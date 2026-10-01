"""Controller-driven landing capture: identical real production environment
(fine terrain with real craters/hills/rocks, regolith material, real sun
lighting, real lunar-gravity PhysX physics) and identical actuator/force
pipeline to `scripts/isaaclab_policy_eval_capture.py` -- but flown by
`lunarsim.control.zemzev_controller.ZemZevController` (the hand-designed,
non-RL ZEM/ZEV guidance + PD attitude law) instead of a trained SAC
checkpoint. This is the real-PhysX, real-rendered version of what
`scripts/render_controller_landing_video.py` shows with matplotlib on the
fast analytic backend -- same controller, same physical vehicle, this time
in the actual production sim.

The controller expects an `AnalyticLanderEnv`-shaped object (`.state` dict +
`.params` + `._ground_z(x, y)`); `_ControllerStateView` below is a thin
adapter that re-packs this script's own real Isaac Sim telemetry (world-
frame position/velocity/quaternion, fuel remaining) into that shape every
step, then reads back the exact same 4-vector action
(`[throttle, pitch_cmd, roll_cmd, yaw_cmd]`, each in [-1, 1]) this script's
actuator/force code already expects -- no other pipeline code had to change.

Run via scripts/run_isaaclab_controller_eval_capture.sh.
"""
import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out-dir", type=str, default="/workspace/LunarSim/out/eval_capture")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--spawn-altitude-min-m", type=float, default=20.0)
parser.add_argument("--spawn-altitude-max-m", type=float, default=60.0)
parser.add_argument("--spawn-xy-radius-m", type=float, default=15.0)
parser.add_argument("--spawn-speed-min-m-s", type=float, default=0.0)
parser.add_argument("--spawn-speed-max-m-s", type=float, default=5.0)
parser.add_argument("--tile-size-m", type=float, default=150.0)
parser.add_argument("--terrain-roughness-scale", type=float, default=1.0,
                     help="0.0 for a near-flat tile, matching an easy/hover-only training stage's terrain; "
                          "1.0 for the full craters/hills/rocks field.")
parser.add_argument("--max-episode-s", type=float, default=60.0)
parser.add_argument("--post-episode-s", type=float, default=3.0,
                     help="keep recording this long after the episode actually ends (landed/crashed/"
                          "timed out), so the outcome is visible, not just the final instant.")
parser.add_argument("--lidar-config", type=str, default="OS2")
parser.add_argument("--lidar-every-n-frames", type=int, default=15)
parser.add_argument("--max-seeds-to-try", type=int, default=1,
                     help="if >1 and the first seed doesn't land safely, bump the seed and retry (up to this "
                          "many attempts) so the capture shows an actual successful landing.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import numpy as np
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.sensors import Imu, ImuCfg, Pva, PvaCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import (
    add_heightfield_collision, apply_regolith_physics_material, build_render_mesh,
)
from lunarsim.adapters.isaac.isaac_lander_env import _quat_to_euler
from lunarsim.adapters.isaac.lander import spawn_apollo_lm
from lunarsim.adapters.isaac.lighting import create_sun_light, disable_ambient, set_no_ambient_render_settings
from lunarsim.adapters.isaac.materials import create_regolith_material
from lunarsim.adapters.isaac.sensors import create_rtx_lidar, export_rtx_point_cloud, get_point_cloud
from lunarsim.control.zemzev_controller import ZemZevController, ZemZevGains
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.core.terrain.rocks import sample_height_at
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, G0, moment_of_inertia
from lunarsim.rl import LanderParams


def _quat_rotate_world(qw: float, qx: float, qy: float, qz: float, v: np.ndarray) -> np.ndarray:
    """Rotates a body-frame vector `v` into world frame by quaternion (w,x,y,z).
    Same fix as `isaaclab_policy_eval_capture.py`'s identical helper -- see
    that script's REAL BUG note on why world frame (not body/IMU frame) is
    required here to match what `IsaacLanderEnv`/`AnalyticLanderEnv` mean by
    vx/vy/vz/wx/wy/wz.
    """
    qv = np.array([qx, qy, qz], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


class _ControllerStateView:
    """Adapts this script's real Isaac Sim telemetry into the
    `.state`/`.params`/`._ground_z()` shape `ZemZevController.act` expects
    (the same shape `AnalyticLanderEnv` exposes) -- see module docstring.
    """

    def __init__(self, params: LanderParams, tile):
        self.params = params
        self.tile = tile
        self.state: dict = {}

    def _ground_z(self, x: float, y: float) -> float:
        return float(sample_height_at(self.tile.height, self.tile.res_m, np.array([x]), np.array([y]))[0])


out_dir = args_cli.out_dir
chase_dir = os.path.join(out_dir, "chase_frames")
nav_dir = os.path.join(out_dir, "nav_frames")
lidar_dir = os.path.join(out_dir, "lidar_scans")
for d in (out_dir, chase_dir, nav_dir, lidar_dir):
    os.makedirs(d, exist_ok=True)

p = LanderParams()  # real Apollo LM numbers + default safety thresholds

# REAL BUG FOUND via this feasibility test's own Isaac Sim capture (see
# diag_rcs_torque_test.py / diag_effective_mass_inertia.py): under GPU
# Direct-GPU-API physics (device="cuda:0", the default this pipeline used
# everywhere before), per-body RCS TORQUE applied via `PhysxForceAPI` is
# silently dropped entirely -- confirmed with a minimal isolated test (a
# free-floating body, full torque commanded, zero rotation over 4.5s) and
# a hard PhysX error when directly setting angular velocity instead
# ("illegal to call ... if PxSceneFlag::eENABLE_DIRECT_GPU_API is
# enabled"). This silently neutered attitude control in every past
# GPU-mode capture using this actuator mechanism (linear thrust still
# worked -- it happens to route through a GPU-compatible path -- which is
# why past captures looked like normal descents with no visible tilt
# response). CPU physics does not have this limitation.
sim_cfg = sim_utils.SimulationCfg(device="cpu", gravity=(0.0, 0.0, -p.gravity_m_s2))
sim = sim_utils.SimulationContext(sim_cfg)
stage = omni.usd.get_context().get_stage()

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(sim_cfg.physics_prim_path))
physx_scene.CreateEnableCCDAttr(True)

rng = np.random.default_rng(args_cli.seed)

# -- real production terrain, roughness dialable per stage (see arg help) --
r = args_cli.terrain_roughness_scale
terrain_seed = int(rng.integers(0, 2**31 - 1))
cfg = TerrainConfig(
    mode="fine", size_m=args_cli.tile_size_m, res_m=max(0.4, args_cli.tile_size_m / 300.0), seed=terrain_seed,
    coarse_source="procedural",
    hills={"amplitude_m": 1.5 * r, "wavelength_m": args_cli.tile_size_m / 20.0, "hurst": 0.75},
    craters={"count_scale": 1.0 * r, "d_min_m": 1.0, "d_max_m": args_cli.tile_size_m / 6.0, "b": 2.5,
             "depth_ratio": 0.1, "age": 0.3},
    rocks={"density_scale": 0.6 * r, "d_max_m": 0.8},
    roi={"regions": [{"x_m": 0.0, "y_m": 0.0, "sigma_m": args_cli.tile_size_m / 4.0, "weight": 1.0}]},
    curvature=False,
)
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)
render_mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=2, uv_tile_size_m=2.0)

regolith_physics_mat = apply_regolith_physics_material(stage, "/World/PhysicsMaterials/Regolith")
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Terrain/Collision")).Bind(
    regolith_physics_mat, materialPurpose="physics")

normal_map_path = "/tmp/lunarsim_regolith_normal.png"
normal_map = bake_regolith_normal_map(
    resolution=1024, physical_size_m=2.0, amplitude_m=0.04, wavelength_m=0.08, seed=cfg.seed)
imageio.imwrite(normal_map_path, normal_map)
material = create_regolith_material(
    stage, "/World/Looks/Regolith", albedo=0.1, brdf="albedo", roughness=0.95, normal_map_path=normal_map_path)
UsdShade.MaterialBindingAPI.Apply(render_mesh.GetPrim()).Bind(material)

rocks_rng = np.random.default_rng(9)
n_rocks = min(len(tile.rocks.x_m), 40)
rock_idx = rocks_rng.choice(len(tile.rocks.x_m), size=n_rocks, replace=False) if len(tile.rocks.x_m) > n_rocks \
    else np.arange(len(tile.rocks.x_m))
for k, ri in enumerate(rock_idx):
    rx, ry = float(tile.rocks.x_m[ri]), float(tile.rocks.y_m[ri])
    diam = max(0.1, float(tile.rocks.diameter_m[ri]))
    rz = float(sample_height_at(tile.height, tile.res_m, np.array([rx]), np.array([ry]))[0])
    rock = UsdGeom.Sphere.Define(stage, f"/World/Rocks/rock_{k}")
    rock.CreateRadiusAttr(diam / 2.0)
    xf = UsdGeom.Xformable(rock.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz + diam * 0.15))
    UsdPhysics.CollisionAPI.Apply(rock.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(rock.GetPrim()).Bind(regolith_physics_mat, materialPurpose="physics")
    UsdShade.MaterialBindingAPI.Apply(rock.GetPrim()).Bind(material)

sun_pos = SunPosition(elevation_deg=30.0, azimuth_deg=140.0)
create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()

specs = ApolloLMSpecs()
spawn_altitude_m = float(rng.uniform(args_cli.spawn_altitude_min_m, args_cli.spawn_altitude_max_m))
spawn_r = float(rng.uniform(0.0, args_cli.spawn_xy_radius_m))
spawn_theta = float(rng.uniform(0.0, 2.0 * np.pi))
spawn_x, spawn_y = spawn_r * np.cos(spawn_theta), spawn_r * np.sin(spawn_theta)
ground_z0 = float(sample_height_at(tile.height, tile.res_m, np.array([spawn_x]), np.array([spawn_y]))[0])

spawn_speed = float(rng.uniform(args_cli.spawn_speed_min_m_s, args_cli.spawn_speed_max_m_s))
to_center = max(1e-6, float(np.hypot(spawn_x, spawn_y)))
vx0 = -spawn_speed * spawn_x / to_center
vy0 = -spawn_speed * spawn_y / to_center

lm_asset_path = os.path.join(args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz")
lm_root = spawn_apollo_lm(stage, "/World/LM", lm_asset_path, specs=specs, fuel_kg=p.initial_fuel_kg,
                           visual_only=False)
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/LM/PhysicsProxy")).Bind(
    regolith_physics_mat, materialPurpose="physics")

spawn_z = ground_z0 + specs.height_m / 2.0 + spawn_altitude_m
xform_api = UsdGeom.XformCommonAPI(lm_root.GetPrim())
xform_api.SetTranslate(Gf.Vec3d(spawn_x, spawn_y, spawn_z))
UsdPhysics.RigidBodyAPI(lm_root.GetPrim()).GetVelocityAttr().Set(Gf.Vec3f(vx0, vy0, p.spawn_v_z_m_s))
print(f"release: ({spawn_x:.1f}, {spawn_y:.1f}) alt={spawn_altitude_m:.1f}m, h_speed={spawn_speed:.1f} m/s, "
      f"controller=ZemZevController")

PhysxSchema.PhysxRigidBodyAPI.Apply(lm_root.GetPrim()).CreateAngularDampingAttr(p.angular_damping_per_s)

force_api = PhysxSchema.PhysxForceAPI.Apply(lm_root.GetPrim())
force_api.CreateForceEnabledAttr(True)
force_api.CreateWorldFrameEnabledAttr(False)
force_api.CreateForceAttr(Gf.Vec3f(0.0, 0.0, 0.0))
force_api.CreateTorqueAttr(Gf.Vec3f(0.0, 0.0, 0.0))

nav_cam_cfg = CameraCfg(
    prim_path="/World/LM/Sensors/NavCam", update_period=0, height=args_cli.height, width=args_cli.width,
    data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -specs.height_m * 0.5 - 1.2), rot=(0.0, 0.0, 0.0, 1.0),
                                convention="opengl"),
    spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4)),
)
nav_camera = Camera(cfg=nav_cam_cfg)

chase_cam_cfg = CameraCfg(
    prim_path="/World/ChaseCamera", update_period=0, height=args_cli.height, width=args_cli.width,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
)
chase_camera = Camera(cfg=chase_cam_cfg)

print(f"creating RTX LiDAR ({args_cli.lidar_config} profile) on the vehicle...")
lidar_sensor = create_rtx_lidar("/World/LM/Sensors/Lidar", config=args_cli.lidar_config, tick_rate=10.0)
lidar_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/LM/Sensors/Lidar"))
lidar_xf.ClearXformOpOrder()
lidar_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -specs.height_m * 0.3))
lidar_xf.AddRotateYOp().Set(60.0)

imu = Imu(cfg=ImuCfg(prim_path="/World/LM"))
pva = Pva(cfg=PvaCfg(prim_path="/World/LM"))

sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()

controller = ZemZevController(p, ZemZevGains())
controller.reset()
state_view = _ControllerStateView(p, tile)


def _aim_chase_camera(vpos, close: bool):
    vx, vy, vz = float(vpos[0]), float(vpos[1]), float(vpos[2])
    offset = torch.tensor([[14.0, -14.0, 8.0]], device=sim.device, dtype=torch.float32) if close else \
        torch.tensor([[25.0, -25.0, 15.0]], device=sim.device, dtype=torch.float32)
    cam_pos = torch.tensor([[vx, vy, vz]], device=sim.device, dtype=torch.float32) + offset
    target = torch.tensor([[vx, vy, vz]], device=sim.device, dtype=torch.float32)
    chase_camera.set_world_poses_from_view(cam_pos, target)


_aim_chase_camera((spawn_x, spawn_y, spawn_z), close=False)
print("warming up renderer...")
for _ in range(30):
    sim.step(render=True)
    nav_camera.update(dt=sim.get_physics_dt())
    chase_camera.update(dt=sim.get_physics_dt())
    imu.update(dt=sim.get_physics_dt())
    pva.update(dt=sim.get_physics_dt())

sim_dt = sim.get_physics_dt()
render_stride = max(1, round((1.0 / args_cli.fps) / sim_dt))
max_steps = max(1, round((args_cli.max_episode_s + args_cli.post_episode_s) / sim_dt))
episode_end_steps = max(1, round(args_cli.post_episode_s / sim_dt))

telemetry_path = os.path.join(out_dir, "telemetry.csv")
telemetry_fields = [
    "frame", "t_s", "pos_x_m", "pos_y_m", "pos_z_m", "alt_m", "roll_deg", "pitch_deg", "yaw_deg",
    "ang_vel_x_rad_s", "ang_vel_y_rad_s", "ang_vel_z_rad_s", "lin_acc_x_m_s2", "lin_acc_y_m_s2", "lin_acc_z_m_s2",
    "throttle", "rcs_pitch", "rcs_roll", "rcs_yaw", "fuel_frac", "rcs_fuel_frac", "reward", "lidar_hit_count",
]
frames_index_path = os.path.join(out_dir, "frames_index.csv")
lidar_index_path = os.path.join(out_dir, "lidar_index.csv")

fuel_kg = p.initial_fuel_kg
rcs_fuel_kg = p.initial_rcs_fuel_kg
saved_frames = 0
lidar_scan_i = 0
episode_ended_step = None
end_reason = "max_episode_s"
landed_safely = None
touchdown_vz = touchdown_vxy = touchdown_tilt_deg = None

with open(telemetry_path, "w", newline="") as tf, \
     open(frames_index_path, "w", newline="") as ff, \
     open(lidar_index_path, "w", newline="") as lf:
    writer = csv.DictWriter(tf, fieldnames=telemetry_fields)
    writer.writeheader()
    frames_writer = csv.writer(ff)
    frames_writer.writerow(["frame_index", "step", "t_s"])
    lidar_writer = csv.writer(lf)
    lidar_writer.writerow(["scan_index", "filename", "step", "t_s", "hit_count"])

    step = 0
    for step in range(max_steps):
        pos_w = pva.data.pos_w[0]
        qx, qy, qz, qw = pva.data.quat_w[0]
        pos_w = pos_w.cpu().numpy() if hasattr(pos_w, "cpu") else np.asarray(pos_w)
        qx, qy, qz, qw = (float(v) for v in (qx, qy, qz, qw))
        roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)
        ang_vel_b = imu.data.ang_vel_b[0]
        ang_vel_b = ang_vel_b.cpu().numpy() if hasattr(ang_vel_b, "cpu") else np.asarray(ang_vel_b)
        lin_vel_b = pva.data.lin_vel_b[0]
        lin_vel_b = lin_vel_b.cpu().numpy() if hasattr(lin_vel_b, "cpu") else np.asarray(lin_vel_b)

        lin_vel_w = _quat_rotate_world(qw, qx, qy, qz, lin_vel_b)
        ang_vel_w = _quat_rotate_world(qw, qx, qy, qz, ang_vel_b)

        ground_z_here = float(sample_height_at(
            tile.height, tile.res_m, np.array([pos_w[0]]), np.array([pos_w[1]]))[0])
        belly_z = float(pos_w[2]) - specs.height_m / 2.0
        alt_m = belly_z - ground_z_here

        # -- feed the real telemetry into the controller (see
        # _ControllerStateView's docstring): tilt_x<->roll, tilt_y<->pitch,
        # wx/wy/wz<->world-frame angular rate, z<->belly (ground-contact
        # reference, matching AnalyticLanderEnv's touchdown convention). --
        state_view.state = {
            "x": float(pos_w[0]), "y": float(pos_w[1]), "z": belly_z,
            "vx": float(lin_vel_w[0]), "vy": float(lin_vel_w[1]), "vz": float(lin_vel_w[2]),
            "tilt_x": float(roll), "tilt_y": float(pitch), "yaw": float(yaw),
            "wx": float(ang_vel_w[0]), "wy": float(ang_vel_w[1]), "wz": float(ang_vel_w[2]),
            "fuel_kg": fuel_kg, "rcs_fuel_kg": rcs_fuel_kg,
        }
        action = controller.act(state_view)
        throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        pitch_cmd = float(np.clip(action[1], -1.0, 1.0))
        roll_cmd = float(np.clip(action[2], -1.0, 1.0))
        yaw_cmd = float(np.clip(action[3], -1.0, 1.0))
        if fuel_kg <= 0.0:
            throttle = 0.0
        if rcs_fuel_kg <= 0.0:
            pitch_cmd = roll_cmd = yaw_cmd = 0.0

        thrust_mag = 0.0 if throttle <= 0.0 else p.dps_thrust_min_n + throttle * (p.dps_thrust_max_n - p.dps_thrust_min_n)
        max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple

        mass_kg = p.dry_mass_kg + fuel_kg
        i_tilt, i_yaw = moment_of_inertia(mass_kg, p.body_radius_m, p.body_height_m)
        # kept for documentation/parity with the training envs' own mass
        # authoring -- REAL BUG FOUND: this override is a confirmed no-op
        # under this pipeline (see diag_effective_mass_inertia.py: a
        # precise single-step force/torque measurement showed PhysX using
        # EXACTLY unit mass (1.0 kg) and unit inertia (1.0 kg.m^2) for this
        # body regardless of what MassAPI is set to, on both CPU and GPU
        # physics). Force/torque below are scaled directly by the REAL
        # mass_kg/i_tilt/i_yaw to compensate -- i.e. what's actually
        # authored on the schema is the desired ACCELERATION (since the
        # solver's effective mass/inertia is 1, force-in = accel-out
        # directly), not real Newtons/N.m. This replaces the old fixed
        # `FORCE_TORQUE_CAL=1/100` empirical fudge factor (calibrated for
        # GPU mode's separate, unrelated ~100x force scale quirk -- see the
        # `sim_cfg` comment above for why GPU mode isn't used at all here
        # anymore) with a mass-accurate correction that stays right as
        # fuel burns and i_tilt changes over the flight.
        mass_api = UsdPhysics.MassAPI(lm_root.GetPrim())
        mass_api.GetMassAttr().Set(mass_kg)
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(i_tilt, i_tilt, i_yaw))

        force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, thrust_mag / mass_kg))
        force_api.GetTorqueAttr().Set(Gf.Vec3f(
            pitch_cmd * max_torque / i_tilt, roll_cmd * max_torque / i_tilt,
            yaw_cmd * max_torque / i_yaw))

        sim.step(render=True)
        nav_camera.update(dt=sim_dt)
        chase_camera.update(dt=sim_dt)
        imu.update(dt=sim_dt)
        pva.update(dt=sim_dt)

        dps_mdot = thrust_mag / (p.dps_isp_s * G0)
        rcs_mdot = (abs(pitch_cmd) + abs(roll_cmd) + abs(yaw_cmd)) * (p.rcs_thrust_n * p.rcs_jets_per_couple) / (
            p.rcs_isp_s * G0)
        fuel_kg = max(0.0, fuel_kg - dps_mdot * sim_dt)
        rcs_fuel_kg = max(0.0, rcs_fuel_kg - rcs_mdot * sim_dt)

        t_s = step * sim_dt
        lidar_hit_count = 0
        if step % args_cli.lidar_every_n_frames == 0:
            pc = get_point_cloud(lidar_sensor)
            lidar_hit_count = int(pc["x_m"].size)
            if lidar_hit_count > 0:
                fn = f"scan_{lidar_scan_i:04d}.ply"
                export_rtx_point_cloud(pc, os.path.join(lidar_dir, fn))
                lidar_writer.writerow([lidar_scan_i, fn, step, round(t_s, 4), lidar_hit_count])
                lidar_scan_i += 1

        ang_vel = imu.data.ang_vel_b[0]
        lin_acc = imu.data.lin_acc_b[0]
        ang_vel = ang_vel.cpu().numpy() if hasattr(ang_vel, "cpu") else np.asarray(ang_vel)
        lin_acc = lin_acc.cpu().numpy() if hasattr(lin_acc, "cpu") else np.asarray(lin_acc)

        writer.writerow({
            "frame": step, "t_s": round(t_s, 4),
            "pos_x_m": round(float(pos_w[0]), 4), "pos_y_m": round(float(pos_w[1]), 4),
            "pos_z_m": round(float(pos_w[2]), 4), "alt_m": round(alt_m, 4),
            "roll_deg": round(np.degrees(roll), 3), "pitch_deg": round(np.degrees(pitch), 3),
            "yaw_deg": round(np.degrees(yaw), 3),
            "ang_vel_x_rad_s": round(float(ang_vel[0]), 6), "ang_vel_y_rad_s": round(float(ang_vel[1]), 6),
            "ang_vel_z_rad_s": round(float(ang_vel[2]), 6),
            "lin_acc_x_m_s2": round(float(lin_acc[0]), 6), "lin_acc_y_m_s2": round(float(lin_acc[1]), 6),
            "lin_acc_z_m_s2": round(float(lin_acc[2]), 6),
            "throttle": round(throttle, 3), "rcs_pitch": round(pitch_cmd, 3), "rcs_roll": round(roll_cmd, 3),
            "rcs_yaw": round(yaw_cmd, 3), "fuel_frac": round(fuel_kg / p.initial_fuel_kg, 4),
            "rcs_fuel_frac": round(rcs_fuel_kg / p.initial_rcs_fuel_kg, 4),
            "reward": "", "lidar_hit_count": lidar_hit_count,
        })

        if step % render_stride == 0:
            _aim_chase_camera(pos_w, close=episode_ended_step is not None)
            for cam, cam_dir in ((nav_camera, nav_dir), (chase_camera, chase_dir)):
                rgb = cam.data.output.get("rgb")
                if rgb is None:
                    continue
                rgb_np = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
                imageio.imwrite(os.path.join(cam_dir, f"frame_{saved_frames:04d}.png"),
                                 np.clip(rgb_np[..., :3], 0, 255).astype(np.uint8))
            frames_writer.writerow([saved_frames, step, round(t_s, 4)])
            saved_frames += 1

        if step % (render_stride * 20) == 0:
            print(f"  step {step}/{max_steps} (t={t_s:.1f}s alt={alt_m:.2f}m throttle={throttle:.2f})")

        if episode_ended_step is None:
            lost_control = bool(np.hypot(roll, pitch) > p.loss_of_control_tilt_rad)
            timed_out = bool(t_s >= args_cli.max_episode_s)
            if belly_z <= ground_z_here + 1e-2:
                episode_ended_step = step
                end_reason = "touchdown"
                touchdown_vz = float(lin_vel_w[2])
                touchdown_vxy = float(np.hypot(lin_vel_w[0], lin_vel_w[1]))
                touchdown_tilt_deg = float(np.degrees(np.hypot(roll, pitch)))
                landed_safely = bool(
                    abs(touchdown_vz) <= p.safe_landing_v_z_m_s
                    and touchdown_vxy <= p.safe_landing_v_xy_m_s
                    and np.hypot(roll, pitch) <= p.safe_landing_tilt_rad
                )
                print(f"  touchdown at step {step} (t={t_s:.2f}s) -- "
                      f"vz={touchdown_vz:.2f} vxy={touchdown_vxy:.2f} tilt={touchdown_tilt_deg:.1f}deg "
                      f"landed_safely={landed_safely}")
            elif lost_control:
                episode_ended_step = step
                end_reason = "lost_control"
                print(f"  lost control at step {step} (t={t_s:.2f}s)")
            elif timed_out:
                episode_ended_step = step
                end_reason = "max_episode_s"

        if episode_ended_step is not None and step >= episode_ended_step + episode_end_steps:
            break

print(f"episode ended: {end_reason}")
print(f"saved {saved_frames} frame pairs to {chase_dir} / {nav_dir}")
print(f"wrote telemetry ({step + 1} rows) -> {telemetry_path}")
print(f"wrote {lidar_scan_i} lidar scans -> {lidar_dir}")
with open(os.path.join(out_dir, "eval_summary.txt"), "w") as f:
    f.write(f"controller=ZemZevController\nend_reason={end_reason}\nt_s={step * sim_dt:.2f}\n"
            f"fuel_frac={fuel_kg / p.initial_fuel_kg:.3f}\nrcs_fuel_frac={rcs_fuel_kg / p.initial_rcs_fuel_kg:.3f}\n"
            f"landed_safely={landed_safely}\n"
            + (f"touchdown_vz_m_s={touchdown_vz:.2f}\ntouchdown_vxy_m_s={touchdown_vxy:.2f}\n"
               f"touchdown_tilt_deg={touchdown_tilt_deg:.1f}\n" if end_reason == "touchdown" else ""))
simulation_app.close()
