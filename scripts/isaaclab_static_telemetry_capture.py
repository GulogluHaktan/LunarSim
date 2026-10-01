"""Uncontrolled-descent telemetry capture: the real, final production
environment (fine terrain with craters/hills/rocks, regolith material +
normal map, real sun lighting -- same pipeline as isaaclab_orbit_demo.py)
with the real Apollo LM vehicle released at a RANDOM altitude/offset/
horizontal speed and left completely alone under real lunar-gravity PhysX
physics -- no RL policy, no controller, no force/torque command is ever
issued, so the vehicle free-falls/coasts on whatever ballistic path its
release state gives it (this is "yorunge-gibi" in the sense of an
unpowered coast toward the surface, NOT real two-body orbital mechanics --
same scope caveat as `lunarsim.rl.analytic_lander_env`'s curriculum spawns)
until it hits the real terrain -- however it hits it (soft, tumbling,
crashing), since nothing is correcting it. That uncorrected outcome is
itself the accuracy check this run exists for: does the vehicle interact
with the real terrain collision believably.

Three vehicle-mounted sensors are read every simulated frame and logged
throughout the whole fall: an onboard nav camera (nose-down, looking at the
landing site), a real RTX LiDAR, and a real IMU (angular velocity + proper
linear acceleration, from `isaaclab.sensors.Imu` -- reads ~0 in freefall,
~(0,0,1.62) m/s^2 once at rest, since an IMU measures proper acceleration,
not coordinate acceleration). A second, world-fixed-but-retargeted "chase"
camera follows the vehicle's world position every frame (a genuinely fixed
camera can't keep a several-hundred-meter fall in frame). All of this --
two video feeds, periodic LiDAR point clouds, and a per-frame telemetry
CSV -- is exactly the video + telemetry material needed for next week's
demo, gathered from one run instead of building each case separately.

Recording stops `--post-touchdown-s` after the vehicle's belly first
reaches the real terrain height (not at a fixed duration -- how long the
fall takes depends on the randomly drawn altitude), capped at
`--max-duration-s` as a safety net in case it somehow never touches down.

Run via scripts/run_isaaclab_static_telemetry_capture.sh.
"""
import argparse
import csv
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out-dir", type=str, default="/workspace/LunarSim/out/static_capture")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=None,
                     help="RNG seed for the spawn altitude/offset/speed draw and the terrain's "
                          "own procedural seed. Omit for a different random release every run.")
parser.add_argument("--spawn-altitude-min-m", type=float, default=120.0)
parser.add_argument("--spawn-altitude-max-m", type=float, default=350.0)
parser.add_argument("--spawn-xy-radius-m", type=float, default=80.0,
                     help="the release point is drawn uniformly within this radius of the tile "
                          "center (world origin), same spawn-scatter idea as AnalyticLanderEnv.")
parser.add_argument("--spawn-speed-min-m-s", type=float, default=8.0)
parser.add_argument("--spawn-speed-max-m-s", type=float, default=35.0,
                     help="horizontal release speed range, aimed from the release point toward "
                          "the tile center (like AnalyticLanderEnv's spawn_horizontal_speed_m_s) "
                          "-- deliberately modest (not literal orbital ~1.6 km/s) so the resulting "
                          "ground track fits a tile this pipeline can still generate/collide "
                          "against at real fine-terrain resolution; see the module docstring.")
parser.add_argument("--tile-size-m", type=float, default=600.0,
                     help="must comfortably exceed 2*(spawn_xy_radius_m + spawn_speed_max_m_s * "
                          "fall_time_s), or the vehicle can coast off the collidable tile before "
                          "it ever reaches the ground -- the fall-time estimate this defaults "
                          "against assumes spawn_altitude_max_m free-fall with no horizontal drag.")
parser.add_argument("--post-touchdown-s", type=float, default=5.0,
                     help="keep recording this long after first ground contact, to also capture "
                          "the resting/settled (or tipped-over) outcome, not just the impact.")
parser.add_argument("--max-duration-s", type=float, default=90.0,
                     help="hard cap in case the vehicle somehow never touches down (e.g. drifts "
                          "off the tile edge).")
parser.add_argument("--lidar-config", type=str, default="OS2",
                     help="Isaac's real hardware LiDAR profiles differ a lot in range -- REAL BUG FOUND VIA "
                          "A DOCKER RUN: the previous default, OS0, is documented (Isaac's own config JSON) "
                          "at farRangeM=75m, so it recorded zero hits for the entire descent from a "
                          "120-350m release and only ever saw anything in the last <1s before touchdown. "
                          "OS2 (the long-range Ouster variant) is documented at farRangeM=350m, matching "
                          "this script's own spawn-altitude range.")
parser.add_argument("--lidar-every-n-frames", type=int, default=15)
parser.add_argument("--enable-impact-marks", action="store_true", default=False,
                     help="disturbed-regolith decals at the impact/drag site (see _spawn_impact_mark) -- "
                          "OFF by default: at production scale their SIZE (Bekker-radius-derived, meant "
                          "to be visible against ~1.5m natural terrain relief) reads as wildly "
                          "disproportionate to the vehicle once actually seen in a full run, not just "
                          "the small-scale test this was tuned against. Left in the code (not deleted) "
                          "for whoever wants to retune the sizing rather than rebuild the mechanism.")
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
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import (
    add_heightfield_collision, apply_regolith_physics_material, build_render_mesh,
)
from lunarsim.adapters.isaac.isaac_lander_env import _quat_to_euler
from lunarsim.adapters.isaac.lander import spawn_apollo_lm
from lunarsim.adapters.isaac.lighting import (
    create_sun_light, disable_ambient, set_no_ambient_render_settings,
)
from lunarsim.adapters.isaac.materials import create_regolith_material
from lunarsim.core.terrain.deformation import BekkerSoilParams, bekker_sinkage_m
from lunarsim.adapters.isaac.sensors import create_rtx_lidar, export_rtx_point_cloud, get_point_cloud
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.core.terrain.rocks import sample_height_at
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs

out_dir = args_cli.out_dir
chase_dir = os.path.join(out_dir, "chase_frames")
nav_dir = os.path.join(out_dir, "nav_frames")
lidar_dir = os.path.join(out_dir, "lidar_scans")
for d in (out_dir, chase_dir, nav_dir, lidar_dir):
    os.makedirs(d, exist_ok=True)

sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"), gravity=(0.0, 0.0, -1.62))
sim = sim_utils.SimulationContext(sim_cfg)

stage = omni.usd.get_context().get_stage()

# scene-wide CCD switch: a per-body `PhysxRigidBodyAPI.enableCCD` (set in
# `lander.py`'s spawn_apollo_lm -- see the REAL BUG note there on the
# ~33 m/s tunneling this fixes) only actually engages if the PHYSICS SCENE
# also has CCD turned on; SimulationCfg's default scene doesn't.
from pxr import PhysxSchema

physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(sim_cfg.physics_prim_path))
physx_scene.CreateEnableCCDAttr(True)

rng = np.random.default_rng(args_cli.seed)

# -- final production terrain: real fine detail (craters+hills+rocks+regolith
# normal map), the same "final ortam" knobs as isaaclab_orbit_demo.py -- NOT
# the flat, physics-matching terrain isaaclab_landing_render.py uses for its
# kinematic trajectory replay. This run does real PhysX collision under the
# vehicle, so it needs the real terrain shape, not a flat stand-in. Sized off
# `--tile-size-m` (not a fixed 120m) since the release point can now be
# hundreds of meters up with real horizontal drift -- see that arg's help
# text for the sizing rationale.
terrain_seed = int(rng.integers(0, 2**31 - 1))
cfg = TerrainConfig(
    mode="fine", size_m=args_cli.tile_size_m, res_m=max(0.4, args_cli.tile_size_m / 300.0), seed=terrain_seed,
    coarse_source="procedural",
    hills={"amplitude_m": 1.5, "wavelength_m": args_cli.tile_size_m / 20.0, "hurst": 0.75},
    craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": args_cli.tile_size_m / 6.0, "b": 2.5,
             "depth_ratio": 0.1, "age": 0.3},
    rocks={"density_scale": 0.6, "d_max_m": 0.8},
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


# -- disturbed-regolith impact/drag mark: a DECAL, not a heightfield edit.
#
# REAL BUG FOUND VIA A DOCKER RUN (three attempts deep): the original plan
# was to stamp a real Bekker-sinkage depression into `tile.height` (see
# `lunarsim.core.terrain.deformation.wheel_footprint_stamp`, already
# verified correct in isolation by scripts/isaac_test_wheel_deformation.py)
# and re-author the terrain's collision+render meshes from it each time
# (remove-then-rebuild, the same pattern proven to work -- confirmed by a
# real PhysX raycast -- in a standalone probe under `isaacsim.core.api.
# World`). Under THIS script's `isaaclab.sim.SimulationContext` stepping
# path specifically, that rebuild silently doesn't take: the height data
# itself demonstrably changed (logged h_before/h_after each stamp), but a
# `raycast_closest` at the exact stamped point kept returning `hit=False`
# even several physics steps after the rebuild -- the query interface
# never picks up a heightfield collider that's REMOVED AND RECREATED
# mid-simulation, only ones present at `sim.reset()`. This is the same
# family of "IsaacLab's stepping path doesn't refresh X" gap already hit
# for PhysX's transform fast-cache and for two Kit extensions this session
# -- not fixable from here in the time this has to ship in.
#
# A disturbed-regolith DECAL sidesteps it entirely: it's an ordinary NEW
# prim, the same kind the (already working, every run) decorative rocks
# already are -- no remove/recreate of anything, so nothing needs a stale
# cache to refresh. Sized off the same Bekker sinkage math (radius) so the
# mark's SIZE is still load-grounded, and given a darker albedo than the
# undisturbed regolith -- churned/freshly-exposed regolith really does read
# darker than the weathered surface layer, so this isn't purely cosmetic
# either, just not a true collision-affecting deformation.
_mark_material = create_regolith_material(
    stage, "/World/Looks/DisturbedRegolith", albedo=0.035, brdf="albedo", roughness=0.98)
_mark_count = 0


def _spawn_impact_mark(x_m: float, y_m: float, ground_z_m: float, radius_m: float):
    global _mark_count
    mark = UsdGeom.Cylinder.Define(stage, f"/World/ImpactMarks/mark_{_mark_count}")
    mark.CreateRadiusAttr(radius_m)
    mark.CreateHeightAttr(0.02)
    mark.CreateAxisAttr("Z")
    xf = UsdGeom.Xformable(mark.GetPrim())
    xf.ClearXformOpOrder()
    # +0.015m: just proud of the real surface so it never z-fights the
    # terrain render mesh, not a claim the regolith actually mounds up here.
    xf.AddTranslateOp().Set(Gf.Vec3d(x_m, y_m, ground_z_m + 0.015))
    UsdShade.MaterialBindingAPI.Apply(mark.GetPrim()).Bind(_mark_material)
    _mark_count += 1

# decorative rocks -- a separate, fixed-seed RNG (independent of the spawn
# draw's `rng` above, so the scenery is reproducible run-to-run even with
# `--seed` omitted), scaled to the actual tile size and pushed out past
# `spawn_xy_radius_m` so decoration can't land inside where the vehicle
# might actually touch down.
rocks_rng = np.random.default_rng(9)
rock_r_min = args_cli.spawn_xy_radius_m + 10.0
rock_r_max = args_cli.tile_size_m * 0.45
for ri in range(24):
    ang = rocks_rng.uniform(0, 2 * np.pi)
    rad = rocks_rng.uniform(rock_r_min, rock_r_max)
    rx, ry = rad * np.cos(ang), rad * np.sin(ang)
    rz = float(sample_height_at(tile.height, tile.res_m, np.array([rx]), np.array([ry]))[0])
    size = rocks_rng.uniform(0.3, 1.3)
    rock = UsdGeom.Sphere.Define(stage, f"/World/Rocks/rock_{ri}")
    rock.CreateRadiusAttr(0.5)
    xf = UsdGeom.Xformable(rock.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz + size * 0.15))
    xf.AddScaleOp().Set(Gf.Vec3f(size, size * rocks_rng.uniform(0.6, 1.0), size * rocks_rng.uniform(0.5, 0.85)))
    UsdShade.MaterialBindingAPI.Apply(rock.GetPrim()).Bind(material)

sun_pos = SunPosition(elevation_deg=30.0, azimuth_deg=140.0)
create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()

# -- vehicle: real physics (visual_only=False), so gravity/collision/IMU are
# all real. No RL policy or controller is ever attached -- it is released at
# a random altitude/offset/horizontal-speed draw and left to coast under
# lunar gravity alone (see the module docstring's scope caveat on this NOT
# being real orbital mechanics).
specs = ApolloLMSpecs()

spawn_altitude_m = float(rng.uniform(args_cli.spawn_altitude_min_m, args_cli.spawn_altitude_max_m))
spawn_r = float(rng.uniform(0.0, args_cli.spawn_xy_radius_m))
spawn_theta = float(rng.uniform(0.0, 2.0 * np.pi))
spawn_x = spawn_r * np.cos(spawn_theta)
spawn_y = spawn_r * np.sin(spawn_theta)
ground_z = float(sample_height_at(tile.height, tile.res_m, np.array([spawn_x]), np.array([spawn_y]))[0])

spawn_speed = float(rng.uniform(args_cli.spawn_speed_min_m_s, args_cli.spawn_speed_max_m_s))
to_center_dist = max(1e-6, float(np.hypot(spawn_x, spawn_y)))
vx0 = -spawn_speed * spawn_x / to_center_dist
vy0 = -spawn_speed * spawn_y / to_center_dist

lm_asset_path = os.path.join(args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz")
lm_root = spawn_apollo_lm(stage, "/World/LM", lm_asset_path, specs=specs, visual_only=False)
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/LM/PhysicsProxy")).Bind(
    regolith_physics_mat, materialPurpose="physics")

spawn_z = ground_z + specs.height_m / 2.0 + spawn_altitude_m
xform_api = UsdGeom.XformCommonAPI(lm_root.GetPrim())
xform_api.SetTranslate(Gf.Vec3d(spawn_x, spawn_y, spawn_z))
# initial horizontal velocity, authored directly on RigidBodyAPI's own
# attribute so PhysX picks it up as the release state at `sim.reset()` --
# the same "set once before reset, PhysX owns it after" pattern the initial
# position above already relies on.
UsdPhysics.RigidBodyAPI(lm_root.GetPrim()).GetVelocityAttr().Set(Gf.Vec3f(vx0, vy0, 0.0))
print(f"release: ({spawn_x:.1f}, {spawn_y:.1f}) alt={spawn_altitude_m:.1f}m above ground_z={ground_z:.3f}m, "
      f"horizontal speed={spawn_speed:.1f} m/s toward the tile center, no vertical release speed")

# -- onboard nav camera: nose-down, child of the vehicle root so it inherits
# the real-physics pose automatically (no manual per-frame repositioning
# needed) -- a landing/hazard-avoidance-style view of the site below.
nav_cam_cfg = CameraCfg(
    prim_path="/World/LM/Sensors/NavCam",
    update_period=0,
    height=args_cli.height,
    width=args_cli.width,
    data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(
        # REAL BUG FOUND VIA A DOCKER RENDER: mounted near the TOP of the
        # body (height_m*0.45 above the CENTER) looking straight down, the
        # camera's own forward ray immediately hit the vehicle's own
        # descent-stage geometry before ever reaching open ground below --
        # every captured frame came back solid black (self-occlusion, not
        # a missing-light or wrong-direction bug). Moved below the body's
        # own bottom face instead, where "straight down" actually looks at
        # the landing site, not through the vehicle itself. REAL BUG FOUND
        # AGAIN once that black-frame bug was fixed: -0.15m clearance put
        # the camera barely above the real (bumpy) terrain once the
        # vehicle actually settled, so every frame from around touchdown
        # onward was an extreme, blurry close-up of dirt texture, not a
        # useful view -- widened the clearance so it still reads as a
        # hazard-cam shot of the landing site once resting, not a macro
        # shot of the regolith directly underneath.
        pos=(0.0, 0.0, -specs.height_m * 0.5 - 1.2),
        # identity rotation in the "opengl" convention (forward = -Z by
        # convention definition) points the camera along the parent's own
        # local -Z with no hand-derived quaternion needed -- simpler and
        # less error-prone than rotating a "world"-convention (+X-forward)
        # offset by hand, which is what produced the wrong-mount-point bug
        # above in the first place.
        rot=(0.0, 0.0, 0.0, 1.0),
        convention="opengl",
    ),
    spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4)),
)
nav_camera = Camera(cfg=nav_cam_cfg)

# -- external chase camera: world-fixed prim, but RE-AIMED every recorded
# frame in the main loop below (via set_world_poses_from_view against the
# vehicle's live position) -- a one-time "establishing shot" framing can't
# keep a several-hundred-meter fall in frame, unlike the earlier fixed-drop
# version of this script.
chase_cam_cfg = CameraCfg(
    prim_path="/World/ChaseCamera",
    update_period=0,
    height=args_cli.height,
    width=args_cli.width,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
)
chase_camera = Camera(cfg=chase_cam_cfg)

# -- vehicle-mounted RTX LiDAR: child of the vehicle root, same rationale as
# the nav camera (inherits the real-physics pose for free). Mounted near the
# vehicle belly.
#
# REAL BUG FOUND VIA A DOCKER RUN: this used to leave the sensor at its
# default (level/forward-looking) orientation, on the assumption that a
# spinning 360-degree-azimuth profile "doesn't need to be aimed at
# anything in particular" -- but a rotating LiDAR's vertical FOV band is
# still centered on its own local reference direction, not world-down. At
# altitude with the vehicle upright, most of a level sensor's scan cone
# missed the ground entirely: telemetry showed zero hits for the whole
# descent, only registering ground returns in roughly the last second
# before touchdown once the vehicle's own tumble happened to tip the cone
# downward. Tilting the mount itself down (same nose-down idea as the nav
# camera) gives it real ground coverage through the actual descent, not
# just after impact.
print(f"creating RTX LiDAR ({args_cli.lidar_config} profile) on the vehicle...")
lidar_sensor = create_rtx_lidar(f"/World/LM/Sensors/Lidar", config=args_cli.lidar_config, tick_rate=10.0)
lidar_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/LM/Sensors/Lidar"))
lidar_xf.ClearXformOpOrder()
lidar_local_z = -specs.height_m * 0.3
lidar_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, lidar_local_z))
lidar_xf.AddRotateYOp().Set(60.0)  # tilt the scan cone's reference direction down toward the ground

# -- IMU: the vehicle root itself is the rigid body it measures --
imu_cfg = ImuCfg(prim_path="/World/LM")
imu = Imu(cfg=imu_cfg)

# -- PVA: same rigid body, world-frame pose -- for telemetry.csv's
# position/orientation columns (see the REAL BUG note below on why this
# replaced two earlier, broken position-read attempts).
pva_cfg = PvaCfg(prim_path="/World/LM")
pva = Pva(cfg=pva_cfg)

sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()

# REAL BUG FOUND VIA A DOCKER RUN, THREE ATTEMPTS DEEP: telemetry.csv's
# position/orientation went through three different read mechanisms before
# one actually tracked the live simulation:
#   1. `UsdGeom.Xformable(...).ComputeLocalToWorldTransform(Usd.TimeCode.
#      Default())` -- reads the AUTHORED (Default-time) transform, which
#      PhysX's per-frame write-back never updates in this pipeline, so
#      every row silently repeated the spawn pose (confirmed via the
#      rendered video itself actually showing the vehicle fall/settle --
#      only this raw-USD read was stale).
#   2. `isaacsim.core.prims.RigidPrim` (the mechanism already verified
#      correct in `lunarsim.adapters.isaac.isaac_lander_env`) -- not even a
#      resolvable extension under `isaaclab.app.AppLauncher` in THIS docker
#      image ("Failed to resolve extension dependencies ... No versions of
#      isaacsim.core.prims").
#   3. `omni.physx.get_physx_interface().get_rigidbody_transformation(...)`
#      -- reads PhysX's "fast cache", which IsaacLab's own (leaner, tensor-
#      API-based) stepping path never refreshes; calling
#      `update_transformations()` first didn't help either, so something
#      about this stepping path never populates that cache at all here.
# `isaaclab.sensors.Pva` (Pose/Velocity/Acceleration) is the SAME sensor
# family as the already-working `Imu` above -- same tensor-API backend, so
# if Imu's ang_vel/lin_acc are real (they are -- they show a real touchdown
# transient), Pva's pos_w/quat_w are read the same correct way.

def _aim_chase_camera(vehicle_pos_w):
    """Re-aim the world-fixed chase camera at the vehicle's current position
    -- offset up/back by a fixed vector so the framing stays consistent as
    the vehicle actually moves hundreds of meters during the fall.

    REAL BUG FOUND VIA A DOCKER RUN: the wide (~38m) fall-tracking offset
    used throughout the whole flight made the impact/drag mark (see the
    crash-stamp code below -- confirmed by direct data to actually be
    there, ~0.3m deep) visually indistinguishable from the terrain's own
    ~1.5m natural relief once the vehicle landed. Once `touchdown_step` is
    set, this tightens to a close-up framing where a meter-scale mark
    actually reads clearly, instead of the same wide framing a fall needs.
    """
    vx, vy, vz = float(vehicle_pos_w[0]), float(vehicle_pos_w[1]), float(vehicle_pos_w[2])
    if touchdown_step is not None:
        # REAL BUG FOUND VIA A FULL-SCALE DOCKER RUN: (8,-8,4) put the
        # camera close enough that an upright (non-tumbled) touchdown's own
        # ~9.4m footpad-span geometry filled/clipped the frame right at
        # impact -- widened for real clearance around the whole vehicle,
        # not just its body core.
        offset = torch.tensor([[14.0, -14.0, 8.0]], device=sim.device, dtype=torch.float32)
    else:
        offset = torch.tensor([[25.0, -25.0, 15.0]], device=sim.device, dtype=torch.float32)
    cam_pos = torch.tensor([[vx, vy, vz]], device=sim.device, dtype=torch.float32) + offset
    target = torch.tensor([[vx, vy, vz]], device=sim.device, dtype=torch.float32)
    chase_camera.set_world_poses_from_view(cam_pos, target)


touchdown_step = None  # set below, but must exist before the first _aim_chase_camera call
_aim_chase_camera((spawn_x, spawn_y, spawn_z))

print("warming up renderer...")
for _ in range(30):
    sim.step(render=True)
    nav_camera.update(dt=sim.get_physics_dt())
    chase_camera.update(dt=sim.get_physics_dt())
    imu.update(dt=sim.get_physics_dt())
    pva.update(dt=sim.get_physics_dt())

sim_dt = sim.get_physics_dt()
render_stride = max(1, round((1.0 / args_cli.fps) / sim_dt))
max_steps = max(1, round(args_cli.max_duration_s / sim_dt))
post_touchdown_steps = max(1, round(args_cli.post_touchdown_s / sim_dt))

telemetry_path = os.path.join(out_dir, "telemetry.csv")
telemetry_fields = [
    "frame", "t_s", "pos_x_m", "pos_y_m", "pos_z_m", "alt_m",
    "roll_deg", "pitch_deg", "yaw_deg",
    "ang_vel_x_rad_s", "ang_vel_y_rad_s", "ang_vel_z_rad_s",
    "lin_acc_x_m_s2", "lin_acc_y_m_s2", "lin_acc_z_m_s2",
    "lidar_hit_count",
]

print(f"capturing up to {max_steps} physics steps ({args_cli.max_duration_s:.0f}s cap), stopping "
      f"{args_cli.post_touchdown_s:.1f}s after first ground contact, rendering every {render_stride} "
      f"steps (~{args_cli.fps:.0f} fps)...")

saved_frames = 0
touchdown_step = None
last_stamp_xy = (0.0, 0.0)
crash_contact_width_m = specs.footpad_span_m * 0.5
crash_sinkage_m = 0.0
lidar_scan_i = 0

# index files so post-processing (scripts/compose_capture_video.py) can
# line up chase/nav PNG frames and lidar_scans/*.ply files against
# telemetry.csv's t_s timeline WITHOUT having to guess a render/lidar
# cadence back out of the run's CLI args -- the exact (frame|scan) -> step,
# t_s mapping actually used this run, straight from the source of truth.
frames_index_path = os.path.join(out_dir, "frames_index.csv")
lidar_index_path = os.path.join(out_dir, "lidar_index.csv")
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
        sim.step(render=True)
        nav_camera.update(dt=sim_dt)
        chase_camera.update(dt=sim_dt)
        imu.update(dt=sim_dt)
        pva.update(dt=sim_dt)

        translation = pva.data.pos_w[0]
        qx, qy, qz, qw = pva.data.quat_w[0]  # (x, y, z, w) order
        translation = translation.cpu().numpy() if hasattr(translation, "cpu") else np.asarray(translation)
        qx, qy, qz, qw = (float(v) for v in (qx, qy, qz, qw))
        euler_rad = _quat_to_euler(qw, qx, qy, qz)
        euler = [np.degrees(a) for a in euler_rad]

        # ground height under the CURRENT (x, y), not the spawn point -- the
        # vehicle has real horizontal drift now, unlike the earlier fixed-
        # drop version where it never left the spawn column.
        ground_z_here = float(sample_height_at(
            tile.height, tile.res_m, np.array([translation[0]]), np.array([translation[1]]))[0])
        belly_z = float(translation[2]) - specs.height_m / 2.0
        in_contact = belly_z <= ground_z_here + 0.05
        if touchdown_step is None and belly_z <= ground_z_here + 1e-2:
            touchdown_step = step
            print(f"  ground contact at step {step} (t={step * sim_dt:.2f}s) -- "
                  f"recording {args_cli.post_touchdown_s:.1f}s more, then stopping")
            # disturbed-regolith mark radius, from the SAME Bekker sinkage
            # math `leg_force_bounds_n` uses elsewhere in this codebase --
            # see `_spawn_impact_mark`'s block comment above for why this
            # ended up a decal (not a real heightfield deformation) and why
            # that's still a load-grounded SIZE, not an arbitrary one.
            crash_weight_n = (specs.dry_mass_kg + specs.descent_propellant_kg) * 1.62
            crash_contact_width_m = max(1.0, tile.res_m * 6.0)
            crash_sinkage_m = max(0.25, min(
                0.6, bekker_sinkage_m(crash_weight_n * 10.0, crash_contact_width_m, BekkerSoilParams())))
            last_stamp_xy = (float(translation[0]), float(translation[1]))
            if args_cli.enable_impact_marks:
                _spawn_impact_mark(last_stamp_xy[0], last_stamp_xy[1], ground_z_here, crash_contact_width_m * 0.6)
        elif touchdown_step is not None and in_contact:
            # dragging/tumbling after the initial impact -- lay down more
            # of the trail only once the vehicle has actually moved a bit,
            # so a vehicle just rotating/rocking in place doesn't spam a
            # pile of overlapping decals at the same spot.
            dx = float(translation[0]) - last_stamp_xy[0]
            dy = float(translation[1]) - last_stamp_xy[1]
            if np.hypot(dx, dy) > 0.3:
                if args_cli.enable_impact_marks:
                    _spawn_impact_mark(
                        float(translation[0]), float(translation[1]), ground_z_here, crash_contact_width_m * 0.45)
                last_stamp_xy = (float(translation[0]), float(translation[1]))

        ang_vel = imu.data.ang_vel_b
        lin_acc = imu.data.lin_acc_b
        ang_vel_np = ang_vel[0].cpu().numpy() if hasattr(ang_vel, "cpu") else np.asarray(ang_vel[0])
        lin_acc_np = lin_acc[0].cpu().numpy() if hasattr(lin_acc, "cpu") else np.asarray(lin_acc[0])

        t_s = step * sim_dt

        lidar_hit_count = 0
        if step % args_cli.lidar_every_n_frames == 0:
            pc = get_point_cloud(lidar_sensor)
            lidar_hit_count = int(pc["x_m"].size)
            if lidar_hit_count > 0:
                scan_filename = f"scan_{lidar_scan_i:04d}.ply"
                scan_path = os.path.join(lidar_dir, scan_filename)
                export_rtx_point_cloud(pc, scan_path)
                lidar_writer.writerow([lidar_scan_i, scan_filename, step, round(t_s, 4), lidar_hit_count])
                lidar_scan_i += 1
        writer.writerow({
            "frame": step, "t_s": round(t_s, 4),
            "pos_x_m": round(float(translation[0]), 4),
            "pos_y_m": round(float(translation[1]), 4),
            "pos_z_m": round(float(translation[2]), 4),
            "alt_m": round(belly_z - ground_z_here, 4),
            "roll_deg": round(float(euler[0]), 3),
            "pitch_deg": round(float(euler[1]), 3),
            "yaw_deg": round(float(euler[2]), 3),
            "ang_vel_x_rad_s": round(float(ang_vel_np[0]), 6),
            "ang_vel_y_rad_s": round(float(ang_vel_np[1]), 6),
            "ang_vel_z_rad_s": round(float(ang_vel_np[2]), 6),
            "lin_acc_x_m_s2": round(float(lin_acc_np[0]), 6),
            "lin_acc_y_m_s2": round(float(lin_acc_np[1]), 6),
            "lin_acc_z_m_s2": round(float(lin_acc_np[2]), 6),
            "lidar_hit_count": lidar_hit_count,
        })

        if step % render_stride == 0:
            _aim_chase_camera(translation)
            for cam, cam_dir in ((nav_camera, nav_dir), (chase_camera, chase_dir)):
                rgb = cam.data.output.get("rgb")
                if rgb is None:
                    continue
                rgb_np = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
                frame_u8 = np.clip(rgb_np[..., :3], 0, 255).astype(np.uint8)
                imageio.imwrite(os.path.join(cam_dir, f"frame_{saved_frames:04d}.png"), frame_u8)
            frames_writer.writerow([saved_frames, step, round(t_s, 4)])
            saved_frames += 1

        if step % (render_stride * 20) == 0:
            print(f"  step {step}/{max_steps} (t={t_s:.1f}s alt={belly_z - ground_z_here:.2f}m) "
                  f"lin_acc_z={lin_acc_np[2]:.2f} m/s^2")

        if touchdown_step is not None and step >= touchdown_step + post_touchdown_steps:
            break

print(f"saved {saved_frames} frame pairs to {chase_dir} / {nav_dir}")
print(f"wrote telemetry ({step + 1} rows) -> {telemetry_path}")
print(f"wrote {lidar_scan_i} lidar scans -> {lidar_dir}")
if touchdown_step is None:
    print(f"WARNING: never detected ground contact within {args_cli.max_duration_s:.0f}s -- "
          f"check --spawn-altitude-max-m / --tile-size-m sizing.")
simulation_app.close()
