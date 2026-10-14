"""Replay a recorded, validated landing in Isaac Sim WITH the descent sensors,
writing the same capture directory the old eval-capture scripts write, so the
existing `scripts/compose_capture_video.py` turns it into the original
4-panel dashboard (chase camera, nav camera, IMU, live LiDAR terrain map)
without a single change to that script.

Why a replay and not `isaaclab_controller_eval_capture.py` with a policy in the
loop: that script builds its own scene, spawn distribution and actuator layer,
so a clip from it would show a different system from the one that scored
172/192 in the paired validation. Here the vehicle is posed from
`record_guided_gain_isaac.py`'s recording, which reproduces the validation wave
episode for episode. The sensors are real, though: the RTX LiDAR and the
cameras are children of the vehicle prim and see the rendered terrain tile that
episode actually flew over, so the map is a genuine scan of that ground.

Sensor mounts are the old capture's exactly: OS2 profile at 10 Hz,
`descent_sensor_mount_z_m(specs)` up the body axis, `LIDAR_MOUNT_TILT_DEG`
forward tilt; nav camera 0.5 m above the contact plane looking straight down.

Output layout (what compose_capture_video.py reads):
  chase_frames/frame_XXXX.png, nav_frames/frame_XXXX.png
  telemetry.csv       one row per recorded control step (20 Hz)
  frames_index.csv    frame_index, step, t_s  -- step indexes telemetry.csv
  lidar_index.csv     scan_index, filename, step, t_s, hit_count, sensor pose
  lidar_scans/scan_XXXX.ply   world-frame points

telemetry.csv notes, since a replay has no IMU: ang_vel is the recorded body
rate; lin_acc is the finite-difference world acceleration of the recorded
velocity plus lunar gravity on z (the proper acceleration an IMU would read,
expressed in the world frame -- the tilts here stay under ~30 deg).

Usage:
  ~/isaac-env/bin/python scripts/isaaclab_replay_capture.py \
      --recording out/recordings/wave90000_policy.npz --env 10 \
      --out-dir out/videos/rl_capture_env10
  ~/isaac-env/bin/python scripts/compose_capture_video.py \
      --capture-dir out/videos/rl_capture_env10 --fps 30 --pose-lag-s 0
"""
import argparse
import csv
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--recording", type=str, required=True)
parser.add_argument("--env", type=int, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--speedup", type=float, default=2.0)
parser.add_argument("--slow-below-m", type=float, default=15.0)
parser.add_argument("--hold-s", type=float, default=2.5)
parser.add_argument("--lidar-config", type=str, default="OS2")
parser.add_argument("--lidar-every-s", type=float, default=0.25,
                    help="recorded-time spacing between kept scans (the old captures kept one per 0.25 s)")
parser.add_argument("--lidar-settle-steps", type=int, default=8,
                    help="render steps with the pose held before a scan is read, so the 10 Hz "
                         "sensor completes a full rotation from that pose")
parser.add_argument("--settle-steps", type=int, default=2,
                    help="render steps per frame that carries no LiDAR scan; the RTX image lags the "
                         "pose, so too few makes the vehicle appear to hover and snap to the ground")
parser.add_argument("--t-start", type=float, default=0.0, help="skip frames before this time (debug)")
parser.add_argument("--upsample", type=int, default=3)
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--no-rtx-lidar", action="store_true",
                    help="skip the RTX LiDAR (cameras + telemetry only); scans can then be added by "
                         "scripts/synth_lidar_scans.py, which ray-casts the recorded terrain")
parser.add_argument("--sim-device", type=str, default="cpu",
                    help="the old LiDAR captures ran with device=cpu; RTX LiDAR has only been seen working that way")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdShade  # noqa: E402

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import build_render_mesh  # noqa: E402
from lunarsim.adapters.isaac.lander import spawn_apollo_lm  # noqa: E402
from lunarsim.adapters.isaac.lighting import (  # noqa: E402
    create_sun_light, disable_ambient, set_no_ambient_render_settings)
from lunarsim.adapters.isaac.materials import create_regolith_material  # noqa: E402
from lunarsim.adapters.isaac.sensors import (  # noqa: E402
    LIDAR_MOUNT_TILT_DEG, create_rtx_lidar, descent_sensor_mount_z_m,
    export_rtx_point_cloud, get_point_cloud, lidar_mount_pose)
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map  # noqa: E402
from lunarsim.core.lighting.sun import SunPosition  # noqa: E402
from lunarsim.core.metadata.lidar import euler_to_rotation_matrix  # noqa: E402
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs  # noqa: E402
from lunarsim.rl.action_map import action_to_throttle  # noqa: E402
from lunarsim.rl.analytic_lander_env import LanderParams  # noqa: E402

SPECS = ApolloLMSpecs()
P = LanderParams()
HALF_H, STANCE = SPECS.height_m / 2.0, SPECS.footpad_span_m / 2.0

# ------------------------------------------------------------------ recording
d = np.load(args_cli.recording)
i = args_cli.env
g = lambda k: d[f"e{i}_{k}"]  # noqa: E731
x, y, alt, ground = g("x"), g("y"), g("alt"), g("ground_z")
tx, ty, yaw = g("tilt_x"), g("tilt_y"), g("yaw")
vx, vy, vz = g("vx"), g("vy"), g("vz")
wx, wy, wz = g("wx"), g("wy"), g("wz")
fuel = g("fuel_kg")
cmd = g("cmd")
outcome = json.loads(str(g("outcome")))
dt = float(d["dt"])
tilt = np.hypot(tx, ty)
z_c = ground + alt + HALF_H * np.cos(tilt) + STANCE * np.sin(tilt)
t = np.arange(len(x)) * dt
T = float(t[-1])
acc = np.stack([np.gradient(v, dt) for v in (vx, vy, vz)], axis=1)
acc[:, 2] += P.gravity_m_s2
print(f"[capture] env {i}: {len(x)} steps, {T:.1f} s, landed={outcome['landed']}", flush=True)


class _Tile:
    def __init__(self, height, res_m, size_m):
        self.height, self.res_m, self.size_m = height, res_m, size_m


h0 = g("tile_height").astype(np.float64)
res0, size0 = float(g("tile_res")), float(g("tile_size"))
pad = h0.shape[0]
h0 = np.pad(h0, pad, mode="edge")  # background beyond the flown tile; centred, so coords unchanged
size0 *= 3.0
up = max(1, int(args_cli.upsample))
src = np.arange(h0.shape[0])
dst = np.linspace(0, h0.shape[0] - 1, (h0.shape[0] - 1) * up + 1)
h1 = np.array([np.interp(dst, src, r) for r in h0])
tile = _Tile(np.array([np.interp(dst, src, c) for c in h1.T]).T.astype(np.float32), res0 / up, size0)

# ------------------------------------------------------------------ scene
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.sim_device))
stage = omni.usd.get_context().get_stage()
mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=0, uv_tile_size_m=3.0)
nmap = "/tmp/lunarsim_regolith_normal_capture.png"
imageio.imwrite(nmap, bake_regolith_normal_map(resolution=1024, physical_size_m=3.0,
                                               amplitude_m=0.09, wavelength_m=0.12, seed=7))
mat = create_regolith_material(stage, "/World/Looks/Regolith", albedo=0.11, brdf="albedo",
                               roughness=0.95, normal_map_path=nmap)
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
create_sun_light(stage, "/World/Sun", SunPosition(elevation_deg=18.0, azimuth_deg=135.0),
                 angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()

lm_root = spawn_apollo_lm(stage, "/World/LM", os.path.join(
    args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz"), visual_only=True)
xform_api = UsdGeom.XformCommonAPI(lm_root.GetPrim())

nav_camera = Camera(cfg=CameraCfg(
    prim_path="/World/LM/Sensors/NavCam", update_period=0, height=args_cli.height,
    width=args_cli.width, data_types=["rgb"],
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -SPECS.height_m * 0.5 + 0.5),
                               rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
    spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955,
                                     clipping_range=(0.05, 1.0e4))))
chase_camera = Camera(cfg=CameraCfg(
    prim_path="/World/ChaseCamera", update_period=0, height=args_cli.height, width=args_cli.width,
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
        focal_length=22.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5))))

lidar = None
if not args_cli.no_rtx_lidar:
    print(f"[capture] RTX LiDAR {args_cli.lidar_config} on the vehicle", flush=True)
    lidar = create_rtx_lidar("/World/LM/Sensors/Lidar", config=args_cli.lidar_config, tick_rate=10.0)
    lxf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/LM/Sensors/Lidar"))
    lxf.ClearXformOpOrder()
    lxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, descent_sensor_mount_z_m(SPECS)))
    lxf.AddRotateYOp().Set(LIDAR_MOUNT_TILT_DEG)

sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()


# Chase-camera heading: follow the velocity direction while moving, HOLD the last heading when
# the horizontal speed drops under 0.5 m/s (it used to snap to 0 there, which jumped the camera
# to the other side of the vehicle right at touchdown), and low-pass it so it never jumps.
_sp = np.hypot(vx, vy)
_raw = np.zeros(len(vx))
_last = float(np.arctan2(vy[np.argmax(_sp > 0.5)], vx[np.argmax(_sp > 0.5)])) if (_sp > 0.5).any() else 0.0
for _k in range(len(vx)):
    if _sp[_k] > 0.5:
        _last = float(np.arctan2(vy[_k], vx[_k]))
    _raw[_k] = _last
_raw = np.unwrap(_raw)
HEAD = np.empty_like(_raw)
_a = 1.0 - np.exp(-dt / 1.5)          # ~1.5 s time constant
HEAD[0] = _raw[0]
for _k in range(1, len(_raw)):
    HEAD[_k] = HEAD[_k - 1] + _a * (_raw[_k] - HEAD[_k - 1])


def set_pose(k):
    xform_api.SetTranslate(Gf.Vec3d(float(x[k]), float(y[k]), float(z_c[k])))
    xform_api.SetRotate(Gf.Vec3f(float(np.degrees(tx[k])), float(np.degrees(ty[k])),
                                 float(np.degrees(yaw[k]))), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    dist = float(np.clip(16.0 + 0.22 * alt[k], 18.0, 55.0))
    head = float(HEAD[k])
    cx = x[k] - dist * np.cos(head) + 0.45 * dist * np.sin(head)
    cy = y[k] - dist * np.sin(head) - 0.45 * dist * np.cos(head)
    cz = z_c[k] + float(np.interp(alt[k], [6.0, 60.0], [0.32, -0.06])) * dist
    chase_camera.set_world_poses_from_view(
        torch.tensor([[cx, cy, cz]], device=sim.device, dtype=torch.float32),
        torch.tensor([[x[k], y[k], z_c[k] - 1.5]], device=sim.device, dtype=torch.float32))


def step_render(n):
    for _ in range(n):
        sim.step(render=True)
    nav_camera.update(dt=sim.get_physics_dt())
    chase_camera.update(dt=sim.get_physics_dt())


set_pose(int(min(np.searchsorted(t, args_cli.t_start, side="right") - 1, len(t) - 1)) if args_cli.t_start > 0 else 0)
step_render(40)

# ------------------------------------------------------------------ schedule
slow = alt < args_cli.slow_below_m
t_slow = float(t[np.argmax(slow)]) if slow.any() else T
times, tt = [], 0.0
while tt < T:
    times.append(tt)
    tt += (args_cli.speedup if tt < t_slow else 1.0) / args_cli.fps
times += [T] * int(args_cli.hold_s * args_cli.fps)
if args_cli.t_start > 0:
    times = [v for v in times if v >= args_cli.t_start]
if args_cli.max_frames:
    times = times[: args_cli.max_frames]

out = args_cli.out_dir
for sub in ("chase_frames", "nav_frames", "lidar_scans"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)

with open(os.path.join(out, "telemetry.csv"), "w", newline="") as tf:
    w = csv.writer(tf)
    w.writerow(["frame", "t_s", "pos_x_m", "pos_y_m", "pos_z_m", "alt_m", "roll_deg", "pitch_deg",
                "yaw_deg", "ang_vel_x_rad_s", "ang_vel_y_rad_s", "ang_vel_z_rad_s",
                "lin_acc_x_m_s2", "lin_acc_y_m_s2", "lin_acc_z_m_s2", "throttle", "rcs_pitch",
                "rcs_roll", "rcs_yaw", "fuel_frac", "rcs_fuel_frac", "reward", "lidar_hit_count"])
    for k in range(len(t)):
        w.writerow([k, round(t[k], 4), round(x[k], 4), round(y[k], 4), round(z_c[k], 4),
                    round(alt[k], 4), round(np.degrees(tx[k]), 3), round(np.degrees(ty[k]), 3),
                    round(np.degrees(yaw[k]), 3), round(wx[k], 6), round(wy[k], 6), round(wz[k], 6),
                    round(acc[k, 0], 6), round(acc[k, 1], 6), round(acc[k, 2], 6),
                    round(float(action_to_throttle(cmd[k, 0])), 3), round(float(cmd[k, 1]), 3),
                    round(float(cmd[k, 2]), 3), round(float(cmd[k, 3]), 3),
                    round(float(fuel[k] / P.initial_fuel_kg), 4), "", "", ""])

fi = open(os.path.join(out, "frames_index.csv"), "w", newline="")
li = open(os.path.join(out, "lidar_index.csv"), "w", newline="")
fw, lw = csv.writer(fi), csv.writer(li)
fw.writerow(["frame_index", "step", "t_s"])
lw.writerow(["scan_index", "filename", "step", "t_s", "hit_count", "sensor_x_m", "sensor_y_m",
             "sensor_z_m"] + [f"sensor_r{a}{b}" for a in range(3) for b in range(3)])

scan_i, next_scan_t = 0, 0.0
print(f"[capture] rendering {len(times)} frames -> {out}", flush=True)
for f, tt in enumerate(times):
    k = int(min(np.searchsorted(t, tt, side="right") - 1, len(t) - 1))
    set_pose(k)
    if lidar is not None and t[k] >= next_scan_t and f < len(times) - int(args_cli.hold_s * args_cli.fps):
        step_render(args_cli.lidar_settle_steps)
        R = euler_to_rotation_matrix(float(tx[k]), float(ty[k]), float(yaw[k]))
        pc = get_point_cloud(lidar, sensor_pose=lidar_mount_pose(
            SPECS, np.array([x[k], y[k], z_c[k]]), R))
        n_hit = int(pc["x_m"].size)
        if n_hit > 0:
            fn = f"scan_{scan_i:04d}.ply"
            export_rtx_point_cloud(pc, os.path.join(out, "lidar_scans", fn))
            pos, rot = pc.get("sensor_pos_m"), pc.get("sensor_rotation")
            pose = ([round(float(v), 5) for v in pos] + [round(float(rot[a][b]), 7)
                    for a in range(3) for b in range(3)]) if pos is not None and rot is not None \
                else [""] * 12
            lw.writerow([scan_i, fn, k, round(t[k], 4), n_hit] + pose)
            scan_i += 1
        next_scan_t = t[k] + args_cli.lidar_every_s
    else:
        step_render(args_cli.settle_steps)
    for cam, sub in ((nav_camera, "nav_frames"), (chase_camera, "chase_frames")):
        rgb = cam.data.output.get("rgb")
        if rgb is None:
            continue
        arr = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
        imageio.imwrite(os.path.join(out, sub, f"frame_{f:04d}.png"),
                        np.clip(arr[..., :3], 0, 255).astype(np.uint8))
    fw.writerow([f, k, round(t[k], 4)])
    if f % 60 == 0:
        print(f"[capture]   frame {f}/{len(times)} t={t[k]:.1f}s alt={alt[k]:.1f}m scans={scan_i}",
              flush=True)

fi.close(); li.close()
imp = outcome["impact"]
with open(os.path.join(out, "eval_summary.txt"), "w") as fh:
    fh.write(f"controller=guided-gain PPO (replay of recorded validation episode)\n"
             f"recording={args_cli.recording} env={i}\nlanded_safely={outcome['landed']}\n"
             f"t_s={T:.2f}\ntouchdown_vz={imp.get('vz', float('nan')):.3f}\n"
             f"touchdown_vxy={float(np.hypot(imp.get('vx', 0), imp.get('vy', 0))):.3f}\n")
print(f"[capture] done: {len(times)} frames, {scan_i} lidar scans -> {out}", flush=True)
simulation_app.close()
