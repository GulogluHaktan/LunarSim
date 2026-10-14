"""Isaac Sim (RTX) render of a RECORDED, validated landing.

`scripts/record_guided_gain_isaac.py` records every control step of the
controller and the guided-gain policy through the same `IsaacLanderVecEnv` and
wave seeds the paired validation used (and checks the outcomes match it
episode for episode). This script replays one recorded episode inside Isaac Sim
with the real Apollo LM asset, the terrain tile that episode actually flew
over, sun lighting and the regolith material -- the same visual pipeline as
`isaaclab_landing_render.py`. The vehicle is visual-only and posed from the
recording each frame, so nothing is re-simulated: what is on screen is the
validated flight, not a new one.

Geometry. The recording stores belly clearance and the ground height under the
footprint, exactly as `contact_clearance_m` defines them, so the body-centre
height is recovered by inverting that function:
    z_centre = ground_z + clearance + half_height*cos(tilt) + stance_radius*sin(tilt)
and the LM asset's root sits at the body centre (`spawn_apollo_lm` shifts the
visual mesh so its footpads are at -height/2).

Terrain. The tile is the recorded height array, resampled linearly for a
smoother render surface. Linear resampling stays on the triangle mesh PhysX
collided with, so the ground on screen is the ground that was flown.

Usage (native Isaac Lab, no docker):
  ~/isaac-env/bin/python scripts/isaaclab_replay_recording.py \
      --recording out/recordings/wave90000_policy.npz --env 10 \
      --label "RL politika" --out out/videos/policy_env10.mp4
"""
import argparse
import json
import os
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--recording", type=str, required=True)
parser.add_argument("--env", type=int, required=True)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--accent", type=str, default="#3987e5",
                    help="label colour (dataviz reference palette: policy #3987e5, "
                         "controller #d95926)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--speedup", type=float, default=3.0)
parser.add_argument("--slow-below-m", type=float, default=15.0)
parser.add_argument("--hold-s", type=float, default=3.0)
parser.add_argument("--upsample", type=int, default=3)
parser.add_argument("--max-frames", type=int, default=0, help="debug: stop after N frames")
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import build_render_mesh  # noqa: E402
from lunarsim.adapters.isaac.lander import spawn_apollo_lm  # noqa: E402
from lunarsim.adapters.isaac.lighting import (  # noqa: E402
    create_sun_light, disable_ambient, set_no_ambient_render_settings)
from lunarsim.adapters.isaac.materials import create_regolith_material  # noqa: E402
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map  # noqa: E402
from lunarsim.core.lighting.sun import SunPosition  # noqa: E402
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs  # noqa: E402

SPECS = ApolloLMSpecs()
HALF_H = SPECS.height_m / 2.0
STANCE = SPECS.footpad_span_m / 2.0
V_XY_LIMIT, V_Z_LIMIT = 1.2, 1.0
GOOD, CRIT, INK, INK2 = (12, 163, 12), (208, 59, 59), (255, 255, 255), (195, 194, 183)


# ---------------------------------------------------------------- recording
d = np.load(args_cli.recording)
i = args_cli.env
g = lambda k: d[f"e{i}_{k}"]  # noqa: E731
x, y, alt, ground = g("x"), g("y"), g("alt"), g("ground_z")
tx, ty, yaw = g("tilt_x"), g("tilt_y"), g("yaw")
vx, vy, vz = g("vx"), g("vy"), g("vz")
cmd = g("cmd")
outcome = json.loads(str(g("outcome")))
dt = float(d["dt"])
tilt = np.hypot(tx, ty)
z_centre = ground + alt + HALF_H * np.cos(tilt) + STANCE * np.sin(tilt)
vxy = np.hypot(vx, vy)
t = np.arange(len(x)) * dt
T = t[-1]
print(f"[replay] env {i}: {len(x)} steps, {T:.1f} s, landed={outcome['landed']}", flush=True)


class _Tile:
    """The two fields `build_render_mesh` reads, from the recorded tile."""

    def __init__(self, height, res_m, size_m):
        self.height, self.res_m, self.size_m = height, res_m, size_m


h0 = g("tile_height").astype(np.float64)
res0, size0 = float(g("tile_res")), float(g("tile_size"))
# Background only: extend the tile by edge replication so the hard edge of the
# flown tile is not in frame. The vehicle never leaves the original tile (the
# env truncates at its boundary), so every metre it flies over is unchanged;
# the padding is scenery beyond it. The mesh is centred, so symmetric padding
# keeps the original tile's coordinates exactly.
pad = h0.shape[0]
h0 = np.pad(h0, pad, mode="edge")
size0 *= 3.0
up = max(1, int(args_cli.upsample))
n0 = h0.shape[0]
src = np.arange(n0)
dst = np.linspace(0, n0 - 1, (n0 - 1) * up + 1)
# separable linear resample: stays on the piecewise-linear surface PhysX used
h1 = np.array([np.interp(dst, src, row) for row in h0])
h_up = np.array([np.interp(dst, src, col) for col in h1.T]).T
tile = _Tile(h_up.astype(np.float32), res0 / up, size0)

# ---------------------------------------------------------------- scene
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0"))
stage = omni.usd.get_context().get_stage()
render_mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=0, uv_tile_size_m=3.0)
normal_map_path = "/tmp/lunarsim_regolith_normal_replay.png"
imageio.imwrite(normal_map_path, bake_regolith_normal_map(
    resolution=1024, physical_size_m=3.0, amplitude_m=0.09, wavelength_m=0.12, seed=7))
material = create_regolith_material(stage, "/World/Looks/Regolith", albedo=0.11, brdf="albedo",
                                    roughness=0.95, normal_map_path=normal_map_path)
UsdShade.MaterialBindingAPI.Apply(render_mesh.GetPrim()).Bind(material)

create_sun_light(stage, "/World/Sun", SunPosition(elevation_deg=18.0, azimuth_deg=135.0),
                 angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()

lm_root = spawn_apollo_lm(stage, "/World/LM", os.path.join(
    args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz"), visual_only=True)
xform_api = UsdGeom.XformCommonAPI(lm_root.GetPrim())


def _emissive(path, color, intensity):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[c * intensity for c in color]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


flame = UsdGeom.Cone.Define(stage, "/World/LM/DPS_Flame")
flame.CreateHeightAttr(1.0); flame.CreateRadiusAttr(0.3); flame.CreateAxisAttr("Z")
fxf = UsdGeom.Xformable(flame.GetPrim()); fxf.ClearXformOpOrder()
flame_t, flame_s = fxf.AddTranslateOp(), fxf.AddScaleOp()
UsdShade.MaterialBindingAPI.Apply(flame.GetPrim()).Bind(_emissive("/World/Looks/Flame", (1.0, 0.55, 0.1), 8.0))

camera = Camera(cfg=CameraCfg(
    prim_path="/World/ChaseCamera", update_period=0, height=args_cli.height, width=args_cli.width,
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
        focal_length=22.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5))))
sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()
for _ in range(40):
    sim.step(render=True)
    camera.update(dt=sim.get_physics_dt())

# ---------------------------------------------------------------- schedule
t_slow = float(t[np.argmax(alt < args_cli.slow_below_m)]) if (alt < args_cli.slow_below_m).any() else T
times, tt = [], 0.0
while tt < T:
    times.append(tt)
    tt += (args_cli.speedup if tt < t_slow else 1.0) / args_cli.fps
times += [T] * int(args_cli.hold_s * args_cli.fps)
if args_cli.max_frames:
    times = times[: args_cli.max_frames]

try:
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 22)
    font_b = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 30)
    font_m = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 20)
except OSError:
    font = font_b = font_m = ImageFont.load_default()
accent = tuple(int(args_cli.accent.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))


def hud(frame, k, final):
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    dr.rectangle([0, 0, W, 44], fill=(13, 13, 13, 200))
    dr.text((14, 8), args_cli.label, font=font_b, fill=accent)
    dr.rectangle([10, 56, 300, 200], fill=(13, 13, 13, 170))
    lines = [f"t      {t[k]:5.1f} s", f"irtifa {alt[k]:6.1f} m",
             f"yanal  {vxy[k]:5.2f} m/s", f"dikey  {abs(vz[k]):5.2f} m/s",
             f"yatış  {np.degrees(tilt[k]):5.1f}°"]
    for j, s in enumerate(lines):
        col = INK
        if j == 2 and alt[k] < 15:
            col = CRIT if vxy[k] > V_XY_LIMIT else INK
        dr.text((22, 64 + 26 * j), s, font=font_m, fill=col)
    if final:
        imp = outcome["impact"]
        ivxy = float(np.hypot(imp.get("vx", np.nan), imp.get("vy", np.nan)))
        ivz = abs(float(imp.get("vz", np.nan)))
        if outcome["landed"]:
            txt, col = f"İNDİ  ✓   yanal {ivxy:.2f} m/s  ·  dikey {ivz:.2f} m/s", GOOD
        else:
            why = (f"yanal {ivxy:.2f} > {V_XY_LIMIT} m/s" if ivxy > V_XY_LIMIT
                   else f"dikey {ivz:.2f} > {V_Z_LIMIT} m/s")
            txt, col = f"ÇAKILDI  ✕   {why}", CRIT
        tw = dr.textlength(txt, font=font_b)
        dr.rectangle([(W - tw) / 2 - 18, H - 78, (W + tw) / 2 + 18, H - 26], fill=(13, 13, 13, 215))
        dr.text(((W - tw) / 2, H - 72), txt, font=font_b, fill=col)
    return np.asarray(im)


frames_dir = os.path.splitext(args_cli.out)[0] + "_frames"
os.makedirs(frames_dir, exist_ok=True)
print(f"[replay] rendering {len(times)} frames -> {frames_dir}", flush=True)
index_rows = []  # (frame number, recording step) -- lets a dashboard sync to each frame
for f, tt in enumerate(times):
    k = int(min(np.searchsorted(t, tt, side="right") - 1, len(t) - 1))
    xform_api.SetTranslate(Gf.Vec3d(float(x[k]), float(y[k]), float(z_centre[k])))
    xform_api.SetRotate(Gf.Vec3f(float(np.degrees(tx[k])), float(np.degrees(ty[k])),
                                 float(np.degrees(yaw[k]))), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    thr = float(np.clip((cmd[k, 0] + 1.0) / 2.0, 0.0, 1.0))
    if thr > 0.03 and alt[k] > 0.05:
        UsdGeom.Imageable(flame.GetPrim()).MakeVisible()
        fh = 0.8 + 3.5 * thr
        flame_t.Set(Gf.Vec3d(0.0, 0.0, -HALF_H - fh / 2))
        flame_s.Set(Gf.Vec3f(0.5 + 0.5 * thr, 0.5 + 0.5 * thr, fh))
    else:
        UsdGeom.Imageable(flame.GetPrim()).MakeInvisible()
    # chase camera: pulls back with altitude so the approach and the ground
    # are both in frame, closes in for touchdown
    dist = float(np.clip(16.0 + 0.22 * alt[k], 18.0, 55.0))
    heading = np.arctan2(vy[k], vx[k]) if vxy[k] > 0.5 else 0.0
    cx = x[k] - dist * np.cos(heading) + 0.45 * dist * np.sin(heading)
    cy = y[k] - dist * np.sin(heading) - 0.45 * dist * np.cos(heading)
    # High up, sit slightly BELOW the vehicle so the horizon and the black sky
    # are in frame and the altitude reads; near the ground, look down on it
    # so the touchdown and the footpads are visible.
    cz = z_centre[k] + float(np.interp(alt[k], [6.0, 60.0], [0.32, -0.06])) * dist
    camera.set_world_poses_from_view(
        torch.tensor([[cx, cy, cz]], device=sim.device, dtype=torch.float32),
        torch.tensor([[x[k], y[k], z_centre[k] - 1.5]], device=sim.device, dtype=torch.float32))
    for _ in range(2):  # let the RTX accumulator settle on the new pose
        sim.step(render=True)
    camera.update(dt=sim.get_physics_dt())
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        continue
    arr = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
    frame = np.clip(arr[..., :3], 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(frames_dir, f"f{f:05d}.png"), hud(frame, k, k == len(t) - 1))
    index_rows.append((f, k))
    if f % 60 == 0:
        print(f"[replay]   frame {f}/{len(times)}  t={t[k]:.1f}s alt={alt[k]:.1f}m", flush=True)

with open(os.path.join(frames_dir, "index.csv"), "w") as fh:
    fh.write("frame,step\n")
    fh.writelines(f"{a},{b}\n" for a, b in index_rows)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args_cli.fps),
                "-i", os.path.join(frames_dir, "f%05d.png"), "-c:v", "libx264", "-pix_fmt",
                "yuv420p", "-crf", "18", args_cli.out], check=True)
print(f"[replay] wrote {args_cli.out}", flush=True)
simulation_app.close()
