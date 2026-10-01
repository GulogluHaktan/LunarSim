"""Renders a real Isaac Sim video of a trained SAC policy's landing
rollout: the real Apollo LM asset (`lunarsim.adapters.isaac.lander`,
correctly-scaled visual mesh + real mass/RCS locators), a real terrain
tile (`lunarsim.adapters.isaac.heightfield`), real sun lighting/regolith
material -- the same validated Isaac pipeline `isaaclab_orbit_demo.py`
uses, but animating the vehicle kinematically along a precomputed
trajectory (from `scripts/render_landing_video.py`'s JSON export) instead
of a static orbit camera.

The vehicle is spawned with `physics:kinematicEnabled=True` so its pose
is driven directly from the trajectory each frame -- PhysX doesn't try to
simulate it (there's no physics to check here, this is a visual replay of
an already-computed analytic rollout, not a re-simulation).

Run via scripts/run_isaaclab_landing_render.sh.
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--trajectory-json", type=str, default="/workspace/LunarSim/out/landing_trajectory.json")
parser.add_argument("--out-dir", type=str, default="/workspace/LunarSim/out/landing_frames")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--cam-mode", type=str, default="chase", choices=["chase", "fixed"])
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
from isaaclab.sensors.camera import Camera, CameraCfg
from pxr import Gf, UsdGeom, UsdShade

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, build_render_mesh
from lunarsim.adapters.isaac.lander import spawn_apollo_lm
from lunarsim.adapters.isaac.lighting import (
    create_sun_light, disable_ambient, set_no_ambient_render_settings,
)
from lunarsim.adapters.isaac.materials import create_regolith_material
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile

os.makedirs(args_cli.out_dir, exist_ok=True)

with open(args_cli.trajectory_json) as f:
    traj = json.load(f)
history = traj["history"]
print(f"loaded trajectory: {len(history)} steps, landed={traj['landed']}")

sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"))
sim = sim_utils.SimulationContext(sim_cfg)

stage = omni.usd.get_context().get_stage()

# flat terrain -- matches the hover_only training stage's
# terrain_roughness_scale=0.0 (see scripts/train_sac_curriculum.py); the
# rollout's physics assumed this exact (flat) ground, so the HEIGHT must
# stay untouched to match what the policy actually flew over -- do not
# add hills/craters here, no matter how much nicer non-flat terrain would
# look. Only material/texture/decoration below are allowed to change.
cfg = TerrainConfig(
    mode="fine", size_m=200.0, res_m=1.0, seed=1,
    coarse_source="procedural",
    hills={"amplitude_m": 0.0, "wavelength_m": 20.0, "hurst": 0.75},
    craters={"count_scale": 0.0, "d_min_m": 1.0, "d_max_m": 10.0, "b": 2.5,
             "depth_ratio": 0.08, "age": 0.5},
    rocks={"density_scale": 0.0, "d_max_m": 0.5},
    roi={"sigma_m": 40.0, "centers": None},
    curvature=False,
)
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)
render_mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=2, uv_tile_size_m=2.0)

# real-scale micro-bump normal map -- REAL BUG FOUND VIA A RENDER: skipping
# this (plain flat-albedo material only) made the ground look like a flat
# gray plane, not lunar regolith, even though the mesh itself is real DEM-
# generated terrain (see lunarsim/adapters/isaac/README.md's own note that
# this is the standard way close-up regolith texture gets added on top of
# the coarser mesh shape).
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map

normal_map_path = "/tmp/lunarsim_regolith_normal.png"
# amplitude/wavelength pushed beyond the module defaults (0.015m/0.15m) --
# this is a TEXTURE (tangent-space normal map), not terrain height, so it
# can't disturb the flat collision the rollout assumed; it just needs to
# read as regolith at chase-cam distance instead of a flat gray plane.
normal_map = bake_regolith_normal_map(resolution=1024, physical_size_m=2.0, amplitude_m=0.04, wavelength_m=0.08, seed=cfg.seed)
imageio.imwrite(normal_map_path, normal_map)

material = create_regolith_material(
    stage, "/World/Looks/Regolith", albedo=0.09, brdf="albedo", roughness=0.95, normal_map_path=normal_map_path
)
UsdShade.MaterialBindingAPI.Apply(render_mesh.GetPrim()).Bind(material)

# -- decorative rocks: visual only, well outside the flight path, so they
# can't interact with the (flat, physics-matching) height field or the
# vehicle at all -- purely so the ground doesn't read as an empty plane.
from lunarsim.core.terrain.rocks import sample_height_at

rng = np.random.default_rng(7)
n_rocks = 22
for ri in range(n_rocks):
    ang = rng.uniform(0, 2 * np.pi)
    rad = rng.uniform(18.0, 55.0)  # well outside the hover_only flight envelope
    rx, ry = rad * np.cos(ang), rad * np.sin(ang)
    rz = float(sample_height_at(tile.height, tile.res_m, np.array([rx]), np.array([ry]))[0])
    size = rng.uniform(0.3, 1.4)
    rock = UsdGeom.Sphere.Define(stage, f"/World/Rocks/rock_{ri}")
    rock.CreateRadiusAttr(0.5)
    xf = UsdGeom.Xformable(rock.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, rz + size * 0.15))
    xf.AddScaleOp().Set(Gf.Vec3f(size, size * rng.uniform(0.6, 1.0), size * rng.uniform(0.5, 0.85)))
    UsdShade.MaterialBindingAPI.Apply(rock.GetPrim()).Bind(material)

sun_pos = SunPosition(elevation_deg=30.0, azimuth_deg=140.0)
create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()

lm_asset_path = os.path.join(args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz")
# visual_only=True: no RigidBodyAPI/collision at all, so nothing in PhysX
# fights the pose we set frame-by-frame below (see spawn_apollo_lm's
# docstring for the render that caught this the first time).
lm_root = spawn_apollo_lm(stage, "/World/LM", lm_asset_path, visual_only=True)

xform_api = UsdGeom.XformCommonAPI(lm_root.GetPrim())


def _emissive_material(path: str, color: tuple[float, float, float], intensity: float = 5.0):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * intensity for c in color]))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


# -- DPS exhaust flame + RCS axis indicators: children of the vehicle root
# (NOT the scaled "Visual" child), so they sit in real-meter local space
# and inherit the vehicle's world pose each frame automatically. These are
# illustrative, not a physically exact per-jet plume -- see the user's
# request to be able to tell "which one is firing" from the video.
flame_mat = _emissive_material("/World/Looks/DPSFlame", (1.0, 0.55, 0.1), intensity=8.0)
flame = UsdGeom.Cone.Define(stage, "/World/LM/DPS_Flame")
flame.CreateHeightAttr(1.0)
flame.CreateRadiusAttr(0.3)
flame.CreateAxisAttr("Z")
flame_xf = UsdGeom.Xformable(flame.GetPrim())
flame_xf.ClearXformOpOrder()
flame_translate_op = flame_xf.AddTranslateOp()
flame_scale_op = flame_xf.AddScaleOp()
UsdShade.MaterialBindingAPI.Apply(flame.GetPrim()).Bind(flame_mat)

rcs_markers = {}
rcs_colors = {"pitch": (1.0, 0.1, 0.1), "roll": (0.1, 1.0, 0.1), "yaw": (0.1, 0.4, 1.0)}
rcs_local_pos = {"pitch": (1.6, 0.0, 3.2), "roll": (0.0, 1.6, 3.2), "yaw": (-1.6, 0.0, 3.2)}
for axis, color in rcs_colors.items():
    mat = _emissive_material(f"/World/Looks/RCS_{axis}", color, intensity=6.0)
    marker = UsdGeom.Sphere.Define(stage, f"/World/LM/RCS_{axis}_marker")
    marker.CreateRadiusAttr(0.35)
    mxf = UsdGeom.Xformable(marker.GetPrim())
    mxf.ClearXformOpOrder()
    t_op = mxf.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*rcs_local_pos[axis]))
    s_op = mxf.AddScaleOp()
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(mat)
    rcs_markers[axis] = (marker, s_op)

camera_cfg = CameraCfg(
    prim_path="/World/ChaseCamera",
    update_period=0,
    height=args_cli.height,
    width=args_cli.width,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
)
camera = Camera(cfg=camera_cfg)

sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()

print("warming up renderer...")
for _ in range(30):
    sim.step(render=True)
    camera.update(dt=sim.get_physics_dt())


def set_pose(x, y, z, tilt_x, tilt_y, yaw):
    # small-angle roll(x)-pitch(y)-yaw(z), matching lunarsim.rl's own
    # _euler_to_quat convention -- degrees for UsdGeom.XformCommonAPI.
    xform_api.SetTranslate(Gf.Vec3d(x, y, z))
    xform_api.SetRotate(Gf.Vec3f(np.degrees(tilt_x), np.degrees(tilt_y), np.degrees(yaw)),
                         UsdGeom.XformCommonAPI.RotationOrderXYZ)


# subsample to the target render fps: rollout dt is 0.05s (20 Hz)
sim_dt = history[1]["t"] - history[0]["t"] if len(history) > 1 else 0.05
step_stride = max(1, round((1.0 / args_cli.fps) / sim_dt))
frames_idx = list(range(0, len(history), step_stride))

print(f"rendering {len(frames_idx)} frames (stride={step_stride}, sim_dt={sim_dt:.3f}s)...")
saved = 0
for out_i, i in enumerate(frames_idx):
    h = history[i]
    set_pose(h["x"], h["y"], h["z"], h["tilt_x"], h["tilt_y"], 0.0)

    # DPS flame: grows with throttle, hidden when the engine is off
    throttle = h.get("throttle", 0.0)
    if throttle > 0.03:
        UsdGeom.Imageable(flame.GetPrim()).MakeVisible()
        flame_h = 0.6 + 3.0 * throttle
        flame_translate_op.Set(Gf.Vec3d(0.0, 0.0, -3.2 - flame_h / 2))
        flame_scale_op.Set(Gf.Vec3f(0.4 + 0.4 * throttle, 0.4 + 0.4 * throttle, flame_h))
    else:
        UsdGeom.Imageable(flame.GetPrim()).MakeInvisible()

    # RCS markers: light up + grow with |command| on their axis
    rcs_cmd = {"pitch": h.get("rcs_pitch", 0.0), "roll": h.get("rcs_roll", 0.0), "yaw": h.get("rcs_yaw", 0.0)}
    for axis, (marker, s_op) in rcs_markers.items():
        mag = abs(rcs_cmd[axis])
        if mag > 0.05:
            UsdGeom.Imageable(marker.GetPrim()).MakeVisible()
            s_op.Set(Gf.Vec3f(0.3 + 0.9 * mag))
        else:
            UsdGeom.Imageable(marker.GetPrim()).MakeInvisible()

    if args_cli.cam_mode == "chase":
        cam_pos = torch.tensor([[h["x"] - 12.0, h["y"] - 12.0, h["z"] + 6.0]], device=sim.device, dtype=torch.float32)
        target = torch.tensor([[h["x"], h["y"], h["z"]]], device=sim.device, dtype=torch.float32)
    else:
        cam_pos = torch.tensor([[15.0, -15.0, history[0]["z"] * 0.6 + 10.0]], device=sim.device, dtype=torch.float32)
        target = torch.tensor([[0.0, 0.0, history[0]["ground_z"]]], device=sim.device, dtype=torch.float32)
    camera.set_world_poses_from_view(cam_pos, target)

    sim.step(render=True)
    camera.update(dt=sim.get_physics_dt())

    rgb = camera.data.output.get("rgb")
    if rgb is None:
        continue
    rgb_np = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
    frame_u8 = np.clip(rgb_np[..., :3], 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(args_cli.out_dir, f"frame_{out_i:04d}.png"), frame_u8)
    saved += 1
    if out_i % 20 == 0:
        print(f"  frame {out_i}/{len(frames_idx)} (t={h['t']:.1f}s alt={h['z']-h['ground_z']:.1f}m) saved")

print(f"saved {saved}/{len(frames_idx)} frames to {args_cli.out_dir}")
simulation_app.close()
