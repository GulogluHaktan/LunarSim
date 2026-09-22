"""Orbit camera demo: flies a camera in a circle around a real lunarsim tile
(real terrain, real regolith material, real fixed sun light -- the same
validated pipeline as isaaclab_test_camera.py) and saves one PNG frame per
orbit step. Not a full-globe render (no global DEM/texture in this repo) --
this is the "orbit effect" over a local tile, which is what actually exists
to render right now.

Run via scripts/run_isaaclab_orbit_demo.sh, which also stitches the frames
into an mp4 with ffmpeg (host-side, after the container exits).
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out-dir", type=str, default="/workspace/LunarSim/out/orbit_frames")
parser.add_argument("--n-frames", type=int, default=90)
parser.add_argument("--radius-m", type=float, default=55.0)
parser.add_argument("--height-m", type=float, default=35.0)
parser.add_argument("--n-orbits", type=float, default=1.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--mesh-lod", type=int, default=2)
parser.add_argument("--settle-steps", type=int, default=12)
parser.add_argument("--path-tracing", action="store_true", default=False)
parser.add_argument("--spp", type=int, default=256)
parser.add_argument("--center-detail-radius-m", type=float, default=45.0,
                     help="ROI falloff radius (sigma_m) for full fine detail at world "
                          "origin (0,0), where the camera orbits -- tapers to coarse "
                          "detail further out.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import numpy as np
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.sensors.camera import Camera, CameraCfg
from pxr import UsdShade

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, build_render_mesh
from lunarsim.adapters.isaac.lighting import create_sun_light
from lunarsim.adapters.isaac.materials import create_regolith_material
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile

os.makedirs(args_cli.out_dir, exist_ok=True)

sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"))
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([0.0, -args_cli.radius_m, args_cli.height_m], [0.0, 0.0, 0.0])

stage = omni.usd.get_context().get_stage()

cfg = TerrainConfig(
    mode="blend", size_m=150.0, res_m=0.25, coarse_res_m=2.0, seed=21,
    coarse_source="procedural",
    hills={"amplitude_m": 3.0, "wavelength_m": 40.0, "hurst": 0.75},
    craters={"count_scale": 1.5, "d_min_m": 1.0, "d_max_m": 30.0, "b": 2.4,
             "depth_ratio": 0.12, "age": 0.2},
    rocks={"density_scale": 1.0, "d_max_m": 1.0},
    # REAL BUG FIXED: {"sigma_m": 40.0, "centers": None} actually meant "one
    # RANDOM center" (see generate.py's `centers is None` branch), not "the
    # tile center" -- explicit regions at world (0,0), where the camera
    # actually orbits, is what guarantees max detail is where it's seen.
    roi={"regions": [{"x_m": 0.0, "y_m": 0.0,
                       "sigma_m": args_cli.center_detail_radius_m, "weight": 1.0}]},
    curvature=False,
)
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)
render_mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=args_cli.mesh_lod, uv_tile_size_m=2.0)

from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map
import imageio.v2 as _imageio_nm

normal_map_path = "/tmp/lunarsim_regolith_normal.png"
normal_map = bake_regolith_normal_map(resolution=1024, physical_size_m=2.0, seed=cfg.seed)
_imageio_nm.imwrite(normal_map_path, normal_map)
print(f"baked regolith micro-bump normal map -> {normal_map_path}")

material = create_regolith_material(
    stage, "/World/Looks/Regolith", albedo=0.11, brdf="albedo", normal_map_path=normal_map_path
)
UsdShade.MaterialBindingAPI.Apply(render_mesh.GetPrim()).Bind(material)

sun_pos = SunPosition(elevation_deg=35.0, azimuth_deg=120.0)
create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)

from lunarsim.adapters.isaac.lighting import disable_ambient, enable_path_tracing, set_no_ambient_render_settings

disable_ambient(stage)  # kill any Kit-seeded default dome light we didn't author
set_no_ambient_render_settings()  # zero the renderer-level ambient fill + use quality-mode DLSS
if args_cli.path_tracing:
    enable_path_tracing(spp=args_cli.spp)

camera_cfg = CameraCfg(
    prim_path="/World/OrbitCamera",
    update_period=0,
    height=args_cli.height,
    width=args_cli.width,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
)
camera = Camera(cfg=camera_cfg)

sim.reset()
disable_ambient(stage)  # re-check: some Kit extensions seed a default dome light on reset
set_no_ambient_render_settings()
if args_cli.path_tracing:
    enable_path_tracing(spp=args_cli.spp)

print("warming up renderer...")
for _ in range(60):
    sim.step(render=True)
    camera.update(dt=sim.get_physics_dt())

import torch

print(f"rendering {args_cli.n_frames} orbit frames...")
saved = 0
for i in range(args_cli.n_frames):
    angle = 2.0 * np.pi * args_cli.n_orbits * i / args_cli.n_frames
    cam_x = args_cli.radius_m * np.cos(angle)
    cam_y = args_cli.radius_m * np.sin(angle)
    cam_z = args_cli.height_m
    cam_pos = torch.tensor([[cam_x, cam_y, cam_z]], device=sim.device, dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
    camera.set_world_poses_from_view(cam_pos, target)

    for _ in range(args_cli.settle_steps):
        sim.step(render=True)
        camera.update(dt=sim.get_physics_dt())

    rgb = camera.data.output.get("rgb")
    if rgb is None:
        print(f"frame {i}: no rgb, skipping")
        continue
    rgb_np = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
    frame_u8 = np.clip(rgb_np[..., :3], 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(args_cli.out_dir, f"frame_{i:04d}.png"), frame_u8)
    saved += 1
    if i % 10 == 0:
        print(f"  frame {i}/{args_cli.n_frames} saved")

print(f"saved {saved}/{args_cli.n_frames} frames to {args_cli.out_dir}")
simulation_app.close()
