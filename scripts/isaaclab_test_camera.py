"""Camera render validation using the ACTUAL working recipe: Isaac Lab's own
`AppLauncher` + `SimulationContext` + `isaaclab.sensors.camera.Camera`
(matches IsaacLab/scripts/tutorials/04_sensors/run_usd_camera.py, and is
architecturally the same path that produced LunarRocket's real rendered
frames). Bare `isaacsim.core.api.World` + `isaacsim.sensors.camera.Camera`
never advanced the render product past frame 0 in this same container (see
adapters/isaac/README.md's "Known gap" section) -- this script exists to
settle whether the Isaac Lab-specific Camera wrapper's explicit
`camera.update(dt=...)` per-step call is the missing piece.

Run via isaaclab.sh, not python.sh directly:
    ./isaaclab.sh -p /workspace/LunarSim/scripts/isaaclab_test_camera.py --headless --enable_cameras
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out", type=str, default="/workspace/LunarSim/out/isaaclab_camera_validation.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import json

import numpy as np
import torch
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, build_render_mesh
from lunarsim.adapters.isaac.lighting import create_sun_light
from lunarsim.adapters.isaac.materials import create_regolith_material
from lunarsim.core.lighting.camera import CameraNoiseModel
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile

sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"))
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([0.0, -25.0, 15.0], [0.0, 0.0, 0.0])

stage = omni.usd.get_context().get_stage()

cfg = TerrainConfig(
    mode="fine", size_m=100.0, res_m=0.5, seed=3,
    coarse_source="procedural",
    hills={"amplitude_m": 1.5, "wavelength_m": 25.0, "hurst": 0.75},
    craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 20.0, "b": 2.5,
             "depth_ratio": 0.12, "age": 0.3},
    rocks={"density_scale": 0.0, "d_max_m": 0.5},
    roi={"sigma_m": 30.0, "centers": None},
    curvature=False,
)
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)
render_mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=0)
material = create_regolith_material(stage, "/World/Looks/Regolith", albedo=0.1, brdf="albedo")

from pxr import UsdShade
UsdShade.MaterialBindingAPI.Apply(render_mesh.GetPrim()).Bind(material)

camera_cfg = CameraCfg(
    prim_path="/World/Camera",
    update_period=0,
    height=240,
    width=320,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
)
camera = Camera(cfg=camera_cfg)

sim.reset()

camera_positions = torch.tensor([[0.0, -25.0, 15.0]], device=sim.device)
camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
camera.set_world_poses_from_view(camera_positions, camera_targets)

noise_model = CameraNoiseModel()
rng = np.random.default_rng(0)
results = []

for elev in (2.0, 15.0, 45.0, 80.0):
    sun_pos = SunPosition(elevation_deg=elev, azimuth_deg=135.0)
    create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)

    for _ in range(20):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    rgb = camera.data.output.get("rgb")
    if rgb is None:
        results.append({"sun_elevation_deg": elev, "error": "no rgb output"})
        print(f"elev={elev}: NO RGB OUTPUT")
        continue

    rgb_np = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
    radiance = rgb_np[..., :3].astype(np.float64).mean(axis=-1) * 15000.0
    dn = noise_model.apply(radiance, exposure_s=0.01, rng=rng)
    max_dn = 2**noise_model.bit_depth - 1
    sat_frac = float(np.mean(dn >= max_dn * 0.99))
    results.append({
        "sun_elevation_deg": elev,
        "rgb_shape": list(rgb_np.shape),
        "mean_raw_rgb": float(rgb_np[..., :3].mean()),
        "mean_dn": float(dn.mean()),
        "saturation_fraction": sat_frac,
    })
    print(f"elev={elev:5.1f} deg  rgb_shape={rgb_np.shape}  mean_raw_rgb={rgb_np[...,:3].mean():.4f}  "
          f"mean_dn={dn.mean():.1f}  sat_frac={sat_frac:.4f}")

with open(args_cli.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {args_cli.out}")

any_ok = any("error" not in r for r in results)
simulation_app.close()
sys.exit(0 if any_ok else 1)
