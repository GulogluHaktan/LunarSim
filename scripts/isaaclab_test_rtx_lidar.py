"""Real RTX LiDAR validation via isaaclab.sim.SimulationContext (bare
isaacsim.core.api.World never ticks the RTX sensor's render pipeline in this
headless setup, same root cause as the camera gap -- see adapters/isaac/README.md).

Spawns an Ouster OS0 sensor profile above a real lunarsim tile, steps the
sim, and checks the returned point cloud actually lands on the real terrain
surface (compares each hit's z against `sample_height_at` at that hit's x/y).
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import isaaclab.sim as sim_utils
from pxr import Gf, UsdGeom, UsdPhysics

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision
from lunarsim.adapters.isaac.sensors import create_rtx_lidar, get_point_cloud
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.core.terrain.rocks import sample_height_at

sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"), gravity=(0.0, 0.0, -1.62))
sim = sim_utils.SimulationContext(sim_cfg)

import omni.usd

stage = omni.usd.get_context().get_stage()

cfg = TerrainConfig(
    mode="fine", size_m=80.0, res_m=0.5, seed=5,
    coarse_source="procedural",
    hills={"amplitude_m": 1.5, "wavelength_m": 20.0, "hurst": 0.75},
    craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 15.0, "b": 2.5,
             "depth_ratio": 0.1, "age": 0.3},
    rocks={"density_scale": 0.0, "d_max_m": 0.5},
    roi={"sigma_m": 20.0, "centers": None},
    curvature=False,
)
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)

print("creating RTX LiDAR (Ouster OS0 profile)...")
sensor = create_rtx_lidar("/World/Lidar", config="OS0", tick_rate=10.0)

lidar_prim = stage.GetPrimAtPath("/World/Lidar")
lidar_xf = UsdGeom.Xformable(lidar_prim)
ops = lidar_xf.GetOrderedXformOps()
sensor_height = 8.0
if ops:
    ops[0].Set(Gf.Vec3d(0.0, 0.0, sensor_height))
else:
    lidar_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, sensor_height))

sim.reset()

print("warming up + stepping sim to collect a lidar frame...")
pc = {"x_m": np.empty(0)}
for i in range(90):
    sim.step(render=True)
    pc = get_point_cloud(sensor)
    if pc["x_m"].size > 0:
        print(f"  step {i}: got {pc['x_m'].size} points")
        break

if pc["x_m"].size == 0:
    print("RTX LIDAR TEST: FAIL (no points ever returned)")
    simulation_app.close()
    sys.exit(1)

ground_z = sample_height_at(tile.height, tile.res_m, pc["x_m"], pc["y_m"])
err = pc["z_m"] - ground_z
in_bounds = (np.abs(pc["x_m"]) < 39) & (np.abs(pc["y_m"]) < 39)

print(f"total points: {pc['x_m'].size}, in-bounds points: {int(in_bounds.sum())}")
if in_bounds.sum() > 0:
    print(f"mean |z_hit - terrain_z| for in-bounds points: {np.mean(np.abs(err[in_bounds])):.3f} m")
    print(f"max |z_hit - terrain_z| for in-bounds points: {np.max(np.abs(err[in_bounds])):.3f} m")
    print(f"intensity range: {pc['intensity'].min():.3f} - {pc['intensity'].max():.3f}")

ok = in_bounds.sum() > 10 and np.mean(np.abs(err[in_bounds])) < 1.0
print("RTX LIDAR TEST:", "PASS" if ok else "FAIL")
simulation_app.close()
sys.exit(0 if ok else 1)
