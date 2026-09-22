"""Dense, multi-position LiDAR point cloud captured ENTIRELY from a live
Isaac Sim simulation -- every point comes from a real
`omni.physx.get_physx_scene_query_interface().raycast_closest()` call
against the actual PhysX collision mesh in the running scene (the same
mechanism validated in scripts/isaac_validation_suite.py's LiDAR section),
not from any standalone/offline Python geometry calculation. Scans a real
lunarsim tile with a prominent crater from several sensor positions around
it (like a descent/orbit survey) and merges the real hits into one cloud.
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out", type=str, default="/workspace/LunarSim/out/crater_scan_sim.ply")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import numpy as np
import omni.physx
import omni.timeline
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World

sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision
from lunarsim.core.metadata.lidar import LidarPointCloud, LidarScanPattern, export_point_cloud
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile

# ---------------------------------------------------------------------------
# Build the real scene: World, real physics scene, real terrain collision mesh
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene")
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
ps = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(1.62)

cfg = TerrainConfig(
    mode="fine", size_m=180.0, res_m=0.3, seed=42,
    coarse_source="procedural",
    hills={"amplitude_m": 1.0, "wavelength_m": 60.0, "hurst": 0.75},
    craters={"count_scale": 0.6, "d_min_m": 2.0, "d_max_m": 140.0, "b": 1.8,
             "depth_ratio": 0.16, "age": 0.15},
    rocks={"density_scale": 0.8, "d_max_m": 1.5},
    roi={"centers": None},
    curvature=False,
)
print("generating real terrain tile (one dominant crater)...")
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile, hide_from_render=False)
print(f"terrain height range: {tile.height.min():.2f} to {tile.height.max():.2f} m")

# advance physics so PhysX actually builds its collision/broadphase structures
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    simulation_app.update()
timeline.pause()

physx_query = omni.physx.get_physx_scene_query_interface()

# ---------------------------------------------------------------------------
# Multi-position survey scan, real PhysX raycasts only.
#
# REAL DENSITY FIX: the first version of this scan (10 positions, 90
# channels, 1.2 deg horizontal res) put only ~1.5 pts/m^2 over the full
# 180x180m tile -- visually it looked like thin scattered lines/rings, not
# a dense point-cloud "surface", because the ray budget was spread across
# too much ground. A close-range descent/landing-site survey (like the
# reference scan this is meant to resemble) covers a much smaller footprint
# at much higher angular density. Tightened the orbit radius/altitude and
# raised channel count + angular resolution so the same real-PhysX-raycast
# approach now lands on a ~70m-wide patch around the crater at roughly two
# orders of magnitude higher point density.
# ---------------------------------------------------------------------------
n_positions = 16
radius = 35.0
altitude = 22.0
max_range = 120.0

all_points = []
all_hit_count = 0
all_ray_count = 0

for i in range(n_positions):
    angle = 2 * np.pi * i / n_positions
    sensor_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), altitude])
    to_center = np.array([-sensor_pos[0], -sensor_pos[1], 0.0])
    yaw_deg = np.degrees(np.arctan2(to_center[1], to_center[0]))

    pattern = LidarScanPattern.spinning(
        n_channels=128, vertical_fov_deg=(-70, 15), horizontal_res_deg=0.25,
        horizontal_fov_deg=(yaw_deg - 45, yaw_deg + 45),
    )
    dirs = pattern.ray_directions()
    all_ray_count += len(dirs)

    for d in dirs:
        origin = Gf.Vec3f(*sensor_pos.tolist())
        direction = Gf.Vec3f(*d.tolist())
        hit_info = physx_query.raycast_closest(origin, direction, max_range)
        if hit_info["hit"]:
            all_points.append([hit_info["position"][0], hit_info["position"][1], hit_info["position"][2]])
            all_hit_count += 1

    print(f"  position {i+1}/{n_positions}: {all_hit_count} cumulative real PhysX hits")

pts = np.array(all_points)
print(f"\nTOTAL: {all_hit_count} real PhysX raycast hits out of {all_ray_count} rays cast, "
      f"from {n_positions} sensor positions, all against the live simulation's collision mesh.")

merged = LidarPointCloud(
    points_m=pts,
    range_m=np.zeros(len(pts)),
    intensity=np.zeros(len(pts)),
    hit_mask=np.ones(len(pts), dtype=bool),
)
export_point_cloud(merged, args.out)
print(f"wrote {args.out}")

# Skip simulation_app.close(): its carb tasking teardown has a known
# benign race ("Destroying busy TaskGroup" assertion -> Aborted) in this
# environment, always AFTER our output is already written. All work is
# done, so exit immediately instead of running that teardown.
import os

sys.stdout.flush()
os._exit(0 if all_hit_count > 0 else 1)
