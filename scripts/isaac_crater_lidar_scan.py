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
# Multi-position survey scan, real PhysX raycasts only
# ---------------------------------------------------------------------------
n_positions = 10
radius = 70.0
altitude = 45.0
max_range = 180.0

all_points = []
all_hit_count = 0
all_ray_count = 0

for i in range(n_positions):
    angle = 2 * np.pi * i / n_positions
    sensor_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), altitude])
    to_center = np.array([-sensor_pos[0], -sensor_pos[1], 0.0])
    yaw_deg = np.degrees(np.arctan2(to_center[1], to_center[0]))

    pattern = LidarScanPattern.spinning(
        n_channels=90, vertical_fov_deg=(-75, 5), horizontal_res_deg=1.2,
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

simulation_app.close()
sys.exit(0 if all_hit_count > 0 else 1)
