"""A single, stationary, 360-degree spinning LiDAR scan (rover-mast-mounted
sensor, like a real Ouster/Velodyne), captured ENTIRELY from a live Isaac
Sim simulation -- every point is a real `omni.physx`
`raycast_closest()` call against the live PhysX collision mesh, no offline
geometry math.

This differs from isaac_crater_lidar_scan.py's multi-position orbital
survey (several partial-FOV viewpoints around the site, meant to look like
a descent/orbit pass -- which weaves into a basket-like overlapping
pattern when rendered). A real rover-mounted spinning LiDAR instead sits
at ONE position and sweeps the full 360 degrees, which is what actually
produces the clean concentric-ring point pattern real LiDAR point clouds
show (each ring = one vertical channel's full azimuth sweep).
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out", type=str, default="/workspace/LunarSim/out/rover_lidar_sim.ply")
parser.add_argument("--sensor-height-m", type=float, default=1.6)
parser.add_argument("--n-channels", type=int, default=128)
parser.add_argument("--horizontal-res-deg", type=float, default=0.12)
parser.add_argument("--terrain-size-m", type=float, default=40.0)
parser.add_argument("--terrain-res-m", type=float, default=0.15,
                     help="collision mesh resolution -- lower = more geometric detail but "
                          "more PhysX triangles to cook (VRAM/time cost); 0.06-0.08 is a "
                          "reasonable high-detail ceiling on an 8GB card for an 80m tile.")
parser.add_argument("--max-range-m", type=float, default=22.0,
                     help="REAL FIX: the same ray/point budget spread over a smaller "
                          "close-range footprint (not a wide 60m+ survey) is what actually "
                          "looks dense up close -- a fixed point count over a bigger area "
                          "just looks sparse again once you frame in on any one part of it.")
parser.add_argument("--n-frames", type=int, default=6,
                     help="a single static spin always shows discrete concentric rings "
                          "with real gaps between channels/azimuth steps -- that's correct "
                          "single-frame LiDAR physics, not a bug (see the reference image "
                          "of a single automotive spinning LiDAR sweep). A dense, near-"
                          "photographic-looking cloud like a real mapping product is built "
                          "by merging MANY frames as the platform drifts/re-levels slightly "
                          "between spins, which interlaces each frame's rings into the "
                          "previous frames' gaps. This does that for real: N real PhysX "
                          "sweeps from N slightly jittered sensor poses, merged.")
parser.add_argument("--jitter-m", type=float, default=0.06,
                     help="max random XY sensor-position offset between frames (m)")
parser.add_argument("--seed", type=int, default=0)
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

world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene")
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
ps = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(1.62)

cfg = TerrainConfig(
    mode="fine", size_m=args.terrain_size_m, res_m=args.terrain_res_m, seed=42,
    coarse_source="procedural",
    hills={"amplitude_m": 0.6, "wavelength_m": 25.0, "hurst": 0.75},
    craters={"count_scale": 1.2, "d_min_m": 1.0, "d_max_m": 30.0, "b": 1.9,
             "depth_ratio": 0.16, "age": 0.1},
    rocks={"density_scale": 1.2, "d_max_m": 1.2},
    roi={"centers": None},
    curvature=False,
)
n_grid = int(round(args.terrain_size_m / args.terrain_res_m))
print(f"generating real terrain tile ({n_grid}x{n_grid} = {n_grid*n_grid:,} verts, "
      f"~{2*(n_grid-1)**2:,} collision triangles)...")
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile, hide_from_render=False)
print(f"terrain height range: {tile.height.min():.2f} to {tile.height.max():.2f} m")

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(5):
    simulation_app.update()
timeline.pause()

physx_query = omni.physx.get_physx_scene_query_interface()

# Sensor sits at the terrain center, at rover mast height, right on the
# real surface below it (real raycast straight down finds ground z).
ground_hit = physx_query.raycast_closest(Gf.Vec3f(0, 0, 50.0), Gf.Vec3f(0, 0, -1), 200.0)
ground_z = ground_hit["position"][2] if ground_hit["hit"] else 0.0
sensor_pos = np.array([0.0, 0.0, ground_z + args.sensor_height_m])
print(f"sensor position: {sensor_pos.tolist()} (ground z={ground_z:.3f})")

rng = np.random.default_rng(args.seed)
max_range = args.max_range_m
all_points = []
hit_count = 0
ray_count = 0

for frame in range(args.n_frames):
    # Small per-frame jitter (position + azimuth/elevation start offset)
    # is what interlaces this frame's rings into the gaps left by the
    # previous frames' rings -- a real effect of the platform never
    # holding EXACTLY the same pose between spins, not synthetic noise.
    dx, dy = rng.uniform(-args.jitter_m, args.jitter_m, size=2)
    frame_pos = sensor_pos + np.array([dx, dy, 0.0])
    az_offset_deg = rng.uniform(0.0, args.horizontal_res_deg)
    el_offset_deg = rng.uniform(-0.15, 0.15)

    pattern = LidarScanPattern.spinning(
        n_channels=args.n_channels,
        vertical_fov_deg=(-55.0 + el_offset_deg, 10.0 + el_offset_deg),
        horizontal_res_deg=args.horizontal_res_deg,
        horizontal_fov_deg=(az_offset_deg, 360.0 + az_offset_deg),
    )
    dirs = pattern.ray_directions()
    ray_count += len(dirs)

    frame_hits = 0
    for d in dirs:
        origin = Gf.Vec3f(*frame_pos.tolist())
        direction = Gf.Vec3f(*d.tolist())
        hit_info = physx_query.raycast_closest(origin, direction, max_range)
        if hit_info["hit"]:
            all_points.append([hit_info["position"][0], hit_info["position"][1], hit_info["position"][2]])
            frame_hits += 1
    hit_count += frame_hits
    print(f"  frame {frame+1}/{args.n_frames} (jitter dx={dx:+.3f} dy={dy:+.3f}m): "
          f"{frame_hits} hits, {hit_count} cumulative")

pts = np.array(all_points)
print(f"\nTOTAL: {hit_count} real PhysX raycast hits out of {ray_count} rays, "
      f"from {args.n_frames} slightly-jittered sensor poses (max range {args.max_range_m}m), "
      f"all against the live simulation's collision mesh.")

merged = LidarPointCloud(
    points_m=pts, range_m=np.zeros(len(pts)), intensity=np.zeros(len(pts)),
    hit_mask=np.ones(len(pts), dtype=bool),
)
export_point_cloud(merged, args.out)
print(f"wrote {args.out}")

import os

sys.stdout.flush()
os._exit(0 if hit_count > 0 else 1)
