"""Real-Isaac-Sim validation suite: LiDAR ground truth vs. the live PhysX
collision scene, camera behavior across sun elevations, and a physics-step
speed benchmark.

Run: /isaac-sim/python.sh scripts/isaac_validation_suite.py --lunarsim-root /workspace/LunarSim
"""
import argparse
import json
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out", type=str, default="/workspace/LunarSim/out/isaac_validation.json")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})

import numpy as np
import omni.physx
from pxr import Gf, UsdGeom, UsdPhysics

try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World

sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, apply_regolith_physics_material
from lunarsim.adapters.isaac.lighting import create_sun_light
from lunarsim.adapters.isaac.sensors import create_camera
from lunarsim.core.lighting.camera import CameraNoiseModel
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.metadata.lidar import LidarPointCloud, LidarScanPattern, compare_point_clouds, raycast_lidar
from lunarsim.core.terrain import TerrainConfig, generate_tile

report = {}


def section(name):
    print(f"\n--- {name} ---")


# ---------------------------------------------------------------------------
# Build the scene: World, real terrain collision mesh, lunar-g physics scene,
# camera -- all BEFORE world.reset(), which is what actually wires up the
# render pipeline (a raw simulation_app.update() loop with no World/reset()
# never advances the render product past frame 0 -- confirmed empirically
# against this same image, and matches the working pattern in
# LunarRocket/app/record_demo_light.py:209-341).
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene")
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

physics_scene = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr(1.62)  # lunar gravity, m/s^2

cfg = TerrainConfig(
    mode="fine",
    size_m=100.0,
    res_m=0.5,
    seed=3,
    coarse_source="procedural",
    hills={"amplitude_m": 1.5, "wavelength_m": 25.0, "hurst": 0.75},
    craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 20.0, "b": 2.5,
             "depth_ratio": 0.12, "age": 0.3},
    rocks={"density_scale": 0.5, "d_max_m": 0.5},
    roi={"sigma_m": 30.0, "centers": None},
    curvature=False,
)
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)
apply_regolith_physics_material(stage, "/World/PhysicsMaterials/Regolith")

lander = UsdGeom.Cube.Define(stage, "/World/Lander")
lander.CreateSizeAttr(1.0)
lander_xform = UsdGeom.Xformable(lander.GetPrim())
lander_xform.ClearXformOpOrder()
lander_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 20.0))
UsdPhysics.RigidBodyAPI.Apply(lander.GetPrim())
UsdPhysics.CollisionAPI.Apply(lander.GetPrim())
UsdPhysics.MassAPI.Apply(lander.GetPrim()).CreateMassAttr(1500.0)

camera = create_camera("/World/Camera", resolution=(320, 240))
cam_xform = UsdGeom.Xformable(camera.prim)
cam_xform.ClearXformOpOrder()
cam_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -25.0, 15.0))
cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 0.0))  # look down-forward at the terrain

fake_sun = SunPosition(elevation_deg=30.0, azimuth_deg=135.0)
create_sun_light(stage, "/World/Sun", fake_sun, angular_diameter_deg=0.53)

print("Resetting world (wires up physics + render pipeline)...")
world.reset()

print("Warming up renderer (45 frames, matches the proven record_demo_light.py recipe)...")
for _ in range(45):
    world.step(render=True)
    simulation_app.update()

# ---------------------------------------------------------------------------
# 1) LiDAR ground truth vs PhysX scene-query raycast against the SAME mesh
# ---------------------------------------------------------------------------
section("LiDAR: analytic ground truth vs. live PhysX raycast")

sensor_pos = np.array([0.0, 0.0, 8.0])
pattern = LidarScanPattern.spinning(n_channels=16, vertical_fov_deg=(-60, -5), horizontal_res_deg=15)
ray_dirs = pattern.ray_directions()

analytic = raycast_lidar(tile.height, tile.res_m, sensor_pos, ray_dirs, max_range_m=50.0)

physx_query = omni.physx.get_physx_scene_query_interface()
physx_range = np.full(len(ray_dirs), np.nan)
physx_hit = np.zeros(len(ray_dirs), dtype=bool)
for i, d in enumerate(ray_dirs):
    origin = Gf.Vec3f(*sensor_pos.tolist())
    direction = Gf.Vec3f(*d.tolist())
    hit_info = physx_query.raycast_closest(origin, direction, 50.0)
    if hit_info["hit"]:
        physx_hit[i] = True
        physx_range[i] = hit_info["distance"]

physx_cloud = LidarPointCloud(
    points_m=np.full((len(ray_dirs), 3), np.nan),
    range_m=physx_range,
    intensity=np.zeros(len(ray_dirs)),
    hit_mask=physx_hit,
)
stats = compare_point_clouds(analytic, physx_cloud)
print("both-hit count:", stats.get("n_compared"), "/ total rays:", len(ray_dirs))
print("analytic hit count:", int(analytic.hit_mask.sum()), " physx hit count:", int(physx_hit.sum()))
print(json.dumps(stats, indent=2))

# diagnostics: rays PhysX hit but analytic missed, and a sample of large disagreements
physx_only = physx_hit & ~analytic.hit_mask
print("rays physx-hit/analytic-miss:", int(physx_only.sum()))
for i in np.where(physx_only)[0][:5]:
    print(f"  ray {i}: dir={ray_dirs[i]}, physx_range={physx_range[i]:.2f}")
both = analytic.hit_mask & physx_hit
err = physx_range[both] - analytic.range_m[both]
worst = np.argsort(-np.abs(err))[:5]
idx_both = np.where(both)[0]
for k in worst:
    i = idx_both[k]
    print(f"  worst ray {i}: dir={ray_dirs[i]}, analytic={analytic.range_m[i]:.2f}, physx={physx_range[i]:.2f}, err={err[k]:.2f}")
report["lidar_vs_physx"] = stats
report["lidar_hit_counts"] = {
    "n_rays": len(ray_dirs),
    "analytic_hits": int(analytic.hit_mask.sum()),
    "physx_hits": int(physx_hit.sum()),
}

# ---------------------------------------------------------------------------
# 2) Camera behavior across sun elevations (saturation should rise as the
#    sun-facing slope gets more direct light; shadow side stays near-black
#    since ambient is off)
#
# KNOWN GAP: in this headless-docker setup, the RGB render product's frame
# counter never advances past 0 through bare isaacsim.core.api.World +
# world.step(render=True), with or without enable_cameras=True or extra
# warm-up ticks (tried up to 100). LunarRocket's actually-working rendered
# output (outputs/videos/trained_lunar_landing.mp4, real frames on disk) was
# produced by driving the camera through an Isaac Lab env's own
# SimulationContext (isaaclab.envs.DirectRLEnv / env.world.step), not bare
# isaacsim.core.api.World -- that's a materially bigger dependency (a real
# Isaac Lab checkout, not just isaac-sim) than this adapter currently pulls
# in. Left as a documented gap rather than silently claimed as working.
# ---------------------------------------------------------------------------
section("Camera: saturation vs. sun elevation")

camera_stats = []
noise_model = CameraNoiseModel()
rng = np.random.default_rng(0)

for elev in (2.0, 15.0, 45.0, 80.0):
    sun_pos = SunPosition(elevation_deg=elev, azimuth_deg=135.0)
    create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)
    for _ in range(10):
        world.step(render=True)
        simulation_app.update()

    frame_dict = camera.get_current_frame()
    rgb = frame_dict.get("rgb")
    if rgb is None:
        camera_stats.append({"sun_elevation_deg": elev, "error": "empty frame (see KNOWN GAP note above)"})
        print(f"elev={elev:5.1f} deg  ERROR: empty frame")
        continue

    radiance = rgb[..., :3].astype(np.float64).mean(axis=-1) * 50.0  # arbitrary linear scale for the noise model
    dn = noise_model.apply(radiance, exposure_s=0.01, rng=rng)
    max_dn = 2**noise_model.bit_depth - 1
    sat_frac = float(np.mean(dn >= max_dn * 0.99))
    camera_stats.append({
        "sun_elevation_deg": elev,
        "mean_raw_rgb": float(rgb[..., :3].mean()),
        "mean_dn": float(dn.mean()),
        "saturation_fraction": sat_frac,
    })
    print(f"elev={elev:5.1f} deg  mean_raw_rgb={rgb[...,:3].mean():.4f}  mean_dn={dn.mean():.1f}  sat_frac={sat_frac:.4f}")

report["camera_vs_sun_elevation"] = camera_stats

# ---------------------------------------------------------------------------
# 3) Physics step speed
# ---------------------------------------------------------------------------
section("Physics step speed")

n_warmup, n_measure = 20, 200
for _ in range(n_warmup):
    world.step(render=False)  # physics-only speed, matches a training loop with rendering off

t0 = time.perf_counter()
for _ in range(n_measure):
    world.step(render=False)
dt = time.perf_counter() - t0

steps_per_s = n_measure / dt
print(f"{n_measure} steps in {dt:.3f}s -> {steps_per_s:.1f} steps/s (single env, physics-only, render off)")
report["physics_speed"] = {"n_steps": n_measure, "wall_time_s": dt, "steps_per_s": steps_per_s}

# ---------------------------------------------------------------------------
with open(args.out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {args.out}")

simulation_app.close()
