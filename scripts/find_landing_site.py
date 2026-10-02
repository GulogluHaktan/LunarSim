"""Pick a landing site on a capture's terrain, the way a real lander would.

On the rough ('cinematic') terrain -- the one that actually looks like the
Moon -- only about a tenth of the surface satisfies
`safe_landing_max_leg_height_diff_m`, the footpad-span flatness term of
`landed_safely`. Releasing a vehicle at the tile centre there and hoping is
not a demonstration of anything: the outcome is decided by where the spawn
happened to point, not by the guidance.

So this does what the onboard hazard map exists to do -- scores every
candidate site by the simulator's OWN landing criterion and hands back a
target to aim at. Feed the result to
`isaaclab_controller_eval_capture.py --target-x/--target-y`.

The terrain is reproduced exactly, not approximated: the capture script
derives its terrain seed and its spawn point from one `default_rng(seed)`
stream, and this replays that stream in the same order.

IMPORTANT: the site returned is tied to the terrain that seed produces, so
the capture has to run with the SAME seed and `--max-seeds-to-try 1`. Letting
the capture bump its seed on a failed landing would regenerate the terrain
underneath a target chosen for the old one, and the run would be aiming at a
spot that no longer exists.

Run: .venv/bin/python scripts/find_landing_site.py --seed 2001 \\
         --terrain-profile cinematic --tile-size-m 1000
"""
from __future__ import annotations

import argparse

import numpy as np

from lunarsim.adapters.isaac.isaac_lander_env import collision_mesh_height_at
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.core.terrain.rocks import sample_height_at
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs
from lunarsim.rl.analytic_lander_env import LanderParams
from lunarsim.rl.curriculum import Stage as CurriculumStage
from lunarsim.rl.curriculum import terrain_config as curriculum_terrain_config

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--terrain-profile", type=str, default="cinematic", choices=["training", "cinematic"])
parser.add_argument("--tile-size-m", type=float, default=1680.0)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--terrain-roughness-scale", type=float, default=1.0)
parser.add_argument("--spawn-altitude-min-m", type=float, default=200.0)
parser.add_argument("--spawn-altitude-max-m", type=float, default=200.0)
parser.add_argument("--spawn-xy-radius-m", type=float, default=80.0)
parser.add_argument("--spawn-speed-min-m-s", type=float, default=10.0)
parser.add_argument("--spawn-speed-max-m-s", type=float, default=30.0)
parser.add_argument("--max-range-m", type=float, default=400.0,
                    help="how far from the release point the site may be. The vehicle has to be able to "
                         "reach it: too far and the guidance spends the whole descent on cross-track.")
parser.add_argument("--step-m", type=float, default=4.0, help="candidate site spacing")
parser.add_argument("--near-x", type=float, default=None)
parser.add_argument("--near-y", type=float, default=None,
                    help="search around this point instead of around the release point. MEASURED REASON "
                         "this exists: the ZemZev controller nulls its horizontal velocity but does not "
                         "fly back to the target -- two captures released at 18.6 m/s both ended ~600 m "
                         "downrange of where they were aimed (hero_fixed was aimed at (0,0) and set down "
                         "at (-596, 142)). So a target picked next to the release point is a target the "
                         "vehicle will overfly. Predict where it will actually arrive -- roughly the "
                         "release point plus 30x the release speed along the release heading -- and look "
                         "for a landable patch THERE.")
args = parser.parse_args()

specs = ApolloLMSpecs()
params = LanderParams()
stance = specs.footpad_span_m / 2.0

# --- replay the capture script's RNG stream, in its order -------------------
rng = np.random.default_rng(args.seed)
terrain_seed = int(rng.integers(0, 2**31 - 1))
r = args.terrain_roughness_scale
if args.terrain_profile == "training":
    cfg = curriculum_terrain_config(
        CurriculumStage(name="eval", tile_size_m=args.tile_size_m, terrain_roughness_scale=r),
        terrain_seed, args.terrain_grid_n)
else:
    cfg = TerrainConfig(
        mode="fine", size_m=args.tile_size_m, res_m=max(0.4, args.tile_size_m / 300.0), seed=terrain_seed,
        coarse_source="procedural",
        hills={"amplitude_m": 1.5 * r, "wavelength_m": args.tile_size_m / 20.0, "hurst": 0.75},
        craters={"count_scale": 1.0 * r, "d_min_m": 1.0, "d_max_m": args.tile_size_m / 6.0, "b": 2.5,
                 "depth_ratio": 0.12, "age": 0.4},
        rocks={"density_scale": 0.6 * r, "d_max_m": 0.8},
        roi={"regions": [{"x_m": 0.0, "y_m": 0.0, "sigma_m": args.tile_size_m / 4.0, "weight": 1.0}]},
        curvature=False,
    )
tile = generate_tile(cfg)

spawn_altitude_m = float(rng.uniform(args.spawn_altitude_min_m, args.spawn_altitude_max_m))
spawn_r = float(rng.uniform(0.0, args.spawn_xy_radius_m))
spawn_theta = float(rng.uniform(0.0, 2.0 * np.pi))
spawn_x, spawn_y = spawn_r * np.cos(spawn_theta), spawn_r * np.sin(spawn_theta)
_ = float(sample_height_at(tile.height, tile.res_m, np.array([spawn_x]), np.array([spawn_y]))[0])
spawn_speed = float(rng.uniform(args.spawn_speed_min_m_s, args.spawn_speed_max_m_s))

print(f"terrain seed {terrain_seed}, {tile.height.shape[0]}x{tile.height.shape[0]} at {tile.res_m:.2f} m/cell")
print(f"release ({spawn_x:.1f}, {spawn_y:.1f}) alt={spawn_altitude_m:.0f} m, h_speed={spawn_speed:.1f} m/s")

# --- score every candidate site with the simulator's own criterion ----------
half = tile.res_m * (tile.height.shape[0] - 1) / 2.0 - stance - 2.0
axis = np.arange(-half, half + args.step_m, args.step_m)
gx, gy = np.meshgrid(axis, axis, indexing="ij")

# the four footpads, exactly as `_footpad_height_diff_m` samples them
pads = [collision_mesh_height_at(tile.height, tile.res_m,
                                 gx + stance * np.cos(a), gy + stance * np.sin(a))
        for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])]
pads = np.stack(pads, axis=0)
leg_diff = pads.max(axis=0) - pads.min(axis=0)

centre_z = collision_mesh_height_at(tile.height, tile.res_m, gx, gy)
dz_dx, dz_dy = np.gradient(centre_z, args.step_m)
slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

landable = (leg_diff <= params.safe_landing_max_leg_height_diff_m) & (
    slope_deg <= np.degrees(params.safe_landing_tilt_rad))
print(f"landable sites: {landable.sum()} of {landable.size} "
      f"({100.0 * landable.mean():.1f}% of the tile)")
if not landable.any():
    raise SystemExit("no landable site on this terrain -- try another seed")

# --- prefer the middle of a landable PATCH, not a lucky single cell ---------
# A site one cell away from rough ground is not a site: real touchdown
# disperses by metres. Score each candidate by how far it is from the
# nearest unlandable cell, so the pick sits as deep inside a good patch as
# the terrain allows.
try:
    from scipy.ndimage import distance_transform_edt
    margin_m = distance_transform_edt(landable) * args.step_m
except ImportError:  # pragma: no cover - scipy ships with this project
    margin_m = landable.astype(float) * args.step_m

anchor_x = spawn_x if args.near_x is None else args.near_x
anchor_y = spawn_y if args.near_y is None else args.near_y
if args.near_x is not None or args.near_y is not None:
    print(f"searching around the predicted arrival ({anchor_x:+.1f}, {anchor_y:+.1f}) m, "
          f"not the release point")
reach = np.hypot(gx - anchor_x, gy - anchor_y)
eligible = landable & (reach <= args.max_range_m)
if not eligible.any():
    raise SystemExit(f"no landable site within {args.max_range_m:.0f} m of ({anchor_x:.0f}, {anchor_y:.0f})")

# among the roomiest sites, take the one closest to the anchor: least
# cross-track for the guidance to fly off, most margin when it arrives.
best_margin = margin_m[eligible].max()
good = eligible & (margin_m >= best_margin - args.step_m)
pick = np.argmin(np.where(good, reach, np.inf))
ix, iy = np.unravel_index(pick, reach.shape)
tx, ty = float(gx[ix, iy]), float(gy[ix, iy])

print(f"\nchosen site ({tx:+.1f}, {ty:+.1f}) m")
print(f"  footpad height spread : {leg_diff[ix, iy]:.3f} m  (limit "
      f"{params.safe_landing_max_leg_height_diff_m:.2f})")
print(f"  slope                 : {slope_deg[ix, iy]:.2f} deg  (limit "
      f"{np.degrees(params.safe_landing_tilt_rad):.0f})")
print(f"  clear radius around it: {margin_m[ix, iy]:.0f} m of continuously landable ground")
print(f"  range from anchor     : {reach[ix, iy]:.0f} m")
print(f"\n  --target-x {tx:.1f} --target-y {ty:.1f}")
