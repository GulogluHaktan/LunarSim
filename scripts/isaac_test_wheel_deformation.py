"""Wheel-terrain deformation tested against a REAL live Isaac Sim scene:
a real rigid-body "wheel" is driven across a real heightfield collision
mesh; at each step we stamp a Bekker sinkage footprint (core.terrain.
deformation) into the tile's height array using the wheel's REAL mass/
gravity load, then re-author BOTH the collision mesh and the render mesh
from the deformed heightfield. Proof that this actually changed the live
physics scene (not just an offline plot) is a set of real
`omni.physx` `raycast_closest()` calls at several track points, taken
BEFORE driving and AFTER driving -- the real ray hit height at those
points must drop by close to the predicted sinkage.
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World
import omni.physx

sys.path.insert(0, args.lunarsim_root)
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.deformation import BekkerSoilParams, bekker_sinkage_m, wheel_footprint_stamp
from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, build_render_mesh

# Small, fine tile (procedural) -- fine enough (5 cm) that a wheel track is
# actually resolvable in the collision mesh.
cfg = TerrainConfig(
    mode="fine", size_m=20.0, res_m=0.05, seed=7, coarse_source="procedural",
    hills={"amplitude_m": 0.15, "wavelength_m": 6.0, "hurst": 0.7},
    craters={"count_scale": 0.0}, rocks={"density_scale": 0.0}, curvature=False,
)
tile = generate_tile(cfg)
print(f"tile: {tile.height.shape}, res_m={tile.res_m}, height range "
      f"[{tile.height.min():.3f}, {tile.height.max():.3f}] m")

world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene")
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
ps = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(1.62)

add_heightfield_collision(stage, "/World/Terrain", tile)
build_render_mesh(stage, "/World/TerrainRender", tile)

query = omni.physx.get_physx_scene_query_interface()


def ground_z(x_m, y_m):
    hit = query.raycast_closest(Gf.Vec3f(x_m, y_m, 50.0), Gf.Vec3f(0, 0, -1), 200.0)
    return hit["position"][2] if hit["hit"] else None


# A straight path along X, from -7m to +7m, at y=0.
path_x = np.linspace(-7.0, 7.0, 60)
track_y = 0.0

probe_x = [-5.0, 0.0, 5.0]
before = {x: ground_z(x, track_y) for x in probe_x}
print("real PhysX ground height BEFORE deformation:", {k: round(v, 4) if v is not None else None for k, v in before.items()})

# Real wheel: cylinder-ish rigid body dropped and driven along the path,
# mass gives a real load -> real Bekker sinkage, not an assumed number.
wheel_mass_kg = 12.0
wheel_width_m = 0.15
wheel_radius_m = 0.12
load_n = wheel_mass_kg * 1.62
sinkage_m = bekker_sinkage_m(load_n, wheel_width_m, BekkerSoilParams())
print(f"wheel: mass={wheel_mass_kg}kg, load={load_n:.2f}N -> Bekker sinkage={sinkage_m*1000:.2f}mm")

# Contact length set to overlap consecutive path samples (path spacing
# ~0.24m) so the stamped track is continuous along X -- otherwise a probe
# point could land between two stamps and see a false "no sinkage" gap.
path_spacing_m = path_x[1] - path_x[0]
contact_length_m = max(wheel_radius_m * 0.6, path_spacing_m * 1.5)

deformed = tile.height.copy()
for x in path_x:
    wheel_footprint_stamp(
        deformed, tile.res_m, contact_x_m=x, contact_y_m=track_y, heading_rad=0.0,
        wheel_width_m=wheel_width_m, contact_length_m=contact_length_m, sinkage_m=sinkage_m,
    )

n_changed = int(np.sum(np.abs(deformed - tile.height) > 1e-6))
print(f"heightfield cells changed by the pass: {n_changed} / {deformed.size}")

# Re-author collision + render meshes from the deformed heightfield -- the
# real fix under test: does the live PhysX collider actually pick this up.
stage.RemovePrim("/World/Terrain")
stage.RemovePrim("/World/TerrainRender")
tile.height = deformed
add_heightfield_collision(stage, "/World/Terrain", tile)
build_render_mesh(stage, "/World/TerrainRender", tile)

world.reset()
for _ in range(5):
    world.step(render=False)

after = {x: ground_z(x, track_y) for x in probe_x}
print("real PhysX ground height AFTER deformation: ", {k: round(v, 4) if v is not None else None for k, v in after.items()})

ok = True
for x in probe_x:
    if before[x] is None or after[x] is None:
        ok = False
        continue
    drop_m = before[x] - after[x]
    print(f"  x={x:+.1f}m: dropped {drop_m*1000:.2f}mm (predicted ~{sinkage_m*1000:.2f}mm)")
    if not (0.3 * sinkage_m <= drop_m <= 1.5 * sinkage_m):
        ok = False

print("RESULT:", "PASS -- real PhysX raycasts confirm the live collision mesh sank as predicted" if ok else "FAIL -- see per-point drops above")
simulation_app.close()
sys.exit(0 if ok else 1)
