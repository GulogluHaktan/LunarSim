"""Smoke test: build a real USD stage from a lunarsim Tile inside a live Isaac
Sim process, using our adapters.isaac authoring code, and report what broke.

Run inside the Isaac Sim Python environment (has `isaacsim`/`omni`/`pxr`
importable), e.g.:

    /isaac-sim/python.sh /workspace/LunarSim/scripts/isaac_smoke_test.py --headless

This is deliberately NOT a pytest test (Isaac Sim's SimulationApp must be
constructed before any `omni`/`pxr` import happens anywhere in the process,
which doesn't play well with pytest collection) -- it's a standalone script,
run once, that exercises every adapters.isaac function against a real stage
and prints a pass/fail per step so failures are easy to triage.
"""
import argparse
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--out", type=str, default="/tmp/lunarsim_smoke.usd")
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

# Everything else must import AFTER SimulationApp is constructed.
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

sys.path.insert(0, args.lunarsim_root)

from lunarsim.core.terrain import TerrainConfig, generate_tile  # noqa: E402

results = []


def step(name, fn):
    try:
        val = fn()
        results.append((name, "OK", ""))
        return val
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        traceback.print_exc()
        return None


def main():
    stage = omni.usd.get_context().get_stage()

    cfg = TerrainConfig(
        mode="fine",
        size_m=64.0,
        res_m=1.0,
        seed=1,
        coarse_source="procedural",
        hills={"amplitude_m": 2.0, "wavelength_m": 30.0, "hurst": 0.75},
        craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 15.0, "b": 2.5,
                 "depth_ratio": 0.1, "age": 0.3},
        rocks={"density_scale": 1.0, "d_max_m": 1.0},
        roi={"sigma_m": 20.0, "centers": None},
        curvature=False,
    )
    tile = step("generate_tile", lambda: generate_tile(cfg))
    if tile is None:
        return

    from lunarsim.adapters.isaac.heightfield import (
        add_heightfield_collision,
        apply_regolith_physics_material,
        build_render_mesh,
    )

    step("add_heightfield_collision", lambda: add_heightfield_collision(stage, "/World/Terrain/Collision", tile))
    step("build_render_mesh", lambda: build_render_mesh(stage, "/World/Terrain/Render", tile, lod=0))
    step("apply_regolith_physics_material", lambda: apply_regolith_physics_material(stage, "/World/PhysicsMaterials/Regolith"))

    from lunarsim.adapters.isaac.materials import create_regolith_material

    step("create_regolith_material", lambda: create_regolith_material(stage, "/World/Looks/Regolith", albedo=0.1, brdf="albedo"))

    from lunarsim.core.lighting.sun import SunPosition
    from lunarsim.adapters.isaac.lighting import create_sun_light, disable_ambient

    fake_sun = SunPosition(elevation_deg=30.0, azimuth_deg=90.0)
    step("create_sun_light", lambda: create_sun_light(stage, "/World/Sun", fake_sun, angular_diameter_deg=0.53))
    step("disable_ambient_noop", lambda: disable_ambient(stage, None))

    def make_rock_prototype():
        proto = UsdGeom.Sphere.Define(stage, "/World/RockPrototypes/Rock0")
        proto.CreateRadiusAttr(0.5)
        return proto

    step("make_rock_prototype", make_rock_prototype)

    from lunarsim.adapters.isaac.rocks import cap_rock_count, spawn_rocks

    capped_tile = step("cap_rock_count", lambda: cap_rock_count(tile, max_count=50, rng=np.random.default_rng(0)))
    if capped_tile is not None:
        step(
            "spawn_rocks",
            lambda: spawn_rocks(
                stage, "/World/Rocks", capped_tile, ["/World/RockPrototypes/Rock0"], np.random.default_rng(0)
            ),
        )

    from lunarsim.adapters.isaac.sensors import create_camera, create_rtx_lidar

    step("create_camera", lambda: create_camera("/World/Camera"))
    step("create_rtx_lidar", lambda: create_rtx_lidar(stage, "/World/Lidar", mode="rtx_sparse", n_channels=16))

    step("export_stage", lambda: stage.Export(args.out))


main()

print("\n=== LunarSim Isaac adapter smoke test ===")
n_fail = sum(1 for _, status, _ in results if status == "FAIL")
for name, status, msg in results:
    print(f"[{status:4s}] {name}  {msg}")
print(f"\n{len(results) - n_fail}/{len(results)} steps OK")

simulation_app.close()
sys.exit(1 if n_fail else 0)
