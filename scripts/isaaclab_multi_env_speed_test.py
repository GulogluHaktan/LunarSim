"""Physics speed test at ONE env count using Isaac Lab's real, supported
pattern for a SHARED terrain across many cloned envs:
`TerrainImporterCfg(terrain_type="usd")` + `InteractiveSceneCfg(num_envs=N)`
with a per-env `RigidObjectCfg` clone. Run once per env count you want to
measure (see scripts/run_isaaclab_speed_test.sh, which loops this) -- tearing
down and rebuilding an InteractiveScene/SimulationContext mid-process was
unreliable (a stale stage handle broke `get_current_stage_id()` on the
second iteration), so one process per env count is the robust way to do
this, matching how LunarRocket's own training scripts are structured
(one process per run) anyway.

This deliberately does NOT spawn one real DEM/USD terrain mesh per env --
LunarRocket's own prior architecture (source/lunar_rocket_lab/.../lunar_lander_env.py,
lunar_lander_env_cfg.py: `legacy_terrain_usd_max_envs = 16`) found that spawning
a UNIQUE real terrain mesh per env stops scaling past ~16 envs (falls back to
a flat ground plane silently past that), and that their actual bulk-training
throughput (512 envs) came from a single shared/GPU-instanced terrain with a
"terrain pool" of pre-generated height data swapped in per reset -- not from
N independent physics meshes. This script measures exactly that shared-terrain
scaling regime: one real lunarsim tile, N cloned rigid bodies on top of it.
"""
import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--out", type=str, default="/workspace/LunarSim/out/isaaclab_multi_env_speed.jsonl")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--n-warmup", type=int, default=20)
parser.add_argument("--n-measure", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from pxr import Usd, UsdGeom

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, apply_regolith_physics_material
from lunarsim.core.terrain import TerrainConfig, generate_tile

# ---------------------------------------------------------------------------
# 1) Export ONE real lunarsim tile to a standalone USD file (the shared terrain)
# ---------------------------------------------------------------------------
cfg = TerrainConfig(
    mode="fine", size_m=80.0, res_m=0.5, seed=11,
    coarse_source="procedural",
    hills={"amplitude_m": 1.0, "wavelength_m": 20.0, "hurst": 0.75},
    craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 15.0, "b": 2.5,
             "depth_ratio": 0.1, "age": 0.3},
    rocks={"density_scale": 0.0, "d_max_m": 0.5},
    roi={"sigma_m": 20.0, "centers": None},
    curvature=False,
)
tile = generate_tile(cfg)

terrain_usd_path = "/tmp/lunarsim_shared_terrain.usd"
proto_stage = Usd.Stage.CreateNew(terrain_usd_path)
UsdGeom.SetStageUpAxis(proto_stage, UsdGeom.Tokens.z)
add_heightfield_collision(proto_stage, "/Terrain", tile)
apply_regolith_physics_material(proto_stage, "/Terrain/PhysMat")
proto_stage.SetDefaultPrim(proto_stage.GetPrimAtPath("/Terrain"))
proto_stage.GetRootLayer().Save()
print(f"exported shared terrain to {terrain_usd_path}")

n_envs = args_cli.num_envs
print(f"\n=== num_envs={n_envs} ===")


@configclass
class SpeedTestSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=terrain_usd_path,
        env_spacing=100.0,  # large enough that clones don't overlap the same tile region
    )
    lander: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lander",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 15.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=1361.0, color=(1.0, 1.0, 0.98)),
    )


sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"), gravity=(0.0, 0.0, -1.62))
sim = sim_utils.SimulationContext(sim_cfg)

scene_cfg = SpeedTestSceneCfg(num_envs=n_envs, env_spacing=100.0)
scene = InteractiveScene(scene_cfg)
sim.reset()

for _ in range(args_cli.n_warmup):
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

t0 = time.perf_counter()
for _ in range(args_cli.n_measure):
    sim.step(render=False)
    scene.update(sim.get_physics_dt())
dt = time.perf_counter() - t0

steps_per_s = args_cli.n_measure / dt
env_steps_per_s = steps_per_s * n_envs
print(f"num_envs={n_envs}: {args_cli.n_measure} steps in {dt:.3f}s -> "
      f"{steps_per_s:.1f} sim-steps/s, {env_steps_per_s:.0f} env-steps/s")

result = {
    "num_envs": n_envs,
    "n_steps": args_cli.n_measure,
    "wall_time_s": dt,
    "sim_steps_per_s": steps_per_s,
    "env_steps_per_s": env_steps_per_s,
}
with open(args_cli.out, "a") as f:
    f.write(json.dumps(result) + "\n")
print(f"appended to {args_cli.out}")

simulation_app.close()
