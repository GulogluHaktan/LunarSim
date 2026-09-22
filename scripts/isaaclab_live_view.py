"""Interactive live view: builds the real lunarsim scene (real terrain,
regolith material + micro-bump normal map, fixed sun light, no ambient) in
a non-headless Isaac Sim window and just idles -- fly/orbit the viewport
yourself with the mouse (standard Isaac Sim navigation: left-drag orbit,
middle-drag pan, scroll zoom, or WASD+right-drag for fly mode).

Needs a display (X11) on the host -- see scripts/run_isaaclab_live_view.sh,
which sets up the X11 forwarding into the container. Won't work over a pure
SSH-without-X session; use the orbit-demo video path for that instead.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
# REAL BUG FIXED: this used to default to path tracing (spp=256), which
# re-accumulates all 256 samples on every camera move -- for an
# interactive viewport that means it never looks "done" while orbiting,
# pins the GPU near 100%, and can look indistinguishable from a hang.
# Real-time raster is the right default for live interaction; pass
# --path-tracing explicitly if you want a single still frame's quality.
parser.add_argument("--path-tracing", dest="path_tracing", action="store_true", default=False)
parser.add_argument("--no-path-tracing", dest="path_tracing", action="store_false")
parser.add_argument("--spp", type=int, default=256)
parser.add_argument("--mesh-lod", type=int, default=2)
parser.add_argument("--size-m", type=float, default=150.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False  # this script is specifically for the interactive window

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import omni.usd
from pxr import UsdShade

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.heightfield import add_heightfield_collision, build_render_mesh
from lunarsim.adapters.isaac.lighting import (
    disable_ambient,
    enable_path_tracing,
    create_sun_light,
    set_no_ambient_render_settings,
)
from lunarsim.adapters.isaac.materials import create_regolith_material
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map
from lunarsim.core.lighting.sun import SunPosition
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile

sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"))
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([0.0, -60.0, 40.0], [0.0, 0.0, 0.0])

stage = omni.usd.get_context().get_stage()

cfg = TerrainConfig(
    mode="blend", size_m=args_cli.size_m, res_m=0.5, coarse_res_m=2.0, seed=21,
    coarse_source="procedural",
    hills={"amplitude_m": 3.0, "wavelength_m": 40.0, "hurst": 0.75},
    craters={"count_scale": 1.5, "d_min_m": 1.0, "d_max_m": 30.0, "b": 2.4,
             "depth_ratio": 0.12, "age": 0.2},
    rocks={"density_scale": 1.0, "d_max_m": 1.0},
    roi={"sigma_m": 40.0, "centers": None},
    curvature=False,
)
print("generating real terrain tile...")
tile = generate_tile(cfg)
add_heightfield_collision(stage, "/World/Terrain/Collision", tile)  # hidden from render by default
render_mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=args_cli.mesh_lod, uv_tile_size_m=2.0)

normal_map_path = "/tmp/lunarsim_regolith_normal.png"
imageio.imwrite(normal_map_path, bake_regolith_normal_map(resolution=1024, physical_size_m=2.0, seed=cfg.seed))
material = create_regolith_material(
    stage, "/World/Looks/Regolith", albedo=0.11, brdf="albedo", normal_map_path=normal_map_path
)
UsdShade.MaterialBindingAPI.Apply(render_mesh.GetPrim()).Bind(material)

sun_pos = SunPosition(elevation_deg=35.0, azimuth_deg=120.0)
create_sun_light(stage, "/World/Sun", sun_pos, angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()
if args_cli.path_tracing:
    enable_path_tracing(spp=args_cli.spp)

sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()
if args_cli.path_tracing:
    enable_path_tracing(spp=args_cli.spp)

print("scene ready -- use the Isaac Sim window to fly/orbit around (left-drag: orbit, "
      "middle-drag: pan, scroll: zoom). Close the window or Ctrl+C here to exit.")
while simulation_app.is_running():
    sim.step(render=True)

simulation_app.close()
