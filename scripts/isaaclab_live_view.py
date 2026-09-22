"""Interactive live view: builds the real lunarsim scene (real terrain,
regolith material + micro-bump normal map, fixed sun light, no ambient) and
just idles so you can fly/orbit around it yourself (standard Isaac Sim
navigation: left-drag orbit, middle-drag pan, scroll zoom, or WASD+right-
drag for fly mode).

Two ways to view it, both via IsaacLab's built-in `--livestream` flag:

  --livestream 0 (default): a native X11 window on THIS machine's display
  (see scripts/run_isaaclab_live_view.sh). Needs a real display; on this
  project's dev machine this got stuck inside AppLauncher() itself with
  ~4% GPU util and no window ever appearing -- a real, unresolved issue
  with GLX/window-surface creation from inside the container against this
  host's NVIDIA driver, not something our own code controls.

  --livestream 2 (RECOMMENDED, WebRTC): runs fully headless -- no X11/GLX
  window is created at all, sidestepping that failure mode entirely -- and
  streams frames out over WebRTC. Connect with the NVIDIA "Isaac Sim
  WebRTC Streaming Client" app, or a browser pointed at this machine's
  streaming port (see scripts/run_isaaclab_livestream_view.sh for the
  exact URL/port and a client-app download link). Works even over a pure
  SSH-without-X session since no local display is touched.
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
parser.add_argument("--center-detail-radius-m", type=float, default=40.0,
                     help="ROI falloff radius (sigma_m) for full fine detail at world "
                          "origin (0,0) -- where the camera starts -- tapering to coarse "
                          "detail further out.")
AppLauncher.add_app_launcher_args(parser)  # adds --livestream {0,1,2}, --headless, --device, ...
args_cli, _ = parser.parse_known_args()
# REAL BUG FIXED: AppLauncher's constructor consumes/removes its own args
# (including `livestream`) from args_cli as it processes them -- reading
# args_cli.livestream again AFTER constructing AppLauncher raises
# AttributeError (confirmed live: got past "Simulation App Startup
# Complete" and into terrain generation, then crashed there). Capture it
# in a plain local variable up front instead.
livestream_mode = args_cli.livestream
# livestream mode renders fully headless (no local X11/GLX window at all --
# that's the whole point, see module docstring); only fall back to a real
# window when streaming is off.
args_cli.headless = livestream_mode in (1, 2)

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
    # REAL BUG FIXED: {"centers": None} actually meant "one RANDOM center"
    # (see generate.py's `centers is None` branch), not "the tile center" --
    # explicit regions at world (0,0) is what actually guarantees max detail
    # right where the camera starts, tapering out over center-detail-radius-m.
    roi={"regions": [{"x_m": 0.0, "y_m": 0.0,
                       "sigma_m": args_cli.center_detail_radius_m, "weight": 1.0}]},
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

if livestream_mode in (1, 2):
    print("scene ready -- connect the Isaac Sim WebRTC Streaming Client (or a browser, see "
          "scripts/run_isaaclab_livestream_view.sh) to this machine now. Ctrl+C here to exit.")
else:
    print("scene ready -- use the Isaac Sim window to fly/orbit around (left-drag: orbit, "
          "middle-drag: pan, scroll: zoom). Close the window or Ctrl+C here to exit.")
while simulation_app.is_running():
    sim.step(render=True)

simulation_app.close()
