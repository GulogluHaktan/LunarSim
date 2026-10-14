"""Screenshot of the crater + boulder terrain: a static scene (no policy is run).

Takes the tile of a recorded episode, scatters a boulder field over the landing area, stands the
lander on the ground and renders oblique / low / top-down views with the same renderer, material
and sun as the landing videos.

  ~/isaac-env/bin/python scripts/isaaclab_terrain_screenshot.py \
      --recording out/crater2/rec/es_ext.npz --env 1 --out-dir out/crater2/shots
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--recording", type=str, default=None)
parser.add_argument("--env", type=int, default=0)
parser.add_argument("--from-config", action="store_true",
                    help="generate the tile with curriculum.terrain_config (env vars select craters / LOLA)")
parser.add_argument("--stage", type=str, default="orbit_descent")
parser.add_argument("--grid-n", type=int, default=512)
parser.add_argument("--sun-elev", type=float, default=18.0)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--rock-density", type=float, default=3.0, help="x the 0.02 /m^2 reference density")
parser.add_argument("--rock-dmax", type=float, default=3.0)
parser.add_argument("--rock-radius", type=float, default=170.0, help="only rocks within this radius are rendered")
parser.add_argument("--max-rocks", type=int, default=7000)
parser.add_argument("--seed", type=int, default=5)
parser.add_argument("--no-rocks", action="store_true")
parser.add_argument("--no-lander", action="store_true")
parser.add_argument("--upsample", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from pxr import Gf, UsdGeom, UsdShade  # noqa: E402

sys.path.insert(0, args_cli.lunarsim_root)
from lunarsim.adapters.isaac.heightfield import build_render_mesh  # noqa: E402
from lunarsim.adapters.isaac.lander import spawn_apollo_lm  # noqa: E402
from lunarsim.adapters.isaac.lighting import (  # noqa: E402
    create_sun_light, disable_ambient, set_no_ambient_render_settings)
from lunarsim.adapters.isaac.materials import create_regolith_material  # noqa: E402
from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map  # noqa: E402
from lunarsim.core.lighting.sun import SunPosition  # noqa: E402
from lunarsim.core.terrain.rocks import sample_height_at, sample_rock_field  # noqa: E402
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs  # noqa: E402

SPECS = ApolloLMSpecs()
HALF_H = SPECS.height_m / 2.0
os.makedirs(args_cli.out_dir, exist_ok=True)

if args_cli.from_config:
    from lunarsim.core.terrain.generate import generate_tile
    from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config
    from scipy.ndimage import uniform_filter

    _t = generate_tile(terrain_config(STAGES_BY_NAME[args_cli.stage], args_cli.seed, args_cli.grid_n))
    h0 = np.asarray(_t.height, dtype=np.float64); res0, size0 = float(_t.res_m), float(_t.size_m)
    n0 = h0.shape[0]
    gy, gx = np.gradient(h0, res0)
    slope = uniform_filter(np.degrees(np.arctan(np.hypot(gx, gy))), size=max(3, int(14 / res0)))
    ax0 = (np.arange(n0) - (n0 - 1) / 2) * res0
    X, Y = np.meshgrid(ax0, ax0, indexing="ij")
    cost = slope + 0.002 * np.hypot(X, Y)          # flattest spot, mildly preferring the centre
    ii, jj = np.unravel_index(int(np.argmin(cost)), cost.shape)
    sx, sy = float(ax0[ii]), float(ax0[jj])
    gz_site = float(sample_height_at(h0, res0, np.array([sx]), np.array([sy]))[0])
    site_tilt = (0.0, 0.0, 0.0)
    print(f"[shot] tile {size0:.0f} m @ {res0:.2f} m, craters {len(_t.craters.x_m)}, relief {np.ptp(h0):.1f} m, "
          f"site slope {slope[ii, jj]:.1f} deg", flush=True)
    _rock_src = _t.rocks
    try:
        import matplotlib; matplotlib.use("Agg")
        from matplotlib.colors import LightSource
        import matplotlib.pyplot as plt
        ls = LightSource(azdeg=135, altdeg=14)
        hs = ls.hillshade(h0.T, vert_exag=2.0, dx=res0, dy=res0)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(hs, cmap="gray", origin="lower", extent=[-size0 / 2, size0 / 2, -size0 / 2, size0 / 2])
        ax.plot(sx, sy, "r+", ms=14); ax.set_title(f"LOLA + sentetik krater, 1 pencere ({size0:.0f} m)")
        fig.savefig(os.path.join(args_cli.out_dir, "hillshade.png"), dpi=110, bbox_inches="tight")
    except Exception as e:  # noqa: BLE001
        print("[shot] hillshade skipped:", e)
else:
    d = np.load(args_cli.recording)
    i = args_cli.env
    h0 = d[f"e{i}_tile_height"].astype(np.float64)
    res0, size0 = float(d[f"e{i}_tile_res"]), float(d[f"e{i}_tile_size"])
    sx, sy = float(d[f"e{i}_x"][-1]), float(d[f"e{i}_y"][-1])
    gz_site = float(sample_height_at(h0, res0, np.array([sx]), np.array([sy]))[0])
    site_tilt = (float(np.degrees(d[f"e{i}_tilt_x"][-1])), float(np.degrees(d[f"e{i}_tilt_y"][-1])),
                 float(np.degrees(d[f"e{i}_yaw"][-1])))
    _rock_src = None

# ---- boulder field, scattered over the whole tile with the generator's own sampler, then cut to a disc
rng = np.random.default_rng(args_cli.seed)
rocks = _rock_src if _rock_src is not None else sample_rock_field(size0, args_cli.rock_density, args_cli.rock_dmax, rng)
near = np.hypot(rocks.x_m - sx, rocks.y_m - sy) < args_cli.rock_radius
rx, ry, rd = rocks.x_m[near], rocks.y_m[near], rocks.diameter_m[near]
if len(rx) > args_cli.max_rocks:
    keep = np.argsort(-rd)[:args_cli.max_rocks]
    rx, ry, rd = rx[keep], ry[keep], rd[keep]
rz = sample_height_at(h0, res0, rx, ry)
print(f"[shot] site ({sx:.0f},{sy:.0f}) ground {gz_site:.2f} m; rocks in disc {len(rx)} "
      f"(>0.5 m: {(rd > 0.5).sum()}, >1 m: {(rd > 1.0).sum()}, >2 m: {(rd > 2.0).sum()})", flush=True)


class _Tile:
    def __init__(self, height, res_m, size_m):
        self.height, self.res_m, self.size_m = height, res_m, size_m


pad = h0.shape[0]
hp = np.pad(h0, pad, mode="edge")
up = max(1, int(args_cli.upsample))
src = np.arange(hp.shape[0]); dst = np.linspace(0, hp.shape[0] - 1, (hp.shape[0] - 1) * up + 1)
h1 = np.array([np.interp(dst, src, r) for r in hp])
tile = _Tile(np.array([np.interp(dst, src, c) for c in h1.T]).T.astype(np.float32), res0 / up, size0 * 3.0)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu"))
stage = omni.usd.get_context().get_stage()
mesh = build_render_mesh(stage, "/World/Terrain/Render", tile, lod=0, uv_tile_size_m=3.0)
nmap = "/tmp/lunarsim_regolith_normal_shot.png"
imageio.imwrite(nmap, bake_regolith_normal_map(resolution=1024, physical_size_m=3.0,
                                               amplitude_m=0.09, wavelength_m=0.12, seed=7))
mat = create_regolith_material(stage, "/World/Looks/Regolith", albedo=0.11, brdf="albedo",
                               roughness=0.95, normal_map_path=nmap)
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
create_sun_light(stage, "/World/Sun", SunPosition(elevation_deg=args_cli.sun_elev, azimuth_deg=135.0), angular_diameter_deg=0.53)
disable_ambient(stage)
set_no_ambient_render_settings()

if not args_cli.no_rocks:
    UsdGeom.Xform.Define(stage, "/World/Rocks")
    rr = np.random.default_rng(args_cli.seed + 1)
    for k in range(len(rx)):
        sph = UsdGeom.Sphere.Define(stage, f"/World/Rocks/r{k}")
        sph.CreateRadiusAttr(0.5)
        xf = UsdGeom.Xformable(sph.GetPrim()); xf.ClearXformOpOrder()
        dm = float(max(rd[k], 0.08))
        sq = rr.uniform(0.55, 1.0)                      # squash: rocks sit and are not balls
        xf.AddTranslateOp().Set(Gf.Vec3d(float(rx[k]), float(ry[k]), float(rz[k]) + dm * 0.5 * sq * 0.6))
        xf.AddRotateZOp().Set(float(rr.uniform(0, 360)))
        xf.AddScaleOp().Set(Gf.Vec3f(dm * rr.uniform(0.8, 1.25), dm * rr.uniform(0.8, 1.25), dm * sq))
        UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mat)

lm_root = None
if not args_cli.no_lander:
    lm_root = spawn_apollo_lm(stage, "/World/LM", os.path.join(
        args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz"), visual_only=True)
    xa = UsdGeom.XformCommonAPI(lm_root.GetPrim())
    z_c = gz_site + HALF_H
    xa.SetTranslate(Gf.Vec3d(sx, sy, z_c))
    xa.SetRotate(Gf.Vec3f(*site_tilt), UsdGeom.XformCommonAPI.RotationOrderXYZ)

cam = Camera(cfg=CameraCfg(
    prim_path="/World/ShotCamera", update_period=0, height=args_cli.height, width=args_cli.width,
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5))))
sim.reset()
disable_ambient(stage)
set_no_ambient_render_settings()

views = {
    # name: (eye offset from the site, look-at offset)
    "yakin": ((-17.0, -9.0, 5.5), (0.0, 0.0, HALF_H - 0.5)),
    "alcak": ((-30.0, 6.0, 1.7), (0.0, 0.0, HALF_H - 1.0)),
    "genis": ((-95.0, -40.0, 38.0), (10.0, 5.0, 0.0)),
    "tepe": ((0.0, 0.1, 230.0), (0.0, 0.0, 0.0)),
}
for name, (eo, to) in views.items():
    eye = torch.tensor([[sx + eo[0], sy + eo[1], gz_site + eo[2]]], device=sim.device, dtype=torch.float32)
    tgt = torch.tensor([[sx + to[0], sy + to[1], gz_site + to[2]]], device=sim.device, dtype=torch.float32)
    cam.set_world_poses_from_view(eye, tgt)
    for _ in range(14):
        sim.step(render=True)
    cam.update(dt=sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    p = os.path.join(args_cli.out_dir, f"{name}.png")
    imageio.imwrite(p, np.clip(rgb[..., :3], 0, 255).astype(np.uint8))
    print("[shot] wrote", p, flush=True)
json.dump(dict(site=[sx, sy], rocks=int(len(rx))),
          open(os.path.join(args_cli.out_dir, "shot_info.json"), "w"))
simulation_app.close()
