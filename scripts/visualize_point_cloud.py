"""Render a LiDAR point cloud (.ply from export_point_cloud) as a real
point-cloud image using Open3D's offscreen renderer -- proper per-point
z-buffering and point splatting, unlike matplotlib's 3D scatter (which has
no real depth compositing and looks muddy/flat for large clouds). This is
pure post-processing of a point cloud Isaac already produced; no Isaac Sim
needed here.

Install: pip install -e .[viz]  (adds open3d)
"""
import argparse

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("ply_path")
parser.add_argument("--out", default=None)
parser.add_argument("--point-size", type=float, default=2.0)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--front", type=float, nargs=3, default=(0.35, -0.8, 0.5),
                     help="camera view direction (oblique elevated by default)")
parser.add_argument("--zoom", type=float, default=0.35,
                     help="lower = closer/more zoomed-in (Open3D convention)")
parser.add_argument("--cmap", default="turbo")
parser.add_argument("--elevation-tint", type=float, default=0.35,
                     help="0 = pure normal-shaded gray (relief only, no elevation color -- "
                          "fixes fine detail washing out into one color band in a locally "
                          "flat area), 1 = pure elevation color (old behavior)")
parser.add_argument("--normal-radius-m", type=float, default=0.25,
                     help="neighborhood size for normal estimation -- shrink this if the "
                          "cloud is very dense/close-range so small rocks/craters still show "
                          "up as shading instead of being smoothed away")
parser.add_argument("--radius-m", type=float, default=None,
                     help="crop to points within this XY radius of the cloud's centroid "
                          "before rendering -- use this to frame in on a small dense patch "
                          "instead of the whole scan (a fixed point budget always looks "
                          "sparse again once you zoom into a small part of a wide scan).")
parser.add_argument("--center", type=float, nargs=2, default=None,
                     help="XY center for --radius-m crop (default: cloud centroid)")
args = parser.parse_args()

with open(args.ply_path) as f:
    lines = f.readlines()
hdr_end = lines.index("end_header\n") + 1
data = np.array([[float(v) for v in l.split()] for l in lines[hdr_end:]])
pts = data[:, :3]

if args.radius_m is not None:
    cx, cy = args.center if args.center else (pts[:, 0].mean(), pts[:, 1].mean())
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    kept = d <= args.radius_m
    print(f"cropping to radius {args.radius_m}m around ({cx:.1f}, {cy:.1f}): "
          f"{kept.sum()}/{len(pts)} points kept")
    pts = pts[kept]

z = pts[:, 2]
area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
print(f"{len(pts)} points, ~{area:.0f} m^2 -> {len(pts)/max(area,1e-6):.2f} pts/m^2")

import matplotlib.pyplot as plt

# REAL FIX for "the highest-detail area just looks like one flat color":
# pure elevation coloring normalizes across the WHOLE cloud's z range, so
# small local relief (a few cm of rock/crater texture near the sensor)
# gets washed into a single color band whenever it's small relative to
# that global range. Real point-cloud viewers (CloudCompare included) show
# fine surface texture via per-point NORMAL-based shading (a light dotted
# against each point's local surface normal), not color alone -- that's
# what actually reveals small bumps regardless of their absolute height.
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius_m, max_nn=30)
)
pcd.orient_normals_towards_camera_location(pts.mean(axis=0) + np.array([0, 0, 1000.0]))
normals = np.asarray(pcd.normals)

light_dir = np.array([0.4, -0.5, 0.75])
light_dir = light_dir / np.linalg.norm(light_dir)
shade = np.clip(normals @ light_dir, 0.05, 1.0)
shade = 0.35 + 0.65 * shade  # keep some ambient so nothing goes pure black

zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
elev_color = plt.get_cmap(args.cmap)(zn)[:, :3]
base_color = args.elevation_tint * elev_color + (1 - args.elevation_tint) * np.ones_like(elev_color)
colors = np.clip(base_color * shade[:, None], 0, 1)
pcd.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=args.width, height=args.height)
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.point_size = args.point_size

ctr = vis.get_view_control()
ctr.set_lookat(pts.mean(axis=0))
ctr.set_front(list(args.front))
ctr.set_up([0, 0, 1])
ctr.set_zoom(args.zoom)

vis.poll_events()
vis.update_renderer()
out = args.out or (args.ply_path.rsplit(".", 1)[0] + "_render.png")
vis.capture_screen_image(out, do_render=True)
vis.destroy_window()
print("saved", out)
