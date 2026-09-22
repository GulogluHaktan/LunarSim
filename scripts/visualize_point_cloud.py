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
parser.add_argument("--zoom", type=float, default=0.6)
parser.add_argument("--cmap", default="turbo")
args = parser.parse_args()

with open(args.ply_path) as f:
    lines = f.readlines()
hdr_end = lines.index("end_header\n") + 1
data = np.array([[float(v) for v in l.split()] for l in lines[hdr_end:]])
pts = data[:, :3]
z = pts[:, 2]
area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
print(f"{len(pts)} points, ~{area:.0f} m^2 -> {len(pts)/max(area,1e-6):.2f} pts/m^2")

import matplotlib.pyplot as plt

zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
colors = plt.get_cmap(args.cmap)(zn)[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
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
