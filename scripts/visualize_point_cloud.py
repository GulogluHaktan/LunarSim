"""Render a LiDAR point cloud (.ply from export_point_cloud) as a clean,
CloudCompare-style elevation-colored image -- no Isaac Sim needed, this is
pure post-processing of a point cloud that Isaac already produced.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("ply_path")
parser.add_argument("--out", default=None)
parser.add_argument("--point-size", type=float, default=0.6)
parser.add_argument("--elev", type=float, default=35.0)
parser.add_argument("--azim", type=float, default=-60.0)
args = parser.parse_args()

with open(args.ply_path) as f:
    lines = f.readlines()
hdr_end = lines.index("end_header\n") + 1
data = np.array([[float(v) for v in l.split()] for l in lines[hdr_end:]])
x, y, z = data[:, 0], data[:, 1], data[:, 2]
print(f"{len(data)} points, area ~{(x.max()-x.min())*(y.max()-y.min()):.0f} m^2 "
      f"-> {len(data)/max((x.max()-x.min())*(y.max()-y.min()), 1e-6):.2f} pts/m^2")

fig = plt.figure(figsize=(12, 10), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_facecolor("black")
ax.scatter(x, y, z, c=z, s=args.point_size, cmap="turbo", linewidths=0, depthshade=True)
ax.view_init(elev=args.elev, azim=args.azim)
ax.set_box_aspect([x.max() - x.min(), y.max() - y.min(), max((z.max() - z.min()) * 3, 1.0)])

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0, 0, 0, 1))
    axis.line.set_color((0, 0, 0, 0))
    axis.set_ticklabels([])
    axis._axinfo["grid"]["color"] = (0, 0, 0, 0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_axis_off()

out = args.out or (args.ply_path.rsplit(".", 1)[0] + "_render.png")
plt.savefig(out, dpi=200, facecolor="black")
print("saved", out)
