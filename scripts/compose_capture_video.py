"""Post-processing only -- no Isaac Sim needed. Composites the outputs of
`scripts/isaaclab_static_telemetry_capture.py` (chase_frames/, nav_frames/,
telemetry.csv, frames_index.csv, lidar_index.csv, lidar_scans/*.ply) into:

1. A 4-panel dashboard video (top-left: external chase camera, top-right:
   nose-down nav camera, bottom-right: the live LiDAR point cloud, bottom-
   left: a scrolling IMU plot with a "now" marker) at the camera frames'
   own timeline/cadence.
2. A standalone LiDAR-only video, one frame per REAL captured scan (not
   held/interpolated to the camera cadence) -- the raw scan sequence "yol
   boyunca" (along the way), as its own clip.

Uses `frames_index.csv`/`lidar_index.csv` (written by the capture script
itself) to know exactly which physics step/t_s each PNG/`.ply` corresponds
to, rather than re-deriving a render/lidar cadence from CLI defaults that
could drift out of sync with what a given run actually used.

Run: .venv/bin/python scripts/compose_capture_video.py --capture-dir out/static_capture
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--capture-dir", type=str, default="out/static_capture")
parser.add_argument("--out", type=str, default=None, help="default: <capture-dir>_dashboard.mp4")
parser.add_argument("--lidar-out", type=str, default=None, help="default: <capture-dir>_lidar.mp4")
parser.add_argument("--fps", type=float, default=None, help="output video fps; default matches the camera frame rate")
parser.add_argument("--lidar-fps", type=float, default=6.0, help="output fps for the standalone lidar-only video")
parser.add_argument("--panel-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
args = parser.parse_args()

capture_dir = args.capture_dir
chase_dir = os.path.join(capture_dir, "chase_frames")
nav_dir = os.path.join(capture_dir, "nav_frames")
lidar_dir = os.path.join(capture_dir, "lidar_scans")
telemetry_path = os.path.join(capture_dir, "telemetry.csv")
frames_index_path = os.path.join(capture_dir, "frames_index.csv")
lidar_index_path = os.path.join(capture_dir, "lidar_index.csv")

out_video = args.out or (capture_dir.rstrip("/") + "_dashboard.mp4")
lidar_video = args.lidar_out or (capture_dir.rstrip("/") + "_lidar.mp4")
panel_w, panel_h = args.panel_size

composite_frames_dir = os.path.join(capture_dir, "dashboard_frames")
lidar_frames_dir = os.path.join(capture_dir, "lidar_panel_frames")
os.makedirs(composite_frames_dir, exist_ok=True)
os.makedirs(lidar_frames_dir, exist_ok=True)


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


telemetry_rows = read_csv_rows(telemetry_path)
frames_rows = read_csv_rows(frames_index_path)
lidar_rows = read_csv_rows(lidar_index_path)

t_all = np.array([float(r["t_s"]) for r in telemetry_rows])
ang_vel = np.array([[float(r["ang_vel_x_rad_s"]), float(r["ang_vel_y_rad_s"]), float(r["ang_vel_z_rad_s"])]
                     for r in telemetry_rows])
lin_acc = np.array([[float(r["lin_acc_x_m_s2"]), float(r["lin_acc_y_m_s2"]), float(r["lin_acc_z_m_s2"])]
                     for r in telemetry_rows])
alt_m = np.array([float(r["alt_m"]) for r in telemetry_rows])

if args.fps is not None:
    out_fps = args.fps
elif len(frames_rows) >= 2:
    dt_between_frames = float(frames_rows[1]["t_s"]) - float(frames_rows[0]["t_s"])
    out_fps = 1.0 / dt_between_frames if dt_between_frames > 0 else 30.0
else:
    out_fps = 30.0
print(f"telemetry rows: {len(telemetry_rows)}, camera frames: {len(frames_rows)}, lidar scans: {len(lidar_rows)}, "
      f"dashboard fps: {out_fps:.1f}")

# per-physics-step vehicle position, for the ray-fan sensor marker below --
# `lidar_index.csv`'s "step" is the exact same physics-step counter
# `telemetry.csv`'s "frame" column is, both written by the same capture
# script loop iteration, so this is an exact lookup, not a nearest-match.
_pos_by_step = {int(r["frame"]): (float(r["pos_x_m"]), float(r["pos_y_m"]), float(r["pos_z_m"]))
                for r in telemetry_rows}


def _read_ascii_ply(path):
    """Minimal ASCII-PLY reader for the exact format
    `lunarsim.core.metadata.lidar.export_point_cloud` writes (x y z intensity
    per vertex line) -- avoids adding a PLY-parsing dependency for a format
    this same codebase controls both ends of."""
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "end_header")
    pts = np.array([[float(v) for v in l.split()] for l in lines[header_end + 1:] if l.strip()])
    if pts.size == 0:
        return np.empty((0, 3)), np.empty(0)
    return pts[:, :3], pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))


def _render_lidar_panel(points_xyz, intensity, w_px, h_px, title, azim_deg: float = -60.0,
                         sensor_pos=None, max_rays: int = 220):
    """3D point-cloud "UAV/ray-fan" view: the real x/y/z hits (colored by
    intensity), a ground reference grid at the hit field's own base level,
    and thin lines from the real vehicle position (`sensor_pos`, from
    telemetry.csv -- see `_pos_by_step`) out to a subsampled set of hit
    points, so the sensor's actual line-of-sight geometry reads at a
    glance instead of just a floating point cloud. `azim_deg` drifts
    slowly across frames (see call sites) for a subtle orbit."""
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.86], projection="3d")
    ax.set_facecolor("black")
    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))
    if len(points_xyz) > 0:
        cx, cy, cz = points_xyz[:, 0].mean(), points_xyz[:, 1].mean(), points_xyz[:, 2].mean()
        radius = max(5.0, float(np.percentile(
            np.hypot(points_xyz[:, 0] - cx, points_xyz[:, 1] - cy), 95)) * 1.3)
        z_lo, z_hi = float(points_xyz[:, 2].min()), float(points_xyz[:, 2].max())
        z_half = max(3.0, (z_hi - z_lo) * 0.75, float(points_xyz[:, 2].std()) * 2.5)
        z_lo = min(z_lo, cz - z_half)

        # ground reference grid, at the hit field's own lowest level
        grid_n = 10
        gx = np.linspace(cx - radius, cx + radius, grid_n)
        gy = np.linspace(cy - radius, cy + radius, grid_n)
        grid_color = (1, 1, 1, 0.18)
        for x in gx:
            ax.plot([x, x], [cy - radius, cy + radius], [z_lo, z_lo], color=grid_color, linewidth=0.5)
        for y in gy:
            ax.plot([cx - radius, cx + radius], [y, y], [z_lo, z_lo], color=grid_color, linewidth=0.5)

        # sensor position + ray fan to a subsampled set of hits
        if sensor_pos is not None:
            sx, sy, sz = sensor_pos
            n = len(points_xyz)
            ray_idx = np.linspace(0, n - 1, min(max_rays, n)).astype(int)
            for k in ray_idx:
                px, py, pz = points_xyz[k]
                ax.plot([sx, px], [sy, py], [sz, pz], color=(1, 1, 1, 0.25), linewidth=0.4)
            ax.scatter([sx], [sy], [sz], c="red", s=40, marker="^", depthshade=False)

        ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2],
                   c=intensity, cmap="viridis", s=3, linewidths=0, depthshade=True)
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_zlim(z_lo, cz + z_half)
    else:
        ax.text2D(0.5, 0.5, "no lidar hits", color="white", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    ax.view_init(elev=22, azim=azim_deg)
    ax.set_title(title, color="white", fontsize=10, pad=0)
    ax.tick_params(colors="white", labelsize=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(buf).resize((w_px, h_px))


def _render_imu_panel(t_now, w_px, h_px):
    fig, axes = plt.subplots(2, 1, figsize=(w_px / 100, h_px / 100), dpi=100, sharex=True)
    fig.patch.set_facecolor("black")
    labels = ["x", "y", "z"]
    colors = ["#ff5555", "#55ff55", "#5599ff"]
    for ax, data, title, ylabel in (
        (axes[0], ang_vel, "angular velocity", "rad/s"),
        (axes[1], lin_acc, "linear acceleration", "m/s^2"),
    ):
        ax.set_facecolor("black")
        for i in range(3):
            ax.plot(t_all, data[:, i], color=colors[i], linewidth=1.0, label=labels[i])
        ax.axvline(t_now, color="white", linewidth=1.2, alpha=0.9)
        ax.set_title(title, color="white", fontsize=9, loc="left")
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.legend(loc="upper right", fontsize=6, facecolor="black", labelcolor="white", framealpha=0.4)
    axes[-1].set_xlabel("t (s)", color="white", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(buf).resize((w_px, h_px))


# ---------------------------------------------------------------------------
# 1) standalone LiDAR video -- one frame per REAL scan, raw cadence
# ---------------------------------------------------------------------------
print(f"rendering {len(lidar_rows)} standalone lidar frames...")
for i, row in enumerate(lidar_rows):
    pts, inten = _read_ascii_ply(os.path.join(lidar_dir, row["filename"]))
    img = _render_lidar_panel(pts, inten, panel_w, panel_h,
                               f"t={float(row['t_s']):.2f}s  hits={row['hit_count']}",
                               azim_deg=-60.0 + 0.7 * i,
                               sensor_pos=_pos_by_step.get(int(row["step"])))
    img.save(os.path.join(lidar_frames_dir, f"frame_{i:04d}.png"))

if lidar_rows:
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(args.lidar_fps),
        "-i", os.path.join(lidar_frames_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        lidar_video,
    ], check=True)
    print(f"wrote {lidar_video}")
else:
    print("no lidar scans captured -- skipping standalone lidar video")

# ---------------------------------------------------------------------------
# 2) 4-panel dashboard video, on the camera frames' own timeline
# ---------------------------------------------------------------------------
print(f"rendering {len(frames_rows)} dashboard frames...")
last_scan_idx = -1
last_scan_pts, last_scan_inten, last_scan_label = np.empty((0, 3)), np.empty(0), "no lidar hits yet"
last_scan_sensor_pos = None

for row in frames_rows:
    fi = int(row["frame_index"])
    t_now = float(row["t_s"])

    chase_path = os.path.join(chase_dir, f"frame_{fi:04d}.png")
    nav_path = os.path.join(nav_dir, f"frame_{fi:04d}.png")
    if not (os.path.exists(chase_path) and os.path.exists(nav_path)):
        continue
    chase_img = Image.open(chase_path).convert("RGB").resize((panel_w, panel_h))
    nav_img = Image.open(nav_path).convert("RGB").resize((panel_w, panel_h))

    # hold the most recent lidar scan at or before this frame's time
    while last_scan_idx + 1 < len(lidar_rows) and float(lidar_rows[last_scan_idx + 1]["t_s"]) <= t_now:
        last_scan_idx += 1
        r = lidar_rows[last_scan_idx]
        last_scan_pts, last_scan_inten = _read_ascii_ply(os.path.join(lidar_dir, r["filename"]))
        last_scan_label = f"t={float(r['t_s']):.2f}s  hits={r['hit_count']}"
        last_scan_sensor_pos = _pos_by_step.get(int(r["step"]))
    lidar_img = _render_lidar_panel(last_scan_pts, last_scan_inten, panel_w, panel_h, last_scan_label,
                                     azim_deg=-60.0 + 0.5 * fi, sensor_pos=last_scan_sensor_pos)
    imu_img = _render_imu_panel(t_now, panel_w, panel_h)

    composite = Image.new("RGB", (panel_w * 2, panel_h * 2))
    composite.paste(chase_img, (0, 0))
    composite.paste(nav_img, (panel_w, 0))
    composite.paste(imu_img, (0, panel_h))
    composite.paste(lidar_img, (panel_w, panel_h))
    composite.save(os.path.join(composite_frames_dir, f"frame_{fi:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-framerate", str(out_fps),
    "-i", os.path.join(composite_frames_dir, "frame_%04d.png"),
    "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
    out_video,
], check=True)
print(f"wrote {out_video}")
