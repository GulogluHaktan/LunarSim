"""Computed (ray-cast) descent-LiDAR scans for a recorded, validated landing.

Adds `lidar_scans/*.ply` and `lidar_index.csv` to a capture directory written by
`isaaclab_replay_capture.py --no-rtx-lidar`, so the existing
`compose_capture_video.py` builds its LiDAR terrain map and landability mask
from them unchanged.

WHY COMPUTED AND NOT RTX. The RTX LiDAR returned zero hits in every replay scene
tried (native Isaac 6.1 and the 6.0.1 Docker image, GPU and CPU physics, with
and without a collision mesh, parented and free-standing). The one pipeline in
the repo where it does return hits (`isaaclab_controller_eval_capture.py`) flies
its own physics loop, and that loop was measured to descend ~3x slower than the
training/validation env from the same release -- so a video from it would not
show the validated flight. This script keeps the validated flight and computes
what the sensor would see. It is labelled as computed wherever it is shown.

WHAT IS THE SAME AS THE REAL SENSOR. The mount (`lidar_mount_pose`: the descent
sensor height on the body axis, `LIDAR_MOUNT_TILT_DEG` forward tilt), the
spherical-to-cartesian convention (`spherical_to_cartesian`), the Ouster OS2's
vertical field of view (+-11.25 deg, 128 channels in the real unit) and a full
360 deg azimuth sweep, a 0.25 s scan cadence (what the old captures kept), and
world-frame output with the sensor pose in `lidar_index.csv`. Rays are
intersected with the recorded heightfield of the tile this episode flew over,
the same piecewise-linear surface PhysX collided with.

WHAT IS NOT. No beam divergence, no noise, no dropouts, no intensity model
beyond the cosine of the incidence angle, and the pattern is subsampled
(`--channels` x `--azimuths`, default 64 x 1024 rays) to keep a scan under a
second. The map it produces is therefore cleaner than a real sensor's.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.adapters.isaac.sensors import export_rtx_point_cloud, lidar_mount_pose  # noqa: E402
from lunarsim.core.metadata.lidar import euler_to_rotation_matrix, spherical_to_cartesian  # noqa: E402
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs  # noqa: E402

SPECS = ApolloLMSpecs()


def bilinear(h, res, x, y):
    """Height of the piecewise-linear surface at (x, y); NaN off the tile.
    `h[i, j]` sits at x = (i - c)*res, y = (j - c)*res (the project's convention,
    see `core.terrain.rocks.sample_height_at`)."""
    n = h.shape[0]
    c = (n - 1) / 2.0
    fi = x / res + c
    fj = y / res + c
    out = np.full(x.shape, np.nan)
    ok = (fi >= 0) & (fi <= n - 1) & (fj >= 0) & (fj <= n - 1)
    i0 = np.clip(np.floor(fi[ok]).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fj[ok]).astype(int), 0, n - 2)
    di = fi[ok] - i0
    dj = fj[ok] - j0
    out[ok] = (h[i0, j0] * (1 - di) * (1 - dj) + h[i0 + 1, j0] * di * (1 - dj)
               + h[i0, j0 + 1] * (1 - di) * dj + h[i0 + 1, j0 + 1] * di * dj)
    return out


def cast(h, res, origin, dirs, max_range, step):
    """First crossing of each ray with the surface: march at `step`, then refine
    linearly inside the bracketing step. Returns hit points and ranges."""
    ts = np.arange(step, max_range + step, step)
    hit_t = np.full(dirs.shape[0], np.nan)
    prev_gap = None
    prev_t = 0.0
    alive = np.ones(dirs.shape[0], dtype=bool)
    for t in ts:
        p = origin[None, :] + dirs * t
        gap = p[:, 2] - bilinear(h, res, p[:, 0], p[:, 1])
        if prev_gap is not None:
            cross = alive & np.isfinite(gap) & np.isfinite(prev_gap) & (prev_gap > 0) & (gap <= 0)
            frac = prev_gap[cross] / (prev_gap[cross] - gap[cross])
            hit_t[cross] = prev_t + frac * (t - prev_t)
            alive &= ~cross
            alive &= ~(~np.isfinite(gap))  # left the tile: no return
        prev_gap, prev_t = gap, t
        if not alive.any():
            break
    ok = np.isfinite(hit_t)
    return origin[None, :] + dirs[ok] * hit_t[ok, None], hit_t[ok], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--env", type=int, required=True)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--every-s", type=float, default=0.25)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--azimuths", type=int, default=1024)
    ap.add_argument("--vfov-deg", type=float, default=22.5, help="Ouster OS2: 22.5 deg")
    ap.add_argument("--max-range-m", type=float, default=200.0)
    ap.add_argument("--step-m", type=float, default=0.75)
    args = ap.parse_args()

    d = np.load(args.recording)
    g = lambda k: d[f"e{args.env}_{k}"]  # noqa: E731
    x, y, alt, ground = g("x"), g("y"), g("alt"), g("ground_z")
    tx, ty, yaw = g("tilt_x"), g("tilt_y"), g("yaw")
    tilt = np.hypot(tx, ty)
    z_c = ground + alt + SPECS.height_m / 2 * np.cos(tilt) + SPECS.footpad_span_m / 2 * np.sin(tilt)
    h = g("tile_height").astype(np.float64)
    res = float(g("tile_res"))
    dt = float(d["dt"])
    t = np.arange(len(x)) * dt

    el = np.linspace(-args.vfov_deg / 2, args.vfov_deg / 2, args.channels)
    az = np.linspace(-180.0, 180.0, args.azimuths, endpoint=False)
    AZ, EL = np.meshgrid(az, el)
    dirs_s = spherical_to_cartesian(AZ.ravel(), EL.ravel(), np.ones(AZ.size))
    dirs_s = dirs_s / np.linalg.norm(dirs_s, axis=1, keepdims=True)

    scan_dir = os.path.join(args.capture_dir, "lidar_scans")
    os.makedirs(scan_dir, exist_ok=True)
    rows = []
    next_t, scan_i = 0.0, 0
    for k in range(len(t)):
        if t[k] + 1e-9 < next_t:
            continue
        next_t = t[k] + args.every_s
        R = euler_to_rotation_matrix(float(tx[k]), float(ty[k]), float(yaw[k]))
        pos, rot = lidar_mount_pose(SPECS, np.array([x[k], y[k], z_c[k]]), R)
        dirs_w = dirs_s @ rot.T
        pts, rng, ok = cast(h, res, pos, dirs_w, args.max_range_m, args.step_m)
        if pts.shape[0] == 0:
            continue
        # cosine of incidence against the local surface normal, as an intensity
        eps = 0.5
        hx = (bilinear(h, res, pts[:, 0] + eps, pts[:, 1]) - bilinear(h, res, pts[:, 0] - eps, pts[:, 1])) / (2 * eps)
        hy = (bilinear(h, res, pts[:, 0], pts[:, 1] + eps) - bilinear(h, res, pts[:, 0], pts[:, 1] - eps)) / (2 * eps)
        nrm = np.stack([-np.nan_to_num(hx), -np.nan_to_num(hy), np.ones(len(pts))], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        inten = np.abs(np.sum(nrm * dirs_w[ok], axis=1))
        fn = f"scan_{scan_i:04d}.ply"
        export_rtx_point_cloud({"x_m": pts[:, 0], "y_m": pts[:, 1], "z_m": pts[:, 2],
                                "intensity": inten, "range_m": rng}, os.path.join(scan_dir, fn))
        rows.append([scan_i, fn, k, round(float(t[k]), 4), int(pts.shape[0])]
                    + [round(float(v), 5) for v in pos]
                    + [round(float(rot[a][b]), 7) for a in range(3) for b in range(3)])
        scan_i += 1
        if scan_i % 20 == 0:
            print(f"  scan {scan_i}: t={t[k]:.1f}s alt={alt[k]:.1f}m hits={pts.shape[0]}", flush=True)

    with open(os.path.join(args.capture_dir, "lidar_index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scan_index", "filename", "step", "t_s", "hit_count", "sensor_x_m", "sensor_y_m",
                    "sensor_z_m"] + [f"sensor_r{a}{b}" for a in range(3) for b in range(3)])
        w.writerows(rows)
    with open(os.path.join(args.capture_dir, "lidar_SOURCE.txt"), "w") as fh:
        fh.write("COMPUTED scans: ray-cast against the recorded heightfield with the OS2 field of view "
                 "and the real mount pose. Not RTX sensor output. See scripts/synth_lidar_scans.py.\n")
    print(f"wrote {scan_i} computed scans -> {scan_dir}")


if __name__ == "__main__":
    main()
