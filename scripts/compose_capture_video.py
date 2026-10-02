"""Post-processing only -- no Isaac Sim needed. Composites the outputs of
the capture scripts (`isaaclab_static_telemetry_capture.py`,
`isaaclab_controller_eval_capture.py`, `isaaclab_policy_eval_capture.py`:
chase_frames/, nav_frames/, telemetry.csv, frames_index.csv,
lidar_index.csv, lidar_scans/*.ply) into:

1. A 4-panel dashboard video (top-left: external chase camera, top-right:
   nose-down nav camera, bottom-left: a scrolling IMU plot with a "now"
   marker, bottom-right: the LiDAR MAP being built) at the camera frames'
   own timeline/cadence.
2. A standalone 2-panel mapping video, one frame per REAL captured scan:
   the accumulating elevation map next to the landability mask derived from
   it, so the map visibly fills in as the vehicle descends.
3. The final map as `<capture>_map.png` and `<capture>_map.npz` (the raw
   layers: elevation, coverage, relief, roughness, slope, landable).

WHAT CHANGED AND WHY (the LiDAR panel used to be a per-scan point cloud):
the old bottom-right panel plotted one scan at a time as a floating 3D
point cloud with a ray fan. Two things were wrong with that.

  * It threw away the only thing a descent LiDAR is actually for. A lander
    does not get one good look at its site, it gets a few hundred partial,
    oblique looks on the way down, and the product is the MAP those build
    up to -- plus, from the map, where the vehicle could actually set down.
    That is what the bottom-right panel now shows, and the landability
    overlay uses the same footpad-span criterion the simulator's own
    `landed_safely` test uses (see `core.metadata.mapping`).

  * REAL BUG: it was plotting the wrong numbers entirely. Isaac's RTX LiDAR
    reports its element arrays in the coordinate type `gmo.elementsCoordsType`
    names, which for these Ouster profiles is SPHERICAL -- azimuth deg,
    elevation deg, range m -- and `adapters.isaac.sensors.get_point_cloud`
    passed those straight through as cartesian metres. So every `.ply` under
    `out/*/lidar_scans/` holds (azimuth, elevation, range) under the property
    names (x, y, z), and the old panel was plotting an azimuth-vs-elevation-
    vs-range scatter as if it were a shape in space. The adapter now converts
    properly; this script detects and converts legacy spherical captures so
    the scans already on disk are usable without re-running Isaac (see
    `--scan-format`).

Reconstructing world points from a legacy spherical capture needs the
sensor's pose per scan, which old `lidar_index.csv` files do not carry, so
it is rebuilt from telemetry.csv's vehicle pose plus the mount constants the
capture scripts author (`--mount-tilt-deg` / `--mount-z-m`). That
reconstruction was validated against the simulator's own ground truth on
`out/static_capture`: cross-scan agreement (per-cell height spread over 98
scans taken from altitudes 227 m down to touchdown) is 0.12 m median, and
the map's height under the vehicle matches `telemetry.csv`'s own
`pos_z - height/2 - alt_m` to a median of -0.13 m. Getting the mount
rotation or the azimuth convention wrong instead gives 18-35 m. See
`--pose-lag-s` for the one remaining correction.

New captures write cartesian world-frame points plus the sensor pose into
`lidar_index.csv`, so none of this reconstruction runs for them.

Uses `frames_index.csv`/`lidar_index.csv` (written by the capture scripts
themselves) to know exactly which physics step/t_s each PNG/`.ply`
corresponds to, rather than re-deriving a render/lidar cadence from CLI
defaults that could drift out of sync with what a given run actually used.

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
from matplotlib.colors import ListedColormap
from PIL import Image

from lunarsim.core.metadata.lidar import (
    euler_to_rotation_matrix,
    sensor_to_world,
    spherical_to_cartesian,
)
from lunarsim.core.metadata.mapping import MapGrid, TerrainMap

# LEGACY mount constants, used only to rebuild the sensor pose for captures
# that predate `lidar_index.csv` carrying it: those runs authored
# `AddTranslateOp(0, 0, -specs.height_m * 0.3)` then `AddRotateYOp(60)` with
# ApolloLMSpecs.height_m = 7.04. Deliberately NOT the current mount (see
# `adapters.isaac.sensors.descent_sensor_mount_z_m`, which moved the sensor
# once the visual mesh was placed on its true contact plane) -- these have to
# describe the run being read, not the code as it stands today. New captures
# record their own pose and never reach this path.
DEFAULT_MOUNT_TILT_DEG = 60.0
DEFAULT_MOUNT_Z_M = -7.04 * 0.3

parser = argparse.ArgumentParser()
parser.add_argument("--capture-dir", type=str, default="out/static_capture")
parser.add_argument("--out", type=str, default=None, help="default: <capture-dir>_dashboard.mp4")
parser.add_argument("--map-out", "--lidar-out", dest="map_out", type=str, default=None,
                    help="standalone mapping video; default: <capture-dir>_map.mp4. "
                         "(--lidar-out is the old name for this, still accepted.)")
parser.add_argument("--fps", type=float, default=None, help="output video fps; default matches the camera frame rate")
parser.add_argument("--map-fps", "--lidar-fps", dest="map_fps", type=float, default=6.0,
                    help="output fps for the standalone mapping video (one frame per real scan)")
parser.add_argument("--panel-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
parser.add_argument("--map-cell-m", type=float, default=2.0,
                    help="map grid cell size. 2 m is a compromise: the footpad span is 9.4 m, so "
                         "cells must be a good bit smaller than that for the landability test to "
                         "mean anything, while a cell has to collect enough hits to have a height.")
parser.add_argument("--map-half-extent-m", type=float, default=None,
                    help="map covers [-X, +X] in world x and y. Default: sized from the scans "
                         "themselves (98th percentile hit radius, rounded up).")
parser.add_argument("--map-min-range-m", type=float, default=None,
                    help="drop returns closer than this. The sensor is mounted ON the vehicle, so "
                         "the nearest returns are the lander's own structure, which would otherwise "
                         "be mapped as terrain. 6 m comes from the vehicle's measured envelope seen "
                         "from the sensor: the engine bell reaches ~1.4 m and the legs and footpads "
                         "~5.2 m, nothing further. The old 12 m default was set when the sensor hung "
                         "below a mis-placed mesh and discarded far more than structure -- on a "
                         "28 m-release capture it threw away 68%% of all returns (1.53M points), most "
                         "of them real ground from the second half of the descent. The last second "
                         "before touchdown still falls inside 6 m and is dropped; by then the map "
                         "under the vehicle is already dense. Default: chosen from the scan format, "
                         "because the two mounts have different envelopes -- 6 m for world-frame "
                         "(current) captures, 12 m for legacy spherical ones, whose sensor hung "
                         "BELOW the mesh and saw its own vehicle out to ~9 m (measured: a clear "
                         "self-hit cluster at 2.9-6 m and a further 0.97%% of returns at 6-9 m).")
parser.add_argument("--scan-format", type=str, default="auto",
                    choices=["auto", "spherical", "world"],
                    help="'spherical' = legacy capture whose .ply columns are really "
                         "(azimuth_deg, elevation_deg, range_m); 'world' = cartesian world-frame "
                         "points. 'auto' decides from the data and prints its evidence.")
parser.add_argument("--pose-lag-s", type=float, default=2.0 / 60.0,
                    help="how far BEFORE a scan's recorded step the sensor pose used to place it "
                         "is taken from, for legacy spherical captures. A spinning LiDAR's frame "
                         "is acquired over the preceding revolution, so the pose logged at the "
                         "step the scan was READ is already too late. Measured against ground "
                         "truth on out/static_capture (a 33 m/s free fall, where this matters "
                         "most): map-vs-truth height error +2.40 m at 0 s, -0.13 m at 2 steps "
                         "(0.033 s), -1.78 m at 3 steps, and cross-scan spread bottoms out at the "
                         "same 2 steps. On a gentle powered descent it is within noise (0.044 vs "
                         "0.046 m cross-scan spread), so this default is safe for both.")
parser.add_argument("--mount-tilt-deg", type=float, default=DEFAULT_MOUNT_TILT_DEG,
                    help="nose-down tilt of the LiDAR mount about the vehicle's Y axis")
parser.add_argument("--mount-z-m", type=float, default=DEFAULT_MOUNT_Z_M,
                    help="LiDAR mount offset along the vehicle's Z axis")
parser.add_argument("--skip-dashboard", action="store_true",
                    help="only build the standalone mapping video + final map artifacts")
args = parser.parse_args()

capture_dir = args.capture_dir
chase_dir = os.path.join(capture_dir, "chase_frames")
nav_dir = os.path.join(capture_dir, "nav_frames")
lidar_dir = os.path.join(capture_dir, "lidar_scans")
telemetry_path = os.path.join(capture_dir, "telemetry.csv")
frames_index_path = os.path.join(capture_dir, "frames_index.csv")
lidar_index_path = os.path.join(capture_dir, "lidar_index.csv")

out_video = args.out or (capture_dir.rstrip("/") + "_dashboard.mp4")
map_video = args.map_out or (capture_dir.rstrip("/") + "_map.mp4")
map_png = capture_dir.rstrip("/") + "_map.png"
map_npz = capture_dir.rstrip("/") + "_map.npz"
panel_w, panel_h = args.panel_size

composite_frames_dir = os.path.join(capture_dir, "dashboard_frames")
map_frames_dir = os.path.join(capture_dir, "map_frames")
os.makedirs(composite_frames_dir, exist_ok=True)
os.makedirs(map_frames_dir, exist_ok=True)


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
track_xy = np.array([[float(r["pos_x_m"]), float(r["pos_y_m"])] for r in telemetry_rows])

if args.fps is not None:
    out_fps = args.fps
elif len(frames_rows) >= 2:
    dt_between_frames = float(frames_rows[1]["t_s"]) - float(frames_rows[0]["t_s"])
    out_fps = 1.0 / dt_between_frames if dt_between_frames > 0 else 30.0
else:
    out_fps = 30.0
print(f"telemetry rows: {len(telemetry_rows)}, camera frames: {len(frames_rows)}, "
      f"lidar scans: {len(lidar_rows)}, dashboard fps: {out_fps:.1f}")

# per-physics-step vehicle pose. `lidar_index.csv`'s "step" is the exact same
# physics-step counter `telemetry.csv`'s "frame" column is, both written by
# the same capture-script loop iteration, so this is an exact lookup rather
# than a nearest-match.
_pose_by_step = {
    int(r["frame"]): (
        np.array([float(r["pos_x_m"]), float(r["pos_y_m"]), float(r["pos_z_m"])]),
        np.deg2rad([float(r["roll_deg"]), float(r["pitch_deg"]), float(r["yaw_deg"])]),
    )
    for r in telemetry_rows
}
_steps_sorted = np.array(sorted(_pose_by_step))
_dt_s = (t_all[1] - t_all[0]) if len(t_all) >= 2 else 1.0 / 60.0
_pose_lag_steps = int(round(args.pose_lag_s / _dt_s)) if _dt_s > 0 else 0


def _read_ascii_ply(path):
    """Minimal ASCII-PLY reader for the exact format
    `lunarsim.core.metadata.lidar.export_point_cloud` writes (three coordinate
    columns + intensity per vertex line) -- avoids adding a PLY-parsing
    dependency for a format this same codebase controls both ends of.

    NOTE the three columns are NOT necessarily x/y/z metres despite the
    property names; see this module's docstring and `--scan-format`.
    """
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "end_header")
    pts = np.array([[float(v) for v in l.split()] for l in lines[header_end + 1:] if l.strip()])
    if pts.size == 0:
        return np.empty((0, 3)), np.empty(0)
    return pts[:, :3], pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts))


# ---------------------------------------------------------------------------
# load scans, decide their coordinate type, convert to world
# ---------------------------------------------------------------------------
raw_scans = []  # (row, cols(n,3), intensity(n,))
for row in lidar_rows:
    path = os.path.join(lidar_dir, row["filename"])
    if not os.path.exists(path):
        print(f"  WARNING: {row['filename']} listed in lidar_index.csv but missing on disk")
        continue
    cols, inten = _read_ascii_ply(path)
    raw_scans.append((row, cols, inten))

if not raw_scans:
    raise SystemExit(f"no readable lidar scans under {lidar_dir}")


def _detect_scan_format(scans) -> str:
    """Decide whether a capture's `.ply` columns are spherical
    (azimuth_deg, elevation_deg, range_m) or cartesian world metres.

    Two independent signatures, both required:

      * bounds -- spherical columns must fit |az| <= 180, |el| <= 90,
        range >= 0. Necessary but not sufficient: a small map of terrain
        that happens to sit above z = 0 satisfies all three too.
      * shape -- in a spherical scan the three columns have wildly
        different spans: azimuth sweeps the full circle, elevation covers
        only the sensor's fixed vertical FOV (tens of degrees at most), and
        range covers the whole descent distance. So span(col1) is a small
        fraction of span(col0), and span(col2) is much LARGER than
        span(col1). A world-frame cloud is the other way round: col0/col1
        are the two horizontal tile extents (comparable to each other, both
        large) and col2 is terrain relief (small).

    Measured on the captures in this repo: the spherical ones give spans of
    360 / 22 / 347, which passes; forcing a correctly-converted world cloud
    through the same test gives roughly 400 / 400 / 2, which fails on both
    shape conditions.
    """
    populated = [c for _r, c, _i in scans if len(c) >= 32]
    if not populated:
        return "world"
    stacked = np.concatenate(populated, axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    span = hi - lo
    bounds_ok = bool(abs(lo[0]) <= 180.001 and abs(hi[0]) <= 180.001
                     and abs(lo[1]) <= 90.001 and abs(hi[1]) <= 90.001
                     and lo[2] >= 0.0)
    shape_ok = bool(span[1] < 0.5 * span[0] and span[2] > span[1])
    print(f"  scan-format detection: column spans "
          f"[{span[0]:.1f}, {span[1]:.1f}, {span[2]:.1f}] over ranges "
          f"col0 [{lo[0]:.1f}, {hi[0]:.1f}], col1 [{lo[1]:.1f}, {hi[1]:.1f}], "
          f"col2 [{lo[2]:.1f}, {hi[2]:.1f}]")
    print(f"    angular bounds {'ok' if bounds_ok else 'violated'}; "
          f"spherical span shape {'ok' if shape_ok else 'violated'}")
    return "spherical" if (bounds_ok and shape_ok) else "world"


scan_format = args.scan_format
if scan_format == "auto":
    scan_format = _detect_scan_format(raw_scans)
    print(f"  -> treating scans as {scan_format.upper()}")
else:
    print(f"  scan format forced to {scan_format.upper()}")

_POSE_KEYS = ["sensor_x_m", "sensor_y_m", "sensor_z_m"] + [
    f"sensor_r{i}{j}" for i in range(3) for j in range(3)
]
_index_has_pose_columns = all(k in lidar_rows[0] for k in _POSE_KEYS)
if _index_has_pose_columns:
    print("  lidar_index.csv carries sensor-pose columns -- using them where they are filled in")


def _recorded_pose(row):
    """The pose the capture recorded for this scan, or None.

    The columns can be present but EMPTY: the capture writes blanks when the
    sensor reported world-frame points directly, because then there was no
    sensor->world transform to record. Parsing those blanks as floats would
    raise, so an unfilled row falls back to reconstruction like a legacy one.
    """
    if not _index_has_pose_columns:
        return None
    try:
        values = [float(row[k]) for k in _POSE_KEYS]
    except (TypeError, ValueError):
        return None
    pos = np.array(values[:3])
    rot = np.array(values[3:]).reshape(3, 3)
    return pos, rot


def _sensor_pose(row):
    """`(sensor_pos_world, sensor_to_world_rotation)` for one scan.

    Prefers the pose the capture recorded; falls back to rebuilding it from
    the vehicle pose in telemetry.csv plus the mount constants.
    """
    recorded = _recorded_pose(row)
    if recorded is not None:
        return recorded
    step = int(row["step"]) - _pose_lag_steps
    # clamp into the recorded range rather than skipping: the first scans of
    # a capture can be read a step or two before the lag window exists.
    step = int(np.clip(step, _steps_sorted[0], _steps_sorted[-1]))
    if step not in _pose_by_step:
        step = int(_steps_sorted[np.searchsorted(_steps_sorted, step).clip(0, len(_steps_sorted) - 1)])
    pos, (roll, pitch, yaw) = _pose_by_step[step]
    r_vehicle = euler_to_rotation_matrix(roll, pitch, yaw)
    r_mount = euler_to_rotation_matrix(0.0, np.deg2rad(args.mount_tilt_deg), 0.0)
    sensor_pos = pos + r_vehicle @ np.array([0.0, 0.0, args.mount_z_m])
    return sensor_pos, r_vehicle @ r_mount


# the legacy mount hung below a mis-placed mesh and saw the vehicle out to
# ~9 m; the current one sits inside the structure's 5.3 m envelope. The scan
# format tells the two apart because both changed in the same fix.
min_range_m = args.map_min_range_m
if min_range_m is None:
    min_range_m = 12.0 if scan_format == "spherical" else 6.0
    print(f"  near-range cull: {min_range_m:.1f} m (from the {scan_format} mount's own envelope)")

print(f"converting {len(raw_scans)} scans to world frame "
      f"(dropping returns closer than {min_range_m:.1f} m)...")
world_scans = []  # (row, points_world(n,3) float32, intensity(n,) float32, sensor_pos(3,))
n_self_hits = 0
for row, cols, inten in raw_scans:
    sensor_pos, rot = _sensor_pose(row)
    if scan_format == "spherical":
        keep = cols[:, 2] >= min_range_m
        n_self_hits += int((~keep).sum())
        pts = sensor_to_world(
            spherical_to_cartesian(cols[keep, 0], cols[keep, 1], cols[keep, 2]), sensor_pos, rot
        )
        inten_keep = inten[keep]
    else:
        rng = np.linalg.norm(cols - sensor_pos, axis=1)
        keep = rng >= min_range_m
        n_self_hits += int((~keep).sum())
        pts, inten_keep = cols[keep], inten[keep]
    world_scans.append((row, pts.astype(np.float32), inten_keep.astype(np.float32), sensor_pos))
print(f"  dropped {n_self_hits} near-range returns (vehicle self-hits)")

all_radius = np.concatenate(
    [np.maximum(np.abs(p[:, 0]), np.abs(p[:, 1])) for _r, p, _i, _s in world_scans if len(p)]
) if any(len(p) for _r, p, _i, _s in world_scans) else np.array([100.0])
if args.map_half_extent_m is not None:
    half_extent = args.map_half_extent_m
else:
    # the 98th percentile of hit radius drops the sparse grazing-angle
    # horizon, but the grid must still contain the whole ground track --
    # on a capture where the vehicle coasts off the collidable tile, the
    # track runs well past the last LiDAR return and would otherwise be
    # clipped out of the panel.
    track_radius = float(np.abs(track_xy).max()) if len(track_xy) else 0.0
    half_extent = max(float(np.percentile(all_radius, 98)), track_radius * 1.05)
    half_extent = float(np.ceil(half_extent / args.map_cell_m) * args.map_cell_m)
    half_extent = float(np.clip(half_extent, 50.0, 1200.0))
grid = MapGrid(cell_m=args.map_cell_m, half_extent_m=half_extent)
print(f"map grid: {grid.n}x{grid.n} cells of {grid.cell_m:.1f} m, covering "
      f"[{-half_extent:.0f}, {half_extent:.0f}] m in x and y")


# ---------------------------------------------------------------------------
# a full pass first, to fix the colour scale (and to report the final map)
# ---------------------------------------------------------------------------
final_map = TerrainMap(grid)
for row, pts, inten, _sensor_pos in world_scans:
    final_map.integrate(pts, inten, int(row["scan_index"]))
final_summary = final_map.summary()
print("final map:")
for k, v in final_summary.items():
    print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

_final_elev = final_map.elevation_m()
if np.isfinite(_final_elev).any():
    # 5-95 rather than the full range: a handful of crater-floor cells sit
    # ~10 m below everything else, and scaling to them flattens the entire
    # map to one colour. The colorbar states the range being shown.
    elev_vmin, elev_vmax = (float(np.nanpercentile(_final_elev, 5)),
                            float(np.nanpercentile(_final_elev, 95)))
    if elev_vmax - elev_vmin < 1e-3:
        elev_vmin, elev_vmax = elev_vmin - 0.5, elev_vmax + 0.5
else:
    elev_vmin, elev_vmax = -1.0, 1.0

np.savez_compressed(
    map_npz,
    elevation_m=_final_elev,
    coverage=final_map.coverage(),
    relief_m=final_map.relief_m(),
    roughness_m=final_map.roughness_m(),
    slope_deg=final_map.slope_deg(),
    intensity=final_map.intensity(),
    landable=final_map.landable_mask(),
    cell_m=grid.cell_m,
    half_extent_m=grid.half_extent_m,
    extent=np.array(grid.extent),
    track_xy=track_xy,
)
print(f"wrote {map_npz}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
_UNSEEN = "#07080c"
_LANDABLE_CMAP = ListedColormap([(0, 0, 0, 0), (0.25, 1.0, 0.45, 0.55)])


# `landable_mask()` bilinearly samples the elevation layer four times over the
# whole grid, and `summary()` calls it again -- on a 450x450 map that is ~0.8M
# samples. The dashboard redraws at the camera rate (30 fps) while scans only
# arrive at 10 Hz, so recomputing per frame does the same work three times
# over. Memoize on the scan count, which is exactly what changes the map.
_derived_cache: dict = {}


def _derived(tmap):
    """`(elevation, landable, summary)` for `tmap`, recomputed only when a new
    scan has actually been folded in."""
    key = (id(tmap), tmap.n_scans, tmap.n_points_integrated)
    hit = _derived_cache.get("key")
    if hit != key:
        _derived_cache["key"] = key
        _derived_cache["value"] = (tmap.elevation_m(), tmap.landable_mask(), tmap.summary())
    return _derived_cache["value"]


def _style_map_axis(ax, title):
    ax.set_facecolor(_UNSEEN)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.set_xlabel("x (m)", color="white", fontsize=7)
    ax.set_ylabel("y (m)", color="white", fontsize=7)


def _draw_track(ax, upto_index, sensor_pos=None):
    if upto_index > 1:
        ax.plot(track_xy[:upto_index, 0], track_xy[:upto_index, 1],
                color="white", linewidth=0.9, alpha=0.75)
    if upto_index >= 1:
        ax.plot([track_xy[upto_index - 1, 0]], [track_xy[upto_index - 1, 1]],
                marker="o", markersize=4, color="#ff4444", markeredgecolor="white", markeredgewidth=0.5)
    if sensor_pos is not None:
        ax.plot([sensor_pos[0]], [sensor_pos[1]], marker="+", markersize=7,
                color="#66ddff", markeredgewidth=1.0)


def _nearest_telemetry_index(t_now):
    return int(np.clip(np.searchsorted(t_all, t_now), 1, len(t_all)))


def _render_map_panel(tmap, w_px, h_px, title, t_now, current_scan=None, sensor_pos=None,
                      show_landable=True, show_colorbar=True):
    """Top-down accumulating elevation map, with the landability mask
    overlaid, the ground track so far, and the latest scan's footprint."""
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.11, 0.11, 0.74 if show_colorbar else 0.86, 0.80])

    elev, landable, summary = _derived(tmap)
    im = ax.imshow(np.ma.masked_invalid(elev), origin="lower", extent=grid.extent,
                   cmap="cividis", vmin=elev_vmin, vmax=elev_vmax, interpolation="nearest")
    if show_landable:
        ax.imshow(landable.astype(float), origin="lower", extent=grid.extent,
                  cmap=_LANDABLE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    if current_scan is not None and len(current_scan):
        step = max(1, len(current_scan) // 3000)
        ax.scatter(current_scan[::step, 0], current_scan[::step, 1], s=0.6,
                   c="#66ddff", alpha=0.35, linewidths=0)

    _draw_track(ax, _nearest_telemetry_index(t_now), sensor_pos)
    _style_map_axis(ax, title)

    ax.text(0.015, 0.975,
            f"scans {summary['n_scans']}   pts {summary['n_points'] / 1e3:.0f}k\n"
            f"mapped {summary['mapped_area_m2'] / 1e3:.1f}k m²  ({summary['coverage_frac'] * 100:.0f}%)\n"
            f"landable {summary['landable_area_m2'] / 1e3:.1f}k m²",
            transform=ax.transAxes, color="white", fontsize=6.5, va="top", ha="left",
            family="monospace",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0))

    if show_colorbar:
        cax = fig.add_axes([0.88, 0.11, 0.03, 0.80])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("elevation (m)", color="white", fontsize=7)
        cb.ax.tick_params(colors="white", labelsize=6)
        cb.outline.set_edgecolor("#555555")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(buf).resize((w_px, h_px))


def _render_map_pair(tmap, w_px, h_px, title, t_now, current_scan=None, sensor_pos=None):
    """Side-by-side elevation + landability, for the standalone video."""
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax_e = fig.add_axes([0.06, 0.10, 0.38, 0.78])
    ax_l = fig.add_axes([0.56, 0.10, 0.38, 0.78])

    elev, landable, summary = _derived(tmap)
    im = ax_e.imshow(np.ma.masked_invalid(elev), origin="lower", extent=grid.extent,
                     cmap="cividis", vmin=elev_vmin, vmax=elev_vmax, interpolation="nearest")
    if current_scan is not None and len(current_scan):
        step = max(1, len(current_scan) // 4000)
        ax_e.scatter(current_scan[::step, 0], current_scan[::step, 1], s=0.6,
                     c="#66ddff", alpha=0.4, linewidths=0)
    idx = _nearest_telemetry_index(t_now)
    _draw_track(ax_e, idx, sensor_pos)
    _style_map_axis(ax_e, "LiDAR elevation map")
    cax = fig.add_axes([0.455, 0.10, 0.016, 0.78])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("m", color="white", fontsize=7)
    cb.ax.tick_params(colors="white", labelsize=6)
    cb.outline.set_edgecolor("#555555")

    observed = tmap.observed
    # 0 = unseen, 1 = seen but rejected, 2 = landable
    classes = np.where(landable, 2.0, np.where(observed, 1.0, 0.0))
    ax_l.imshow(classes, origin="lower", extent=grid.extent, vmin=0, vmax=2,
                cmap=ListedColormap([_UNSEEN, "#8a3b3b", "#3fd97a"]), interpolation="nearest")
    _draw_track(ax_l, idx, sensor_pos)
    _style_map_axis(ax_l, "landable (footpad span ≤ 0.16 m, slope ≤ 15°)")

    frac = (summary["cells_landable"] / summary["cells_observed"]) if summary["cells_observed"] else 0.0
    ax_l.text(0.015, 0.975,
              f"mapped   {summary['mapped_area_m2'] / 1e3:7.1f}k m²\n"
              f"landable {summary['landable_area_m2'] / 1e3:7.1f}k m²  ({frac * 100:.0f}% of mapped)",
              transform=ax_l.transAxes, color="white", fontsize=6.5, va="top", ha="left",
              family="monospace",
              bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0))

    fig.suptitle(title, color="white", fontsize=10, y=0.975)
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
# 1) standalone mapping video -- one frame per REAL scan, raw cadence
# ---------------------------------------------------------------------------
print(f"rendering {len(world_scans)} standalone mapping frames...")
build_map = TerrainMap(grid)
for i, (row, pts, inten, sensor_pos) in enumerate(world_scans):
    build_map.integrate(pts, inten, int(row["scan_index"]))
    t_s = float(row["t_s"])
    img = _render_map_pair(
        build_map, panel_w * 2, panel_h,
        f"t = {t_s:5.2f} s     scan {i + 1}/{len(world_scans)}     {len(pts)} returns",
        t_s, current_scan=pts, sensor_pos=sensor_pos,
    )
    img.save(os.path.join(map_frames_dir, f"frame_{i:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-framerate", str(args.map_fps),
    "-i", os.path.join(map_frames_dir, "frame_%04d.png"),
    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
    map_video,
], check=True)
print(f"wrote {map_video}")

# final map still, at a readable size
_render_map_pair(
    final_map, 1920, 960,
    f"LiDAR terrain map -- {final_summary['n_scans']} scans, "
    f"{final_summary['n_points'] / 1e6:.2f}M returns",
    t_all[-1],
).save(map_png)
print(f"wrote {map_png}")

if args.skip_dashboard:
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 2) 4-panel dashboard video, on the camera frames' own timeline
# ---------------------------------------------------------------------------
print(f"rendering {len(frames_rows)} dashboard frames...")
dash_map = TerrainMap(grid)
next_scan = 0
last_scan_pts = None
last_scan_sensor_pos = None
last_scan_label = "waiting for first scan"

for row in frames_rows:
    fi = int(row["frame_index"])
    t_now = float(row["t_s"])

    chase_path = os.path.join(chase_dir, f"frame_{fi:04d}.png")
    nav_path = os.path.join(nav_dir, f"frame_{fi:04d}.png")
    if not (os.path.exists(chase_path) and os.path.exists(nav_path)):
        continue
    chase_img = Image.open(chase_path).convert("RGB").resize((panel_w, panel_h))
    nav_img = Image.open(nav_path).convert("RGB").resize((panel_w, panel_h))

    # fold in every scan acquired at or before this frame's time
    while next_scan < len(world_scans) and float(world_scans[next_scan][0]["t_s"]) <= t_now:
        srow, spts, sinten, spos = world_scans[next_scan]
        dash_map.integrate(spts, sinten, int(srow["scan_index"]))
        last_scan_pts, last_scan_sensor_pos = spts, spos
        last_scan_label = f"scan {next_scan + 1}/{len(world_scans)} @ t={float(srow['t_s']):.2f}s"
        next_scan += 1

    map_img = _render_map_panel(
        dash_map, panel_w, panel_h, f"LiDAR map -- {last_scan_label}", t_now,
        current_scan=last_scan_pts, sensor_pos=last_scan_sensor_pos,
    )
    imu_img = _render_imu_panel(t_now, panel_w, panel_h)

    composite = Image.new("RGB", (panel_w * 2, panel_h * 2))
    composite.paste(chase_img, (0, 0))
    composite.paste(nav_img, (panel_w, 0))
    composite.paste(imu_img, (0, panel_h))
    composite.paste(map_img, (panel_w, panel_h))
    composite.save(os.path.join(composite_frames_dir, f"frame_{fi:04d}.png"))

subprocess.run([
    "ffmpeg", "-y", "-framerate", str(out_fps),
    "-i", os.path.join(composite_frames_dir, "frame_%04d.png"),
    "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
    out_video,
], check=True)
print(f"wrote {out_video}")
