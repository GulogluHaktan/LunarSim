"""Landing-hazard map from a heightfield: where is the lander's footpad-height criterion satisfiable?

`landed_safely` requires the four footpads (radius ``footpad_span/2`` around the vehicle) to sit within
``safe_landing_max_leg_height_diff_m`` (0.16 m) of each other. On real highland terrain only a few percent of
locations pass, so a lander that cannot see terrain is capped at that percentage. This module turns a
tile's heightfield into

  * ``diff``  : footpad height spread at every grid cell (m),
  * ``dist``  : distance (m) from every cell to the nearest cell whose spread is <= ``diff_max``.

``dist`` is the quantity a planner/policy needs: it is 0 on usable ground and grows smoothly away from it.
The height sampler is the same triangulated-mesh lookup the Isaac contact model uses
(`collision_mesh_height_at`), so a cell that passes here passes the env's own criterion.
"""
from __future__ import annotations

import numpy as np


def _mesh_height(height, res_m, x_m, y_m):
    n = height.shape[0]
    c = (n - 1) / 2.0
    fi = np.clip(np.asarray(x_m, dtype=np.float64) / res_m + c, 0.0, n - 1.0)
    fj = np.clip(np.asarray(y_m, dtype=np.float64) / res_m + c, 0.0, n - 1.0)
    i0 = np.minimum(fi.astype(np.int64), n - 2)
    j0 = np.minimum(fj.astype(np.int64), n - 2)
    a, b = fi - i0, fj - j0
    h00, h10 = height[i0, j0], height[i0 + 1, j0]
    h01, h11 = height[i0, j0 + 1], height[i0 + 1, j0 + 1]
    return np.where(a >= b, h00 + (h10 - h00) * a + (h11 - h10) * b, h00 + (h01 - h00) * b + (h11 - h01) * a)


def footpad_spread(height: np.ndarray, res_m: float, radius_m: float = 4.7) -> np.ndarray:
    n = height.shape[0]
    ax = (np.arange(n) - (n - 1) / 2.0) * res_m
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    z = [_mesh_height(height, res_m, X + radius_m * np.cos(a), Y + radius_m * np.sin(a))
         for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])]
    from lunarsim.core.terrain.criterion import effective_leg_diff
    return effective_leg_diff(np.stack(z), radius_m).astype(np.float32)


def hazard_maps(height: np.ndarray, res_m: float, radius_m: float = 4.7, diff_max: float = 0.12):
    """-> (diff, dist): float32 (n, n) arrays, index order (i <-> x, j <-> y) like the heightfield."""
    from scipy.ndimage import distance_transform_edt

    diff = footpad_spread(np.asarray(height, dtype=np.float64), res_m, radius_m)
    safe = diff <= diff_max
    if not safe.any():
        dist = np.full(diff.shape, 1.0e3, dtype=np.float32)
    else:
        dist = (distance_transform_edt(~safe) * res_m).astype(np.float32)
    return diff, dist


def attach_hazard(tile, radius_m: float = 4.7, diff_max: float | None = None):
    """Cache ``tile.haz_diff`` / ``tile.haz_dist`` (computed once per tile and threshold).

    `LUNARSIM_HAZ_DIFFMAX` overrides the 'usable footpad site' threshold (default 0.12 m); a stricter value
    targets sites with margin below the 0.16 m landing criterion."""
    import os
    if diff_max is None:
        diff_max = float(os.environ.get("LUNARSIM_HAZ_DIFFMAX", "0.12"))
    from lunarsim.core.terrain.criterion import criterion_key
    key = (round(diff_max, 9),) + criterion_key()
    if getattr(tile, "haz_dist", None) is None or getattr(tile, "haz_key", None) != key:
        tile.haz_diff, tile.haz_dist = hazard_maps(tile.height, tile.res_m, radius_m, diff_max)
        tile.haz_thr = diff_max
        tile.haz_key = key
    return tile
