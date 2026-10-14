"""Terrain features for the ES policy: a 'hazard radar' read from the tile's distance-to-safe-cell map.

Seven numbers, all in units of 100 m, clipped to [0, 3]:
  d0       distance to the nearest usable footpad site from the vehicle's own position
  dA..dD   the same distance at points 50 / 100 / 150 / 250 m ahead along the velocity direction
  dL, dR   the same distance 100 m ahead and 40 m to the left / right of the velocity direction
A policy that sees these can trade braking distance and drift against where usable ground lies.
"""
from __future__ import annotations

import numpy as np

N_HAZ = 7
_AHEAD = (50.0, 100.0, 150.0, 250.0)


def _bilerp(m: np.ndarray, res_m: float, x, y):
    n = m.shape[0]; c = (n - 1) / 2.0
    fi = np.clip(np.asarray(x, dtype=np.float64) / res_m + c, 0.0, n - 1.0)
    fj = np.clip(np.asarray(y, dtype=np.float64) / res_m + c, 0.0, n - 1.0)
    i0 = np.minimum(fi.astype(np.int64), n - 2); j0 = np.minimum(fj.astype(np.int64), n - 2)
    a, b = fi - i0, fj - j0
    return (m[i0, j0] * (1 - a) * (1 - b) + m[i0 + 1, j0] * a * (1 - b)
            + m[i0, j0 + 1] * (1 - a) * b + m[i0 + 1, j0 + 1] * a * b)


def haz_features(x: float, y: float, vx: float, vy: float, dist_map: np.ndarray, res_m: float) -> np.ndarray:
    sp = float(np.hypot(vx, vy))
    ux, uy = (vx / sp, vy / sp) if sp > 0.5 else (1.0, 0.0)
    nx, ny = -uy, ux
    pts = [(x, y)] + [(x + L * ux, y + L * uy) for L in _AHEAD] \
        + [(x + 100.0 * ux + 40.0 * nx, y + 100.0 * uy + 40.0 * ny),
           (x + 100.0 * ux - 40.0 * nx, y + 100.0 * uy - 40.0 * ny)]
    px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
    return np.clip(_bilerp(dist_map, res_m, px, py) / 100.0, 0.0, 3.0)


def safe_dir(x: float, y: float, dist_map: np.ndarray, res_m: float) -> np.ndarray:
    """Unit vector (world frame) from (x, y) towards the nearest usable cell, scaled by min(d/100 m, 1);
    zero on usable ground. The negative gradient of the distance-to-safe map."""
    e = 2.0 * res_m
    gx = _bilerp(dist_map, res_m, np.array([x + e]), np.array([y]))[0] - _bilerp(dist_map, res_m, np.array([x - e]), np.array([y]))[0]
    gy = _bilerp(dist_map, res_m, np.array([x]), np.array([y + e]))[0] - _bilerp(dist_map, res_m, np.array([x]), np.array([y - e]))[0]
    nrm = float(np.hypot(gx, gy))
    if nrm < 1e-6:
        return np.zeros(2)
    d = float(_bilerp(dist_map, res_m, np.array([x]), np.array([y]))[0])
    return -np.array([gx, gy]) / nrm * min(d / 100.0, 1.0)


def aim_point(x, y, dist_map, res_m, offset_m=(0.0, 0.0)):
    """Rule-based aim point: the nearest usable cell (via the distance map's gradient) plus a policy offset (m)."""
    d = float(_bilerp(dist_map, res_m, np.array([x]), np.array([y]))[0])
    u = safe_dir(x, y, dist_map, res_m)
    n = float(np.hypot(u[0], u[1]))
    ux, uy = (u[0] / n, u[1] / n) if n > 1e-9 else (0.0, 0.0)
    return (x + ux * d + offset_m[0], y + uy * d + offset_m[1])
