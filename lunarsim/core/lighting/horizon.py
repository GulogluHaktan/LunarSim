"""Heightmap-only horizon map: max blocking elevation angle per azimuth from a point.

Used as an independent ground truth to validate renderer shadows/penumbra
against (plan section 7: "heightmap'ten bağımsız horizon map hesapla, render
gölgeleriyle kıyasla"). Also used directly to decide whether the sun is
geometrically visible from a site (useful near the south pole, where the sun
sits at 0-3 deg elevation and long shadows are the norm).
"""
from __future__ import annotations

import numpy as np


def horizon_profile(
    height: np.ndarray,
    res_m: float,
    site_px: tuple[float, float],
    n_azimuths: int = 360,
    max_range_m: float | None = None,
    eye_height_m: float = 0.0,
) -> np.ndarray:
    """Max elevation angle (deg) blocking the sky at each azimuth, seen from `site_px`.

    `site_px` = (row, col) in heightmap pixel coordinates (can be fractional).
    Azimuth 0 = +row direction ("north" in array space), increasing toward +col ("east").
    Returns an (n_azimuths,) array of horizon elevation angles in degrees.
    """
    n = height.shape[0]
    row0, col0 = site_px
    z0 = _bilinear(height, row0, col0) + eye_height_m

    if max_range_m is None:
        max_range_m = n * res_m * 0.5

    n_steps = int(max_range_m / res_m)
    ranges_m = (np.arange(1, n_steps + 1)) * res_m

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    horizon = np.zeros(n_azimuths)

    for k, az in enumerate(azimuths):
        az_rad = np.deg2rad(az)
        d_row, d_col = np.cos(az_rad), np.sin(az_rad)

        rows = row0 + d_row * ranges_m / res_m
        cols = col0 + d_col * ranges_m / res_m

        valid = (rows >= 0) & (rows < n - 1) & (cols >= 0) & (cols < n - 1)
        if not valid.any():
            continue
        rows, cols, r = rows[valid], cols[valid], ranges_m[valid]

        z = _bilinear_vec(height, rows, cols)
        elev = np.rad2deg(np.arctan2(z - z0, r))
        horizon[k] = elev.max() if elev.size else -90.0

    return horizon


def sun_visible(horizon: np.ndarray, n_azimuths: int, sun_elevation_deg: float, sun_azimuth_deg: float) -> bool:
    """Whether the sun clears the horizon at its azimuth (nearest sampled bin)."""
    idx = int(round(sun_azimuth_deg / 360.0 * n_azimuths)) % n_azimuths
    return sun_elevation_deg > horizon[idx]


def _bilinear(height: np.ndarray, row: float, col: float) -> float:
    return float(_bilinear_vec(height, np.array([row]), np.array([col]))[0])


def _bilinear_vec(height: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    n = height.shape[0]
    r0 = np.clip(np.floor(rows).astype(int), 0, n - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, n - 2)
    fr, fc = rows - r0, cols - c0

    z00 = height[r0, c0]
    z10 = height[r0 + 1, c0]
    z01 = height[r0, c0 + 1]
    z11 = height[r0 + 1, c0 + 1]

    return (z00 * (1 - fr) * (1 - fc) + z10 * fr * (1 - fc)
            + z01 * (1 - fr) * fc + z11 * fr * fc)
