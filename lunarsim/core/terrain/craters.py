"""Crater fields: power-law size distribution, parabolic-bowl-plus-rim profile."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_crater_diameters(
    n: int, d_min: float, d_max: float, b: float, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF sampling of a cumulative power-law N(>D) ~ D^-b."""
    u = rng.random(n)
    a, c = d_min**-b, d_max**-b
    return (a - u * (a - c)) ** (-1.0 / b)


def _crater_count(area_m2: float, d_min: float, d_max: float, b: float, count_scale: float) -> int:
    """Rough target count from a reference density so bigger tiles get proportionally more craters."""
    reference_density_per_m2 = 5e-4  # tuned for d_min ~ 0.5 m fields; scaled by count_scale
    return max(0, int(area_m2 * reference_density_per_m2 * count_scale))


@dataclass
class CraterField:
    x_m: np.ndarray  # crater center x, meters, relative to tile center
    y_m: np.ndarray
    diameter_m: np.ndarray
    depth_ratio: np.ndarray
    age: np.ndarray


def sample_crater_field(
    size_m: float,
    d_min_m: float,
    d_max_m: float,
    b: float,
    count_scale: float,
    depth_ratio_range: tuple[float, float],
    age_range: tuple[float, float],
    rng: np.random.Generator,
) -> CraterField:
    n = _crater_count(size_m * size_m, d_min_m, d_max_m, b, count_scale)
    x = rng.uniform(-size_m / 2, size_m / 2, n)
    y = rng.uniform(-size_m / 2, size_m / 2, n)
    d = sample_crater_diameters(n, d_min_m, d_max_m, b, rng)
    depth_ratio = rng.uniform(*depth_ratio_range, n)
    age = rng.uniform(*age_range, n)
    return CraterField(x, y, d, depth_ratio, age)


def _single_crater_profile(r_norm: np.ndarray) -> np.ndarray:
    """Normalized radial profile (r_norm = r / (D/2)): parabolic bowl inside rim, raised rim, decaying outside.

    Returns depth-normalized height offset in [-1 (bowl floor), rim_height (>0) at rim, decaying to 0 far out].
    """
    rim_height = 0.04  # fraction of depth, small raised rim typical of simple craters
    profile = np.empty_like(r_norm)

    inside = r_norm <= 1.0
    profile[inside] = r_norm[inside] ** 2 - 1.0  # parabolic bowl, 0 at rim, -1 at center

    outside = ~inside
    # rim taper: exponential falloff beyond the crater radius
    falloff = np.exp(-(r_norm[outside] - 1.0) * 3.0)
    profile[outside] = rim_height * falloff

    return profile


def rasterize_craters(
    height: np.ndarray, res_m: float, field: CraterField, age_blur_max_sigma_px: float = 2.5
) -> np.ndarray:
    """Stamp each crater's profile into the heightfield (additive, in-place copy returned).

    `age` (0=fresh,1=old) increases rim/floor blur, softening the crater over time.
    """
    from scipy.ndimage import gaussian_filter

    n = height.shape[0]
    ax = (np.arange(n) - (n - 1) / 2) * res_m
    out = height.copy()

    for x0, y0, d, depth_ratio, age in zip(
        field.x_m, field.y_m, field.diameter_m, field.depth_ratio, field.age
    ):
        radius = d / 2.0
        depth = d * depth_ratio
        pad = radius * 1.5
        i_lo = np.searchsorted(ax, y0 - pad)
        i_hi = np.searchsorted(ax, y0 + pad)
        j_lo = np.searchsorted(ax, x0 - pad)
        j_hi = np.searchsorted(ax, x0 + pad)
        if i_hi <= i_lo or j_hi <= j_lo:
            continue

        yy = ax[i_lo:i_hi][:, None]
        xx = ax[j_lo:j_hi][None, :]
        r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        r_norm = r / radius
        mask = r_norm <= 1.5
        if not mask.any():
            continue

        patch = _single_crater_profile(r_norm) * depth

        sigma_px = age * age_blur_max_sigma_px
        if sigma_px > 0.3:
            patch = gaussian_filter(patch, sigma_px)

        out[i_lo:i_hi, j_lo:j_hi] += patch

    return out
