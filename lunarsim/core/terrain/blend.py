"""Frequency-separated coarse/fine blending with Gaussian ROI weighting.

Blending heights directly produces seams/steps at the boundary. Instead we
take low frequency from `coarse` and only the high-frequency detail from
`fine`; a Gaussian mask controls where and how much detail is injected.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def roi_weight(
    n: int, res_fine_m: float, centers_px: list[tuple[float, float]], sigma_m: float
) -> np.ndarray:
    """Sum of Gaussian bumps at each (row, col) center, clipped to [0, 1]."""
    if not centers_px:
        return np.ones((n, n))
    y, x = np.mgrid[0:n, 0:n]
    w = np.zeros((n, n))
    for cy, cx in centers_px:
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) * res_fine_m**2
        w = np.maximum(w, np.exp(-r2 / (2 * sigma_m**2)))
    return np.clip(w, 0.0, 1.0)


def blend(
    coarse: np.ndarray,
    fine: np.ndarray,
    res_fine_m: float,
    split_m: float = 20.0,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Coarse low-frequency + weighted fine high-frequency detail.

    `weight` is an (n, n) array in [0, 1]; None means full-strength fine detail
    everywhere (mode `fine`). Pass an all-zero weight for mode `coarse`.
    """
    n = fine.shape[0]
    c = zoom(coarse, n / coarse.shape[0], order=3)[:n, :n]

    s = split_m / res_fine_m
    detail = fine - gaussian_filter(fine, s)

    w = np.ones((n, n)) if weight is None else weight
    return c + w * detail
