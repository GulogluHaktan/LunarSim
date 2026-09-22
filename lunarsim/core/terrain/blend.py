"""Frequency-separated coarse/fine blending with Gaussian ROI weighting.

Blending heights directly produces seams/steps at the boundary. Instead we
take low frequency from `coarse` and only the high-frequency detail from
`fine`; a Gaussian mask controls where and how much detail is injected --
this is the mechanism behind "make one area's detail level higher than the
rest" (plan section 5): put an ROI center there, the detail fades out with
distance per that ROI's own `sigma_m`, and everywhere with no ROI coverage
stays at the coarse source's detail level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


@dataclass
class RoiSpec:
    """One region-of-interest: a Gaussian detail bump centered at `center_px`
    (row, col, in fine-grid pixels), `sigma_m` wide, peaking at `weight`
    (1.0 = full fine detail at the center; use e.g. 0.5 for "somewhat more
    detail than the surroundings" rather than "full resolution here").
    """

    center_px: tuple[float, float]
    sigma_m: float
    weight: float = 1.0


def roi_weight(
    n: int,
    res_fine_m: float,
    centers_px: list[tuple[float, float]] | list[RoiSpec],
    sigma_m: float | None = None,
) -> np.ndarray:
    """Per-pixel detail weight in [0, 1]: the max over all ROIs of each
    ROI's own Gaussian falloff (so overlapping ROIs don't stack additively
    past 1.0, they just take whichever ROI's contribution is highest there).

    Two calling conventions:
    - `centers_px` a list of `(row, col)` pairs + a single shared `sigma_m`
      (the original uniform-detail-radius behavior, e.g. `random_16`).
    - `centers_px` a list of `RoiSpec` (per-ROI `sigma_m` and `weight`) --
      use this to give different areas different detail levels/radii in
      the same tile; `sigma_m` argument is ignored in this form.
    """
    if not centers_px:
        return np.ones((n, n))

    specs = (
        [RoiSpec(c, sigma_m, 1.0) for c in centers_px]
        if not isinstance(centers_px[0], RoiSpec)
        else centers_px
    )

    y, x = np.mgrid[0:n, 0:n]
    w = np.zeros((n, n))
    for spec in specs:
        cy, cx = spec.center_px
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) * res_fine_m**2
        w = np.maximum(w, spec.weight * np.exp(-r2 / (2 * spec.sigma_m**2)))
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
