"""Planetary curvature correction (drops the tile's height as a function of range)."""
from __future__ import annotations

import numpy as np

MOON_RADIUS_M = 1737.4e3


def add_curvature(height: np.ndarray, res_m: float, radius_m: float = MOON_RADIUS_M) -> np.ndarray:
    """Subtract R - sqrt(R^2 - r^2) from height, r measured from tile center."""
    n = height.shape[0]
    ax = (np.arange(n) - (n - 1) / 2) * res_m
    x, y = np.meshgrid(ax, ax, indexing="ij")
    drop = radius_m - np.sqrt(radius_m**2 - (x**2 + y**2))
    return height - drop
