"""Rock/boulder field: a position+size list, kept separate from the heightfield.

Adapters turn this into instanced meshes; core only produces the list and,
optionally, samples ground height/slope at each rock so an adapter can place
rocks flush with terrain without re-touching the heightfield.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RockField:
    x_m: np.ndarray
    y_m: np.ndarray
    diameter_m: np.ndarray


def sample_rock_field(
    size_m: float,
    density_scale: float,
    d_max_m: float,
    rng: np.random.Generator,
    reference_density_per_m2: float = 0.02,
    d_min_m: float = 0.05,
) -> RockField:
    """Exponential size distribution (most rocks small, few large), uniform position."""
    area = size_m * size_m
    n = max(0, int(area * reference_density_per_m2 * density_scale))

    x = rng.uniform(-size_m / 2, size_m / 2, n)
    y = rng.uniform(-size_m / 2, size_m / 2, n)

    mean_d = max(d_min_m, d_max_m / 8.0)
    d = rng.exponential(mean_d, n) + d_min_m
    d = np.clip(d, d_min_m, d_max_m)

    return RockField(x, y, d)


def sample_height_at(height: np.ndarray, res_m: float, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Bilinear-free nearest-sample lookup of heightfield value under each (x, y)."""
    n = height.shape[0]
    center = (n - 1) / 2
    j = np.clip(np.round(x_m / res_m + center).astype(int), 0, n - 1)
    i = np.clip(np.round(y_m / res_m + center).astype(int), 0, n - 1)
    return height[i, j]
