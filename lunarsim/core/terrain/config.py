"""YAML terrain config loading + per-tile parameter sampling.

Every parameter is either a fixed scalar or a [min, max] range; ranges are
sampled once per tile from the tile's seeded RNG so a given seed always
reproduces the same terrain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


def _sample(value, rng: np.random.Generator):
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(v, (int, float)) for v in value
    ):
        lo, hi = value
        return float(rng.uniform(lo, hi))
    return value


def _sample_dict(d: dict, rng: np.random.Generator) -> dict:
    return {k: _sample(v, rng) for k, v in d.items()}


@dataclass
class TerrainConfig:
    mode: str = "blend"  # coarse | fine | blend
    size_m: float = 1000.0
    res_m: float = 0.5
    coarse_res_m: float = 4.0
    seed: int = 0

    # Coarse (large-scale) elevation source. "dem" reads a real LOLA/LRO raster
    # (see core/terrain/dem.py) and is the default for anything meant to
    # represent an actual place on the Moon. "procedural" (fBm hills) is only
    # for synthetic training tiles where the specific region doesn't matter —
    # it must never be treated as ground truth for a real site.
    coarse_source: str = "dem"  # dem | procedural
    dem_path: str | None = None
    site_lat_deg: float = 0.0
    site_lon_deg: float = 0.0

    hills: dict = field(default_factory=dict)
    craters: dict = field(default_factory=dict)
    rocks: dict = field(default_factory=dict)
    roi: dict = field(default_factory=dict)
    curvature: bool = True

    split_m: float = 20.0

    @classmethod
    def from_yaml(cls, path: str) -> "TerrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        t = raw["terrain"]
        return cls(
            mode=t.get("mode", "blend"),
            size_m=t["size_m"],
            res_m=t["res_m"],
            coarse_res_m=t.get("coarse_res_m", 4.0),
            seed=t.get("seed", 0),
            coarse_source=t.get("coarse_source", "dem"),
            dem_path=t.get("dem_path"),
            site_lat_deg=t.get("site_lat_deg", 0.0),
            site_lon_deg=t.get("site_lon_deg", 0.0),
            hills=t.get("hills", {}),
            craters=t.get("craters", {}),
            rocks=t.get("rocks", {}),
            roi=t.get("roi", {}),
            curvature=t.get("curvature", True),
            split_m=t.get("split_m", 20.0),
        )


@dataclass
class SampledParams:
    """Concrete (non-range) parameters resolved for one tile from a TerrainConfig."""

    hills: dict
    craters: dict
    rocks: dict
    roi: dict


def sample_params(cfg: TerrainConfig, rng: np.random.Generator) -> SampledParams:
    hills = _sample_dict(cfg.hills, rng)
    craters = dict(cfg.craters)
    craters["count_scale"] = _sample(craters.get("count_scale", 1.0), rng)
    craters["b"] = _sample(craters.get("b", 2.5), rng)
    craters["d_min_m"] = _sample(craters.get("d_min_m", 0.5), rng)
    craters["d_max_m"] = _sample(craters.get("d_max_m", 120.0), rng)
    depth_ratio = craters.get("depth_ratio", [0.05, 0.2])
    craters["depth_ratio_range"] = tuple(depth_ratio) if isinstance(depth_ratio, (list, tuple)) else (depth_ratio, depth_ratio)
    age = craters.get("age", [0.0, 1.0])
    craters["age_range"] = tuple(age) if isinstance(age, (list, tuple)) else (age, age)

    rocks = dict(cfg.rocks)
    rocks["density_scale"] = _sample(rocks.get("density_scale", 1.0), rng)
    rocks["d_max_m"] = _sample(rocks.get("d_max_m", 2.0), rng)

    roi = dict(cfg.roi)
    roi["sigma_m"] = _sample(roi.get("sigma_m", 200.0), rng)

    return SampledParams(hills=hills, craters=craters, rocks=rocks, roi=roi)
