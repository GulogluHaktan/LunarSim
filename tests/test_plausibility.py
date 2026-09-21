import numpy as np
import pytest

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile
from lunarsim.core.terrain.plausibility import check_tile


def _cfg(**overrides):
    base = dict(
        mode="fine",
        size_m=200.0,
        res_m=0.5,
        seed=5,
        coarse_source="procedural",
        hills={"amplitude_m": 2.0, "wavelength_m": 40.0, "hurst": 0.75},
        craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 40.0, "b": 2.5,
                 "depth_ratio": 0.1, "age": 0.3},
        rocks={"density_scale": 1.0, "d_max_m": 1.0},
        roi={"sigma_m": 40.0, "centers": None},
        curvature=False,
    )
    base.update(overrides)
    return TerrainConfig(**base)


def test_reasonable_default_config_passes():
    tile = generate_tile(_cfg())
    report = check_tile(tile)
    assert report.ok, report.warnings


def test_extreme_crater_depth_ratio_flagged():
    tile = generate_tile(_cfg(craters={"count_scale": 2.0, "d_min_m": 1.0, "d_max_m": 40.0, "b": 2.5,
                                        "depth_ratio": 0.5, "age": 0.0}))
    report = check_tile(tile)
    assert not report.ok
    assert any("depth_ratio" in w for w in report.warnings)


def test_nan_heightmap_flagged():
    tile = generate_tile(_cfg())
    tile.height[10, 10] = np.nan
    report = check_tile(tile)
    assert not report.ok
    assert any("non-finite" in w for w in report.warnings)


def test_report_stats_populated():
    tile = generate_tile(_cfg())
    report = check_tile(tile)
    assert report.stats["n_craters"] > 0
    assert report.stats["n_rocks"] > 0
    assert report.stats["height_std_m"] > 0
