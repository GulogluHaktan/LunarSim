import numpy as np
import pytest

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.craters import sample_crater_diameters
from lunarsim.core.terrain.curvature import MOON_RADIUS_M, add_curvature
from lunarsim.core.terrain.generate import generate_tile


def _small_cfg(mode: str, seed: int = 42) -> TerrainConfig:
    return TerrainConfig(
        mode=mode,
        size_m=64.0,
        res_m=1.0,
        coarse_res_m=4.0,
        seed=seed,
        coarse_source="procedural",  # these tests exercise the synthetic fBm path
        hills={"amplitude_m": 2.0, "wavelength_m": 30.0, "hurst": 0.75},
        craters={"count_scale": 1.0, "d_min_m": 1.0, "d_max_m": 20.0, "b": 2.5,
                 "depth_ratio": 0.1, "age": 0.3},
        rocks={"density_scale": 1.0, "d_max_m": 1.0},
        roi={"sigma_m": 20.0, "centers": None},
        curvature=True,
    )


@pytest.mark.parametrize("mode", ["coarse", "fine", "blend"])
def test_same_seed_same_output(mode):
    cfg = _small_cfg(mode)
    t1 = generate_tile(cfg)
    t2 = generate_tile(cfg)
    np.testing.assert_array_equal(t1.height, t2.height)
    np.testing.assert_array_equal(t1.craters.diameter_m, t2.craters.diameter_m)
    np.testing.assert_array_equal(t1.rocks.diameter_m, t2.rocks.diameter_m)


def test_different_seed_different_output():
    t1 = generate_tile(_small_cfg("fine", seed=1))
    t2 = generate_tile(_small_cfg("fine", seed=2))
    assert not np.array_equal(t1.height, t2.height)


def test_output_shape_matches_size_and_resolution():
    cfg = _small_cfg("fine")
    tile = generate_tile(cfg)
    n = int(cfg.size_m / cfg.res_m)
    assert tile.height.shape == (n, n)


def test_crater_diameter_power_law_slope():
    """Fit the cumulative N(>D) ~ D^-b slope from a large sample and check it recovers b."""
    rng = np.random.default_rng(0)
    b_true = 2.5
    d_min, d_max = 1.0, 1000.0
    d = sample_crater_diameters(200_000, d_min, d_max, b_true, rng)

    bins = np.logspace(np.log10(d_min), np.log10(d_max * 0.5), 25)
    counts_gt = np.array([(d > x).sum() for x in bins])
    valid = counts_gt > 50
    log_d = np.log10(bins[valid])
    log_n = np.log10(counts_gt[valid])
    slope, _ = np.polyfit(log_d, log_n, 1)

    assert -slope == pytest.approx(b_true, abs=0.15)


def test_curvature_drop_matches_reference_table():
    # plan section 3: curvature drop at given center distances
    res = 1.0
    n = 10001  # center-to-edge = 5000 m
    height = np.zeros((n, n))
    dropped = add_curvature(height, res, MOON_RADIUS_M)
    center = (n - 1) // 2

    checks = {250: 0.018, 500: 0.07, 1000: 0.29, 2000: 1.15, 5000: 7.2}
    for dist_m, expected_drop in checks.items():
        idx = center + int(dist_m / res)
        drop = -dropped[center, idx]  # add_curvature subtracts drop, so height = -drop
        assert drop == pytest.approx(expected_drop, rel=0.05)


def test_curvature_zero_at_center():
    height = np.zeros((21, 21))
    dropped = add_curvature(height, 1.0)
    center = 10
    assert dropped[center, center] == pytest.approx(0.0, abs=1e-9)


def test_coarse_mode_is_smoother_than_fine():
    coarse_tile = generate_tile(_small_cfg("coarse"))
    fine_tile = generate_tile(_small_cfg("fine"))
    coarse_roughness = np.std(np.diff(coarse_tile.height, axis=0))
    fine_roughness = np.std(np.diff(fine_tile.height, axis=0))
    assert coarse_roughness < fine_roughness


def _dem_cfg(mode: str, dem_path: str) -> TerrainConfig:
    return TerrainConfig(
        mode=mode,
        size_m=2000.0,
        res_m=20.0,
        coarse_res_m=50.0,
        seed=7,
        coarse_source="dem",
        dem_path=dem_path,
        site_lat_deg=0.0,
        site_lon_deg=0.0,
        hills={},
        craters={"count_scale": 0.5, "d_min_m": 5.0, "d_max_m": 100.0, "b": 2.5,
                 "depth_ratio": 0.1, "age": 0.3},
        rocks={"density_scale": 0.5, "d_max_m": 1.0},
        roi={"sigma_m": 300.0, "centers": None},
        curvature=True,
    )


def test_generate_tile_requires_dem_path_when_source_is_dem():
    cfg = _dem_cfg("fine", dem_path=None)
    cfg.dem_path = None
    with pytest.raises(ValueError, match="dem_path"):
        generate_tile(cfg)


@pytest.mark.parametrize("mode", ["coarse", "fine", "blend"])
def test_generate_tile_from_real_dem_source(synthetic_lunar_dem, mode):
    path, _ = synthetic_lunar_dem
    tile = generate_tile(_dem_cfg(mode, str(path)))
    n = int(2000.0 / 20.0)
    assert tile.height.shape == (n, n)
    assert tile.coarse_source == "dem"
    # DEM elevation is already sphere-relative -> curvature must NOT be double-applied
    assert np.abs(tile.height).max() < 2000.0


def test_dem_backed_tile_is_seed_reproducible(synthetic_lunar_dem):
    path, _ = synthetic_lunar_dem
    cfg = _dem_cfg("blend", str(path))
    t1 = generate_tile(cfg)
    t2 = generate_tile(cfg)
    np.testing.assert_array_equal(t1.height, t2.height)
