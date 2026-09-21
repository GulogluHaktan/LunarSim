import pytest

from lunarsim.core.terrain.curvature import MOON_RADIUS_M
from lunarsim.core.terrain.dem import DemSource


def test_dem_source_is_geographic(synthetic_lunar_dem):
    path, _ = synthetic_lunar_dem
    src = DemSource(str(path))
    assert src.is_geographic


def test_dem_crop_matches_expected_center_value(synthetic_lunar_dem):
    path, elevation = synthetic_lunar_dem
    src = DemSource(str(path))

    tile = src.read_tile(center_lat_deg=0.0, center_lon_deg=0.0, size_m=50_000.0, res_m=250.0, radius_m=MOON_RADIUS_M)
    n = tile.shape[0]
    center_val = tile[n // 2, n // 2]

    # at lon=0, lat=0 the synthetic plane elevation is ~0
    assert center_val == pytest.approx(0.0, abs=15.0)


def test_dem_crop_shape_matches_size_and_res(synthetic_lunar_dem):
    path, _ = synthetic_lunar_dem
    src = DemSource(str(path))
    tile = src.read_tile(center_lat_deg=10.0, center_lon_deg=-5.0, size_m=10_000.0, res_m=100.0)
    assert tile.shape == (100, 100)


def test_dem_crop_gradient_direction(synthetic_lunar_dem):
    """East side of the tile (positive lon direction) should read higher than west, matching
    the synthetic plane's +x (lon) gradient."""
    path, _ = synthetic_lunar_dem
    src = DemSource(str(path))
    tile = src.read_tile(center_lat_deg=0.0, center_lon_deg=0.0, size_m=100_000.0, res_m=500.0)
    west_mean = tile[:, :10].mean()
    east_mean = tile[:, -10:].mean()
    assert east_mean > west_mean
