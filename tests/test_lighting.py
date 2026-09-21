import numpy as np
import pytest

from lunarsim.core.lighting.horizon import horizon_profile, sun_visible
from lunarsim.core.lighting.materials import (
    albedo_lambertian,
    hapke_approx,
    lommel_seeliger,
    reflectance,
    surface_normals_from_heightmap,
)


def test_flat_plane_normals_point_up():
    height = np.zeros((20, 20))
    normals = surface_normals_from_heightmap(height, res_m=1.0)
    np.testing.assert_allclose(normals[10, 10], [0, 0, 1], atol=1e-9)


def test_reflectance_zero_at_grazing_incidence():
    mu_i = np.array([0.0])
    mu_e = np.array([1.0])
    assert albedo_lambertian(mu_i, 0.1)[0] == pytest.approx(0.0)
    assert lommel_seeliger(mu_i, mu_e, 0.1)[0] == pytest.approx(0.0)


def test_reflectance_positive_at_normal_incidence():
    mu_i = np.array([1.0])
    mu_e = np.array([1.0])
    assert albedo_lambertian(mu_i, 0.1)[0] > 0
    assert lommel_seeliger(mu_i, mu_e, 0.1)[0] > 0
    r = hapke_approx(mu_i, mu_e, np.array([0.5]), w=0.1)
    assert r[0] > 0


def test_hapke_opposition_surge_brightens_near_zero_phase():
    mu_i = np.array([1.0])
    mu_e = np.array([1.0])
    r_no_surge = hapke_approx(mu_i, mu_e, np.array([0.001]), w=0.1, opposition_surge=False)
    r_surge = hapke_approx(mu_i, mu_e, np.array([0.001]), w=0.1, opposition_surge=True)
    assert r_surge[0] > r_no_surge[0]


def test_reflectance_dispatch_matches_direct_call():
    mu_i, mu_e = np.array([0.8]), np.array([0.9])
    assert reflectance("albedo", mu_i, mu_e, albedo=0.12)[0] == albedo_lambertian(mu_i, 0.12)[0]
    assert reflectance("lommel_seeliger", mu_i, mu_e, albedo=0.12)[0] == lommel_seeliger(mu_i, mu_e, 0.12)[0]

    with pytest.raises(ValueError):
        reflectance("hapke", mu_i, mu_e, albedo=0.12)  # missing g_rad


def test_horizon_flat_ground_is_zero_everywhere():
    height = np.zeros((401, 401))
    h = horizon_profile(height, res_m=1.0, site_px=(200, 200), n_azimuths=72)
    np.testing.assert_allclose(h, 0.0, atol=1e-6)


def test_horizon_wall_blocks_low_sun():
    height = np.zeros((401, 401))
    height[:, 300:] = 50.0  # a wall to the "east" (+col) side
    h = horizon_profile(height, res_m=1.0, site_px=(200, 200), n_azimuths=72)
    east_idx = 72 // 4  # azimuth 90 deg ~ +col direction in this array convention
    assert h[east_idx] > 10.0  # wall well above the horizontal


def test_sun_visible_respects_horizon():
    horizon = np.zeros(36)
    horizon[9] = 10.0  # a 10-deg obstruction at azimuth 90
    assert not sun_visible(horizon, 36, sun_elevation_deg=5.0, sun_azimuth_deg=90.0)
    assert sun_visible(horizon, 36, sun_elevation_deg=15.0, sun_azimuth_deg=90.0)
