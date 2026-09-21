import numpy as np
import pytest

from lunarsim.core.dust.plume import (
    MOON_GRAVITY_M_S2,
    DustEvent,
    flat_ground_landing_time,
    landing_positions_on_heightmap,
    position_at,
    sample_launch_velocities,
)


def _event(n=1000, seed=1):
    return DustEvent(origin_m=np.array([0.0, 0.0, 0.0]), n_particles=n, seed=seed)


def test_launch_velocity_shape_and_speed_bounds():
    event = _event()
    v = sample_launch_velocities(event)
    assert v.shape == (event.n_particles, 3)
    speed = np.linalg.norm(v, axis=-1)
    assert speed.min() >= event.speed_range_m_s[0] - 1e-6
    assert speed.max() <= event.speed_range_m_s[1] + 1e-6


def test_low_angle_bias_mostly_horizontal():
    event = _event()
    v = sample_launch_velocities(event)
    elevation = np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))
    assert elevation.min() >= event.elevation_angle_range_deg[0] - 1e-6
    assert elevation.max() <= event.elevation_angle_range_deg[1] + 1e-6


def test_position_at_zero_time_is_origin():
    event = _event(n=5)
    v = sample_launch_velocities(event)
    pos = position_at(event, v, 0.0)
    np.testing.assert_allclose(pos, np.tile(event.origin_m, (5, 1)))


def test_single_particle_matches_hand_computed_parabola():
    event = DustEvent(origin_m=np.array([0.0, 0.0, 0.0]), n_particles=1)
    v = np.array([[3.0, 0.0, 4.0]])
    t = 1.0
    pos = position_at(event, v, t)
    expected_z = 4.0 * t - 0.5 * MOON_GRAVITY_M_S2 * t**2
    assert pos[0, 0] == pytest.approx(3.0)
    assert pos[0, 2] == pytest.approx(expected_z)


def test_flat_ground_landing_time_matches_analytic_range():
    event = DustEvent(origin_m=np.array([0.0, 0.0, 0.0]), n_particles=1)
    v = np.array([[5.0, 0.0, 5.0]])
    t_land = flat_ground_landing_time(event, v, ground_z_m=0.0)
    pos = position_at(event, v, t_land)
    assert pos[0, 2] == pytest.approx(0.0, abs=1e-6)
    # classic range formula: R = v^2 sin(2*theta) / g (theta=45 deg here since vx=vz)
    expected_range = (5.0**2 + 5.0**2) / MOON_GRAVITY_M_S2  # sin(90deg)=1
    assert pos[0, 0] == pytest.approx(expected_range, rel=1e-3)


def test_landing_positions_on_flat_heightmap_matches_flat_ground_formula():
    event = DustEvent(
        origin_m=np.array([0.0, 0.0, 0.0]), n_particles=200,
        speed_range_m_s=(0.5, 5.0), seed=1,
    )
    v = sample_launch_velocities(event)
    height = np.zeros((201, 201))  # half-extent 100 m, comfortably covers the ~8 m max range at 5 m/s
    res_m = 1.0

    land_pos, land_t = landing_positions_on_heightmap(event, v, height, res_m)
    t_analytic = flat_ground_landing_time(event, v, ground_z_m=0.0)

    in_bounds = (np.abs(land_pos[:, 0]) < 90) & (np.abs(land_pos[:, 1]) < 90)
    assert in_bounds.sum() > 100
    assert np.mean(np.abs(land_pos[in_bounds, 2])) < 0.5  # coarse time-stepping tolerance
    # landing time should roughly track the analytic flat-ground formula
    assert np.corrcoef(land_t[in_bounds], t_analytic[in_bounds])[0, 1] > 0.95


def test_lunar_ejecta_travels_farther_than_earth_would():
    """No-drag lunar ballistic range should exceed the naive Earth-gravity range
    for the same launch velocity -- the real physical reason lunar dust arcs
    are so much longer than terrestrial ones (besides the vacuum removing drag
    entirely, which this module already assumes)."""
    event = DustEvent(origin_m=np.array([0.0, 0.0, 0.0]), n_particles=1)
    v = np.array([[5.0, 0.0, 5.0]])
    t_moon = flat_ground_landing_time(event, v, ground_z_m=0.0, g=MOON_GRAVITY_M_S2)
    t_earth = flat_ground_landing_time(event, v, ground_z_m=0.0, g=9.81)
    range_moon = position_at(event, v, t_moon, g=MOON_GRAVITY_M_S2)[0, 0]
    range_earth = position_at(event, v, t_earth, g=9.81)[0, 0]
    assert range_moon > range_earth * 5  # g ratio is ~6x
