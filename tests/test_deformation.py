import numpy as np
import pytest

from lunarsim.core.terrain.deformation import BekkerSoilParams, bekker_sinkage_m, wheel_footprint_stamp


def test_sinkage_zero_for_zero_load():
    assert bekker_sinkage_m(0.0, 0.2) == 0.0


def test_sinkage_increases_with_load():
    s1 = bekker_sinkage_m(500.0, 0.2)
    s2 = bekker_sinkage_m(2000.0, 0.2)
    assert 0 < s1 < s2


def test_sinkage_decreases_with_wheel_width():
    narrow = bekker_sinkage_m(1000.0, 0.1)
    wide = bekker_sinkage_m(1000.0, 0.3)
    assert wide < narrow


def test_sinkage_is_reasonable_magnitude_for_a_rover_wheel():
    """A ~1500 kg lander/rover-class load spread over 4 wheels under lunar
    gravity should sink a few cm, not meters, for plausible soil params."""
    load_per_wheel_n = (1500.0 * 1.62) / 4.0
    s = bekker_sinkage_m(load_per_wheel_n, wheel_width_m=0.2)
    assert 0.001 < s < 0.5


def test_footprint_stamp_creates_depression_at_center():
    n = 201
    height = np.zeros((n, n))
    wheel_footprint_stamp(
        height, res_m=0.05, contact_x_m=0.0, contact_y_m=0.0, heading_rad=0.0,
        wheel_width_m=0.2, contact_length_m=0.15, sinkage_m=0.03,
    )
    center = n // 2
    assert height[center, center] < 0
    assert height[center, center] == pytest.approx(-0.03, abs=0.005)


def test_footprint_stamp_has_berm_outside_track_width():
    n = 201
    height = np.zeros((n, n))
    res_m = 0.02
    wheel_footprint_stamp(
        height, res_m=res_m, contact_x_m=0.0, contact_y_m=0.0, heading_rad=0.0,
        wheel_width_m=0.2, contact_length_m=0.15, sinkage_m=0.03, berm_ratio=0.2,
    )
    center = n // 2
    # just outside the track width (half_w=0.1m -> ~1.15*half_w berm center),
    # cross-track direction is the y-axis when heading=0 (along=x)
    berm_row = center + int(0.115 / res_m)
    assert height[berm_row, center] > 0  # raised berm


def test_footprint_stamp_respects_heading_direction():
    """A footprint stamped along the x-axis (heading=0) vs. rotated 90 deg
    (heading=pi/2) should produce different (transposed-like) depression
    shapes, not an isotropic blob."""
    n = 201
    res_m = 0.03
    h0 = wheel_footprint_stamp(
        np.zeros((n, n)), res_m, 0.0, 0.0, 0.0, wheel_width_m=0.15, contact_length_m=0.4, sinkage_m=0.02,
    )
    h90 = wheel_footprint_stamp(
        np.zeros((n, n)), res_m, 0.0, 0.0, np.pi / 2, wheel_width_m=0.15, contact_length_m=0.4, sinkage_m=0.02,
    )
    assert not np.allclose(h0, h90)


def test_footprint_stamp_no_op_for_zero_sinkage():
    height = np.zeros((51, 51))
    result = wheel_footprint_stamp(height, 0.1, 0.0, 0.0, 0.0, 0.2, 0.15, sinkage_m=0.0)
    assert np.all(result == 0.0)


def test_footprint_stamp_out_of_bounds_is_noop_not_crash():
    height = np.zeros((21, 21))
    result = wheel_footprint_stamp(height, 0.1, 1000.0, 1000.0, 0.0, 0.2, 0.15, sinkage_m=0.05)
    assert np.all(result == 0.0)
