"""Needs network access (downloads/caches real NAIF kernels on first run,
like tests/test_dem.py implicitly relies on skyfield's de421.bsp elsewhere).
Skipped automatically if the kernels can't be fetched."""
from datetime import datetime, timezone

import pytest

from lunarsim.core.lighting.sun import SunEphemeris


@pytest.fixture(scope="module")
def eph():
    try:
        return SunEphemeris()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load ephemeris/kernels (no network?): {e}")


def test_sun_position_returns_finite_values(eph):
    pos = eph.sun_position(datetime(2026, 9, 21, 12, tzinfo=timezone.utc), lat_deg=0.0, lon_deg=0.0)
    assert -90.0 <= pos.elevation_deg <= 90.0
    assert 0.0 <= pos.azimuth_deg < 360.0


def test_south_pole_sun_is_low(eph):
    """The whole point of the south-pole use case (plan section 7): sun
    elevation there should be within a few degrees of the horizon."""
    pos = eph.sun_position(datetime(2026, 9, 21, 12, tzinfo=timezone.utc), lat_deg=-89.0, lon_deg=0.0)
    assert 0.0 <= pos.elevation_deg <= 5.0


def test_different_times_give_different_sun_positions(eph):
    pos1 = eph.sun_position(datetime(2026, 9, 21, 12, tzinfo=timezone.utc), lat_deg=10.0, lon_deg=10.0)
    pos2 = eph.sun_position(datetime(2026, 10, 21, 12, tzinfo=timezone.utc), lat_deg=10.0, lon_deg=10.0)
    assert abs(pos1.elevation_deg - pos2.elevation_deg) > 0.1 or abs(pos1.azimuth_deg - pos2.azimuth_deg) > 0.1


def test_near_horizon_call_succeeds_with_no_refraction_path(eph):
    """Regression guard on the pressure_mbar=0 (no-atmosphere) altaz() call
    itself, at a site/time expected to be near the horizon."""
    pos = eph.sun_position(datetime(2026, 9, 21, 12, tzinfo=timezone.utc), lat_deg=-89.9, lon_deg=180.0)
    assert -90.0 <= pos.elevation_deg <= 90.0
