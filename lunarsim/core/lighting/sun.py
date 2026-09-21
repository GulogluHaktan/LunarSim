"""Sun elevation/azimuth at a selenographic site, from a real JPL ephemeris.

Geometry: look up true Sun and Moon barycentric vectors at the given UTC
time (via skyfield + a cached DE421 kernel), take the Moon-to-Sun direction
in the inertial (J2000 equatorial) frame, then rotate it into the Moon's
body-fixed frame using the IAU Moon rotation model (mean pole + W, with the
primary E1-E13 physical-libration periodic terms — see "Report of the IAU
Working Group on Cartographic Coordinates and Rotational Elements: 2009").
That gives the subsolar direction in selenographic coordinates, which is
then projected into the local east-north-up frame of the requested site.

The sun is ~1.5e8 km away vs. a Moon radius of ~1737 km, so it is treated as
a directional (parallel-ray) light source — no parallax correction needed
for elevation/azimuth at the surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SOLAR_CONSTANT_W_M2 = 1361.0
SUN_ANGULAR_DIAMETER_DEG = 0.53

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"


@dataclass
class SunPosition:
    elevation_deg: float
    azimuth_deg: float  # 0 = selenographic north, clockwise (east positive)


def _iau_moon_rotation_deg(d_days: float) -> tuple[float, float, float]:
    """IAU Moon mean rotation elements (alpha0, delta0, W), all in degrees.

    `d_days` = days since J2000.0 TDB (JD_TDB - 2451545.0).
    """
    T = d_days / 36525.0

    E = np.deg2rad(np.array([
        125.045 - 0.0529921 * d_days,
        250.089 - 0.1059842 * d_days,
        260.008 + 13.0120009 * d_days,
        176.625 + 13.3407154 * d_days,
        357.529 + 0.9856003 * d_days,
        311.589 + 26.4057084 * d_days,
        134.963 + 13.0649930 * d_days,
        276.617 + 0.3287146 * d_days,
        34.226 + 1.7484877 * d_days,
        15.134 - 0.1589763 * d_days,
        119.743 + 0.0036096 * d_days,
        239.961 + 0.1643573 * d_days,
        25.053 + 12.9590088 * d_days,
    ]))
    s = np.sin(E)
    c = np.cos(E)

    alpha0 = (269.9949 + 0.0031 * T
              - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2]
              - 0.0172 * s[3] + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12])
    delta0 = (66.5392 + 0.0130 * T
              + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2]
              + 0.0068 * c[3] - 0.0029 * c[5] + 0.0009 * c[6]
              + 0.0008 * c[9] - 0.0009 * c[12])
    W = (38.3213 + 13.17635815 * d_days - 1.4e-12 * d_days**2
         + 3.5610 * s[0] + 0.1208 * s[1] - 0.0642 * s[2]
         + 0.0158 * s[3] + 0.0252 * s[4] - 0.0066 * s[5]
         - 0.0047 * s[6] - 0.0046 * s[7] + 0.0028 * s[8]
         + 0.0052 * s[9] + 0.0040 * s[10] + 0.0019 * s[11]
         - 0.0044 * s[12])
    return alpha0, delta0, W % 360.0


def _body_fixed_rotation_matrix(alpha0_deg: float, delta0_deg: float, w_deg: float) -> np.ndarray:
    """ICRF (J2000 equatorial) -> Moon body-fixed frame rotation matrix."""
    a0, d0, w = np.deg2rad([alpha0_deg, delta0_deg, w_deg])

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    return rz(w) @ rx(np.pi / 2 - d0) @ rz(np.pi / 2 + a0)


class SunEphemeris:
    """Loads a DE421 kernel once (downloading + caching it under .cache/ on first use)."""

    def __init__(self, cache_dir: str | Path | None = None):
        from skyfield.api import Loader

        cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        loader = Loader(str(cache_dir))
        self._ts = loader.timescale()
        self._eph = loader("de421.bsp")
        self._sun = self._eph["sun"]
        self._moon = self._eph["moon"]

    def sun_position(self, utc: datetime, lat_deg: float, lon_deg: float) -> SunPosition:
        """Sun elevation/azimuth at a selenographic (lat, lon) site, at the given UTC time.

        `lon_deg` follows the IAU_MOON positive-east convention (0-360 or
        -180..180, either works).
        """
        t = self._ts.from_datetime(utc.astimezone(tz=_utc_tz()))
        moon_ecliptic = self._moon.at(t)
        sun_from_moon = (self._sun.at(t) - moon_ecliptic).position.km
        sun_dir_icrf = sun_from_moon / np.linalg.norm(sun_from_moon)

        d_days = t.tdb - 2451545.0
        alpha0, delta0, w = _iau_moon_rotation_deg(d_days)
        rot = _body_fixed_rotation_matrix(alpha0, delta0, w)
        sun_dir_body = rot @ sun_dir_icrf

        return sun_position_from_body_frame_direction(sun_dir_body, lat_deg, lon_deg)


def _utc_tz():
    from datetime import timezone

    return timezone.utc


def sun_position_from_body_frame_direction(sun_dir_body: np.ndarray, lat_deg: float, lon_deg: float) -> SunPosition:
    """Project a Moon-body-fixed sun direction vector onto a site's local ENU frame."""
    lat, lon = np.deg2rad([lat_deg, lon_deg])

    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)

    s_up = float(np.dot(sun_dir_body, up))
    s_east = float(np.dot(sun_dir_body, east))
    s_north = float(np.dot(sun_dir_body, north))

    elevation = np.rad2deg(np.arcsin(np.clip(s_up, -1.0, 1.0)))
    azimuth = np.rad2deg(np.arctan2(s_east, s_north)) % 360.0
    return SunPosition(elevation_deg=elevation, azimuth_deg=azimuth)
