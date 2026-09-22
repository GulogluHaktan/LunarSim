"""Sun elevation/azimuth at a selenographic site, from a real JPL ephemeris.

Uses `skyfield.api.PlanetaryConstants` with the real NAIF Moon frame/orientation
kernels (`moon_080317.tf`, `pck00008.tpc`, `moon_pa_de421_1900-2050.bpc` --
the same kernel trio a peer lunar-robotics simulator, OmniLRS, uses for this
exact purpose) instead of a hand-derived IAU rotation matrix.

REAL BUG FIXED HERE: an earlier version of this module implemented the IAU
Moon mean-rotation elements (alpha0/delta0/W with the E1-E13 physical-
libration terms) by hand and built its own body-fixed rotation matrix +
local-ENU projection from scratch. That approach was already caught getting
a *different* hand-rolled rotation wrong once this session (see
`adapters/isaac/lighting.py`'s sun-direction fix) -- rather than keep
trusting a second independently-hand-derived rotation, this now goes
through skyfield's own tested `PlanetaryConstants.build_latlon_degrees` +
`.observe().apparent().altaz()`, which is exactly the mechanism skyfield
ships (and a real peer project uses) for "observer standing on a body other
than Earth, real orientation kernel, real light-time/aberration handled
along the way." `pressure_mbar=0` disables the atmospheric refraction
`altaz()` applies by default (built for Earth observers) -- the Moon has no
atmosphere to refract through.

The sun is ~1.5e8 km away vs. a Moon radius of ~1737 km, so it is treated as
a directional (parallel-ray) light source in the rest of lunarsim -- no
parallax correction needed for elevation/azimuth at the surface, and
skyfield's light-time/aberration corrections here are far more precision
than lunarsim actually needs, but cost nothing to get for free.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SOLAR_CONSTANT_W_M2 = 1361.0
SUN_ANGULAR_DIAMETER_DEG = 0.53

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"

# Standard NAIF generic kernels (downloaded once, cached under .cache/ like
# de421.bsp already was): Moon frame definitions, planetary pole/radii
# constants, and the DE421-consistent Moon physical-libration binary PCK.
_MOON_TF = "moon_080317.tf"
_PCK = "pck00008.tpc"
_MOON_PA = "moon_pa_de421_1900-2050.bpc"
_MOON_FRAME = "MOON_ME_DE421"  # mean-Earth/polar-axis frame -- the standard selenographic convention


@dataclass
class SunPosition:
    elevation_deg: float
    azimuth_deg: float  # 0 = selenographic north, clockwise (east positive), standard astronomical convention


class SunEphemeris:
    """Loads DE421 + the Moon orientation kernels once (downloading + caching
    them under .cache/ on first use)."""

    def __init__(self, cache_dir: str | Path | None = None):
        from skyfield.api import Loader, PlanetaryConstants

        cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        loader = Loader(str(cache_dir))
        self._ts = loader.timescale()
        self._eph = loader("de421.bsp")
        self._sun = self._eph["sun"]
        self._moon = self._eph["moon"]

        self._pc = PlanetaryConstants()
        self._pc.read_text(loader(_MOON_TF))
        self._pc.read_text(loader(_PCK))
        self._pc.read_binary(loader(_MOON_PA))
        self._frame = self._pc.build_frame_named(_MOON_FRAME)

    def sun_position(self, utc: datetime, lat_deg: float, lon_deg: float) -> SunPosition:
        """Sun elevation/azimuth at a selenographic (lat, lon) site, at the given UTC time.

        `lon_deg` follows the MOON_ME positive-east convention (0-360 or
        -180..180, either works).
        """
        t = self._ts.from_datetime(utc.astimezone(timezone.utc))
        observer = self._moon + self._pc.build_latlon_degrees(self._frame, lat_deg, lon_deg)
        apparent = observer.at(t).observe(self._sun).apparent()
        alt, az, _distance = apparent.altaz(pressure_mbar=0)  # no atmosphere on the Moon -> no refraction
        return SunPosition(elevation_deg=alt.degrees, azimuth_deg=az.degrees % 360.0)
