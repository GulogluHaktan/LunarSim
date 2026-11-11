"""Radial fade profiles for the detail-injection weight (ported from the first-version terrain study).

Every fade maps a distance ``r`` (m) to a weight in [0, 1] and is forced to exactly 0 for ``r >= radius``
(hard truncation at the same radius, so the methods are compared at an equal transition budget).

    smoothstep           t = clamp(1 - r/R); t^2 (3 - 2t)                 compact support
    gaussian_shipped     exp(-1 (r/R)^2)  -> 0.368 left at R, then cut    legacy constant
    gaussian_calibrated  exp(-ln(1000) (r/R)^2)  -> 1e-3 at R             calibrated constant
    wendland             (1 - r/R)^4 (4 r/R + 1)  (Wendland C2)           compact support
    hybrid               gaussian near ``contact_radius``, smoothstep beyond, joined by a smoothstep switch
"""
from __future__ import annotations

import math

import numpy as np

K_SHIPPED = 1.0
K_CALIBRATED = math.log(1000.0)
FADE_METHODS = ("smoothstep", "gaussian_shipped", "gaussian_calibrated", "wendland", "hybrid")


def _ss(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep(r, R):
    return _ss(1.0 - r / R)


def _gauss(r, R, k):
    return np.exp(-k * (r / R) ** 2)


def _wendland(r, R):
    t = np.clip(r / R, 0.0, 1.0)
    return (1.0 - t) ** 4 * (4.0 * t + 1.0)


def make_fade(method: str, radius_m: float, contact_radius_m: float | None = None, switch_width_m: float | None = None):
    """Return ``fade(r)``: array of distances (m) -> weights, exactly 0 for r >= radius_m."""
    R = float(radius_m)
    if method == "smoothstep":
        f = lambda r: _smoothstep(r, R)
    elif method == "gaussian_shipped":
        f = lambda r: _gauss(r, R, K_SHIPPED)
    elif method == "gaussian_calibrated":
        f = lambda r: _gauss(r, R, K_CALIBRATED)
    elif method == "wendland":
        f = lambda r: _wendland(r, R)
    elif method == "hybrid":
        rc = float(contact_radius_m if contact_radius_m is not None else 0.1875 * R)  # 1.5 m at R = 8 m, as before
        w = float(switch_width_m if switch_width_m is not None else 0.0375 * R)        # 0.3 m at R = 8 m
        lo, hi = rc - w / 2.0, rc + w / 2.0

        def f(r):
            s = _ss((hi - r) / max(hi - lo, 1e-9))
            return s * _gauss(r, R, K_CALIBRATED) + (1.0 - s) * _smoothstep(r, R)
    else:
        raise ValueError(f"unknown fade method: {method!r} (choose from {FADE_METHODS})")
    return lambda r: np.where(np.asarray(r) >= R, 0.0, f(np.asarray(r, dtype=float)))
