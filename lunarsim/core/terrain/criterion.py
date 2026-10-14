"""Footpad-flatness term of `landed_safely`, in one place so the envs, the hazard map and the capture scripts agree.

Two definitions, picked by ``LUNARSIM_LANDING_CRITERION``:

``plane`` (default)
    What the lunar-landing literature checks: terrain *slope* under the pads (<= ``LUNARSIM_MAX_SLOPE_DEG``,
    default 10 deg, Nova-C-style) and *roughness* = pad-height residual after removing the best-fit plane
    (<= 0.16 m). Both are folded into one number so every consumer keeps its ``leg_diff <= 0.16`` test:
    ``eff = max(residual, 0.16 * slope / max_slope)`` -- continuous, and it crosses 0.16 exactly where either
    limit is crossed.
``spread``
    Legacy: raw max-min of the four pad heights in the world frame. Over a 9.4 m span that is a ~1 deg slope
    limit, inconsistent with the 15 deg tilt limit; it passes 1.6% of Apollo 16 terrain (see
    ``ObsidianVault/01-Projects/Isaac-Sim/lunarsim-apollo16-ceiling``).
"""
from __future__ import annotations

import os

import numpy as np

LEG_LIMIT_M = 0.16


def criterion_mode() -> str:
    m = os.environ.get("LUNARSIM_LANDING_CRITERION", "plane").strip().lower()
    return m if m in ("plane", "spread") else "plane"


def max_slope_deg() -> float:
    return float(os.environ.get("LUNARSIM_MAX_SLOPE_DEG", "10.0"))


def criterion_key() -> tuple:
    """Hashable identity of the active definition (cache key for per-tile hazard maps)."""
    return (criterion_mode(), max_slope_deg() if criterion_mode() == "plane" else 0.0)


def effective_leg_diff(z, radius_m: float):
    """z: pad heights, leading axis of length 4 at azimuths 0/90/180/270 deg. -> effective leg diff (m)."""
    z = np.asarray(z, dtype=np.float64)
    spread = z.max(0) - z.min(0)
    if criterion_mode() == "spread":
        return spread
    gx = (z[0] - z[2]) / (2.0 * radius_m)
    gy = (z[1] - z[3]) / (2.0 * radius_m)
    resid = np.abs(z[0] + z[2] - z[1] - z[3]) / 2.0          # max-min of the 4 plane-fit residuals
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return np.maximum(resid, LEG_LIMIT_M * slope / max(max_slope_deg(), 1e-6))
