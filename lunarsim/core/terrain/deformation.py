"""Wheel-terrain deformation: a real (if simplified) Bekker pressure-sinkage
estimate stamped into the heightmap as the wheel moves, the same way
`craters.py` stamps a crater profile -- no soft-body/FEM solve, just a
direct height-field edit at the contact patch each step.

This mirrors the approach a peer lunar-robotics simulator (OmniLRS) takes
(`src/terrain_management/deformation_engine.py`: footprint stamped directly
into the heightmap) rather than a full terramechanics FEM solve, which
neither project has shipped as complete -- OmniLRS's own
`terramechanics_solver.py` is marked WIP in their source.

Bekker sinkage here is the classic single-wheel static estimate
    z = (W / (b * (k_c/b + k_phi)))^(1/n)
(mean ground pressure p = W / (b*l) approximated with contact length l ~ b;
the more accurate l ~ 2*sqrt(r*z) iterative form is NOT implemented -- this
gives the right order of magnitude and load/width sensitivity for a visual/
terrain-effect deformation, not a traction-accurate solve). See Bekker,
M.G. (1969) "Introduction to Terrain-Vehicle Systems" for the underlying
pressure-sinkage relation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BekkerSoilParams:
    """Pressure-sinkage soil parameters. Defaults are commonly-cited
    lunar-regolith-like values (order-of-magnitude from Apollo LRV-era
    terramechanics literature) -- treat as a reasonable starting point to
    tune per use case, not a precise measured constant."""

    k_c: float = 1400.0  # N/m^(n+1), cohesive modulus
    k_phi: float = 820_000.0  # N/m^(n+2), frictional modulus
    n: float = 1.0  # sinkage exponent


def bekker_sinkage_m(load_n: float, wheel_width_m: float, params: BekkerSoilParams = BekkerSoilParams()) -> float:
    """Static sinkage estimate (m) for a wheel of `wheel_width_m` carrying
    `load_n` newtons, per the simplified Bekker relation (see module docstring).
    """
    if load_n <= 0:
        return 0.0
    k = params.k_c / wheel_width_m + params.k_phi
    return (load_n / (wheel_width_m * k)) ** (1.0 / params.n)


def wheel_footprint_stamp(
    height: np.ndarray,
    res_m: float,
    contact_x_m: float,
    contact_y_m: float,
    heading_rad: float,
    wheel_width_m: float,
    contact_length_m: float,
    sinkage_m: float,
    berm_ratio: float = 0.15,
) -> np.ndarray:
    """Stamp one wheel-contact deformation into `height` (in-place edit,
    also returned for chaining) -- an elongated depression along the wheel's
    rolling direction (`heading_rad`), `wheel_width_m` wide,
    `contact_length_m` long, `sinkage_m` deep at the center, with small
    raised berms (`berm_ratio` of `sinkage_m`) just outside the track edges
    -- the displaced-soil ridge real wheel tracks show.
    """
    if sinkage_m <= 0:
        return height

    n = height.shape[0]
    ax = (np.arange(n) - (n - 1) / 2) * res_m
    pad = max(wheel_width_m, contact_length_m) * 1.5
    i_lo = np.searchsorted(ax, contact_y_m - pad)
    i_hi = np.searchsorted(ax, contact_y_m + pad)
    j_lo = np.searchsorted(ax, contact_x_m - pad)
    j_hi = np.searchsorted(ax, contact_x_m + pad)
    if i_hi <= i_lo or j_hi <= j_lo:
        return height

    yy = ax[i_lo:i_hi][:, None] - contact_y_m
    xx = ax[j_lo:j_hi][None, :] - contact_x_m

    # rotate into the wheel's own (along-track, cross-track) frame
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    along = xx * c + yy * s
    cross = -xx * s + yy * c

    half_w = wheel_width_m / 2.0
    half_l = contact_length_m / 2.0

    # smooth (cosine-tapered) depression footprint, zero outside the contact patch
    within_l = np.abs(along) <= half_l
    within_w = np.abs(cross) <= half_w
    footprint = np.zeros_like(along)
    footprint[within_l & within_w] = 1.0

    cross_taper = np.clip(1.0 - np.abs(cross) / half_w, 0.0, 1.0) ** 0.5
    along_taper = np.where(within_l, 1.0, np.clip(1.0 - (np.abs(along) - half_l) / (0.3 * half_l + 1e-9), 0.0, 1.0))
    depression = -sinkage_m * cross_taper * along_taper * (within_w | (np.abs(cross) <= half_w * 1.3))

    # berm: a small raised ridge just outside the track width, tapering to zero further out
    berm_center = half_w * 1.15
    berm_width = half_w * 0.6
    berm_dist = np.abs(np.abs(cross) - berm_center)
    berm = berm_ratio * sinkage_m * np.clip(1.0 - berm_dist / berm_width, 0.0, 1.0) * along_taper

    height[i_lo:i_hi, j_lo:j_hi] += depression + berm
    return height
