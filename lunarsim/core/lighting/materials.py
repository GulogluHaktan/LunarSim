"""Regolith reflectance models, from simplest to most physically accurate
(plan section 7: "Basit albedo -> Lommel-Seeliger -> Hapke yaklaşığı").

All functions take angles in radians and a single-scattering albedo `w`
(not the same as normal (geometric) albedo; typical lunar regolith
w ~ 0.1-0.3 depending on wavelength/maturity) and return a dimensionless
reflectance factor (radiance factor, I/F) suitable for multiplying by
incident solar irradiance to get outgoing radiance.

- i = incidence angle (Sun to surface normal)
- e = emission angle (surface normal to sensor/camera)
- g = phase angle (Sun-surface-sensor angle)
"""
from __future__ import annotations

import numpy as np


def albedo_lambertian(mu_i: np.ndarray, albedo: float) -> np.ndarray:
    """Flat Lambertian reflectance: I/F = albedo * cos(i). Cheapest, least correct
    (real regolith is markedly non-Lambertian, especially near full phase)."""
    return albedo * np.clip(mu_i, 0.0, None)


def lommel_seeliger(mu_i: np.ndarray, mu_e: np.ndarray, albedo: float) -> np.ndarray:
    """Single-scattering Lommel-Seeliger law: good first-order fit for dark,
    porous regolith; captures the basic limb-darkening/brightening behavior
    that Lambertian misses.
    """
    mu_i = np.clip(mu_i, 0.0, None)
    mu_e = np.clip(mu_e, 1e-6, None)
    return albedo * mu_i / (mu_i + mu_e)


def _hapke_h_function(x: np.ndarray, w: float) -> np.ndarray:
    """Hapke's H-function approximation (Hapke 2002), isotropic multiple scattering."""
    gamma = np.sqrt(1.0 - w)
    x = np.clip(x, 1e-6, None)
    return 1.0 / (1.0 - w * x * (gamma + (1.0 - 2.0 * gamma * x) / 2.0 * np.log1p(1.0 / x)))


def hapke_approx(
    mu_i: np.ndarray,
    mu_e: np.ndarray,
    g_rad: np.ndarray,
    w: float,
    b0: float = 1.0,
    h: float = 0.06,
    opposition_surge: bool = False,
) -> np.ndarray:
    """Simplified Hapke bidirectional reflectance (isotropic single-particle
    phase function, single-term Henyey-Greenstein omitted for simplicity —
    "yaklaşık" per the plan). Adds the shadow-hiding opposition surge term
    when requested (b0, h are its amplitude/width parameters).

    Reference: Hapke, B. (2012), "Theory of Reflectance and Emittance
    Spectroscopy", 2nd ed. Returns the bidirectional reflectance r; the
    radiance factor (I/F) a renderer wants is `pi * r / mu_i`.
    """
    mu_i = np.clip(mu_i, 0.0, None)
    mu_e = np.clip(mu_e, 1e-6, None)

    p_g = 1.0  # isotropic single-particle phase function approximation
    surge = 1.0
    if opposition_surge:
        surge = 1.0 + b0 / (1.0 + np.tan(g_rad / 2.0) / h)

    h_i = _hapke_h_function(mu_i, w)
    h_e = _hapke_h_function(mu_e, w)

    return (w / (4.0 * np.pi)) * (mu_i / (mu_i + mu_e)) * (p_g * surge + h_i * h_e - 1.0)


def reflectance(
    brdf: str,
    mu_i: np.ndarray,
    mu_e: np.ndarray,
    g_rad: np.ndarray | None = None,
    albedo: float = 0.10,
    opposition_surge: bool = False,
) -> np.ndarray:
    """Dispatch by name, matching the `material.brdf` quality-profile key
    (albedo | lommel_seeliger | hapke_approx | hapke)."""
    if brdf == "albedo":
        return albedo_lambertian(mu_i, albedo)
    if brdf == "lommel_seeliger":
        return lommel_seeliger(mu_i, mu_e, albedo)
    if brdf in ("hapke_approx", "hapke"):
        if g_rad is None:
            raise ValueError("hapke BRDFs require the phase angle g_rad")
        return hapke_approx(mu_i, mu_e, g_rad, w=albedo, opposition_surge=opposition_surge)
    raise ValueError(f"unknown brdf: {brdf!r}")


def cos_incidence_angle(surface_normal: np.ndarray, sun_dir: np.ndarray) -> np.ndarray:
    """cos(i) from a (..., 3) unit surface normal field and a (3,) unit sun direction."""
    return np.clip(np.tensordot(surface_normal, sun_dir, axes=([-1], [0])), 0.0, 1.0)


def surface_normals_from_heightmap(height: np.ndarray, res_m: float) -> np.ndarray:
    """Central-difference unit surface normals, shape (n, n, 3)."""
    dzdy, dzdx = np.gradient(height, res_m)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal
