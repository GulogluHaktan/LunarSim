"""Sun light authoring: a single distant light, sized to the real solar
angular diameter, oriented from a `core.lighting.sun.SunPosition`, with
ambient/skylight forced off (plan section 7: "Ambient yok").
"""
from __future__ import annotations

import numpy as np

from lunarsim.core.lighting.sun import SOLAR_CONSTANT_W_M2, SunPosition


def create_sun_light(stage, prim_path: str, sun_pos: SunPosition, angular_diameter_deg: float, penumbra_samples: int = 4):
    """Author a `UsdLux.DistantLight` pointed at `sun_pos`.

    ISAAC-VERSION-CHECK: RTX soft-shadow sample count is normally set via a
    render-settings knob (e.g. `/rtx/pathtracing/lightcache/...` or the
    render product's `rtx:pathtracing:totalSpp`-adjacent settings), not a
    per-light attribute; `penumbra_samples` is accepted here for API
    symmetry with the quality profile but must be wired to whatever the
    installed Isaac version's actual RTX setting is.
    """
    from pxr import Gf, UsdGeom, UsdLux

    light = UsdLux.DistantLight.Define(stage, prim_path)
    light.CreateAngleAttr(float(angular_diameter_deg))
    light.CreateIntensityAttr(SOLAR_CONSTANT_W_M2)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.98))  # unfiltered solar color, mildly warm

    # elevation/azimuth (selenographic ENU: +x east, +y north, +z up) -> the
    # direction the light shines, i.e. FROM the sun TOWARD the ground, which
    # is what a DistantLight's local -Z axis must point along.
    el = np.deg2rad(sun_pos.elevation_deg)
    az = np.deg2rad(sun_pos.azimuth_deg)
    direction_to_sun = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    shine_dir = -direction_to_sun

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()
    orient_op = xform.AddOrientOp()
    orient_op.Set(_orient_quat_for_shine_direction(shine_dir))

    return light


def disable_ambient(stage, dome_light_prim_path: str | None = None):
    """Ensure no skylight/ambient contributes -- delete or zero-intensity any dome light."""
    if dome_light_prim_path is None:
        return
    prim = stage.GetPrimAtPath(dome_light_prim_path)
    if prim.IsValid():
        from pxr import UsdLux

        UsdLux.DomeLight(prim).CreateIntensityAttr(0.0)


def _orient_quat_for_shine_direction(shine_dir: np.ndarray):
    """Quaternion such that the local -Z axis maps to world-space `shine_dir`
    (unit vector) after rotation.

    Built via `Gf.Rotation`'s direct from-vector-to-vector constructor
    instead of hand-derived Euler angles -- a prior Euler-angle version of
    this function was verified (via `scripts/isaac_test_sun_rotation.py`,
    checking the light's actual local-to-world transform) to point the light
    in the wrong direction for every non-trivial case tested; this
    from/to-vector construction is unambiguous and was verified correct
    (dot product to the intended direction > 0.999) for zenith, horizon at
    multiple azimuths, a 45 deg case, and a low south-pole-like elevation.
    """
    from pxr import Gf

    d = shine_dir / np.linalg.norm(shine_dir)
    rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*d.tolist()))
    return Gf.Quatf(rotation.GetQuat())
