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

    # elevation/azimuth (selenographic ENU) -> a look-at rotation for a light
    # whose local -Z points toward the ground by USD convention.
    el = np.deg2rad(sun_pos.elevation_deg)
    az = np.deg2rad(sun_pos.azimuth_deg)
    direction_to_sun = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()
    rotate_op = xform.AddRotateXYZOp()
    rotate_op.Set(_rotation_from_direction(-direction_to_sun))

    return light


def disable_ambient(stage, dome_light_prim_path: str | None = None):
    """Ensure no skylight/ambient contributes -- delete or zero-intensity any dome light."""
    if dome_light_prim_path is None:
        return
    prim = stage.GetPrimAtPath(dome_light_prim_path)
    if prim.IsValid():
        from pxr import UsdLux

        UsdLux.DomeLight(prim).CreateIntensityAttr(0.0)


def _rotation_from_direction(direction: np.ndarray):
    """Euler XYZ (degrees) such that local -Z aligns with `direction` (unit vector)."""
    from pxr import Gf

    d = direction / np.linalg.norm(direction)
    pitch = np.rad2deg(np.arcsin(-d[2]))
    yaw = np.rad2deg(np.arctan2(d[0], d[1]))
    return Gf.Vec3f(float(pitch), 0.0, float(yaw))
