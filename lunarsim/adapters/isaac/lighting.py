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


def set_no_ambient_render_settings(dlss_quality: bool = True, ambient_occlusion: bool = True):
    """Fix render-level settings that inject light/detail our own scene
    doesn't author -- these are NOT USD prims, so `disable_ambient` (which
    only scans for `UsdLux.DomeLight` prims) can't see or fix them:

    - `/rtx/sceneDb/ambientLightIntensity` defaults to **1.0** in Isaac
      Sim's own shipped rendering-mode presets (both "quality" and
      "performance" .kit configs) -- a flat renderer-level ambient fill
      that directly contradicts the plan's "no ambient, shadows go to
      black" requirement and was the real cause of an early render looking
      washed-out/glossy rather than looking like unlit vacuum shadow. Set
      to 0.0 here.
    - `/rtx/post/dlss/execMode` defaults to 0 ("Performance", the most
      aggressive/lowest-quality upscale mode) in the headless app config
      used to run these scripts. At the camera resolutions used here (a
      few hundred px), Performance-mode DLSS reconstruction produces a
      visible checkerboard/dither artifact at shadow boundaries -- set to
      2 ("Quality") to avoid it. Pass `dlss_quality=False` to skip (e.g.
      if you deliberately want the cheaper mode for a fast preview).

    Call this once after the sim/renderer is up, before capturing frames
    you care about the look of.
    """
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
    if dlss_quality:
        settings.set("/rtx/post/dlss/execMode", 2)
    if ambient_occlusion:
        settings.set("/rtx/ambientOcclusion/enabled", True)


def enable_path_tracing(spp: int = 256):
    """Switch the renderer from real-time raster ("RaytracedLighting", the
    Isaac Sim default) to path tracing.

    The real-time mode's screen-space shadow/AO/upscale approximations
    (soft shadows via a shadow map, DLSS reconstruction, etc.) produced
    visible checkerboard/dither artifacts at shadow boundaries and a
    lingering specular sheen in early lunarsim orbit-demo renders that
    survived multiple targeted real-time-specific fixes (ambient/DLSS
    settings, material specular workflow). Path tracing computes shadows
    and shading from actual traced rays instead of those approximations,
    which is the more direct fix when the real-time pipeline's
    approximations themselves are the problem. Expect meaningfully slower
    renders in exchange.
    """
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", spp)
    settings.set("/rtx/pathtracing/totalSpp", spp)
    settings.set("/rtx/pathtracing/maxSamplesPerLaunch", 1_000_000)


def disable_ambient(stage, dome_light_prim_path: str | None = None):
    """Ensure no skylight/ambient contributes -- zero-intensity any dome light.

    With an explicit `dome_light_prim_path`, only that prim is touched. With
    `None` (the common case -- Kit/Isaac Sim sometimes seeds a new stage with
    a default dome light we never authored and don't have a path for), the
    whole stage is scanned for `UsdLux.DomeLight` prims and all of them are
    zeroed; this is the call to make right before capturing a frame if
    unexplained specular/ambient brightness shows up despite `create_sun_light`
    being the only light explicitly authored.
    """
    from pxr import UsdLux

    if dome_light_prim_path is not None:
        prim = stage.GetPrimAtPath(dome_light_prim_path)
        if prim.IsValid():
            UsdLux.DomeLight(prim).CreateIntensityAttr(0.0)
        return

    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
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
