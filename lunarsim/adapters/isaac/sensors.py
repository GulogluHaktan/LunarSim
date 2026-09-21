"""Camera and RTX LiDAR sensor setup, parameterized by the active `Quality`
profile (plan sections 8-9).

Camera: uses Isaac Sim's high-level `isaacsim.sensors.camera.Camera` wrapper
(falls back to the pre-4.5 `omni.isaac.sensor.Camera` namespace) rather than
authoring `UsdGeom.Camera` by hand -- a prior working Isaac integration on
this same task used the high-level wrapper and found it both simpler and
more reliable than raw USD authoring. Auto-exposure is always left off
(every bundled profile has `camera.auto_exposure: false` -- lunar dynamic
range makes auto-exposure behave badly); `core.lighting.camera` applies the
actual exposure/noise model downstream of the raw rendered frame instead of
relying on an in-renderer auto-exposure pass.

LiDAR: `lidar.mode` selects between the analytic `raycast` path (see
`core.metadata.lidar.raycast_lidar`, no Isaac needed at all) and Isaac's RTX
LiDAR (`rtx_sparse` / `rtx_full` / `rtx`). Unlike the camera path, this has
NOT been validated against a working prior integration -- the one available
reference project never implemented RTX LiDAR either. Treat
`create_rtx_lidar` below as an unverified starting point only.
"""
from __future__ import annotations


def create_camera(prim_path: str, resolution: tuple[int, int] = (1280, 720), focal_length_mm: float = 18.0):
    """Create and return an `isaacsim.sensors.camera.Camera` at `prim_path`.

    `exposure_s`/gain from the active `Quality.camera` profile are applied by
    `core.lighting.camera.apply_noise` on the returned RGBA frame, not here --
    Isaac's camera render product doesn't have a reliable direct
    "auto-exposure off, fixed exposure X seconds" knob across versions
    (ISAAC-VERSION-CHECK if a newer Isaac Sim adds one and you want to move
    exposure into the render itself instead of post-processing).
    """
    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera  # pre-4.5 namespace

    camera = Camera(prim_path=prim_path, resolution=resolution)
    camera.set_focal_length(focal_length_mm / 10.0)  # USD camera focal length is in scene units (cm), not mm
    camera.initialize()
    return camera


def create_rtx_lidar(
    stage,
    prim_path: str,
    mode: str,
    n_channels: int,
    horizontal_fov_deg: float = 360.0,
    vertical_fov_deg: tuple[float, float] = (-15.0, 15.0),
    max_range_m: float = 100.0,
):
    """Author an RTX LiDAR sensor prim.

    UNVERIFIED (see module docstring). ISAAC-VERSION-CHECK: RTX LiDAR
    authoring moved from `omni.isaac.sensor.LidarRtx` (pre-4.5) to
    `isaacsim.sensors.rtx` (4.5+), and the config-profile mechanism (a
    JSON/YAML sensor profile referenced by name, e.g. "Example_Rotary") is
    the actual source of channel/FOV/range truth in recent Isaac versions
    rather than USD attributes set directly -- the attributes set below are
    a structural placeholder; check the installed version's sensor creation
    API (`isaacsim.sensors.rtx.LidarRtx` docs) before relying on them, or
    prefer `core.metadata.lidar.raycast_lidar` (verified, Isaac-independent)
    for anything that doesn't specifically need RTX's GPU-accelerated path.
    """
    from pxr import UsdGeom

    lidar = UsdGeom.Camera.Define(stage, prim_path)  # placeholder prim type
    prim = lidar.GetPrim()
    prim.CreateAttribute("lunarsim:lidar_mode", _sdf_string()).Set(mode)
    prim.CreateAttribute("lunarsim:n_channels", _sdf_int()).Set(n_channels)
    prim.CreateAttribute("lunarsim:horizontal_fov_deg", _sdf_double()).Set(horizontal_fov_deg)
    prim.CreateAttribute("lunarsim:vertical_fov_deg", _sdf_double2()).Set(vertical_fov_deg)
    prim.CreateAttribute("lunarsim:max_range_m", _sdf_double()).Set(max_range_m)
    return lidar


def _sdf_double():
    from pxr import Sdf

    return Sdf.ValueTypeNames.Double


def _sdf_double2():
    from pxr import Sdf

    return Sdf.ValueTypeNames.Double2


def _sdf_int():
    from pxr import Sdf

    return Sdf.ValueTypeNames.Int


def _sdf_string():
    from pxr import Sdf

    return Sdf.ValueTypeNames.String
