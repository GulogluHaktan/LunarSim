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
relying on an in-renderer auto-exposure pass. NOTE: for actual pixel capture
in headless docker, bare `isaacsim.core.api.World` never advances the render
product -- drive the camera through `isaaclab.sim.SimulationContext` +
`isaaclab.sensors.camera.Camera` instead (see adapters/isaac/README.md's
"RESOLVED" section; `create_camera` here still works fine for authoring/
non-headless use, just not for headless-docker pixel capture on its own).

LiDAR: `lidar.mode` selects between the analytic `raycast` path (see
`core.metadata.lidar.raycast_lidar`, no Isaac needed at all, cross-validated
against live PhysX raycasts) and Isaac's real RTX LiDAR
(`isaacsim.sensors.experimental.rtx`), verified here against a set of real
hardware sensor profiles Isaac Sim ships (Ouster OS0/OS1/OS2/VLS-128, Hesai
XT32, SICK units, etc. -- see `SUPPORTED_LIDAR_CONFIGS` in that extension).
"""
from __future__ import annotations

import numpy as np


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


def create_rtx_lidar(prim_path: str, config: str = "OS0", tick_rate: float = 10.0):
    """Author + wire up a real RTX LiDAR sensor at `prim_path`, using one of
    Isaac Sim's built-in real hardware profiles (default "OS0" -- an Ouster
    OS0, a common short/mid-range rover-analog LiDAR; other options include
    "OS1", "OS2", "VLS_128", "XT32_SD10", the SICK units, etc. -- see
    `isaacsim.sensors.experimental.rtx.rtx_lidar_configs.SUPPORTED_LIDAR_CONFIGS`
    for the full list).

    Returns a `LidarSensor` with a `"generic-model-output"` annotator already
    attached; after stepping the sim, call `get_point_cloud(sensor)` to read
    the current frame's hit points.
    """
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

    lidar = Lidar.create(prim_path, config=config, tick_rate=tick_rate)
    sensor = LidarSensor(lidar, annotators=["generic-model-output"])
    return sensor


def get_point_cloud(sensor) -> dict[str, np.ndarray]:
    """Read the current frame's point cloud from an RTX `LidarSensor`.

    Returns a dict with `x_m`/`y_m`/`z_m` (hit points, in the sensor's
    configured frame of reference -- world by default) and `intensity`
    (the GenericModelOutput "scalar" field, reflectivity-like), all
    length-`n_hits` numpy arrays. Empty arrays if no data is available yet
    (e.g. called before the first render tick after creation).
    """
    from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data

    data, _info = sensor.get_data("generic-model-output")
    if data is None:
        return {"x_m": np.empty(0), "y_m": np.empty(0), "z_m": np.empty(0), "intensity": np.empty(0)}

    gmo = parse_generic_model_output_data(data)
    n = gmo.numElements
    return {
        "x_m": np.array(gmo.x[:n]),
        "y_m": np.array(gmo.y[:n]),
        "z_m": np.array(gmo.z[:n]),
        "intensity": np.array(gmo.scalar[:n]) if n > 0 else np.empty(0),
    }


def export_rtx_point_cloud(pc: dict, path: str, fmt: str | None = None) -> str:
    """Write an RTX LiDAR point cloud (the dict `get_point_cloud` returns) to
    disk in the same formats as `core.metadata.lidar.export_point_cloud`
    (`.ply` / `.npy` / `.csv`) -- separate from that function since this
    dict has no `hit_mask` (the RTX sensor already only reports actual hits).
    """
    from lunarsim.core.metadata.lidar import LidarPointCloud, export_point_cloud

    n = pc["x_m"].size
    points_m = np.stack([pc["x_m"], pc["y_m"], pc["z_m"]], axis=-1) if n > 0 else np.empty((0, 3))
    wrapped = LidarPointCloud(
        points_m=points_m,
        # only meaningful as a true sensor range if the points are in the
        # sensor's local frame; not used by export_point_cloud anyway (it
        # only writes points_m + intensity), kept just to satisfy the dataclass.
        range_m=np.linalg.norm(points_m, axis=-1) if n > 0 else np.empty(0),
        intensity=pc["intensity"],
        hit_mask=np.ones(n, dtype=bool),
    )
    return export_point_cloud(wrapped, path, fmt=fmt)
