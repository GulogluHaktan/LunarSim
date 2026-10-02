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


def enable_isaac_extension(*extension_names: str) -> None:
    """Force-enable one or more Kit extensions via `omni.kit.app` directly.

    REAL BUG FOUND (twice, via a local non-docker run and then again
    against the docker image itself): several `isaacsim.*` packages that
    import fine under a bare `isaacsim.SimulationApp` raise
    `ModuleNotFoundError` under `isaaclab.app.AppLauncher` -- the
    corresponding Kit extension just isn't enabled by IsaacLab's base app
    profile (confirmed for `isaacsim.sensors.experimental.rtx` and
    `isaacsim.core.prims`; likely others too). Enabling explicitly here
    fixes both paths (a no-op if something else already enabled it).

    Goes through `omni.kit.app` directly rather than the
    `isaacsim.core.utils.extensions` wrapper some Isaac Sim docs point to --
    that wrapper PACKAGE is itself unavailable under IsaacLab's app profile
    for the same reason (`ModuleNotFoundError: No module named
    'isaacsim.core.utils'`), while `omni.kit.app` (what the wrapper calls
    internally) is always present, under any Kit app profile.
    """
    import omni.kit.app

    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    for name in extension_names:
        ext_mgr.set_extension_enabled_immediate(name, True)


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

    See `enable_isaac_extension`'s docstring for why this needs to force-
    enable its own extensions under `isaaclab.app.AppLauncher`.
    """
    enable_isaac_extension("omni.usd.schema.omni_sensors", "isaacsim.sensors.experimental.rtx")
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

    lidar = Lidar.create(prim_path, config=config, tick_rate=tick_rate)
    sensor = LidarSensor(lidar, annotators=["generic-model-output"])
    return sensor


# Vehicle-frame mount height shared by the downward-looking descent sensors
# (nav camera and LiDAR), and the LiDAR's own down-tilt.
#
# REAL BUG THIS ENCODES, found by running a capture after the visual mesh was
# moved onto its true contact plane: with the LiDAR at z = -height*0.3
# (-2.11 m) a single scan came back with 88074 hits, EVERY ONE of them
# between 0.8 and 3.0 m from the sensor -- the vehicle's own descent stage,
# no terrain at all. The old mount only ever worked because the visual mesh
# was mis-placed 3.66 m too high, leaving the sensor in open space below it.
#
# Where a downward sensor can actually live was then measured off the real
# mesh, binning all 76209 vertices by (radius, z) in the corrected body
# frame: the descent stage fills radius 0..4 m from z = -2.4 upward, so
# anything above that stares into structure. Below -2.4 the only occupied
# radii are the engine bell (r < ~1.0) and the legs (r > ~3.0). This height
# sits in that clear band, 0.5 m above the footpad contact plane, and is the
# same height the nav camera already renders a clean ground view from.
LIDAR_MOUNT_TILT_DEG = 60.0
_DESCENT_SENSOR_ABOVE_CONTACT_PLANE_M = 0.5


def descent_sensor_mount_z_m(specs) -> float:
    """Vehicle-frame z of the downward-looking descent sensors."""
    return -specs.height_m * 0.5 + _DESCENT_SENSOR_ABOVE_CONTACT_PLANE_M


def lidar_mount_pose(specs, vehicle_pos_m, vehicle_rotation):
    """`(sensor_pos_world, sensor_to_world_rotation)` for the descent LiDAR,
    composed from the vehicle's CURRENT pose and the authored mount.

    Pass the result to `get_point_cloud(..., sensor_pose=...)`. Reading the
    pose off the USD stage instead does NOT work during a running sim: under
    Isaac Lab the live pose lives in Fabric/PhysX and the stage keeps the
    authored spawn transform. Measured in a real capture -- the recorded
    sensor z stayed at 29.50 m for every scan while the vehicle actually
    descended from 25.2 m altitude to touchdown.
    """
    from lunarsim.core.metadata.lidar import euler_to_rotation_matrix

    offset = np.array([0.0, 0.0, descent_sensor_mount_z_m(specs)])
    rotation = np.asarray(vehicle_rotation, dtype=float)
    mount = euler_to_rotation_matrix(0.0, np.deg2rad(LIDAR_MOUNT_TILT_DEG), 0.0)
    return np.asarray(vehicle_pos_m, dtype=float) + rotation @ offset, rotation @ mount


def sensor_world_transform(sensor, prim_path: str | None = None):
    """`(position, rotation)` of an RTX sensor in the world frame, read from
    USD -- a (3,) translation and a 3x3 sensor->world rotation matrix in the
    column-vector convention (`p_world = rotation @ p_sensor + position`).

    Read off the prim's `ComputeLocalToWorldTransform` rather than off the
    GMO's own `frameStart`/`frameEnd` pose: that pose carries a 4-float
    `orientation` whose component order (wxyz vs xyzw) is not documented in
    the extension's type stub, and guessing it wrong silently mirrors the
    whole point cloud. A USD matrix has no such ambiguity. (USD composes
    with row vectors -- `v_world = v_local * M` -- so the translation is
    M's last ROW and the column-convention rotation is M's upper-left 3x3
    TRANSPOSED.)
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    prim = None
    if prim_path is not None:
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    else:
        authored = getattr(sensor, "lidar", None)
        prims = getattr(authored, "prims", None) if authored is not None else None
        if prims:
            prim = prims[0]
    if prim is None or not prim.IsValid():
        raise ValueError(
            "could not resolve the sensor's USD prim; pass prim_path= explicitly"
        )

    matrix = np.array(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=float,
    )
    return matrix[3, :3].copy(), matrix[:3, :3].T.copy()


def get_point_cloud(sensor, prim_path: str | None = None, to_world: bool = True,
                    sensor_pose: tuple | None = None) -> dict[str, np.ndarray]:
    """Read the current frame's point cloud from an RTX `LidarSensor`.

    Returns a dict of length-`n_hits` arrays: `x_m`/`y_m`/`z_m` (CARTESIAN
    hit points, in the world frame when `to_world` and the sensor pose is
    resolvable, otherwise in the sensor frame) and `intensity` (the
    GenericModelOutput "scalar" field, reflectivity-like). When the sensor
    reports spherically it also returns the raw `azimuth_deg`,
    `elevation_deg` and `range_m`. `sensor_pos_m`/`sensor_rotation` carry
    the pose the conversion used, and `coords_type`/`frame_of_reference`
    record what the sensor actually reported. Empty arrays if no data is
    available yet (e.g. called before the first render tick after creation).

    `sensor_pose` is `(position, 3x3 rotation)` for the sensor in world
    coordinates; pass it (see `lidar_mount_pose`) whenever a simulation is
    running. Without it the pose is read off the USD stage, which under
    Isaac Lab keeps the authored SPAWN transform while the live pose lives
    in Fabric/PhysX -- a capture built that way recorded the same sensor
    position for every scan of an entire descent.

    REAL BUG THIS FIXES, found by reading a finished capture's own exports:
    this used to return `gmo.x/y/z` directly, described as "in the sensor's
    configured frame of reference -- world by default". Both halves of that
    were wrong for the Ouster profiles this project uses. `gmo` reports in
    whatever `gmo.elementsCoordsType` says, and that is SPHERICAL --
    azimuth deg, elevation deg, range m -- in the SENSOR frame. So every
    `.ply` written by every capture run so far holds angles-and-a-range
    under the property names (x, y, z): a 600 m tile's scans had an "x"
    column spanning exactly -179.9997..179.9998 and a "y" column spanning
    -11.11..10.79 with 0.176 deg spacing that never moved between scans --
    a full 360 deg azimuth sweep and the OS2's 22.5 deg vertical FOV over
    128 channels, not metres. Anything downstream that treated those as a
    shape in space (the dashboard's LiDAR panel; any CloudCompare/MeshLab
    open of those files) was looking at an artifact of the mistake.
    """
    from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data

    empty = {
        "x_m": np.empty(0), "y_m": np.empty(0), "z_m": np.empty(0), "intensity": np.empty(0),
    }
    data, _info = sensor.get_data("generic-model-output")
    if data is None:
        return empty

    gmo = parse_generic_model_output_data(data)
    n = int(gmo.numElements)
    if n <= 0:
        return empty

    coords_type = getattr(gmo.elementsCoordsType, "name", str(gmo.elementsCoordsType))
    frame_of_reference = getattr(gmo.frameOfReference, "name", str(gmo.frameOfReference))
    c0 = np.array(gmo.x[:n], dtype=float)
    c1 = np.array(gmo.y[:n], dtype=float)
    c2 = np.array(gmo.z[:n], dtype=float)
    intensity = np.array(gmo.scalar[:n], dtype=float)

    out: dict[str, np.ndarray] = {"intensity": intensity}
    if coords_type == "SPHERICAL":
        from lunarsim.core.metadata.lidar import spherical_to_cartesian

        out["azimuth_deg"], out["elevation_deg"], out["range_m"] = c0, c1, c2
        points = spherical_to_cartesian(c0, c1, c2)
        points_are_world = False
    else:
        points = np.stack([c0, c1, c2], axis=-1)
        out["range_m"] = np.linalg.norm(points, axis=-1)
        points_are_world = frame_of_reference == "WORLD"

    if to_world and not points_are_world:
        from lunarsim.core.metadata.lidar import sensor_to_world

        if sensor_pose is not None:
            position, rotation = sensor_pose
            position = np.asarray(position, dtype=float)
            rotation = np.asarray(rotation, dtype=float)
        else:
            position, rotation = sensor_world_transform(sensor, prim_path)
        points = sensor_to_world(points, position, rotation)
        out["sensor_pos_m"] = position
        out["sensor_rotation"] = rotation

    out["x_m"], out["y_m"], out["z_m"] = points[:, 0], points[:, 1], points[:, 2]
    out["coords_type"] = coords_type
    out["frame_of_reference"] = frame_of_reference
    return out


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
        # the sensor's own reported range when `get_point_cloud` has it;
        # the vector norm is only the true range while the points are in
        # the sensor frame, and they are world-frame by default now. Not
        # used by `export_point_cloud` either way (it only writes points_m
        # + intensity), so this just keeps the dataclass honest.
        range_m=pc.get("range_m", np.linalg.norm(points_m, axis=-1) if n > 0 else np.empty(0)),
        intensity=pc["intensity"],
        hit_mask=np.ones(n, dtype=bool),
    )
    return export_point_cloud(wrapped, path, fmt=fmt)
