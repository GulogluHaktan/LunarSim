# Isaac adapter

Turns `lunarsim.core` output (heightmap + crater/rock ground truth + sun
position + quality profile) into an Isaac Sim USD stage: heightfield-equivalent
collision, a separate (finer) render mesh, rock instancing, sun light,
regolith material, and camera/LiDAR sensor prims.

**Status: fully validated against a real Isaac Sim 6.0.1 container**, GPU/render
included (RTX 5060 Laptop, 8GB VRAM; `lunarsim-isaacsim:6.0.1` for everything
except camera pixels, `lunarsim-isaaclab:6.0.1` for those -- see
`docker/Dockerfile.isaacsim` / `docker/Dockerfile.isaaclab`). Run
`scripts/run_all_isaac_validations.sh` for the full one-command regression
suite (structural authoring, sun-direction math, rock collision physics,
dust trajectories, LiDAR-vs-PhysX, physics speed); camera rendering has its
own script (`scripts/isaaclab_test_camera.py`) since it needs the Isaac Lab
image.

## What's been verified for real (not just written against API docs)

- **Collision authoring**: Isaac 6.0.1 has no dedicated heightfield-collider
  API; the correct, tested pattern is a plain `UsdGeom.Mesh` +
  `UsdPhysics.CollisionAPI.Apply()` + `physxCollision:approximation = "none"`
  (exact triangle mesh). `heightfield.py` does this.
- **LiDAR ground truth accuracy**: `core.metadata.lidar.raycast_lidar`
  (pure Python/numpy, no Isaac needed) was cross-checked against
  `omni.physx.get_physx_scene_query_interface().raycast_closest()` on the
  SAME collision mesh, 384 rays: 341/341 hit-count match, mean signed error
  **0.37 mm**, RMSE 1.58 m (driven by ~4-9 grazing/near-edge rays out of 337
  compared -- worth tightening the analytic raycaster's step size for
  shallow-angle rays if sub-meter accuracy near a tile's edge matters for a
  given use case; the un-biased mean confirms no systematic error in the
  ground-truth model itself).
- **Physics step speed**: ~1900-2100 steps/s, single env, physics-only
  (`world.step(render=False)`), on the RTX 5060 above. This is a floor, not a
  target -- plan section 10 calls for 512-1024 parallel envs; that needs an
  actual multi-env Isaac Lab task wrapper (not built here) to measure
  properly, since per-env cost drops sharply under GPU-batched PhysX.
- **Rock referencing**: a real bug was caught and fixed here -- authoring the
  rock host prim with an explicit `Xform` typeName before adding the asset
  reference shadows the referenced asset's real type (USD type resolution is
  local-strongest), leaving rocks with no visible/collidable geometry despite
  the reference existing. Fixed by defining the prim with no typeName so the
  referenced asset's type composes through (verified: prim resolves to the
  prototype's actual type, e.g. `Sphere`, after the fix).
- **Two physics scenes is a real footgun**: `isaacsim.core.api.World()`
  creates its own default `/physicsScene` unless you pass
  `physics_prim_path=` pointing at your own scene prim. Having both a custom
  `/World/PhysicsScene` (lunar gravity) and World's own default one active
  at once measurably corrupted PhysX raycast results (all 384/384 rays
  suddenly "hit" and showed a large systematic range bias) until fixed by
  passing `physics_prim_path="/World/PhysicsScene"` to `World()`.
- **Rock collision, physically confirmed**: dropped a 0.2 m rigid ball from
  10 m above a 2 m rock (real external `.usd` prototype asset, real
  `convexHull` collision) under lunar gravity and stepped real PhysX to
  settle. It came to rest at z=2.200 m -- exactly the analytically predicted
  contact height (rock top at 2.0 m + ball radius 0.2 m), confirming
  `spawn_rocks`'s collision setup is physically correct, not just
  structurally present. (`scripts/isaac_test_rock_collision.py`.)
- **Sun light direction, found badly wrong and fixed**: the original
  `create_sun_light` derived a hand-rolled Euler-XYZ (pitch, 0, yaw) from the
  target direction, assuming a specific rotation composition order. Checked
  directly against the light's actual computed local-to-world transform
  (`scripts/isaac_test_sun_rotation.py`, no render/GPU needed -- pure
  transform math) across 5 cases (zenith, two horizon azimuths, 45 deg, a
  low south-pole-like elevation): **every single case pointed the light in
  the wrong direction** (dot product with the intended direction was 0 or
  0.5, should be 1.0). This would have produced completely wrong shadow/
  terminator geometry with no obvious symptom short of comparing against
  ephemeris ground truth. Fixed by replacing the Euler derivation with
  `Gf.Rotation(fromVec, toVec)` (USD's direct vector-to-vector rotation
  constructor) -> quaternion; re-verified, all 5 cases now dot=1.0000.
- **Dust particles, authored and verified moving**: `dust.py` bakes
  `core.dust.plume`'s exact vacuum-ballistic trajectories as USD
  time-sampled `PointInstancer` positions. Verified: 410-particle burst at
  0.8 intensity produced 265 time samples, particles reached up to ~89 m
  horizontal range (physically correct for the no-drag low-angle-ejecta
  model at ~12.6 m/s peak speed under 1.62 m/s² gravity -- this is the real
  reason lunar dust famously travels in such long, low arcs), and
  per-instance scale shrinks as each particle settles. One bug found+fixed
  along the way: `UsdGeom.PointInstancer` is a schema wrapper, not a
  `Usd.Prim` -- `instancer.CreateAttribute(...)` doesn't exist, needed
  `instancer.GetPrim().CreateAttribute(...)`; also switched the settling-fade
  signal from a nonstandard custom `"widths"` attribute to
  `PointInstancer.GetScalesAttr()`, which is the actual USD-standard
  per-instance size control (the custom attribute also tripped a harmless
  but pointless Fabric/Hydra warning). (`scripts/isaac_test_dust.py`.)

## RESOLVED: RGB camera rendering works via Isaac Lab's SimulationContext

`create_camera` (bare `isaacsim.sensors.camera.Camera` + `isaacsim.core.api.World`)
never got the render product's frame counter past 0 in this headless docker
setup -- tried with/without `enable_cameras=True`, up to 100 warm-up ticks,
`--network=host --ipc=host`, both the `lunar-rocket-isaaclab:6.0.1` and
stock `nvcr.io/nvidia/isaac-sim:6.0.1` images. **Fix: drive the camera
through `isaaclab.sim.SimulationContext` + `isaaclab.sensors.camera.Camera`
instead** (see `scripts/isaaclab_test_camera.py`, following IsaacLab's own
`scripts/tutorials/04_sensors/run_usd_camera.py` reference pattern:
`AppLauncher(args_cli)` with `--enable_cameras`, `sim.step()` +
`camera.update(dt=sim.get_physics_dt())` each tick, reading
`camera.data.output["rgb"]`). Confirmed real RGB frames (240x320x3) with a
**physically correct trend**: mean scene brightness rose monotonically with
sun elevation (raw RGB mean 2°→28.0, 15°→54.4, 45°→94.1, 80°→106.5; after
`core.lighting.camera`'s sensor model at a fixed exposure, mean DN out of a
4095 max 2°→516, 15°→1003, 45°→1735, 80°→1964) -- exactly the expected "no
ambient, brightness/signal tracks direct illumination angle" behavior, and
confirms the sun-direction fix above is visually correct too, not just
numerically. Note the low-elevation case genuinely only fills ~13% of the
sensor's dynamic range at this fixed exposure -- the real, physical reason
low-sun operations (e.g. near the south pole) are harder to see by: less
signal reaches the sensor, not a rendering artifact. Building this took a
real Isaac Lab checkout (`docker/Dockerfile.isaaclab`, pinned
`v3.0.0-beta2.patch1`, ~5GB of torch+cu128 -- lesson from an earlier failed
attempt: bake a toolchain install like this into a Dockerfile `RUN` step,
never a throwaway `docker run --rm` container, or everything pip-installed
is destroyed the instant that container exits).

Bare `isaacsim.core.api.World` is still fine (and much lighter/faster to
start) for everything that doesn't need camera pixels -- collision, LiDAR
ground truth, dust, physics speed were all validated on it. Only reach for
the Isaac Lab image when you actually need rendered images.

## Attempted and NOT working: RTX LiDAR (`sensors.py::create_rtx_lidar`)

This is now a real implementation against the actual API (not a placeholder):
`isaacsim.sensors.experimental.rtx.Lidar.create(path, config="OS0", ...)` +
`LidarSensor(lidar, annotators=["generic-model-output"])`, matching NVIDIA's
own test suite (`isaacsim.sensors.experimental.rtx/.../tests/test_lidar.py`,
`test_lidar_sensor.py`) and using a real hardware sensor profile (Ouster OS0;
also available: OS1/OS2/VLS-128, Hesai XT32, several SICK units -- see
`SUPPORTED_LIDAR_CONFIGS` in that extension). Two bring-up paths were tried
and **both failed** (`scripts/isaac_test_rtx_lidar.py`,
`scripts/isaaclab_test_rtx_lidar.py`):

- Bare `isaacsim.core.api.World` (with or without `enable_cameras=True`,
  60 step attempts): the annotator never returns valid data --
  `parse_generic_model_output_data` logs "Invalid magic number" on every
  single attempt, meaning the render pipeline hands back an uninitialized/
  garbage buffer instead of a real GenericModelOutput frame.
- `isaaclab.sim.SimulationContext` (the path that *does* work for the
  camera, see below): the process **segfaults** (exit 139) during startup/
  sensor creation, before any of our own code's print statements even run.

This extension is explicitly namespaced `experimental` by NVIDIA, and both
failure modes point at real instability in this specific Isaac Sim 6.0.1 /
Isaac Lab `v3.0.0-beta2.patch1` combination in a headless container, not an
obvious usage mistake (the calls match the vendor's own tests verbatim).
**Use `core.metadata.lidar.raycast_lidar` instead** -- it's cross-validated
to 0.37mm mean error against live PhysX raycasts (see above) and has no
such instability. If GPU-raytraced RTX LiDAR is specifically required for a
future need, the next things to try: a non-experimental/older RTX LiDAR
namespace if the installed Isaac version has one, a newer/older Isaac Sim
patch release, or running with a real display instead of fully headless.

## Other unverified pieces (documented, not yet exercised against a live scene)

- **Hapke MDL material**: `materials.py` still falls back to a flat
  `UsdPreviewSurface` approximation for `hapke`/`hapke_approx`; no custom MDL
  shader is bundled.

## Modules
- `heightfield.py` — exact-triangle-mesh PhysX collision (verified) + a
  separate, denser USD render mesh, and a physics (friction/restitution)
  material factory.
- `materials.py` — `UsdPreviewSurface` visual material (verified pattern,
  reliable across render modes) with an optional real MDL asset bind-through.
- `lighting.py` — distant light sized from `SUN_ANGULAR_DIAMETER_DEG`,
  oriented from `core.lighting.sun.SunPosition`, ambient forced off. Verified
  two ways: pure transform math (light's actual local-to-world -Z direction
  matches the intended sun direction, dot=1.0000) and visually (real
  rendered frames show brightness rising monotonically with sun elevation).
- `rocks.py` — individual per-rock prims referencing a prototype asset
  (verified pattern + the typeName bug above), plus `cap_rock_count` for the
  quality profile's `rocks.max_count`.
- `sensors.py` — `isaacsim.sensors.camera.Camera` wrapper: authors correctly;
  for actual frame capture use `isaaclab.sensors.camera.Camera` +
  `isaaclab.sim.SimulationContext` instead (see the RESOLVED section above) --
  bare `isaacsim.core.api.World` never advances the render product. RTX LiDAR
  remains a structural placeholder.
- `dust.py` — bakes `core.dust.plume` ballistic trajectories as a
  time-sampled `PointInstancer` (verified, see above); `dust_event_from_disturbance`
  is a convenience constructor scaling particle count/speed from a single
  0-1 "intensity" (plume throttle, wheel slip, footstep force, etc).
