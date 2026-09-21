# Isaac adapter

Turns `lunarsim.core` output (heightmap + crater/rock ground truth + sun
position + quality profile) into an Isaac Sim USD stage: heightfield-equivalent
collision, a separate (finer) render mesh, rock instancing, sun light,
regolith material, and camera/LiDAR sensor prims.

**Status: validated against a real Isaac Sim 6.0.1 container** (GPU: RTX 5060
Laptop, 8GB VRAM, via the `lunar-rocket-isaaclab:6.0.1` / `nvcr.io/nvidia/isaac-sim:6.0.1`
images) — see `scripts/isaac_smoke_test.py` (structural: every authoring
function runs against a live stage, no errors) and
`scripts/isaac_validation_suite.py` (behavioral: LiDAR ground truth vs. live
PhysX, physics step speed). Run them with `scripts/run_isaac_smoke_test.sh`.

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

## Known gap: RGB camera rendering in headless docker

`create_camera` (via `isaacsim.sensors.camera.Camera`) authors correctly and
the prim exists on the stage, but in this environment the render product's
frame counter never advances past 0 through bare `isaacsim.core.api.World` +
`world.step(render=True)` -- tried with/without `enable_cameras=True`, with
up to 100 warm-up ticks, with `--network=host --ipc=host`, and against both
the `lunar-rocket-isaaclab:6.0.1` and stock `nvcr.io/nvidia/isaac-sim:6.0.1`
images (same result on both, ruling out the custom image as the cause).

The one thing on this machine that *did* produce real rendered frames
(`LunarRocket/outputs/videos/trained_lunar_landing.mp4`, real PNGs on disk)
drove the camera through an actual Isaac Lab environment's own
`SimulationContext` (`isaaclab.envs.DirectRLEnv` / `env.world.step`), not
bare `isaacsim.core.api.World`. That's a materially bigger dependency (a real
Isaac Lab task, not just isaac-sim) than this adapter currently pulls in --
left as a documented, reproducible gap (see the "KNOWN GAP" comment in
`scripts/isaac_validation_suite.py`) rather than silently claimed as working.
Anyone picking this up: try driving the camera through a minimal
`isaaclab.sim.SimulationContext` instead of `isaacsim.core.api.World` first.

## Other unverified pieces (documented, not yet exercised against a live scene)

- **RTX LiDAR** (`sensors.py::create_rtx_lidar`): still a structural
  placeholder, same as before -- no reference implementation was found even
  in the prior LunarRocket integration (it never implemented RTX LiDAR
  either, only the analytic path). Prefer `core.metadata.lidar.raycast_lidar`
  (now cross-validated, see above) unless GPU-accelerated RTX LiDAR is
  specifically required.
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
  oriented from `core.lighting.sun.SunPosition`, ambient forced off (verified
  to author correctly; not yet validated as visually correct without working
  camera rendering).
- `rocks.py` — individual per-rock prims referencing a prototype asset
  (verified pattern + the typeName bug above), plus `cap_rock_count` for the
  quality profile's `rocks.max_count`.
- `sensors.py` — `isaacsim.sensors.camera.Camera` wrapper (authors/creates
  correctly; frame capture blocked by the gap above) and the RTX LiDAR
  placeholder.
