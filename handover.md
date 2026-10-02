# LunarSim — Handover (2026-10-01, evening)

Read this fully before doing anything. This REPLACES the earlier
2026-10-01 06:33 handover, whose central hypothesis (curriculum shape)
was investigated and turned out not to be the binding problem. Two
different root causes were found and fixed instead. Nothing here is
committed to git yet.

## What this project is

LunarSim: an Apollo Lunar Module landing simulator on NVIDIA Isaac
Sim/IsaacLab, training a SAC policy to fly a physically-real lander from
a high-altitude/high-speed release to a soft touchdown on real cratered
terrain. There is also a fast CPU-only `AnalyticLanderEnv`, but **only
Isaac Sim results count** as progress — the analytic env is for unit
tests and feasibility sweeps, not for reporting policy performance.

## THE HEADLINE: two independent root causes, each alone fatal

Both are fixed. Both were verified by direct measurement, not inference.

### 1. A soft landing was physically undetectable

`isaac_lander_env.py`'s touchdown test was
`belly_z <= ground_z + 1e-3`: the body-frame centre point coming within
1 mm of a single sampled ground height. The collision body is a 2.1 m
radius cylinder, so when it rests on real terrain its belly centre stays
ABOVE the highest ground under the footprint — measured resting
clearance, median per stage: ramp_20m 0.059 m, ramp_50m 0.032,
ramp_100m 0.015, ramp_150m 0.007, orbit_descent 0.006. All 6-59x the
tolerance. (A second, independent error compounded it:
`sample_height_at` returns nearest-sample height while PhysX collides
with the interpolated triangle mesh — measured max disagreement 0.054 m,
and it does not shrink with resolution.)

Direct proof, from `out/eval_snapshots/orbit_descent_multiseed_44/`
(ZemZev controller, real Isaac Sim):

```
t=62.98  pos=(-296.599, 49.109)  alt=0.067  vx=-4.6e-05  vy=4.2e-04  vz=-1.5e-05
eval_summary.txt:  end_reason=max_episode_s   landed_safely=None
```

The vehicle landed at t≈47 s and sat motionless for 16 seconds. The sim
recorded a timeout.

**Consequence: the only kind of touchdown training ever saw was a crash**
— the landing bonus had never once been paid. Every past conclusion
about `landing_bonus_scale` vs `crash_penalty` balance came from runs
where the bonus was unreachable.

### 2. The discount horizon was 7-12x shorter than an episode

`SAC(...)` ran SB3's default `gamma=0.99`. At `dt_s=0.05` that is an
effective horizon of `1/(1-gamma)` = 100 steps = **5.0 seconds**, against
episodes of 20-60 seconds. A touchdown 700 steps (35 s) out was
discounted by `0.99**700 = 8.8e-4`, making `landing_bonus_scale`=1800
worth **1.58 return units** at the release point.

Four strategy archetypes scored end-to-end under the real reward:

| | gamma=0.99 | gamma=0.999 |
|---|---:|---:|
| good landing (35 s) | -198.2 | **+159.1** |
| **stall at 100 m** | **-146.9** | -1343.3 |
| free-fall crash | -276.2 | -1527.9 |
| climb-away | -194.1 | -1597.6 |

**At the shipped gamma, hovering was the optimal policy** — better than
landing successfully, with climbing away tied with landing. A joint
gamma/coefficient sweep showed gamma is the binding constraint: at 0.99
stalling wins for every coefficient variant tried; at 0.999 landing wins
for every one. gamma=0.9995 was tried and rejected — it inverts the
required crash-worse-than-timeout ordering.

This also means: **the undiscounted episode-total arithmetic that most of
`reward.py`'s comments reason with is only a valid proxy while the
discount horizon covers the episode.** If `max_episode_s` or `dt_s`
change, gamma must move with them.

## Everything else that was found and fixed

### reward.py
- **Altitude gate was blind to the sign of vz.** `_braking_ratio` squares
  vz, so a fast CLIMB relaxed the descend incentive exactly as much as a
  fast descent. Measured at alt=100 m: `cost(vz=-10) = cost(vz=+10) =
  249.63`, both less than half of `cost(vz=0) = 507.75` — climbing was
  strictly cheaper than holding station. Gate now fed `min(vz, 0)`;
  descent side bit-for-bit unchanged.
- **`_time_to_ground_s` was discontinuous at vz=0** and unbounded as
  |vz|→0. At alt=200 m, v_xy=25 m/s, going from vz=-1e-7 to -1e-3 moved
  tgo 15.71 s → 200000 s and the horizontal-braking penalty 400 → 0: a
  stalled vehicle shed the entire term by sinking 1 cm/s. Replaced with
  the single ballistic solve `t = (vz + sqrt(vz^2 + 2*g*alt))/g`.
- **`alive_tax_per_step` was 320x off and the wrong sign for the goal.**
  It summed into `r` before `reward_scale`=0.0025 while terminals are
  added after: 1200 steps delivered 3.75 units against a 1200 crash
  penalty. Applied correctly it buys land-vs-stall margin by destroying
  crash-vs-stall margin (605 → 176 at 1.0/step), against the standing
  "crashing must stay worse than hovering" requirement. Set to 0.0.
- **`kappa_alt_sqrt` 50.64 → 25.32.** The 4x raise was justified as
  compensating `reward_scale`'s 4x cut, but the thing actually
  neutralising the term in the climb-away regime was the broken gate
  (4.67x discount at alt=1000/vz=+50). At 4x it was 42.7% of the
  braking-phase shaping sum, quietly undoing the `t0` 1.0→20.0 fix.
- **NEW `_angular_rate_penalty`.** `landed_safely` requires |w| ≤ 0.5
  rad/s but nothing in the shaping read wx/wy/wz, and yaw was never
  penalised at all. See the measured evidence below. Gentle quadratic
  (`omega_k`=60) plus a steep barrier at the limit
  (`omega_cutoff_k`=600, power 6), whole term capped at 800.
- `shaping_clip_abs` 4200 → 5000, re-derived from the caps rather than
  estimated (worst simultaneous sum is 4890; 4200 would have clipped it).

### Simulation / env
- Real triangulation sampling (`collision_mesh_height_at`), tilted-
  cylinder lowest-point clearance (`contact_clearance_m`),
  `TOUCHDOWN_CONTACT_EPS_M = 0.05`. `touched_down`, `info["altitude_m"]`
  and the observation now share ONE definition of altitude, and it
  reaches exactly 0 at contact.
- **Vehicle was flying off the terrain tile.** `sample_height_at` clamps
  out-of-bounds to the edge while the PhysX mesh simply ends, so episodes
  continued over phantom ground — `margins_check_2` recorded a
  "touchdown" at **alt = -32.8 m**, 524 m from centre on a 600 m tile.
  Tile sizing had a factor-2 error (drift must fit the HALF-width) and
  used free-fall time. Now `out_of_tile()` → `truncated` with
  `info["left_tile"]`, and tiles are 60/240/640/1200/1680 m.
- **Eval terrain differed from training terrain.** Capture scripts built
  their own `TerrainConfig` (hills 1.5 m vs 0.3, 10x crater density).
  P(leg_height_diff ≤ 0.16) was 100% on training terrain and **9.7%** on
  capture terrain — a perfect landing had ~1/10 odds of being recorded as
  safe. Now `--terrain-profile training` is the default.
- **The "easiest" curriculum stage was the hardest.** Terrain scales were
  fractions of tile size, so the 60 m tile got 10x steeper slopes than
  the 600 m one. ramp_20m had 42% of episodes unlandable by any policy.
  Scales fixed to absolute metres.
- **Train ran at 20 Hz, eval at 60 Hz.** Capture scripts queried the
  policy every physics step. Now 20 Hz ZOH via `decision_stride`.
- Single shared `lunarsim/rl/curriculum.py`; the stage table had been
  copy-pasted into 4 scripts and diverged (`test_landing_feasibility.py`
  was still certifying stages that no longer existed).

### Training setup
- **`gamma=0.999`** (CLI `--gamma`), forced on the warm-start path too —
  every checkpoint on disk has 0.99 baked in and silently restores it.
- **`gradient_steps=-1`.** SB3 with a VecEnv collects `num_envs`
  transitions per rollout but took ONE gradient step: a 2M-timestep run
  did only 125000 updates. Every "converged to a local optimum, ent_coef
  went flat" conclusion in the old handover came from runs with 16x fewer
  updates than their step counts imply.
- `learning_starts` re-armed per stage (the replay buffer is reset at
  stage boundaries but `learning_starts` compares against the GLOBAL step
  count, so stages 2+ had no warmup and fit the critic from a 16-element
  buffer).
- `--torch-device` (default auto→cuda) for SAC's MLPs only.
- Per-stage `sanity_rollout()` after every stage save, reporting
  landed_safely / lost_control / left_tile counts. Previously the only
  rollout ran once at the very end on the hardest stage, so a multi-hour
  run produced no landing signal until it was over.

## The one measurement we have under the fixed code

`out/diag_ramp_20m_fixed_v1_detail.log` — 500k-step ramp_20m checkpoint,
16 episodes, real Isaac Sim:

```
=== ramp_20m: 2/16 landed_safely (12%), 0/16 lost_control, 10/16 timeout, 0/16 left_tile ===
```

**This is the first non-zero `landed_safely` in the project's recorded
history** (every prior measurement was 0/16). Read it carefully:

- **Proven fixed:** the terminal bonus is now actually payable (two clean
  landings: `vz=0.49 vxy=1.01 tilt=3.3` and `vz=-0.18 vxy=0.68 tilt=4.2`).
  Climb-away is gone — `lost_control` 0/16, highest vertical speed
  +0.88 m/s, where telemetry used to show +28..+50 m/s full-throttle
  climbs to 680-1372 m. `left_tile` 0/16.
- **The angular-rate gap, measured:**
  `[ep 15] CRASH vz=-0.21 vxy=0.48 tilt=7.1 w=0.65
   margins={v_z=0.79, v_xy=0.60, tilt=0.53, w=0.00} worst=w`
  — every other criterion comfortably inside its limit, rejected purely
  on rate. Of the three touchdowns that came closest, two had w as the
  binding margin. This is what `_angular_rate_penalty` was added for; it
  has NOT yet been trained against.
- **Still broken — 10/16 never reach the ground.** Final altitudes
  (spawn is 20 m): 25.8 / 25.8 / 25.5 (climbed above spawn and hung
  there), 20.3 / 20.4 (never descended), 13.2 / 12.6 / 8.6 / 6.0
  (partial), 8.9 (descending at -5.66, ran out of clock). So the stall
  attractor is not gone, it is 50x smaller — 25.8 m instead of 1372 m.

### Why it still stalls — the current best diagnosis

An offline CPU probe of that checkpoint (grid of hand-built normalized
observations matching `_observation()` exactly) shows a clear learned
rule, `net_a` in m/s², positive = climbing:

```
dx=0 (over the target)           dx=15 m (offset)
  alt   vx=0   vx=1   vx=2  vx=5      alt   vx=0   vx=1
 20.0  +1.22  +1.08  -0.20 -1.58     20.0  -1.28  -1.10
  5.0  +0.77  +0.56  -0.37 -1.20      5.0  -0.75  -0.41
  0.5  +0.69  +0.99  +0.59 -1.08      0.5  -0.33  -0.29
```

**"If you are over the target and nearly stopped horizontally, climb;
otherwise descend."** That is exactly inverted at the moment of landing:
the instant the vehicle satisfies the landing condition it throttles up.
It explains all 16 episodes, including why `ep 0` registered a landing
with `vz=+0.49` — moving UP at contact, having dipped below the 0.05 m
threshold and pushed back.

**This is not a reward-ordering bug.** Verified: the per-step reward
penalises altitude monotonically — hovering at 0.5 m costs 0.0450/step,
at 25 m costs 0.3168/step (7x), and the `sqrt(alt)` derivative is
strongest near the ground. The policy is acting AGAINST the shaping
gradient. The ordering is right, the gradient is right; what is missing
is learning. The downward pull is worth ~89 return units over an
episode while the landing bonus is ~1400 but only paid if the hard
contact problem is actually solved — a textbook sparse-reward credit
assignment gap.

Caveat on the probe: single-state queries, not closed loop, at states
(`qw=1`, all rates 0) that never occur exactly. Trust it because it
independently agrees with all 16 rollouts, not on its own.

## WHAT TO DO NEXT (start here tomorrow)

The reward and sim fixes are all in and tested. The next lever is
**giving the critic real examples of a completed landing**, which is what
demo bootstrapping was supposed to do and never did.

1. **Re-collect the ZemZev demos.** The existing
   `out/zemzev_orbit_descent_demos.npz` is unusable and the loader now
   rejects it: 39824 transitions but only **4 terminal transitions** (the
   collector packed episodes into lanes and truncated each group to its
   shortest member, so 44 of 48 landings were cut off before the end),
   plus a 3.52 m altitude bias fed to the controller, plus it was flown
   over the old undersized tile. All three are fixed in the collector.
   ```
   /home/haktan/isaac-env/bin/python scripts/collect_zemzev_demos.py \
     --episodes 48 --out out/zemzev_orbit_descent_demos_v2.npz
   ```
   **Verify the printed `terminal transitions kept: 48/48` before using it.**

2. **Re-run the curriculum from scratch with demo seeding.** All existing
   checkpoints are stale (the altitude observation changed meaning and
   the terrain distribution changed).
   ```
   nohup /home/haktan/isaac-env/bin/python scripts/train_sac_isaac.py \
     --steps-per-stage 500000 --out-dir out/sac_training_run_fixed_v2 \
     --demo-path out/zemzev_orbit_descent_demos_v2.npz \
     > out/train_fixed_v2.log 2>&1 &
   ```
   Per-stage `=== SANITY <stage>: landed_safely=N/16 ... ===` lines now
   appear in the log after every stage save — watch those, not just fps.

3. **Measure each stage checkpoint** as it lands. This can run CONCURRENTLY
   with training (see operational notes).
   ```
   /home/haktan/isaac-env/bin/python scripts/diag_stage_landing_rate.py \
     --checkpoint out/sac_training_run_fixed_v2/sac_lunar_lander_isaac_<stage>.zip \
     --stage <stage> --episodes 16
   ```
   It now prints `alt`, `w`, and the per-criterion `margins` with the
   binding one named, so a rejected touchdown says WHY.

4. **If stalling persists with demo seeding**, the next thing to try is
   potential-based shaping (`F(s,s') = gamma*Phi(s') - Phi(s)`), which
   makes the shaping sum telescope to a constant independent of episode
   length and removes the entire "shaping total vs terminal" balancing
   problem these comments have been fighting for sessions, with a
   policy-invariance guarantee. Shift Phi to be non-negative so the
   `(1-gamma)*|Phi|` survival bonus does not reappear.

## Open decisions (not made, deliberately)

1. **The curriculum is not monotonic.** Under the proven ZemZev
   controller, 24 episodes/stage: ramp_20m **88%**, ramp_50m 54%,
   ramp_100m **38%**, ramp_150m 54%, orbit_descent 75%. ramp_100m is
   harder than the final target, which contradicts the "small delta each
   step" premise the ramp was built on. Rebalancing is a design change
   and was left alone.
2. **Stage advancement still does not gate on competence** — it advances
   on a fixed step budget. A policy that cannot do ramp_20m is
   warm-started into ramp_50m regardless. The per-stage sanity rollout
   now makes this visible; turning it into a gate changes training
   semantics and is a user call.
3. **`--terrain-grid-n 160`.** Currently 80, so orbit_descent's 1680 m
   tile has 21 m cells and the 9.4 m leg span falls inside one cell,
   effectively disabling `safe_landing_max_leg_height_diff_m`.
   `generate_tile` is free at 160 but PhysX bakes 50562 triangles instead
   of 12482 per reset; measure wall-clock before adopting.
4. **PhysX on GPU.** Currently CPU — GPU sat at 0% utilization through a
   whole run while the process used 3 of 16 cores. Moving SAC's MLPs to
   CUDA gave 166 → 231 fps. Note the ceiling: with a 1:1 replay ratio,
   throughput is `N / (t_env(N) + N*t_grad)`, which converges to
   `1/t_grad` ≈ 390 fps on this box no matter how many envs — so
   **raising `--n-envs` cannot raise throughput**, only sample diversity.
   Moving PhysX to GPU is worth at most 231 → ~390, and needs the
   numpy state-read path (`np.asarray(self.body.get_world_poses())`)
   moved to torch or the transfers will eat the gain.

## Known-good operational patterns

- Training: `/home/haktan/isaac-env/bin/python scripts/train_sac_isaac.py`
  under `nohup ... & disown`. **Log into `out/`, not `/tmp`** — a reboot
  wiped the logs for an entire day of runs and left only checkpoints.
- Two startup errors (`libxml2.so.2`, `omni.kit.tool.asset_importer` /
  `OmniClientWrapper`) are expected and benign. Anchor any error grep to
  start AFTER the first `=== stage:` line; they appear before it.
- **`diag_*.py` CAN run concurrently with training.** The old "never run
  eval and training together" rule was about docker-based RENDER captures
  (large render buffers → OOM). The diag scripts are headless with no
  camera/lidar: training used 1.2 GB of 8.1 GB VRAM and 3 of 16 cores.
  Docker `run_isaaclab_*_capture.sh` runs still need exclusive GPU.
- `.venv/bin/python` for CPU work (pytest, reward arithmetic, offline
  checkpoint probes via `SAC.load(..., device="cpu")` — no Isaac needed).
- Tests: `.venv/bin/python -m pytest tests/ -q` → **7 failed, 121
  passed**. The 7 are pre-existing `tests/test_apollo_lm_asset.py`
  failures from a local `.venv` missing `PhysxSchema`; not ours.
- `eval_summary.txt` reports `landed_safely` and the touchdown numbers —
  never trust `end_reason=touchdown` alone.

## Stale artifacts

- **All checkpoints**, including `out/sac_training_run_fixed_v1/
  sac_lunar_lander_isaac_ramp_20m.zip`: the altitude observation changed
  meaning, the terrain distribution changed, and the angular-rate term
  was added after that checkpoint was trained.
- **`out/zemzev_orbit_descent_demos.npz`** — rejected by the loader, see
  step 1 above.
- **`reward_coefficients.md`** is badly out of date: it still documents
  `t0=1.0`, `landing_bonus_scale=500`, `crash_penalty=700`,
  `timeout_penalty=400`, `shaping_clip_abs=400`, and describes the
  braking term in its old `exp()` form. Regenerate it from the current
  field comments when the coefficients settle.

## 2026-10-02: the LiDAR was never producing a point cloud

Separate from everything above (nothing here touches the reward, the env,
or training). Driven by a request to make the dashboard's LiDAR panel do
MAPPING instead of showing one scan at a time. Fixing the panel turned up
why the old panel never looked like terrain.

### The bug

`adapters/isaac/sensors.get_point_cloud` returned `gmo.x/y/z` as if they
were cartesian metres. They are not. Isaac's RTX LiDAR reports in whatever
`gmo.elementsCoordsType` says, and for the Ouster profiles this project
uses that is **SPHERICAL** -- azimuth deg, elevation deg, range m, in the
SENSOR frame. Read straight off the captured files:

```
out/static_capture/lidar_scans/*.ply
  col0 ("x"): -179.9997 .. 179.9998      <- a full 360 deg azimuth sweep
  col1 ("y"):  -11.1125 ..  10.7883      <- the OS2's 22.5 deg vertical FOV,
                                            0.176 deg spacing = 128 channels,
                                            IDENTICAL in every scan while the
                                            vehicle fell 227 m
  col2 ("z"):    2.8846 .. 349.9725      <- range, against a farRangeM of 350
```

So every `.ply` under `out/*/lidar_scans/` -- all 10 captures on disk --
holds angles-and-a-range under the property names (x, y, z). Consequences:

- The old dashboard LiDAR panel and `*_lidar.mp4` were plotting an
  azimuth-vs-elevation-vs-range scatter as a shape in space. The thing on
  screen was an artifact of the mistake, not terrain.
- Those `.ply` files open as garbage in CloudCompare/MeshLab too.
- `scripts/isaaclab_test_rtx_lidar.py` compares `pc["z_m"]` against
  `sample_height_at(..., pc["x_m"], pc["y_m"])` and passes if the mean
  error is under 1 m. It was comparing range against terrain height
  sampled at (azimuth, elevation). **That test is now the natural
  first GPU check of this fix** -- it should pass for real.

Note the analytic path was never affected: `isaac_crater_lidar_scan.py` /
`isaac_rover_lidar_scan.py` go through `core.metadata.lidar.raycast_lidar`,
which has always produced real cartesian points.

### The fix

- `get_point_cloud(sensor, prim_path=None, to_world=True)` now reads
  `elementsCoordsType` / `frameOfReference`, converts spherical to
  cartesian, and transforms into the world frame. It also returns the raw
  `azimuth_deg`/`elevation_deg`/`range_m` and the `sensor_pos_m` /
  `sensor_rotation` it used.
- The sensor pose comes from the prim's USD `ComputeLocalToWorldTransform`,
  NOT from the GMO's `frameStart`/`frameEnd`. That pose carries a 4-float
  `orientation` whose component order (wxyz vs xyzw) is undocumented in the
  extension's type stub, and guessing wrong silently mirrors the cloud. A
  USD matrix has no such ambiguity. (USD composes with row vectors, so the
  translation is the matrix's last ROW and the column-convention rotation
  is the upper-left 3x3 TRANSPOSED.)
- All three capture scripts now record the sensor pose per scan into
  `lidar_index.csv` (`sensor_x_m`/`y`/`z` + `sensor_r00..r22`), so nothing
  downstream has to reconstruct it.

**Not yet run on a GPU.** The adapter change is the one piece of this that
no test on this box can exercise. Run
`scripts/isaaclab_test_rtx_lidar.py` first.

### New: incremental terrain mapping

`lunarsim/core/metadata/mapping.py` (`MapGrid` + `TerrainMap`, pure numpy,
no Isaac): successive world-frame scans are folded into one world-aligned
grid as sum / sum-of-squares / min / max per cell, so memory is fixed by
grid size rather than by descent length. Derived layers: elevation,
coverage, within-cell relief, roughness, slope, and a **landability mask**
that applies the simulator's OWN criterion -- the four-footpad height
spread over the 9.4 m span against
`safe_landing_max_leg_height_diff_m` = 0.16 m, exactly as
`isaac_lander_env._footpad_height_diff_m` does -- but computed from the
LiDAR map instead of from the ground-truth heightfield. Unobserved cells
are never landable, and a cell whose four footpad samples are not ALL
observed returns NaN rather than an optimistic partial spread.

`core/metadata/lidar.py` gained `spherical_to_cartesian`, `sensor_to_world`
and `euler_to_rotation_matrix` (the shared, Isaac-free conversion math).

`tests/test_mapping.py`: 28 tests. The substantive one raycasts a KNOWN
heightfield with the analytic LiDAR from six poses, re-encodes each scan
the way the RTX sensor reports one, pushes it back through the conversion
and the map, and checks the result against the heightfield it started
from. A sign error or transposed rotation anywhere in the chain fails it.

### The dashboard panel

`scripts/compose_capture_video.py`'s bottom-right panel is now the
accumulating map (elevation + landability overlay + ground track +
current scan footprint). The standalone `*_lidar.mp4` is replaced by
`*_map.mp4`: elevation and landability side by side, one frame per real
scan, so the map visibly fills in during the descent. It also writes
`*_map.png` and `*_map.npz` (all layers, for the demo site). `--lidar-out`
still works as an alias for `--map-out`.

Because the scans on disk are spherical, the script detects that
(`--scan-format auto`, from the columns' span shape) and rebuilds the
sensor pose from `telemetry.csv` + the mount constants, so **the existing
captures are usable without re-running Isaac**. New captures carry their
own pose and skip all of it.

### Why the reconstruction is trusted

Validated on `out/static_capture` against the simulator's own ground truth
(98 scans, from 227 m down to touchdown, a 33 m/s free fall):

| mount/frame hypothesis | cross-scan spread per 2 m cell | map vs `pos_z - h/2 - alt_m` |
|---|---:|---:|
| **sensor frame, T(0,0,-2.112) then RotY(60)** | **0.12 m** | **-0.13 m** |
| no mount tilt | 17.8 m | +46.6 m |
| angles already world-referenced | 17.8 m | +83.1 m |
| tilt the other way | 34.7 m | +296 m |

A sweep over how far back the pose is taken put BOTH the absolute error
and the cross-scan spread at a minimum at the same place, 2 physics steps
(0.033 s) before the step the scan was read -- which is what a spinning
sensor acquiring over the preceding revolution should look like. Hence
`--pose-lag-s` defaulting to 2/60. On a gentle powered descent it is
within noise (0.044 vs 0.046 m), so the default is safe for both.

Independent corroboration: the map's landable fraction on the harsh
capture terrain is 7% of mapped area, against the 9.7% this handover
already records for `P(leg_height_diff <= 0.16)` measured from the
ground-truth DEM on that same profile. On the gentler training terrain
(the successful-landing capture) it is 32%. The map reads lower than
ground truth because map noise adds into a 0.16 m threshold -- i.e. it
reports what the sensor can CERTIFY, not what is actually safe. That gap
is real, not a bug, but do not quote the map fraction as the terrain's
true landability.

### Artifacts rebuilt

- `out/static_capture_map.{mp4,png,npz}`
- `out/eval_snapshots/orbit_descent_controller_isaacsim_seed2001_terrainfix/`
  `{map.mp4,dashboard.mp4}` + `..._map.{png,npz}` -- this is the capture
  with `landed_safely=True` (`vz=-0.07`, `vxy=0.18`, `tilt=1.0`), i.e. the
  successful controller landing, 367k m^2 mapped from 253 scans / 3.5M
  returns.

Tests: `.venv/bin/python -m pytest tests/ -q` -> **7 failed, 149 passed**.
Same 7 pre-existing `test_apollo_lm_asset.py` PhysxSchema failures as
before; the count went 121 -> 149 from `tests/test_mapping.py`.

## 2026-10-02 (later): the vehicle was never standing on its legs

Started from a user observation while watching the end of a landing video --
"the legs aren't touching, it's balancing on something". That turned out to
be three separate defects, found by measuring the asset and then by running
a real capture. NONE of them changed any physics RESULT (mass, inertia and
the landing criteria all come from `specs`, never from the mesh), but two of
them did change what the simulation looked like, and one broke the LiDAR.

### 1. The visual LM floated 3.66 m above its own contact plane

The raw mesh stands ON its origin rather than being centred on it: its
lowest point is at +6.14 cm in raw units. Scaled, the visual LM occupied
z = +0.14 .. +7.18 m in the body frame while the collision cylinder occupied
-3.52 .. +3.52. So the physics rested on a face 3.66 m below anything
visible. Fixed with a compensating translate; `_RAW_MESH_BOTTOM_M` records
the measurement.

### 2. The LM rendered at 45% of its real width

`_RAW_MESH_FOOTPRINT_M` was 8.622 m, taken from the asset's WHOLE bounding
box. That box is dominated by one prim, `Object_8` -- a thin boom high on
the ascent stage (y = 231..294 cm) reaching x = +669.8 cm while every other
prim stays inside +/-192.4 cm. Excluding it the body measures
384.7 x 300.0 x 384.6 cm, so the real figure is **3.847 m**. The height
constant was never affected. Measured before/after:

| | visual z range | ground-contacting prim width |
|---|---|---|
| before | +0.144 .. +7.184 (floating) | 4.19 x 4.19 m |
| after | -3.519 .. +3.521 (contact plane -3.520) | **9.40 x 9.40 m** |

### 3. The collider was a 2.1 m disc, not a 9.4 m leg base

This is the one that caused the rocking the user actually noticed. The whole
vehicle collided as ONE `body_radius_m` cylinder, so it rested on a support
polygon 4.5x narrower than its real gear. On the successful ZemZev capture:
first contact at t=44 s (angular-rate spike), then **~10 s of rocking** --
tilt swinging 14.9 -> 6.9 -> 10.4 -> 3.1 deg with the engine already off
(throttle 0.005) and clearance stuck between 0.10 and 0.26 m, a gap free
fall would have closed in half a second.

Now modelled as built: four footpad spheres at the real stance carry
touchdown, with the descent-stage cylinder raised
`_BODY_UNDERSIDE_ABOVE_PADS_M` above them. The centre of mass is pinned to
the origin explicitly -- with several shapes PhysX would otherwise derive it
from the geometry and silently move every thrust/RCS moment arm.

`contact_clearance_m` moved with it: the contact ring is now the footpad
stance, not the body radius, and the stencil went 8 -> 16 azimuths. Note the
landing-safety test ALREADY measured terrain spread under exactly these four
points, so the collider and the criterion only now agree.

`TOUCHDOWN_CONTACT_EPS_M` stays 0.05 m, re-measured for the new geometry:
worst-case stencil error is now **0.0215 m** (was 0.039 m) -- better,
because 16 samples on the bigger ring have comparable arc spacing while
covering relief the small disc missed entirely. 0.0215 + 0.02 (PhysX contact
offset) = 0.0415 m.

### What a real capture then showed

A short GPU capture (`out/eval_snapshots/verify_fixed_geometry`, seed 2001,
14 s) confirmed the contact model and exposed two more bugs:

```
touchdown at step 837 (t=13.95s) -- vz=-0.80 vxy=0.71 tilt=12.0deg
                                    w=0.035 leg_diff=0.026m landed_safely=True
```

- **The RTX LiDAR only saw the vehicle.** Every return in every scan was
  0.8-3.0 m away -- 88074 of them in one scan. Moving the mesh onto its true
  contact plane had put the sensor (z = -2.11) INSIDE the descent stage. The
  old mount only ever worked because the mesh was mis-placed above it.
  Where a downward sensor can live was measured off the real mesh (76209
  vertices binned by radius and z): the descent stage fills radius 0..4 m
  from z = -2.4 upward; below that only the engine bell (r < ~1.0) and the
  legs (r > ~3.0) are occupied. Both descent sensors now mount at
  `descent_sensor_mount_z_m` = -height/2 + 0.5, in that clear band -- the
  height the nav camera already renders a clean ground view from.
- **The recorded sensor pose was frozen.** It read 29.50 m for every scan
  while the vehicle descended from 25.2 m altitude to touchdown.
  `ComputeLocalToWorldTransform` returns the authored SPAWN transform during
  a running Isaac Lab sim -- the live pose is in Fabric/PhysX. The capture
  scripts now compose the pose from the vehicle's live pose and the authored
  mount (`lidar_mount_pose`) and pass it to `get_point_cloud(sensor_pose=)`.
  The USD read remains only as a fallback, with that caveat in its docstring.

### How much the altitude definition moved

Worth knowing before reading any altitude number across the change: moving
the contact ring from the 2.1 m body radius to the 4.7 m footpad stance
makes the reported clearance LOWER, by the geometric term
`(4.7 - 2.1) * sin(tilt)` plus whatever extra relief the bigger ring finds.
Measured over 900 random sites on 3 orbit_descent tiles:

| tilt | median shift | p95 | max |
|---|---:|---:|---:|
| 0 deg | +0.003 m | +0.006 | +0.071 |
| 15 deg | +0.676 m | +0.679 | +0.744 |

So at low tilt it is nothing, and at the tilt limit it is bounded at ~0.7 m.
Note this when a capture's descent looks different from an older one: a
first `hero_fixed` attempt was still at 42 m after 55 s where the original
run had touched down by then, but that is trajectory variation from a
different spawn draw, NOT this shift -- 0.7 m cannot slow a descent by 40%.
It does mean `--max-episode-s 60` is no longer a safe default for a 200 m
release; the re-capture uses 95.

### Verified on real hardware

A second GPU capture (`out/eval_snapshots/verify_v2`, same seed and length)
confirms all of it:

```
 scan   t_s  sensor_z  vehicle_alt  n_hit  median range  median hit z
    0  0.25     25.73        25.21  32069         32.70        -0.17
   15  4.00     13.47        12.62  10368         16.13        -0.19
   30  7.75      6.56         5.40  50830          7.83         0.04
   45 11.50      3.56         1.92  45104          3.84         0.03
   62 15.75      0.55        -0.03  23434          0.88        -0.00
```

The sensor z now tracks the descent instead of sitting frozen, the hits land
on the terrain (median z ~0 m at every altitude, where before every return
clustered at the sensor's own height), and the median range scales with
altitude the way a down-tilted LiDAR's should. `compose_capture_video.py`
auto-detected the capture as WORLD and never entered the legacy spherical
reconstruction path. Touchdown: `landed_safely=True`, `w=0.035 rad/s`,
`leg_diff=0.026 m`. The chase frames show the vehicle resting on its four
footpads at the correct stance.

The nav camera was separately moved off its old mount: at
`-height/2 - 1.2` it sat 1.2 m BELOW the contact plane -- the clearance had
been widened in the wrong direction -- and the feed measured solid black
(mean 0.0, std 0.0) for 20.4 s of the 63 s successful landing, covering the
entire final approach. At the new height it measures std 1.16 -> 38.91
through the descent.

### Correction to an earlier note

The 7 failing `tests/test_apollo_lm_asset.py` tests are NOT "a local .venv
missing PhysxSchema". They fail inside the Isaac container too: `PhysxSchema`
is only importable with a Kit app running, which pytest does not provide.
Everything authorable with plain `UsdPhysics` was moved into
`lander.author_physics` so it CAN be tested, and 6 tests there now pass
locally (visual placement, footpad stance, raised body cylinder, centre of
mass). The remaining 7 still exercise `spawn_apollo_lm` end-to-end and still
cannot run outside Kit.

Tests: **7 failed, 156 passed**.

## MEASURED: the controller nulls velocity but does not reach its target

Found while trying to land on a site chosen from the hazard map. The ZemZev
controller drives horizontal VELOCITY to zero -- it does that very well --
but it does not fly back to `target_x/target_y`. Two captures, both released
at 18.6 m/s:

| capture | target | touchdown | miss |
|---|---|---|---:|
| hero_fixed | (0, 0) | (-596, 142) | **613 m** |
| hero_cinematic (1st try) | (40, 44) | (70, 517), left the tile | **473 m** |

hero_fixed is the run this project calls a successful landing: `vz=-0.80`,
`vxy=0.08`, `landed_safely=True`. It set down 613 m from where it was aimed.

Nothing caught this because `landed_safely` has five terms -- vertical speed,
horizontal speed, tilt, angular rate, footpad flatness -- and **not one of
them is distance to the target**. A controller that lands beautifully in the
wrong place scores identically to one that lands on the pad.

The miss is predictable rather than random: both runs travelled roughly
`30 x release_speed` along the release heading, and hero_fixed's touchdown
was within 15 m of that estimate. `scripts/find_landing_site.py --near-x/-y`
exists to use it -- predict the arrival, then look for a landable patch
THERE rather than next to the release point.

Two consequences worth acting on:
- A divert/landing-site-selection demo needs either a controller that closes
  position error, or a target placed where the vehicle is already going.
- If target accuracy is supposed to matter, `landed_safely` needs a sixth
  term. That is a design change and was left alone.

## RL: what today's bugs did and did NOT touch

Asked directly ("could the RL failure be caused by a bug in the model?"), so
here is the audit.

**Did not touch training at all:**
- The LiDAR spherical/cartesian bug. `_observation` is
  `[x-target, y-target, clearance, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz,
  fuel_frac, rcs_fuel_frac, 0.0]` -- 16 elements, **no LiDAR**. Training
  never saw a point cloud, correct or otherwise.
- The visual mesh placement and scale. The mesh is render-only; mass,
  inertia and collision all come from `specs`. Training runs headless.

**Did touch training:**
- The collision geometry. Every checkpoint on disk was trained while the
  vehicle rested on a 2.1 m disc, and was judged by a flatness test
  measured 4.7 m out -- the collider and the criterion disagreed about
  where the vehicle touches. The footpad fix changes contact dynamics, so
  **all checkpoints are stale again**, for a new reason.
  Caveat on how much this mattered: on the TRAINING terrain the flatness
  term passes ~100% of the time, so it was probably not the binding
  failure there.

**Latent skew worth knowing about (not yet shown to have fired):** the
third observation element means different things in different envs.

| env | element 3 |
|---|---|
| `AnalyticLanderEnv` | `z - ground_z` (body origin above ground) |
| `IsaacLanderEnv` / `IsaacLanderVecEnv` | `contact_clearance_m` (footpad ring's lowest point) |

Those differ by `height_m/2` = **3.52 m** at zero tilt, plus the tilt and
relief terms. `train_sac_isaac.py` (Isaac) and `train_sac_curriculum.py`
(analytic) therefore feed the SAME network input slot two different
quantities, and `train_sac_isaac.py --warm-start` will happily load an
analytic-trained checkpoint into the Isaac env. Any such warm start hands
the policy an altitude channel shifted by 3.5 m. Either make the two
definitions agree or refuse the cross-env load.

(The ZemZev controller is NOT affected: it computes `alt = z - ground_z`
itself from the state dict, the same way in both backends.)

Also noted: `leg_force_frac`, the 16th observation element, is hardcoded
0.0 in all three envs. Not a bug -- the legs only load on the terminal
step -- but it is an input the network can never learn anything from.

**Bottom line:** none of today's findings overturn the handover's own
diagnosis (a sparse-reward credit-assignment gap: ~89 return units of
shaping pull toward the ground against a ~1400 landing bonus that is only
paid once the hard contact problem is already solved). But the contact
change does invalidate the checkpoints, so the next training run is also
the test of whether it moves the needle.

## The demo site

`docs/` is a self-contained static site for GitHub Pages: TR/EN toggle, ~11 MB,
no external requests at all (it works on a trade-show wifi, or none). Publish
it from Settings -> Pages -> Source `main` / `/docs`, which serves it at
https://gulogluhaktan.github.io/LunarSim/ . It is NOT published yet -- that is
the owner's call.

It shows ONE combined 4-panel dashboard per capture rather than scattering the
chase/nav/map feeds across separate players, with the panels explained in
cards beside the video. Four videos total: the landing dashboard, the LiDAR
mapping clip, the free-fall dashboard, and a terrain flyover.

Every number on the page comes from a capture's own outputs and is
regenerated, never hand-edited:

```
./scripts/refresh_demo_site.sh landing out/eval_snapshots/hero_fixed
.venv/bin/python scripts/refresh_demo_site_numbers.py out/eval_snapshots/hero_fixed
```

`refresh_demo_site.sh` re-encodes for the web and picks each poster by image
CONTRAST rather than a fixed timestamp (a fixed one gave a solid-black nav
poster once). `refresh_demo_site_numbers.py` rewrites the 13 figures on the
page from `eval_summary.txt` / `telemetry.csv` / the map `.npz`, and refuses
to run if `landed_safely` is not true. It was validated by running it against
the ORIGINAL capture: the page came back byte-identical.

### The landing it now shows

`out/eval_snapshots/hero_cinematic` -- seed 2001 on the `orbit_descent`
stage preset (200 m release, 18.6 m/s horizontal, 1680 m tile), on the
ROUGH `cinematic` terrain, with every fix from this session in place.

```
touchdown t=91.8s  vz=-0.80  vxy=0.07  tilt=3.2deg  w=0.012
                   leg_diff=0.193m   landed_safely=False
```

Four criteria cleared with wide margins; the fifth missed by 21%. The site
says so, per criterion, rather than calling this a success -- and
`refresh_demo_site_numbers.py` now refuses to publish a failed run unless
`--allow-failed-landing` is passed explicitly.

What makes it worth showing is WHY it missed. The target was chosen by
predicting where the vehicle would actually arrive (release + ~30x release
speed along the release heading) and finding a landable patch there:

| | miss from target |
|---|---:|
| aimed at the tile centre (hero_fixed) | 613 m |
| aimed at the predicted arrival | **11 m** |

11 m is a 56x improvement -- and still not enough, because the patch found
there had only a 12 m clear radius, so an 11 m miss put the footpads on its
edge: 0.106 m spread at the target, 0.193 m where it actually touched down.
The hazard map was right; the guidance could not use it. That is the open
problem the site now states plainly.

Map: 463.2k m2 from 379 scans / 12.87M returns, 40% of it landable,
cross-scan agreement 0.032 m. Terrain relief -1.38..+1.23 m, against
0.56 m on the training-profile tile -- this is why `--terrain-profile
cinematic` is the right one for a video (the flag's own help says so).

`--max-episode-s 120` and `--max-seeds-to-try 1`: the stage default of 60 s
is far too short for a 200 m release (~95 s), and bumping the seed would
regenerate the terrain under a target chosen for the old one.

### Performance note, since it cost this session hours

Captures are NOT slow. Measured from frame mtimes:

| capture | frames/min |
|---|---:|
| pre-change `..._terrainfix` | 156 |
| `hero_fixed` (this session's code) | 223 |
| `hero_cinematic` (cinematic terrain) | 237 |

The fixes made capture ~40% faster, and a full run is roughly 4 min of
Isaac startup + 13 min of capture + 13 min of compose. The hours went to
three aborted attempts of mine (episode cap too short; tile too small for
the measured drift) and to reading progress off a BLOCK-BUFFERED log
instead of the frame files, which produced wildly wrong ETAs. Read
`ls chase_frames | wc -l` over a real interval; the log lags by minutes.

### Still open

**The free-fall section's video still shows the pre-fix geometry.** A
re-capture was attempted and FAILED -- do not just repeat it:

```
./scripts/run_isaaclab_static_telemetry_capture.sh --seed 7 \
  --width 960 --height 720 --fps 30 \
  --spawn-altitude-min-m 220 --spawn-altitude-max-m 230 \
  --spawn-speed-min-m-s 18 --spawn-speed-max-m-s 24      # <-- 600 m tile: WRONG
```

It released at 229 m with 19.8 m/s and flew off the default 600 m tile at
t=22 s, then kept falling THROUGH the terrain (`alt` went -1 -> -44 m) --
`sample_height_at` clamps out of bounds while the PhysX mesh simply ends.
That script's own docstring gives the rule: the tile must exceed
`2 * (spawn_xy_radius + speed_max * fall_time)`. From 229 m the fall takes
~16.8 s, so at 24 m/s that is ~970 m. **Add `--tile-size-m 1200`** (or drop
the release speed), then
`./scripts/refresh_demo_site.sh freefall out/static_capture`.

Two more things about that script:
- **It deletes `out/static_capture` in place.** The dataset the legacy
  spherical reconstruction was validated against is preserved at
  `out/static_capture_legacy_spherical/` (scans + telemetry only, see its
  README.txt). `out/static_capture` itself currently holds the broken
  fell-through-the-terrain run.
- The site's free-fall prose still quotes the OLD capture's numbers
  (223 m, 21 m/s, 26.9 m/s peak). `refresh_demo_site_numbers.py` only
  covers the landing section, so update those three by hand.

**Smaller:** the landing section's hero stat strip shows vz / tilt / mapped
area / fuel. Now that the run misses one criterion, "11 m from the selected
site" would be a more informative fourth tile than fuel remaining -- it is
the number the whole section is about.

## Git

Nothing from today is committed. Working tree touches
`lunarsim/rl/reward.py`, `lunarsim/adapters/isaac/isaac_lander_env.py`,
`lunarsim/adapters/isaac/isaac_lander_vec_env.py`,
`scripts/train_sac_isaac.py`, `scripts/isaaclab_policy_eval_capture.py`,
`scripts/isaaclab_controller_eval_capture.py`,
`scripts/test_landing_feasibility.py`, `tests/test_reward.py`,
`tests/test_isaac_adapter_pure.py`, plus new files
`lunarsim/rl/curriculum.py`, `scripts/collect_zemzev_demos.py`,
`scripts/diag_policy_telemetry.py`, `scripts/diag_stage_landing_rate.py`.

The 2026-10-02 LiDAR/mapping work adds `lunarsim/core/metadata/mapping.py`
and `tests/test_mapping.py`, and touches
`lunarsim/adapters/isaac/sensors.py`, `lunarsim/core/metadata/lidar.py`,
`lunarsim/core/metadata/__init__.py`, `scripts/compose_capture_video.py`,
`scripts/isaaclab_static_telemetry_capture.py`,
`scripts/isaaclab_controller_eval_capture.py`,
`scripts/isaaclab_policy_eval_capture.py`,
`scripts/run_isaaclab_controller_eval_capture.sh`,
`scripts/run_isaaclab_policy_eval_capture.sh`.

The later 2026-10-02 geometry/contact work additionally touches
`lunarsim/core/vehicle/apollo_lm.py`, `lunarsim/adapters/isaac/lander.py`,
`lunarsim/adapters/isaac/isaac_lander_env.py`,
`lunarsim/adapters/isaac/isaac_lander_vec_env.py`,
`tests/test_apollo_lm_asset.py`, `tests/test_isaac_adapter_pure.py`,
and adds the demo site under `docs/`.

Ask before committing.
