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

## 2026-10-02 (evening): the reward was scoring the proven landing as a loss

Method, and the reason to trust this over any previous round of coefficient
tuning: instead of reasoning about what the terms *ought* to reward, replay a
flight that is *known* to be correct. `out/eval_snapshots/hero_fixed/` is a
real Isaac Sim ZemZev landing with `landed_safely=True` (vz=-0.80, vxy=0.08,
tilt=3.3 deg, w=0.014, leg_diff=0.006). Score it step by step under the real
reward, discount at the shipped gamma=0.999, and compare with stall/crash
archetypes.

```
                            BEFORE      AFTER (x10 rescale applied)
  controller landing (95s)  -895.6      +14.58   <- now best
  stall @100 m, timeout     -759.0      -75.90
  stall @ 20 m, timeout     -514.5      -51.45
  free-fall crash          -1328.8     -118.95   <- stays worst
```

**Stalling at 20 m beat a perfect landing by 381 units.** That is exactly the
measured behaviour of every run: the ramp_20m probe in the section above hung
10/16 episodes at 20-25 m. The policy was not failing to learn the task; it
was learning what this reward defined. The previous diagnosis in this file --
"a sparse-reward credit assignment gap... the ordering is right, the gradient
is right; what is missing is learning" -- was wrong on the first clause. The
ordering was inverted.

Term-by-term bill of that winning flight under the old weights (2076 units
against an 1800 bonus, i.e. net negative):

| term | raw | share |
|---|---:|---:|
| altitude | 322543 | 38.8% |
| **xy_pos** | 264135 | **31.8%** |
| **braking_xy** | 223633 | **26.9%** |
| vel_track | 19541 | 2.4% |
| everything else | <600 | 0.1% |

### What changed

- **`x1` 0.05 -> 0.0.** 31.8% of the winning flight's cost, levied because it
  touched down 613 m from `target_x/target_y`. `landed_safely` has five
  criteria and not one is distance to target (this file already records that
  finding). A shaping term must not be the largest penalty on a flight the
  success test calls perfect, for a quantity that test never reads.
- **`t0_xy` 20.0 -> 2.0.** Was an explicitly UNTUNED guess mirrored off the
  vertical `t0`. It charged 26.9% of the bill for executing the
  reference-correct braking profile (18.6 -> 0 m/s over 90 s).
- **`landing_bonus_scale` 1800 -> 4500**, then the whole scale /10.

### Then it diverged, and that is also measured

The 4500 bonus made landing win but blew up the value scale. Run
`out/train_cal_v1.log`, 268k steps:

```
   step    critic_loss   ent_coef
  11696       7.15e+03      0.635
 132496       7.69e+04       2.08
 266592       3.03e+06       6.61   <- 400x, monotonic, still climbing
```

SAC's alpha is tuned only against `target_entropy` = -dim(A) = -4, so a large
reward scale pulls the policy deterministic, entropy falls under target, alpha
climbs to push back -- and alpha re-enters the critic target
(`Q = r + gamma*(Q' - alpha*log_pi)`), which at gamma=0.999 compounds over a
~1000-step horizon. Feedback loop, not a fitting problem. Two fixes, both in:

- **Uniform /10 on the entire reward**: `reward_scale` 0.0025 -> 0.00025,
  `landing_bonus_scale` 4500 -> 450, `crash_penalty` 1200 -> 120,
  `timeout_penalty` 1050 -> 105. A uniform positive scale cannot change the
  optimal policy, so the table above holds exactly (verified: every figure is
  the pre-scale one / 10). **These four are ONE scale -- move them together.**
- **`--ent-coef`, default 0.01, no longer `auto`.** Sized against the measured
  per-step reward on the winning run (mean |r| = 0.0483), so the entropy term
  `alpha*|target_entropy|` = 0.04 is ~0.8x the task signal per step and ~8% of
  the discounted terminal over an episode. Forced on the warm-start path too,
  like gamma -- a checkpoint saved mid-runaway carries its own `log_ent_coef`
  and would silently resume at 6.6. Verified against real SB3 2.9.0 that both
  the fresh and the auto->fixed path train, save and reload (SB3 branches
  `_get_torch_save_params` on `ent_coef_optimizer is not None`).

### Ready to run

Demos re-collected under the fixed reward (the loader rejects a file whose
stored rewards came from different weights):
`out/zemzev_orbit_descent_demos_v3.npz` — check its log reports
`terminal transitions kept: 48/48`.

```
setsid nohup /home/haktan/isaac-env/bin/python scripts/train_sac_isaac.py \
  --only-stage orbit_descent --steps-per-stage 500000 \
  --out-dir out/sac_training_run_cal_v2 \
  --demo-path out/zemzev_orbit_descent_demos_v3.npz --demo-repeat 6 \
  > out/train_cal_v2.log 2>&1 < /dev/null &
```

~156 steps/s measured (not the 399 the first log line suggests -- that is the
pre-warmup collection rate, before `gradient_steps=-1` starts costing), so
500k steps is ~55 min. Watch `grep -E "SANITY|critic_loss|ent_coef"`: if
critic_loss climbs past ~1e4 again the scale fix was insufficient.

Note `--only-stage orbit_descent`, not the full ramp: the replay buffer is
reset at every stage boundary, so demos seeded at stage 1 are discarded before
the stage they were collected for.

## 2026-10-04: three parallel audits — four independent defects

Dispatched three agents over reward.py / the Isaac envs / the SAC setup. Every
claim below was re-verified by hand before acting on it.

### 1. Training had collision geometry the termination test cannot see

`isaac_lander_vec_env.py:195-213` spawns up to `--max-rocks-per-env 10`
*collidable* spheres per env per reset. `IsaacLanderEnv` (demos, every
`diag_*.py`) spawns **none**. `contact_clearance_m` derives `ground_z` purely
from `collision_mesh_height_at(height, ...)` — the heightfield — so a vehicle
resting on a rock reports clearance up to **0.65 m** against
`TOUCHDOWN_CONTACT_EPS_M = 0.05`. `touched_down` never fires; the episode runs
to timeout while physically parked.

This is headline bug #1 of this file (a soft landing being undetectable)
reintroduced through a channel the fix never covered, and it is invisible to
every diagnostic we have, because the diagnostics run the rock-free env.
Measured P(a capped rock inside the 4.7 m footpad ring at a random site):
**ramp_20m 25.8%**, ramp_50m 0.8%, others <=0.1% — worst on the one stage the
curriculum must bootstrap from.

Mitigated with `--max-rocks-per-env 0`, which also restores train/demo parity.
NOT fixed properly: making the contact test rock-aware needs the per-env
spawned-rock subset threaded into `contact_clearance_m`, and
`_footpad_height_diff_m` is heightfield-only too, so a partial fix would just
move the inconsistency.

### 2. The target network was 5x faster than the horizon it stabilises

`tau` was never set -> SB3 default 0.005 -> target time constant `1/tau` = 200
gradient steps, against the `1/(1-gamma)` = 1000-step bootstrap horizon at
gamma=0.999. SB3's default is calibrated for its own gamma=0.99 (horizon 100),
where the target is 2x SLOWER — the stable ordering. At our gamma that
ordering inverts. Independent of the entropy runaway; pinning alpha does not
address it. Added `--tau`, default **0.001**, forced on the warm-start path.

### 3. The braking term is anti-correlated with success

Measured on all 48 demo episodes (20 landed / 28 failed): successful landings
pay **1.40x** what failures pay (23121 vs 16527 raw). Cost/step landed vs
failed by band: 60-120 m 36.3/27.0, 30-60 m 69.9/42.8, 10-30 m 97.6/56.4.
A successful descent is simply FASTER through the mid-band (mean vz -5.24 vs
-4.52 at 10-30 m) and `ratio = vz^2/(2*alt)` charges exactly that.

Not retunable: sweeping `braking_authority_margin` 0.25->0.85 and `t0` 20->8
leaves the landed/failed ratio at 1.40 throughout — any monotone transform
hits both groups. Only silenceable in the known-good regime:
`braking_authority_margin` **0.25 -> 0.70** puts the proven winning flight's
peak ratio at 0.96 (just under the knee, was 2.35 — fully on during a textbook
landing), cutting the term 87%.

### 4. `leg_diff` was tested but never graded

`landed_safely` checks five criteria; `landing_margins` carried four.
`_terminal_reward` averages those margins, so a touchdown rejected purely on
footpad flatness reported margins that all looked comfortable — the "CRASH
with every printed number inside its limit" case `diag_stage_landing_rate.py`'s
own comment describes. Added to both envs.

### Also changed

- `--ent-coef` default 0.01 -> **0.05**. At 0.01 the Q term outweighed entropy
  ~1e4:1, i.e. sigma->0 and SAC degenerates to TD3 with no exploration noise —
  the wrong side to err on here. 0.05 puts entropy at ~9% of the measured
  ~1074-unit Q spread, against the 16x-of-spread the auto runaway reached.
- Demo seeding dropped the array tail to fill whole `n_envs` chunks, and that
  tail is always a TERMINAL (log said "47 terminal per repeat" for 48
  episodes). Now drops from the front. Verified: "48 terminal transitions per
  repeat".
- `--demo-repeat` 6 -> 3. At 6, demos are 29.9% of the buffer and each of the
  35521 unique demo transitions is drawn ~1077 times over 500k gradient steps.

### Run `out/train_cal_v2.log` — all of the above, gamma still 0.999

critic_loss against the diverged v1 run at matched steps:

```
  ~100k:  v1 7.99e+04   v2 276        (289x)
  ~200k:  v1 3.74e+04   v2 552        (68x)
  ~300k:  v1 6.91e+06   v2 1.89e+03   (3656x)
  ~370k:  v1 2.60e+06   v2 3.75e+03   (693x)
```

ent_coef pinned at 0.05, no runaway. But critic_loss still climbs ~1 order per
150k steps (72 -> 150 -> 550 -> 1.4e3 -> 3.2e3) and actor_loss falls
monotonically to -1100, i.e. Q ~ +1100 against a realistic ceiling of +199.
Overestimation, slower than v1 but the same direction.

### MEASURED: gamma can come down to 0.998

Discounted return over the 48 real demo episodes plus the stall archetype:

| gamma | horizon | worst landing | best failure | separation | stall@20m | landing's margin over stall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.990 | 100 | -10.7 | -8.8 | **-1.9** | -2.8 | **-7.9** stall wins |
| 0.995 | 200 | -13.5 | -17.2 | +3.6 | -5.9 | **-7.6** stall wins |
| 0.997 | 333 | +1.5 | -29.2 | +30.7 | -12.0 | +13.6 |
| **0.998** | **500** | **+30.8** | **-43.0** | **+73.8** | -22.4 | **+53.2** |
| 0.999 | 1000 | +102.8 | -69.6 | +172.4 | -51.4 | +154.2 |

0.99 and 0.995 are genuinely out — this file's original finding stands, stall
beats landing there. But **0.998 halves the horizon** (the thing compounding
the overestimation above) and still keeps landing ahead of stalling by +53,
with the worst landing (+30.8) clear of the best failure (-43.0). It also puts
the already-set `tau=0.001` (time constant 1000) exactly 2x slower than the
500-step horizon — the same ratio SB3's own defaults have at gamma=0.99, and
the ordering that was inverted at 0.999.

**Next run to try, not yet run:** identical to the v2 command but
`--gamma 0.998`. v2 is its control: only gamma differs.

### Deferred, with numbers

Potential-based shaping. The shaping currently carries almost no outcome
information: mean discounted shaping landed -38.1 vs failed -47.2, i.e. **9.2
units of discrimination inside a 176.6-unit gap** — 95% of the signal is the
terminal, and the critic is asked to solve 700-step credit assignment from the
other 5%. `_altitude_penalty` is 70-80% of all shaping and barely separates
(94125 vs 101808, ratio 0.92); corr(episode length, shaping total) = **-0.927**,
so it is measuring duration, not quality. `_angular_rate_penalty` is
numerically dead in the operating regime (|w|=0.1 -> 0.00016/step; raising
`omega_k` 60->400 changed measured separation by 0.0).

Measured PBS proposal, Phi = 60*max(3 - D(s), 0) with
D = sqrt(alt/200) + 0.05*(vxy/lim + |vz|/lim + tilt/lim + |w|/lim):

| | current | PBS |
|---|---:|---:|
| landed | [102.9, 150.4] | [216.9, 243.5] |
| failed | [-87.5, -73.7] | [-53.6, -23.4] |
| gap | 176.6 | **270** |
| within-landed spread | 47.5 | **26.6** |

Deferred because that Phi is an untuned first proposal and the measured
corr(steps, PBS) = +0.807 is confounded with outcome. The agent's other
deferred proposal: a BC-regularised actor loss (TD3+BC style,
`actor_loss += lambda*||a - a_demo||^2` on demo samples), on the grounds that
buffer-seeded SACfD only teaches the CRITIC that the landing region is
valuable while the actor is pulled only by grad-Q, which is unreliable where
the actor never visits.

### v2 RESULT: 0/16, and worse than every archetype

```
=== SANITY orbit_descent: landed_safely=0/16 lost_control=7 left_tile=2 other=7 mean_reward=-179.5 ===
checkpoint: out/sac_training_run_cal_v2/sac_lunar_lander_isaac_orbit_descent.zip
```

Against the archetypes this reward was calibrated on:

| | discounted return |
|---|---:|
| controller landing | +14.86 |
| stall @20 m | -51.45 |
| free-fall crash | -110.34 |
| **this policy (mean)** | **-179.5** |

The policy scores below doing nothing and below falling out of the sky. The
reward ordering is not the explanation — it was verified correct on 48 real
episodes (0/560 landed-vs-failed pairs inverted). Sharper: 30% of its replay
buffer was demos, 20 of those episodes are real landings, and even the FAILED
demos score -91…-75. **The policy ended up worse than the worst demonstration
in its own buffer.** An actor following a sound critic should at minimum
regress toward demo-like behaviour.

That moves the blame off exploration and off the reward, onto value learning:

```
adim      critic_loss   actor_loss
   12288         72.7        -4.88
  198704          552         -348
  365776     2.28e+03         -955
  484448     7.48e+03        -1630
  499168          999        -1770      <- Q ~ +1770
```

Realistic ceiling is +199 (a perfect landing). The critic inflated Q **9x**;
the actor is climbing a fictitious value landscape, which is also why the
failure mode moved from passive stalling to 7/16 active loss of control.

So the four defects fixed above were real and bought 2-3 orders of magnitude
over v1, but they were not the binding constraint. The binding constraint is
overestimation compounding over the 1000-step horizon at gamma=0.999 — exactly
what the gamma table above says can be halved at 0.998 while keeping landing
ahead of stalling by +53.

Two levers, in order, neither yet run:
1. `--gamma 0.998` (v2 is the control: only gamma differs).
2. BC-regularised actor loss on demo samples — pulls the actor toward
   demonstrated actions directly instead of through a grad-Q it currently
   cannot trust.

## 2026-10-05: the hover attractor was the ACTION SPACE, not the reward

Measured, and it reframes most of this file. The old mapping was
`throttle = (action[0] + 1) / 2`, thrust linear in throttle:

```
action[0] = 0  (box centre, and an untrained SAC policy's mean)
    -> throttle 0.50 -> 24856 N against 24467 N weight -> T/W = 1.0159
exact hover:                       action[0] = -0.019
|net accel| <= 0.05 m/s^2 band:    action[0] in [-0.057, +0.018]  = 3.7% OF THE BOX
ZemZev (which lands) flies:        throttle 0.46-0.52 = action[0] in [-0.080, +0.040]
```

The centre of the action box is a near-perfect hover, and the band the vehicle
must hold to descend under control is 3.7% wide while an exploring policy's
noise spans the whole box. "The policy always hovers" has been blamed on the
reward for most of this project; this is the mechanism, and it is upstream of
the reward entirely.

### Fix: `lunarsim/rl/action_map.py`, the ONE definition

`throttle = 0.5 + 0.5*(e*a0^3 + (1-e)*a0)` with `e = THROTTLE_EXPO = 0.8` —
the linear/cubic blend RC pilots call expo — plus its exact inverse
`throttle_to_action` (Cardano; the curve is monotone so there is one real
root; round-trip error 5e-15).

```
expo e   fine band   d(throttle)/d(a0) at centre
  0.0       3.7%       0.500   <- the old linear mapping
  0.8      16.0%       0.100   <- chosen
  1.0      32.3%       0.000   <- pure cubic: widest, but the centre
                                  derivative vanishes, so small actions stop
                                  moving the vehicle at all
```

Endpoints are untouched (a0=0 -> throttle 0.5, a0=+-1 -> 1/0), so the
REACHABLE thrust set is unchanged (+1.36 / -1.31 m/s^2 at PDI mass). Only
resolution is redistributed: 4.3x more where control actually happens.

**Validated before use:** ZemZev through the real Isaac env, 24 episodes,
**11/24 (46%) landed_safely**, 0 lost_control, 0 left_tile — against 20/48
(42%) on the linear mapping. Controllability is preserved; the curve only
changes how the action box is spent.

The forward mapping had been hand-written in FIVE places and the inverse in a
sixth, with a seventh copy in `tests/test_rl_env.py`. All now call the module.
That test is how the change was caught being load-bearing: its hand-rolled
`2*t - 1` asked for throttle 0.4951 instead of 0.4757 under the new curve,
close enough to hover that the vehicle no longer reached the ground inside the
episode. Same copy-paste shape `lunarsim/rl/curriculum.py` was created to end.

### Run v3: no demos at all, per user direction

"buffer'a bir şey koyma, kendi bulsun" — the v2 post-mortem supports it: demos
come from ONE terrain realization (the single env has no `tile_fn`) while
training regenerates per reset, and at `--demo-repeat 3` each of 35521 demo
transitions is drawn ~540 times. Heavily-repeated off-distribution data is a
textbook driver of extrapolation error, which is what Q -> +1770 against a
+199 ceiling looks like. Also: no pure-exploration run has EVER been a clean
test — every historical 0/16 had at least one fatal bug (gamma=0.99,
undetectable landings, gradient_steps=1, rocks).

```
scripts/train_sac_isaac.py --only-stage orbit_descent --steps-per-stage 500000 \
  --out-dir out/sac_training_run_v3_expo_nodemo --gamma 0.998 --max-rocks-per-env 0
```

gamma 0.998 chosen off the sweep table above: halves the bootstrap horizon
(1000 -> 500 steps, the thing compounding the overestimation) while keeping
landing +53 ahead of stalling, and it puts `tau=0.001` exactly 2x slower than
the horizon — SB3's own stable ratio, which was inverted at 0.999.

### v3 result: 0/16, and the failure mode is new — it flies away sideways

```
=== SANITY orbit_descent: landed_safely=0/16 lost_control=2 left_tile=13 other=1 mean_reward=-202.6 ===
=== SANITY orbit_descent: landed_safely=0/16 lost_control=7 left_tile=6  other=3 mean_reward=-209.3 ===
```

**But the learning itself is finally healthy.** Against v2 at matched steps:

| step | v2 critic | v3 critic | v2 Q | v3 Q |
|---:|---:|---:|---:|---:|
| 100k | 276 | **14.8** | 176 | **7** |
| 200k | 552 | **40.5** | 348 | **28** |
| 300k | 1.89e3 | **176** | 711 | **64** |
| 360k | 4.24e3 | **15.4** | 915 | **86** |

critic_loss 20-275x lower and oscillating rather than climbing (74.7 -> 3.99 ->
15.4), Q=86 inside the +199 realistic ceiling where v2 was at 915. Dropping the
demos and shortening the horizon did end the overestimation.

Telemetry of the checkpoint (`out/diag_v3.log`, 10 episodes) shows one
signature:

```
[ep 2] LOST_CONTROL  alt 222 m  vz +2.1  v_xy 32.6 -> 33.0 RISING  tilt 59.6->60.1  roll_cmd +0.98
[ep 9] LEFT_TILE     alt 360 m  vz +1.2  v_xy 42.0 -> 42.3 RISING  tilt 51.9        roll_cmd +1.00
[ep 1] CRASH         alt   2 m  vz -11   v_xy 36.5                 tilt 42.1        roll_cmd +1.00
```

roll_cmd pinned at the box corner, tilt held at 45-60 deg, v_xy driven ABOVE the
10-30 m/s release range, altitude rising. Not "fails to brake" — actively
accelerating sideways off a 1680 m tile.

Two hypotheses were tested against this and BOTH were refuted by measurement,
which is worth recording so they are not re-tried:

1. *"Leaving the tile is a cheap early exit."* No: `left_tile` truncates with
   `-timeout_penalty`, and an EARLIER exit discounts that penalty LESS. At
   0.10/step shaping, exiting at 20 s totals -74.7 against -55.0 for riding the
   full 60 s. Leaving early is worse, not cheaper.
2. *"The attitude gate rewards keeping speed."* The gate
   (`urgency = clip(1 - v_xy/12)`) really is blind to tilt DIRECTION, so in
   principle a policy can keep its own tilt cheap by accelerating. But the term
   it scales is numerically dead: `quad = mu*tilt^2` in RADIANS is 0.0001/step
   at mu_base and 52 deg, against a `tilt_cutoff` barrier of 0.159/step at the
   same point. A direction-aware gate was implemented, measured to move the
   exploit state from 0.159 to 0.160/step, and REVERTED. See the note left in
   `_attitude_hold_penalty`.

### The reward is thinner than it looks

Per-term decomposition of the winning controller flight under CURRENT weights
(scaled, undiscounted):

```
altitude      86.10   <- 94% of all shaping
vel_track      4.89
braking_xy     0.71
braking_z / attitude / ang_rate / throttle / proximity / xy_pos   <= 0.02 each
TOTAL         91.74        against landing_bonus = 450
```

Everything except altitude is numerically dead. In particular the terms that
should say "kill your horizontal speed" are worth **5.60 units = 1.2% of the
landing bonus**, after `x1 -> 0`, `t0_xy 20 -> 2` and
`braking_authority_margin 0.25 -> 0.70` compounded to ~45x. That is the same
compounding mistake made earlier with `tilt_cutoff` (5x cut x 4x scale = 20x),
and it is the likeliest reason nothing opposes the sideways flight.

Note also, measured: at gamma=0.998 the ranking holds on the REAL demo
population (median 37 s: worst landing +30.8, stall -22.4, margin +53.2) but
INVERTS on a 94.5 s flight (landing -25.5 vs stall -22.4), because
0.998^1891 = 0.023 leaves a 450 bonus worth 10. The reward's ordering is
sensitive to episode length; long flights lose their bonus.

### v4: does it land on the EASIEST stage at all?

Before tuning orbit_descent further, establish learnability.
`--only-stage ramp_20m` (20 m, 0-2 m/s, 20 s episodes). Decisive: if the policy
cannot land there, nothing about orbit_descent matters. It is also structurally
easier to learn from -- 400-step episodes mean 500k steps is ~20000 terminal
events against orbit_descent's ~660, so the landing bonus is sampled ~30x more
often. handover records a PRE-fix checkpoint already reaching 2/16 there.

### v4 (ramp_20m, old reward): 0/16 — and it explains everything

```
=== SANITY ramp_20m: landed_safely=0/16 lost_control=0 left_tile=6 other=10 mean_reward=-117.9 ===
```

Even the easiest stage. The per-episode detail (`out/diag_v4_ramp20.log`) shows
two degenerate modes and no third:

```
3 touchdowns:  vz = -7.60, -7.57, -3.72   (limit 1.0)  margins v_z=0 v_xy=0 tilt=0
7 timeouts:    final alt 23-33 m (spawn is 20 m!)      vz = +0.14 .. +2.45, i.e. CLIMBING
6 left_tile:   alt 0.8-4.2 m, vz -6.6..-7.4            diving, drifts off a 60 m tile
```

Free fall from 20 m arrives at 8.05 m/s; the policy arrives at 6.6-7.6. It is
dropping, not descending. And the touchdowns are not near misses.

**Why, exactly.** The reward says landing beats diving by ~570 units, so the
ranking was never the problem. But the policy has never once landed, so the
+450 has never entered the replay buffer. In the MDP it has actually
EXPERIENCED there are only two outcomes, and these were their values:

```
free-fall crash   -104.5      <- margins {v_z=0.00, v_xy=0.00, tilt=0.00, w=0.92, leg_diff=0.40}
hover to timeout  -105.0
```

Crashing was *cheaper than doing nothing*. The policy found the optimum of the
reward it could see. Cause: the penalty was graded by the MEAN of five margins,
each clipped to [0,1], so the two criteria that are trivially satisfied while
falling straight down (you are not rotating; the ground happens to be flat)
masked a 7.6x velocity overshoot. Total spread from "20% too fast" to "free
fall" was 19.5 units.

### The reward rewrite (2026-10-05)

User's framing, which matches the measurements exactly: *"rewards aren't
guiding enough, there are too many distractions, crashing straight down must
not pay, it should want to slow down and land."*

1. **Crash graded by OVERSHOOT, not by mean margin.** New
   `_touchdown_severity`: `max` over criteria of value/limit, computed from
   the STATE (margins are clipped and cannot express magnitude). Multiplier
   `crash_severity_base + crash_severity_k*(severity-1)`, capped. Max not
   mean, because one catastrophic axis is a catastrophe however good the
   others look.

   | | severity | old | new |
   |---|---:|---:|---:|
   | free fall (measured) | 7.60 | -104.5 | **-360.0** |
   | hard arrival | 2.50 | -93.3 | -141.0 |
   | near miss | 1.20 | -85.0 | **-70.8** |
   | tumble (lost_control) | – | -120.0 | -360.0 |
   | hover to timeout | – | -105.0 | -105.0 |
   | safe landing | – | +450 | +450 |

   Resulting order: **landing > near miss > hovering > hard arrival > dive =
   tumble**. Attempting and nearly making it still beats never trying (that
   requirement is from an earlier session and still holds); diving is now
   3.4x worse than hovering instead of equal to it.

2. **Measured-dead terms zeroed**: `alpha` (proximity), `s` (throttle
   effort), `mu_base`/`mu_scaled` (the attitude quadratic). Each was <= 0.02
   over an entire 94 s winning flight against a 450 bonus. Tilt is still
   enforced in flight by `tilt_cutoff` (1000x larger wherever it matters) and
   at touchdown by the severity grading.

   `_angular_rate_penalty` was zeroed too and then **RESTORED**. The audit
   measured it as non-discriminating — but on 48 demo episodes flown by a
   controller whose |w| never leaves 0.003. A population where nothing spins
   cannot show whether a spin penalty matters, and this file already records
   |w| as the BINDING margin on two of the three closest touchdowns ever
   measured. Two regression tests assert its purpose; they caught the
   deletion.

End-to-end episode returns under the new reward (gamma=0.998):

```
ramp_20m:       soft landing (1 m/s) +197.1 | hover -55.0 | engine-off dive -299.1
orbit_descent:  soft landing          -41.5 | hover -50.2 | engine-off dive -208.9
```

(The orbit_descent "soft landing" archetype is a 1 m/s descent from 200 m,
which takes 200 s against a 60 s cap, so that row understates it; the real
controller profile descends at -3.9 .. -0.8.)

Run v5: `--only-stage ramp_20m --gamma 0.998 --max-rocks-per-env 0`.

## 2026-10-05 (evening): IT LANDS. The wall was one line of stage config.

```
=== SANITY ramp_20m: landed_safely=3/16 lost_control=0 left_tile=0 mean_reward=+65.5 ===
=== SANITY ramp_20m: landed_safely=1/16 lost_control=0 left_tile=0 mean_reward=+14.1 ===
```

First non-zero result of the session, and the first POSITIVE mean episode
reward this project has recorded (the previous seven runs sat at -116 to -209).

### What it actually was

Seven runs were spent fixing real, measured defects -- rock/contact parity,
`tau` vs the discount horizon, the braking term being anti-correlated with
success, `leg_diff` never being graded, the action box giving 3.7% control
resolution, the tile being too small for an episode's drift, and finally a
from-scratch reward rewrite. Every one of those was a genuine bug and every
one was verified. **None of them was the binding constraint.**

The binding constraint was `ramp_20m`'s `spawn_v_z_m_s = 0.0`. Measured with
a RANDOM policy, 40 episodes each:

```
spawn_v_z =  0.0  ->  touchdowns  0/40,  timeouts 40/40
spawn_v_z = -2.0  ->  touchdowns 40/40,  timeouts  0/40
                      severity: median 1.67x, best 1.16x, binding criterion v_z in 40/40
```

This vehicle's thrust-to-weight at the centre of the action box is **1.0159**.
"Do nothing" -- which is exactly the mean action of an untrained policy -- is
a hover. Released at rest, the vehicle therefore never reaches the ground, the
terminal reward is never sampled, and no reward function can teach what the
agent never experiences. Released already descending, every episode ends in a
touchdown and a random policy is already within 1.16x of the limit, so the
success region sits immediately adjacent to random behaviour and
`crash_severity` grades every m/s shaved off the way in.

The deleted `hover_only_easy` stage used `spawn_v_z_m_s = -2.0`. This file
records it as "presumed solved". That is why.

**Method note worth keeping:** this was found by asking what a RANDOM policy
can do, not what the trained one does. Seven runs of trajectory diagnostics
never surfaced it, because they all answered "why is this policy bad" when the
question was "is the goal reachable at all". Run the random-policy probe first
on any new stage.

### Then: the vertical channel is solved, the gap is lateral

24-episode diagnostic of the v8 checkpoint -- 2/24 landed (8%), 0 lost_control,
and **10 of the 12 touchdowns were rejected on v_xy and nothing else**:

```
[ep 19] CRASH  vz=-0.11  vxy=1.50  tilt=4.5  w=0.03   margins{v_z=0.90, v_xy=0.00, ...} worst=v_xy
[ep  8] CRASH  vz=-0.21  vxy=1.57  tilt=4.0  w=0.03   margins{v_z=0.79, v_xy=0.00, ...} worst=v_xy
[ep 13] LANDED vz=-0.67  vxy=1.14
[ep 22] LANDED vz=-0.10  vxy=1.16
```

v_z margins of 0.33-0.90 mean the descent profile is genuinely learned. The
entire remaining gap is 0.3-0.5 m/s of lateral speed at contact.

Cause, measured: at `kxy=12` the per-step difference between arriving at 1.7
and at 1.1 m/s was **0.005**, against ~1.0/step for the descent progress
reward -- "go down" outweighed "bleed off lateral" by 200x, so the policy
correctly ignored the latter. Raised to 400 (0.17/step difference, same order
as the progress term) with `kxy_cap` 20000 so it does not saturate at the
later stages' release speeds.

Archetypes after the change, ramp_20m:

```
  landing, lateral cut to 1.1   +322.2
  touchdown, 1.6 left            -11.5      <- the 334-unit gap to be learned
  hover                          -88.6
  engine-off dive               -289.1
```

v9 is running, warm-started from v8 (the vertical channel is worth keeping).

### What to do with the other stages

`spawn_v_z_m_s = 0.0` is still set on ramp_50m / ramp_100m / ramp_150m /
orbit_descent, and the tile/episode-drift mismatch recorded above is still
unfixed on all four (ramp_50m has 90.3 m of usable lateral room against 240 m
of drift over a full episode). Both will bite exactly as they did here. Run
the random-policy probe on each stage before training it.

### ramp_50m: the two shaping terms scale in opposite directions

First attempt at ramp_50m (warm-started from the solved ramp_20m) regressed to
0/16 with mean_reward -325/-377. 11/12 episodes timed out, one of them
**hovering 0.20 m off the ground with 3.78 m/s of lateral still on**.

That last detail is the reward being RIGHT: touching down at 3.78 m/s is
severity ~3.1 and scores -173, against -105 for timing out. Refusing a bad
touchdown is correct. The vehicle simply could not bleed the lateral speed in
time, because the reward had told it to prioritise that over everything.

Measured ratio of lateral penalty to descent-progress reward, per step:

```
ramp_20m (release 0-2 m/s):   0.2x at 20 m,   0.6x at 1 m    balanced
ramp_50m (release 2-8 m/s):   2.8x at 50 m,  16.3x at 1 m    swamped
```

The terms scale in OPPOSITE directions across the curriculum: the lateral
penalty is quadratic in a release speed that grows stage to stage, while the
progress reward is normalised by `h0` and therefore shrinks. Any kxy tuned on
one stage is wrong on the next. Fixed by pulling the ground weight in,
`kxy_alt_ref_m` 20 -> 4 m, which concentrates the charge where lateral speed
actually breaks legs:

```
ramp_50m after:  0.7x at 50 m,  1.4x at 20 m,  5.3x at 5 m,  13.7x at 1 m
ramp_20m after:  0.1x at 20 m,  0.5x at 1 m    (unchanged in character)
```

### MEASURED: the last two stages' episode budgets are too short

Time needed = descent at the envelope, plus killing the release lateral speed
at the ~20 deg tilt the policy actually uses, with partial overlap:

| stage | descent s | lateral bleed s | needed | budget | ok? |
|---|---:|---:|---:|---:|:--|
| ramp_20m | 11.5 | 3.4 | 12.7 | 20 | yes |
| ramp_50m | 18.1 | 13.6 | 22.9 | 30 | yes |
| ramp_100m | 25.6 | 27.1 | 36.1 | 40 | yes |
| **ramp_150m** | 31.4 | 40.7 | **51.7** | 50 | **no** |
| **orbit_descent** | 36.3 | 50.9 | **63.6** | 60 | **no** |

Not yet fixed, because raising them interacts with gamma: at 0.998 an 85 s
orbit_descent episode (1700 steps) discounts the 450 bonus to 15 units at the
release point. Lengthening those budgets means revisiting gamma for those
stages (0.999 gives 82 units over the same horizon). Decide the pair together,
the way `profile_c` and the budgets are already coupled.

### Do NOT continue training a stage. Measured, three runs.

All three evaluated on the SAME 24 episodes (`--seed0 31000`), ramp_20m_fast:

```
v12  ent_coef 0.05, first 500k on the stage   10/24 = 42%   Q=150
v13  ent_coef 0.05, +500k more                 2/24 =  8%   Q=211   (18/24 timeout)
v14  ent_coef 0.01, +500k from v12            4/24 = 17%   Q=226
```

Continuing training on a stage degrades it, and `ent_coef` is not the cause --
annealing it to 0.01 for the consolidation run recovered some of v13's loss but
still landed well below the checkpoint it started from. In all three Q climbs
monotonically (150 -> 211 -> 226) while real performance falls: off-policy
overestimation accumulating over the 500-step horizon gamma=0.998 implies,
with the actor then exploiting its own critic's error back into the stall
attractor (18/24 and 15/24 timeouts).

A hypothesis that was tested and REFUTED, so it is not re-tried: "ent_coef is
pinned at 0.05 and never anneals, so a permanently stochastic policy caps the
precision reachable in the measured 16%-wide control band." Plausible, and
wrong -- v14 is the experiment.

**Recipe: 500k per stage, keep that checkpoint, move on.** The best checkpoint
for a stage is the first run on it, not the longest.

Best per stage so far:
```
ramp_20m        out/sac_training_run_v9_kxy/   20/24 = 83%
ramp_20m_fast   out/sac_training_run_v12_fast/ 10/24 = 42%
```

### The tile/drift mismatch is a per-stage blocker, and the probe is how you find it

`ramp_50m` stayed at 0/16 from three different warm starts. The random-policy
probe (the method that cracked ramp_20m, and which I failed to run here before
training it three times) said why immediately:

```
RANDOM / ramp_50m:  land 0,  touchdown 2,  timeout 16,  LEFT_TILE 22   (of 40)
```

55% of episodes end by leaving the map. Same blocker ramp_20m had. Audit of
every stage, usable lateral room (half-width - footpad margin - spawn radius)
against drift over a full episode at the release speed:

```
stage            tile   usable   episode drift   ok?
ramp_20m          120     45.3            40.0   yes
ramp_20m_fast     120     45.3           100.0   NO   <- a stage I added, with
                                                         the same defect baked in
ramp_50m          240     90.3           240.0   NO
ramp_100m         640    265.3           640.0   NO
ramp_150m        1200    530.3          1200.0   NO
orbit_descent    1680    755.3          1800.0   NO
```

Fixed: ramp_20m_fast 120 -> 260, ramp_50m 240 -> 600. The 42% measured on
ramp_20m_fast was scored under the handicap, so it is being re-run clean.

The last three are NOT fixed, and should not be fixed blindly: covering
orbit_descent's worst case needs a 3800 m tile, which at `--terrain-grid-n 80`
is 47 m cells -- effectively flat ground. These numbers are the worst case for
a policy that never brakes; the ZemZev demos leave no tile at all (0/48). So
**probe each stage before training it** and enlarge only if the incoming policy
actually leaves. That is cheaper than guessing and it is what this section is.

**Standing method note, now twice validated:** before training a new stage, run
the random-policy probe. Both times it answered in four minutes what multiple
50-minute training runs could not.

### Current state (2026-10-06 02:15)

Best checkpoint per stage, all measured on 24 episodes with `--seed0 31000`:

```
ramp_20m        out/sac_training_run_v9_kxy/            20/24 = 83%
ramp_20m_fast   out/sac_training_run_v16_fast_bigtile/  11/24 = 46%
ramp_50m        -- 0%, tile just enlarged 240 -> 600, v17 testing now
```

ramp_20m_fast went 42% (small tile) -> 46% (260 m tile). That is within noise
on n=24; the tile fix removed a real defect but this stage is simply harder
than ramp_20m, not handicapped into failure the way ramp_50m was.

Note on measurement: the two sanity rolls inside a single run routinely
disagree by a lot (v16 gave 1/16 then 8/16; v12 gave 4/16 then 5/16). Terrain
is regenerated per env per reset, so 16 episodes is a noisy estimate. Use
`diag_stage_landing_rate.py --episodes 24 --seed0 31000` for any comparison
between checkpoints -- same seeds, same terrain draws, or the numbers are not
comparable.

### Settled recipe

```
--gamma 0.998 --ent-coef 0.05 --tau 0.001 --max-rocks-per-env 0
--steps-per-stage 500000, warm-started from the previous stage's best
```
- 500k per stage, then MOVE ON. Continuing a stage degrades it (42 -> 8 -> 17%).
- Probe a new stage with a random policy before training it.
- Check `usable lateral room > episode drift` before training it.

### The bootstrap descent velocity has to RISE with the stage, not fall

`ramp_50m` stayed at 0/16 through four warm starts and a tile fix. The probe,
re-run after the tile was enlarged, isolated it:

```
vz0 = -1.5:   land 0,  touchdown  0,  timeout 40,  left_tile 0   median final alt 18.7 m
vz0 = -3.0:   land 0,  touchdown 40,  timeout  0,  left_tile 0   severity median 3.95x, min 2.33x
```

The first version of the bootstrap velocity TAPERED down as the curriculum
climbed (-2.0, -1.5, -1.0, -0.5, 0.0), reasoning that later stages inherit a
policy that already descends. That is backwards. Reachability needs at least
`h0 / max_episode_s` of average descent, and h0 grows faster than the budget,
so the requirement RISES while I was lowering the help:

```
stage          h0/T needed    was      now    envelope at h0
ramp_20m          1.00       -2.0     -2.0        3.49
ramp_50m          1.67       -1.5     -3.0        5.52
ramp_100m         2.50       -1.0     -4.0        7.80
ramp_150m         3.00       -0.5     -5.0        9.55
orbit_descent     3.33        0.0      0.0       11.03
```

Every value is inside the descent envelope at its own release altitude, so the
episode does not begin in penalty. orbit_descent stays at 0.0 deliberately --
it is the real scenario, an orbital release, and by then the policy has to
initiate its own descent.

**This is the third time the same class of defect has been the blocker**:
ramp_20m (vz0=0, goal unreachable), ramp_50m tile (55% flew off the map),
ramp_50m vz0 (goal unreachable again). All three were invisible to training
diagnostics and obvious to a four-minute random-policy probe. The probe is not
optional.

### ramp_35m: the rules now produce a working stage on the first try

`ramp_50m` reached 2/16 and 1/16 once its reachability was fixed -- first
landings there, but weak, and mean_reward -320/-390 (most episodes still
arrive hard). Cause is the familiar one: ramp_20m_fast -> ramp_50m moves
altitude 2.5x AND release speed 1.6x together.

Inserted `ramp_35m`, altitude alone (20 -> 35 m) at ramp_20m_fast's own
2-5 m/s. Every number derived from the three rules rather than picked:

```
budget 25 s      -> needs h0/T = 1.40 m/s average descent
vz0 = -2.8       =  2x that, and inside the 4.61 m/s envelope at 35 m
drift 5*25 = 125 -> tile >= 2*(125 + 4.7 + 15) = 289  -> 320 m
```

Probe, first try: **touchdown 40/40, timeout 0, left_tile 0**, severity median
2.88x / min 1.25x. That sits exactly between ramp_20m (1.67x / 1.16x) and
ramp_50m (3.95x / 2.33x), which is what an intermediate stage should look like.

Stage ladder as it now stands, with the random-policy severity as a difficulty
measure:

```
stage           h0   lateral   vz0    tile    random severity (median/min)
ramp_20m        20   0-2      -2.0     120    1.67 / 1.16     trained 83%
ramp_20m_fast   20   2-5      -2.0     260    --              trained 46%
ramp_35m        35   2-5      -2.8     320    2.88 / 1.25     v19 training
ramp_50m        50   2-8      -3.0     600    3.95 / 2.33     trained 10%
ramp_100m      100   6-16     -4.0     640    -- probe first, tile too small
ramp_150m      150  10-24     -5.0    1200    -- probe first, tile too small
orbit_descent  200  10-30      0.0    1680    -- probe first, tile too small
```

## 2026-10-06 05:30: every stage number in this file was measured at the WORST point of its run

`--checkpoint-every` was added to test one hypothesis: that performance peaks
INSIDE a 500k stage run and the script, which saves only at stage end, was
systematically keeping the over-trained tail. Five snapshots from one
ramp_35m run, all evaluated on the same 24 episodes (`--seed0 31000`):

```
step        landed_safely
1,600,000      2/24 =  8%
1,700,000      4/24 = 17%   <- peak, 200k into the run
1,800,000      1/24 =  4%
1,900,000      0/24 =  0%
2,000,000      0/24 =  0%   <- the point the script saves, and what was reported
```

Confirmed, and worse than expected: the curve peaks at ~200k and collapses to
**zero** by 400k. The saved endpoint is the worst point sampled.

**Consequences.**
- Every per-stage figure recorded above (ramp_20m_fast 46%, ramp_35m 6%,
  ramp_50m 10%) is a tail measurement, not a peak. The real peaks are higher
  and unknown, because no snapshots exist for those runs.
- The rule written earlier in this file -- "do not continue training a stage,
  it degrades (42 -> 8 -> 17%)" -- was not a rule about stage boundaries at
  all. It was this same collapse, observed across runs instead of within one.
- ramp_20m's 83% (v9) may also be below its own peak.

**Root cause is still the overestimation**, not the saving policy: Q rose
monotonically 150 -> 211 -> 226 across the three ramp_20m_fast runs while real
performance fell 42 -> 8 -> 17%. Snapshotting is a mitigation; the collapse
itself is off-policy Q growth over the 500-step horizon gamma=0.998 implies.
Candidates not yet tried, in order: a lower replay ratio
(`--gradient-steps` below -1), `--gamma 0.997` (shorter horizon, and the sweep
table above shows landing still wins by +13.6 there), or SB3's
`target_entropy`/n-critics options.

**Revised recipe.**
```
--steps-per-stage 200000 --checkpoint-every 50000
then evaluate every snapshot with
  diag_stage_landing_rate.py --episodes 24 --seed0 31000
and carry the BEST one forward, not the last.
```
Shorter, lands near the peak, and 2.5x faster per stage.

### Snapshot-and-keep-best works, and the instability is large

Re-ran ramp_35m at the revised recipe (200k steps, snapshots every 50k),
same warm start, all four snapshots on the same 24 episodes:

```
1,550,000   2/24 =  8%
1,600,000   1/24 =  4%
1,650,000   5/24 = 21%   <- best
1,700,000   1/24 =  4%   <- endpoint, i.e. what would have been saved
```

So keeping the best snapshot took this stage from the **6%** originally
reported to **21%**. The recipe change is real and worth keeping.

But the curve does not peak and decay cleanly -- it oscillates (8, 4, 21, 4).
At n=24 the gap between 1 and 5 successes is ~2.5 sd, so it is marginally
significant rather than noise, and the honest reading is that the policy swings
between roughly 4% and 21% during training. Picking the best snapshot harvests
the top of the swing; it does not make the swing go away.

### Best checkpoint per stage (2026-10-06 06:00)

```
stage           rate   checkpoint                                          note
ramp_20m         83%   out/sac_training_run_v9_kxy/                        endpoint, no snapshots -- may be below its peak
ramp_20m_fast    46%   out/sac_training_run_v16_fast_bigtile/              endpoint, same caveat
ramp_35m         21%   .../v21_ramp35_short/snapshots/ramp_35m_1650000_steps.zip
ramp_50m         10%   out/sac_training_run_v18_ramp50_reach/              endpoint, same caveat
```

Re-running the first two with snapshots is cheap and would likely raise both.

### Next: attack the instability itself

Everything above is mitigation. The cause is Q growing while performance falls
(150 -> 211 -> 226 across runs; 8/4/21/4% within one). Candidates in order of
cost:

1. **`--gamma 0.997`** -- horizon 500 -> 333 steps, so less compounding, and
   the sweep table in this file shows landing still beats stalling by +13.6
   there. v22 is running this now as a direct A/B against v21's 21%: same
   stage, same warm start, same budget, only gamma differs.
2. Lower the replay ratio (`--gradient-steps` between 1 and -1). The 1:1 ratio
   was set to fix a real bug (SB3 was training at 1/16), but 1:1 with a
   16-env VecEnv is a lot of updates per unique transition.
3. SB3 SAC takes `policy_kwargs={"n_critics": N}`; more critics reduce the
   max-bias that drives overestimation. Not exposed on the CLI yet.

### notify_telegram.py sent with parse_mode=HTML but only escaped the log tail

A message containing `<- best` came back `HTTP 400 Bad Request`: Telegram
parsed it as an unclosed tag. The `--file` tail path escaped `& < >`, the
`--text` path did not. Since this script exists to relay log and error output
-- which is full of those characters -- the fix is to escape everything that is
not the `<pre>` wrapper the HTML mode is there for. Added `_esc()` and applied
it to both paths.
