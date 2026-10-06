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

### gamma 0.997 does slow the Q growth (v22 vs v21, aligned by step)

Same stage, same warm start, same 200k budget, only gamma differs:

```
step        v21 (g=0.998)   v22 (g=0.997)
1.504M          127             134
1.522M          166             157
1.546M          181             175
1.559M          201             173
```

v21's Q grew 127 -> 329 over its 200k steps (2.6x) -- that is the
overestimation signature in plain sight. v22 tracks ~14% lower at the same
step. Direction is right; whether it converts into landing rate is what the
snapshot evaluation decides.

(v22's first logged critic_loss is 452 against v21's 32.6. That is the
warm-start transient, not instability: reloading a critic trained at g=0.998
and continuing at 0.997 makes every stored value slightly wrong for the new
horizon, so the loss spikes and then recovers -- it was back to 66 within a
few hundred steps.)

### gamma 0.997 does NOT improve landing rate (v22) -- gamma is not the lever

```
snapshot    v21 (g=0.998)   v22 (g=0.997)
1.55M            8%              8%
1.60M            4%              0%
1.65M           21%              4%
1.70M            4%             12%
total         9/96            6/96
```

6 vs 9 successes out of 96 is inside the noise. So the 14% reduction in Q
growth that v22 demonstrably achieved did NOT convert into landing rate, and
the "Q inflation -> poor performance" link I assumed is not confirmed. Do not
retry gamma as the fix.

The mechanism is coherent and worth keeping: lowering gamma sets two effects
against each other. Compounding of the critic's bootstrap shrinks (good), but
the distant landing bonus is also discounted harder relative to near-term
shaping, so STALLING becomes relatively more attractive (bad). v22's timeout
counts (6/12/8/9 of 24) are consistent with the second effect. The earlier
sweep that showed landing still ahead at 0.997 measured a fixed pair of
trajectories; it could not see the policy shifting toward the stall.

### The dominant failure mode is not stalling -- it is missing a criterion at touchdown

From the v22 breakdown, e.g. the 1.55M snapshot: of 24 episodes, 2 landed,
6 timed out, and **16 touched down but failed one of the five
`landed_safely` criteria**. So two thirds of episodes get the vehicle to the
ground and lose on precision, not on willingness to descend. That makes the
binding criterion the thing to measure next, rather than another
hyperparameter.

### Two saturation defects in the lateral term, both measured, both fixed

Per-episode margins on the best ramp_35m checkpoint (21%, v21 1.65M) gave the
first clear picture of HOW it fails. Of 24 episodes: 5 landed, 5 timed out,
**14 touched down and missed a criterion**. The binding criterion is lateral
speed, not vertical:

```
v_xy at or over its limit:  13 of the 14 crashes
v_z  at or over its limit:   7 of the 14
landed  v_xy: 0.35 1.03 1.05 1.12 1.17      (limit ~1.2 m/s)
crashed v_xy: 1.26 1.57 1.59 1.88 1.94 1.95 2.21 2.53 3.15 3.37 4.70 6.50 14.47
```

Most failures are NEAR misses. And the reward was flat in exactly that region,
for two independent reasons:

**1. `kxy_cap` saturated near the ground.** The cap was sized against the
release speeds, where the ground weight is tiny (orbit_descent at 200 m:
weight 0.0196, so the cap bound at 50 m/s -- correct). At ground level the
weight is 1.0 and the same cap bound at `sqrt(20000/400)` = **7.07 m/s**,
inside the operating range. Both low-altitude timeouts lived there:

```
ep13  alt=2.75  v_xy=12.68   (saturation at that altitude: 9.18)
ep15  alt=0.45  v_xy=13.62   (saturation at that altitude: 7.46)
```

Both were skimming the surface at ~13 m/s and refusing to land, with
`d(penalty)/d(v_xy) = 0` -- nothing anywhere told them to slow down.

**2. `shaping_clip_abs` was being filled by the lateral term alone.** Its
comment estimated the worst sum at 7400 including "kxy 1200", a figure from
before kxy was raised 12 -> 400. At 400 the lateral term alone reaches 8100 at
v_xy=4.5 near the ground, so it clipped the whole sum by itself and flattened
the descent-envelope, tilt, omega AND progress gradients at the same time. The
crashes at v_xy 4.70, 6.50 and 14.47 were all in that state.

**The fix** is a knee, not a bigger cap: quadratic below `kxy_knee_m_s=2.0`,
linear above (`knee**2 + 2*knee*(v_xy - knee)`), which is value- and
slope-continuous at the knee. It keeps the magnitude and restores the slope:

```
state                     old       new     d/dv_xy old   new
ep15 skim 13.62@0.45    20000     18150            0.0   1438
ep13 skim 12.68@2.75    20000     11074            0.0    948
ep19 crash 14.47@0.02   20000     21445            0.0   1592
ep18 crash  1.26@0.00     635       635           1008   1008
```

`kxy_cap` 20000 -> 60000 and `shaping_clip_abs` 8000 -> 60000, both now true
safety nets that no stage's release speed reaches.

Verified before training:
  - below the knee the term is **bit-identical** to the old quadratic (54/54
    sampled points over the whole ramp_20m touchdown range), so the proven 83%
    regime is untouched; the multiplication order is deliberately preserved to
    make that exact rather than equal-to-within-rounding
  - the proven ZemZev landing still beats the failure modes by a wide margin
    (+288 discounted, was +311), and its OWN score rose 141.7 -> 151.5,
    because the controller unavoidably carries >2 m/s laterally early in the
    descent and the knee stops over-taxing it
  - three regression tests added for the defect class (gradient present at the
    speeds that actually fail; sub-knee values unchanged; lateral term alone
    cannot fill the clip). 17 reward tests pass, suite at baseline.

v23 is the A/B: same stage, same warm start, gamma 0.998, 200k, snapshots --
directly against v21's 21%.

### v23: the knee works, and it moved the binding criterion to v_z

```
snapshot    v21 (old)   v22 (gamma)   v23 (knee)
1.55M          8%           8%            4%
1.60M          4%           0%           12%
1.65M         21%           4%           25%   <- best
1.70M          4%          12%           17%
total       9/96         6/96         14/96
```

14 vs 9 successes out of 96 is about 1 sd, so the aggregate is NOT significant
on its own. The mechanistic evidence is much stronger than the rate, and it is
what matters:

```
                      v21 best (21%)                         v23 best (25%)
landed  v_xy    0.35 1.03 1.05 1.12 1.17          0.50 0.73 0.86 0.88 0.91 0.95
crashed v_xy    1.26 ... 3.37 4.70 6.50 14.47     1.47 1.55 1.65 1.67
crashes         14                                4
binding         v_xy on 13 of 14                  v_z on 3 of 4
```

The 2-14 m/s touchdown tail is gone and the binding criterion has moved to the
vertical channel. That is exactly what the change was aimed at.

Cost: timeouts rose 5 -> 14, in two distinct populations. Eight sit at 0.8-3.6 m
with v_xy 2.7-8.1, i.e. they now correctly refuse to touch down fast but cannot
null the lateral speed inside the 25 s budget. Six never descend at all
(15-22 m, vz at or above zero) -- v21 had 3 of those, so at n=24 that part is
inside the noise.

### The vertical channel was 12x weaker than the lateral one

Measured per step at alt=0.2 m, the charge separating a landing from a crash:

```
vertical, vz  -0.8 -> -1.8 :   63.7
lateral,  vxy  0.9 ->  1.7 :  792.4     (12.4x)
```

Same defect as kxy=12, same fix: `profile_k` 35 -> 400 brings it to 416/step,
a 0.53 ratio against the lateral channel instead of 0.08.

A correction to an earlier reading of this: exceeding the envelope was NOT
profitable, at any altitude -- the progress reward is capped at the envelope, so
extra speed earns nothing and only incurs the penalty. What was wrong was the
MAGNITUDE: 0.5 m/s of excess cost 8.8/step, which against lateral terms of
several hundred per step was noise. It now costs 100/step, and a free-fall
arrival costs 20851 instead of 2027. The sign was always right; it just did not
matter.

Two coupled numbers had to move with it:
  - `profile_cap` 3000 -> 25000. At k=400 a cap of 3000 saturates at 2.74 m/s
    of excess, which is inside the range a near-ground free-fall arrival
    reaches (~8 m/s) -- i.e. it would have recreated vertically the exact
    defect just removed laterally. The block's own comment demands the term
    stay unsaturated out to ~7 m/s.
  - `profile_climb_k` 70 -> 800, holding the 2x ratio to profile_k that was
    measured to stop the climb-away exploit. v23 still timed out 6 of 24 at
    15-22 m with vz at or above zero, so that relationship is still carrying
    load and must not become relatively free.
  - `shaping_clip_abs` 60000 -> 80000 to keep the clip a safety net.

### `profile_alt_floor_m` was documented as "not a tunable" and was the most load-bearing number in the block

It floors the altitude inside the sqrt, which means it sets the envelope AT
CONTACT -- the touchdown speed the reward demands. At 0.25 m that was
`0.78*sqrt(0.25)` = **0.39 m/s against a `safe_landing_v_z_m_s` of 1.0**: the
reward was asking for a touchdown 2.5x gentler than the vehicle needs, and
(before profile_k rose) had no authority to enforce even that. Raised to 1.0 m,
giving a 0.78 m/s floor, a 22% margin under the real limit.

This also attacks the hover-just-above-the-ground population directly, because
the progress reward is capped at the envelope: at alt=0.2 m the rate it pays
for goes 0.39 -> 0.78 m/s, doubling the reward for descending the last metre
(267 -> 535 per step) rather than hanging there.

Verified: the proven ramp_20m touchdown regime (vz -0.04..-0.95) still pays
essentially nothing (0.0/0.0/0.2/11.6 per step at alt=0.2, against
0.0/0.4/5.9/11.0 before); the proven ZemZev landing still beats the failure
modes by +295 discounted and its own score rose again; no saturation out to
7 m/s of excess; 4 more regression tests (envelope floor vs the real limit,
channel balance, diving never pays, no flat spot through free fall). Suite:
163 passed, 7 pre-existing failures.

v24 is the A/B: same stage, same warm start, gamma 0.998, 200k, snapshots.

### v24: the vertical fix lands it. ramp_35m 6% -> 29% over the night.

```
run                 snapshots (24 eps each, seed0=31000)   best   total
v21  old reward      8,  4, 21,  4                          21%    9/96
v22  gamma 0.997     8,  0,  4, 12                          12%    6/96
v23  lateral knee    4, 12, 25, 17                          25%   14/96
v24  + vertical     29,  4, 17, 21                          29%   17/96
```

Statistically honest: 17/96 vs 9/96 is a two-proportion z of 1.69, p ~ 0.09.
Not significant at 0.05 -- a trend. The evidence that matters is mechanistic:
each intervention made a specific prediction about the margins and each
prediction held.

```
                 landed v_xy median    crashed v_xy max    timeouts at best
v21                    1.05                 14.47                5
v23                    0.87                  1.67               14
v24                    0.49                  4.22                4
```

The predicted side effect of raising `profile_alt_floor_m` -- doubling the
reward for descending the last metre -- shows up exactly where it should:
v23's 14 timeouts at its best snapshot fall to 4 in v24.

**The reward is now balanced, and that is the real result.** The binding
criterion on v24's remaining failures is spread across v_xy (7), v_z (5),
tilt (3) and |w| (1) instead of being one criterion on 13 of 14. A single
criterion dominating was the signature of a reward defect; a spread means
further progress needs capability or training time, not more reshaping. The
remaining crashes are near misses -- ep12 touched down at v_xy=1.29 against a
1.2 limit, i.e. it missed by 0.09 m/s.

So: stop reshaping the reward. The three defects found tonight (lateral
saturation, clip filled by one term, vertical channel 12x too weak) were all
real and all measured; there is no evidence of a fourth.

### Next: rebuild the ladder on the corrected reward, from the bottom

v24's best snapshot is its EARLIEST (1.55M, 50k steps in), then 4%, 17%, 21%.
The warm-started policy is already near its best under the new reward and
training initially makes it worse. So more ramp_35m steps is the wrong spend;
the earlier rungs are, because every later stage inherits them.

Note the measured rates of the existing checkpoints do NOT change with the
reward -- the reward only affects what training does, not what a trained policy
achieves. ramp_20m stays 83% and ramp_20m_fast stays 46%. What changes is where
further training from them can get to.

v25 retrains ramp_20m_fast (the 46% rung) under the corrected reward, warm
started from v9's 83% ramp_20m.

### v25: restarting a rung from below loses to the rung's own checkpoint

Retrained ramp_20m_fast under the corrected reward, warm started from v9's 83%
ramp_20m, 200k steps:

```
1.05M  25%      v16's existing ramp_20m_fast checkpoint, re-measured on the
1.10M  33%      IDENTICAL sample (24 eps, seed0=31000): 11/24 = 46%
1.15M  12%
1.20M  33%   <- best
```

So 33% against a 46% incumbent. This is NOT evidence that the new reward is
worse, and it must not be read that way: v25 had only 200k steps on this stage
while v16's checkpoint carries a much longer history, and the two started from
different policies. What it measures is that 200k steps cannot reproduce that
history.

The planning error is mine and worth recording: "rebuild the ladder from the
bottom on the corrected reward" throws away training history for no reason.
The reward only changes what FURTHER training does, so the right move is to
continue each rung from its own best checkpoint, not to restart it from the
rung below. v26 does that -- ramp_20m_fast from v16 itself, 200k, new reward,
against v16's own 46%.

### v26: continuing ramp_20m_fast from its OWN best checkpoint also degrades it

```
1.55M  17%
1.60M  21%
1.65M  29%   <- best
1.70M   0%
```

v16's own checkpoint measures 46% on the same 24 episodes. So 200k more steps
from it, under the corrected reward, took the stage DOWN from 46% to 29%.

This contradicts the assumption I had been working under all night -- that the
reward was the lever. Two readings fit:
  (a) the new reward helps ramp_35m (21 -> 29%) and hurts ramp_20m_fast
      (46 -> 29%)
  (b) further training degrades the stage regardless of reward, and v16's 46%
      was a peak that any continuation destroys

(b) is consistent with the mid-run collapse this file already documents, and it
would mean the reward conclusions drawn tonight are confounded. The two are
distinguishable with one control: continue from v16 for 200k under the OLD
weights. v27 is that run.

### `--reward-weight NAME=VALUE` added to the trainer

Repeatable override on any RewardWeights field, so a reward A/B is a
reproducible command instead of an edit to reward.py that has to be remembered
and undone. Verified it reproduces the old reward exactly: with
`kxy_knee_m_s=1e9 kxy_cap=20000`, the lateral charge at v_xy=13.62 / alt=0.45
is 20000 again, i.e. the old saturated value. Passing no override returns None
and the env keeps its own default, so the normal path is untouched.

### THE CONTROL SETTLES IT: continuation degrades the stage under EITHER reward

```
run (200k from v16's ramp_20m_fast)   snapshots        best   total
v16 incumbent                          --               46%     --
v27  OLD reward (via --reward-weight)  17, 33, 21, 21   33%   22/96
v26  NEW reward                        17, 21, 29,  0   29%   17/96
```

So reading (b) is the right one: **further training takes the stage down
regardless of reward.** v16's 46% was a peak and any continuation destroys it.
On this stage old and new reward are indistinguishable (22 vs 17 of 96, sd of
the difference ~6.2).

This does NOT invalidate the ramp_35m reward comparison, and it is worth being
precise about why: v21/v23/v24 all had the same structure -- same warm start,
same 200k budget, differing only in reward -- so the degradation pressure was
equal in every arm and the comparison stays controlled. What it changes is the
interpretation of the headline. "ramp_35m 6% -> 29%" is two separate effects:
snapshot-picking (6 -> 21) and the reward fixes (21 -> 29).

That the reward fixes do nothing on ramp_20m_fast is consistent rather than
contradictory: both defects were SATURATION defects, and saturation only binds
at speed. ramp_20m_fast is the gentlest rung (release 2-5 m/s from 20 m), so
there was little for them to do there.

**The dominant problem is no longer the reward. It is that training degrades
the policy**, and every "best checkpoint" in this file is a peak that the run
then walked away from. Stop spending 200k continuations expecting improvement
until this is addressed.

First candidate, from the list already in this file: the replay ratio. With
n_envs=16 and `--gradient-steps -1` SB3 does one gradient step per env step,
i.e. 16 updates per vec step -- a lot of updates per unique transition, which
is the classic driver of critic overfitting. `--gradient-steps` is already
exposed. v28 tests 1:4.

## MEASUREMENT ERROR: 24 episodes cannot rank checkpoints, and "degradation" was an artifact

v28 (replay ratio 1:4) came back at best 29%, 17/96 -- identical to 1:1. So
gamma, reward (on this stage) and replay ratio have all been refuted as causes
of the apparent decline. At that point the right move was to question the
measurement rather than find a fourth cause, and that was the answer:

```
checkpoint        seed 31000   seed 52000   seed 73000   pooled
v16                   46%           8%          33%      21/72 = 29%
v27 best              33%          21%          29%      20/72 = 28%
```

**v16's 46% is a sampling artifact.** The same policy scores 8% on another 24
episodes. Pooled over 72 episodes v16 and v27's best are indistinguishable,
so there was never a 46% to degrade from.

The error was mine and it is a textbook one: v16 was SELECTED as the best
ramp_20m_fast checkpoint using the seed0=31000 sample, and then the decline was
measured on that same sample. That is selection and comparison on identical
data, so regression to the mean reads as degradation.

### What this invalidates

The spread on 24 episodes is roughly +/-15 points -- one policy ranged 8% to
46%. So every single-sample figure in this file carries that much noise, and
these conclusions are NOT established:

  - ramp_35m "21% -> 25% -> 29%" across v21/v23/v24. Single sample each; the
    progression is within noise.
  - ramp_20m's 83% and every other per-stage headline rate.
  - "performance peaks mid-run and collapses", and the non-monotonic snapshot
    curves (8/4/21/4 and the rest). Much of that oscillation is measurement
    noise, not the policy changing. The underlying claim may still be true but
    it is not established by those numbers.

### What survives

Nothing that depends on a 24-episode rate. These do not:

  - The two saturation defects, which are properties of the reward function
    proven by direct computation -- the lateral term was provably flat above
    7.07 m/s near the ground, the clip was provably filled by one term, and the
    vertical channel was provably 12x weaker than the lateral one.
  - The margin distributions, which aggregate every episode of a run rather
    than resting on a success count: crashes 14 -> 4, worst crash v_xy
    14.47 -> 1.67, median landed v_xy 1.05 -> 0.49, and the binding criterion
    going from one criterion on 13 of 14 failures to a spread across four.

### The standard from here

`--episodes 24` is for a smoke test, never for a comparison. At a ~30% rate,
+/-5 points needs n ~ 84, so a real comparison is 96+ episodes, and a
checkpoint must never be selected and then judged on the same seeds.

## THE RESULT, properly measured: the reward fixes roughly double the landing rate

96 episodes, seed0=41000 -- a sample never used to select either checkpoint:

```
checkpoint                              landed     timeout
v21 best (OLD reward)                 11/96 = 11.5%    26
v24 best (NEW reward)                 22/96 = 22.9%    22
```

Two-proportion z = 2.09, **p ~ 0.037**. Significant, and the improvement is not
bought with timeouts (26 -> 22).

Both checkpoints were picked as the best of four snapshots on the seed0=31000
sample, so both carry the same selection advantage and the fresh sample makes
the DIFFERENCE trustworthy even though it inflated both absolute figures.

So the corrected headline for the night: **ramp_35m went from 11.5% to 22.9%**,
not 6% to 29%. The earlier numbers were inflated by a lucky 24-episode sample.
The improvement is real and about 2x; the absolute level is lower than reported.

### Settled, in order of confidence

1. Three saturation defects in the reward were real, proven by direct
   computation, and fixing them roughly doubles the landing rate (p~0.037).
2. The margin distributions shifted exactly as each fix predicted (crashes
   14 -> 4, worst crash v_xy 14.47 -> 1.67, binding criterion spread from one
   to four).
3. 24 episodes cannot rank checkpoints (one policy spanned 8-46%). Comparisons
   need 96+, on seeds not used for selection.
4. The apparent mid-run collapse and the "degradation on continuation" were
   largely this measurement error. Not disproven, but not established either.

### Open

- ramp_20m_fast, ramp_20m, ramp_50m rates are all unmeasured to this standard.
  ramp_20m_fast pools to ~29% over 72 episodes, well below its reported 46%.
- Whether training degrades a policy at all is now an open question, and the
  honest answer is that the evidence gathered for it does not survive.
- ramp_50m and beyond still need the random-policy reachability probe and the
  `usable lateral room > episode drift` check before training. That rule came
  out of seven failed runs and still stands.

### Stage precondition audit, re-run 2026-10-06

Both rules from this file, applied to every stage as configured now:

```
stage            vz0   h0/T needed  vert   drift(vxy*T)  tile needed   tile now
ramp_20m         2.0      1.00       ok          40          120         120  ok
ramp_20m_fast    2.0      1.00       ok         100          260         260  ok
ramp_35m         2.8      1.40       ok         125          289         320  ok
ramp_50m         3.0      1.67       ok         240          539         600  ok
ramp_100m        4.0      2.50       ok         640         1389         640  TOO SMALL
ramp_150m        5.0      3.00       ok        1200         2539        1200  TOO SMALL
orbit_descent    0.0      3.33       --        1800         3769        1680  TOO SMALL
```

The vertical rule is satisfied everywhere it applies. orbit_descent's vz0=0.0
is deliberate and correct -- the comment in curriculum.py says so: it is the
real scenario, and the policy arriving there is not random but trained through
ramp_150m, so the reachability argument that justified vz0=-2.0 on ramp_20m
does not transfer. Flagging it as a failure would be a false positive.

The lateral blocker is real on the last three and remains the thing to fix
before any of them can be trained. But the number should not be picked from a
model:

  - `drift = vxy * T` assumes the vehicle never brakes. That is right for the
    RANDOM-policy probe, and it is what validated ramp_35m's 320 m tile.
  - A policy inherited from ramp_50m does brake, and the corrected reward now
    pushes hard on exactly that. If lateral speed bleeds to zero over about
    half the episode, the requirement drops a lot: 800 / 1400 / 2000 instead
    of 1440 / 2560 / 3840.

The two differ by 2x on orbit_descent, and the conservative number is not free:
`terrain_grid_n` is global (default 80), so a 3840 m tile is 48 m per cell,
coarse enough to matter for a landing. Measure before choosing -- run the
probe on ramp_100m at the current tile and read the left_tile count, exactly
as the rule in this file prescribes.

## THE CURRICULUM WAS NOT MONOTONIC IN DIFFICULTY -- three rungs were unsolvable

Flew the ZemZev controller -- the reference for a healthy landing, and a far
better reachability test than a random policy -- over all seven stages:

```
stage            T      controller   drift max / usable room
ramp_20m         20 s     24/24          14.7 /  60
ramp_20m_fast    20 s      1/24  <--     49.3 / 130
ramp_35m         25 s     21/24          54.1 / 160
ramp_50m         30 s     24/24         102.1 / 300
ramp_100m        40 s      0/24  <--    293.7 / 320
ramp_150m        50 s      0/24  <--    529.3 / 600
orbit_descent    60 s     22/24         638.1 / 840
```

The FINAL stage was easier than three of the rungs below it. The policy was
being asked to climb through stages that the reference controller cannot pass.

**Two distinct defects, both measured:**

`ramp_100m` and `ramp_150m` were SHORT ON CLOCK, not hard. Every failure was
identical and showed the vehicle doing everything right:

```
ramp_100m  TIMEOUT  t=40.0  alt=4.45  vz=-0.80  vxy=0.56
ramp_150m  TIMEOUT  t=50.0  alt=4.65  vz=-0.80  vxy=0.48
```

Lateral speed already nulled, descending at the controller's 0.80 m/s terminal
rate, and the clock ran out 4.5 m above the ground -- 5.6 s short. Fixed:
40 -> 48 s and 50 -> 60 s. Both go to **24/24**, and drift does not grow with
the extra time because it is spent in the terminal descent at ~0.5 m/s lateral
(293.7 -> 292.2 m).

`ramp_20m_fast` was IMPOSSIBLE, and more clock does not fix it: 1/24 at every
budget from 20 s to 34 s. 20 m of altitude is not enough to bleed off 5 m/s of
lateral speed -- the vehicle reaches the ground before it finishes braking, at
v_xy 0.87-2.50 against a 1.2 limit. Measured envelope at 20 m:

```
v_xy (2.0, 5.0)   1/24        v_xy (1.5, 2.5)  16/24
v_xy (2.0, 3.5)   1/24        v_xy (2.0, 2.5)  11/24
v_xy (2.0, 3.0)   1/24
raising altitude instead: h0=28 -> 17/24, h0=30 -> 22/24 (= ramp_35m again)
```

Fixed to (1.5, 2.5) -> 16/24. **Every rate ever measured on this stage (42%,
46%, 29%) was scored on the impossible version**, including last night's
"ramp_20m_fast degrades from 46%" work.

### The tile question is settled, and the answer is "change nothing"

The conservative model (`drift = vxy * T`, never brakes) said the last three
tiles were 2-3x too small. Measured drift from a controller that actually
lands says otherwise -- every current tile already fits:

```
stage           measured max drift   tile half   models said (no-brake / brake)
ramp_35m               54.1            160          125 /  62
ramp_50m              102.1            300          240 / 120
ramp_100m             292.2            320          640 / 320
ramp_150m             527.9            600         1200 / 600
orbit_descent         638.1            840         1800 / 900
```

The worst-case model is right for a RANDOM-policy probe and wrong for sizing a
tile a competent policy will fly in. No tile changes needed, and the coarse
48 m/cell terrain that a 3840 m orbit_descent tile would have forced is avoided.

### Ladder after the fixes -- every rung solvable by the reference

```
ramp_20m 24/24 | ramp_20m_fast 16/24 | ramp_35m 21/24 | ramp_50m 24/24
ramp_100m 24/24 | ramp_150m 24/24 | orbit_descent 22/24
```

### Target agreed with the user: 75% landing rate, given a safe site exists

## Overestimation is NOT the cause of the collapse -- three experiments say so

```
intervention                       Q behaviour              landing rate
gamma 0.998 -> 0.997               Q -14%                   no change
replay ratio 1:1 -> 1:4            --                       no change
learning rate 3e-4 -> 5e-5         inflation STOPPED        still collapses
```

v29 (lr 3e-4): Q 151 -> 246 -> 323 -> 391 -> **406** -> 352 -> 288 -> 263
v30 (lr 5e-5): Q 143 -> 197 -> 217 -> **222** -> 215 -> **175**

Low LR gave exactly the stability we wanted -- Q plateaus and comes back down --
and the policy collapsed anyway (8%, 4%, 0%, 0%, with 13-15 of 24 timing out).

So `n_critics` should NOT be the next thing tried, even though it is the
textbook answer: it targets the same max-bias mechanism that these three
experiments exonerate. Keep it as a last resort.

### The actual signature: the policy has collapsed to bang-bang

```python
m.predict(obs, deterministic=True)   -> [-1.  1. -1.  1.]
m.predict(obs, deterministic=False)  -> [-1.  1. -1.  1.]   (identical)
```

Deterministic and stochastic predictions are bit-identical because the policy's
PRE-TANH std is 4.7-5.8, so tanh is saturated and sampling cannot move the
output off the box corners. Measured over 48 episodes, the stochastic and
deterministic landing rates are identical (10/48 and 14/48 timeouts, both).

That explains the near misses directly -- a saturated policy cannot make small
corrections, and ramp_35m failures were losing by 0.09 m/s of lateral speed. And
the std GROWS in the runs that collapse:

```
v24 best (22.9%)   std = 4.75  3.76  4.98  3.74
v30 last (0%)      std = 4.73  4.47  5.75  4.52
```

This is a known SAC pathology on a bounded action space: with the entropy
coefficient pinned too high, the cheapest way to earn the entropy bonus is to
widen the pre-squash distribution into tanh's saturation region, where extra
width costs nothing in behavioural diversity but destroys fine control.

And the coefficient is mis-sized BY MY OWN CHANGE: handover records 0.05 as
sized against the OLD reward's measured per-step magnitude (mean |r| = 0.0483),
and the reward was then rewritten from scratch with different magnitudes
throughout. v31 tests ent_coef 0.005 (lr held at 5e-5, which demonstrably fixed
the Q inflation).

### `--stochastic` added to diag_stage_landing_rate.py

Training collects data by sampling the policy while evaluation takes its mean,
so the gap between the two is worth being able to measure. Here it is zero,
which is itself the finding.

## ent_coef is not it either -- FOUR knobs now refuted

v31 (ent_coef 0.005, lr 5e-5): **21% at 75k steps in, 4% at 150k** (19 of 24
timing out). Same collapse, same shape.

```
knob                    what it did to the symptom      landing rate
gamma 0.998 -> 0.997    Q -14%                          no change
replay ratio 1:1 -> 1:4 --                              no change
lr 3e-4 -> 5e-5         Q inflation stopped outright     still collapses
ent_coef 0.05 -> 0.005  pre-tanh std 4.5 -> 3.2          still collapses
```

### Correction: bang-bang saturation is NOT the collapse mechanism

I reported the saturated policy as the cause. It is not. Saturation is the same
in the GOOD policy:

```
                        pre-tanh std        saturated actions
v24 best (22.9%)        4.54 - 4.62              95.0%
v30 last (0%)           4.50 - 5.70              95.8%
v31 (ent 0.005)         2.75 - 3.51              96.8%
```

95% saturated and it lands 22.9% of the time, so bang-bang is the normal
operating mode of these policies, not a defect -- which makes sense at 20 Hz,
where switching the throttle gives a PWM-like average thrust. Lowering the
entropy coefficient did move the std exactly as predicted and did nothing to
saturation or to the collapse. The saturation measurement stands; the causal
claim does not.

### What the collapse is invariant to, and what that points at

It happens within ~100k steps of every warm start, at every gamma, learning
rate, entropy coefficient, replay ratio and under both rewards. The one thing
common to all of them that has NOT been tested: **the replay buffer is reset on
every warm start and the critic is refit from a narrow, on-policy distribution.**

`--keep-buffer` cannot test this -- `SAC.load` does not restore a buffer, so the
flag only skips the reset of an already-empty one. Testing it needs
`save_replay_buffer` / `load_replay_buffer` wired into the checkpoint path.
**That is the next experiment**, and it is the one candidate the four refuted
knobs do not touch.

Second candidate, cheaper to try: freeze the actor for the first N steps after a
warm start so the critic can re-fit before the policy starts chasing it.

## v32: the direct jump to orbit_descent does not transfer

Warm started from the best ramp_35m policy (22.9%), 200k steps, lr 5e-5, new
reward:

```
snapshot    landed   lost_control   left_tile   (16 episodes each)
1.60M        0/16        4/16          2/16
1.65M        0/16        4/16          0/16
1.70M        0/16        5/16          0/16
1.75M        0/16        5/16          1/16
```

0/16 throughout, and the failure mode is DIFFERENT from ramp_35m's: the vehicle
tumbles (lost_control) and crashes rather than hovering. The 35 m / 5 m/s ->
200 m / 30 m/s jump is too large for the policy to carry, which matches the
earlier attempt this file records. Worth trying -- the stage is now verified
reachable, since the reference controller lands 22/24 there -- but the answer is
no, not in one step.

(16 episodes is a smoke test, not a ranking measurement. It does not need to be
better than that: 0/16 four times over is not a sampling question.)

# ================================================================
# HANDOVER -- state at 2026-10-06 12:00
# ================================================================

## Where the project actually is

```
stage            best verified rate   checkpoint
ramp_35m              22.9%           out/sac_training_run_v24_vert/snapshots/
                 (22/96, seed0=41000)   ramp_35m_1550000_steps.zip
orbit_descent          0%             not reachable by training yet
```

That 22.9% is the only number in this project measured to the standard below.
Treat every other rate in this file as provisional.

Target agreed with the user: **75%**, given a safe landing site exists.
For reference, the ZemZev controller lands 22/24 (92%) on orbit_descent, so the
target is physically achievable -- the gap is in training, not in the vehicle,
the reward, or the stage.

## The measurement standard -- read this before trusting any number

24 episodes CANNOT rank checkpoints. One policy scored 8%, 33% and 46% on three
different 24-episode samples. Comparisons need 96+ episodes, and a checkpoint
must never be selected and then judged on the same seeds -- doing that is what
produced a phantom "training degrades the policy" result earlier in this file.

```
smoke test      --episodes 16 or 24
comparison      --episodes 96, on seeds not used for selection (e.g. 41000)
```

## What is settled

1. **Three saturation defects in the reward, all proven by direct computation
   and worth ~2x the landing rate** (11.5% -> 22.9%, p~0.037 on 96 held-out
   episodes): `kxy_cap` saturating at 7.07 m/s near the ground, the shaping clip
   being filled by the lateral term alone, and the vertical channel being 12x
   weaker than the lateral one. Details above; seven regression tests guard them.

2. **The curriculum had three unsolvable rungs.** Flying the ZemZev controller
   over all seven stages -- a far better reachability test than a random policy
   -- showed the FINAL stage was easier than three rungs below it. ramp_100m and
   ramp_150m were 6 s short on clock; ramp_20m_fast was geometrically impossible
   (20 m is not enough altitude to bleed 5 m/s of lateral speed, and no budget
   fixes it). All three fixed and verified. Every rate ever measured on
   ramp_20m_fast was scored on the impossible version.

3. **Tile sizes need no change.** Measured drift from a controller that lands is
   inside every current tile (orbit_descent: 638 m against 840 m of room). The
   `drift = vxy * T` bound is right for a random-policy probe and wrong for
   sizing a tile a competent policy flies in.

4. **The collapse is real** -- a checkpoint measured at 22.9% on 96 episodes
   goes to 0/24 on five consecutive snapshots after 600k more steps -- **and it
   is not overestimation.** Four knobs refuted: gamma, replay ratio, learning
   rate (which stopped the Q inflation outright and changed nothing), entropy
   coefficient.

## The next experiment, and why it is the right one

The collapse happens within ~100k steps of EVERY warm start, at every gamma,
learning rate, entropy coefficient and replay ratio, under both rewards. That
invariance is the clue. The one thing common to all of them that has never been
tested:

**the replay buffer is reset on every warm start, so the critic is refit from a
narrow on-policy distribution.**

`--keep-buffer` cannot test this: `SAC.load` does not restore a buffer, so the
flag only skips the reset of an already-empty one. The test needs
`save_replay_buffer` / `load_replay_buffer` wired into the checkpoint and
warm-start paths. Concretely:

```python
# where the CheckpointCallback saves, also:
model.save_replay_buffer(path.with_suffix(".buffer.pkl"))
# on the warm-start path, after SAC.load and BEFORE the reset:
if buf.exists(): model.load_replay_buffer(buf)   # and skip replay_buffer.reset()
```

Second candidate, cheaper: freeze the actor for the first N steps after a warm
start so the critic can re-fit before the policy starts chasing it.

Do NOT spend hours on `n_critics` / training from scratch. It is the textbook
answer for overestimation and the four experiments above exonerate that
mechanism. The user has approved it, but it should stay a last resort.

## The ladder, as it now stands (reference controller, 24 episodes)

```
ramp_20m 24/24 | ramp_20m_fast 16/24 | ramp_35m 21/24 | ramp_50m 24/24
ramp_100m 24/24 | ramp_150m 24/24 | orbit_descent 22/24
```

Every rung is solvable by the reference. The direct ramp_35m -> orbit_descent
jump does not transfer (v32, 0/16), so the intermediate rungs have to be walked
-- but they are only worth walking once the collapse is fixed, because today any
200k continuation loses what it started with.

## Tooling added this session

```
--reward-weight NAME=VALUE   trainer: reward A/B as a reproducible command
--learning-rate              trainer: forced on the warm-start path too, because
                             SB3 reads lr_schedule and would silently keep the
                             checkpoint's rate
--stochastic                 diag_stage_landing_rate: sampled vs mean policy
lunarsim/rl/action_map.py    the ONE action[0] -> throttle map and its inverse
scripts/notify_telegram.py   run output to Telegram (creds only in
                             ~/.config/lunarsim/telegram.env, never the repo)
scripts/telegram_control.py  command dispatcher, one chat_id, fixed whitelist
```

Tests: 163 passed, 7 pre-existing failures (test_apollo_lm_asset.py needs the
Isaac python, not the venv one).

## v33: from scratch with a 5-critic ensemble (user's call)

Launched 2026-10-06 12:06, `out/sac_training_run_v33_scratch5`:

```
scripts/train_sac_isaac.py --steps-per-stage 200000 --checkpoint-every 50000 \
  --out-dir out/sac_training_run_v33_scratch5 \
  --n-critics 5 --gamma 0.998 --max-rocks-per-env 0
```

No `--only-stage`, so it walks the whole corrected ladder from ramp_20m. Seven
stages at 200k is ~1.4M steps, roughly 2.5 h. Learning rate stays at the 3e-4
default -- 5e-5 is for fine-tuning an already-good policy and would be far too
slow from random init.

To be explicit about what this is: the four experiments above exonerate
overestimation, which is the mechanism more critics address, so this is not
expected on the evidence to fix the collapse. It is being run because the user
chose to, and because the 22.9% it would otherwise preserve is far enough below
the 75% target not to be worth protecting.

**If it was interrupted**, resume from the last snapshot in
`out/sac_training_run_v33_scratch5/snapshots/` with the SAME `--n-critics 5`
(a 5-critic checkpoint will not load into the 2-critic default):

```
--warm-start out/sac_training_run_v33_scratch5/snapshots/<last>.zip \
--only-stage <the stage it was on> --n-critics 5
```

### `--n-critics` and `_expand_critic_ensemble`

The flag also supports growing the ensemble on a warm start WITHOUT discarding
a trained policy, which is the path to prefer if this is ever revisited with a
policy worth keeping. `SAC.load` cannot restore a 2-critic checkpoint into a
5-critic model, so the helper copies the actor exactly and CLONES the trained
critics into the new slots with a 1% perturbation. The cloning is the point:
SAC takes the min over the ensemble, so a randomly initialised critic would
dominate that min with untrained garbage and bootstrap it into the target. The
perturbation keeps the clones from being exact duplicates, which would leave the
min unchanged and the extra critics redundant.

## Literature review + two new measurements: it is neither overestimation NOR plasticity loss

Full review: `rl-cokus-literatur-taramasi.md` (30 papers, arXiv + OpenAlex MCP).

### What the literature reframed

Klein et al. (2024), *Plasticity Loss in Deep RL: A Survey*, states plainly that
loss of plasticity "contributes to scaling failures, **overestimation bias**, and
insufficient exploration". So Q inflation is a SYMPTOM in that framework, not the
disease -- which is exactly why gamma and (predictably) n_critics do nothing, and
why lowering the learning rate stopped the Q inflation without stopping the
collapse. Our whole diagnostic frame was the 2018-2021 one.

Our setup is also a textbook primacy-bias generator (Nikishin et al. 2022): we
reset the replay buffer on every warm start, so the critic refits from a narrow
early window; the policies we warm-start from have 1.5M steps on them (cf.
*Policy Plasticity Matters in Offline-to-Online RL*, 2026); and the curriculum is
deliberate non-stationarity (Abbas et al. 2023).

### But the plasticity hypothesis does not survive measurement either

Dormant neuron fraction (ReDo's metric, tau=0.025):

```
checkpoint              actor dormant   critic dormant
v24 BEST   (22.9%)          26.0%           12.5%
v29 COLLAPSED (0%)          28.9%           12.8%
v30 COLLAPSED (0%)          25.6%           13.4%
v31 COLLAPSED (4%)          25.0%           13.0%
```

Identical. (Caveat: measured on random Gaussian observations rather than real
rollout batches, which is how ReDo defines it -- worth redoing on real data, but
a 3-point spread is not going to become a 2x one.)

Weight norms and feature rank:

```
checkpoint              actor |W|   critic |W|   feat rank   srank(0.99)
v24 BEST   (22.9%)         190.7       484.8      137/256        29
v29 COLLAPSED (0%)         223.5       598.3      130/256        24
v30 COLLAPSED (0%)         192.2       486.8      144/256        31
```

**v30 is the decisive counterexample.** v29 does show the classic plasticity
signature (norm growth, rank drop), but v30 collapsed to 0% with its weight norm
essentially unchanged, its feature rank HIGHER than the good checkpoint's, and its
dormant fraction flat. A network whose statistics barely moved lost all of its
performance.

### The synthesis that survives: the policy is brittle because it sits on the bound

v30 went from 22.9% to 0% while its actor norm moved 190.7 -> 192.2. A tiny
parameter change produced a total behavioural change. That is what a SATURATED
policy does: at `|a| = 1` the parameter-to-behaviour map is a step function, so
small gradient steps flip discrete action choices instead of refining them.

This puts the bang-bang measurement back in the picture, correctly this time --
not as the CAUSE of the collapse, but as the reason training is a random walk in
behaviour space rather than an improvement process. The literature has the
mechanism: Shamass (2026), *Unthrottling the Tanh Jacobian in SAC*, notes
`da/du = 1 - a^2` vanishes as `|a| -> 1`, "starv[ing] the actor of critic signal
exactly where extreme actions are optimal".

**And that paper is a warning, not a recipe.** It tried to restore the missing
gradient and the result was negative: the ungated version drove the policy to 99%
saturation and collapsed return from -31.6 to -195.5. Its closing line applies
to us directly: *"Saturating a bound is not the same as solving a problem whose
optimum lives on that bound."* So do not "fix" the Jacobian.

### What the landing literature actually does -- we are on the hard road

Of the lunar-landing guidance work found, the large majority does NOT use RL. It
trains networks SUPERVISED on optimal trajectories: Wang/Chen/Li (2024) generate
the dataset from Pontryagin's Minimum Principle; Origer & Izzo (2024) build
Guidance & Control Networks that represent an optimal control policy; Wang (2026)
does Optimality-Informed NNs; Shen et al. (2022) pair convex optimisation with a
network. The RL branch is mostly one group (Gaudet/Linares/Furfaro 2018, 2021),
and SAC specifically is near-absent.

We have a ZemZev controller that lands 22/24 (92%) on orbit_descent. The
literature-standard move is to imitate it (behaviour cloning / DAgger / G&CNET
style) and RL-finetune if needed. The user has explicitly chosen not to seed the
buffer with demos ("kendi bulsun"), which is a legitimate call -- but it is worth
re-raising as a decision, because it is the field's main road.

### Revised next steps, cheapest first

```
1  LayerNorm in policy_kwargs, one run.        Klein+24: general regularisation
                                              beats domain-specific fixes;
                                              Lyle+24: also fights overestimation
2  Periodic partial resets with snapshots.     Nikishin+22; Calibrated Partial
                                              Resets (2026) targets "policy
                                              collapse" by name
3  Narrow / re-scale the action box so the      our own v30 measurement: the
   operating point is interior rather than      policy is brittle because it
   on the tanh bound                            lives on the bound
4  Keep the replay buffer across warm starts.   demoted from #1 by this review
5  Behaviour cloning from ZemZev + RL finetune. the landing field's main road
```

Do NOT: unthrottle the tanh Jacobian (published negative result), or spend more
on n_critics (two independent lines of evidence against it now).

## All five review items implemented (2026-10-06 12:35)

### 5. Behaviour cloning from the controller — DONE, needs evaluating

`scripts/train_bc_from_demos.py` fits the SAC actor to the ZemZev controller's
actions and saves a REAL SAC checkpoint, so `diag_stage_landing_rate.py`,
`--warm-start` and the snapshot machinery all work on it unchanged. Only the
actor is fitted; the critic stays at init, which is right for something that
will be evaluated directly or RL-finetuned.

Two details that matter:
  - it regresses the PRE-TANH mean (`atanh(a)`), not the squashed action.
    Regressing the squashed output would push the loss through a saturating
    nonlinearity and give almost no gradient exactly where the controller
    commands extremes -- which is most of the time.
  - it sets the policy's log_std head to a small constant. SB3's `log_std` is a
    state-dependent `Linear`, not a parameter, so the weight is zeroed and the
    bias set to -3. Left at init, a std of 4.5+ would saturate tanh and throw
    the fit away.

Trained on the existing 48-episode orbit_descent demo set:
`val MSE 0.078, mean |da| 0.046` (on a +/-1 action scale, ~2.3% of range).
Checkpoint: `out/bc_orbit_descent.zip`.

**NOT YET EVALUATED** -- this is the first thing to run:

```
scripts/diag_stage_landing_rate.py --checkpoint out/bc_orbit_descent.zip \
    --stage orbit_descent --episodes 96 --seed0 41000
```

The controller it cloned lands 22/24 (92%), so this measurement is the real
test of whether the field's main road works here. `collect_zemzev_demos.py` now
takes `--stage`, so demos can be gathered for any rung.

### 1-3. LayerNorm, periodic resets, action-saturation penalty

New module `lunarsim/rl/plasticity.py` (LayerNorm weaving, ReDo's
dormant-neuron metric as a diagnostic, partial resets + an SB3 callback), wired
into the trainer:

```
--layer-norm                 LayerNorm in the actor and critic trunks. Fresh
                             models only. Klein+24: general regularisation
                             usually beats domain-specific fixes.
--reset-every N              periodic partial reset (Nikishin+22)
--reset-alpha A              1.0 = full reinit of the scope, <1 pulls partway
                             (Calibrated Partial Resets, 2026)
--reset-scope last|critic|all  Ma+23 find the modules differ, so this is
                             separable on purpose
--save-buffer                item 4: save the replay buffer beside every
                             checkpoint and restore it on --warm-start
```

Item 3 is `action_saturation_k` in RewardWeights, **default 0.0 (off)**. Quartic
charge on `|a|`: nearly free through the usable band (0.06x at |a|=0.5), biting
only in the last 20% (0.66x at 0.9, full at 1.0). It attacks the brittleness
from the other side than the literature's failed attempt: instead of trying to
make the tanh bound differentiable (Shamass 2026, return -31.6 -> -195.5), it
moves the optimum off the bound.

Smoke-tested: LayerNorm rebuilds 5 trunks and forward/predict still work,
resets touch 3 layers and the model still predicts, dormant_fraction computes.
Suite at baseline: 163 passed, 7 pre-existing failures.

### Suggested run order when the machine is next free

```
# 0. the one that already exists -- just measure it
scripts/diag_stage_landing_rate.py --checkpoint out/bc_orbit_descent.zip \
    --stage orbit_descent --episodes 96 --seed0 41000

# 1. LayerNorm, from scratch up the corrected ladder
scripts/train_sac_isaac.py --steps-per-stage 200000 --checkpoint-every 50000 \
    --out-dir out/sac_v34_layernorm --layer-norm --gamma 0.998 --max-rocks-per-env 0

# 2. resets, from the 22.9% checkpoint
scripts/train_sac_isaac.py --only-stage ramp_35m --steps-per-stage 300000 \
    --checkpoint-every 50000 --out-dir out/sac_v35_resets \
    --warm-start out/sac_training_run_v24_vert/snapshots/ramp_35m_1550000_steps.zip \
    --reset-every 50000 --reset-alpha 0.8 --reset-scope last \
    --gamma 0.998 --max-rocks-per-env 0

# 3. action-saturation charge (needs the weight switched on)
#    --reward-weight action_saturation_k=2000

# 4. buffer kept across the warm start
#    add --save-buffer to both the run that produces the checkpoint and the one
#    that warm-starts from it
```

Evaluate with 96 episodes on seeds not used for selection. 24 is a smoke test.

# ================================================================
# FOUR-AGENT AUDIT, 2026-10-06 — the measurement was broken
# ================================================================

Seven hyperparameter interventions had failed, all of them invariant. That
pattern says "something below the hyperparameters is wrong", so four agents were
pointed at four areas hyperparameters cannot reach. Three of them found real
defects and one of those changes the project's headline number.

## THE BIG ONE: the touchdown label was inflating every RL rate ~4.5x

Termination was graded from the state read AFTER all `_n_substeps` PhysX
substeps. A control step is `dt_s=0.05 s`, so the vehicle travels `0.05*|vz|` m,
while `TOUCHDOWN_CONTACT_EPS_M` is a fixed 0.05 m. **Above 1 m/s the contact
band is narrower than one step of travel**, so contact happens mid-loop, and the
regolith is authored with `restitution=0.0` -- PhysX has already stopped the
vehicle before the grader looks. Roughly `1/|vz|` of fast impacts were caught
honestly; the rest were graded at `vz ~ 0`.

So `landed_safely` passed slams, and `_touchdown_severity` -- the function whose
only job is "how many times over its limit was this touchdown" -- returned ~0
for an arbitrarily hard impact.

**Measured consequence.** The v24 checkpoint, the one this file has been calling
the 22.9% baseline:

```
                                   landed      timeout
old (post-substep) grading         22/96 = 23%   22/96
FIXED (pre-substep) grading         5/96 =  5%    8/96
```

17 of the 22 "landings" were slams. The real ramp_35m figure is **5%**, and the
controller's **85%** stands, because it flies its terminal descent at 0.80 m/s
(0.04 m of travel against the 0.05 m band) -- inside the honest regime. An RL
policy exploring faster descents was graded by coin flip.

### What this invalidates

Everything measured through `landed_safely` before this fix, which is every rate
in this file. In particular, today's hyperparameter comparisons are confounded:

  - the 11.5% -> 22.9% reward result (p~0.037)
  - the ramp_35m progression across v21/v23/v24
  - the gamma, replay-ratio, learning-rate, entropy and log-std comparisons
  - the proxy-vs-true correlation (r=+0.84, r=+0.92) and the training-time
    7.5%/12.5% figures -- the "true" column was the broken label

The REWARD-SHAPE findings survive, because they were proven by direct
computation rather than by training outcome: the lateral term was provably flat
above 7.07 m/s near the ground, the clip was provably filled by one term, and
the vertical channel was provably 12x weaker than the lateral one. The margin
DISTRIBUTIONS also survive as distributions, but the pass/fail verdicts on them
do not.

Fixed by snapshotting vz/v_xy/tilt/|w| before the substep loop and grading from
that, in both envs, including the leg-force kinetic energy. Costs no extra reads;
off by at most one step of acceleration (~0.08 m/s against a 1.0 m/s limit)
instead of by the whole impact velocity. Regression test added.

## The behaviour clone flies AWAY, and the loss cannot see it

The ramp_35m clone (4/96, 63 timeouts) does not hover -- its vertical feedback
gain has the **wrong sign**:

```
                 d(throttle)/d(vz)        d(throttle)/d(alt)
teacher               -0.98                    -0.047
clone 5-10 m          +0.088                   +0.024
clone 10-20 m         +0.306                   +0.010
clone 20+ m           +0.247                   +0.008
```

Positive is positive feedback: with `d(az)/d(a0) = 0.504 m/s^2` the growth rate
is +0.154/s, i.e. 47x over a 25 s episode. The measured trace does exactly that
-- descends to 18 m, crosses vz=0 at t=9 s, runs the throttle to +0.68, climbs
back to 55.6 m at +4.75 m/s.

The throttle bias hypothesis was real but is NOT the cause: the +0.0152 bias
(43 sigma) is 29x too small, and adding exactly it to the TEACHER's action
changes nothing (23/24 still land).

**Root cause is the expert data's geometry.** On the kept trajectories
`corr(alt, vz) = -0.98`, and vz never exceeds -0.80 m/s -- not one ascending
state. On that near-1D manifold MSE is nearly invariant to how throttle is
apportioned between collinear regressors, so a sign-flipped partial is almost
free. A per-transition validation split cannot see it either, because adjacent
steps of one episode land on both sides. This is why MSE 0.068 and a mean action
error of 0.0485 -- numbers that look like a good fit -- coexisted with 4%.

**Fixed and verified.** `--dagger-iters` added to `train_bc_from_demos.py`:
roll the clone out in the ANALYTIC env (which reproduces the Isaac failure at
~20 s of CPU per 24 episodes), relabel every visited state with the teacher,
aggregate, refit. Three iterations took `d/dvz` from +0.306 to **-0.96..-1.54**,
i.e. the teacher's own gain. Noise injection alone does NOT work -- the teacher
drags the state back onto the manifold and `corr(alt, vz)` stays at -0.98.

`--gain-check` (on by default) finite-differences both gains per altitude band
and REJECTS a positive one. It takes seconds and catches the entire defect class
that the loss is blind to. Verified: it passes the DAgger clone and fires on
`out/bc_ramp35.zip`. Note it also separates the two BC failures -- the
orbit_descent clone's gains are all NEGATIVE (-1.2..-1.7), so that 0/96 is a
different problem (42% teacher, 20 episodes, longer horizon).

## Also fixed

  - `info["action"]` is now written by both envs. `reward.py` read it and
    nothing wrote it, so `_action_saturation_penalty` was identically zero --
    which VOIDS the v34 experiment and the conclusion that the penalty "failed
    and made things worse". That 8% was variance on a disabled term.
  - the dead `obs[15]` slot (`leg_force_frac`, hardcoded 0.0 everywhere) now
    carries TIME REMAINING. The envs truncate on the clock and charge 105
    against a 225-450 landing bonus, so two bit-identical observations had
    returns differing by ~105 on an unobservable variable. Caveat from the
    audit: a clock is ALSO collinear with alt/vz on expert trajectories, so this
    helps the MDP but does not by itself fix a BC gain sign.
  - a `NameError` of mine: `step_wait` binds the batch as `a`, so the vec env's
    `info["action"]` needed `a[i]`.

## Clean bills of health (verified against SB3's own source, not from memory)

  - episode boundary handling: `info["terminal_observation"]` is set BEFORE the
    reset and SB3 reads exactly that key, so the critic never bootstraps across
    a boundary
  - terminated vs truncated: `TimeLimit.truncated` plumbed correctly, so a
    timeout bootstraps and a crash does not
  - observation normalization: one static `OBS_SCALE`, no `VecNormalize`,
    identical element-for-element between the vec and single envs
  - the landing criteria and spawn block are literally the same code (the vec
    env imports them), byte-identical after normalizing away `[i]` indexing

## Known, recorded, NOT yet fixed

  1. **Replay buffer capacity is `buffer_size // n_envs` = 62,500 rows, not
     1e6.** The demo-seeding comment is wrong by 16x, so demos WERE evicted --
     which invalidates the earlier conclusion that demo bootstrapping did not
     help.
  2. **`left_tile` is both bootstrapped as a truncation and charged the timeout
     penalty.** On some stages it is the dominant ending, so a large share of
     terminal targets are extrapolations from edge-of-map states.
  3. **Evaluation uses one fixed terrain (seed 7) for every episode** while
     training regenerates it per reset. Every headline rate is a single-tile
     measurement; more episodes do not average that out. Site flatness alone
     caps ramp_20m at 91.5%.
  4. **A warm start without `--save-buffer` refits the critic on 10k uniform
     random transitions** before its first gradient step, because SB3 samples
     randomly below `learning_starts` and the buffer was just reset.
  5. **Rocks exist only in training** (vec env default 10/env, colliders the
     contact test cannot see; a rock top sits 0.65*d above ground against a
     0.05 m band). `--max-rocks-per-env 0` is effectively mandatory and was
     passed on every run today, but the DEFAULT is 10 and run args are not
     persisted next to checkpoints, so older runs are unauditable.
  6. `env_spacing_m` is sized off the largest stage, so env 15 sits at
     x ~ 37.8 km where float32 spacing is 3.9 mm -- 8% of the contact band.

## DAgger WORKS: 31% on ramp_35m, 6x the best RL policy

All three measured on the same 96 episodes (seed0=41000) under the CORRECTED
touchdown grading:

```
plain BC clone (collinear expert data)      4/96 =  4%
best RL policy (v24, the ex-"22.9%")        5/96 =  5%
DAgger clone (3 iterations)                30/96 = 31%
ZemZev controller                          41/48 = 85%   (measured under the OLD
                                                          grading -- being redone)
```

This is the first artefact in the project that substantially works, and the
margin over RL is not a hyperparameter — it is data geometry. Plain cloning
learned the SIGN of the vertical feedback gain wrong (+0.306 against the
teacher's -0.98) because expert trajectories are collinear (corr(alt,vz) = -0.98,
no ascending state anywhere) and MSE on that manifold is nearly indifferent to
the sign. DAgger breaks the collinearity by asking the teacher what to do in the
states the CLONE visits; the gain goes to -0.96..-1.54 and the rate goes 4% ->
31%.

Checkpoint: `out/bc_ramp35_dagger.zip`. Reproduce with

```
.venv/bin/python scripts/train_bc_from_demos.py \
    --demos out/zemzev_ramp35_demos.npz --stage ramp_35m \
    --epochs 400 --dagger-iters 3 --dagger-episodes 24 \
    --out out/bc_ramp35_dagger.zip
```

Caveat, from the harness audit: this is a SINGLE-TERRAIN measurement (the eval
script reuses one tile at `--terrain-seed 7` for every episode), so read it as
"31% on seed 7", not as an estimate over the training distribution.

Two things this makes possible that were not possible before:
  1. A warm start whose critic can actually see landings. Every RL run so far
     started from a policy that lands 5% of the time, so the critic almost never
     sampled a success. 31% changes that materially.
  2. A clean isolation of the RL question. If RL started from this and still
     collapsed, "the policy was bad to begin with" would no longer be available
     as an explanation.

## CORRECTION: the "22.9% -> 5%" headline was wrong, and the cause was my own confound

Two changes landed in the same commit -- the touchdown-grading fix AND the
repurposing of observation slot 15 from a dead constant 0.0 to time-remaining --
and the v24 re-measurement attributed the whole drop to the grading. It was the
other one. Isolated with `--legacy-obs15`:

```
                                                        landed
old grading, old observation                            22/96 = 23%
FIXED grading, old observation (--legacy-obs15)         21/96 = 22%
FIXED grading, NEW observation                           5/96 =  5%
```

**The grading fix moved exactly one episode.** The 17-episode collapse was a
trained policy being fed a 1.0 -> 0.0 ramp in a slot it had only ever seen as
zero, which it loads without complaint because the vector stayed 16-wide.

So:
  - the ~22% ramp_35m baseline STANDS
  - today's hyperparameter comparisons are NOT invalidated by the grading fix,
    contrary to what the previous section claimed
  - the touchdown label bug was real as a mechanism, provable by computation,
    and its effect on THIS checkpoint's measured rate was negligible, because
    its successful landings were already slow. It will matter for a policy that
    arrives fast, and the reward-side consequence (severity flat above 1 m/s,
    making a slam cheaper than a timeout) was severe regardless of the rate.

The error was mine and it is the ordinary one: two variables, one measurement,
one explanation. It was caught only because the harness audit flagged the
observation-vintage trap independently.

**Standing rule from this:** never change the observation layout and anything
else in the same measurement. `--legacy-obs15` exists so a pre-change checkpoint
can be scored on the input it was trained with; use it for every checkpoint
produced before 2026-10-06, and note that every demo set on disk also carries
0.0 in slot 15.

## Where things actually stand, measured honestly (2026-10-06, post-audit)

All on 96 episodes, seed0=41000, ramp_35m, FIXED grading, correct observation
vintage (`--legacy-obs15` for every pre-change checkpoint), exclusive counters:

```
                          landed        crash   timeout
best RL policy (v24)    21/96 = 22%       53      22
DAgger clone            36/96 = 38%       47      13
ZemZev controller       22/24 = 92%        2       0    (24 eps, same grading)
```

CRASH dominates the failures everywhere, which the old summary line could not
show. The project spent today reading "the policy will not descend" off a
counter that hid two thirds of the outcomes; it descends and misses on precision.

And the reference ceiling is NOT 100%. Isaac controller baselines, post-fix,
24 episodes each:

```
ramp_20m 96%   ramp_20m_fast 71%   ramp_35m 92%   ramp_50m 100%
ramp_100m 100%  ramp_150m 100%     orbit_descent 46%
```

Every one of the 23 failures across that whole sweep is the same mode: a
correctly-executed touchdown rejected on lateral speed. `safe_landing_v_xy_m_s`
is 1.2 and the controller's terminal v_xy sits at 0.5-1.3, straddling it. So an
RL rate on ramp_20m_fast should be read against 71%, and on orbit_descent
against 46%, not against 100%.

## Today's three curriculum changes: confirmed in Isaac, plus three more fixed

The Isaac re-validation measured the PRE-change configuration rather than
inferring, and the kinematic reasoning transferred exactly: ramp_100m at T=40
gave 0/24 with 24 timeouts at 4.27 m of final clearance against the analytic
prediction of 4.45 m. ramp_150m likewise (4.58 m). ramp_20m_fast at the old
(2.0, 5.0) scored 4/24, not the 1/24 the analytic audit reported -- so
"impossible" was overstated, but 17% is unusable as a rung and the direction and
magnitude of the change were both right.

Three further defects, now fixed:
  - **ramp_50m had ZERO budget slack.** Over 48 controller episodes touchdown
    times were 29.6 s median with a max of EXACTLY 30.0 s and 9 of 48 at
    >= 29.8 s -- one landed in the final control step. `spawn_v_z_m_s`
    -3.0 -> -3.4.
  - **ramp_100m's tile was 4.1 m from its own truncation boundary** (311.2 m
    measured track against 315.3 m). It was the only stage never re-derived from
    a measured track; every other carries 2.1-2.5x. 640 -> 800 m.
  - **ramp_20m_fast's tile was ~5x oversized** after its lateral release was cut,
    and because `terrain_grid_n` is global that gave it 3.25 m cells against
    ramp_20m's 1.5 m -- the stage meant to add ONE dimension was also flattening
    the terrain. 260 -> 120 m.

**The reachability rule in this file was too lax by ~1.9x.** Worst-case touchdown
time is a tight multiple of the free-coast time at every stage (h0/|vz0| vs
t_td_max: 1.69, 1.70, 1.82, 1.79, 1.83, 1.87), so the usable form is
`T >= ~1.9 * h0/|vz0|`. The "~2x h0/T" heuristic already used for
`spawn_v_z_m_s` was right; the bare `>= h0/T` written elsewhere in the comments
is what let an unreachable stage through. All seven stages satisfy the corrected
rule now.

# ================================================================
# THE RECIPE THAT WORKS (2026-10-07)
# ================================================================

After twenty interventions that each failed in the same way, the problem turned
out not to be learning rate, critic quality, exploration, reward shape or
optimiser. It was that **nothing held the policy anywhere**. Every intervention
changed how fast a warm-started policy left the good region; none changed
whether it left.

Mechanism, measured: these policies are ~96% saturated, so competence depends on
the pre-tanh mean being large. Moving it from ~5 to ~1 takes actions from 1.0 to
0.76 -- tiny in parameter space, catastrophic in behaviour -- and any optimiser
covers that distance in the 10,000 updates that fit in one 10k-step window. The
collapse was identical under Adam and SGD, at every learning rate, with a
bounded critic and a diverging one, across a 100x range of reward scales.

The fix is the standard offline-to-online constraint (TD3+BC, AWAC), applied as
a proximal pull rather than a loss term so SB3's `train()` needs no surgery.

## Working configuration on ramp_35m

```
scripts/train_sac_isaac.py --only-stage ramp_35m \
    --steps-per-stage 400000 --checkpoint-every 50000 \
    --out-dir out/<name> \
    --warm-start out/bc_ramp35_dagger_cov.zip \
    --gamma 0.99 --max-rocks-per-env 0 --ent-coef 0.005 \
    --learning-rate 3e-4 --actor-lr 3e-5 --bc-anchor 1e-2 \
    --reward-weight reward_total_scale=0.1 \
    --log-std-max 0.0 --onpolicy-warmup --critic-only-steps 40000 \
    --proxy-log-every 10000
```

Training-time landing rate (stochastic, so the deterministic figure is higher):

```
release           47.4%
 60k               5.6%   <- the dip the anchor cannot prevent, only survive
 70k-110k      20-30%
140k              61.5%   <- above the policy it started from
380k-390k      58-62%
```

Each flag earns its place, and the ones that do NOT appear were measured and
dropped:

  --bc-anchor 1e-2     THE one that matters. At 1e-3 the policy still dips to 0%
                       and only climbs back to ~13%; at 1e-2 it recovers past its
                       own start.
  --gamma 0.99         ends the critic divergence outright (Q stayed in [-14,+6]
                       where every other setting ran away: +100, -171, +1972,
                       +109). Reachable only because the shaping ranks landing
                       above stalling on its own (+2.64 shaping-only), so the
                       terminal is allowed to discount away.
  reward_total_scale   0.1. At 1.0 returns are O(400) and the critic diverges; at
                       0.01, combined with gamma 0.99, the value range collapses
                       to ~1 and the gradient is noise.
  --critic-only-steps  lets the critic fit before the actor chases it. Necessary
                       but nowhere near sufficient on its own.
  --onpolicy-warmup    SB3 samples uniformly at random below `learning_starts`,
                       so without this a warm start refits its critic on pure
                       random-policy data.
  --log-std-max 0.0    stops entropy pressure re-inflating the std into tanh
                       saturation.

## Pipeline from scratch for a new stage

```
# 1. controller demos IN ISAAC (not the analytic env -- they disagree)
scripts/collect_zemzev_demos.py --stage <stage> --episodes 48 --seed0 41000 \
    --out out/zemzev_<stage>_demos.npz

# 2. clone with DAgger and the gain check (both are load-bearing)
.venv/bin/python scripts/train_bc_from_demos.py \
    --demos out/zemzev_<stage>_demos.npz --stage <stage> \
    --epochs 400 --dagger-iters 3 --out out/bc_<stage>_dagger.zip

# 3. RL with the configuration above, --warm-start out/bc_<stage>_dagger.zip
```

Step 2 is not optional. Plain cloning learns the SIGN of the vertical feedback
gain wrong (+0.306 against the teacher's -0.98) because expert trajectories are
collinear, and the loss cannot see it: MSE 0.068 with a mean action error of
0.0485 coexisted with a 4% landing rate. DAgger takes the gain to -0.96..-1.54
and the rate from 4% to 58%.

## Reference ceilings, REAL Isaac, post-grading-fix

```
ramp_20m 96%   ramp_20m_fast 71%   ramp_35m 92%   ramp_50m 100%
ramp_100m 100% ramp_150m 100%      orbit_descent 46%
```

Every one of the 23 controller failures across all seven stages is a correct
touchdown rejected on lateral speed, with its terminal v_xy straddling the
1.2 m/s limit. So RL rates are read against 92% on ramp_35m and 46% on
orbit_descent, not against 100%.
