# LunarSim — Handover (2026-10-01)

Read this fully before doing anything. Previous session's context got very
long; this is a clean handover so a fresh session can go DIRECTLY to fixing
the curriculum (see "WHAT TO DO NEXT" at the bottom) without re-deriving
everything above it. Do not re-run already-settled investigations (terrain
bug, force/torque calibration, observation normalization) — trust this doc.

## What this project is

LunarSim: an Apollo Lunar Module (LM) landing simulator on NVIDIA Isaac
Sim/IsaacLab, training an RL (SAC) policy to fly a physically-real lander
from a high-altitude/high-speed release to a soft touchdown on real
cratered terrain. There's also a fast CPU-only analytic env
(`AnalyticLanderEnv`) for quick iteration, but **the user has explicitly
said IsaacSim-only results matter** — do not report/tune against the
analytic env's numbers as if they were the real target; it does not
reflect real IsaacSim behavior (confirmed this session: a hand-designed
controller got 75% `landed_safely` on the analytic env but only ~40-50%
on real IsaacSim).

**The unsolved problem, unchanged in nature from before: the trained SAC
policy does not reliably achieve a genuinely safe landing on the
`orbit_descent` stage** (200m release altitude, 10-30 m/s horizontal
speed randomized per episode, 80m xy spawn radius, real terrain). It no
longer reliably crashes (see "real progress" below), but it has not yet
landed safely in any sanity-check/eval sample this session, despite ~13M
cumulative training steps on this one stage.

## Real progress THIS session (don't re-investigate these, they're fixed)

Four real, root-caused, fixed bugs, each verified by before/after
telemetry:

1. **Terrain x/y transpose bug** (`lunarsim/core/terrain/rocks.py`,
   `sample_height_at`): ground-height lookups had row/column swapped —
   invisible on flat/gentle terrain, real (~1m) wherever terrain is
   anisotropic (crater rims, slopes — exactly where a drifting descent
   touches down). Affected BOTH training envs and all eval/render
   scripts, since they all share this one function. Fixed.

2. **Braking-envelope term far too weak relative to the altitude-closure
   term** (`lunarsim/rl/reward.py`, `RewardWeights.t0`): at the old
   `t0=1.0`, the braking penalty was ~34x weaker than the "descend"
   incentive at alt=100m on a realistic trajectory — no real
   counter-pressure against falling fast until the vehicle was already
   inside ~30-50m of the ground. Raised `t0` 1.0→20.0 (with matching
   `braking_penalty_cap` 250→400 and `shaping_clip_abs` raises) so the two
   terms are comparable well before the danger zone. Verified
   numerically (see the `t0` field comment for the exact before/after
   table) — this was found from a **direct user observation** of a live
   training run ("çok hızlı iniyor, hızı tutmaya çalışmıyor") before any
   telemetry was even pulled, and the numbers confirmed it immediately.

3. **No observation normalization anywhere in the pipeline**
   (`lunarsim/rl/obs_norm.py`, new file): position offset (~0-80m+),
   altitude (~0-200m), velocity (~0-30+ m/s), angular rate, quaternion,
   fuel fractions were all fed raw/unnormalized into a standard SB3 MLP —
   2-3 orders of magnitude of scale mismatch. No `VecNormalize` anywhere.
   Added a fixed-scale (not running-statistics) normalizer, applied
   identically in `AnalyticLanderEnv`, `IsaacLanderEnv`,
   `IsaacLanderVecEnv`, and `scripts/isaaclab_policy_eval_capture.py`'s
   hand-built observation (4 call sites, all patched — check
   `lunarsim/rl/obs_norm.py`'s own docstring if a 5th call site ever gets
   added). **This made every pre-existing checkpoint stale** (trained on
   raw-scale inputs) — full curriculum was retrained from scratch after
   this fix.

4. **Attitude-hold term had no velocity-independent ceiling near the
   loss-of-control tilt cutoff** (`lunarsim/rl/reward.py`,
   `_attitude_hold_penalty`): the existing velocity-gated tilt term
   (correctly) makes tilt nearly free during high-speed braking, but
   nothing independently discouraged tilting all the way toward the 60°
   `loss_of_control_tilt_rad` cutoff itself (the feasibility controller's
   own successful braking only needs ~25-35°). Real telemetry showed
   `lost_control` rate climbing 0%→44%→69%→100% across successive
   training extensions of the SAME run, despite healthy critic_loss/
   ent_coef throughout — a genuine behavioral drift toward over-tilting,
   not a training-instability artifact. Added a steep, velocity-
   INDEPENDENT barrier (`tilt_cutoff_k`/`tilt_cutoff_power`/
   `tilt_cutoff_cap`) that's negligible through the controller-proven
   0-35° range but climbs sharply past 45-50°. **This fix alone took
   `lost_control` from 16/16 down to 1-6/16** across subsequent training
   runs — the single biggest improvement this session.

Also added this session, per direct user request, NOT yet validated by a
full training run to convergence (see "what to do next"):

5. **Gated xy-position-pull term re-added**
   (`lunarsim/rl/reward.py`, `_xy_position_penalty`, fields `x1`/`x2`/
   `position_alt_gate_m`/`xy_position_penalty_cap`): a PREVIOUS session
   had removed any pull toward `(target_x, target_y)` because a flat,
   always-on version fought the final touchdown unwind maneuver (see the
   `x5`/`x6` field comment's bug history #3 for that old incident). This
   new version is gated by altitude — full weight above
   `position_alt_gate_m=40.0`, fading linearly to exactly zero at
   touchdown — so it can't repeat that old conflict. Verified numerically
   (see the sanity-check table in the field comment). One ~2M-step
   training run was done after adding this; still 0/N `landed_safely` in
   that run's own 16-episode sanity check, but `lost_control` stayed low
   (6/16) and reward values clustered much more tightly than before
   (-2100 to -3200 vs. a much wider spread previously) — a soft positive
   signal, not yet a resolved one.

## Current best checkpoint

`out/sac_training_run/sac_lunar_lander_isaac_orbit_descent.zip` — ~13M
cumulative training steps (fresh curriculum from scratch after the obs-
normalization fix, then many `--only-stage orbit_descent --warm-start
<prev>` continuations). This is the END of the run that included fix #5
above (the xy-position term). Do NOT assume this checkpoint is
"finished converging" — `ent_coef` was still ~0.01-0.015 (low but not
zero) and the run was still healthy (not diverging) when it was last
saved; it was killed mid-continuation (not mid-save) to free the GPU for
other checks, so the saved file itself is a valid, complete checkpoint.

`out/sac_training_run/sac_lunar_lander_isaac_hover_only_easy.zip` and
`..._hover_only.zip` (saved 23:07/23:11 today, from the fresh post-fix
curriculum run): presumed solved, same as they always have been (these
stages use flat terrain + near-zero spawn speed, so none of this
session's 4 fixes should materially change their behavior) — **not
independently re-verified this session**, inferred only.

`out/sac_training_run/sac_lunar_lander_isaac_final_approach.zip` (saved
23:15 today): **checked this session, result is concerning and changes
the diagnosis** — `out/eval_snapshots/final_approach_check_v1/`
(35m altitude, 0-3 m/s, 15m radius, seed 7): `end_reason=max_episode_s`,
**never touched down at all**, timed out at t=47s. This is ONE seed, not
a confirmed rate, but it means the problem may not be isolated to the
`orbit_descent` curriculum jump — `final_approach` itself (much easier:
35m altitude, 0-3 m/s horizontal speed, vs. `orbit_descent`'s 200m/
10-30 m/s) isn't landing either, at least not on this seed. Two
non-exclusive possibilities to check before committing fully to the
speed-sub-curriculum plan below:
1. The "hover forever to dodge risk" terminal-reward-calculus problem
   (documented extensively elsewhere in `reward.py`'s own comments,
   supposedly fixed by moving `crash_penalty` below `landing_bonus_scale`)
   may be resurfacing — possibly because `shaping_clip_abs` has been
   raised repeatedly this session (900→3000→3200) alongside new terms,
   which could have shifted the per-step economics again without anyone
   re-checking the terminal-reward breakeven math against the NEW shaping
   magnitudes.
2. This could simply be one unlucky seed/terrain draw at `final_approach`
   (its own known failure mode historically was ROUGH TERRAIN under the
   landing pad causing a leg-height-diff failure, not a refusal to
   attempt landing at all — this "never touched down" result looks
   different from that old pattern).
Re-run with 2-3 more seeds before concluding `final_approach` is broken;
command to reproduce:
```
./scripts/run_isaaclab_policy_eval_capture.sh \
  out/sac_training_run/sac_lunar_lander_isaac_final_approach.zip final_approach_check_v2 \
  --spawn-altitude-min-m 35 --spawn-altitude-max-m 35 --spawn-xy-radius-m 15 \
  --spawn-speed-min-m-s 0 --spawn-speed-max-m-s 3 --tile-size-m 60 \
  --terrain-roughness-scale 1.0 --max-episode-s 45 --seed <different seed> --post-episode-s 2
```

## What's actually still broken — the CURRICULUM-SHAPE hypothesis

This is the live, untested hypothesis the user wants investigated NEXT,
and it's a genuinely different angle from anything tried before:

**Across the last several 2M-step `orbit_descent` training extensions
(cumulative ~13M steps on this stage alone), `landed_safely` stayed at
0/16 in every sanity check while `ent_coef` decayed to a low, stable
~0.01-0.02 and stopped moving.** That combination — low, flat entropy
+ zero improvement in the actual objective — is the signature of a
policy that has CONVERGED to a local optimum, not one that's still
searching and just needs more wall-clock time. Blindly launching more
2M-step extensions of the same thing is unlikely to help on its own
(this was tried repeatedly this session with no trend toward success).

**The specific suspected mechanism:** `orbit_descent` is reached via
curriculum warm-start from `final_approach`, whose spawn distribution is
0-3 m/s horizontal speed at 35m altitude — then `orbit_descent` jumps
straight to 10-30 m/s at 200m altitude. That's a cliff, not a smooth
progression (contrast with `hover_only_easy`→`hover_only`, which only
changes safety THRESHOLDS, not the physical spawn distribution). The
network inherits both weights AND a mostly-decayed exploration entropy
from `final_approach`'s "gentle hover and land" regime, then has to
discover a qualitatively different "commit to 25-35° of sustained tilt
and brake hard" strategy for `orbit_descent` — but by the time training
reaches that stage, there may not be enough exploration left to find it.

**Additional supporting evidence (3-seed direct eval, done right before
the xy-position fix, i.e. against a checkpoint with ~1-6/16 lost_control
already):**
- Seed A: touchdown, vz=-1.74 (near its 1.0 limit) and tilt=6.3° (well
  under 15°), but vxy=17.45 m/s — way over its 1.2 limit. Vertical/
  attitude control is fine; horizontal braking essentially isn't
  happening.
- Seed B: touchdown, vz=-15.0, vxy=28.7, tilt=24° — a near-free-fall hard
  impact. Nothing under control.
- Seed C: never touched down, timed out at t=62s.

Three distinct failure modes from the SAME checkpoint across different
release conditions (seeds sample different points in the 10-30 m/s / 80m
radius distribution) is itself evidence the policy hasn't found one
strategy that generalizes across the full release envelope — consistent
with "does okay at the easy/slow end, falls apart at the hard/fast end."

## WHAT TO DO NEXT (start here)

1. **Check `out/eval_snapshots/final_approach_check_v1/eval_summary.txt`**
   first (see above) — tells you if `final_approach` itself is actually
   solid or if there's ALSO a problem one stage earlier than
   `orbit_descent`, which would change the diagnosis.

2. **Implement a speed sub-curriculum WITHIN `orbit_descent`**, not just
   threshold relaxation (that pattern — `hover_only_easy`→`hover_only` —
   already exists and isn't the same fix needed here). Concretely: add
   one or more intermediate stages between `final_approach` and the real
   `orbit_descent` in `scripts/train_sac_isaac.py`'s `STAGES` list, e.g.:
   - `orbit_descent_slow`: same 200m altitude/80m radius/real terrain,
     but `spawn_horizontal_speed_m_s=(10.0, 15.0)` instead of
     `(10.0, 30.0)` — the easy end of the real range.
   - Then the existing `orbit_descent` stage (10-30 m/s) as the final
     stage, warm-started from `orbit_descent_slow` instead of directly
     from `final_approach`.
   - Consider a THIRD intermediate point (e.g. 10-22 m/s) if the jump
     from `orbit_descent_slow` straight to the full 10-30 m/s range still
     shows the same convergence failure.
   - Match `test_landing_feasibility.py`'s `orbit_descent` stage
     (`tile_size_m=600.0`, same terrain config) for the new stage(s) so
     they stay comparable/consistent with existing eval tooling.

3. **Verify via the SAME 3-seed-style direct eval pattern used this
   session** (not just the training script's own 16-episode sanity
   rollout, which samples the FULL 10-30 m/s range every time regardless
   of which sub-stage was most recently trained) — i.e. explicitly test
   at both the EASY end (10-15 m/s) and the HARD end (25-30 m/s) of the
   real range after each stage, to see whether the curriculum actually
   closes the gap at the hard end or just shifts where the policy gives
   up.

4. If the speed sub-curriculum genuinely helps, consider whether
   `ent_coef` needs to be prevented from decaying too far before each new
   sub-stage (e.g. SB3's `target_entropy` override, or simply accept the
   fresh replay buffer at each stage boundary naturally re-raises
   `ent_coef` somewhat, as was observed happening — check whether that's
   enough or needs an explicit nudge).

## Known-good operational patterns (unchanged from before, still true)

- Run training via `/home/haktan/isaac-env/bin/python
  scripts/train_sac_isaac.py` in the background (`nohup ... & disown`),
  tail/grep the log file to monitor. Two benign startup errors
  (`libxml2.so.2`, `omni.kit.tool.asset_importer`/`OmniClientWrapper`)
  are expected and NOT real failures — training proceeds fine after them.
  Don't let a naive `grep -q "Traceback"` re-trigger on these later in
  the log; anchor any error-grep to start AFTER the line containing
  `=== stage:` or `warm-started from`.
- Run eval/controller captures via `./scripts/run_isaaclab_policy_eval_capture.sh`
  / `./scripts/run_isaaclab_controller_eval_capture.sh` (docker-based).
  **Never run training and an eval/controller capture at the same time**
  — single shared GPU, causes OOM/silent black-frame renders. Kill one
  before starting the other.
- `eval_summary.txt` in each `out/eval_snapshots/<run>/` reports
  `landed_safely`, `touchdown_vz_m_s`, `touchdown_vxy_m_s`,
  `touchdown_tilt_deg` when `end_reason=touchdown` — always check these
  numbers, never trust `end_reason=touchdown` alone.
- All reward/env tests: `.venv/bin/python -m pytest tests/test_reward.py
  tests/test_rl_env.py -q` — keep passing before/after any reward.py or
  analytic_lander_env.py edit. (7 unrelated pre-existing failures in
  `tests/test_apollo_lm_asset.py` are a local `.venv` missing
  `PhysxSchema` — not something this session broke, don't try to fix it
  unless asked.)
- The user wants **IsaacSim results only** for anything reported as
  progress/success — do not use `AnalyticLanderEnv`/
  `scripts/train_sac_curriculum.py` (the CPU-fast trainer) as a stand-in
  signal; it was explicitly called out this session as not representative
  of real IsaacSim behavior.
- No video needed for routine eval checks per user request — reading
  `eval_summary.txt` (and raw `telemetry.csv` if finer detail is needed)
  is sufficient; only send a video if the user asks to see one directly.

## Nothing from this session is committed to git

Check with the user before committing.
