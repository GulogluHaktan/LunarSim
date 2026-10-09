# LunarSim — Handover (2026-10-09)

Read this fully before doing anything. It replaces the 2026-10-01 handover.
The short version: **the goal is not met**, every competent policy in this
project is a behaviour clone, and the reason RL cannot improve on one is now
measured rather than guessed.

---

## 1. Where the numbers stand

All deterministic, 96 episodes x 2 seed blocks (7000, 41000) = 192, terrain
seed 7, `scripts/diag_stage_landing_rate.py`.

### orbit_descent (the stage the goal is stated on)

| artefact | total | note |
|---|---|---|
| controller (ZemZev) | ~50% | the bar |
| SAC clone | 51.0% (98/192) | imitation |
| **PPO clone** | **52.6% (101/192)** | imitation, current best baseline |
| v77 SAC (from the 19.8% clone) | 27.1% | +7.3 over ITS clone — the only measured RL gain |
| v78 SAC @410k | 51.0% | +0.0 |
| v79 SAC @410k | 50.5% | |
| v86 PPO @1.5M | **54.2% (104/192)** | +1.6 over its clone, p=0.76 |
| v87 PPO @750k | 55.7% | peak of a DECLINING run — do not count |
| v87 PPO @1.5M | 46.9% | below its clone; trend -4.06 pts/1M |
| v88 clone (coupled schedule refit) | 39.1% | the refit broke the observation |
| v89 PPO (decoupled refit) | ~49.9% train-time, flat | |
| PPO from scratch | 0.0% (0/192) | |
| SAC from scratch | 0.0% | |

### land_handoff (the split's landing half)

| artefact | total | note |
|---|---|---|
| controller | 12.0% (23/192) | the bar on this stage |
| **landing clone** | **11.5% (22/192)** | matches its teacher exactly |
| PPO from it, 200k-800k | 10% per block | at or below the clone |
| SAC from scratch | 0.0% | |
| PPO from scratch | 0.0-1.2% | |

**Nothing in this table is a success.** The largest honest estimate is +5.2
points and the measurement cannot resolve it (see §3).

---

## 2. The measured root cause

**A learned Q(s,a) cannot rank actions on this task.** `scripts/diag_action_signal_to_noise.py`:

```
two independent policy samples differ by   0.0428   in action space
the full action box spans                  3.663 m/s^2 (11-tonne vehicle)
  -> one 0.05 s step turns that into       0.0039 m/s
dV/dv, from the realised return            0.1936 value units per m/s
  -> SIGNAL, one step                      0.00074
calibrated critic residual std (NOISE)     1.884
      horizon      signal      ratio
            1    0.000736   1 : 2559
          100    0.073615    1 : 26
          714    0.525611     1 : 4   (a whole episode)
```

And the LATERAL channel — which causes **94% of the failures** — is far worse,
because thrust is fixed along the body axis so lateral acceleration is four
integrations from the command (torque → angular accel → rate → tilt → accel):
a 0.05 s command tilts the vehicle 0.002° and the signal is **1 : 10,495,747**
under the noise.

That single fact explains, at once: the action gradient reading 52.6% toward
the expert (a coin flip), best-of-16 selection being WORSE than random on every
checkpoint, and CQL / Cal-QL / TD3+BC / PEX / LayerNorm / dual-gamma each
failing in turn — they were all improving properties of a quantity that never
carried the signal. It also explains why the analytic controller works: it
inverts the dynamics instead of estimating values.

**PPO was chosen from this measurement**, not by trying another algorithm: its
advantage is (realised return − V(s)), so it never needs Q(s,a), and the
on-policy estimator is quieter AND unbiased (`diag_onpolicy_noise.py`):
critic residual 1.884 *systematic*, realised-return spread 0.266 *zero-mean*.

---

## 3. What the measurement budget can and cannot say

`scripts/diag_power.py`. At 192 episodes per arm, the smallest resolvable
effect is **~14 points** and the 95% interval on a difference is **±10 points**.

- v86's +1.6 points would need 15,258 episodes per arm. It was never resolvable.
- So "statistically indistinguishable", said a dozen times in the last session,
  means *the measurement cannot say*, not *RL added nothing*. Every RL result
  here is consistent with anywhere from −10 to +10 points.
- The target itself IS measurable: 50% → 75% is +25 points, which 166
  episodes/arm resolves comfortably. The unmeasurable part is the +2 to +5
  point steps on the way.

**The protocol, declared in advance** (`scripts/run_highpower_validation.sh`):
pick the checkpoint on block **7000 alone**, never on the pooled number and
never on the training log; score it on K **fresh** blocks. ±14.1 at one block,
±10.0 at two, ±7.1 at four, ±5.0 at eight. Do not report a gain without it.

---

## 4. What has been ruled out

- **Imitation is not the bottleneck any more.** More demos took the clone from
  19.8% → 52.6% (48 → 255 landed episodes). Its held-out action error is 0.059
  against an observation floor of ~0.047, so ~25% of the imitation gap is left,
  not 80%.
- **The reward is dense and continuous.** `diag_reward_continuity.py`: 0
  discontinuities over 8000 states × 11 dimensions, 100% response in every
  velocity channel.
- **Three reward changes with MEASURABLE improvements to the reward all failed
  to move the landing rate**: altitude-normalised shaping (decisive band's share
  1.5% → 16%) trended −4.06 pts/1M; the schedule refit coupled (2-5 m outcome
  AUC 0.516 → 0.746) took the clone to 39.1%; decoupled, it sat flat.
- **Horizon length is not the obstacle.** land_handoff has a 150-400 step
  credit distance, a weak 11.5% clone, 88 points of headroom and n≈92 episodes
  per reading — and the trained policy measures 10% against the clone's 11%.
- **The task is feasible and the stage is not mis-sized.** 100% of
  land_handoff's entry distribution is vertically feasible (T/W 2.52, worst
  corner needs 5.1 m of the 8 m available), and a vehicle entering at 6 m/s
  travels 17 m while braking against 50 m to the tile edge. Fuel is at 96% at
  touchdown, so braking harder is always available.
- **Terrain observability is not the ceiling.** 0 of 17 failures bind on
  `leg_diff`; 94% bind on `v_xy` at touchdown (median overshoot 1.39× against
  a 1.2 m/s limit).

---

## 5. The pattern every RL run shows

Measured with `diag_policy_vs_expert.py --vs <its own clone>`. A random walk
grows as sqrt(n); a directed move grows linearly.

| run | 250k | 500k/600k | 1M-1.5M | verdict |
|---|---|---|---|---|
| v86 (γ=0.999) | 0.0170 | 0.0240 | 0.0386 | sqrt(n) exactly — **diffusion** |
| v87 (altitude norm) | 0.0159 | 0.0223 | 0.0572 | 1.47× sqrt(n) — directed, toward WORSE |
| land specialist | 0.0152 | 0.0276 | 0.0322 | sqrt(n) — **diffusion** |

So the policies move as far as their clone's entire remaining error and gain
nothing. v87 is the informative exception: a reward that separates landing from
crashing *half as well* (`diag_reward_separation.py`) produced a *stronger*
gradient pointing at worse landings. A stronger gradient on a worse objective.

**Use `diag_reward_separation.py` before spending a run on a reward change.**
v87 was predictable from a table that takes a minute.

---

## 6. State of the machine

Nothing should be running. Check with
`ps -eo cmd | grep -E "train_sac_isaac|diag_stage" | grep -v "grep -E"`.

The last pipeline (`land_warm.sh`) was finishing its block-7000 selection sweep:
200k/400k/600k/800k all 10/96. The 1M cell may still be in flight.

Artefacts worth keeping:
- `out/bc_ppo_orbit_v2.zip` — the 52.6% PPO clone (policy + value head, γ=0.99)
- `out/bc_ppo_orbit_g999.zip` — same policy, value head at γ=0.999
- `out/bc_orbit_big.zip` — the 51.0% SAC clone
- `out/bc_ppo_land.zip` — the 11.5% land_handoff clone
- `out/ppo_v86_g999/snapshots/orbit_descent_1500000_steps.zip` — the 54.2%
- `out/zemzev_orbit_demos_s2[0-3]000.npz` + `..._v20.npz` — 576 episodes, 303 landed
- `out/zemzev_band25.npz` — the 0-25 m band, 61,837 transitions

---

## 7. Diagnostics built (use these before running anything)

| script | answers |
|---|---|
| `diag_action_signal_to_noise.py` | can a learned Q rank this policy's actions? (no) |
| `diag_onpolicy_noise.py` | is the on-policy estimator quieter? (7×, and unbiased) |
| `diag_reward_separation.py` | does a reward change help the reward tell landing from crashing? |
| `diag_critic_vs_true_value.py` | Q against the realised two-stream reward-to-go, per altitude band |
| `diag_policy_vs_expert.py --vs` | how far a run moved from its clone; diffusion vs learning |
| `diag_training_trend.py` | a run's landed-rate slope WITH a confidence interval |
| `diag_power.py` | what a given episode budget can resolve |
| `diag_reward_continuity.py` | cliffs vs kinks over the whole envelope |
| `diag_shaping_weight_by_altitude.py` | where the shaping reward lives |
| `diag_field_schedule_fit.py` | does the field's schedule match what lands? |
| `diag_profile_landed_vs_failed.py` | where landed and failed trajectories separate |
| `diag_best_of_n.py` | can the critic rank in-distribution actions? (no) |
| `diag_handoff_envelope.py` | where a landing specialist succeeds, by entry condition |
| `diag_controller_stage_rate.py` | the controller's rate on ANY stage, on real Isaac |

---

## 8. The trap this project keeps falling into

**Nine times in the last session a mechanism flag was accepted and silently
dropped**, and each one would have been read as "the method does not work":
`--layer-norm` (twice: ignored on warm start, then built on CPU so dead on
GPU), the whole `CQLSAC` class (v68's 18% was plain SAC), `--altitude-replay`
(computed and thrown away on the warm-start path), `ent_coef` (missing from
`PPO.load`'s kwargs), `--reward-weight` on the slot-repair script (defaults
used silently), and `--reward-weight` on the EVALUATION script (which read 7%
for a clone whose real rate was 39.1%).

**The rule:** assert on the OBJECT, never print an attribute you just set.
A print of a value you assigned proves nothing.

**And the one that costs most:** a parameter that feeds BOTH the reward and the
observation (anything in `target_velocity`) must be passed at *every* entry
point — trainer, demo repair, evaluation, demo collection. Slots 16-19 are
`target_velocity()` evaluated at the current state; shifting them under a
trained policy took this clone from 22% to 2% once, and from 52.6% to an
apparent 7% a second time. `--obs-field-default` now lets the reward's field
and the observation's field differ on purpose.

Also: `pgrep -f` self-matches (use `ps -eo cmd | grep -F`); zsh does not
word-split `$VAR` (use arrays); piping a long-running command through `tail`
hides its output until it exits.

---

## 9. Open leads, in the order the measurements favour

1. **The guidance field does two jobs and they conflict.** Refitting the
   descent schedule onto what lands improves the reward and destroys the
   observation feature, because `vz - vz_target` is informative BECAUSE the old
   target is offset — a target that tracks the behaviour perfectly makes the
   feature ~0 and subtracts the signal out. `--obs-field-default` decouples
   them; v89 tried it and was flat, but the *observation* side is unexplored:
   what feature SHOULD slot 17 carry?
2. **Compose the split and measure it where the goal lives.**
   `diag_stage_landing_rate.py --handoff-checkpoint X --handoff-alt 25` flies a
   specialist below 25 m and an approach policy above, on orbit_descent. The
   specialist is currently no better than its clone, so this is on hold — but
   the harness exists and `diag_handoff_envelope.py` maps where a specialist
   succeeds, which is what an approach policy's terminal reward should be built
   from.
3. **n-step returns.** `NStepReplayBuffer` exists and has never been used. The
   action-SNR table says the signal grows with the horizon over which an
   action's effect accumulates; this is the one estimator change that attacks
   the numerator and has not been tried.
4. **Action repeat.** The lateral channel's leverage grows as k³ in the hold
   length (0.002° at one step, 0.79° at 1 s, 19.7° at 5 s). It must be
   implemented INSIDE the env, not as a VecEnvWrapper — a wrapper discards the
   first k steps of episodes that terminate early, and those are the release
   state. Note recorded in `diag_action_signal_to_noise.py`.
5. **Analytic / model-based policy gradient.** The reward is dense, continuous
   and tighter than the controller at every altitude; its slope is available in
   closed form through a simple thrust model. No critic, no bootstrapping.
   Biggest build, strongest measured justification.

---

## 10. What NOT to do

- Do not trust a training-time landing rate. n=12-31 on orbit_descent misled
  four separate conclusions. (n≈92 on land_handoff is better but still showed
  15.0% where the deterministic rate was 10%.)
- Do not pick the best checkpoint from a run. Two of five runs peaked
  mid-training and declined (v77 at 310k, v87 at 750k).
- Do not read a single seed block. The PPO clone looked 7 points better than
  the SAC clone on block 7000 and 4 points worse on 41000; pooled, +1.6.
- Do not enable `action_saturation_k`. Measured twice: it does not reduce
  saturation (95.8% → 96.2%) and it takes ramp_35m from 22.9% to 8%.
- Do not use `--stateless-tgo`. The controller's `tgo` is a carried integrator;
  re-solving it every step gives 0/200 landings because the estimate never
  shrinks and the ZEM/ZEV command scales with 1/tgo.
