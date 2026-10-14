# LunarSim — Handover (2026-10-09, evening)

Read this fully before doing anything. It replaces the earlier 2026-10-09
handover, and it **contradicts it on the root cause**. The short version:

- **There is now a measured RL gain**, on fresh pre-registered seed blocks:
  `land_handoff` 9.9% → 31.2%, +21.4 points, McNemar 41–0, p = 9.1e-13.
- The previous handover's central claim — "a learned Q(s,a) cannot rank
  actions on this task" — **does not hold**, and three of its load-bearing
  numbers were artifacts. §2 has the arithmetic.
- What actually blocked every earlier run was three things in the training
  setup, each independently sufficient, all measurable from the saved
  checkpoints. §3.
- The fix was an action-space change, not another algorithm. §4.

---

## 0. LATEST (2026-10-09 night) — read this first

**Isaac `orbit_descent`, the goal stage, 192 paired fresh episodes**
(`scripts/validate_guided_gain_isaac.py`, 16 waves x 12 envs, wave seeds
90000-105000, none seen in training): ZEM/ZEV controller **101/192 = 52.6%**,
guided-gain PPO at 30k decisions **131/192 = 68.2%**, **+15.6 points**,
95% bootstrap CI [+8.9, +22.9], McNemar 41-11, p = 3.6e-5, policy >= controller
in **16/16** waves. Checkpoint: `out/gg_isaac_orbit_v2_snap30k_frozen.zip`;
per-episode vectors: `out/gg_isaac_orbit_v2_30k_validation.json`. Not yet the
75% goal (policy 95% CI [61.5, 75.0]); the run `out/gg_isaac_orbit_v2` was
still training and read 79.2% on its selection block at 45k.

**That run used an outcome-only reward**: `--reward-weight reward_scale=0
--reward-weight touchdown_k=0 --reward-weight touchdown_penalty_cap=60
--reward-weight landed_step_k=100` (+5 for landing, 0 for a failed touchdown,
-3 timeout/tumble, no per-step shaping). Reason, measured with
`scripts/diag_reward_rescore.py` on 192 paired land_handoff episodes: the
shipped dense `vfield` shaping ranks policies AGAINST landing rate (the
31.2%-landing policy scored worse than the 9.9% controller), while the terminal
alone or a binary outcome ranks them correctly 100% of the time. Step bonuses,
`touchdown_penalty_cap` compression and `vfield_cap` clipping each FAIL to fix
the ranking at sane sizes. Literature agrees (Gaudet: flat failure + step on
the conjunction; HER: shaping "compromises the binary success metric"); the
identity J = F + p(S-F) explains it — make F policy-independent and return is
monotone in success rate. See `.cache/lit_reward_design.md`,
`.cache/lit_sac_critic.md`.

On `land_handoff` the reward change did NOT clearly matter: fresh-block rates
were 31.2% / 21.9% (shipped reward, two seeds), 31.8% (terminal-only), 24.5%
(outcome-only), all vs a 9.9% controller with ~zero regressions. Seed variance
there is as large as the reward effect. The stage, not the reward, binds: a
200-sample gain-space oracle reaches 31.2% (the earlier 24-sample "79% infeasible"
figure was undersampled — the oracle was still rising).

## 1. Where the numbers stand

### land_handoff — measured on four FRESH blocks (90000/120000/150000/180000)

192 paired episodes, terrain seed 7, `scripts/validate_guided_gain.py`,
analytic backend. Block 7000 selected the checkpoint and is excluded.

| arm | landed | rate |
|---|---|---|
| ZemZev controller (zero action) | 19/192 | 9.9% |
| **guided-gain PPO** | **60/192** | **31.2%** |

Per block: 5→17, 8→16, 5→15, 1→12 — every block improved.
McNemar exact: the policy landed on 41 episodes the controller did not, and
the controller on **0** the policy did not. p = 9.095e-13. Strict dominance.
Per-episode 0/1 vectors: `out/gg_land_v2_validation.json`.

The declared protocol asked for ±7.1 points at four blocks; the effect is
+21.4. This is the first result in this project that clears its own bar.

### What the gain space can and cannot reach

`scripts/diag_gain_space_ceiling.py` draws N random gain vectors per episode
and asks whether ANY lands — the ceiling for a *state-independent* gain
policy, and a lower bound on a state-dependent one.

| stage (analytic) | controller | any-of-N | no sample ever landed |
|---|---|---|---|
| land_handoff | 6.2% | 20.8% | **79.2%** |
| orbit_descent | 87.5% | 95.8% | **0.0%** |

So `land_handoff`'s apparent "88 points of headroom" is about 15: four fifths
of its episodes are **attitude-infeasible**. Its release draw reaches 21.1° of
tilt (`spawn_tilt_rad` is sampled independently per axis, so the diagonal is
0.26·√2) with up to −5 m/s of sink from 8 m — 1.6 s of flight — while the RCS
needs ~8.7 s to slew to the 20° final-approach cap at 2.3°/s. The trained
policy reaches 25% on the selection block, i.e. *above* the state-independent
ceiling, which is what state-dependence should buy.

**This reframes the previous handover's "strongest single constraint."** The
five identical 10/96 readings across 1M steps were taken on a stage whose
reachable ceiling is ~21%. They did not rule out horizon length, baseline
strength and measurement power together; they measured a stage near its limit.

### orbit_descent — the stage the goal is stated on

Controller bar, re-measured with the altitude bug in §5 fixed:

| harness | landed |
|---|---|
| `diag_controller_stage_rate.py`, single Isaac env | 24/48 = 50% — **all 24 failures are crashes**, 0 timeout, 0 lost_control, 0 left_tile |
| through the guided-gain training env (12 envs) | 27/48 = 56.2% |

A guided-gain run is in progress here (`out/gg_isaac_orbit_v1`). Score it with
`scripts/validate_guided_gain_isaac.py`, not with the trainer's own eval —
see §6 on why the trainer's eval can select but cannot score.

---

## 2. What the previous handover got wrong

Its causal chain was: a one-step action perturbation changes the return by
0.00074 while the critic's residual is 1.884, so the ratio is 1:2559, so no
learned Q can rank this policy's actions, so PPO was chosen because its
advantage needs no Q. Three independent defects, and the first two invert it.

**The noise is the wrong quantity, and the dominant part cancels.**
`diag_action_signal_to_noise.py` computes `std(Q(s, a_det) − V_expert)` across
states. The quantity that governs *ranking* is `std(Q(s,a₁) − Q(s,a₂))` at the
same state. Measured on the very checkpoint cited:

```
residual mean                                  +2.014   cancels exactly
residual std (the "1.884")                      1.907
R^2 of residual on quadratic state features     0.913   a smooth b(s); cancels
corr(Q(s,a1), Q(s,a2))                          0.9983
std(Q(s,a1) - Q(s,a2))                          0.0483   <- the real figure
```

39× smaller. The script never evaluates `Q(s,a₁)` at all.

**The signal is attenuated ~100× by pooling across altitude.** `dV/dv` is one
regression over all altitudes; per band it ranges 0.02 → 26.5 and flips sign.
In the 0–1 m band where the outcome is decided it is 137× the pooled value.
The same regression puts the *lateral* coefficient at 0.0007–0.011 while the
handover says lateral decides 94% of outcomes — both cannot be true.

**The lateral "1 : 10,495,747" figure appears in no script.** It truncates a
one-step tilt impulse at the end of the 0.05 s hold. But attitude is an
integrator and only the *rate* is damped (`angular_damping_per_s = 0.5`,
τ = 2 s), so one command deposits a permanent tilt offset:

```
Δtilt_inf = α·dt·τ = 2.78e-3 rad (0.159 deg)        ~80x the quoted 0.002 deg
a_lat     = g·Δtilt = 4.5e-3 m/s^2, for the REST of the episode
Δv_lat over ~35 s = 0.161 m/s                       ~1.7e5 x the value used
```

A single one-step lateral command is worth about an eighth of the 1.2 m/s
limit. This also means the lateral leverage of a held command is **linear in
hold length, not cubic** — so the old action-repeat sizing argument ("k=20
moves it four orders of magnitude") was wrong in the generous direction.

**And the decisive internal contradiction:** PPO uses no Q, and PPO failed
identically. A Q-noise explanation cannot cover that.

Two more numbers that were treated as physics and are harness artifacts:

- The **"~0.047 irreducible observation floor"** is a hardcoded string in a
  print statement (`diag_policy_vs_expert.py`). The k-NN estimate behind it is
  linear in neighbour distance and extrapolates to ≈0.007 at zero distance;
  median neighbour distance is 0.674 z-units across 20 standardised dims, so
  "near-identical observations" were not near-identical. A PPO clone already
  reads 0.0405, below its own alleged floor.
- The **controller's hidden state is one scalar** (`self.tgo`) and nothing
  else — no integrators, no filters, no latches. The project's own ablation
  attributes **13%** of the held-out BC residual to it, and the final-approach
  branch discards it entirely, so the regime where touchdown is graded is
  memoryless by construction. 0.059 is an imitation gap, not a floor.

What *does* survive: `diag_stage_landing_rate.py`'s outcome accounting and
overshoot ranking, `diag_critic_vs_true_value.py`'s +38 forensic decomposition
and its CQL-scale conclusion, the trap list in §7, and best-of-16 being
reliably worse than random — though that last one is a ranker pointed *away*
from the teacher (within-state Spearman −0.058, SE 0.013), which is
informative-but-misaligned, not "cannot rank".

---

## 3. What actually blocked the earlier runs

Three defects, each independently sufficient, all readable off the checkpoints.

**(a) `gae_lambda` threw away the terminal reward that `gamma` was chosen to
keep.** GAE weights step t+l by (γλ)^l, so the advantage — the only thing the
policy gradient sees — has horizon 1/(1−γλ).

| λ | advantage horizon | weight on the landing bonus at a 714-step release |
|---|---|---|
| 0.95 (shipped default) | 19.6 steps | 6e-17 |
| **0.99 (every real run)** | **91 steps** | **3.7e-04** |
| pure γ=0.999, no GAE | — | 0.4895 |

γ=0.999 was set deliberately so the bonus would be visible (`refit_ppo_value_head.py`
quantifies it as "the signal 565×"). λ=0.99 kept 0.037% of it. The policy was
climbing the dense shaping alone — and the dense shaping's decisive 0–5 m band
holds 2.0% of the reward mass (`vfield_norm_p` ships at 0.0).

**(b) PPO's exploration scale was inherited from the BC fit residual and never
moved.** `bc_to_ppo.py` sets `log_std = (mean_action − target).std().log()`:

```
bc_ppo_orbit_v2.zip              sigma = [0.0221, 0.0493, 0.0490, 0.0037]
ppo_v86_g999 @ 1.5M steps        sigma = [0.0221, 0.0494, 0.0491, 0.0037]
policy_std over the whole run    0.03106081 -> 0.03108847   (+0.0085%)
```

1.1–2.5% of the half-range, 0.19% in yaw, for 1.5M steps. The entropy bonus's
gradient on `log_std` is a constant −ent_coef per dim, which at 0.003 and
lr=1e-5 is orders of magnitude under the policy-gradient term and cannot widen
it. Wagenmaker et al. (arXiv:2512.16911) *prove* standard BC can fail the
action-coverage condition that is minimally necessary for RL fine-tuning.
Worse, the clone's mean action lies outside the box on ~25% of steps in both
attitude channels (max +2.599 on a [−1,1] box), where ~24% of samples are
clipped — so the gradient is computed on actions the env never executed.

**(c) 0.05 s control steps against a 6.3 s attitude timescale.** From the
code's own constants at the reachable mass (15103 kg at PDI, only 886 kg burns
in 60 s, so `I_tilt` ∈ [74392, 79029] — *not* the 57,666 in
`diag_action_signal_to_noise.py`), with damping included:

```
max RCS couple 1584.2 N.m,  alpha = 0.0200 rad/s^2,  w_ss = alpha/0.5 = 2.30 deg/s
time to 10 deg of tilt at FULL torque:  6.3 s        control step: 0.05 s
```

A 125:1 ratio, so one lateral braking manoeuvre is 250–380 consecutive
commands and independent per-step noise over that span cancels. This is a
theorem, not an observation: Tallec, Blier & Ollivier (ICML 2019,
arXiv:1901.09732) Thm 2 gives `Q_dt(s,a) = V_dt(s) + dt·A_dt(s,a)` — "if the
approximation error is much larger than O(dt), this error dominates, and the
ranking of actions given by the approximated Q is destroyed" — and their Thm 4
shows per-step i.i.d. exploration noise provably collapses to a deterministic
policy as dt → 0, which is the measured sqrt(n) "diffusion". Park, Kim & Kim
(arXiv:2111.03941) extend the divergence to the policy-gradient variance,
which is why switching SAC → PPO changed nothing.

Measured directly, `scripts/diag_exploration_coherence.py`, holding the policy
fixed and varying only the noise's time correlation at **identical** σ:

```
scheme (same sigma)        within-seed return std     vs white x1
white  x1 (what PPO did)            0.309                 1.0x
repeat x1 k=20                      0.947                 3.1x
ou     x1 tau=2s                    2.393                 7.8x
```

The within-seed std is the numerator of the policy gradient's SNR. PPO's
actual setting is the lowest row in the table. There was no signal to estimate.

---

## 4. The fix: modulate the controller, do not replace it

`lunarsim/rl/guided_gain_env.py` (analytic) and
`lunarsim/rl/guided_gain_vec_env.py` (Isaac, vectorized). PPO outputs
multiplicative modulations of `ZemZevGains`, held for `hold_steps` control
steps, while the controller keeps flying at 20 Hz.

Why this shape, and it is the literature's answer rather than a guess:

- **A zero action reproduces the controller exactly** — the map is
  `gain · ratio^action`. Verified bit-for-bit: identical return and outcome on
  24 orbit_descent seeds. The floor is the controller itself, not a 52.6%
  clone of it with 1.2% action coverage.
- The controller's inner PD attitude loop and its carried `tgo` clock stay
  inside the loop, so the policy never learns the four-integration
  torque→lateral-velocity chain, and the unclonable hidden state is moot.
- 12–60 decisions per episode instead of 400–1200, so γ and λ can keep the
  terminal in the advantage (0.41–0.70 instead of 0.00037).
- Exploration is temporally coherent by construction — the PPO-compatible form
  of the correlated noise Thm 4 requires.

Precedent: Furfaro, Scorsoglio, Linares & Massari (Acta Astronautica 2020,
arXiv:2003.02182) learn `K_R`, `K_V`, `T_f` of this same ZEM/ZEV law under
zero-order hold. Kaufmann, Bauersfeld & Scaramuzza (arXiv:2202.10796) ran the
decisive action-space ablation at matched compute: setpoint-level commands over
an inner loop 0.6–12.3 cm tracking error, direct actuator commands undeployable
on hardware. Jiang et al. (arXiv:2407.15083) take **rocket landing from 8% to
97%** using the analytic controller as a guide policy rather than a BC target.
Metelli et al. (ICML 2020, arXiv:2002.06836) formalise the persistence.
Full sweeps: `.cache/lit_rl_landing.md`, `.cache/lit_estimator_snr.md`.

### What the policy learned, and it is interpretable

`land_handoff`, best checkpoint, mean gain over 600 decisions:

| knob | nominal | learned | near the ground |
|---|---|---|---|
| `kd_xy_final` | 0.700 | **0.371** | ×0.52 at 0–2 m |
| `max_tilt_cmd_final_rad` | 20° | 16.4° | ×0.73 at 0–2 m |
| `kp_vz_final` | 1.500 | 1.78 | ×1.34 at 0–2 m |
| `tgo_floor_s` | 3.00 | 3.87 | ×1.24 |

It learned to command **less** attitude, not more. The reason is the measured
limit: at v_xy = 1.67 m/s the nominal final-approach command is
`atan2(0.7·1.67, 1.62) = 35.8°`, clipped to 20°, and the RCS needs 8.7 s to
reach 20° inside a ~10 s window — the command was saturated and unreachable.
At `kd_xy_final = 0.371` it becomes 20.9°, i.e. just at the achievable limit.
The policy found the de-saturation fix, state-dependently, which no fixed
hand-tuned gain can do.

### Hyperparameters, and why each one

`scripts/train_guided_gain.py` / `scripts/train_guided_gain_isaac.py`.
`--gae-lambda 0.98–0.99` and `--gamma 0.99–0.995` so the terminal keeps 0.4–0.7
of its weight over the episode's *decision* count (§3a). `--log-std-init -1.0`
→ σ = 0.368, declared rather than inherited (§3b). `--hold-steps 10–20` =
0.5–1.0 s (§3c). A fresh policy, not a warm start: SB3 initialises `action_net`
with gain 0.01, so the initial mean action is ≈0 and the jump-start property is
free. Every one of those three is asserted **on the object** after construction.

---

## 5. Two live bugs found and fixed

**`_edge_penalty` was identically zero in the only env the trainer uses.**
`reward.py` reads the tile via `getattr(getattr(env, "tile", None), "size_m", 0.0)`,
and `IsaacLanderVecEnv` passed `reward_fn` an `_EnvView` whose `__slots__` had
no `tile` — so the lookup gave 0.0 and the term's first guard short-circuited.
Live in both other envs, dead in this one. The term exists because a learner
*already exploited* `left_tile` truncation paying no timeout penalty, making
flying off the map a free exit from a crash worth −6 to −24, and `edge_k` is
6400.0. **Every RL run in this project trained with that price set to zero.**
Fixed; pinned by `tests/test_vec_env_edge_penalty_is_live.py`.

**`diag_controller_stage_rate.py` fed the controller the body centre, not the
belly.** Its `_View` forwarded `state["z"]` straight through, and
`IsaacLanderEnv` places that at `ground_z + alt + 3.52`, so the controller
believed it was 3.52 m high (4.62 m at 15° tilt) while its ZEM/ZEV vertical
channel solved for touchdown that far below the ground. This script is the
source of the **"controller 12.0% on land_handoff"** bar, and that stage starts
at 8–25 m — the bias was 14–58% of the altitude budget, and it moved the
final-approach trigger from 8 m to a true 3.4–4.5 m, cutting the terminal
descent window from 10.0 s to 4.2–5.6 s. Fixed the same way
`collect_zemzev_demos._BellyStateView` already did it; `left_tile` is also no
longer miscounted as a timeout.

Still open, not yet fixed (each is a one-liner with a real consequence):

- `train_sac_isaac.py:703-710` compares the demo reward-weight fingerprint
  against **hard-coded defaults** rather than the run's own `--reward-weight`,
  so overridden runs silently seed the critic with defaults-computed rewards —
  the exact failure the guard's error message describes. Neither
  `repair_demo_guidance_slots.py` nor `recompute_demo_rewards.py` refreshes
  that metadata, so it can never fire. Five demo files on disk currently carry
  a fingerprint claiming defaults while two of them are a mirrored skew pair.
- `--dual-gamma`, `--mc-terminal` and `--altitude-replay` are silent no-ops on
  the **from-scratch** SAC path whenever `--cql-alpha` is 0 (`:1213` discards
  the kwargs), and all seven verification assertions live only in the
  warm-start branch. `--pex-temperature` is inert without `--warm-start`.
  `--net-arch` never reaches SAC at all. `--log-std-max` is inert for PPO.
- `NStepReplayBuffer` **does not exist** — the previous handover's lead #3 is a
  phantom. For the PPO line, n-step returns already exist as λ.
- `collect_zemzev_demos.py` has no `--reward-weight`, so the fourth entry point
  in the project's own four-entry-point rule still cannot be controlled.

---

## 6. Measurement: what changed and why it matters

**The design is paired and the test was not.** Both arms fly the same episode
seeds on the same terrain, so McNemar is correct and the unpooled
two-proportion z in `run_highpower_validation.sh` is not. At n=192 the z
reports ±10.0 points; the paired test at the observed 21.4% discordance
resolves far better. The project was discarding ~1.5–2× of its own power and
then citing the loss as grounds for "the measurement cannot say."

`scripts/validate_guided_gain.py` (analytic) and
`scripts/validate_guided_gain_isaac.py` (Isaac) both print the paired and the
unpaired test, and both **persist per-episode 0/1 vectors** to JSON.
`diag_stage_landing_rate.py` keeps only its printed summary, which is why "10/96
at all five checkpoints" can never be checked for whether it was the same ten
episodes — those vectors were never saved and the runs are gone.

**The Isaac trainer's own eval can select but cannot score.**
`IsaacLanderVecEnv` draws every spawn from one internal `_rng` inside
`_reset_one` and auto-resets each env as its episode ends, so a policy that
changes episode lengths changes the order of those draws and the two arms fly
different release states. `validate_guided_gain_isaac.py` fixes this with a
wave design: re-seed `_rng` to a declared value, reset all envs at once, and
collect exactly one episode per env, so re-running a wave under a different
policy reproduces the same releases exactly.

---

## 7. The trap this project keeps falling into

Unchanged and still the most valuable section. **Assert on the OBJECT, never
print an attribute you just set.** A print of a value you assigned proves
nothing; §5 is two more instances, and the from-scratch SAC path has four.

**A parameter that feeds BOTH the reward and the observation** (anything in
`target_velocity`) must be passed at *every* entry point — trainer, demo
repair, evaluation, demo collection. Slots 16–19 are `target_velocity()` at the
current state; shifting them under a trained policy took a clone from 22% to 2%
once and from 52.6% to an apparent 7% a second time.

Also: `pgrep -f` self-matches — it bit again this session, silently turning a
launch wrapper into an infinite loop (use `ps -eo args | awk`). Do not name a
scratch script `inspect.py`; it shadows the stdlib and breaks torch's import.

---

## 8. Open leads, in the order the measurements now favour

1. **Finish and score the Isaac orbit_descent run.** The bar is 50–56.2%, the
   analytic gain-space ceiling is 95.8% with **zero** infeasible episodes, and
   a quarter of random gain vectors land. Score with the wave validator.
2. **Reward/outcome divergence.** On land_handoff the landing rate rose
   6.2% → 25% while the mean return *fell* −28.9 → −38.7. The policy is paying
   dense-shaping cost to land more, which means the reward is not aligned with
   the metric. Worth fixing before any reward-based comparison is trusted — and
   `vfield_norm_p` shipping at 0.0 puts 2.0% of the reward in the band that
   decides 94% of failures.
3. **`residual_scale > 0`** — a small additive correction on the controller's
   own output, sized per Staessens et al. (arXiv:2110.02566). Plumbed and
   defaulting to off. This is the lever for failures no gain setting can reach.
4. **Attitude-setpoint action space** with the analytic loop underneath — the
   CTBR analogue from arXiv:2202.10796, one level more expressive than gains
   and still not end-to-end.
5. **Q-chunking** (arXiv:2507.07969), purpose-built for an offline prior plus
   online RL with temporally consistent exploration and unbiased n-step
   backups. The member of that family that has not been tried.

## 9. What NOT to do

- Do not plan from the previous handover's §2. See §2 here.
- Do not quote a baseline from a previous handover. Re-measure it through the
  harness you are comparing in, on the same seeds. Two of them were artifacts.
- Do not use `land_handoff` as a headroom stage. 79.2% of it is infeasible.
- Do not trust a training-time landing rate, and do not pick the best
  checkpoint from a run on the pooled number — select on one block, score on
  fresh ones.
- Do not read an unpaired z on a paired design.
- Do not enable `action_saturation_k`; do not use `--stateless-tgo` (0/200).
- Do not add action repeat to a warm-started raw-action policy. Measured: it
  degrades the clone monotonically (12/24 at k=1 → 5/24 at k=20 → 1/24 at
  k=40). The coarse decision rate has to come with a setpoint-level action
  space, which is what §4 is.

---

## 10. Artefacts and tooling added this session

New envs/trainers:
- `lunarsim/rl/guided_gain_env.py`, `lunarsim/rl/guided_gain_vec_env.py`
- `scripts/train_guided_gain.py`, `scripts/train_guided_gain_isaac.py`
- `scripts/validate_guided_gain.py`, `scripts/validate_guided_gain_isaac.py`

New diagnostics:
- `scripts/diag_exploration_coherence.py` — does the exploration produce any
  behavioural difference at all? Separates "the estimator cannot see the
  signal" from "there was no signal".
- `scripts/diag_gain_space_ceiling.py` — is a flat curve an optimisation
  problem or a reachability one? This is what reframed `land_handoff`.

Checkpoints worth keeping:
- `out/gg_land_v2/best_on_selection_block` — the 31.2% land_handoff policy
- `out/gg_land_v2_validation.json` — its per-episode outcome vectors
- `out/gg_isaac_orbit_v1/` — the orbit_descent run in progress
- everything listed in the previous handover's §6 is still valid
