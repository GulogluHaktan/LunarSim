# Paper plan: simulation-fidelity study on LunarSim

Status: plan v2, 2026-11-11. Nothing here has been run yet.

## Fixed decisions (from the authors)

| Item | Decision |
|---|---|
| Scope | Simulation study. **Update 2026-11-11: learned models are IN** (PPO, ES), but they are never compared with each other. Each learner is compared across blend methods only (e.g. smoothstep-PPO vs wendland-PPO; Gaussian-ES vs smoothstep-ES). The deterministic controller stays as the fixed probe for E0-E6 |
| Backend | **Every reported result comes from Isaac Sim / Isaac Lab.** The analytic backend is a last resort, for debugging scripts only (a similar article was rejected for analytic tests). No analytic number enters the paper. Repo docs and `analytic_lander_env.py` now say so |
| Venue | Same as the first version: *Autonomous Intelligent Systems* (Springer, ISSN 2730-616X, no APC) |
| Format | LaTeX, Springer Nature `sn-jnl` class. Reuse the package in `LunarRocket/experiments/terrain_transition/paper/paper_sn/` (`sn-jnl.cls`, `sn-mathphys-num.bst`, build with `tectonic -X compile main_sn.tex`) |
| Sites | Apollo 11 and Apollo 16 only. No procedural tiles in the results |
| Old paper | Not released. It is **not cited**. Its method and lessons are reused, and its blend study is re-done inside this paper under the new simulation |
| Stage | `orbit_descent` only |
| Behavioural probe | Fixed deterministic `ZemZevController` (default gains). MPC only as an optional second probe |

## 1. What to carry over from the old paper

| Old element | Here |
|---|---|
| One variable per experiment, all else fixed | Keep |
| Fair comparison at equal budget (hard truncation at equal radius) | Keep: compare variants at equal radius / resolution / tile size |
| C0/C1 boundary continuity, 720 angular samples, detail-only mode | Keep. Reuse `terrain_transition/metrics.py` and `blend_functions.py`. **Measure on the Isaac collision mesh** (raycast), not on the height array |
| 30 seeds, Wilcoxon on log ratios, bootstrap CI | Keep for geometry (cheap even in Isaac: raycasts, no episodes) |
| Four ablation sweeps, noise floor named in each | Keep the style |
| SAC, n=3 per arm | **Dropped.** Replaced by the deterministic controller, paired by terrain seed |
| Unexplained wall-clock difference | Time each stage separately (mesh build, physics step, controller) |
| Reviewer points: n/power caveat in the abstract, name the pure-noise hypothesis first, give a practical recommendation | Keep as writing rules |

## 2. What is different now

- The new blend (`lunarsim/core/terrain/blend.py`) injects high-pass detail weighted by a Gaussian ROI. It is not a radial fade between two meshes. The old seam defect may not exist in it. **Phase P1 locates real seams before we claim any.** Candidates: the edge of `generate_hires_patch` (micro-detail looks unfaded in the code), the DEM taper at the tile border, ROI weight tails.
- The criterion changed on 2026-10-14 (`plane` default, `spread` legacy). Old numbers do not compare. Every result stores `criterion_key()`.
- Apollo 16 has a hard landing ceiling under `plane` (see [[01-Projects/Isaac-Sim/lunarsim-apollo16-ceiling/lunarsim-apollo16-ceiling]] for the earlier `spread` analysis). Apollo 11 is the easy site. Reporting both gives contrast; effects hidden by the ceiling at A16 are shown with geometry metrics (hazard maps, footpad residuals), which need no landings.

## 3. Research questions

Working title: *How much terrain and physics fidelity does a lunar-landing simulator need? A controlled Isaac Sim study on Apollo 11 and Apollo 16 terrain.*

| RQ | Question | Exp |
|---|---|---|
| RQ1 | Where do multi-resolution seams exist in the Isaac terrain, and how large are they (C0/C1)? Does a compact-support fade remove them at equal cost? | E1 |
| RQ2 | How does terrain resolution change footpad criterion, hazard map and safe-cell fraction (convergence order, noise floor)? | E2 |
| RQ3 | Do sub-grid layers (fine detail, craters, rocks) change the safe-site fraction and the controller landing rate? | E3 |
| RQ4 | How sensitive is the landing rate to the criterion definition (plane vs spread, slope limit)? | E4 |
| RQ5 | How sensitive are outcomes to PhysX settings (collision representation, substep size, solver iterations)? | E5 |
| RQ6 | What does each fidelity level cost (mesh build, VRAM, step time, throughput)? | E6 |

## 4. Experiment matrix (all on Isaac Sim / Lab)

Sites: A11 and A16 in every row. Stage: `orbit_descent` only (decided). "Episode" = one controller landing from that stage.

| ID | Factor varied | Held fixed | Metrics | Design / statistics | Episodes? | New code |
|---|---|---|---|---|---|---|
| E0 | None (A/A): same seeds repeated K times | Everything | Outcome flip rate per seed; touchdown speed/tilt spread | K>=5 repeats of 24 seeds per site. Gives the noise floor for E3-E5 | Yes | Small driver |
| E1 | Fade shape at joins: current, smoothstep, truncated Gaussian (k=1, k=ln 1000), Wendland C2, hybrid | Tile, seed, radius, resolution | C0/C1 (RMS, max) from raycasts on the collision mesh; fade cost | 30+ seeds per variant per site, Wilcoxon on log ratios; radius sweep for scale invariance | No (raycasts) | Selectable fade in `blend.py` / hires patch; Isaac raycast probe; port old `metrics.py` |
| E2 | `res_m` in {0.25, 0.5, 1, 2} m | Site, tile seed, criterion | Footpad slope and roughness error vs finest reference (raycast); hazard-map IoU and kappa; safe-cell fraction | Reference = finest grid; log-log convergence slope with named floor; 50+ tiles per site | No | Reference/agreement script |
| E3 | Terrain `mode` (coarse / fine / blend), crater density, rock density | Site, seed list, controller, criterion | Landing rate; touchdown speed, tilt, footpad residual | Paired by seed; McNemar on `landed_safely`; paired bootstrap CI | Yes | Driver over `TerrainConfig` variants |
| E4 | Criterion: `plane` vs `spread`; max slope in {5, 8, 10, 12, 15} deg | Episodes from E3 baseline arm | Landing rate vs criterion | **Re-score stored touchdown states** (no new episodes). Needs the logger to save the full touchdown state | Re-use | State logger; re-scoring script |
| E5 | PhysX: collision approximation (triangle mesh vs alternatives), `dt` (1/60, 1/120, 1/240), solver iterations | Terrain, seeds, controller | Landing rate, touchdown stats, step time | Paired by seed; the E0 floor is the reference for "no effect" | Yes | Config switches in the Isaac adapter |
| E6 | Quality profile (fast / balanced / high / reference), tile size | Site | Mesh build time, VRAM, step time, fps | Fill blank `step_s/fps/vram_mb` in `benchmarks/benchmarks.csv`; repeat K times | No | Isaac benchmark pass |
| E7 (optional) | LiDAR vs terrain resolution | Site, scan pattern | Point-cloud error vs reference (`compare_point_clouds`) | Only if the rest is done | No | Small |

The old "analytic vs Isaac" comparison is dropped: it needs analytic results, which are out of scope.

## 5. Budget: measure first, then pick arm counts

Isaac episodes are the scarce resource. Earlier numbers (`RESUME.md`): about 20 decisions/s while training with 12 envs; the controller arm finishes 24-48 episodes in a few minutes on 3 envs. We do not have a measured episodes-per-hour figure for the controller probe at 12 envs. **P0 must measure it.** Then:

`total episodes = sites x arms x seeds x repeats`, `hours = total / (episodes per hour)`.

Detectable effect, binary landing rate near 50%, alpha 0.05, power 0.8 (unpaired, a conservative bound):

| Episodes per arm | Minimum detectable difference |
|---|---|
| 96 | about 20 points |
| 192 | about 14 points |
| 384 | about 10 points |

Pairing by seed lowers these. E0 measures the real discordance and the real noise floor. Report the minimum detectable difference next to every episode-based result. Geometry experiments (E1, E2, E6) need no episodes and can use many seeds.

If the budget is tight, cut in this order: E7, E5 solver-iteration axis, E3 rock axis, E6 repeats. Keep E0, E1, E2, E4.

## 5b. Measured timing (P0 probe, 2026-11-11)

Setup: Isaac Sim 6.0.1, RTX 5060 Laptop 8 GB, `validate_guided_gain_isaac.py`, controller arm, `orbit_descent`, 12 envs, held-out pools. Raw files: `experiments/paper/results/p0_timing/`. Git tree was NOT clean (uncommitted edits), so this is a timing probe, not a result.

| Run | Wall time (s) |
|---|---|
| A16, 1 wave | 128 (cold caches) |
| A16, 3 waves | 124 |
| A11, 1 wave | 56 |
| A11, 3 waves | 130 |

- Per wave of 12 episodes: about 37 s (from A11: (130 - 56) / 2). Startup: about 13-19 s warm. The A16 numbers are consistent with this but do not resolve it alone, because its 1-wave run was cold.
- Throughput: about 3.1 s per episode, about 1170 episodes per hour per arm-run, at 12 envs. More envs are untested (8 GB GPU).
- Free A/A signal: wave 0 (seed 90000) was run in two separate processes per site. All 12 outcomes and all 12 returns were bit-identical, on both sites (max difference 0.0). That is 24 episode pairs. It is encouraging for repeatability, but it is not E0: E0 still needs more seeds, more repeats and rock-loaded scenes.

Estimated cost per experiment (episodes at 3.1 s each, both sites). Items marked "not measured" are estimates from code, not timings.

| Exp | Design | GPU time | Basis |
|---|---|---|---|
| E0 | 48 seeds x 10 repeats x 2 sites, fresh process per repeat | about 1.0 h | measured wave time |
| E1 | seam audit + fade comparison, raycasts, no episodes | not measured (needs probe code) | |
| E2 | 4 resolutions, 50 tiles per site, raycasts | not measured (needs probe code) | |
| E3 | 6 arms x 192 episodes x 2 sites | about 2.0 h | measured wave time; rock-loaded arms may be slower, not measured |
| E4 | re-score stored states | about 0 h | needs touchdown-state logger (code work) |
| E5 | 5 arms x 192 episodes x 2 sites | about 1.7-2.2 h | dt = 1/240 s arm may cost up to 2x, not measured |
| E6 | benchmark profiles, K repeats | under 0.5 h | not measured |

Total episode-based GPU time at 192 episodes per arm: about 5 h. At 384 per arm: about 9 h. Code work is the larger unknown: the touchdown logger, the Isaac raycast probe for E1/E2, and the PhysX switches for E5.

## 5c. Time cost of each project method (from existing logs, 2026-11-11)

Source: wall-clock lines in `out/` logs (Isaac "Simulation App Shutting Down" times, SB3 `time_elapsed`, ES/MPC `t=` columns). Same machine, `orbit_descent`. "Isaac" = measured on Isaac. "Analytic" = measured on the analytic backend, which is out of scope for results. "Est." = my arithmetic from measured Isaac rates, not a timing.

| Method | Phase | Time | Backend | Note |
|---|---|---|---|---|
| ZEM/ZEV controller | evaluate 12 episodes | about 37 s | Isaac | P0 probe; about 3.1 s per episode |
| PPO guided-gain | train | about 17-19 decisions/s; 61,440 decisions in 4010 s (67 min, with four 48-episode evals) | Isaac | `out/gg_isaac_orbit_v4_seed21_repro.log`; 150k decisions about 2.3-2.5 h |
| PPO guided-gain | paired validation, 192 episodes | about 15 min (878 s) | Isaac | |
| MPC | evaluate 96 episodes | 924 s (A16), 1269 s (A11) | Isaac | about 70-100 s per 12 episodes, about 2-3x the controller |
| MPC | evaluate 192 episodes | about 15-21 min (924 s and 1233 s in two logs) | Isaac | |
| MPC | tune cost weights, 14 generations | 2858 s (48 min) | **Analytic** | on Isaac: est. about 3.5 h |
| ES (guided-aim policy) | train, 60 generations | 242 s | **Analytic** | on Isaac: est. about 7 h |
| ES (guided-aim policy) | train, 171 generations | 1560 s (26 min) | **Analytic** | on Isaac: est. about 20 h |
| ES policy | evaluate 96 episodes with controller | 714 s | Isaac | |
| PPO from `train_ppo_analytic.py` | train | not timed | **Analytic** | out of scope |

Est. basis: ES-aim does 16 pairs x 2 x 4 = 128 episodes per generation at about 3.3 s per episode (7 min per generation). MPC tuning does 8 pairs x 2 x 8 = 128 episodes per generation at about 7 s per MPC episode (15 min per generation). Isaac cannot use the 8 CPU workers the analytic runs used, so these are single-process estimates.

## 6. Protocol rules

1. **Freeze the code.** The tree has uncommitted edits (`zemzev_controller.py`, `isaac_lander_vec_env.py`, `guided_gain_vec_env.py`, scripts). Commit and tag first. Each result JSON stores git hash, `criterion_key()`, Isaac version, GPU/driver, seed list, stage, controller gains.
2. **Declare seed lists before running.** One file per experiment. Never reuse a list for tuning.
3. **Pair everything** by terrain seed. Report effect size and 95% bootstrap CI, not only p.
4. **Run E0 first.** If the same seed gives different outcomes, that rate is the noise floor and is written in the abstract.
5. **Name the floor in every sweep** (the old paper's negative convergence slope was a truncation floor).
6. **Regenerate every number.** Handover figures include harness artifacts. Use the paired validator pattern, never a trainer's own eval.
7. **Back up `out/`** outside the repo after each phase (wiped once on 2026-10-09).

## 7. Phases

Sizes are relative effort. Every phase needs the GPU except writing.

| Phase | Work | Size | Exit criterion |
|---|---|---|---|
| P0 Setup | Clean commit/tag; check `isaac-env` (`nvidia-smi`, torch CUDA, one 3-env validator run per `RESUME.md`); locate A11/A16 DEM and tile pools (`LUNARSIM_LOLA_IMG`, `LUNARSIM_TILE_POOL`); measure episodes/hour; create `experiments/paper/`; copy `paper_sn` template into `paper/` | S | Episodes/hour number; one controller wave per site prints a rate |
| P1 Seam audit | Raycast the Isaac collision mesh at every candidate seam | S-M | Table of seams with magnitudes, or a documented "no seam" |
| P2 E0, E6 | A/A repeats; cost benchmark | M | Noise floor and cost table |
| P3 E1, E2 | Fade comparison; resolution convergence | M | RQ1, RQ2 figures and tables |
| P4 E3, E4 | Sub-grid layers; criterion re-scoring | L | RQ3, RQ4 tables with CIs |
| P5 E5 | PhysX sensitivity | M-L | RQ5 table |
| P6 Write-up | LaTeX draft, figures from scripts, `/ars-reviewer` panel, revise | M | Reviewer-ready `main_sn.pdf` |

Gate after P2: if E0 shows a large flip rate, drop E5 and raise repeats on E3.

## 8. Repository layout

```
experiments/paper/
  README.md                  # how to reproduce each figure
  seeds/                     # one seed-list file per experiment
  e0_aa_repeat.py ... e6_cost.py
  common/                    # episode logger (full touchdown state), paired stats, metrics
  results/<exp>/<githash>/   # JSON + CSV, never edited by hand
paper/                       # copy of paper_sn: main_sn.tex, body.tex, sections/, tables/, figures/, refs.bib
  build_tables.py            # writes tables/*.tex from results/ (no hand-typed numbers)
```

Reuse from the old project: `metrics.py`, `blend_functions.py`, `ablation_experiments.py`, bootstrap code in `analyze_training_comparison.py`, `LITERATURE_REVIEW.md`, and the vault review [[06-Literature-Reviews/isairas-isparo-2026-lunarsim/isairas-isparo-2026-lunarsim-literatur-taramasi]].

## 9. Paper outline

1. Introduction: simulator fidelity as an unexamined variable; contributions (RQ1-RQ6).
2. Related work: terrain LOD and blending, RBF and compact support (Hardy, Wendland), lunar-landing hazard criteria, simulator sensitivity in robotics. Add a 2023-2026 lunar-landing/TRN source.
3. Simulator: terrain pipeline, Apollo 11/16 data, criterion, Isaac scene, controller probe.
4. Protocol: section 6 of this plan.
5. Results: E1 to E6.
6. Discussion: a table "which fidelity for which question"; the Isaac noise floor; limits (one controller, two sites, no learning).
7. Conclusion.

## 10. Risks

| Risk | Mitigation |
|---|---|
| P1 finds no real seam in the new blend | Report the null honestly; RQ2-RQ6 still carry the paper |
| Isaac episodes too slow for the plan | P0 measurement, then cut order in section 5 |
| Apollo 16 ceiling hides effects | Use geometry metrics and the E4 re-scoring there; A11 carries the landing-rate claims |
| Isaac nondeterminism | E0 first; state the floor |
| One controller only | Optional MPC probe on a small n for E3/E4 |
| GPU or driver problem | `RESUME.md` has check commands |

## 11. Open questions (small)

1. Journal length limit and reference style: check the AIS author guidelines at P6 start.
2. Is MPC allowed as a second probe, given run cost?

## 12. Learner extension (decided 2026-11-11)

Question (RQ7): does the blend method change what PPO and ES learn, measured as paired landing rate of the trained policy on held-out episodes? The comparison is always same learner, different blend method. PPO is never compared with ES.

Rules carried over from the first version's lesson (n=1 looked significant, n=3 did not):
- Several training seeds per (learner, blend method, site). Seeds are declared before training.
- Evaluate every trained policy with the paired validator (`validate_guided_gain_isaac.py`, 192 episodes, fresh seeds, never the trainer's own eval).
- Analyse per-seed final rates as the unit (not per-episode pooling). Report the minimum detectable difference. A null result is reported as "no difference above X".
- Add a no-blend `fine` reference arm per learner so effects are read against a baseline.

Measured cost (Isaac, orbit_descent, this machine class):

| Item | Time |
|---|---|
| PPO training, 60k decisions | about 1.1 h (laptop 15.9 decisions/s; desktop 20.2/s) |
| PPO training, 150k decisions | about 2.1-2.6 h |
| ES training, 60 generations (128 episodes each) | about 7.0 h (396 s per generation + 72 s selection every 3 gens; measured) |
| ES training, 30 generations | about 3.5 h |
| Paired validation of one policy, 192 episodes | about 15 min |

Budget = runs x time. Two machines run in parallel, so wall time is about half.

| Design | Runs | GPU hours (1 machine) | Wall hours (2 machines) |
|---|---|---|---|
| PPO 60k, 3 methods x 2 sites x 3 seeds | 18 | about 20 + 5 validation | about 12 |
| PPO 60k, 6 methods x 2 sites x 3 seeds | 36 | about 40 + 9 | about 25 |
| PPO 150k, 3 methods x 2 sites x 3 seeds | 18 | about 45 + 5 | about 25 |
| ES 60 gens, 3 methods x 1 site x 3 seeds | 9 | about 63 + 2 | about 33 |
| ES 30 gens, 3 methods x 1 site x 3 seeds | 9 | about 32 + 2 | about 17 |
| ES 60 gens, 3 methods x 2 sites x 3 seeds | 18 | about 126 + 5 | about 65 |

Recommended starting point (to confirm): PPO 60k with smoothstep, wendland, gaussian_calibrated (+ `fine` reference) on both sites, 3 seeds; ES 30 gens on one site, same methods, 3 seeds. About 30 wall hours on two machines.

Paper impact: the introduction currently says the probe is not learned. Revise it to "a fixed controller probe plus learned policies compared across blend methods". Add RQ7 and section "Learned policies" in results.

## 13. Final experiment design under the 12-hour-per-machine cap (2026-11-11)

Constraints from the authors: PPO runs on the laptop, ES runs on the HP desktop (`noron`), at most 12 hours each, ES at most 30 generations. Every experiment must produce a table and a figure, all in English. The terrain generator and each learning algorithm have a flow chart in the paper (`paper/figures/flow_*.tex`).

Blend methods compared (each learner against itself only): smoothstep, Wendland C2, calibrated Gaussian (k = ln 1000). PPO also has a no-blend `fine` reference arm.

| Learner | Machine | Design | Per run | Total |
|---|---|---|---|---|
| PPO | laptop | 4 arms (3 methods + fine) x 2 sites x 3 seeds = 24 runs would need 13.4 h, so the `fine` reference uses 1 seed per site: 3 x 2 x 3 + 2 = 20 runs | about 34 min: 15,360 decisions (20 updates), 3 selection blocks of 24 episodes, policy-only validation of 192 held-out episodes | about 11.3 h |
| ES | HP desktop | 3 methods x Apollo 11 x 3 seeds = 9 runs, 18 generations x (8 pairs x 4 episodes x 2) = 64 episodes per generation | about 74 min: 60 min training, 2 selection blocks, validation 192 episodes | about 11.1 h |

Reasons for the sizes, from measurements: PPO reached 77% on a 48-episode block at 15k decisions in the earlier Isaac run, so 15k decisions shows learning. ES generation time is 396 s for 128 episodes (3.1 s per episode); halving the population keeps 18 generations inside the cap. ES on Apollo 16 is not in this budget; it is the first thing to add if time remains.

Validation: every trained policy is scored on the same held-out pool of the site (the canonical `pool_val.pkl`), with declared wave seeds 90000.. and 16 waves x 12 envs. The controller arm is measured once per site and is not repeated per policy (`--policy-only`). The per-seed final landing rate is the analysis unit.

Honest limits to state in the paper: 3 seeds give low power; ES at 18 generations is a short run and may not improve on the controller; learners are not compared with each other. Each limit is reported with the minimum detectable difference.
