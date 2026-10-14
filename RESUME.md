# Resume point — 2026-10-09, stopped for an NVIDIA driver update

GPU is released (41 MiB). Nothing is running. `handover.md` has the full
picture; this file is only what to do next and what was in flight.

## What is safe on disk

| artefact | what it is |
|---|---|
| `out/gg_land_v2/best_on_selection_block.zip` | the 31.2% land_handoff policy |
| `out/gg_land_v2_validation.json` | its 192 paired per-episode outcomes |
| `out/gg_land_v2b/best_on_selection_block.zip` | the replicate, 21.9% |
| `out/gg_land_v2b_validation.json` | the replicate's 192 paired outcomes |
| `out/gg_isaac_orbit_v1/snap_15000.zip` | the Isaac run's only snapshot |
| `out/ctrl_orbit_fixed.log` | controller 24/48 on Isaac orbit_descent, bug fixed |

Both analytic results are complete and validated. The Isaac run was killed at
~27k of 150k decisions; its last snapshot is 15k, so ~12k decisions (~15 min of
wall clock) are lost. Nothing else.

## After the reboot: check Isaac still works before trusting a run

A driver bump can leave the Python bindings mismatched. One command:

```bash
nvidia-smi
/home/haktan/isaac-env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
/home/haktan/isaac-env/bin/python scripts/validate_guided_gain_isaac.py \
    --stage orbit_descent --gain-set attitude --n-envs 3 --waves 1
```

The last one should print a controller arm near 50% in a couple of minutes. If
`torch.cuda.is_available()` is False, the venv's torch was built against the
old CUDA — reinstall torch in `/home/haktan/isaac-env`, not in `.venv` (the
`.venv` is CPU-only by design and is what the analytic scripts use).

## Then relaunch the Isaac arm

This is the stage the goal is stated on. Warm-starting from the 15k snapshot
saves the lost time and is legitimate — that snapshot was selected on block
7000, which is the declared selection block:

```bash
nohup /home/haktan/isaac-env/bin/python scripts/train_guided_gain_isaac.py \
  --stage orbit_descent --gain-set attitude --hold-steps 20 \
  --gamma 0.995 --gae-lambda 0.99 --n-steps 64 --batch-size 192 \
  --steps 150000 --n-envs 12 --seed 3 \
  --eval-episodes 48 --eval-every 15000 \
  --out out/gg_isaac_orbit_v2 > out/gg_isaac_orbit_v2.log 2>&1 &
```

It runs ~20 decisions/s with nothing else on the box, so 150k is about 2 h.
Score the result with the PAIRED validator, never with the trainer's own eval:

```bash
/home/haktan/isaac-env/bin/python scripts/validate_guided_gain_isaac.py \
  out/gg_isaac_orbit_v2/best_on_selection_block \
  --stage orbit_descent --gain-set attitude --waves 16 --n-envs 12 \
  --wave-seed0 90000 --out-json out/gg_isaac_orbit_v2_validation.json
```

## The change worth making FIRST, on measured grounds

See `handover.md` §8 lead 2 — but the arithmetic got sharper after the
replicate validated, so here it is:

The metric is binary (`landed_safely`) and the terminal reward is **continuous
through zero at exactly that boundary**:

```
terminal = max(touchdown_k*(1 - severity), -cap) * reward_total_scale
         = max(120*(1 - severity), -480) * 0.05

severity 0.99 -> +0.060      severity 1.00 -> 0.000      severity 1.01 -> -0.060
```

Crossing the success boundary by ±1% of a limit is worth **0.120 reward
units**. Against that, measured on the 192 paired validation episodes:

| | v2 | v2b |
|---|---|---|
| landed | 60/192 (31.2%) | 42/192 (21.9%) |
| mean return vs controller | **-3.02** | **+1.59** |
| return gain on episodes converted to a landing | +2.52 | +2.60 |
| return change on episodes that still failed | **-5.21** | **+1.62** |

So the run that landed MORE scored a WORSE return, and the run that scored a
better return landed less. PPO maximises return, so its objective actively
preferred the weaker-landing policy. The reason is in the last two rows: the
gradient is dominated by shaping movement on the ~85% of episodes that still
fail, and the binary event being measured contributes ±0.06 at the margin.

Note what this is NOT: the reward is not mis-ranked. AUC(return -> landed) is
**1.000** for the controller arm and 0.973-0.987 for the policies, so given a
touchdown the reward orders outcomes correctly. The defect is in the gradient,
not the ranking.

**The fix the literature already prescribes.** Gaudet/Linares/Furfaro
(arXiv:1810.08719) pay `kappa = 10` as a **step**, only when position,
velocity, attitude AND rate are all inside their limits. This repo's own
analysis of that paper says so and calls it corroboration —
`rl-cokus-literatur-taramasi.md`, section 7: "Their terminal bonus IS a step
function ... a hard step, exactly the shape this project replaced." The
implication was never drawn: replacing the step with a continuous ramp removed
the reward's signal for the one event the project measures.

So add a step at the boundary ON TOP of the continuous grading, rather than
instead of it — the continuity was added to give gradient methods something to
climb, and that reason still holds. Something like

```python
if landed_safely:
    r += w.landed_step_k * S      # a step, ~1-3 units, on top of 120*(1-sev)*S
```

sized so one landing outweighs the shaping range of a failing episode (the
measured range is ~5 units, so the step needs to be at least that order).
It is a new weight with a default of 0.0, so every existing checkpoint and
every number above stays reproducible, and `diag_reward_separation.py` can
screen it before a run is spent — that screen costs a minute.

**Re-run both analytic arms with the step before spending Isaac time on it**,
because land_handoff is 20 minutes and the comparison is already set up:

```bash
.venv/bin/python scripts/train_guided_gain.py --stage land_handoff \
  --gain-set attitude --hold-steps 10 --steps 400000 --n-envs 10 --seed 11 \
  --out out/gg_land_v3     # plus whatever flag exposes the new weight
.venv/bin/python scripts/validate_guided_gain.py out/gg_land_v3/best_on_selection_block \
  --stage land_handoff --gain-set attitude --hold-steps 10 --blocks 4 \
  --out-json out/gg_land_v3_validation.json
```

v2 and v2b give a two-run baseline to beat: +21.4 and +12.0 points, both with
zero regressions.

## One correction to make to my own earlier claim

`scripts/diag_gain_space_ceiling.py` reported "79.2% of land_handoff episodes:
no sample ever landed" and I read that as attitude-infeasibility. That reading
is **not safe**: the search was 24 uniform samples in an **11-dimensional**
box, which is essentially no coverage. The 20.8% figure is a valid LOWER bound
on the gain space's reach and not an upper bound on anything. Before quoting a
ceiling again, either raise the sample count by orders of magnitude or replace
the uniform draw with a real optimiser (CMA-ES) on a handful of episodes. The
physical argument for infeasibility at the worst corner (8 m and -5 m/s gives
1.6 s of flight against an ~8.7 s RCS slew to 20 deg) still stands on its own
and is worth checking directly with `scripts/test_landing_feasibility.py`.
