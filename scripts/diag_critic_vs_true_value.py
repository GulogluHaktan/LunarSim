"""Compare a critic's Q against the REALISED reward-to-go, per altitude band.

This is the diagnostic that finally said what is wrong with this project's critics, after
a calibrated-value check (Q0 vs the episode's own return) had said "calibrated" and a
gradient check had said "uninformative" without either explaining the other.

It works on DEMONSTRATION data, where the ground truth is available exactly: for a landed
demo episode the terminal reward lives entirely in the last step (the reward function adds
it only when `info["terminated"]`), so the two streams can be separated without needing
the stored info, and the reward-to-go can be accumulated backwards in the critic's OWN
units -- shaping at gamma_shaping, terminal at gamma_terminal.

MEASURED on v77@310k, the best checkpoint of the first run that ever improved on its
warm start (19.8% -> 27.1% deterministic):

    band          TRUE V    critic Q     error
    0-1 m          +1.00      +28.84    +27.84
    5-20 m         +0.54      +32.13    +31.59
    20-80 m        -1.09      +33.65    +34.74
    80-201 m       -2.83      +42.06    +44.88
    correlation(Q, TRUE V) = -0.187

The true value RISES toward the ground (fewer penalty steps left, and the positive
terminal arriving sooner) by +3.83 units across the descent. The critic FALLS by 14.40.
It has learned, in effect, that altitude is valuable -- the hover attractor, written into
the value function -- and it is ANTI-correlated with the truth.

WHERE THE +38 OFFSET COMES FROM, by elimination rather than by guess:
  * the alive bonus is 250 * reward_scale * reward_total_scale = +0.0031/step, worth
    +0.31 in V at gamma=0.99 -- 7x too small, and measured, not assumed.
  * SAC's entropy term goes the WRONG WAY here: log pi is POSITIVE (+18) because the
    BC-fitted policy's std is ~0.04, so -ent_coef*log pi = -0.045/step, i.e. -4.5 in V.
    (The hypothesis was the opposite -- a saturated tanh policy making log pi very
    negative and the entropy bonus acting as a survival bonus. Measurement refused it.)
  * pure TD's fixed point is therefore (r - gamma*ent*log pi)/(1-gamma) = -7.5.
  * the only other force in `critic_loss` is CQL's conservative term, whose alpha is a
    per-sample force on Q(s, a_data) that 1/(1-gamma) = 100 amplifies in the value.

Which makes the finding a SCALE MISMATCH: the true value spans about 4 units on this
reward, and --cql-alpha 5.0 is a larger force than the whole range it is regularising.
CQL's published alphas come with D4RL reward scales where returns are O(100).

Run:
    .venv/bin/python scripts/diag_critic_vs_true_value.py \
        --checkpoint out/sac_v77_rebuilt/snapshots/orbit_descent_310000_steps.zip \
        --demos out/zemzev_orbit_demos_s20000.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--demos", default="out/zemzev_orbit_demos_s20000.npz", action="append")
p.add_argument("--samples", type=int, default=6000)
p.add_argument("--gamma-shaping", type=float, default=0.99)
p.add_argument("--gamma-terminal", type=float, default=0.999)
args = p.parse_args()

from lunarsim.rl.eval_loader import load_with_critic_for_diag  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

ALT_SCALE = float(OBS_SCALE[2])


def load(paths):
    O, A, V = [], [], []
    for f in paths if isinstance(paths, list) else [paths]:
        d = np.load(f, allow_pickle=True)
        # The demos store ALREADY-NORMALISED observations (the env applies normalize_obs
        # before returning them): obs[:,2] is ~0.999 at a 200 m release, not 199.8.
        # Normalising again here fed the networks states 200x too small and invalidated
        # three measurements before this script existed.
        obs = d["obs"].astype(np.float32)
        act = d["actions"].astype(np.float32)
        r = d["rewards"].astype(np.float64)
        ends, landed = d["episode_ends"], d["landed"].astype(bool)
        starts = np.concatenate([[0], ends[:-1]])
        for s, e, ok in zip(starts, ends, landed):
            if not ok:
                continue
            seg = r[s:e]
            n = len(seg)
            shap = seg.copy()
            shap[-1] = 0.0                       # the last step's reward IS the terminal
            vs = np.zeros(n)
            acc = 0.0
            for i in range(n - 1, -1, -1):
                acc = shap[i] + args.gamma_shaping * acc
                vs[i] = acc
            vt = (args.gamma_terminal ** np.arange(n - 1, -1, -1)) * seg[-1]
            O.append(obs[s:e])
            A.append(act[s:e])
            V.append(vs + vt)
    return np.concatenate(O), np.concatenate(A), np.concatenate(V)


def main():
    obs, act, V = load(args.demos)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(obs), min(args.samples, len(obs)), replace=False)
    o = th.as_tensor(obs[idx])
    a = th.as_tensor(act[idx])
    Vt = V[idx]
    alt = obs[idx][:, 2] * ALT_SCALE

    m, dual = load_with_critic_for_diag(args.checkpoint, "cpu")
    with th.no_grad():
        qs = [q.reshape(-1).numpy() for q in m.critic(o, a)]
    if dual:
        h = len(qs) // 2
        Q = np.minimum.reduce(qs[:h]) + np.minimum.reduce(qs[h:])
        detail = (np.minimum.reduce(qs[:h]), np.minimum.reduce(qs[h:]))
    else:
        Q = np.minimum.reduce(qs)
        detail = None

    print(f"[critic-vs-true] {args.checkpoint}")
    print(f"  dual_gamma={dual}  n={len(idx)} expert (s,a) pairs from landed episodes\n")
    print(f"  {'band':<12} {'n':>6} {'TRUE V':>9} {'critic Q':>10} {'error':>9}")
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 20), (20, 80), (80, 1e9)):
        msk = (alt >= lo) & (alt < hi)
        if msk.sum() < 20:
            continue
        label = f"{lo}-{hi:g} m" if hi < 1e9 else f"{lo}+ m"
        print(f"  {label:<12} {int(msk.sum()):>6} {Vt[msk].mean():>9.2f} "
              f"{Q[msk].mean():>10.2f} {(Q[msk] - Vt[msk]).mean():>+9.2f}")
    print(f"  {'overall':<12} {len(idx):>6} {Vt.mean():>9.2f} {Q.mean():>10.2f} "
          f"{(Q - Vt).mean():>+9.2f}")

    low, high = alt < 2, alt >= 80
    print(f"\n  TRUE V across the descent: {Vt[high].mean():+.2f} at 80+ m -> "
          f"{Vt[low].mean():+.2f} at 0-2 m   (rises by {Vt[low].mean() - Vt[high].mean():+.2f})")
    print(f"  critic Q:                  {Q[high].mean():+.2f} -> {Q[low].mean():+.2f}"
          f"   ({Q[low].mean() - Q[high].mean():+.2f})")
    cc = float(np.corrcoef(Q, Vt)[0, 1])
    print(f"\n  correlation(Q, TRUE V) = {cc:+.3f}")
    if cc < 0:
        print("  NEGATIVE: the critic ranks states in the OPPOSITE order to their true"
              "\n  value. An actor ascending this Q is being pushed away from the"
              "\n  behaviour that actually lands.")
    if detail is not None:
        q_s, q_t = detail
        print(f"\n  per stream: shaping {q_s.mean():+8.2f} (gamma={args.gamma_shaping}) "
              f"| terminal {q_t.mean():+8.2f} (gamma={args.gamma_terminal})")
        print(f"  implied constant per-step reward behind each level: "
              f"shaping {q_s.mean() * (1 - args.gamma_shaping):+.4f}, "
              f"terminal {q_t.mean() * (1 - args.gamma_terminal):+.4f}")


main()
