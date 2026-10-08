"""Can a learned Q POSSIBLY rank this policy's actions? Compare the signal to the noise.

Everything this project tried on the critic failed for what looked like different reasons
-- the action gradient a coin flip, best-of-16 worse than random, CQL/Cal-QL/TD3+BC/PEX
each failing in turn -- and fixing the critic's CALIBRATION did not fix its RANKING:
correlation(Q, TRUE V) went from -0.19 to +0.282 and best-of-16 stayed at -28.7% of
oracle. This script measures the quantity that explains all of it at once.

    signal  how much the true value differs between two actions the policy would take
    noise   the standard deviation of the calibrated critic's own error

MEASURED on orbit_descent (v81@60k, the first calibrated critic here):

    two independent policy samples differ by      0.0428  in action space
    the full action box spans                     3.663 m/s^2
      -> one 0.05 s step turns that into          0.0039 m/s of velocity difference
    dV/dv, regressed from the realised return     0.1936  value units per m/s
      -> signal, one step                         0.00076 value units
    calibrated critic residual std                1.884

        horizon n      signal     ratio
                1     0.000759   1 : 2483
               10     0.007588    1 : 248
              100     0.075880     1 : 25
              714     0.541780      1 : 3    (a whole episode)

So the difference a learned Q must resolve to prefer one action over another is three
orders of magnitude under its own error, and even accumulated over an entire episode it
stays below it. This is not a hyperparameter: it follows from dt = 0.05 s, an 11-tonne
vehicle, and a value function whose total range across the descent is about 4 units.

It also explains why the ANALYTIC controller works where Q-learning does not -- it does
not estimate values, it inverts the dynamics -- and it points at the estimators that do
not need Q(s,a):

  * an on-policy advantage (PPO/GAE) puts the action's effect into the REALISED return
    over the whole trajectory and needs only V(s), never Q(s,a);
  * action repeat multiplies the per-decision leverage by k while leaving the critic's
    error where it is;
  * an analytic/model-based policy gradient uses the reward's own slope, which is dense,
    continuous and -- measured separately -- tighter than the controller at every altitude.

Run:
    .venv/bin/python scripts/diag_action_signal_to_noise.py \
        --checkpoint out/sac_v81_calibrated/snapshots/orbit_descent_60000_steps.zip
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch as th

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--demos", default="out/zemzev_orbit_demos_s2*.npz")
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--gamma-shaping", type=float, default=0.99)
p.add_argument("--gamma-terminal", type=float, default=0.999)
p.add_argument("--samples", type=int, default=6000)
args = p.parse_args()

from lunarsim.rl.curriculum import STAGES_BY_NAME  # noqa: E402
from lunarsim.rl.eval_loader import load_with_critic_for_diag  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

params = STAGES_BY_NAME[args.stage].params
files = sorted(glob.glob(args.demos))
if not files:
    raise SystemExit(f"no demos matched {args.demos!r}")

O, V = [], []
for f in files[:2]:
    d = np.load(f, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    r = d["rewards"].astype(np.float64)
    ends, landed = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], ends[:-1]])
    for s, e, ok in zip(starts, ends, landed):
        if not ok:
            continue
        seg = r[s:e]
        n = len(seg)
        shap = seg.copy()
        shap[-1] = 0.0
        vs = np.zeros(n)
        acc = 0.0
        for i in range(n - 1, -1, -1):
            acc = shap[i] + args.gamma_shaping * acc
            vs[i] = acc
        vt = (args.gamma_terminal ** np.arange(n - 1, -1, -1)) * seg[-1]
        O.append(obs[s:e])
        V.append(vs + vt)
obs = np.concatenate(O)
Vtrue = np.concatenate(V)

# dV/dv, regressed from the realised return. Coarse by design -- a linear fit -- and
# that is enough when the answer is three orders of magnitude.
vel = obs[:, 3:6] * OBS_SCALE[3:6]
alt = obs[:, 2] * float(OBS_SCALE[2])
A = np.column_stack([vel, alt, np.ones(len(vel))])
coef, *_ = np.linalg.lstsq(A, Vtrue, rcond=None)
grad_v = float(np.linalg.norm(coef[:3]))

m, dual = load_with_critic_for_diag(args.checkpoint, "cpu")
rng = np.random.default_rng(0)
idx = rng.choice(len(obs), min(args.samples, len(obs)), replace=False)
o = th.as_tensor(obs[idx])
with th.no_grad():
    a1, _ = m.policy.actor.action_log_prob(o)
    a2, _ = m.policy.actor.action_log_prob(o)
    a_det = m.policy.actor(o, deterministic=True)
    qs = [q.reshape(-1).numpy() for q in m.critic(o, a_det)]
spread = float(th.norm(a1 - a2, dim=1).mean())
if dual:
    h = len(qs) // 2
    Q = np.minimum.reduce(qs[:h]) + np.minimum.reduce(qs[h:])
else:
    Q = np.minimum.reduce(qs)
noise = float((Q - Vtrue[idx]).std())

mass = params.dry_mass_kg + 0.5 * params.initial_fuel_kg
a_span = (params.dps_thrust_max_n - params.dps_thrust_min_n) / mass
dv_full = a_span * params.dt_s
dv_samp = dv_full * spread / 2.0
signal1 = dv_samp * grad_v

print(f"[action-SNR] {args.checkpoint}\n")
print(f"  two independent policy samples differ by   {spread:.4f}  in action space")
print(f"  full action box spans                      {a_span:.3f} m/s^2 "
      f"(mass ~{mass:.0f} kg)")
print(f"    -> one {params.dt_s} s step turns that into    {dv_samp:.6f} m/s")
print(f"  dV/dv from the realised return             {grad_v:.4f} value units per m/s")
print(f"    -> SIGNAL, one step                      {signal1:.6f} value units")
print(f"  critic residual std (the NOISE)            {noise:.3f}")
print(f"  critic residual mean                       {(Q - Vtrue[idx]).mean():+.3f}\n")
print(f"  {'horizon n':>10} {'signal':>12} {'ratio':>12}")
for n in (1, 10, 50, 100, 400, 714):
    sig = signal1 * n
    print(f"  {n:>10} {sig:>12.6f} {'1 : %.0f' % (noise / sig):>12}")
print("\n  A ratio far below 1:1 means no learned Q can prefer one of the policy's own")
print("  actions over another, however well calibrated it is. The estimators that do not")
print("  need Q(s,a) -- an on-policy advantage, action repeat, an analytic policy")
print("  gradient -- are the ones this number points at.")
