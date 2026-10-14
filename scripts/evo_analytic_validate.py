"""Paired validation of an ES policy vs the ZEM/ZEV controller, analytic backend.
Same episode seeds for both arms (env.reset(seed=s) fixes spawn + terrain)."""
import argparse, json, math, multiprocessing as mp, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.evo.policies import MLPPolicy

p = argparse.ArgumentParser()
p.add_argument("npz"); p.add_argument("--stage", default="ramp_35m")
p.add_argument("--seed0", type=int, default=90000); p.add_argument("--n", type=int, default=192)
p.add_argument("--hold", type=int, default=1); p.add_argument("--out-json", default=None)
a = p.parse_args()
_G = {}

def _env():
    if "e" not in _G:
        from lunarsim.rl.curriculum import STAGES_BY_NAME, make_tile_fn
        from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv
        from lunarsim.rl.reward import make_apollo_reward_fn
        st = STAGES_BY_NAME[a.stage]
        _G["e"] = AnalyticLanderEnv(tile_fn=make_tile_fn(st, 80), params=st.params,
                                    reward_fn=make_apollo_reward_fn(), seed=0)
        _G["st"] = st
    return _G["e"], _G["st"]

def fly(args):
    seed, arm = args
    env, st = _env()
    obs, _ = env.reset(seed=seed)
    if arm == "ctrl":
        from lunarsim.control.zemzev_controller import ZemZevController
        c = ZemZevController(st.params); c.reset()
    else:
        z = np.load(a.npz)
        pol = MLPPolicy(20, 4, tuple(int(h) for h in z["sizes"][1:-1]), float(z["throttle_bias"]))
        theta, mu, sd = z["theta"], z["mu"], z["sd"]
    while True:
        act = c.act(env) if arm == "ctrl" else pol.act(theta, (np.asarray(obs, float) - mu) / sd)
        for _ in range(1 if arm == "ctrl" else a.hold):
            obs, r, te, tr, info = env.step(act)
            if te or tr: break
        if te or tr: break
    S = env.state
    why = "landed" if info["landed_safely"] else "tumble" if info["lost_control"] else "crash" if te else "timeout"
    return dict(landed=bool(info["landed_safely"]), why=why, vz=float(S["vz"]),
                vxy=float(np.hypot(S["vx"], S["vy"])), tilt=float(np.hypot(S["tilt_x"], S["tilt_y"])),
                w=float(np.linalg.norm([S["wx"], S["wy"], S["wz"]])), t=float(info["t_s"]))

def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def mcnemar(x, y):  # exact two-sided
    b = int(np.sum(y & ~x)); c = int(np.sum(x & ~y)); n = b + c
    if n == 0: return b, c, 1.0
    k = min(b, c); pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, pv

if __name__ == "__main__":
    seeds = [a.seed0 + i for i in range(a.n)]
    with mp.get_context("fork").Pool(min(16, mp.cpu_count())) as pool:
        C = pool.map(fly, [(s, "ctrl") for s in seeds]); P = pool.map(fly, [(s, "pol") for s in seeds])
    out = {}
    for name, R in (("controller", C), ("es_policy", P)):
        k = sum(r["landed"] for r in R); lo, hi = wilson(k, a.n)
        why = {w: sum(r["why"] == w for r in R) for w in ("landed", "crash", "tumble", "timeout")}
        L = [r for r in R if r["landed"]]
        print(f"{name:11s} {k}/{a.n} = {100*k/a.n:.1f}%  95%CI [{100*lo:.1f},{100*hi:.1f}]  {why}"
              + (f"  landed: vz={np.mean([abs(r['vz']) for r in L]):.2f} vxy={np.mean([r['vxy'] for r in L]):.2f} "
                 f"tilt={np.degrees(np.mean([r['tilt'] for r in L])):.1f}deg t={np.mean([r['t'] for r in L]):.1f}s" if L else ""))
        out[name] = dict(landed=[r["landed"] for r in R], why=why, k=k)
    b, c, pv = mcnemar(np.array(out["controller"]["landed"]), np.array(out["es_policy"]["landed"]))
    print(f"McNemar: policy-only {b}, controller-only {c}, p={pv:.2e}")
    if a.out_json:
        json.dump(dict(stage=a.stage, seeds=seeds, npz=a.npz, **out, mcnemar=[b, c, pv]), open(a.out_json, "w"))
