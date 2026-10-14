"""ES in the guided-aim action space (analytic backend): 11 ZEM/ZEV gains + 2 lateral-aim components.

The policy sees the 20 env obs + 3 guidance extras + 7 hazard-radar + 2 direction-to-safe-cell features (32-d).
Zero action = the stock controller. Fitness = env reward - LAMBDA*min(dist-to-safe at touchdown,150)/150.
CRN: every candidate flies the same seeds within a generation.
"""
import argparse, csv, multiprocessing as mp, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.evo.es import OpenAIES
from lunarsim.evo.hazfeat import _bilerp
from lunarsim.core.terrain.hazard import attach_hazard

p = argparse.ArgumentParser()
p.add_argument("--stage", default="orbit_descent"); p.add_argument("--hidden", default="")
p.add_argument("--pairs", type=int, default=16); p.add_argument("--eps-per-cand", type=int, default=4)
p.add_argument("--sigma", type=float, default=0.15); p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--gens", type=int, default=60); p.add_argument("--aim-scale", type=float, default=15.0)
p.add_argument("--lam", type=float, default=60.0); p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=8); p.add_argument("--sel", type=int, default=24)
p.add_argument("--bias-only", type=int, default=0, help="1: constant action (13 numbers) -- sample-efficient"); p.add_argument("--out", required=True)
a = p.parse_args()
OVR = ["reward_scale=0", "touchdown_k=120", "touchdown_penalty_cap=480", "landed_step_k=100", "timeout_penalty=480"]
_G = {}
GAINS = "attitude"

def _env():
    if "env" not in _G:
        from lunarsim.rl.curriculum import STAGES_BY_NAME, make_tile_fn
        from lunarsim.rl.guided_gain_env import GuidedGainEnv, GAIN_SETS
        from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides
        st = STAGES_BY_NAME[a.stage]; w = weights_from_overrides(OVR)
        _G["env"] = GuidedGainEnv(tile_fn=make_tile_fn(st, 80), params=st.params, reward_fn=make_apollo_reward_fn(w),
                                  gain_set=None, gain_spec=GAIN_SETS[GAINS], hold_steps=20,
                                  aim_scale=a.aim_scale, hazard_obs=True) if False else GuidedGainEnv(
            tile_fn=make_tile_fn(st, 80), params=st.params, reward_fn=make_apollo_reward_fn(w),
            gain_spec=GAIN_SETS[GAINS], hold_steps=20, aim_scale=a.aim_scale, hazard_obs=True)
        _G["scale"] = w.reward_total_scale
    return _G["env"], _G["scale"]

def forward(theta, x, sizes):
    if sizes[0] == 0: return np.tanh(theta[:sizes[1]])      # constant-action policy
    k = 0
    for li in range(len(sizes) - 1):
        ai, bi = sizes[li], sizes[li + 1]
        x = x @ theta[k:k + ai * bi].reshape(ai, bi) + theta[k + ai * bi:k + ai * bi + bi]; k += ai * bi + bi
        if li < len(sizes) - 2: x = np.tanh(x)
    return np.tanh(x)

def rollout(args):
    theta, seeds, sizes, mu, sd = args
    env, scale = _env()
    N0 = max(sizes[0], 1); tot, landed, osum, osq, on = 0.0, 0, np.zeros(N0), np.zeros(N0), 0
    for s in seeds:
        obs, _ = env.reset(seed=int(s)); R = 0.0
        while True:
            o = np.asarray(obs, np.float64); on += 1
            if sizes[0]: osum += o; osq += o * o
            obs, r, te, tr, info = env.step(forward(theta, (o - mu) / sd, sizes).astype(np.float32)); R += r
            if te or tr: break
        tl = attach_hazard(env.env.tile); st_ = env.env.state
        td = float(_bilerp(tl.haz_dist, tl.res_m, np.array([st_["x"]]), np.array([st_["y"]]))[0])
        tot += R / scale - a.lam * min(td, 150.0) / 150.0; landed += bool(info["landed_safely"])
    return tot / len(seeds), landed / len(seeds), (osum, osq, on)

if __name__ == "__main__":
    env, _ = _env()
    n_obs = int(env.observation_space.shape[0]); n_act = int(env.action_space.shape[0])
    hidden = [int(x) for x in a.hidden.split(",")] if a.hidden else []
    sizes = [n_obs] + hidden + [n_act]
    if a.bias_only: sizes = [0, n_act]
    n_par = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
    theta0 = np.random.default_rng(a.seed).normal(0, 0.02, n_par) if not a.bias_only else np.zeros(n_act)
    es = OpenAIES(theta0, a.sigma, a.lr, a.pairs, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    pool = mp.get_context("fork").Pool(a.workers)
    f = open(os.path.join(a.out, "curve.csv"), "w", newline=""); wr = csv.writer(f)
    wr.writerow(["gen", "mean_fit", "mean_land", "best_land", "sel_land", "sel_fit"])
    if a.bias_only: n_obs = 1
    mu, sd = np.zeros(n_obs), np.ones(n_obs); S = [np.zeros(n_obs), np.zeros(n_obs), 0]
    best = (-1e9, None); t0 = time.time()
    for g in range(a.gens):
        seeds = 200000 + 1000 * g + np.arange(a.eps_per_cand)
        pop, eps = es.ask()
        res = pool.map(rollout, [(th, seeds, sizes, mu, sd) for th in pop])
        fit = np.array([r[0] for r in res]); land = np.array([r[1] for r in res])
        for r in res: S[0] += r[2][0]; S[1] += r[2][1]; S[2] += r[2][2]
        mu = S[0] / S[2]; sd = np.sqrt(np.maximum(S[1] / S[2] - mu * mu, 0.0)) + 1e-2
        es.tell(eps, fit)
        sel_fit = sel_land = float("nan")
        if g % 3 == 0 or g == a.gens - 1:
            chunks = np.array_split(150000 + np.arange(a.sel), a.workers)
            rr = pool.map(rollout, [(es.theta, c, sizes, mu, sd) for c in chunks])
            sel_fit = float(np.mean([r[0] for r in rr])); sel_land = float(np.mean([r[1] for r in rr]))
            if sel_fit > best[0]:
                best = (sel_fit, es.theta.copy())
                np.savez(os.path.join(a.out, "best_on_selection.npz"), theta=best[1], sizes=np.array(sizes), mu=mu, sd=sd,
                         aim_scale=a.aim_scale, gain_set=GAINS)
            print(f"g{g:3d} cand fit={fit.mean():7.1f} land={land.mean():.2f} best={land.max():.2f} | SEL land={sel_land:.2f} fit={sel_fit:7.1f} t={time.time()-t0:.0f}s", flush=True)
        wr.writerow([g, fit.mean(), land.mean(), land.max(), sel_land, sel_fit]); f.flush()
