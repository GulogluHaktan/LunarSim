"""Controller-free ES landing on the ANALYTIC backend (quick prototype before Isaac).

Fitness F_graded with touchdown_penalty_cap=480 and timeout_penalty=480 (EVO_PLAN's
cap=60 is flat for every crash with severity>=1.5, and hovering would beat a crash).
CRN: every candidate in a generation flies the same episode seeds.
"""
import argparse, csv, json, multiprocessing as mp, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.evo.es import OpenAIES
from lunarsim.evo.policies import MLPPolicy

p = argparse.ArgumentParser()
p.add_argument("--stage", default="ramp_35m")
p.add_argument("--hidden", default="32,32")
p.add_argument("--pairs", type=int, default=16)
p.add_argument("--eps-per-cand", type=int, default=4)
p.add_argument("--sigma", type=float, default=0.1)
p.add_argument("--lr", type=float, default=0.03)
p.add_argument("--gens", type=int, default=100)
p.add_argument("--bias", type=float, default=-0.05)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init-npz", default=None)
p.add_argument("--stop-land", type=float, default=0.0)
p.add_argument("--obs-norm", type=int, default=1)
p.add_argument("--init-scale", type=float, default=0.1)
p.add_argument("--hold", type=int, default=1, help="action repeat (control steps)")
p.add_argument("--out", default="out/evo/analytic")
a = p.parse_args()

OVR = ["reward_scale=0", "touchdown_k=120", "touchdown_penalty_cap=480",
       "landed_step_k=100", "timeout_penalty=480"]
_G = {}

def _env():
    if "env" not in _G:
        from lunarsim.rl.curriculum import STAGES_BY_NAME, make_tile_fn
        from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv
        from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides
        st = STAGES_BY_NAME[a.stage]
        w = weights_from_overrides(OVR)
        assert w.touchdown_penalty_cap == 480 and w.timeout_penalty == 480 and w.reward_scale == 0
        _G["env"] = AnalyticLanderEnv(tile_fn=make_tile_fn(st, 80), params=st.params,
                                      reward_fn=make_apollo_reward_fn(w), seed=0)
        _G["scale"] = w.reward_total_scale
    return _G["env"], _G["scale"]

def rollout(args):
    theta, seeds, hidden, mu, sd = args
    env, scale = _env()
    pol = MLPPolicy(20, 4, hidden, a.bias)
    tot, landed = 0.0, 0
    osum = np.zeros(20); osq = np.zeros(20); on = 0
    for s in seeds:
        obs, _ = env.reset(seed=int(s)); R = 0.0
        while True:
            o64 = np.asarray(obs, dtype=np.float64)
            osum += o64; osq += o64 * o64; on += 1
            act = pol.act(theta, (o64 - mu) / sd)
            done = False
            for _ in range(a.hold):
                obs, r, term, trunc, info = env.step(act); R += r
                if term or trunc: done = True; break
            if done: break
        tot += R / scale; landed += bool(info["landed_safely"])
    return tot / len(seeds), landed / len(seeds), (osum, osq, on)

if __name__ == "__main__":
    hidden = tuple(int(x) for x in a.hidden.split(",")) if a.hidden else ()
    pol = MLPPolicy(20, 4, hidden, a.bias)
    theta0 = pol.init_params(np.random.default_rng(a.seed), a.init_scale)
    mu0 = np.zeros(20); sd0 = np.ones(20); N0 = 0
    if a.init_npz:
        z = np.load(a.init_npz); theta0 = z["theta"].copy(); mu0, sd0, N0 = z["mu"], z["sd"], 20000
        assert len(theta0) == pol.n_params, "init npz shape mismatch"
    es = OpenAIES(theta0, a.sigma, a.lr, a.pairs, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    pool = mp.get_context("fork").Pool(min(16, mp.cpu_count()))
    f = open(os.path.join(a.out, "curve.csv"), "w", newline=""); wr = csv.writer(f)
    wr.writerow(["gen", "episodes", "mean_fit", "mean_land", "best_land", "sel_land", "sel_fit"])
    mu, sd = mu0, sd0; S = [mu0 * N0, (sd0 ** 2 + mu0 ** 2) * N0, N0]
    streak = 0
    episodes = 0; t0 = time.time(); best = (-1e9, None)
    for g in range(a.gens):
        seeds = 200000 + 1000 * g + np.arange(a.eps_per_cand)
        pop, eps = es.ask()
        res = pool.map(rollout, [(th, seeds, hidden, mu, sd) for th in pop])
        fit = np.array([r[0] for r in res]); land = np.array([r[1] for r in res])
        if a.obs_norm:
            for r in res:
                S[0] += r[2][0]; S[1] += r[2][1]; S[2] += r[2][2]
            mu = S[0] / S[2]; sd = np.sqrt(np.maximum(S[1] / S[2] - mu * mu, 0.0)) + 1e-2
        es.tell(eps, fit); episodes += len(pop) * a.eps_per_cand
        sel_fit, sel_land = (np.nan, np.nan)
        if g % 5 == 0 or g == a.gens - 1:
            sel = 150000 + np.arange(24)      # selection block, mean theta
            chunks = np.array_split(sel, 8)
            rr = pool.map(rollout, [(es.theta, c, hidden, mu, sd) for c in chunks])
            sel_fit = float(np.mean([r[0] for r in rr])); sel_land = float(np.mean([r[1] for r in rr]))
            if sel_fit > best[0]:
                best = (sel_fit, es.theta.copy()); pol.save(os.path.join(a.out, "best_on_selection.npz"), best[1], mu=mu, sd=sd)
        wr.writerow([g, episodes, fit.mean(), land.mean(), land.max(), sel_land, sel_fit]); f.flush()
        if not np.isnan(sel_land):
            streak = streak + 1 if (a.stop_land and sel_land >= a.stop_land) else 0
        if streak >= 2:
            print(f"g{g} stop: selection land {sel_land:.2f} twice", flush=True); break
        if g % 5 == 0:
            print(f"g{g:3d} ep={episodes} fit={fit.mean():8.1f} cand_land={land.mean():.2f} "
                  f"best_cand={land.max():.2f} SEL land={sel_land:.2f} fit={sel_fit:.1f} t={time.time()-t0:.0f}s", flush=True)
