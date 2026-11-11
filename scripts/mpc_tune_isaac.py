"""Tune the MPC cost weights with OpenAI-ES ON ISAAC (port of mpc_tune.py).

Same compute shape as the analytic tuner: `pairs` antithetic pairs -> 2*pairs candidates per generation, every
candidate flies one wave of `--n-envs` (= the analytic `--n`, default 8) episodes from the SAME wave seed, so
common random numbers hold exactly (same env index -> same spawn). Fitness = landing rate of the wave.
The MPC setup (MPPIMPC, planner config, hazard maps, replanning) is copied from scripts/mpc_isaac_eval.py.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--gens", type=int, default=1)
p.add_argument("--pairs", type=int, default=8)
p.add_argument("--n-envs", type=int, default=8)
p.add_argument("--sigma", type=float, default=0.25)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--init", default=None)
p.add_argument("--out", required=True)
a = p.parse_args()

app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.core.terrain.hazard import attach_hazard  # noqa: E402
from lunarsim.evo.es import OpenAIES  # noqa: E402
from lunarsim.evo.mpc import DEFAULT_LOG, PARAM_NAMES, MPCConfig, MPPIMPC  # noqa: E402
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.reward import RewardWeights, make_apollo_reward_fn  # noqa: E402


def main():
    st = STAGES_BY_NAME[a.stage]; n = a.n_envs
    venv = IsaacLanderVecEnv(
        num_envs=n, tile_fn=make_tile_fn(st), params=st.params,
        env_spacing_m=max(s.tile_size_m for s in STAGES) * 1.5, seed=7, max_rocks_per_env=0,
        reward_fn=make_apollo_reward_fn(RewardWeights()))
    cfg = MPCConfig(K=512, M=20, hold=12, exec_steps=4, roll_sub=2, margin=0.7)
    theta0 = np.load(a.init) if a.init else DEFAULT_LOG.copy()
    mpc = MPPIMPC(st.params, RewardWeights(), cfg, theta0)
    max_raw = int(st.params.max_episode_s / st.params.dt_s) + 4 \
        + int(float(os.environ.get("LUNARSIM_SETTLE_S", "4.0")) / st.params.dt_s) + 8

    def state():
        s = venv.state
        S = {k: np.asarray(s[k], dtype=np.float64).copy() for k in
             ("vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz", "fuel_kg")}
        S["alt"] = np.array([venv._contact(i)[0] for i in range(n)])
        S["px"] = np.array([float(s["x"][i]) - float(venv.env_origins[i][0]) for i in range(n)])
        S["py"] = np.array([float(s["y"][i]) - float(venv.env_origins[i][1]) for i in range(n)])
        S["t_left"] = venv.params.max_episode_s - np.asarray(venv._t, dtype=np.float64)
        return S

    def fly(lp, wave_seed):
        """One wave of n episodes with MPC weights `lp` (log10 params). Returns landed fraction."""
        mpc.W = dict(zip(PARAM_NAMES, (10.0 ** np.asarray(lp)).tolist()))
        venv._rng = np.random.default_rng(wave_seed)
        venv.reset(); mpc.U = None; mpc._since_shift = 0; mpc.reset(n)
        done = np.zeros(n, bool); landed = np.zeros(n, bool); tile_ids = []
        A = np.zeros((n, 4), np.float32)
        for step in range(max_raw):
            if step % cfg.exec_steps == 0:
                ids = [id(t) for t in venv._tiles]
                if ids != tile_ids:
                    tl = [attach_hazard(t) for t in venv._tiles]
                    mpc.set_hazard([t.haz_dist for t in tl], tl[0].size_m, tl[0].res_m, [t.height for t in tl]); tile_ids = ids
                A = mpc.act(state()).astype(np.float32)
            venv.step_async(A); _, _, dones, infos = venv.step_wait()
            for i in range(n):
                if not done[i] and dones[i]:
                    done[i] = True; landed[i] = bool(infos[i].get("landed_safely", False))
            if done.all():
                break
        return float(landed.mean())

    es = OpenAIES(theta0, a.sigma, a.lr, a.pairs, seed=0)
    os.makedirs(a.out, exist_ok=True)
    rows = []; t_all = time.time()
    print(f"MPC tuning on ISAAC {a.stage}: {a.gens} gens, {2 * a.pairs} candidates x {n} episodes", flush=True)
    for g in range(a.gens):
        t0 = time.time()
        pop, eps = es.ask()
        fit = np.array([fly(th, 200000 + 1000 * g) for th in pop])
        es.tell(eps, fit)
        rows.append(dict(gen=g, gen_seconds=time.time() - t0, mean_land=float(fit.mean()), best_land=float(fit.max())))
        print(f"g{g:3d} gen {rows[-1]['gen_seconds']:7.1f}s  mean land {fit.mean():.2f}  best {fit.max():.2f}", flush=True)
    json.dump(dict(stage=a.stage, n_envs=n, pairs=a.pairs, rows=rows, total_seconds=time.time() - t_all),
              open(os.path.join(a.out, "timing.json"), "w"))


main()
app.close()
