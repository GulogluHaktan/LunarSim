"""OpenAI-ES in the guided-aim action space, trained ON ISAAC (port of evo_guided_aim_train.py, bias-only variant).

The analytic script runs 8 CPU workers; Isaac runs one process with `--n-envs` parallel envs, and every env in a
wave flies its own (candidate, seed) job, so one wave = `n_envs` episodes. Same compute shape as the analytic
script: `pairs` antithetic pairs x `eps-per-cand` episodes per candidate, a `--sel`-episode selection block every
3 generations. Policy = constant 13-number action tanh(theta) (the `--bias-only 1` variant).

Differences from the analytic script, stated plainly:
  * fitness = episode return + 100 * landed (no hazard-distance term: the term needs per-env hazard maps),
  * common random numbers hold only per env index, not per candidate (spawns follow the env index),
  * the 2 aim dims are present (LUNARSIM_AIM_SCALE=15) but the policy is open-loop.
Wall-clock per generation is the quantity of interest; it is dominated by the simulation, not by the fitness form.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
import numpy as np
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--gens", type=int, default=2)
p.add_argument("--pairs", type=int, default=16)
p.add_argument("--eps-per-cand", type=int, default=4)
p.add_argument("--sigma", type=float, default=0.15)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--sel", type=int, default=24)
p.add_argument("--sel-every", type=int, default=3, help="selection block every N generations (and at the last)")
p.add_argument("--n-envs", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--aim-scale", type=float, default=15.0)
p.add_argument("--out", required=True)
a = p.parse_args()

os.environ["LUNARSIM_AIM_SCALE"] = str(a.aim_scale)
app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.evo.es import OpenAIES  # noqa: E402
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS  # noqa: E402
from lunarsim.rl.guided_gain_vec_env import GuidedGainVecEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402

OVR = ["reward_scale=0", "touchdown_k=120", "touchdown_penalty_cap=480", "landed_step_k=100", "timeout_penalty=480"]


def main():
    st = STAGES_BY_NAME[a.stage]
    venv = IsaacLanderVecEnv(
        num_envs=a.n_envs, tile_fn=make_tile_fn(st, 80), params=st.params,
        env_spacing_m=max(s.tile_size_m for s in STAGES) * 1.5, seed=7, max_rocks_per_env=0,
        reward_fn=make_apollo_reward_fn(weights_from_overrides(OVR)))
    gv = GuidedGainVecEnv(venv, gain_spec=GAIN_SETS["attitude"], hold_steps=20, residual_scale=0.0)
    n_act = int(gv.action_space.shape[0]); n = a.n_envs
    max_dec = int(st.params.max_episode_s / st.params.dt_s / 20) + 4 \
        + int(float(os.environ.get("LUNARSIM_SETTLE_S", "4.0")) / st.params.dt_s / 20) + 2

    def wave(acts: np.ndarray, wave_seed: int):
        """acts: (n, n_act) one fixed action per env. Returns (return, landed) per env (first episode only)."""
        gv.venv._rng = np.random.default_rng(wave_seed)
        gv.reset()
        ret = np.zeros(n); landed = np.zeros(n, bool); done = np.zeros(n, bool)
        for _ in range(max_dec):
            _, rew, dones, infos = gv.step(acts.astype(np.float32))
            ret += np.where(done, 0.0, rew)
            for i in range(n):
                if not done[i] and dones[i]:
                    done[i] = True; landed[i] = bool(infos[i].get("landed_safely", False))
            if done.all():
                break
        return ret, landed

    def evaluate(thetas, seeds_base, eps_per):
        """Fly every (theta, k) job, n_envs at a time. Returns mean fitness and landed rate per theta."""
        jobs = [(c, k) for c in range(len(thetas)) for k in range(eps_per)]
        fit = np.zeros(len(thetas)); land = np.zeros(len(thetas))
        for w0 in range(0, len(jobs), n):
            batch = jobs[w0:w0 + n]
            batch = batch + [batch[-1]] * (n - len(batch))             # pad the last wave
            acts = np.stack([np.tanh(thetas[c]) for c, _ in batch])
            ret, ok = wave(acts, seeds_base + w0)
            for j, (c, _) in enumerate(jobs[w0:w0 + n]):
                fit[c] += ret[j] + 100.0 * ok[j]; land[c] += ok[j]
        return fit / eps_per, land / eps_per

    es = OpenAIES(np.zeros(n_act), a.sigma, a.lr, a.pairs, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows = []; best = [-1e18]
    t_all = time.time()
    print(f"ES on ISAAC {a.stage}: {a.gens} gens, {a.pairs} pairs x2 x {a.eps_per_cand} eps = "
          f"{2 * a.pairs * a.eps_per_cand} episodes/gen, {n} envs/wave", flush=True)
    for g in range(a.gens):
        t0 = time.time()
        pop, eps = es.ask()
        fit, land = evaluate(pop, 200000 + 1000 * g, a.eps_per_cand)
        es.tell(eps, fit)
        t_train = time.time() - t0
        t_sel = 0.0; sel_land = float("nan")
        if g % a.sel_every == a.sel_every - 1 or g == a.gens - 1:
            ts = time.time()
            sf, sl = evaluate(es.theta[None, :], 150000, a.sel)
            sel_land = float(sl[0]); t_sel = time.time() - ts
            if float(sf[0]) > best[0]:
                best[0] = float(sf[0])
                np.savez(os.path.join(a.out, "best_on_selection.npz"), theta=es.theta.copy(), sizes=np.array([0, n_act]),
                         mu=np.zeros(1), sd=np.ones(1), aim_scale=a.aim_scale, gain_set="attitude")
        rows.append(dict(gen=g, gen_seconds=t_train, sel_seconds=t_sel, cand_land=float(land.mean()), sel_land=sel_land))
        print(f"g{g:3d} gen {t_train:6.1f}s  sel {t_sel:5.1f}s  cand land {land.mean():.2f}  sel land {sel_land:.2f}", flush=True)
    json.dump(dict(stage=a.stage, n_envs=n, pairs=a.pairs, eps_per_cand=a.eps_per_cand, rows=rows,
                   total_seconds=time.time() - t_all), open(os.path.join(a.out, "timing.json"), "w"))


main()
app.close()
