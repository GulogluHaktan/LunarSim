"""Paired Isaac evaluation: ES raw-action policy (no controller) vs the ZEM/ZEV
controller, same wave seeds, same process. Wave logic copied from
scripts/validate_guided_gain_isaac.py (SimulationApp must precede lunarsim imports).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("npz", nargs="?", default=None)
p.add_argument("--stage", default="ramp_35m")
p.add_argument("--n-envs", type=int, default=12)
p.add_argument("--waves", type=int, default=2)
p.add_argument("--wave-seed0", type=int, default=90000)
p.add_argument("--no-controller", action="store_true")
p.add_argument("--terrain-seed", type=int, default=7)
p.add_argument("--out-json", default=None)
a = p.parse_args()

app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS  # noqa: E402
from lunarsim.rl.guided_gain_vec_env import GuidedGainVecEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402
from lunarsim.evo.policies import MLPPolicy
from lunarsim.evo.hazfeat import N_HAZ, haz_features
from lunarsim.core.terrain.hazard import attach_hazard  # noqa: E402


def mcnemar(x, y):
    b = int(np.sum(y & ~x)); c = int(np.sum(x & ~y)); n = b + c
    if n == 0: return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def why_of(info):
    if info.get("landed_safely"): return "landed"
    if info.get("lost_control"): return "tumble"
    if info.get("left_tile"): return "left_tile"
    if info.get("timed_out"): return "timeout"
    return "crash"


def wave_raw(venv, act_fn, seed, max_steps):
    n = venv.num_envs
    venv._rng = np.random.default_rng(seed)
    obs = venv.reset()
    res = [None] * n
    for _ in range(max_steps):
        venv.step_async(act_fn(obs)); obs, rew, dones, infos = venv.step_wait()
        for i in range(n):
            if res[i] is None and dones[i]:
                res[i] = why_of(infos[i])
        if all(r is not None for r in res): break
    return [r or "unfinished" for r in res]


def main():
    st = STAGES_BY_NAME[a.stage]
    w = weights_from_overrides(["reward_scale=0", "touchdown_k=120", "touchdown_penalty_cap=480",
                                "landed_step_k=100", "timeout_penalty=480"])
    venv = IsaacLanderVecEnv(
        num_envs=a.n_envs, tile_fn=make_tile_fn(st), params=st.params,
        env_spacing_m=max(s.tile_size_m for s in STAGES) * 1.5, seed=a.terrain_seed,
        max_rocks_per_env=0, reward_fn=make_apollo_reward_fn(w))
    gv = GuidedGainVecEnv(venv, gain_spec=GAIN_SETS["attitude"], hold_steps=20, residual_scale=0.0)
    max_raw = int(st.params.max_episode_s / st.params.dt_s) + 4
    seeds = [a.wave_seed0 + 1000 * k for k in range(a.waves)]
    print(f"stage {a.stage} on ISAAC, {a.waves} waves x {a.n_envs}, seeds {seeds[0]}..{seeds[-1]}", flush=True)
    pol = None
    if a.npz:
        z = np.load(a.npz)
        obs_dim = int(z["sizes"][0])
        pol = MLPPolicy(obs_dim, 4, tuple(int(h) for h in z["sizes"][1:-1]), float(z["throttle_bias"]))
        theta, mu, sd = z["theta"], z["mu"], z["sd"]
        assert len(theta) == pol.n_params and obs_ok(venv)
        use_haz = obs_dim == 20 + N_HAZ

        def aug(o):
            if not use_haz:
                return o
            rows = []
            for i in range(o.shape[0]):
                tl = attach_hazard(venv._tiles[i]); ox, oy = float(venv.env_origins[i][0]), float(venv.env_origins[i][1])
                rows.append(np.concatenate([o[i], haz_features(float(venv.state["x"][i]) - ox, float(venv.state["y"][i]) - oy,
                                                              float(venv.state["vx"][i]), float(venv.state["vy"][i]),
                                                              tl.haz_dist, tl.res_m)]))
            return np.stack(rows)
    C, P = [], []
    for sdv in seeds:
        t0 = time.time()
        if not a.no_controller:
            # controller arm = zero guided-gain action (exactly the controller), 1 s holds
            gv.venv._rng = np.random.default_rng(sdv)
            obs = gv.reset(); res = [None] * a.n_envs
            for _ in range(max_raw // 20 + 4):
                obs, rew, dones, infos = gv.step(np.zeros((a.n_envs, 11), np.float32))
                for i in range(a.n_envs):
                    if res[i] is None and dones[i]: res[i] = why_of(infos[i])
                if all(r is not None for r in res): break
            C += [r or "unfinished" for r in res]
        if pol is not None:
            P += wave_raw(venv, lambda o: np.stack([pol.act(theta, (a_[i].astype(float) - mu) / sd)
                                                    for a_ in [aug(o)] for i in range(a_.shape[0])]).astype(np.float32),
                          sdv, max_raw)
        print(f"  seed {sdv}: ctrl {sum(r=='landed' for r in C)}/{len(C)} "
              f"es {sum(r=='landed' for r in P)}/{len(P)}  ({time.time()-t0:.0f}s)", flush=True)
    out = {}
    for name, R in (("controller", C), ("es_policy", P)):
        if not R: continue
        why = {k: R.count(k) for k in sorted(set(R))}
        print(f"{name:11s} {R.count('landed')}/{len(R)} = {100*R.count('landed')/len(R):.1f}%  {why}")
        out[name] = dict(landed=[r == "landed" for r in R], why=why)
    if C and P:
        b, c, pv = mcnemar(np.array(out["controller"]["landed"]), np.array(out["es_policy"]["landed"]))
        print(f"McNemar: policy-only {b}, controller-only {c}, p={pv:.2e}")
    if a.out_json:
        json.dump(dict(stage=a.stage, seeds=seeds, npz=a.npz, **out), open(a.out_json, "w"))


def obs_ok(venv):
    return venv.observation_space.shape[0] == 20


main()
app.close()
