"""Paired Isaac evaluation of the MPPI-MPC (no ZEM/ZEV, no learned policy) vs the ZEM/ZEV
controller, same wave seeds, same process. Wave logic copied from scripts/evo_isaac_eval.py."""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("params_npy")
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--n-envs", type=int, default=12)
p.add_argument("--waves", type=int, default=16)
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
from lunarsim.rl.reward import RewardWeights, make_apollo_reward_fn  # noqa: E402
from lunarsim.evo.mpc import MPPIMPC, MPCConfig  # noqa: E402
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


def mpc_state(venv, n):
    s = venv.state
    S = {k: np.asarray(s[k], dtype=np.float64).copy() for k in
         ("vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz", "fuel_kg")}
    S["alt"] = np.array([venv._contact(i)[0] for i in range(n)])
    S["px"] = np.array([float(s["x"][i]) - float(venv.env_origins[i][0]) for i in range(n)])
    S["py"] = np.array([float(s["y"][i]) - float(venv.env_origins[i][1]) for i in range(n)])
    S["t_left"] = venv.params.max_episode_s - np.asarray(venv._t, dtype=np.float64)
    return S


def main():
    st = STAGES_BY_NAME[a.stage]
    venv = IsaacLanderVecEnv(
        num_envs=a.n_envs, tile_fn=make_tile_fn(st), params=st.params,
        env_spacing_m=max(s.tile_size_m for s in STAGES) * 1.5, seed=a.terrain_seed,
        max_rocks_per_env=0, reward_fn=make_apollo_reward_fn(RewardWeights()))
    gv = GuidedGainVecEnv(venv, gain_spec=GAIN_SETS["attitude"], hold_steps=20, residual_scale=0.0)
    cfg = MPCConfig(K=512, M=20, hold=12, exec_steps=4, roll_sub=2, margin=0.7)
    lp = np.load(a.params_npy)
    mpc = MPPIMPC(st.params, RewardWeights(), cfg, lp)
    max_raw = int(st.params.max_episode_s / st.params.dt_s) + 4
    seeds = [a.wave_seed0 + 1000 * k for k in range(a.waves)]
    print(f"MPC on ISAAC {a.stage}: {a.waves} waves x {a.n_envs}, seeds {seeds[0]}..{seeds[-1]}", flush=True)
    C, P, times = [], [], []
    IMP = []   # per-episode touchdown state (first episode of each env)
    for sdv in seeds:
        t0 = time.time()
        if not a.no_controller:
            gv.venv._rng = np.random.default_rng(sdv)
            gv.reset(); res = [None] * a.n_envs
            for _ in range(max_raw // 20 + 4):
                _, _, dones, infos = gv.step(np.zeros((a.n_envs, 11), np.float32))
                for i in range(a.n_envs):
                    if res[i] is None and dones[i]: res[i] = why_of(infos[i])
                if all(r is not None for r in res): break
            C += [r or "unfinished" for r in res]
        # MPC arm: raw actions on the same venv, replan every exec_steps
        venv._rng = np.random.default_rng(sdv)
        venv.reset(); mpc.reset(a.n_envs)
        mpc.U = None; mpc._since_shift = 0; mpc.reset(a.n_envs)
        res = [None] * a.n_envs; A = np.zeros((a.n_envs, 4), np.float32); tm = time.time(); tile_ids = []
        imp = [None] * a.n_envs
        for step in range(max_raw):
            if step % cfg.exec_steps == 0:
                if os.environ.get("LUNARSIM_HAZ", "1") != "0":
                    ids = [id(t) for t in venv._tiles]
                    if ids != tile_ids:
                        tl = [attach_hazard(t) for t in venv._tiles]
                        mpc.set_hazard([t.haz_dist for t in tl], tl[0].size_m, tl[0].res_m, [t.height for t in tl]); tile_ids = ids
                A = mpc.act(mpc_state(venv, a.n_envs)).astype(np.float32)
            venv.step_async(A); _, _, dones, infos = venv.step_wait()
            for i in range(a.n_envs):
                if res[i] is None and dones[i]:
                    res[i] = why_of(infos[i])
                    inf = infos[i]; ist = inf.get("impact_state") or {}
                    imp[i] = dict(
                        why=res[i], t=float(venv._t[i]) if False else None,
                        impact={k: float(v) for k, v in ist.items() if isinstance(v, (int, float, np.floating))},
                        margins={k: float(v) for k, v in (inf.get("landing_margins") or {}).items()})
            if all(r is not None for r in res): break
        P += [r or "unfinished" for r in res]
        IMP += [x if x is not None else dict(why="unfinished") for x in imp]
        times.append(time.time() - t0)
        print(f"  seed {sdv}: ctrl {sum(r=='landed' for r in C)}/{len(C)} "
              f"mpc {sum(r=='landed' for r in P)}/{len(P)}  ({times[-1]:.0f}s, mpc {time.time()-tm:.0f}s)", flush=True)
    out = {}
    for name, R in (("controller", C), ("mpc", P)):
        if not R: continue
        why = {k: R.count(k) for k in sorted(set(R))}
        print(f"{name:11s} {R.count('landed')}/{len(R)} = {100*R.count('landed')/len(R):.1f}%  {why}")
        out[name] = dict(landed=[r == "landed" for r in R], why=why)
    if C and P:
        b, c, pv = mcnemar(np.array(out["controller"]["landed"]), np.array(out["mpc"]["landed"]))
        print(f"McNemar: mpc-only {b}, controller-only {c}, p={pv:.2e}")
    if a.out_json:
        json.dump(dict(stage=a.stage, seeds=seeds, params=np.load(a.params_npy).tolist(),
                       cfg=cfg.__dict__, wave_seconds=times, mpc_impact=IMP, **out), open(a.out_json, "w"))


main()
app.close()
