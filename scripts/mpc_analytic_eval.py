"""MPC (MPPI) on the analytic backend: n envs in lock-step, one batched replan every `hold` sim steps."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.evo.mpc import MPPIMPC, MPCConfig, PARAM_NAMES, DEFAULT_LOG
from lunarsim.core.terrain.hazard import attach_hazard

def make_env(stage, seed):
    from lunarsim.rl.curriculum import STAGES_BY_NAME, make_tile_fn
    from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv
    from lunarsim.rl.reward import make_apollo_reward_fn
    st = STAGES_BY_NAME[stage]
    tmax = float(os.environ.get("LUNARSIM_TMAX", "0"))
    if tmax > 0:   # tighter budget than the real one: tunes in the safety margin the Isaac backend turned out to need
        from dataclasses import replace
        st = replace(st, params=replace(st.params, max_episode_s=tmax))
    return AnalyticLanderEnv(tile_fn=make_tile_fn(st, 80), params=st.params,
                             reward_fn=make_apollo_reward_fn(), seed=seed), st

def run(stage, seeds, log_params=None, cfg=None, verbose=False):
    from lunarsim.rl.reward import RewardWeights
    n = len(seeds)
    envs = []
    for s in seeds:
        e, st = make_env(stage, s); e.reset(seed=int(s)); envs.append(e)
    mpc = MPPIMPC(st.params, RewardWeights(), cfg, log_params)
    mpc.reset(n)
    if os.environ.get("LUNARSIM_HAZ", "1") != "0":
        from lunarsim.core.terrain.hazard import attach_hazard
        tl = [attach_hazard(e.tile) for e in envs]
        mpc.set_hazard([t.haz_dist for t in tl], tl[0].size_m, tl[0].res_m, [t.height for t in tl])
    done = np.zeros(n, bool); res = [None] * n; t0 = time.time(); nrep = 0
    HANDOFF = float(os.environ.get("LUNARSIM_MPC_HANDOFF", "0"))
    handed = np.zeros(n, bool)
    if HANDOFF > 0:
        from lunarsim.control.zemzev_controller import ZemZevController
        ctrls = [ZemZevController(st.params) for _ in range(n)]
    cap = int(st.params.max_episode_s / st.params.dt_s) + 5
    stepi = 0
    while not done.all() and stepi < cap:
        S = {k: np.zeros(n) for k in ("alt", "vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz", "fuel_kg")}
        for i, e in enumerate(envs):
            s_ = e.state
            for k in S:
                S[k][i] = (s_["z"] - e._ground_z(s_["x"], s_["y"])) if k == "alt" else s_[k]
        S["t_left"] = np.array([st.params.max_episode_s - e._t for e in envs])
        S["px"] = np.array([e.state["x"] for e in envs]); S["py"] = np.array([e.state["y"] for e in envs])
        A = mpc.act(S); nrep += 1
        for _ in range(mpc.c.exec_steps):
            for i, e in enumerate(envs):
                if done[i]: continue
                act_i = A[i]
                if HANDOFF > 0:
                    if S["alt"][i] < HANDOFF or handed[i]:
                        handed[i] = True
                        act_i = ctrls[i].act(e)
                _, r, te, tr, info = e.step(act_i)
                if te or tr:
                    done[i] = True
                    res[i] = "landed" if info["landed_safely"] else "tumble" if info["lost_control"] else "crash" if te else "timeout"
            stepi += 1
    for i in range(n):
        if res[i] is None: res[i] = "timeout"
    fin = []; ratios = []
    pp = st.params
    for e in envs:
        s_ = e.state
        vxy = float(np.hypot(s_["vx"], s_["vy"])); tilt = float(np.hypot(s_["tilt_x"], s_["tilt_y"]))
        w = float(np.linalg.norm([s_["wx"], s_["wy"], s_["wz"]]))
        rt = (abs(s_["vz"]) / pp.safe_landing_v_z_m_s, vxy / pp.safe_landing_v_xy_m_s,
              tilt / pp.safe_landing_tilt_rad, w / pp.safe_landing_w_rad_s)
        ratios.append(rt)
        fin.append(max(rt))
    run.last_final_severity = fin
    run.last_ratios = ratios
    try:
        run.last_td_dist = [float(attach_hazard(e.tile).haz_dist[
            int(np.clip(round(e.state["x"] / e.tile.res_m + (e.tile.height.shape[0] - 1) / 2), 0, e.tile.height.shape[0] - 1)),
            int(np.clip(round(e.state["y"] / e.tile.res_m + (e.tile.height.shape[0] - 1) / 2), 0, e.tile.height.shape[0] - 1))])
            for e in envs]
    except Exception:
        run.last_td_dist = []
    return res, (time.time() - t0), nrep

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="ramp_35m"); p.add_argument("--seed0", type=int, default=150000)
    p.add_argument("--n", type=int, default=12); p.add_argument("--K", type=int, default=512)
    p.add_argument("--M", type=int, default=20)
    a = p.parse_args()
    cfg = MPCConfig(K=a.K, M=a.M)
    res, dt, nrep = run(a.stage, [a.seed0 + i for i in range(a.n)], None, cfg)
    print(a.stage, {k: res.count(k) for k in sorted(set(res))}, f"{dt:.1f}s, {nrep} replans, {1000*dt/nrep:.0f} ms/replan")


def eval_candidate(args):
    """Top-level (picklable) worker for spawn pools: (stages, log_params, seeds, cfg dict) -> (fitness, landing rate)."""
    stages, lp, seeds, cfg = args
    from lunarsim.evo.mpc import MPCConfig
    c = MPCConfig(**cfg)
    tot, land = 0.0, 0.0
    for st in stages:
        res, dt, _ = run(st, seeds, np.clip(np.asarray(lp), -3.0, 4.0), c)
        sev = np.array(run.last_final_severity)
        l = np.array([r == "landed" for r in res], float)
        tot += float(np.mean(l - 0.05 * np.minimum(sev, 10.0) * (1 - l))); land += l.mean()
    return tot / len(stages), land / len(stages)
