"""Hand-coded aim rule (v_ref = A * direction-to-nearest-safe-cell) on the analytic backend: how much does lateral aim buy?"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.rl.guided_gain_env import GuidedGainEnv
from lunarsim.rl.curriculum import STAGES_BY_NAME, make_tile_fn
from lunarsim.rl.reward import make_apollo_reward_fn
from lunarsim.core.terrain.hazard import attach_hazard
from lunarsim.evo.hazfeat import safe_dir, _bilerp
p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=48); p.add_argument("--seed0", type=int, default=150000)
p.add_argument("--A", type=float, nargs="+", default=[0, 4, 8, 12]); p.add_argument("--fade", type=float, default=40.0); p.add_argument("--acc", type=float, default=0.0); p.add_argument("--pos", type=int, default=0); p.add_argument("--kp", type=float, default=0.0); p.add_argument("--ahead", type=int, default=0); p.add_argument("--pred", type=float, default=0.0); p.add_argument("--rmax", type=float, default=1e9); p.add_argument("--stage", default="orbit_descent")
a = p.parse_args()
st = STAGES_BY_NAME[a.stage]
env = GuidedGainEnv(tile_fn=make_tile_fn(st, 80), params=st.params, reward_fn=make_apollo_reward_fn(), hold_steps=20)
for A in a.A:
    land = 0; dists = []; tos = 0
    for k in range(a.n):
        obs, _ = env.reset(seed=a.seed0 + k); tl = attach_hazard(env.env.tile)
        while True:
            s = env.env.state
            d = safe_dir(s["x"], s["y"], tl.haz_dist, tl.res_m)
            if a.acc > 0 and A > 0:
                dist = float(_bilerp(tl.haz_dist, tl.res_m, np.array([s["x"]]), np.array([s["y"]]))[0])
                u = d / (np.linalg.norm(d) + 1e-9); sp = min(A, np.sqrt(2 * a.acc * dist)); d = u * (sp / max(A, 1e-9))
            env.controller.final_pos_kp = a.kp; env.controller.terrain_ahead = bool(a.ahead); env.controller.aim_fade_alt_m = a.fade; env.controller.pos_fade_alt_m = a.fade
            if a.pos and a.pred > 0:
                from lunarsim.evo.hazfeat import aim_point
                tg = env.controller.tgo
                if tg is None: tg = 0.0
                alt_ = s["z"] - env.env._ground_z(s["x"], s["y"])
                tg = max(tg, 2 * alt_ / max(-s["vz"], 1.0))
                px, py = s["x"] + s["vx"] * tg / a.pred, s["y"] + s["vy"] * tg / a.pred
                env.controller.aim_pos = aim_point(px, py, tl.haz_dist, tl.res_m)
            elif a.pos:
                dist = float(_bilerp(tl.haz_dist, tl.res_m, np.array([s["x"]]), np.array([s["y"]]))[0])
                u = d / (np.linalg.norm(d) + 1e-9); ap = (s["x"] + u[0] * dist * A, s["y"] + u[1] * dist * A) if dist > 0 else None
                env.controller.aim_pos = ap if dist <= a.rmax else None
            else:
                env.controller.v_ref = (A * d[0], A * d[1])
            obs, r, te, tr, info = env.step(np.zeros(env.action_space.shape[0], np.float32))
            if te or tr: break
            # env.step resets controller? v_ref is re-set each decision above
        s = env.env.state
        dists.append(float(_bilerp(tl.haz_dist, tl.res_m, np.array([s["x"]]), np.array([s["y"]]))[0]))
        land += bool(info["landed_safely"]); tos += int(bool(tr))
        if False:
            s_ = env.env.state; print("  near", round(dists[-1],1), "vxy %.2f vz %.2f tilt %.1f deg w %.2f" % (np.hypot(s_["vx"], s_["vy"]), s_["vz"], np.degrees(np.hypot(s_["tilt_x"], s_["tilt_y"])), np.linalg.norm([s_["wx"], s_["wy"], s_["wz"]])), {k: v for k, v in info.items() if isinstance(v, (bool, str))})
    print("dists", np.round(sorted(dists)).astype(int).tolist())
    print("timeouts", tos); print(f"A={A:5.1f}  landed {land}/{a.n}  touchdown dist-to-safe median {np.median(dists):6.1f} m  mean {np.mean(dists):6.1f}", flush=True)
