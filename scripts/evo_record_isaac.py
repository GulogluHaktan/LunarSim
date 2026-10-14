"""[evo] Record trajectories of the controller and the controller-free ES policy, on Isaac,
(copied from record_guided_gain_isaac.py; the ES arm flies RAW [throttle,pitch,roll,yaw]).

Original: Record full trajectories of the controller and a guided-gain policy, on Isaac,
through the SAME env and the SAME wave seeds the paired validation used.

The video of a result should show the result. `isaaclab_controller_eval_capture.py`
builds its own Isaac Lab scene with a different spawn distribution and its own
actuator layer, so a clip from it would show a different system from the one
that scored 172/192. This script instead drives `GuidedGainVecEnv` over
`IsaacLanderVecEnv` exactly as `validate_guided_gain_isaac.py` does -- re-seed
`_rng`, reset all envs, keep each env's FIRST episode -- and taps every control
step, so a wave recorded here reproduces that validation wave's outcomes
episode for episode (the script checks this against the validation JSON when
one is given).

Output: one .npz per arm with per-control-step state for every env, the
per-decision gain multipliers, each episode's outcome and graded touchdown
state, and the terrain height map each env flew over -- everything a renderer
needs and nothing it has to re-simulate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("npz")
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--stage", type=str, default="orbit_descent")
parser.add_argument("--gain-set", type=str, default="attitude")
parser.add_argument("--hold-steps", type=int, default=20)
parser.add_argument("--n-envs", type=int, default=12)
parser.add_argument("--wave-seed", type=int, default=90000)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--reward-weight", action="append", default=None)
parser.add_argument("--check-json", type=str, default=None,
                    help="validation JSON to cross-check this wave's outcomes against")
parser.add_argument("--out-dir", type=str, default="out/recordings")
parser.add_argument("--torch-device", type=str, default="cuda")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.guided_gain_env import GAIN_SETS  # noqa: E402
from lunarsim.rl.guided_gain_vec_env import GuidedGainVecEnv  # noqa: E402
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides  # noqa: E402
from lunarsim.evo.policies import MLPPolicy  # noqa: E402
from lunarsim.evo.hazfeat import N_HAZ, haz_features  # noqa: E402
from lunarsim.core.terrain.hazard import attach_hazard  # noqa: E402

FIELDS = ("x", "y", "vx", "vy", "vz", "tilt_x", "tilt_y", "yaw", "wx", "wy", "wz",
          "fuel_kg", "throttle")


def record_wave(gvenv, model, wave_seed, max_decisions, raw=False):
    venv = gvenv.venv
    n = gvenv.num_envs
    n_act = int(gvenv.action_space.shape[0])
    steps = [[] for _ in range(n)]           # per env: list of per-step dicts
    gains = [[] for _ in range(n)]           # per env: per-decision multipliers
    outcome = [None] * n
    settled = np.zeros(n, dtype=bool)

    orig_step = venv.step

    def tapped(cmd):
        # Snapshot BEFORE the physics step: after a terminating step the env has
        # already auto-reset, so the post-step state belongs to the next episode.
        for i in range(n):
            if settled[i]:
                continue
            clearance, ground = venv._contact(i)
            ox, oy, _ = venv.env_origins[i]
            row = {k: float(venv.state[k][i]) for k in FIELDS}
            row["x"] -= float(ox)
            row["y"] -= float(oy)
            row["alt"] = float(clearance)
            row["ground_z"] = float(ground)
            row["cmd"] = [float(c) for c in cmd[i]]
            steps[i].append(row)
        obs, rew, dones, infos = orig_step(cmd)
        for i in range(n):
            if settled[i] or not dones[i]:
                continue
            info = infos[i]
            imp = info.get("impact_state") or {}
            outcome[i] = {
                "landed": bool(info.get("landed_safely", False)),
                "lost_control": bool(info.get("lost_control", False)),
                "timed_out": bool(info.get("timed_out", False)),
                "touched": bool(info.get("landing_margins")),
                "margins": {k: float(v) for k, v in (info.get("landing_margins") or {}).items()},
                "impact": {k: float(v) for k, v in imp.items()
                           if isinstance(v, (int, float, np.floating))},
            }
            settled[i] = True
        return obs, rew, dones, infos

    venv.step = tapped
    try:
        venv._rng = np.random.default_rng(wave_seed)
        obs = venv.reset() if raw else gvenv.reset()
        tiles = [venv._tiles[i] for i in range(n)]
        for _ in range(max_decisions):
            if raw:
                act = model(obs)
                for i in range(n):
                    if not settled[i]:
                        gains[i].append([float(v) for v in np.clip(act[i], -1, 1)])
                obs, _r, _d, _inf = venv.step(act)
                if settled.all():
                    break
                continue
            if model is None:
                act = np.zeros((n, n_act), dtype=np.float32)
            else:
                act, _ = model.predict(obs, deterministic=True)
            for i in range(n):
                if not settled[i]:
                    gains[i].append([
                        float(k.ratio ** float(np.clip(act[i, j], -1, 1)))
                        for j, k in enumerate(gvenv.gain_spec)])
            obs, _r, _d, _inf = gvenv.step(act)
            if settled.all():
                break
    finally:
        venv.step = orig_step
    return steps, gains, outcome, tiles


def save(path, steps, gains, outcome, tiles, gain_names, dt, hold):
    out = {"dt": dt, "hold_steps": hold, "gain_names": np.array(gain_names)}
    for i, (st, g, oc, tile) in enumerate(zip(steps, gains, outcome, tiles)):
        for k in steps[i][0]:
            if k == "cmd":
                out[f"e{i}_cmd"] = np.array([r["cmd"] for r in st], dtype=np.float32)
            else:
                out[f"e{i}_{k}"] = np.array([r[k] for r in st], dtype=np.float64)
        out[f"e{i}_gains"] = np.array(g, dtype=np.float32)
        out[f"e{i}_outcome"] = np.array(json.dumps(oc))
        out[f"e{i}_tile_height"] = np.asarray(tile.height, dtype=np.float32)
        out[f"e{i}_tile_res"] = float(tile.res_m)
        out[f"e{i}_tile_size"] = float(tile.size_m)
    np.savez_compressed(path, n_envs=len(steps), **out)


def main():
    stage = STAGES_BY_NAME[args.stage]
    spec = GAIN_SETS[args.gain_set]
    max_dec = int(stage.params.max_episode_s / stage.params.dt_s / args.hold_steps) + 4
    max_raw = int(stage.params.max_episode_s / stage.params.dt_s) + 4
    max_tile = max(s.tile_size_m for s in STAGES)
    venv = IsaacLanderVecEnv(
        num_envs=args.n_envs, tile_fn=make_tile_fn(stage, args.terrain_grid_n),
        params=stage.params, env_spacing_m=max_tile * 1.5, seed=args.terrain_seed,
        max_rocks_per_env=0,
        reward_fn=make_apollo_reward_fn(weights_from_overrides(
            ["reward_scale=0", "touchdown_k=120", "touchdown_penalty_cap=480",
             "landed_step_k=100", "timeout_penalty=480"])))
    gvenv = GuidedGainVecEnv(venv, gain_spec=spec, hold_steps=args.hold_steps, residual_scale=0.0)
    z = np.load(args.npz)
    obs_dim = int(z["sizes"][0])
    pol = MLPPolicy(obs_dim, 4, tuple(int(h) for h in z["sizes"][1:-1]), float(z["throttle_bias"]))
    theta, mu, sd = z["theta"], z["mu"], z["sd"]
    use_haz = obs_dim == 20 + N_HAZ

    def es_act(o):
        rows = []
        for i in range(o.shape[0]):
            row = o[i].astype(float)
            if use_haz:
                tl = attach_hazard(venv._tiles[i]); ox, oy = float(venv.env_origins[i][0]), float(venv.env_origins[i][1])
                row = np.concatenate([row, haz_features(float(venv.state["x"][i]) - ox, float(venv.state["y"][i]) - oy,
                                                        float(venv.state["vx"][i]), float(venv.state["vy"][i]),
                                                        tl.haz_dist, tl.res_m)])
            rows.append(pol.act(theta, (row - mu) / sd))
        return np.stack(rows).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for arm in ("controller", "policy"):
        if arm == "controller":
            steps, gains, outcome, tiles = record_wave(gvenv, None, args.wave_seed, max_dec)
            names, hold = [k.name for k in spec], args.hold_steps
        else:
            steps, gains, outcome, tiles = record_wave(gvenv, es_act, args.wave_seed, max_raw, raw=True)
            names, hold = ["throttle", "pitch", "roll", "yaw"], 1
        path = os.path.join(args.out_dir, f"evo_wave{args.wave_seed}_{arm}.npz")
        save(path, steps, gains, outcome, tiles, names, stage.params.dt_s, hold)
        results[arm] = [o["landed"] for o in outcome]
        print(f"[{arm}] landed {sum(results[arm])}/{len(outcome)}  -> {path}", flush=True)
    disc = [i for i in range(args.n_envs) if results["policy"][i] and not results["controller"][i]]
    print(f"[discordant] policy landed, controller did not: envs {disc}", flush=True)
    if args.check_json:
        d = json.load(open(args.check_json))
        seeds = d["seeds"]; w = seeds.index(args.wave_seed); n = args.n_envs
        ok_c = d["controller"]["landed"][w * n:(w + 1) * n] == results["controller"]
        ok_p = d["es_policy"]["landed"][w * n:(w + 1) * n] == results["policy"]
        print(f"[check] matches validation wave {args.wave_seed}: controller {ok_c} policy {ok_p}", flush=True)


main()
simulation_app.close()
