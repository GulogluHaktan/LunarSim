"""Is a SPLIT critic's VALUE calibrated? Measured offline, in the critic's own units.

WHY. v77's training log reported a Q0-MC gap pinned at +28..+30 with CQL(alpha=5), the
Cal-QL clamp, LayerNorm and dual-gamma all verified active, which reads as "none of them
calibrated anything". That reading was not available from that number: the in-training
gauge was comparing the min over the WHOLE four-head ensemble against a single-gamma
total return, while a dual-gamma critic represents

    Q = min(shaping pair)  +  min(terminal pair)
      ~ sum g_s^t r_shaping  +  sum g_t^t r_terminal

so neither side was the other's target (fixed in lunarsim/rl/monitoring.py, pinned by
tests/test_proxy_gauge_dual_gamma.py). This script answers the question for a checkpoint
already on disk, without retraining, by rolling the deterministic policy out and
accumulating each stream with ITS OWN discount from `info["reward_terminal"]` --
the same split the replay buffer feeds the critic.

It also reports the gap the OLD convention would have printed, so the correction's size
is visible rather than asserted, and splits the result by outcome: a critic can be
calibrated on the episodes that land and badly wrong on the ones that crash, and the
mean hides exactly that.

Run:
    /home/haktan/isaac-env/bin/python scripts/diag_critic_calibration.py \
        --checkpoint out/sac_v77_rebuilt/snapshots/orbit_descent_410000_steps.zip \
        --stage orbit_descent --episodes 32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str,
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stage", type=str, required=True)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--seed0", type=int, default=7000)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--max-rocks-per-env", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})
sys.path.insert(0, args.lunarsim_root)

import torch as th  # noqa: E402

from lunarsim.adapters.isaac.isaac_lander_env import IsaacLanderEnv  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.eval_loader import load_with_critic_for_diag  # noqa: E402
from lunarsim.rl.reward import default_reward_fn  # noqa: E402


def main():
    if args.stage not in STAGES_BY_NAME:
        raise SystemExit(f"no stage named {args.stage!r}")
    stage = STAGES_BY_NAME[args.stage]
    params = stage.params
    tile = generate_tile(terrain_config(stage, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=params, reward_fn=default_reward_fn,
                         seed=args.seed0, lunarsim_root=args.lunarsim_root)

    model, dual = load_with_critic_for_diag(args.checkpoint, device="cpu")
    if dual is None:
        print("[calib] this checkpoint is NOT dual-gamma; the single-gamma convention "
              "applies and the in-training gauge was already measuring the right thing.")
    g_s, g_t = dual if dual is not None else (0.99, 0.99)
    max_steps = int(params.max_episode_s / params.dt_s) + 5

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed0 + ep)
        a0 = None
        s0 = np.array(obs, dtype=np.float32)
        disc_s = disc_t = disc_single = 0.0
        info = {}
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            if a0 is None:
                a0 = np.array(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            r_term = float(info.get("reward_terminal", 0.0))
            r_shape = float(reward) - r_term
            disc_s += (g_s ** t) * r_shape
            disc_t += (g_t ** t) * r_term
            disc_single += (0.99 ** t) * float(reward)
            if terminated or truncated:
                break
        with th.no_grad():
            qs = model.critic(th.as_tensor(s0).unsqueeze(0), th.as_tensor(a0).unsqueeze(0))
            vals = [float(q) for q in qs]
        half = len(vals) // 2
        q_dual = min(vals[:half]) + min(vals[half:])
        q_old = min(vals)
        rows.append({
            "landed": bool(info.get("landed_safely")),
            "q_dual": q_dual, "mc_dual": disc_s + disc_t,
            "q_old": q_old, "mc_old": disc_single,
            "mc_shaping": disc_s, "mc_terminal": disc_t,
            "q_shaping": min(vals[:half]), "q_terminal": min(vals[half:]),
        })
        print(f"  ep {ep:>3} landed={rows[-1]['landed']!s:<5} "
              f"Q={q_dual:+9.2f} MC={disc_s + disc_t:+9.2f} gap={q_dual - (disc_s + disc_t):+8.2f}",
              flush=True)

    def summarise(sel, label):
        sub = [r for r in rows if sel(r)]
        if not sub:
            print(f"  {label:<14} (none)")
            return
        gd = np.mean([r["q_dual"] - r["mc_dual"] for r in sub])
        go = np.mean([r["q_old"] - r["mc_old"] for r in sub])
        print(f"  {label:<14} n={len(sub):<4} "
              f"CORRECT gap {gd:+8.2f}   (old convention would print {go:+8.2f})   "
              f"Q {np.mean([r['q_dual'] for r in sub]):+8.2f} "
              f"MC {np.mean([r['mc_dual'] for r in sub]):+8.2f}")

    print(f"\n[calib] {args.checkpoint}  stage={args.stage} dual_gamma={dual} "
          f"episodes={len(rows)}")
    summarise(lambda r: True, "all")
    summarise(lambda r: r["landed"], "landed")
    summarise(lambda r: not r["landed"], "failed")
    print(f"  per stream:   shaping Q {np.mean([r['q_shaping'] for r in rows]):+8.2f} "
          f"vs MC {np.mean([r['mc_shaping'] for r in rows]):+8.2f}   |   "
          f"terminal Q {np.mean([r['q_terminal'] for r in rows]):+8.2f} "
          f"vs MC {np.mean([r['mc_terminal'] for r in rows]):+8.2f}")
    print("  A stream whose Q and MC disagree is the one to fix; the sum can look "
          "calibrated while both halves are wrong in opposite directions.")


main()
simulation_app.close()
