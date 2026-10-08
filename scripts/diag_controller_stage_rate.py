"""The CONTROLLER's landed_safely rate on any stage, on the REAL Isaac env.

scripts/test_landing_feasibility.py measures the controller too, but on AnalyticLanderEnv
(where it reads 84.5% against ~50% on Isaac) and only over the names in STAGES, which
excludes the SPECIALIST_STAGES the two-model split uses. A split whose landing half is
being judged needs the number its own backend produces on its own stage, or there is
nothing to compare the specialist against.

Deliberately the same env/stage/seed construction as diag_stage_landing_rate.py, so a
controller row and a policy row on the same stage and seed block are comparable.

Run:
    /home/haktan/isaac-env/bin/python scripts/diag_controller_stage_rate.py \
        --stage land_handoff --episodes 96 --seed0 7000
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
parser.add_argument("--stage", type=str, required=True)
parser.add_argument("--episodes", type=int, default=96)
parser.add_argument("--seed0", type=int, default=7000)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})
sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.isaac_lander_env import IsaacLanderEnv  # noqa: E402
from lunarsim.control.zemzev_controller import ZemZevController  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.reward import default_reward_fn  # noqa: E402


class _View:
    """`ZemZevController.act` wants an env with `.params`, `.state` and `._ground_z`."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


def main():
    if args.stage not in STAGES_BY_NAME:
        raise SystemExit(f"no stage named {args.stage!r}; "
                         f"choices: {list(STAGES_BY_NAME)}")
    stage = STAGES_BY_NAME[args.stage]
    params = stage.params
    tile = generate_tile(terrain_config(stage, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=params, reward_fn=default_reward_fn,
                         seed=args.seed0, lunarsim_root=args.lunarsim_root)
    ctrl = ZemZevController(params)
    view = _View(env)
    max_steps = int(params.max_episode_s / params.dt_s) + 5

    safe = crash = timeout = lost = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed0 + ep)
        ctrl.reset()
        info = {}
        for _ in range(max_steps):
            a = ctrl.act(view)
            _o, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            if term or trunc:
                break
        if info.get("landed_safely"):
            safe += 1
        elif info.get("lost_control"):
            lost += 1
        elif info.get("timed_out", False) or (not info.get("terminated")):
            timeout += 1
        else:
            crash += 1
        if (ep + 1) % 24 == 0:
            print(f"  {ep + 1}/{args.episodes}: {safe} landed", flush=True)

    n = args.episodes
    print(f"\ncontroller on {args.stage}: {safe}/{n} landed_safely "
          f"({safe / n:.0%}), {crash} crash, {lost} lost_control, {timeout} timeout "
          f"(seed0={args.seed0}, terrain_seed={args.terrain_seed})")


main()
simulation_app.close()
