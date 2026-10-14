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
    """`ZemZevController.act` wants an env with `.params`, `.state` and
    `._ground_z` -- and it must be handed the altitude the env actually flies
    at, not the rigid body's centre height.

    REAL BUG THIS FIXES. This class used to forward every attribute straight
    through to `IsaacLanderEnv`, including `state` and `_ground_z`. The
    controller computes `alt = state["z"] - _ground_z(x, y)`, which is correct
    for `AnalyticLanderEnv` (a zero-height point model) but wrong here:
    `IsaacLanderEnv.state["z"]` is the body CENTRE, and `reset` places it at
    `ground_z + alt + half_height_m` with `half_height_m = 7.04/2 = 3.52`. So
    the controller believed it was 3.52 m higher than it was at zero tilt, and
    `3.52*cos(tilt) + 4.70*sin(tilt)` = 4.62 m higher at 15 deg -- while its
    ZEM/ZEV vertical channel solved for touchdown that far BELOW the ground.

    It mattered most on exactly the stage it was used to measure. This script
    is the source of the `land_handoff` controller baseline, and that stage
    releases from `spawn_altitude_m = (8.0, 25.0)`, so the bias was 14-58% of
    the whole altitude budget. Concretely, the controller's final-approach
    trigger `alt <= max(8.0, ...)` fired at a TRUE clearance of 4.48 m (zero
    tilt) to 3.38 m (15 deg) instead of 8 m, cutting the terminal descent
    window from 10.0 s to 5.6-4.2 s -- against the ~8.7 s the RCS needs to
    slew to its 20 deg final-approach cap. Any baseline this script printed
    measured the harness, not the controller.

    `collect_zemzev_demos._BellyStateView` and
    `scripts/isaaclab_controller_eval_capture.py` already did this correctly;
    this was the one remaining site that did not. Using `env._contact()` means
    "altitude 0" denotes the same thing to the controller, to
    `info["altitude_m"]`, to observation slot 2 and to the termination test.
    """

    def __init__(self, env):
        self._env = env
        self.params = env.params
        self.state: dict = {}
        self._ground_z_m = 0.0

    def sync(self) -> None:
        clearance_m, ground_z = self._env._contact()
        self.state = dict(self._env.state)
        self.state["z"] = ground_z + clearance_m
        self._ground_z_m = ground_z

    def _ground_z(self, x: float, y: float) -> float:
        return self._ground_z_m

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

    safe = crash = timeout = lost = left = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed0 + ep)
        ctrl.reset()
        info = {}
        for _ in range(max_steps):
            view.sync()
            a = ctrl.act(view)
            _o, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            if term or trunc:
                break
        if info.get("landed_safely"):
            safe += 1
        elif info.get("lost_control"):
            lost += 1
        elif info.get("timed_out", False):
            timeout += 1
        elif not info.get("terminated"):
            # Truncated without timing out is `left_tile`, a different failure
            # from running out of clock -- it used to be counted as a timeout,
            # which hid flying off the map inside the one bucket nobody reads
            # as a failure of control.
            left += 1
        else:
            crash += 1
        if (ep + 1) % 24 == 0:
            print(f"  {ep + 1}/{args.episodes}: {safe} landed", flush=True)

    n = args.episodes
    print(f"\ncontroller on {args.stage}: {safe}/{n} landed_safely "
          f"({safe / n:.0%}), {crash} crash, {lost} lost_control, "
          f"{timeout} timeout, {left} left_tile "
          f"(seed0={args.seed0}, terrain_seed={args.terrain_seed})")


main()
simulation_app.close()
