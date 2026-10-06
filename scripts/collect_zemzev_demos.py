"""Collect `ZemZevController` demonstration rollouts from the REAL Isaac
Sim/PhysX `IsaacLanderEnv` (not `AnalyticLanderEnv`) at `orbit_descent`'s
release condition, to seed SAC's replay buffer before training starts.

Why Isaac Sim, not the fast analytic env `test_landing_feasibility.py`
uses: the user has been explicit all session that analytic-env numbers
don't represent real Isaac Sim behavior (confirmed: this same controller
got 75% landed_safely on the analytic env but only ~40-50% on real Isaac
Sim). Seeding an Isaac-trained critic's replay buffer with analytic-env
transitions would feed it wrong next_obs/reward dynamics for real
transitions it will actually be asked to value -- so demos have to come
from the same backend the policy trains against.

Why bootstrap at all: two from-scratch/warm-started orbit_descent runs
this session (one 13M-step warm start, one fresh 2M-step run), BOTH under
the now-correctly-scaled reward, still reached 0/16 `landed_safely` --
near-hover is a cheap, easy-to-find attractor under random exploration
(not because the vehicle is underpowered -- T/W is 1.84 at PDI mass --
but because action[0]=0, the centre of the action box and the mean of an
untrained policy, maps to T/W 1.016; see train_sac_isaac.py's STAGES
comment for the arithmetic), so the policy never once samples a real
touchdown to learn its value from. Seeding the
critic with real successful (and failed) landing trajectories from a
controller already PROVEN to solve this scenario gives it that signal
directly, without relying on the policy's own exploration to find it.

Output format (v2): FLAT arrays shaped (N_transitions, ...) plus
`episode_ends`, not the old (T, group_size, ...) lane layout.

REAL BUG THIS REPLACES, found by inspecting the demo file an actual
2M-step demo-bootstrapped training run consumed
(`out/zemzev_orbit_descent_demos.npz`, 48 episodes, 2489 x 16 = 39824
transitions): the v1 writer packed episodes into `group_size` parallel
"lanes" and then truncated every lane in a group to `t_min`, the length of
the SHORTEST episode in that group -- so every longer episode lost its
tail, INCLUDING its terminal transition. The file contains **4 `done=True`
flags in 39824 transitions**, out of 48 episodes that each ended in one,
and `truncated` is all-False. The entire point of seeding the buffer is to
show the critic what a completed landing is worth; 44 of the 48 landings
were silently cut off before they happened. Episode lengths at this
release condition genuinely vary ~2-3x (22-62 s), so the truncation was
not a small trim. A flat layout has no lane length to reconcile and keeps
every transition; `train_sac_isaac.py` chunks it into `n_envs`-sized
`ReplayBuffer.add` calls at load time, which also drops the old
"group_size must equal --n-envs" constraint.

`reward_weights` is recorded alongside the transitions so the loader can
refuse a file whose stored rewards were computed by a DIFFERENT reward
function than the one training is about to optimize -- `reward.py` was
re-tuned several times a day during this work, and stale demo rewards
poison the critic exactly where it is being asked to learn the most.

Run:
    /home/haktan/isaac-env/bin/python scripts/collect_zemzev_demos.py \
        --episodes 48 --out out/zemzev_orbit_descent_demos.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--episodes", type=int, default=48)
parser.add_argument("--out", type=str, default="out/zemzev_orbit_descent_demos.npz")
parser.add_argument("--stage", type=str, default="orbit_descent",
                     help="which curriculum stage to fly. Was hard-wired to "
                          "orbit_descent; behaviour cloning needs demos from "
                          "whichever stage is being imitated.")
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80,
                     help="must match train_sac_isaac.py's --terrain-grid-n for the run that "
                          "will consume these demos -- it sets res_m, i.e. the terrain the "
                          "transitions were actually flown over.")
parser.add_argument("--seed0", type=int, default=1000)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, args.lunarsim_root)

from lunarsim.adapters.isaac.isaac_lander_env import IsaacLanderEnv  # noqa: E402
from lunarsim.control.zemzev_controller import ZemZevController  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.reward import RewardWeights, default_reward_fn  # noqa: E402

# the ONE curriculum definition -- this used to be a hand-copied
# orbit_descent Stage + terrain config (see lunarsim/rl/curriculum.py's
# docstring on the four copies that had drifted apart).
_STAGE = STAGES_BY_NAME[args.stage]
ORBIT_DESCENT_PARAMS = _STAGE.params  # kept name; it is args.stage's params
DEMO_FORMAT_VERSION = 2


class _BellyStateView:
    """Adapter so `ZemZevController.act` sees the altitude the Isaac env
    actually flies at.

    REAL BUG FOUND (this session): this script used to call
    `controller.act(env)` on an `IsaacLanderEnv` directly. The controller
    computes `alt = state["z"] - _ground_z(x, y)` -- correct for
    `AnalyticLanderEnv`, whose vehicle is a zero-height point, but
    `IsaacLanderEnv.state["z"]` is the real rigid body's CENTRE, half the
    vehicle's 7.04 m height above its belly. So every demo in
    `out/zemzev_orbit_descent_demos.npz` was flown by a controller that
    believed it was **3.52 m higher than it was**, and whose ZEM/ZEV
    vertical channel was therefore solving for touchdown 3.52 m BELOW the
    ground. `scripts/isaaclab_controller_eval_capture.py` already gets this
    right (it explicitly feeds `"z": belly_z` into its own state view);
    only this script did not. 3.52 m is ~4.4x the controller's own
    `final_approach_alt_m = 8.0 m` trigger's worth of error at the single
    most safety-critical point of the flight.

    Uses the same contact geometry the env's own touchdown test uses (see
    `isaac_lander_env.contact_clearance_m`), so "altitude 0" means the same
    thing to the controller and to the termination check.
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


def main():
    tile = generate_tile(terrain_config(_STAGE, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=ORBIT_DESCENT_PARAMS, reward_fn=default_reward_fn,
                          seed=args.seed0, lunarsim_root=args.lunarsim_root)
    controller = ZemZevController(ORBIT_DESCENT_PARAMS)
    view = _BellyStateView(env)
    max_steps = int(ORBIT_DESCENT_PARAMS.max_episode_s / ORBIT_DESCENT_PARAMS.dt_s) + 5

    obs_l, next_obs_l, act_l, rew_l, done_l, trunc_l, ends = [], [], [], [], [], [], []
    n_safe = n_lost = n_left = 0
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        obs, _ = env.reset(seed=seed)
        controller.reset()
        info = {}
        n_before = len(obs_l)
        for _ in range(max_steps):
            view.sync()
            action = controller.act(view)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_l.append(obs)
            next_obs_l.append(next_obs)
            act_l.append(action)
            rew_l.append(reward)
            done_l.append(terminated or truncated)
            trunc_l.append(truncated and not terminated)
            obs = next_obs
            if terminated or truncated:
                break
        ends.append(len(obs_l))

        landed = bool(info.get("landed_safely"))
        lost = bool(info.get("lost_control"))
        left = bool(info.get("left_tile"))
        n_safe += landed
        n_lost += lost
        n_left += left
        print(f"[demo ep {ep}] landed_safely={landed} lost_control={lost} left_tile={left} "
              f"steps={len(obs_l) - n_before} t_s={info.get('t_s', float('nan')):.1f}")

    print(f"=== {n_safe}/{args.episodes} landed_safely ({100 * n_safe / args.episodes:.0f}%), "
          f"{n_lost}/{args.episodes} lost_control, {n_left}/{args.episodes} left_tile "
          f"-- REAL Isaac Sim ===")

    dones = np.array(done_l, dtype=bool)
    # the whole reason this file exists: if this count is not == --episodes,
    # the terminal transitions are being lost again (see module docstring --
    # the v1 format shipped 4 of them in 39824 rows).
    print(f"terminal transitions kept: {int(dones.sum())}/{args.episodes}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        format_version=np.array(DEMO_FORMAT_VERSION),
        obs=np.array(obs_l, dtype=np.float32),
        next_obs=np.array(next_obs_l, dtype=np.float32),
        actions=np.array(act_l, dtype=np.float32),
        rewards=np.array(rew_l, dtype=np.float32),
        dones=dones,
        truncated=np.array(trunc_l, dtype=bool),
        episode_ends=np.array(ends, dtype=np.int64),
        # fingerprint of the reward function that produced `rewards`, so the
        # loader can refuse a stale file (see module docstring).
        reward_weights=np.array(json.dumps(asdict(RewardWeights()), sort_keys=True)),
    )
    print(f"saved {len(obs_l)} transitions from {args.episodes} episodes -> {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
