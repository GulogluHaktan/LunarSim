"""Fly a trained SAC checkpoint through ITS OWN stage's release condition
(not necessarily orbit_descent) on the real local IsaacLanderEnv, and
report landed_safely / lost_control / timeout counts -- a quick "did this
intermediate ramp-curriculum checkpoint actually solve ITS OWN easier
stage" check, distinct from scripts/diag_policy_telemetry.py (which is
orbit_descent-only and prints full step-by-step telemetry).

Run:
    /home/haktan/isaac-env/bin/python scripts/diag_stage_landing_rate.py \
        --checkpoint out/sac_training_run_ramp/sac_lunar_lander_isaac_ramp_20m.zip \
        --stage ramp_20m --episodes 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--stage", type=str, required=True,
                     help="a stage name from lunarsim/rl/curriculum.py's STAGES")
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80,
                     help="must match the --terrain-grid-n the checkpoint was trained with")
parser.add_argument("--seed0", type=int, default=7000)
parser.add_argument("--legacy-obs15", action="store_true",
                     help="emit the ORIGINAL 16-wide observation, slot 15 held at the "
                          "constant 0.0 it had before it was repurposed to carry "
                          "time-remaining. Required to evaluate a checkpoint trained "
                          "before that change: back when the live vector was also "
                          "16-wide it LOADED without complaint while the policy had "
                          "arbitrary weights on a slot it only ever saw as zero, so a "
                          "measured difference would be attributed to whatever else "
                          "changed. The live vector is now 20-wide, so such a checkpoint "
                          "is rejected on shape unless this is passed.")
parser.add_argument("--legacy-obs16", action="store_true",
                     help="emit the 16-wide observation as it stood AFTER slot 15 became "
                          "time-remaining and BEFORE the 16 -> 20 widening added the "
                          "velocity-error/t_go slots. For checkpoints trained in that "
                          "window.")
parser.add_argument("--stochastic", action="store_true",
                     help="sample from the policy instead of taking its mean. Training "
                          "collects data this way while evaluation is deterministic, so a "
                          "large gap between the two means exploration noise is costing "
                          "landings that the mean policy would make -- which poisons the "
                          "replay buffer with failures the policy did not intend.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, args.lunarsim_root)

from stable_baselines3 import SAC  # noqa: E402

from lunarsim.adapters.isaac.isaac_lander_env import IsaacLanderEnv  # noqa: E402
from lunarsim.core.terrain.generate import generate_tile  # noqa: E402
from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config  # noqa: E402
from lunarsim.rl.reward import default_reward_fn  # noqa: E402

# the ONE curriculum definition (see lunarsim/rl/curriculum.py's docstring
# on the four hand-copied, drifted versions this replaces -- this script's
# own copy was already a second place the ramp stages had to be edited by
# hand to stay in step with the trainer).



def _overshoot_ratios(state, info, params) -> dict:
    """How far over its limit each `landed_safely` criterion actually is.

    The clipped margins cannot answer this -- they saturate at 0.00 the moment a
    limit is exceeded, so they rank nothing and hide magnitude. Read the impact
    state the env publishes where available, for the same reason the env grades
    from it: after contact the post-substep velocity is ~0.
    """
    import numpy as _np
    s = info.get("impact_state") or state
    vz = abs(float(s.get("vz", 0.0)))
    vxy = float(_np.hypot(float(s.get("vx", 0.0)), float(s.get("vy", 0.0))))
    tilt = float(_np.hypot(float(s.get("tilt_x", 0.0)), float(s.get("tilt_y", 0.0))))
    w = float(_np.linalg.norm([float(s.get("wx", 0.0)), float(s.get("wy", 0.0)),
                               float(s.get("wz", 0.0))]))
    margins = info.get("landing_margins") or {}
    out = {
        "v_z": vz / max(params.safe_landing_v_z_m_s, 1e-6),
        "v_xy": vxy / max(params.safe_landing_v_xy_m_s, 1e-6),
        "tilt": tilt / max(params.safe_landing_tilt_rad, 1e-6),
        "w": w / max(params.safe_landing_w_rad_s, 1e-6),
    }
    # leg_diff is not recoverable from the state, so invert its margin
    if "leg_diff" in margins:
        m = float(margins["leg_diff"])
        out["leg_diff"] = (1.0 - m) if m > 0.0 else 1.0
    return out


def main():
    if args.stage not in STAGES_BY_NAME:
        raise SystemExit(f"no stage named {args.stage!r}; choices: {list(STAGES_BY_NAME)}")
    stage = STAGES_BY_NAME[args.stage]
    params = stage.params
    tile = generate_tile(terrain_config(stage, args.terrain_seed, args.terrain_grid_n))
    env = IsaacLanderEnv(tile=tile, params=params, reward_fn=default_reward_fn,
                          seed=args.seed0, lunarsim_root=args.lunarsim_root,
                          legacy_obs15=args.legacy_obs15,
                          legacy_obs16=args.legacy_obs16)
    model = SAC.load(args.checkpoint, device="cpu")
    max_steps = int(params.max_episode_s / params.dt_s) + 5

    n_safe = n_lost = n_timeout = n_left = n_crash = 0
    rows = []
    # UNDISCOUNTED shaped return per episode, kept alongside the outcome tag.
    # Without it the two questions "does the policy land" and "does the reward
    # PREFER the policy that lands" cannot be separated, and this project spent
    # a long time assuming the second without measuring it: if a policy that
    # lands 44% of the time scores a higher return than one that lands 58%, no
    # amount of RL tuning will close the gap, because RL is maximising the thing
    # it was given. The reward_fn here is the same `default_reward_fn` the
    # training runs use, so these numbers are directly comparable to them.
    returns_by_tag = {}
    ep_returns = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        obs, _ = env.reset(seed=seed)
        info = {}
        ep_return = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                break
        landed = bool(info.get("landed_safely"))
        lost = bool(info.get("lost_control"))
        # left_tile is a THIRD way to not land, distinct from timing out:
        # the vehicle drifted off the terrain collider entirely (see
        # isaac_lander_env.out_of_tile). Counting it separately is the
        # whole point -- lumped into "timeout" it looks like indecision
        # when it is actually an undersized tile.
        left = bool(info.get("left_tile"))
        timed_out = truncated and not terminated and not left
        # EXCLUSIVE accounting. The four original counters were not
        # exhaustive: an episode that touched down, stayed upright and missed a
        # `landed_safely` criterion incremented nothing and appeared only as a
        # CRASH tag in the per-episode rows. Measured over this project's own
        # logs that was the DOMINANT outcome -- across four v21 snapshots the
        # exclusive tags were CRASH 45, TIMEOUT 42, LANDED 9, while the summary
        # line reported 9 + 42 + 0 + 0 and left 45 episodes invisible. Reading
        # those summaries, "the policy will not commit to descending" was the
        # natural diagnosis when in fact it committed every time and missed on
        # precision.
        if landed:
            n_safe += 1
        elif lost:
            n_lost += 1
        elif left:
            n_left += 1
        elif timed_out:
            n_timeout += 1
        else:
            n_crash += 1
        s = env.state
        v_xy = float(np.hypot(s["vx"], s["vy"]))
        tilt_deg = float(np.degrees(np.hypot(s["tilt_x"], s["tilt_y"])))
        w_mag = float(np.linalg.norm([s.get("wx", 0.0), s.get("wy", 0.0), s.get("wz", 0.0)]))
        tag = ("LANDED" if landed else "LOST_CONTROL" if lost
               else "LEFT_TILE" if left else "TIMEOUT" if timed_out else "CRASH")
        # A touchdown that misses `landed_safely` used to print only vz/vxy/
        # tilt, and a real measured episode came back CRASH with vz=-0.21,
        # vxy=0.48, tilt=7.1 -- every printed number comfortably inside its
        # limit. The deciding criterion was one of the two this line never
        # showed: |w| (limit `safe_landing_w_rad_s`) and the leg height
        # difference (limit `safe_landing_max_leg_height_diff_m`). Those are
        # also exactly the two criteria the per-step reward has no term for,
        # so a silent failure on them reads as an unexplained crash. Print
        # the clearance, |w|, and whichever margins the env actually
        # computed, and name the binding criterion outright.
        margins = info.get("landing_margins") or {}
        # `min(margins, key=margins.get)` was WRONG and quietly so: every margin
        # is `clip(1 - x/limit, 0, 1)`, so any criterion that EXCEEDS its limit
        # saturates to exactly 0.00 and all violated criteria tie at the floor.
        # `min` then returned whichever came first in the dict, which is `v_z`.
        # Measured over 183 real touchdown rows: 50 had two or more criteria at
        # 0.00 and 44 of those 50 reported `worst=v_z` by accident, i.e. 24% of
        # all touchdown rows named a criterion at random. Conclusions about which
        # criterion binds were drawn from this.
        #
        # Rank by how far OVER the limit each criterion actually is, which the
        # clipped margin cannot express (1.001x over and 12x over are both 0.00).
        over = _overshoot_ratios(env.state, info, params)
        worst = max(over, key=over.get) if over else None
        detail = ((f"  margins={{" + ", ".join(f"{k}={v:.2f}" for k, v in margins.items()) + "}"
                   + (f" worst={worst}({over[worst]:.2f}x)" if worst else ""))
                  if margins else "")
        returns_by_tag.setdefault(tag, []).append(ep_return)
        ep_returns.append(ep_return)
        rows.append(f"[ep {ep}] {tag:12s} R={ep_return:9.2f}  t_s={info.get('t_s', float('nan')):5.1f}  "
                     f"alt={info.get('altitude_m', float('nan')):6.3f}  "
                     f"vz={s['vz']:7.2f}  vxy={v_xy:6.2f}  tilt={tilt_deg:5.1f}  w={w_mag:5.2f}{detail}")
        print(rows[-1])

    # counters are exclusive and sum to --episodes; the assert is the point
    assert n_safe + n_lost + n_left + n_timeout + n_crash == args.episodes
    print(f"\n=== {args.stage}: {n_safe}/{args.episodes} landed_safely "
          f"({100*n_safe/args.episodes:.0f}%), {n_crash} crash, {n_timeout} timeout, "
          f"{n_lost} lost_control, {n_left} left_tile "
          f"(sum={n_safe + n_lost + n_left + n_timeout + n_crash}) "
          f"-- checkpoint={args.checkpoint} stage={args.stage} "
          f"episodes={args.episodes} seed0={args.seed0} "
          f"terrain_seed={args.terrain_seed} grid_n={args.terrain_grid_n} "
          f"stochastic={args.stochastic} legacy_obs15={args.legacy_obs15} "
          f"legacy_obs16={args.legacy_obs16} ===")
    # Mean AND worst decile: a mean return hides the tail that the landing rate
    # is actually made of, and this project's own notes call for tracking the
    # worst percentile rather than the mean for exactly that reason.
    ordered = sorted(ep_returns)
    decile = max(1, len(ordered) // 10)
    print(f"=== return: mean={np.mean(ep_returns):.2f} "
          f"worst_decile={np.mean(ordered[:decile]):.2f} "
          f"best_decile={np.mean(ordered[-decile:]):.2f} ===")
    for tag in sorted(returns_by_tag):
        v = returns_by_tag[tag]
        print(f"===   {tag:12s} n={len(v):3d}  mean_R={np.mean(v):9.2f}  "
              f"min={min(v):9.2f}  max={max(v):9.2f} ===")


if __name__ == "__main__":
    main()
    simulation_app.close()
