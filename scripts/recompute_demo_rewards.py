"""Recompute the demos' per-step reward under different RewardWeights, exactly.

Needed because a PPO clone's value head is fitted to the demos' discounted return, so
changing the reward invalidates it -- and running a new reward with a value head fitted to
the old one would change two things at once, leaving the result unattributable.

The method keeps the TERMINAL exact. The demos do not store `impact_state` or
`landing_margins`, so the touchdown severity cannot be regraded from them. It does not
need to be: the terminal is unchanged by a shaping-weight edit, so

    new[t] = stored[t] - old_shaping[t] + new_shaping[t]

leaves whatever terminal the last step carried untouched and replaces only the part that
moved. Both shaping values are computed from the SAME stored observation, so the
subtraction cancels every term the edit did not touch, including any the reconstruction
would get slightly wrong.

The state is rebuilt from the stored observation, which carries position offsets,
altitude, velocities, the quaternion and the body rates -- everything the shaping terms
read.

Run:
    .venv/bin/python scripts/recompute_demo_rewards.py \
        --demos out/zemzev_orbit_demos_s20000.npz --out-suffix _vfnorm \
        --reward-weight vfield_norm_p=0.75 --reward-weight vfield_norm_c=3.8171
"""
from __future__ import annotations

import argparse
from dataclasses import fields

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--demos", action="append", required=True)
p.add_argument("--out-suffix", default="_rw")
p.add_argument("--reward-weight", action="append", default=[], metavar="NAME=VALUE")
p.add_argument("--stage", default="orbit_descent")
args = p.parse_args()

from lunarsim.rl.curriculum import STAGES_BY_NAME  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402
from lunarsim.rl.reward import RewardWeights, make_apollo_reward_fn  # noqa: E402

valid = {f.name for f in fields(RewardWeights)}
over = {}
for item in args.reward_weight:
    name, _, val = item.partition("=")
    if name not in valid:
        raise SystemExit(f"no RewardWeights field named {name!r}")
    over[name] = float(val)
if not over:
    raise SystemExit("pass at least one --reward-weight, or there is nothing to recompute")
print(f"[recompute] overrides: {over}")

stage = STAGES_BY_NAME[args.stage]
fn_old = make_apollo_reward_fn(RewardWeights())
fn_new = make_apollo_reward_fn(RewardWeights(**over))


class _Tile:
    size_m = stage.tile_size_m


class _Env:
    params = stage.params
    tile = _Tile()

    def __init__(self):
        self.state = {}


def _state_from_obs(o):
    """The shaping terms' inputs, out of the stored (normalised) observation."""
    u = o * OBS_SCALE[:len(o)]
    qw, qx, qy, qz = float(u[6]), float(u[7]), float(u[8]), float(u[9])
    # small-angle tilt from the quaternion, the same convention the envs publish
    tilt_x = 2.0 * (qw * qx + qy * qz)
    tilt_y = 2.0 * (qw * qy - qz * qx)
    return {"x": float(u[0]), "y": float(u[1]), "z": float(u[2]),
            "vx": float(u[3]), "vy": float(u[4]), "vz": float(u[5]),
            "tilt_x": float(tilt_x), "tilt_y": float(tilt_y),
            "wx": float(u[10]), "wy": float(u[11]), "wz": float(u[12]),
            "fuel_kg": float(u[13]) * stage.params.initial_fuel_kg,
            "rcs_fuel_kg": float(u[14]) * stage.params.initial_rcs_fuel_kg}


for f in args.demos:
    d = dict(np.load(f, allow_pickle=True))
    obs = d["obs"].astype(np.float32)
    act = d["actions"].astype(np.float32)
    r_old_stored = d["rewards"].astype(np.float64)
    env = _Env()
    new = np.empty_like(r_old_stored)
    for i in range(len(obs)):
        env.state = _state_from_obs(obs[i])
        info = {"altitude_m": env.state["z"], "action": act[i],
                "terminated": False, "truncated": False}
        s_old = fn_old(env, info)
        env.state = _state_from_obs(obs[i])
        info2 = {"altitude_m": env.state["z"], "action": act[i],
                 "terminated": False, "truncated": False}
        s_new = fn_new(env, info2)
        new[i] = r_old_stored[i] - s_old + s_new
        if i % 100000 == 0 and i:
            print(f"  {f}: {i}/{len(obs)}", flush=True)
    d["rewards"] = new.astype(np.float32)
    out = f.replace(".npz", f"{args.out_suffix}.npz")
    np.savez_compressed(out, **d)
    print(f"[recompute] {f}\n  stored mean {r_old_stored.mean():+.4f} -> "
          f"{new.mean():+.4f}   (shaping only; terminal preserved)\n  -> {out}", flush=True)
