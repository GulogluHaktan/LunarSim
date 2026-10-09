"""Recompute observation slots 16-19 in a recorded demo file.

Slots 16-19 are a deterministic function of slots 0-5 (altitude, pad offsets, velocity)
through `obs_norm.guidance_obs`, so when the guidance field changes they can be rebuilt
exactly, without re-running Isaac. Everything else in the file is untouched.

Why this is needed rather than optional. The lateral target changed from zero to a measured
speed schedule, which moved slot 16 by a mean of 8.055 m/s. A clone trained before that change
is handed an input whose meaning has shifted under it, and the orbit_descent clone duly went
from 22% to 2% on the SAME checkpoint and the SAME seed block. Nothing errors; the numbers
just quietly stop meaning what they did. Repairing the file and retraining is the only way the
clone and the env agree again.

The file is rewritten in place only after the new slots are computed successfully, and the
original is kept alongside with a .pre-repair suffix.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunarsim.rl.obs_norm import OBS_SCALE, guidance_obs  # noqa: E402
from lunarsim.rl.reward import RewardWeights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--reward-weight", action="append", default=[], metavar="NAME=VALUE",
                help="override a RewardWeights field, repeatable. REQUIRED when the "
                     "field's own constants are what changed: the slots must be rebuilt "
                     "with the SAME weights the run will use, or the policy is trained on "
                     "a guidance error computed from one schedule while the reward grades "
                     "it against another. Without this the script silently used the "
                     "defaults, which is the train/eval skew that took this project's "
                     "orbit clone from 22%% to 2%% on an unchanged checkpoint.")
args = ap.parse_args()

from dataclasses import fields as _fields  # noqa: E402

_valid = {f.name for f in _fields(RewardWeights)}
_over = {}
for _item in args.reward_weight:
    if "=" not in _item:
        raise SystemExit(f"--reward-weight wants NAME=VALUE, got {_item!r}")
    _n, _, _v = _item.partition("=")
    _n = _n.strip()
    if _n not in _valid:
        raise SystemExit(f"no RewardWeights field named {_n!r}")
    _over[_n] = float(_v)
w = RewardWeights(**_over)
print(f"[repair] rebuilding slots 16-19 with weights: "
      f"{_over if _over else 'DEFAULTS'}", flush=True)
for path in args.files:
    d = dict(np.load(path, allow_pickle=True))
    for key in ("obs", "next_obs"):
        if key not in d:
            continue
        o = d[key].astype(np.float64)
        if o.shape[1] != len(OBS_SCALE):
            raise SystemExit(f"{path}: {key} is {o.shape[1]}-wide, expected {len(OBS_SCALE)}")
        raw = o * OBS_SCALE
        new = np.array([guidance_obs(w, raw[i, 2], raw[i, 0], raw[i, 1],
                                     raw[i, 3], raw[i, 4], raw[i, 5])
                        for i in range(len(raw))], dtype=np.float64)
        old_n = raw[:, 16:20]
        shift = np.abs(new - old_n).mean(axis=0)
        raw[:, 16:20] = new
        d[key] = (raw / OBS_SCALE).astype(np.float32)
        print(f"{path}  {key}: mean |shift| per slot = {np.round(shift, 3)}", flush=True)
    if args.dry_run:
        print(f"{path}: dry run, not written", flush=True)
        continue
    backup = path + ".pre-repair"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    np.savez_compressed(path, **d)
    print(f"{path}: rewritten (original kept at {backup})", flush=True)
