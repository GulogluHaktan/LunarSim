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
args = ap.parse_args()

w = RewardWeights()
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
