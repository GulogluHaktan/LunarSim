"""Cut the demos down to an altitude band, keeping them a valid demo file.

Needed to seed a landing specialist from data that already exists. The orbit demos hold
61,837 transitions below 25 m from 303 landed episodes -- the entry envelope `land_handoff`
releases into -- and from-scratch RL measured 0.0% (SAC) and 0.0-1.2% (PPO) on that stage,
while the only thing that has ever produced a competent policy in this project is a clone.

What this does NOT give you, and why DAgger is not optional afterwards: the low-altitude
segments come from the controller's own descent, so the state at 25 m is whatever the
controller happened to produce, not land_handoff's randomised entry (tilt +/-15 deg, vz in
[-5,-1], lateral 0-6 m/s). A clone fitted on this alone is good near the controller's own
handoff and blind at the edges -- which is the "handoff distribution does not occur"
failure the stage definition itself warns about. DAgger rolling the clone out IN
land_handoff and relabelling with the controller is what covers the rest.

Episode boundaries are rebuilt per contiguous run inside the band, so an episode that
enters, leaves and re-enters becomes separate episodes rather than one with a
discontinuity. `landed` is carried from the source episode and only set for runs that
reach the ground.

Run:
    .venv/bin/python scripts/extract_altitude_band.py --max-alt 25 \
        --out out/zemzev_band25.npz out/zemzev_orbit_demos_s2*.npz
"""
from __future__ import annotations

import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--out", required=True)
p.add_argument("--max-alt", type=float, default=25.0)
p.add_argument("--min-alt", type=float, default=0.0)
p.add_argument("--min-run", type=int, default=20, help="drop runs shorter than this")
p.add_argument("--landed-only", action="store_true", default=True)
args = p.parse_args()

from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

ALT = float(OBS_SCALE[2])
keys = ("obs", "next_obs", "actions", "rewards", "dones", "truncated")
acc = {k: [] for k in keys}
ends, landed = [], []
total = 0
src_eps = kept_eps = 0

for f in args.files:
    d = np.load(f, allow_pickle=True)
    have = [k for k in keys if k in d]
    alt_all = d["obs"].astype(np.float32)[:, 2] * ALT
    e_ends, e_land = d["episode_ends"], d["landed"].astype(bool)
    starts = np.concatenate([[0], e_ends[:-1]])
    for s, e, ok in zip(starts, e_ends, e_land):
        src_eps += 1
        if args.landed_only and not ok:
            continue
        a = alt_all[s:e]
        inside = (a >= args.min_alt) & (a <= args.max_alt)
        # contiguous runs inside the band
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        for run in np.split(idx, splits + 1):
            if run.size < args.min_run:
                continue
            lo, hi = s + run[0], s + run[-1] + 1
            for k in have:
                acc[k].append(d[k][lo:hi])
            total += hi - lo
            ends.append(total)
            # only a run that actually reaches the ground inherits the landing
            landed.append(bool(ok and alt_all[hi - 1] <= 0.5))
            kept_eps += 1

if not ends:
    raise SystemExit("nothing in the band")
out = {k: np.concatenate(v) for k, v in acc.items() if v}
out["episode_ends"] = np.asarray(ends, dtype=np.int64)
out["landed"] = np.asarray(landed, dtype=bool)
src = np.load(args.files[0], allow_pickle=True)
for k in ("format_version", "reward_weights"):
    if k in src:
        out[k] = src[k]
np.savez_compressed(args.out, **out)
print(f"[band] {args.min_alt:g}-{args.max_alt:g} m from {len(args.files)} file(s)")
print(f"  source episodes {src_eps}, kept runs {kept_eps}, transitions {total}")
print(f"  runs that reach the ground: {int(np.sum(landed))}/{len(landed)}")
print(f"  -> {args.out}")
