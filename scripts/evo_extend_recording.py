"""Append a post-touchdown REST tail to one env of a recording.

The recorder snapshots state BEFORE each step and the env terminates at first
contact, so a recording ends a few cm above the ground with the vehicle still
moving. This script (1) lets the vehicle finish the last cm at its recorded
vertical speed, then (2) holds it at rest on the ground for --rest-s seconds
with the engine cut. That is what the env's contact model produces anyway
(impact_state: restitution 0 stops the vehicle); it is an appended hold, not
re-simulation, and the report says so.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("recording"); ap.add_argument("--env", type=int, required=True)
ap.add_argument("--rest-s", type=float, default=4.0); ap.add_argument("--out", required=True)
a = ap.parse_args()
d = dict(np.load(a.recording, allow_pickle=False))
dt = float(d["dt"]); e = a.env
alt = d[f"e{e}_alt"]; vz = d[f"e{e}_vz"]
n_fall = int(np.ceil(max(alt[-1], 0.0) / max(abs(vz[-1]) * dt, 1e-6))) if alt[-1] > 1e-3 else 0
n_fall = min(n_fall, 40); n_rest = int(round(a.rest_s / dt))
tail = n_fall + n_rest
zeros_on_rest = ("vx", "vy", "vz", "wx", "wy", "wz", "throttle")
for key in [k for k in d if k.startswith(f"e{e}_")]:
    name = key[len(f"e{e}_"):]
    arr = d[key]
    if name in ("outcome", "tile_height", "tile_res", "tile_size", "gains") or arr.ndim == 0 or len(arr) != len(alt):
        continue
    ext = np.repeat(arr[-1:], tail, axis=0)
    if name == "alt":
        ext[:n_fall] = np.maximum(alt[-1] - abs(vz[-1]) * dt * np.arange(1, n_fall + 1), 0.0)
        ext[n_fall:] = 0.0
    if name in zeros_on_rest:
        ext[n_fall:] = 0.0
    if name == "cmd":
        ext[:] = 0.0
    d[key] = np.concatenate([arr, ext], axis=0)
g = d[f"e{e}_gains"]
d[f"e{e}_gains"] = np.concatenate([g, np.zeros((tail + 2, g.shape[1]), g.dtype)], axis=0)
np.savez_compressed(a.out, **d)
print(f"env {e}: +{n_fall} fall steps +{n_rest} rest steps ({a.rest_s}s) -> {a.out}")
