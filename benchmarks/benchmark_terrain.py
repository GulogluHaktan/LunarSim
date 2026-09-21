"""Per-profile terrain generation benchmark -> benchmarks/benchmarks.csv.

Only measures what is meaningful without a renderer/GPU (Isaac adapter isn't
wired up yet): tile generation wall time at each profile's terrain
resolution. `step_s`, `fps`, `vram_mb` columns are left blank here for the
Isaac adapter's benchmark pass to fill in once it exists (see plan section 9,
rule 4: "Her profil için otomatik benchmark").

Usage: python benchmarks/benchmark_terrain.py [--size-m 500] [--out benchmarks/benchmarks.csv]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from lunarsim.core.quality.profiles import _DEFAULT_PROFILES_PATH, quality_from_dict
from lunarsim.core.terrain import TerrainConfig, generate_tile

import yaml


def run(size_m: float, out_path: str, seed: int = 0) -> None:
    with open(_DEFAULT_PROFILES_PATH) as f:
        profile_names = list(yaml.safe_load(f)["profiles"].keys())

    rows = []
    for name in profile_names:
        q = quality_from_dict({"quality": {"profile": name, "overrides": {}}})

        cfg = TerrainConfig(
            mode="fine",
            size_m=size_m,
            res_m=q.terrain["res_m"],
            seed=seed,
            hills={"amplitude_m": 3.0, "wavelength_m": 100.0, "hurst": 0.75},
            craters={"count_scale": 1.0, "d_min_m": 0.5, "d_max_m": 60.0, "b": 2.5,
                     "depth_ratio": 0.1, "age": 0.3},
            rocks={"density_scale": 1.0, "d_max_m": 1.0},
            roi={"sigma_m": 100.0, "centers": None},
            curvature=True,
        )

        t0 = time.perf_counter()
        tile = generate_tile(cfg)
        dt = time.perf_counter() - t0

        rows.append({
            "profile": name,
            "size_m": size_m,
            "res_m": q.terrain["res_m"],
            "grid_n": tile.height.shape[0],
            "terrain_gen_s": round(dt, 4),
            "n_craters": len(tile.craters.diameter_m),
            "n_rocks": len(tile.rocks.diameter_m),
            "step_s": "",
            "fps": "",
            "vram_mb": "",
        })
        print(f"{name:10s} res={q.terrain['res_m']:>5} grid={tile.height.shape[0]:>5} gen={dt:.3f}s")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-m", type=float, default=500.0)
    parser.add_argument("--out", type=str, default="benchmarks/benchmarks.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.size_m, args.out, args.seed)
