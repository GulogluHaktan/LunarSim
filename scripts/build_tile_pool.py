"""Pre-generate a pool of real-DEM terrain tiles (pickle list of Tile) for fast training/eval.

  LUNARSIM_LOLA_IMG=~/lunarsim_data/lroc/NAC_DTM_APOLLO16.TIF LUNARSIM_LOLA_SPLIT=train \
    python scripts/build_tile_pool.py --n 96 --seed0 0 --grid 384 --out out/a16/pool_train.pkl
"""
import argparse, os, pickle, sys, time
from dataclasses import replace
from multiprocessing import get_context

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=96)
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--grid", type=int, default=384)
p.add_argument("--stage", default="orbit_descent")
p.add_argument("--max-rocks", type=int, default=3000)
p.add_argument("--workers", type=int, default=12)
p.add_argument("--out", required=True)
a = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(seed):
    import numpy as np
    from lunarsim.core.terrain.generate import generate_tile
    from lunarsim.core.terrain.rocks import RockField
    from lunarsim.rl.curriculum import STAGES_BY_NAME, terrain_config

    cfg = terrain_config(STAGES_BY_NAME[a.stage], seed, a.grid)
    t = generate_tile(cfg)
    r = t.rocks
    if len(r.x_m) > a.max_rocks:
        k = np.argsort(-r.diameter_m)[: a.max_rocks]
        t = replace(t, rocks=RockField(r.x_m[k], r.y_m[k], r.diameter_m[k]))
    t = replace(t, height=np.asarray(t.height, dtype=np.float32))
    return seed, t, (cfg.site_lat_deg, cfg.site_lon_deg)


if __name__ == "__main__":
    t0 = time.time()
    seeds = list(range(a.seed0, a.seed0 + a.n))
    with get_context("spawn").Pool(a.workers) as pool:
        out = pool.map(build, seeds)
    tiles = [t for _, t, _ in out]
    sites = [s for _, _, s in out]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(tiles, fh)
    import numpy as np
    s = np.array(sites)
    print(f"{len(tiles)} tiles in {time.time() - t0:.0f}s -> {a.out}  lon {s[:, 1].min():.3f}..{s[:, 1].max():.3f} "
          f"lat {s[:, 0].min():.3f}..{s[:, 0].max():.3f}", flush=True)
