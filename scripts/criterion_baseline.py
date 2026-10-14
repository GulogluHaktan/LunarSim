"""Landing rate of the controller and MPC under a chosen landing criterion (analytic backend, orbit_descent).

  LUNARSIM_LOLA_IMG=<tif> LUNARSIM_TILE_POOL=<pool.pkl> LUNARSIM_LANDING_CRITERION=plane|spread \
      python scripts/criterion_baseline.py --mpc out/a16/mpc_v3.npy --n 48
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=48); p.add_argument("--seed0", type=int, default=150000)
p.add_argument("--stage", default="orbit_descent"); p.add_argument("--mpc", default=None)
p.add_argument("--K", type=int, default=512); p.add_argument("--M", type=int, default=20)
a = p.parse_args()
from lunarsim.core.terrain.criterion import criterion_key
from lunarsim.control.zemzev_controller import ZemZevController
import mpc_analytic_eval as M  # scripts/ is on sys.path when run as a script

seeds = [a.seed0 + i for i in range(a.n)]
print("criterion", criterion_key(), "stage", a.stage, flush=True)
t0 = time.time(); land = 0
for s in seeds:
    env, st = M.make_env(a.stage, s); env.reset(seed=int(s)); c = ZemZevController(st.params)
    while True:
        _, r, te, tr, info = env.step(c.act(env))
        if te or tr: break
    land += bool(info["landed_safely"])
print(f"controller  {land}/{a.n} = {100*land/a.n:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
if a.mpc:
    from lunarsim.evo.mpc import MPCConfig
    t0 = time.time(); lp = np.load(a.mpc)
    res = []
    for k in range(0, a.n, 12):
        r, _, _ = M.run(a.stage, seeds[k:k + 12], lp, MPCConfig(K=a.K, M=a.M)); res += r
    print(f"mpc         {res.count('landed')}/{a.n} = {100*res.count('landed')/a.n:.1f}%  "
          f"{ {k: res.count(k) for k in sorted(set(res))} }  ({time.time()-t0:.0f}s)", flush=True)
