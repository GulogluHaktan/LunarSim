"""Does a hazard term in the MPC cost move touchdowns onto usable ground? (analytic backend, train pool)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.mpc_analytic_eval import run
from lunarsim.evo.mpc import MPCConfig
base = np.load(os.path.expanduser("~/lunarsim_artifacts/evo/mpc_params_v2.npy"))
cfg = MPCConfig(K=512, M=20, hold=12, exec_steps=4, roll_sub=2, margin=0.7)
seeds = [300000 + i for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 12)]
for lw in [float(v) for v in (sys.argv[2:] or ["-3", "0.0", "0.7", "1.4"])]:
    lp = np.concatenate([base, [lw]])
    res, dt, _ = run("orbit_descent", seeds, lp, cfg)
    d = np.array(run.last_td_dist)
    print(f"log10 w_haz={lw:+.1f}: landed {res.count('landed')}/{len(res)}  {dict((k,res.count(k)) for k in sorted(set(res)))}  "
          f"td dist-to-safe median {np.median(d):.0f} m  mean {d.mean():.0f} m  ({dt:.0f}s)", flush=True)
