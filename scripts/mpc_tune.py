"""Tune the MPC cost weights (log10 space) with OpenAI-ES on the analytic backend.
CRN: every candidate of a generation flies the same seeds. Fitness = landing rate
minus a small graded final-severity term. Selection block is separate from training."""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lunarsim.evo.es import OpenAIES
from lunarsim.evo.mpc import DEFAULT_LOG, MPCConfig, PARAM_NAMES
import multiprocessing as mp
from scripts.mpc_analytic_eval import run, eval_candidate

p = argparse.ArgumentParser()
p.add_argument("--stages", default="orbit_descent")
p.add_argument("--n", type=int, default=8)
p.add_argument("--pairs", type=int, default=8)
p.add_argument("--gens", type=int, default=12)
p.add_argument("--sigma", type=float, default=0.25)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--K", type=int, default=256)
p.add_argument("--M", type=int, default=16)
p.add_argument("--hold", type=int, default=15)
p.add_argument("--roll-sub", type=int, default=3)
p.add_argument("--exec-steps", type=int, default=6)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--margin", type=float, default=0.7)
p.add_argument("--init", default=None)
p.add_argument("--out", default="out/evo/mpc_tune")
a = p.parse_args()
def main():
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(K=a.K, M=a.M, hold=a.hold, roll_sub=a.roll_sub, exec_steps=a.exec_steps, margin=a.margin)
    stages = a.stages.split(",")
    theta0 = np.load(a.init) if a.init else DEFAULT_LOG.copy()
    es = OpenAIES(theta0, a.sigma, a.lr, a.pairs, seed=0)

    def fitness(lp, seeds):
        return eval_candidate((stages, lp, seeds, cfg))

    f = open(os.path.join(a.out, "curve.csv"), "w", newline=""); wr = csv.writer(f)
    wr.writerow(["gen", "mean_fit", "best_fit", "mean_land", "best_land", "sel_land", "sel_fit"] + list(PARAM_NAMES))
    best = (-1e9, None); t0 = time.time()
    pool = mp.get_context('spawn').Pool(a.workers)
    for g in range(a.gens):
        seeds = [200000 + 100 * g + i for i in range(a.n)]
        pop, eps = es.ask()
        out = pool.map(eval_candidate, [(stages, th, seeds, cfg) for th in pop])
        fit = np.array([o[0] for o in out]); land = np.array([o[1] for o in out])
        es.tell(eps, fit)
        sel = [160000 + i for i in range(a.n)]
        sf, sl = fitness(es.theta, sel)
        if sf > best[0]:
            best = (sf, es.theta.copy()); np.save(os.path.join(a.out, "best_params.npy"), best[1])
        np.save(os.path.join(a.out, "last_params.npy"), es.theta)
        wr.writerow([g, fit.mean(), fit.max(), land.mean(), land.max(), sl, sf] + [round(float(v), 3) for v in es.theta]); f.flush()
        print(f"g{g:2d} cand fit={fit.mean():+.3f} land={land.mean():.2f} best_land={land.max():.2f} | mean-theta SEL land={sl:.2f} fit={sf:+.3f} "
              f"t={time.time()-t0:.0f}s  {dict(zip(PARAM_NAMES, np.round(es.theta, 2)))}", flush=True)


if __name__ == '__main__':
    main()
