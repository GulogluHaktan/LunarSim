"""Is a training run IMPROVING, with a confidence interval rather than an eyeball?

Training-time landing rates in this project come with n = 12-31 episodes per reading, and
four separate conclusions were drawn from such numbers and later overturned by a 96-episode
deterministic matrix. That rule stands for RANKING checkpoints. But a run's whole log is a
few hundred episodes, and "is there a trend" is a different and answerable question -- as
long as it is answered with a slope and its standard error, not by looking at the last line.

What this exists to prevent is the opposite error to the one the matrix rule prevents.
Reading v85's proxy series by eye gave "no trend"; the regression gives a landing-rate
slope of +13.4 points per 1M steps with a 95% interval of [-12.7, +39.6]. That does not
show improvement -- and it does not show its absence either. The honest statement is that
440k steps cannot resolve the two, and the fix is more steps rather than a stronger claim:
the slope's standard error falls with both the span of the x-range and sqrt(n), so
finishing a 1.5M run shrinks it by roughly 3.4x.

Run:
    .venv/bin/python scripts/diag_training_trend.py out/train_ppo_v85.log
"""
from __future__ import annotations

import re
import sys

import numpy as np

if len(sys.argv) < 2:
    raise SystemExit(__doc__.strip().splitlines()[-1])

rows = []
for path in sys.argv[1:]:
    for ln in open(path):
        m = re.search(r"step\s+(\d+)\s+n=\s*(\d+)\s+proxy\s+(-?[\d.]+)"
                      r".*?landed\s+([\d.]+)%", ln)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)),
                         float(m.group(3)), float(m.group(4))))
if len(rows) < 4:
    raise SystemExit(f"only {len(rows)} readings found; need at least 4 for a slope")

s = np.array([r[0] for r in rows], float)
n = np.array([r[1] for r in rows], float)
pr = np.array([r[2] for r in rows], float)
ld = np.array([r[3] for r in rows], float) / 100.0

print(f"{len(rows)} readings, {int(n.sum())} episodes, steps "
      f"{int(s[0]):,} .. {int(s[-1]):,}\n")


def trend(y, name, w=None, pct=False):
    X = np.column_stack([s / 1e6, np.ones(len(s))])
    w = np.ones(len(s)) if w is None else w
    W = np.diag(w / w.mean())
    beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum() / (len(s) - 2))
    se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0]))
    k, unit = (100.0, " pts") if pct else (1.0, "")
    lo, hi = (beta[0] - 1.96 * se) * k, (beta[0] + 1.96 * se) * k
    verdict = ("IMPROVING" if lo > 0 else
               "DEGRADING" if hi < 0 else "UNRESOLVED (interval spans 0)")
    print(f"  {name:<16} slope {beta[0] * k:+8.3f}{unit}/1M   SE {se * k:7.3f}   "
          f"t={beta[0] / se:+5.2f}")
    print(f"  {'':16} 95% CI [{lo:+.3f}, {hi:+.3f}]{unit}   -> {verdict}")


trend(pr, "proxy return")
trend(ld, "landed rate", w=n, pct=True)
half = len(ld) // 2
print(f"\n  weighted mean landed {np.average(ld, weights=n):.1%}   "
      f"first half {np.average(ld[:half], weights=n[:half]):.1%}   "
      f"second half {np.average(ld[half:], weights=n[half:]):.1%}")
print("\n  A deterministic matrix is still the only thing that may RANK two checkpoints;")
print("  this only says whether one run is trending, and how wide the doubt is.")
