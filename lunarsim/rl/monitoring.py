"""Track the PROXY and the TRUE objective side by side during training.

The reward this project optimises is a dense shaped signal. The thing we
actually want is binary and comparatively rare: `landed_safely`. Those are not
the same objective, and the literature on proxy rewards (Laidlaw et al. 2024)
describes the failure directly -- optimise the proxy hard enough and the true
objective starts to come apart, with the shaped return still climbing while the
success rate goes flat or falls.

This project could not have SEEN that happen. Until now nothing recorded either
curve during a run: the VecEnv is not wrapped in a Monitor, so SB3's log carries
only `time/` and `train/` sections with no `rollout/ep_rew_mean`, and the only
success number printed was a 16-episode sanity check AFTER the stage finished.
Every claim in handover.md about what happened mid-run was reconstructed from
snapshots evaluated afterwards.

So this callback records, per window of steps:
  - mean shaped episode return          <- the proxy
  - landed_safely rate                  <- the true objective
  - the breakdown of how the rest failed
  - the WORST-DECILE episode return, because a mean hides the tail and the tail
    is where a landing task actually lives

Divergence between the first two columns is the signal to watch. Both falling
together means training is failing; the proxy rising while the rate falls means
the shaping has started to mislead.
"""
from __future__ import annotations

import csv
import os

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - SB3 absent
    BaseCallback = object  # type: ignore


class ProxyVsTrueCallback(BaseCallback):  # type: ignore[misc]
    """Log proxy return and true success rate on the same cadence.

    Reads the per-step `dones`/`infos`/`rewards` that SB3 puts in `self.locals`,
    so it needs no cooperation from the environment beyond the `landed_safely`
    flag the envs already report.
    """

    def __init__(self, log_every: int = 10_000, csv_path: str | None = None,
                 verbose: int = 1):
        super().__init__(verbose)
        self.log_every = int(log_every)
        self.csv_path = csv_path
        self._running: dict[int, float] = {}
        self._returns: list[float] = []
        self._landed: list[bool] = []
        self._lost: list[bool] = []
        self._left: list[bool] = []
        self._next = None
        self._wrote_header = False

    def _on_training_start(self) -> None:
        self._next = self.model.num_timesteps + self.log_every
        if self.csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)) or ".",
                        exist_ok=True)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if rewards is None or dones is None:
            return True
        rewards = np.atleast_1d(rewards)
        dones = np.atleast_1d(dones)
        for i, r in enumerate(rewards):
            self._running[i] = self._running.get(i, 0.0) + float(r)
        for i, done in enumerate(dones):
            if not done:
                continue
            self._returns.append(self._running.pop(i, 0.0))
            info = infos[i] if infos is not None and i < len(infos) else {}
            self._landed.append(bool(info.get("landed_safely")))
            self._lost.append(bool(info.get("lost_control")))
            self._left.append(bool(info.get("left_tile")))

        if self._next is not None and self.model.num_timesteps >= self._next:
            self._flush()
            self._next = self.model.num_timesteps + self.log_every
        return True

    def _flush(self) -> None:
        n = len(self._returns)
        if n == 0:
            return
        rets = np.asarray(self._returns, dtype=float)
        landed = int(np.sum(self._landed))
        lost = int(np.sum(self._lost))
        left = int(np.sum(self._left))
        row = {
            "step": int(self.model.num_timesteps),
            "episodes": n,
            "proxy_mean_return": float(rets.mean()),
            # the mean hides the tail, and a landing task lives in the tail
            "proxy_worst_decile": float(np.percentile(rets, 10)),
            "true_landed_rate": landed / n,
            "landed": landed,
            "lost_control": lost,
            "left_tile": left,
            "other_failures": n - landed - lost - left,
        }
        if self.verbose:
            print(f"[proxy-vs-true] step {row['step']:>9}  n={n:>4}  "
                  f"proxy {row['proxy_mean_return']:>9.2f} "
                  f"(worst10% {row['proxy_worst_decile']:>9.2f})  "
                  f"landed {row['true_landed_rate']:>6.1%}  "
                  f"lost={lost} left={left} other={row['other_failures']}",
                  flush=True)
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if not self._wrote_header and fh.tell() == 0:
                    w.writeheader()
                self._wrote_header = True
                w.writerow(row)
        self._returns.clear()
        self._landed.clear()
        self._lost.clear()
        self._left.clear()

    def _on_training_end(self) -> None:
        self._flush()
