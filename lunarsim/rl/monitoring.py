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
        # DISCOUNTED return per episode, plus the (s0, a0) it started from, so the
        # critic's estimate can be compared against the realised value IN THE SAME
        # UNITS. Without this the only available comparison is Q against the
        # UNDISCOUNTED proxy return, which is not the same quantity at all: at
        # gamma=0.995 over ~450 steps a uniform reward stream discounts to roughly
        # 0.4x its undiscounted sum, so that comparison can manufacture an
        # "overestimation" of several units out of nothing. This project read
        # Q~-1 against a proxy of -11.20 and called it a 10-unit overestimate; the
        # honest version of that claim needs this number.
        self._disc_running: dict[int, float] = {}
        self._disc_t: dict[int, int] = {}
        self._ep_disc: list[float] = []
        self._ep_s0: list = []
        self._ep_a0: list = []
        self._pending_s0: dict[int, object] = {}
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
        gamma = float(getattr(self.model, "gamma", 0.99))
        # capture (s0, a0) for any env whose episode has just begun
        acts = self.locals.get("actions")
        last_obs = getattr(self.model, "_last_obs", None)
        if last_obs is not None and acts is not None:
            acts = np.atleast_2d(acts)
            for i in range(len(rewards)):
                if i not in self._pending_s0:
                    try:
                        self._pending_s0[i] = (np.array(last_obs[i], dtype=np.float32),
                                               np.array(acts[i], dtype=np.float32))
                    except (IndexError, TypeError):
                        pass
        for i, r in enumerate(rewards):
            self._running[i] = self._running.get(i, 0.0) + float(r)
            tt = self._disc_t.get(i, 0)
            self._disc_running[i] = self._disc_running.get(i, 0.0) + (gamma ** tt) * float(r)
            self._disc_t[i] = tt + 1
        for i, done in enumerate(dones):
            if not done:
                continue
            self._returns.append(self._running.pop(i, 0.0))
            self._ep_disc.append(self._disc_running.pop(i, 0.0))
            self._disc_t.pop(i, None)
            s0a0 = self._pending_s0.pop(i, None)
            if s0a0 is not None:
                self._ep_s0.append(s0a0[0])
                self._ep_a0.append(s0a0[1])
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
        # SAC's own train/ values, which SB3 only prints via _dump_logs -- and
        # that never fires here because the VecEnv is not Monitor-wrapped. They
        # live in logger.name_to_value until dumped, so read them directly.
        # Without this the critic is invisible at exactly the moment a
        # warm-started policy collapses.
        try:
            lv = self.model.logger.name_to_value
            row["critic_loss"] = float(lv.get("train/critic_loss", float("nan")))
            row["actor_loss"] = float(lv.get("train/actor_loss", float("nan")))
            row["ent_coef"] = float(lv.get("train/ent_coef", float("nan")))
        except Exception:
            row["critic_loss"] = row["actor_loss"] = row["ent_coef"] = float("nan")

        # THE OVERESTIMATION NUMBER. Q(s0, a0) from the current critic against the
        # realised discounted return of the episode that actually started there.
        # A positive gap is the critic valuing its own policy above what the policy
        # delivered, which is the mechanism every collapse in this project points
        # at -- and it is only meaningful because both sides are discounted.
        row["q_s0"] = row["mc_s0"] = row["q_minus_mc"] = float("nan")
        if self._ep_s0 and len(self._ep_s0) == len(self._ep_disc):
            try:
                import torch as th
                dev = self.model.device
                with th.no_grad():
                    s0 = th.as_tensor(np.asarray(self._ep_s0), device=dev)
                    a0 = th.as_tensor(np.asarray(self._ep_a0), device=dev)
                    if hasattr(self.model, "critic"):
                        # SAC/TD3: Q(s,a), the min over the ensemble -- the same scalar the
                        # actor ascends.
                        qs = th.cat(self.model.critic(s0, a0), dim=1).min(dim=1).values
                    else:
                        # PPO: V(s). There is no action argument, which is precisely why PPO
                        # is being tried -- the overestimation that broke every critic-based
                        # mechanism here comes from bootstrapping Q(s', a') onto the policy's
                        # own possibly out-of-distribution action, and V has no such argument.
                        # Measuring the same gap for PPO is how that claim gets tested rather
                        # than asserted: if V is also +40 above the realised return, the
                        # reasoning was wrong.
                        qs = self.model.policy.predict_values(s0).reshape(-1)
                    q = float(qs.mean().item())
                mc = float(np.mean(self._ep_disc))
                row["q_s0"], row["mc_s0"], row["q_minus_mc"] = q, mc, q - mc
            except Exception:
                pass

        if self.verbose:
            print(f"[proxy-vs-true] step {row['step']:>9}  n={n:>4}  "
                  f"proxy {row['proxy_mean_return']:>9.2f} "
                  f"(worst10% {row['proxy_worst_decile']:>9.2f})  "
                  f"landed {row['true_landed_rate']:>6.1%}  "
                  f"lost={lost} left={left} other={row['other_failures']}  "
                  f"Q0 {row['q_s0']:+.2f} vs MC {row['mc_s0']:+.2f} (gap {row['q_minus_mc']:+.2f})  "
                  f"Q~{-row['actor_loss']:.0f} critic={row['critic_loss']:.1f}",
                  flush=True)
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if not self._wrote_header and fh.tell() == 0:
                    w.writeheader()
                self._wrote_header = True
                w.writerow(row)
        self._returns.clear()
        self._ep_disc.clear()
        self._ep_s0.clear()
        self._ep_a0.clear()
        self._landed.clear()
        self._lost.clear()
        self._left.clear()

    def _on_training_end(self) -> None:
        self._flush()
