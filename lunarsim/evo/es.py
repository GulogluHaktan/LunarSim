"""OpenAI-ES: antithetic sampling, centered-rank shaping, Adam. numpy only."""
from __future__ import annotations

import numpy as np


def centered_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


class OpenAIES:
    def __init__(self, theta0: np.ndarray, sigma: float = 0.1, lr: float = 0.03,
                 pairs: int = 16, weight_decay: float = 0.0, seed: int = 0):
        self.theta = theta0.astype(np.float64).copy()
        self.sigma, self.lr, self.pairs, self.wd = sigma, lr, pairs, weight_decay
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros_like(self.theta); self.v = np.zeros_like(self.theta); self.t = 0

    def ask(self):
        eps = self.rng.standard_normal((self.pairs, self.theta.size))
        pop = np.concatenate([self.theta + self.sigma * eps, self.theta - self.sigma * eps])
        return pop, eps

    def tell(self, eps: np.ndarray, fitness: np.ndarray):
        n = self.pairs
        s = centered_ranks(np.asarray(fitness, dtype=np.float64))
        w = s[:n] - s[n:]                      # antithetic difference
        g = (w[:, None] * eps).sum(0) / (2 * n * self.sigma)
        g = g - self.wd * self.theta
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh, vh = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        self.theta = self.theta + self.lr * mh / (np.sqrt(vh) + 1e-8)  # ascent
