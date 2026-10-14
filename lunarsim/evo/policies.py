"""Numpy policies with a flat parameter vector (no torch), for ES.

Observations arrive already scaled by `obs_norm.normalize_obs` (fixed constants),
so there is no running normalisation here.
"""
from __future__ import annotations

import numpy as np


class MLPPolicy:
    """tanh MLP, `sizes` e.g. (20, 32, 32, 4). hidden=() gives a linear policy.

    Output is clipped to [-1, 1]. `throttle_bias` is added to the pre-clip
    throttle output (declared prior: slightly below hover).
    """

    def __init__(self, obs_dim: int, act_dim: int = 4, hidden: tuple = (32, 32),
                 throttle_bias: float = -0.05):
        self.sizes = (obs_dim, *hidden, act_dim)
        self.throttle_bias = float(throttle_bias)
        self.shapes = []
        for i in range(len(self.sizes) - 1):
            self.shapes.append((self.sizes[i], self.sizes[i + 1]))
        self.n_params = sum(a * b + b for a, b in self.shapes)

    def init_params(self, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
        # small weights: the initial policy is ~ constant action = throttle_bias
        return rng.normal(0.0, scale, self.n_params)

    def act(self, theta: np.ndarray, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        k = 0
        for li, (a, b) in enumerate(self.shapes):
            W = theta[k:k + a * b].reshape(a, b); k += a * b
            bb = theta[k:k + b]; k += b
            x = x @ W + bb
            if li < len(self.shapes) - 1:
                x = np.tanh(x)
        x = x.copy()
        x[0] += self.throttle_bias
        return np.clip(x, -1.0, 1.0)

    def save(self, path: str, theta: np.ndarray, **meta):
        np.savez(path, theta=theta, sizes=np.array(self.sizes),
                 throttle_bias=self.throttle_bias, **meta)
