"""Does our LayerNorm actually bound Q on out-of-distribution actions?

RLPD (Ball, Smith, Kostrikov, Levine -- arXiv:2302.02948) has a section titled
"Layer Normalization Mitigates Catastrophic Overestimation", and the mechanism is
specific: with LayerNorm before the final linear layer the critic's output is bounded
by the norm of that layer's weights, so Q cannot run away however far the queried
action sits outside the data.

That is exactly the failure this project measured. With --bc-anchor 1e-3 the policy
died in one 10k window while critic_loss stayed at 0.2-2.3 -- the critic fit its buffer
perfectly and was confidently wrong about the actions the actor had just begun
proposing, and the actor's job is to find precisely those.

`--layer-norm` already exists here. It was added for a plasticity-loss experiment and
never used for this, so before any run claims "layer norm did not help" the property
has to be shown to hold in OUR implementation. Four interventions in this project were
invalidated by exactly that omission.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from stable_baselines3 import SAC


class _Sp(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (16,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    def reset(self, **kw):
        return np.zeros(16, np.float32), {}

    def step(self, a):
        return np.zeros(16, np.float32), 0.0, False, False, {}


def _q_spread(model, scale):
    """Worst-case |Q| over inputs pushed `scale` times outside the normal range."""
    rng = np.random.default_rng(0)
    obs = th.as_tensor((rng.normal(size=(512, 16)) * scale).astype(np.float32))
    # actions are nominally in [-1, 1]; the actor can only ever emit that range, so
    # pushing the OBSERVATION out of range is the honest stress here, plus actions
    # driven hard to the bounds.
    act = th.as_tensor(np.clip(rng.normal(size=(512, 4)) * 3.0, -1, 1).astype(np.float32))
    with th.no_grad():
        q = th.cat(model.critic(obs, act), dim=1)
    return float(q.abs().max())


def _build(layer_norm: bool):
    th.manual_seed(0)
    m = SAC("MlpPolicy", _Sp(), device="cpu", verbose=0, seed=0)
    if layer_norm:
        from lunarsim.rl.plasticity import add_layer_norm_to_sac
        n = add_layer_norm_to_sac(m)
        assert n > 0, "add_layer_norm_to_sac rebuilt nothing -- it is a no-op here"
    return m


def test_layer_norm_is_actually_installed_in_the_critic():
    """The flag must change the critic, not just the actor or nothing at all."""
    import torch.nn as nn
    plain = _build(False)
    normed = _build(True)
    def count(mod):
        return sum(1 for m in mod.modules() if isinstance(m, nn.LayerNorm))
    assert count(plain.critic) == 0
    assert count(normed.critic) > 0, (
        "no LayerNorm in the critic after add_layer_norm_to_sac -- the runs that "
        "used --layer-norm were not testing the published mechanism")
    assert count(normed.critic_target) > 0, "target critic left un-normalised"


def test_layer_norm_bounds_q_as_the_input_leaves_the_data_range():
    """The RLPD property: |Q| must stop growing as inputs go further OOD.

    Without normalisation a ReLU MLP is positively homogeneous in its input, so |Q|
    grows roughly linearly with input scale and the actor can chase that growth. With
    LayerNorm the pre-output activations are rescaled to unit norm, so the output is
    bounded by the final layer's weights.
    """
    plain, normed = _build(False), _build(True)
    scales = [1.0, 10.0, 100.0, 1000.0]
    p = [_q_spread(plain, s) for s in scales]
    n = [_q_spread(normed, s) for s in scales]

    plain_growth = p[-1] / max(p[0], 1e-9)
    normed_growth = n[-1] / max(n[0], 1e-9)
    assert plain_growth > 10.0, (
        f"the un-normalised critic did not blow up ({plain_growth:.1f}x over a 1000x "
        f"input range), so this test cannot distinguish the two: {p}")
    assert normed_growth < 3.0, (
        f"LayerNorm did NOT bound Q: it grew {normed_growth:.1f}x over a 1000x input "
        f"range ({n}). The published mechanism is not holding in this implementation.")
    assert normed_growth < plain_growth / 10.0, (p, n)


def test_layer_norm_leaves_q_usable_in_distribution():
    """Bounding must not flatten Q where it has to discriminate.

    A critic that returns a constant is bounded and useless -- and this project has
    already chased one suspected constant critic (it turned out to have Q std 3.26 and
    |dQ/da| 1.41, so the gradient existed and pointed the wrong way). Guard the other
    failure mode explicitly.
    """
    normed = _build(True)
    rng = np.random.default_rng(1)
    obs = th.as_tensor(rng.normal(size=(512, 16)).astype(np.float32))
    act = th.as_tensor(rng.uniform(-1, 1, size=(512, 4)).astype(np.float32))
    with th.no_grad():
        q = normed.critic(obs, act)[0]
    assert float(q.std()) > 1e-4, f"normalised critic is nearly constant (std {float(q.std()):.2e})"
    a = act.clone().requires_grad_(True)
    normed.critic(obs, a)[0].sum().backward()
    assert float(a.grad.abs().mean()) > 1e-5, "no gradient through the action"
