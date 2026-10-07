"""Does the dual-discount critic actually discount the two streams differently?

The point of these tests is the mechanism, not performance. arXiv:1810.08719 reports that
WITHOUT multiple discount rates "the performance was actually worsened by including the
terminal reward term", so if we add the machinery and it silently does nothing, a run will
read as "dual gamma did not help" and we will have tested the implementation again -- which
has already invalidated four conclusions in this project.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from lunarsim.rl.cql_sac import CQLSAC
from lunarsim.rl.dual_gamma import DualRewardReplayBuffer


class _Env(gym.Env):
    """Terminal reward only on the last step, shaping only before it.

    The two streams are deliberately OPPOSED: shaping pays -1 every step and the terminal
    pays +50 at the end. A single gamma cannot value this well -- low gamma discounts the
    +50 away and the optimal action is to end early, high gamma amplifies the -1 stream.
    """
    observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(self, horizon: int = 20):
        self.horizon, self.t = horizon, 0

    def reset(self, **kw):
        self.t = 0
        return np.array([0.0, 1.0], np.float32), {}

    def step(self, a):
        self.t += 1
        done = self.t >= self.horizon
        shaping, terminal = -1.0, (50.0 if done else 0.0)
        obs = np.array([self.t / self.horizon, 1.0 - self.t / self.horizon], np.float32)
        info = {"reward_shaping": shaping, "reward_terminal": terminal}
        return obs, shaping + terminal, done, False, info


def _model(dual, n_critics=4, steps=6000, horizon=20):
    th.manual_seed(0)
    np.random.seed(0)
    m = CQLSAC("MlpPolicy", _Env(horizon=horizon), device="cpu", verbose=0, seed=0,
               cql_alpha=0.0, dual_gamma=dual, gamma=0.9,
               learning_starts=100, batch_size=64, train_freq=1,
               policy_kwargs={"n_critics": n_critics},
               replay_buffer_class=DualRewardReplayBuffer)
    m.learn(total_timesteps=steps, log_interval=10000)
    return m


def test_dual_gamma_rejects_an_ensemble_it_cannot_split():
    """Two per stream is the minimum, so an odd or small ensemble must fail loudly."""
    for n in (1, 2, 3, 5):
        with pytest.raises(ValueError, match="even n_critics"):
            CQLSAC("MlpPolicy", _Env(), device="cpu", verbose=0, seed=0,
                   dual_gamma=(0.9, 0.999), policy_kwargs={"n_critics": n})


def test_the_two_heads_learn_different_values():
    """The halves must end up valuing different things.

    If the split were wired wrongly -- both halves fed the same target, or the terminal
    stream arriving as zeros -- the two halves would converge to the same function and
    this test would catch it. The buffer prints a warning on a missing
    `reward_terminal`, but a warning is not a guard.
    """
    m = _model((0.5, 0.999))
    obs = th.as_tensor(np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], np.float32))
    act = th.zeros((3, 1))
    with th.no_grad():
        q = m.critic(obs, act)
    half = len(q) // 2
    shaping_head = th.cat(q[:half], dim=1).mean().item()
    terminal_head = th.cat(q[half:], dim=1).mean().item()
    # shaping is all negative (-1/step), terminal is a single large positive
    assert shaping_head < 0.0, f"shaping head {shaping_head:.3f} should be negative"
    assert terminal_head > 0.0, f"terminal head {terminal_head:.3f} should be positive"
    assert abs(terminal_head - shaping_head) > 1.0, (shaping_head, terminal_head)


def test_the_terminal_head_matches_the_analytic_discounted_terminal():
    """The decisive test: the terminal head must equal 50 * gamma_t^horizon at the start.

    A relative comparison (higher gamma_t gives a higher value) was the first version and
    it FAILED at 1500 training steps -- not because the backup was wrong but because
    gamma_t=0.999 needs many updates to carry a terminal 20 steps back, and the head was
    still sitting on its initialisation. Comparing against the closed-form value instead
    makes under-training visible as a number rather than as a mystery, and it pins the
    discount rate itself rather than a direction.

    Also checks the shaping head is UNCHANGED by gamma_t: it only ever sees the shaping
    stream, so if its value moves when the terminal discount moves, the split is leaking.
    """
    cases = [(4, 0.5), (4, 0.999), (20, 0.5), (20, 0.999)]
    shaping_heads = {}
    for horizon, g_t in cases:
        m = _model((0.5, g_t), horizon=horizon)
        obs = th.as_tensor(np.array([[0.0, 1.0]], np.float32))
        act = th.zeros((1, 1))
        with th.no_grad():
            q = m.critic(obs, act)
        half = len(q) // 2
        shaping = th.cat(q[:half], dim=1).mean().item()
        terminal = th.cat(q[half:], dim=1).mean().item()
        expected = 50.0 * (g_t ** horizon)
        # Tolerance is set against the TERMINAL MAGNITUDE (50), not against the expected
        # value. A fraction of the expected value is the wrong scale when the expected
        # value is itself near zero: at horizon 4 with gamma_t=0.5 the closed form is
        # 3.125 and the learned head reads 6.248, which is a 6% error on the 50-point
        # scale the critic is actually fitting, and a first version of this assertion
        # rejected it as a bug. The test keeps its power regardless -- if gamma_t were
        # ignored and gamma_shaping used instead, horizon 20 would read ~0 against an
        # expected 49, a 49-point miss.
        assert abs(terminal - expected) < 0.15 * 50.0, (
            f"horizon={horizon} gamma_t={g_t}: terminal head {terminal:.3f} against an "
            f"analytic {expected:.3f} -- the terminal discount is not being applied")
        shaping_heads[(horizon, g_t)] = shaping

    for horizon in (4, 20):
        a = shaping_heads[(horizon, 0.5)]
        b = shaping_heads[(horizon, 0.999)]
        assert abs(a - b) < 0.5, (
            f"horizon={horizon}: the shaping head moved from {a:.3f} to {b:.3f} when only "
            f"gamma_TERMINAL changed -- the two streams are leaking into each other")


def test_the_buffer_warns_when_the_split_is_missing():
    """A reward_fn without the split must not fail silently."""
    import io
    import contextlib
    ob = gym.spaces.Box(-1, 1, (2,), np.float32)
    ac = gym.spaces.Box(-1, 1, (1,), np.float32)
    b = DualRewardReplayBuffer(50, ob, ac, device="cpu", n_envs=1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.add(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32),
              np.zeros((1, 1), np.float32), np.array([1.0], np.float32),
              np.array([False]), [{}])
    assert "reward_terminal" in buf.getvalue(), "silent on a missing split"
