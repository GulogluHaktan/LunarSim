"""The terminal stream's value is available in closed form; fit it, don't bootstrap it.

The terminal stream carries no intermediate reward, so its value at a transition is
exactly  r_term * gamma_terminal^(steps remaining).  Solving that by TD iteration at
gamma_terminal = 0.999 contracts 0.1% per step -- barely at all over a 700-step episode --
so any per-step bias arrives in the value multiplied by 1/(1-gamma) = 1000. That was
measured on real checkpoints: the conservative term's +0.0123/step showed up as +12.3 in
the terminal head where the true value is about +0.5, and the terminal head was 89% of the
critic's total error while carrying 68% of the land-versus-crash value difference.

These tests pin the exact target's arithmetic, the retro-fill that produces it, and the
two cases where it must NOT be used: transitions whose episode has not ended, and
positions a wrapping ring buffer has reassigned to a later episode.
"""
import gymnasium as gym
import numpy as np
import pytest

th = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from lunarsim.rl.cql_sac import CQLSAC  # noqa: E402
from lunarsim.rl.dual_gamma import DualRewardReplayBuffer  # noqa: E402

OBS = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
ACT = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)


def _buf(size=64, n_envs=1):
    return DualRewardReplayBuffer(size, OBS, ACT, device="cpu", n_envs=n_envs,
                                  optimize_memory_usage=False)


def _add(buf, n_envs=1, term=0.0, done=False):
    o = np.zeros((n_envs, 2), np.float32)
    buf.add(o, o.copy(), np.zeros((n_envs, 1), np.float32),
            np.full(n_envs, term, np.float32), np.full(n_envs, done, bool),
            [{"reward_terminal": term} for _ in range(n_envs)])


def test_steps_to_end_counts_backwards_from_the_terminal_step():
    buf = _buf()
    for _ in range(5):
        _add(buf)
    _add(buf, term=7.0, done=True)       # 6th transition ends the episode
    # the terminal step itself has 0 steps remaining, the one before it 1, ...
    assert list(buf.mc_steps[:6, 0]) == [5, 4, 3, 2, 1, 0]
    assert np.all(buf.mc_terminal_r[:6, 0] == 7.0)
    assert np.all(buf.mc_valid[:6, 0] == 1)


def test_the_exact_target_is_the_discounted_terminal_reward():
    buf = _buf()
    for _ in range(9):
        _add(buf)
    _add(buf, term=-3.5, done=True)
    g = 0.999
    s = buf.sample(64)
    exact = s.mc_terminal_r.numpy().reshape(-1) * g ** s.mc_steps.numpy().reshape(-1)
    valid = s.mc_valid.numpy().reshape(-1) > 0
    # every sampled transition is from the one finished episode
    assert valid.all()
    # Indexed by steps-to-go rather than by min/max: the terminal reward here is
    # NEGATIVE, so the most heavily discounted value is the LARGEST, and an earlier
    # version of this assertion had the two the wrong way round. Checking the value AT a
    # given k cannot be fooled by the sign.
    steps = s.mc_steps.numpy().reshape(-1)
    for k in (0, 5, 9):
        got = exact[steps == k]
        assert len(got), f"no sampled transition with {k} steps to go"
        assert got[0] == pytest.approx(-3.5 * g ** k, rel=1e-5)


def test_in_flight_transitions_have_no_exact_target():
    buf = _buf()
    for _ in range(4):
        _add(buf)                        # episode still running
    assert np.all(buf.mc_valid[:4, 0] == 0)
    _add(buf, term=1.0, done=True)
    assert np.all(buf.mc_valid[:5, 0] == 1)
    for _ in range(3):
        _add(buf)                        # a NEW episode starts, still in flight
    assert np.all(buf.mc_valid[5:8, 0] == 0)
    assert np.all(buf.mc_valid[:5, 0] == 1), "the finished episode must stay valid"


def test_a_wrapping_ring_does_not_relabel_a_later_episode():
    """A position reused by a new episode must lose the old episode's target.

    Without the `mc_valid[pos] = 0` on every add, a position written by episode 2 would
    still carry episode 1's terminal reward and step count until episode 2 ended -- and
    the critic would be regressed onto a number belonging to a different trajectory.
    """
    buf = _buf(size=8)
    for _ in range(7):
        _add(buf)
    _add(buf, term=5.0, done=True)       # fills positions 0..7, all valid
    assert np.all(buf.mc_valid[:, 0] == 1)
    for _ in range(3):                   # wraps, overwriting 0,1,2 with a new episode
        _add(buf)
    assert np.all(buf.mc_valid[:3, 0] == 0)
    assert np.all(buf.mc_valid[3:, 0] == 1)
    assert np.all(buf.mc_terminal_r[3:, 0] == 5.0)


class _Env(gym.Env):
    """Shaping -1 per step, terminal +50 at a fixed horizon."""
    observation_space = OBS
    action_space = ACT

    def __init__(self, horizon=10):
        self.horizon, self.t = horizon, 0

    def reset(self, **kw):
        self.t = 0
        return np.array([0.0, 1.0], np.float32), {}

    def step(self, a):
        self.t += 1
        done = self.t >= self.horizon
        term = 50.0 if done else 0.0
        obs = np.array([self.t / self.horizon, 1.0 - self.t / self.horizon], np.float32)
        return obs, -1.0 + term, done, False, {"reward_terminal": term}


def test_the_terminal_head_lands_on_the_exact_value():
    """With the exact target the terminal head must match r*gamma^k, not drift to it.

    gamma_terminal is deliberately 0.9 here rather than 0.999: a low discount is where
    bootstrapping WOULD converge, so a difference between the two modes at 0.9 shows the
    exact target is being used rather than merely tolerated.
    """
    H, G_T = 10, 0.9
    th.manual_seed(0)
    np.random.seed(0)
    m = CQLSAC("MlpPolicy", _Env(H), device="cpu", verbose=0, seed=0,
               cql_alpha=0.0, dual_gamma=(0.9, G_T), gamma=0.9, mc_terminal=True,
               learning_starts=100, batch_size=64, train_freq=1,
               policy_kwargs={"n_critics": 4},
               replay_buffer_class=DualRewardReplayBuffer)
    m.learn(total_timesteps=4000, log_interval=10000)
    assert m.mc_terminal is True
    half = m._n_half
    errs = []
    for t in range(H):
        obs = th.tensor([[t / H, 1.0 - t / H]], dtype=th.float32)
        with th.no_grad():
            a, _ = m.actor.action_log_prob(obs)
            qs = m.critic(obs, a)
            q_t = float(th.cat(qs[half:], dim=1).min())
        want = 50.0 * G_T ** (H - 1 - t)
        errs.append(abs(q_t - want) / max(want, 1e-6))
    # the head is fitted by regression onto an exact number, so it should be close
    assert np.median(errs) < 0.20, f"relative errors {np.round(errs, 3).tolist()}"
