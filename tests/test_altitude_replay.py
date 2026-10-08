"""The altitude-stratified buffer must change WHICH transitions are sampled and nothing
else about them.

Two things have to hold at once, and the second is the one that bites. Stratifying by
altitude means choosing (position, env) PAIRS, while SB3's `ReplayBuffer._get_samples`
draws the env half itself -- so a stratified sampler that only chooses positions would
re-randomise the env index and hand the critic a batch whose terminal rewards belong to
different transitions than its observations. That failure is invisible in aggregate
statistics, so the fixture below encodes each transition's identity INTO its reward:
reward_terminal = altitude * 1000 + env_index. Any mispairing then shows up as an exact
mismatch rather than as a distribution that still looks plausible.
"""
import re

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from gymnasium import spaces  # noqa: E402

from lunarsim.rl.altitude_replay import AltitudeStratifiedReplayBuffer  # noqa: E402
from lunarsim.rl.dual_gamma import DualRewardReplayBuffer  # noqa: E402
from lunarsim.rl.obs_norm import OBS_SCALE  # noqa: E402

N_ENVS = 2
OBS_DIM = 20
ALT_SCALE = float(OBS_SCALE[2])
DECISIVE_M = 5.0
# The measured orbit_descent imbalance: most of the descent is cruise.
LOW_SHARE = 0.14
N_TRANS = 2000


def _fill(cls, **kw):
    buf = cls(N_TRANS, spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
              spaces.Box(-1.0, 1.0, (4,), np.float32), device="cpu",
              n_envs=N_ENVS, optimize_memory_usage=False, **kw)
    rng = np.random.default_rng(0)
    for t in range(N_TRANS):
        obs = np.zeros((N_ENVS, OBS_DIM), dtype=np.float32)
        alts = np.where(rng.random(N_ENVS) < LOW_SHARE,
                        rng.uniform(0.0, DECISIVE_M, N_ENVS),
                        rng.uniform(80.0, 150.0, N_ENVS))
        obs[:, 2] = alts / ALT_SCALE
        # Identity baked into the terminal reward. The altitude is a continuous draw, so
        # it is unique across every (position, env) pair in the buffer -- which makes
        # `term == alt * 1000` an identity check that catches a mispaired ENV index just
        # as surely as a mispaired position. An earlier version of this fixture wrote
        # `alt * 1000 + env_index` and tried to recover the env index with `% 1000`;
        # that is not injective (a 0.197 m altitude lands on 197.0 too), and the defect
        # was in the test, not the buffer.
        term = alts * 1000.0
        buf.add(obs, obs.copy(), np.zeros((N_ENVS, 4), np.float32),
                term.astype(np.float32), np.zeros(N_ENVS, bool),
                [{"reward_terminal": float(term[i])} for i in range(N_ENVS)])
    return buf


def _low_fraction(samples):
    alt = samples.observations[:, 2].numpy() * ALT_SCALE
    return float(np.mean(alt <= DECISIVE_M))


def test_uniform_buffer_reproduces_the_imbalance():
    """The baseline this exists to fix: ~14% of a uniform batch is decisive."""
    buf = _fill(DualRewardReplayBuffer)
    frac = np.mean([_low_fraction(buf.sample(256)) for _ in range(20)])
    assert frac == pytest.approx(LOW_SHARE, abs=0.04)


@pytest.mark.parametrize("want", [0.5, 0.35])
def test_stratified_buffer_hits_its_quota(want):
    buf = _fill(AltitudeStratifiedReplayBuffer,
                decisive_alt_m=DECISIVE_M, decisive_frac=want)
    frac = np.mean([_low_fraction(buf.sample(256)) for _ in range(20)])
    assert frac == pytest.approx(want, abs=0.02)


def test_pairing_survives_stratification():
    """observation, reward and rewards_terminal must all describe the SAME transition.

    This is the test that catches re-randomised env indices: the reward encodes the
    altitude the observation carries, so a mismatch is an exact arithmetic failure.
    """
    buf = _fill(AltitudeStratifiedReplayBuffer,
                decisive_alt_m=DECISIVE_M, decisive_frac=0.5)
    s = buf.sample(512)
    alt = s.observations[:, 2].numpy() * ALT_SCALE
    term = s.rewards_terminal.numpy().reshape(-1)
    np.testing.assert_allclose(term / 1000.0, alt, rtol=1e-4, atol=1e-3)
    # the total reward is the same quantity in this fixture, so it must match too
    np.testing.assert_allclose(s.rewards.numpy().reshape(-1), term, rtol=1e-5)


def test_batch_keeps_its_size_when_a_band_is_empty(capsys):
    """A run that has not reached the ground yet must still get full batches."""
    buf = AltitudeStratifiedReplayBuffer(
        64, spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        spaces.Box(-1.0, 1.0, (4,), np.float32), device="cpu", n_envs=1,
        optimize_memory_usage=False, decisive_alt_m=DECISIVE_M, decisive_frac=0.5)
    for _ in range(64):   # every transition high up: the decisive band is empty
        obs = np.zeros((1, OBS_DIM), dtype=np.float32)
        obs[:, 2] = 120.0 / ALT_SCALE
        buf.add(obs, obs.copy(), np.zeros((1, 4), np.float32),
                np.zeros(1, np.float32), np.zeros(1, bool), [{"reward_terminal": 0.0}])
    s = buf.sample(32)
    assert s.observations.shape[0] == 32
    # Every sample comes from the high band, because that is all there is -- and the
    # buffer must SAY the quota went unmet. Without that line a run that never reaches
    # the ground is indistinguishable from one where stratification worked.
    out = capsys.readouterr().out
    assert "quota UNMET" in out
    alt = s.observations[:, 2].numpy() * ALT_SCALE
    assert np.all(alt > DECISIVE_M)


def test_optimize_memory_usage_is_refused():
    with pytest.raises(ValueError, match="optimize_memory_usage"):
        AltitudeStratifiedReplayBuffer(
            64, spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
            spaces.Box(-1.0, 1.0, (4,), np.float32), device="cpu", n_envs=1,
            optimize_memory_usage=True)


def test_realised_fraction_is_reported_periodically(capsys):
    """The quota-unmet warning fires once, so it cannot report the steady state.

    An early shortfall is expected -- before anything reaches the ground the buffer holds
    only the first seconds of each episode -- so a single warning line says nothing about
    the batches the critic actually trained on. The periodic line is what does.
    """
    buf = _fill(AltitudeStratifiedReplayBuffer, decisive_alt_m=DECISIVE_M,
                decisive_frac=0.5, report_every=5)
    capsys.readouterr()
    for _ in range(5):
        buf.sample(128)
    out = capsys.readouterr().out
    assert "last 5 batches" in out, out
    m = re.search(r"last 5 batches: ([\d.]+)% of sampled", out)
    assert m, out
    assert abs(float(m.group(1)) - 50.0) < 5.0, out


def test_reporting_can_be_switched_off():
    buf = _fill(AltitudeStratifiedReplayBuffer, decisive_alt_m=DECISIVE_M,
                decisive_frac=0.5, report_every=0)
    for _ in range(3):
        buf.sample(64)   # must not raise on the modulo with 0
