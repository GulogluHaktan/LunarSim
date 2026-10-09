"""The shaping reward must be spread across altitude the way the task's difficulty is.

Measured over 480 orbit_descent demo episodes, the absolute-error field put 82.5% of the
shaping reward in the 50-200 m cruise and 1.5% in the decisive 0-5 m band -- which is
14.8% of the steps and 94% of the failures. v86 then confirmed the consequence from the
other side: PPO improved the proxy return with a 95% interval excluding zero and left the
landing rate unresolved, because the whole gain came from the cruise.

These tests pin the three properties the fix has to have, each of which could silently
fail on its own: the default must be the old behaviour exactly, the exponent must move the
band distribution to the measured target, and the constant must leave the MEAN penalty
alone so the shaping-to-terminal balance is not quietly changed at the same time.
"""
import numpy as np
import pytest

from lunarsim.rl.reward import (RewardWeights, _velocity_field_penalty,
                                target_velocity)

TUNED = dict(vfield_norm_p=0.75, vfield_norm_c=3.8171)


def _err_and_alt(n=4000, seed=0):
    """Field errors sampled across the release envelope, with their altitudes."""
    rng = np.random.default_rng(seed)
    alt = rng.uniform(0.0, 200.0, n)
    w = RewardWeights()
    out = []
    for a in alt:
        tl, _, tz, _ = target_velocity(w, float(a))
        # a plausible tracking error: a fraction of the local target, both channels
        lat = tl * rng.uniform(0.6, 1.6)
        vz = tz * rng.uniform(0.6, 1.6)
        out.append((float(a), float(lat), 0.0, float(vz)))
    return out


def test_default_is_the_absolute_error_form_bit_for_bit():
    w0, wd = RewardWeights(), RewardWeights(vfield_norm_p=0.0, vfield_norm_c=1.0)
    for a, lat, _, vz in _err_and_alt(400):
        assert (_velocity_field_penalty(w0, a, 0.0, 0.0, lat, 0.0, vz)
                == _velocity_field_penalty(wd, a, 0.0, 0.0, lat, 0.0, vz))


def _demo_states(limit=20000):
    """Real demo states, because vfield_norm_c is a property of the REAL error
    distribution and nothing else can check it.

    The first version of these two tests used the synthetic fixture above, whose errors
    are proportional to the local target by construction and so have a different
    altitude profile -- it reported the mean penalty moving 27% and failed a constant
    that is correct on the data it was measured from. A constant fitted to real data has
    to be tested against real data.
    """
    import glob
    from lunarsim.rl.obs_norm import OBS_SCALE
    files = sorted(glob.glob("out/zemzev_orbit_demos_s*.npz"))
    if not files:
        pytest.skip("no demo files on disk; vfield_norm_c cannot be checked without them")
    alt, lat, vz = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        o = d["obs"].astype(np.float32)
        alt.append(o[:, 2] * OBS_SCALE[2])
        lat.append(np.hypot(o[:, 3] * OBS_SCALE[3], o[:, 4] * OBS_SCALE[4]))
        vz.append(o[:, 5] * OBS_SCALE[5])
    alt, lat, vz = (np.concatenate(x) for x in (alt, lat, vz))
    if len(alt) > limit:                      # a stride, so every altitude stays represented
        k = len(alt) // limit
        alt, lat, vz = alt[::k], lat[::k], vz[::k]
    return alt, lat, vz


def _penalties(w, alt, lat, vz):
    return np.array([_velocity_field_penalty(w, float(a), 0.0, 0.0, float(l), 0.0, float(z))
                     for a, l, z in zip(alt, lat, vz)])


def test_the_decisive_band_gets_its_share_of_the_shaping_reward():
    alt, lat, vz = _demo_states()
    abs_share = (lambda p: p[alt < 5.0].sum() / p.sum())(
        _penalties(RewardWeights(), alt, lat, vz))
    new_share = (lambda p: p[alt < 5.0].sum() / p.sum())(
        _penalties(RewardWeights(**TUNED), alt, lat, vz))
    step_share = float((alt < 5.0).mean())
    assert abs_share < 0.05, f"the absolute form already gives 0-5 m {abs_share:.1%}"
    # its share of the reward should land near its share of the STEPS, which is the
    # target the exponent was chosen against -- not far past it, since the lateral speed
    # that causes the failures is set up high up.
    assert 0.6 * step_share <= new_share <= 1.6 * step_share, (
        f"0-5 m share {new_share:.1%} against a {step_share:.1%} step share")


def test_the_mean_penalty_is_preserved_so_only_the_distribution_moves():
    """Otherwise the change would also move the shaping-to-terminal balance, and a run
    could not be attributed to one or the other."""
    alt, lat, vz = _demo_states()
    a_mean = _penalties(RewardWeights(), alt, lat, vz).mean()
    n_mean = _penalties(RewardWeights(**TUNED), alt, lat, vz).mean()
    assert n_mean == pytest.approx(a_mean, rel=0.10), (
        f"mean penalty moved {a_mean:.3f} -> {n_mean:.3f}; vfield_norm_c needs remeasuring")


def test_normalisation_raises_the_penalty_near_the_ground_and_lowers_it_high_up():
    w_abs, w_new = RewardWeights(), RewardWeights(**TUNED)
    # the SAME absolute error, charged at two altitudes
    err = 0.9
    for alt, direction in ((1.0, "up"), (200.0, "down")):
        tl, _, tz, _ = target_velocity(w_abs, alt)
        a = _velocity_field_penalty(w_abs, alt, 0, 0, tl + err, 0, tz)
        b = _velocity_field_penalty(w_new, alt, 0, 0, tl + err, 0, tz)
        if direction == "up":
            assert b > a * 2.0, f"at {alt} m the penalty only went {a:.1f} -> {b:.1f}"
        else:
            assert b < a, f"at {alt} m the penalty went {a:.1f} -> {b:.1f}, expected lower"


def test_the_cap_still_binds():
    w = RewardWeights(**TUNED)
    huge = _velocity_field_penalty(w, 0.5, 0.0, 0.0, 500.0, 0.0, -500.0)
    assert huge == pytest.approx(w.vfield_cap)
