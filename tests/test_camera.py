import numpy as np
import pytest

from lunarsim.core.lighting.camera import CameraNoiseModel, apply_noise


def test_brighter_signal_gives_higher_mean_dn():
    rng = np.random.default_rng(0)
    dim = np.full((32, 32), 100.0)
    bright = np.full((32, 32), 10_000.0)
    model = CameraNoiseModel()
    dn_dim = model.apply(dim, exposure_s=0.01, rng=rng)
    dn_bright = model.apply(bright, exposure_s=0.01, rng=rng)
    assert dn_bright.mean() > dn_dim.mean()


def test_saturation_clips_at_max_dn():
    rng = np.random.default_rng(0)
    model = CameraNoiseModel(bit_depth=8)
    huge = np.full((16, 16), 1e9)
    dn = model.apply(huge, exposure_s=1.0, rng=rng)
    assert dn.max() == 2**8 - 1


def test_noise_none_mode_is_deterministic():
    img = np.full((16, 16), 500.0)
    dn1 = apply_noise(img, exposure_s=0.02, rng=np.random.default_rng(1), mode="none")
    dn2 = apply_noise(img, exposure_s=0.02, rng=np.random.default_rng(2), mode="none")
    np.testing.assert_array_equal(dn1, dn2)


def test_shot_read_mode_is_stochastic():
    img = np.full((16, 16), 500.0)
    dn1 = apply_noise(img, exposure_s=0.02, rng=np.random.default_rng(1), mode="shot_read")
    dn2 = apply_noise(img, exposure_s=0.02, rng=np.random.default_rng(2), mode="shot_read")
    assert not np.array_equal(dn1, dn2)


def test_zero_signal_gives_zero_or_near_zero_dn():
    rng = np.random.default_rng(0)
    model = CameraNoiseModel(read_noise_electrons=0.0)
    dn = model.apply(np.zeros((8, 8)), exposure_s=0.01, rng=rng)
    assert dn.max() == 0
