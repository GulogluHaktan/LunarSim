import numpy as np
import pytest

from lunarsim.core.lighting.regolith_texture import bake_regolith_normal_map


def test_normal_map_shape_and_dtype():
    nm = bake_regolith_normal_map(resolution=64, seed=0)
    assert nm.shape == (64, 64, 3)
    assert nm.dtype == np.uint8


def test_normal_map_mostly_points_up():
    """Flat-ish micro-bump: z-component (blue channel) should dominate, i.e.
    average close to 255 (normal.z ~ 1) rather than 128 (flat/no bias)."""
    nm = bake_regolith_normal_map(resolution=128, amplitude_m=0.01, seed=1)
    assert nm[..., 2].mean() > 200


def test_normal_map_is_deterministic_per_seed():
    a = bake_regolith_normal_map(resolution=32, seed=5)
    b = bake_regolith_normal_map(resolution=32, seed=5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = bake_regolith_normal_map(resolution=32, seed=1)
    b = bake_regolith_normal_map(resolution=32, seed=2)
    assert not np.array_equal(a, b)


def test_higher_amplitude_gives_more_normal_variation():
    low = bake_regolith_normal_map(resolution=128, amplitude_m=0.005, seed=3)
    high = bake_regolith_normal_map(resolution=128, amplitude_m=0.05, seed=3)
    # more bump amplitude -> normals tilt further from straight up -> lower mean z
    assert high[..., 2].mean() < low[..., 2].mean()
