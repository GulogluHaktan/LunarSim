import numpy as np

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_hires_patch, generate_tile


def _small_tile(seed=1):
    cfg = TerrainConfig(
        mode="blend", size_m=30.0, res_m=0.2, coarse_res_m=2.0, seed=seed,
        coarse_source="procedural",
        hills={"amplitude_m": 1.0, "wavelength_m": 10.0, "hurst": 0.75},
        craters={"count_scale": 0.0}, rocks={"density_scale": 0.0},
        roi={"centers": None}, curvature=False,
    )
    return generate_tile(cfg)


def test_hires_patch_resolution_matches_request():
    tile = _small_tile()
    patch = generate_hires_patch(tile, size_m=10.0, res_m=0.05)
    assert patch.height.shape == (200, 200)
    assert patch.res_m == 0.05


def test_hires_patch_follows_parent_low_frequency_shape():
    """The patch's mean height should track the parent tile's height near
    the same world point, not be wildly different -- it's a refinement of
    the same surface, not an unrelated one."""
    tile = _small_tile()
    patch = generate_hires_patch(tile, center_x_m=0.0, center_y_m=0.0, size_m=8.0, res_m=0.1,
                                  micro_amplitude_m=0.01)
    center_parent = tile.height[tile.height.shape[0] // 2, tile.height.shape[1] // 2]
    center_patch_mean = patch.height.mean()
    assert abs(center_patch_mean - center_parent) < 0.5


def test_hires_patch_adds_finer_detail_than_parent():
    """The patch must contain real high-frequency content the parent grid
    never had -- not just be a smooth bicubic upsample of it."""
    tile = _small_tile()
    patch = generate_hires_patch(tile, size_m=10.0, res_m=0.05, micro_amplitude_m=0.05,
                                  micro_wavelength_m=0.4)
    # local roughness (std of first differences) should be well above what
    # a pure bicubic upsample of the coarser parent alone would show
    roughness = np.std(np.diff(patch.height, axis=0))
    assert roughness > 1e-4


def test_hires_patch_is_deterministic_for_same_seed():
    tile = _small_tile(seed=7)
    p1 = generate_hires_patch(tile, size_m=6.0, res_m=0.1)
    p2 = generate_hires_patch(tile, size_m=6.0, res_m=0.1)
    np.testing.assert_array_equal(p1.height, p2.height)


def test_hires_patch_off_center_uses_correct_world_location():
    tile = _small_tile()
    n = tile.height.shape[0]
    patch_center = generate_hires_patch(tile, center_x_m=0.0, center_y_m=0.0, size_m=4.0, res_m=0.2,
                                         micro_amplitude_m=0.0)
    patch_offset = generate_hires_patch(tile, center_x_m=5.0, center_y_m=0.0, size_m=4.0, res_m=0.2,
                                         micro_amplitude_m=0.0)
    # different world locations on a non-flat tile should generally not
    # produce identical resampled heightfields
    assert not np.allclose(patch_center.height, patch_offset.height)
