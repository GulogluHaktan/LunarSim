import numpy as np
import pytest

from lunarsim.core.terrain.blend import RoiSpec, roi_weight
from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import generate_tile


def test_roi_weight_backward_compatible_single_sigma():
    w = roi_weight(101, 1.0, [(50, 50)], sigma_m=10.0)
    assert w.shape == (101, 101)
    assert w[50, 50] == pytest.approx(1.0)
    assert w[0, 0] < 0.01  # far corner should be ~unaffected


def test_roi_weight_per_region_sigma_and_strength():
    specs = [
        RoiSpec(center_px=(20, 20), sigma_m=5.0, weight=1.0),   # small, full-strength
        RoiSpec(center_px=(80, 80), sigma_m=20.0, weight=0.5),  # wide, half-strength
    ]
    w = roi_weight(101, 1.0, specs)
    assert w[20, 20] == pytest.approx(1.0)
    assert w[80, 80] == pytest.approx(0.5)
    # the wide region's falloff should reach further than the narrow one
    assert w[80, 90] > w[20, 30]


def test_roi_weight_overlap_takes_max_not_sum():
    specs = [
        RoiSpec(center_px=(50, 50), sigma_m=10.0, weight=0.8),
        RoiSpec(center_px=(50, 51), sigma_m=10.0, weight=0.8),
    ]
    w = roi_weight(101, 1.0, specs)
    assert w.max() <= 1.0 + 1e-9


def test_blend_mechanism_gives_full_detail_at_roi_and_none_far_away():
    """Direct, deterministic check of the blend+ROI mechanism itself (not a
    full stochastic generate_tile pipeline, which mixes in the coarse
    layer's own randomness and makes a statistical roughness comparison
    fragile): with a known synthetic coarse/fine pair, the blended output
    at the ROI center must equal coarse+full fine detail, and far outside
    every ROI must equal the coarse value alone (zero fine contribution)."""
    from lunarsim.core.terrain.blend import blend

    n = 201
    res_m = 1.0
    coarse = np.zeros((n, n))  # flat coarse source
    rng = np.random.default_rng(0)
    fine = rng.normal(0, 1.0, (n, n)) * 0.0
    # a single fine-scale bump the Gaussian high-pass will preserve as "detail"
    fine[100, 100] = 5.0

    roi_specs = [RoiSpec(center_px=(100, 100), sigma_m=5.0, weight=1.0)]
    w = roi_weight(n, res_m, roi_specs)
    blended = blend(coarse, fine, res_m, split_m=2.0, weight=w)

    far_specs = [RoiSpec(center_px=(100, 100), sigma_m=1.0, weight=1.0)]
    w_far_zero = roi_weight(n, res_m, far_specs)
    assert w_far_zero[0, 0] < 1e-6  # confirms (0,0) is effectively outside a tight ROI

    assert blended[100, 100] > coarse[100, 100]  # ROI center: detail injected
    assert blended[0, 0] == pytest.approx(coarse[0, 0], abs=1e-6)  # far away: no detail leaks in


def test_generate_tile_accepts_explicit_roi_regions_config():
    """Config-level wiring smoke test: `roi.regions` (world-meter x_m/y_m +
    per-region sigma_m/weight) parses and produces a tile of the right shape."""
    cfg = TerrainConfig(
        mode="blend", size_m=200.0, res_m=1.0, coarse_res_m=4.0, seed=3,
        coarse_source="procedural",
        hills={"amplitude_m": 3.0, "wavelength_m": 20.0, "hurst": 0.75},
        craters={"count_scale": 0.5, "d_min_m": 1.0, "d_max_m": 10.0, "b": 2.5,
                 "depth_ratio": 0.1, "age": 0.5},
        rocks={"density_scale": 0.0, "d_max_m": 0.5},
        roi={"regions": [
            {"x_m": 60.0, "y_m": 0.0, "sigma_m": 15.0, "weight": 1.0},
            {"x_m": -60.0, "y_m": 40.0, "sigma_m": 30.0, "weight": 0.4},
        ]},
        curvature=False,
        split_m=5.0,
    )
    tile = generate_tile(cfg)
    n = int(cfg.size_m / cfg.res_m)
    assert tile.height.shape == (n, n)
    assert np.isfinite(tile.height).all()
