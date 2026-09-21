import numpy as np
import pytest

from lunarsim.core.metadata.lidar import LidarScanPattern, compare_point_clouds, raycast_lidar


def test_scan_pattern_ray_directions_are_unit():
    pattern = LidarScanPattern.spinning(n_channels=8, vertical_fov_deg=(-15, 5), horizontal_res_deg=10)
    dirs = pattern.ray_directions()
    norms = np.linalg.norm(dirs, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_raycast_hits_flat_ground_at_expected_range():
    height = np.zeros((201, 201))  # res=1 -> extent [-100, 100]
    sensor_pos = np.array([0.0, 0.0, 5.0])
    # a single straight-down ray
    dirs = np.array([[0.0, 0.0, -1.0]])
    pc = raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=dirs, max_range_m=20.0)
    assert pc.hit_mask[0]
    assert pc.range_m[0] == pytest.approx(5.0, abs=0.05)


def test_raycast_misses_beyond_max_range():
    height = np.zeros((201, 201))
    sensor_pos = np.array([0.0, 0.0, 50.0])
    dirs = np.array([[0.0, 0.0, -1.0]])
    pc = raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=dirs, max_range_m=10.0)
    assert not pc.hit_mask[0]
    assert np.isnan(pc.range_m[0])


def test_raycast_on_slope_gives_expected_range():
    n = 201
    res = 1.0
    ax = (np.arange(n) - (n - 1) / 2) * res
    height = np.tile(ax * 0.5, (n, 1))  # slope: z = 0.5 * x
    sensor_pos = np.array([0.0, 0.0, 10.0])
    dirs = np.array([[0.0, 0.0, -1.0]])
    pc = raycast_lidar(height, res_m=res, sensor_pos_m=sensor_pos, ray_dirs=dirs, max_range_m=30.0)
    assert pc.hit_mask[0]
    expected_z = 0.5 * 0.0  # ray stays at x=0
    assert pc.range_m[0] == pytest.approx(10.0 - expected_z, abs=0.05)


def test_intensity_higher_at_normal_incidence_than_grazing():
    height = np.zeros((401, 401))
    sensor_pos = np.array([0.0, 0.0, 10.0])
    straight_down = np.array([[0.0, 0.0, -1.0]])
    grazing = np.array([[0.995, 0.0, -0.1]])
    grazing = grazing / np.linalg.norm(grazing)

    pc_down = raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=straight_down, max_range_m=50.0)
    pc_graze = raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=grazing, max_range_m=50.0)

    assert pc_down.intensity[0] > pc_graze.intensity[0]


def test_compare_point_clouds_zero_error_for_identical_scans():
    height = np.zeros((201, 201))
    sensor_pos = np.array([0.0, 0.0, 5.0])
    dirs = LidarScanPattern.spinning(n_channels=4, vertical_fov_deg=(-30, -10), horizontal_res_deg=45).ray_directions()

    pc1 = raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=dirs, max_range_m=50.0)
    pc2 = raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=dirs, max_range_m=50.0)

    stats = compare_point_clouds(pc1, pc2)
    assert stats["n_compared"] == int(pc1.hit_mask.sum())
    assert stats["rmse_m"] == pytest.approx(0.0, abs=1e-9)
