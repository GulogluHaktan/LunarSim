import numpy as np
import pytest

from lunarsim.core.metadata.lidar import LidarScanPattern, export_point_cloud, raycast_lidar


def _sample_cloud():
    height = np.zeros((201, 201))
    height[100:150, 100:150] = 3.0  # a raised block so points aren't all coplanar
    sensor_pos = np.array([0.0, 0.0, 15.0])
    pattern = LidarScanPattern.spinning(n_channels=8, vertical_fov_deg=(-60, -10), horizontal_res_deg=10)
    return raycast_lidar(height, res_m=1.0, sensor_pos_m=sensor_pos, ray_dirs=pattern.ray_directions(), max_range_m=50.0)


def test_export_ply(tmp_path):
    pc = _sample_cloud()
    path = export_point_cloud(pc, str(tmp_path / "cloud.ply"))
    text = open(path).read()
    assert text.startswith("ply\n")
    n_hits = int(pc.hit_mask.sum())
    assert f"element vertex {n_hits}" in text
    lines = text.strip().splitlines()
    header_len = lines.index("end_header") + 1
    assert len(lines) == header_len + n_hits


def test_export_npy_roundtrips(tmp_path):
    pc = _sample_cloud()
    path = export_point_cloud(pc, str(tmp_path / "cloud.npy"))
    loaded = np.load(path)
    n_hits = int(pc.hit_mask.sum())
    assert len(loaded) == n_hits
    np.testing.assert_allclose(loaded["x"], pc.points_m[pc.hit_mask, 0], atol=1e-3)
    np.testing.assert_allclose(loaded["intensity"], pc.intensity[pc.hit_mask], atol=1e-3)


def test_export_csv(tmp_path):
    pc = _sample_cloud()
    path = export_point_cloud(pc, str(tmp_path / "cloud.csv"))
    lines = open(path).read().strip().splitlines()
    assert lines[0] == "x_m,y_m,z_m,intensity"
    assert len(lines) == 1 + int(pc.hit_mask.sum())


def test_export_unknown_format_raises(tmp_path):
    pc = _sample_cloud()
    with pytest.raises(ValueError):
        export_point_cloud(pc, str(tmp_path / "cloud.xyz"))


def test_export_only_writes_hit_points_not_nan_misses(tmp_path):
    pc = _sample_cloud()
    assert not pc.hit_mask.all()  # sanity: this scan pattern does have some misses
    path = export_point_cloud(pc, str(tmp_path / "cloud.csv"))
    lines = open(path).read().strip().splitlines()[1:]
    assert len(lines) < len(pc.hit_mask)
    for line in lines:
        assert "nan" not in line.lower()
