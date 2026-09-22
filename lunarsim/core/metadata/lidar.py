"""Analytic (heightfield) LiDAR raycasting: ground truth for validating a
renderer's RTX LiDAR output (plan section 8: "Analitik ray casting ile
ground truth mesafe hesaplanıp Isaac çıktısıyla kıyaslanır").

No atmosphere on the Moon -> no range attenuation; intensity is modeled as
reflectance(incidence angle) only (see core.lighting.materials).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarsim.core.lighting.materials import reflectance, surface_normals_from_heightmap


@dataclass
class LidarScanPattern:
    """A set of ray directions in the sensor frame (+x forward, +z up),
    built from channel/FOV specs typical of a mechanical or solid-state
    spinning LiDAR."""

    azimuth_rad: np.ndarray  # (n_rays,)
    elevation_rad: np.ndarray  # (n_rays,)

    @classmethod
    def spinning(
        cls,
        n_channels: int,
        vertical_fov_deg: tuple[float, float],
        horizontal_res_deg: float,
        horizontal_fov_deg: tuple[float, float] = (0.0, 360.0),
    ) -> "LidarScanPattern":
        elev = np.deg2rad(np.linspace(vertical_fov_deg[0], vertical_fov_deg[1], n_channels))
        az = np.deg2rad(np.arange(horizontal_fov_deg[0], horizontal_fov_deg[1], horizontal_res_deg))
        el_grid, az_grid = np.meshgrid(elev, az, indexing="ij")
        return cls(azimuth_rad=az_grid.ravel(), elevation_rad=el_grid.ravel())

    def ray_directions(self) -> np.ndarray:
        """(n_rays, 3) unit vectors in sensor frame."""
        ca = np.cos(self.azimuth_rad)
        sa = np.sin(self.azimuth_rad)
        ce = np.cos(self.elevation_rad)
        se = np.sin(self.elevation_rad)
        return np.stack([ce * ca, ce * sa, se], axis=-1)


@dataclass
class LidarPointCloud:
    """All arrays are full-length (n_rays,)/(n_rays, 3), NaN where `hit_mask` is False,
    so same-index comparison across two scans (e.g. analytic vs. RTX) is always valid.
    """

    points_m: np.ndarray  # (n_rays, 3), sensor frame
    range_m: np.ndarray  # (n_rays,)
    intensity: np.ndarray  # (n_rays,), reflectance-based, in [0, ~1]
    hit_mask: np.ndarray  # (n_rays,) bool, which rays hit terrain within max_range


def _bilinear_height(height: np.ndarray, res_m: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = height.shape[0]
    center = (n - 1) / 2
    col = x / res_m + center
    row = y / res_m + center
    r0 = np.clip(np.floor(row).astype(int), 0, n - 2)
    c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
    fr, fc = row - r0, col - c0
    z00 = height[r0, c0]
    z10 = height[r0 + 1, c0]
    z01 = height[r0, c0 + 1]
    z11 = height[r0 + 1, c0 + 1]
    return (z00 * (1 - fr) * (1 - fc) + z10 * fr * (1 - fc)
            + z01 * (1 - fr) * fc + z11 * fr * fc)


def raycast_lidar(
    height: np.ndarray,
    res_m: float,
    sensor_pos_m: np.ndarray,
    ray_dirs: np.ndarray,
    max_range_m: float = 100.0,
    step_m: float | None = None,
    brdf: str = "lommel_seeliger",
    albedo: float = 0.10,
) -> LidarPointCloud:
    """March each ray in fixed steps, refine the first heightfield crossing by
    bisection, and compute range + reflectance-based intensity.

    `sensor_pos_m` is (3,) in the same world frame as the heightmap (origin
    at tile center, z = height). `ray_dirs` is (n_rays, 3), unit vectors.
    """
    n = height.shape[0]
    step_m = step_m or res_m * 0.5
    n_steps = int(max_range_m / step_m)

    origin = np.asarray(sensor_pos_m, dtype=float)
    dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    n_rays = dirs.shape[0]

    t = np.arange(1, n_steps + 1) * step_m  # (n_steps,)
    pts = origin[None, None, :] + dirs[:, None, :] * t[None, :, None]  # (n_rays, n_steps, 3)

    half_extent = (n - 1) / 2 * res_m
    in_bounds = (np.abs(pts[..., 0]) <= half_extent) & (np.abs(pts[..., 1]) <= half_extent)

    ground_z = np.full(pts.shape[:2], np.nan)
    valid_idx = np.where(in_bounds)
    ground_z[valid_idx] = _bilinear_height(height, res_m, pts[..., 0][valid_idx], pts[..., 1][valid_idx])

    below = pts[..., 2] <= ground_z
    below = np.where(np.isnan(ground_z), False, below)

    first_hit_step = np.argmax(below, axis=1)
    any_hit = below.any(axis=1)

    range_m = np.full(n_rays, np.nan)
    hit_points = np.full((n_rays, 3), np.nan)

    for i in range(n_rays):
        if not any_hit[i]:
            continue
        k = first_hit_step[i]
        t_lo = t[k - 1] if k > 0 else 0.0
        t_hi = t[k]
        for _ in range(20):
            t_mid = 0.5 * (t_lo + t_hi)
            p = origin + dirs[i] * t_mid
            if abs(p[0]) > half_extent or abs(p[1]) > half_extent:
                break
            gz = _bilinear_height(height, res_m, np.array([p[0]]), np.array([p[1]]))[0]
            if p[2] <= gz:
                t_hi = t_mid
            else:
                t_lo = t_mid
        range_m[i] = t_hi
        hit_points[i] = origin + dirs[i] * t_hi

    hit_mask = any_hit & ~np.isnan(range_m)

    normals = surface_normals_from_heightmap(height, res_m)
    intensity = np.zeros(n_rays)
    if hit_mask.any():
        hp = hit_points[hit_mask]
        n_hat = _sample_normal(normals, res_m, hp[:, 0], hp[:, 1])
        incidence_dir = -dirs[hit_mask]  # from surface toward sensor
        mu_i = np.clip(np.einsum("ij,ij->i", n_hat, incidence_dir), 0.0, 1.0)
        mu_e = mu_i  # monostatic sensor: incidence == emission angle
        intensity[hit_mask] = reflectance(brdf, mu_i, mu_e, albedo=albedo)

    return LidarPointCloud(
        points_m=hit_points,
        range_m=range_m,
        intensity=intensity,
        hit_mask=hit_mask,
    )


def _sample_normal(normals: np.ndarray, res_m: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = normals.shape[0]
    center = (n - 1) / 2
    col = np.clip(np.round(x / res_m + center).astype(int), 0, n - 1)
    row = np.clip(np.round(y / res_m + center).astype(int), 0, n - 1)
    return normals[row, col]


def export_point_cloud(pc: LidarPointCloud, path: str, fmt: str | None = None) -> str:
    """Write hit points (NaN/miss rays dropped) to disk.

    `fmt` is inferred from `path`'s extension if not given: `.ply` (ASCII,
    xyz + intensity, opens directly in CloudCompare/MeshLab/Blender/rviz),
    `.npy` (a structured numpy array, round-trips exactly via
    `numpy.load(..., allow_pickle=False)`), or `.csv` (x_m,y_m,z_m,intensity
    header, human-readable/Excel-friendly). Returns the path written.
    """
    fmt = fmt or path.rsplit(".", 1)[-1].lower()
    hits = pc.hit_mask
    pts = pc.points_m[hits]
    intensity = pc.intensity[hits]

    if fmt == "ply":
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float intensity\nend_header\n")
            for (x, y, z), i in zip(pts, intensity):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {i:.4f}\n")
    elif fmt == "npy":
        structured = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
        structured["x"], structured["y"], structured["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        structured["intensity"] = intensity
        np.save(path, structured)
    elif fmt == "csv":
        with open(path, "w") as f:
            f.write("x_m,y_m,z_m,intensity\n")
            for (x, y, z), i in zip(pts, intensity):
                f.write(f"{x:.4f},{y:.4f},{z:.4f},{i:.4f}\n")
    else:
        raise ValueError(f"unknown point cloud format: {fmt!r} (expected 'ply', 'npy', or 'csv')")

    return path


def compare_point_clouds(ground_truth: LidarPointCloud, other: LidarPointCloud) -> dict:
    """Basic per-ray range error stats between the analytic ground truth and
    an external (e.g. Isaac RTX LiDAR) point cloud sharing the same ray order
    (both must be full-length (n_rays,) arrays, as `raycast_lidar` returns).
    """
    both_hit = ground_truth.hit_mask & other.hit_mask
    if not both_hit.any():
        return {"n_compared": 0}

    err = other.range_m[both_hit] - ground_truth.range_m[both_hit]
    return {
        "n_compared": int(both_hit.sum()),
        "mean_error_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "max_abs_error_m": float(np.max(np.abs(err))),
    }
