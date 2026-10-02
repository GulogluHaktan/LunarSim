"""Tests for `core.metadata.mapping` plus the spherical->world frame
conversion in `core.metadata.lidar` that feeds it.

The substantive test here is `test_map_reconstructs_known_heightfield`: it
raycasts a KNOWN heightfield from several poses with the analytic LiDAR,
re-encodes each scan the way Isaac's RTX sensor reports one (azimuth deg /
elevation deg / range m, in the sensor frame), pushes it back through
`spherical_to_cartesian` + `sensor_to_world` + `TerrainMap.integrate`, and
then checks the resulting map against the heightfield it started from. That
closes the loop on both the conversion and the accumulation at once -- a
sign error or a transposed rotation in either one shows up as a map that
does not match the terrain that produced it.
"""
import numpy as np
import pytest

from lunarsim.core.metadata.lidar import (
    LidarScanPattern,
    _bilinear_height,
    euler_to_rotation_matrix,
    raycast_lidar,
    sensor_to_world,
    spherical_to_cartesian,
)
from lunarsim.core.metadata.mapping import (
    FOOTPAD_SPAN_M,
    MapGrid,
    TerrainMap,
)


# ---------------------------------------------------------------------------
# MapGrid
# ---------------------------------------------------------------------------
def test_grid_shape_and_extent():
    grid = MapGrid(cell_m=2.0, half_extent_m=100.0)
    assert grid.n == 100
    assert grid.extent == (-100.0, 100.0, -100.0, 100.0)


def test_grid_rejects_degenerate_config():
    with pytest.raises(ValueError):
        MapGrid(cell_m=0.0, half_extent_m=100.0)
    with pytest.raises(ValueError):
        MapGrid(cell_m=2.0, half_extent_m=-1.0)
    with pytest.raises(ValueError):
        MapGrid(cell_m=100.0, half_extent_m=50.0)  # 1 cell per side


def test_grid_indexing_covers_corners_and_flags_outside():
    grid = MapGrid(cell_m=1.0, half_extent_m=5.0)
    x = np.array([-5.0, -4.5, 0.0, 4.9, -5.1, 5.0, 0.0])
    y = np.array([-5.0, -4.5, 0.0, 4.9, 0.0, 0.0, 99.0])
    ix, iy, inside = grid.index_of(x, y)
    assert inside.tolist() == [True, True, True, True, False, False, False]
    assert (ix[0], iy[0]) == (0, 0)
    assert (ix[2], iy[2]) == (5, 5)


def test_cell_centers_are_cell_centered():
    grid = MapGrid(cell_m=2.0, half_extent_m=10.0)
    centers = grid.cell_centers_1d()
    assert centers[0] == pytest.approx(-9.0)
    assert centers[-1] == pytest.approx(9.0)
    # a cell center must index back to its own cell
    ix, iy, inside = grid.index_of(centers, centers)
    assert inside.all()
    assert ix.tolist() == list(range(grid.n))


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------
def test_integrate_accumulates_mean_min_max_and_count():
    grid = MapGrid(cell_m=1.0, half_extent_m=2.0)
    tmap = TerrainMap(grid)
    # three hits in the same cell (centered at (0.5, 0.5)), one in another
    pts = np.array([
        [0.5, 0.5, 1.0],
        [0.6, 0.4, 3.0],
        [0.7, 0.7, 2.0],
        [-1.5, -1.5, 10.0],
    ])
    assert tmap.integrate(pts) == 4

    ix, iy, _ = grid.index_of(np.array([0.5]), np.array([0.5]))
    r, c = int(iy[0]), int(ix[0])
    assert tmap.coverage()[r, c] == 3
    assert tmap.elevation_m()[r, c] == pytest.approx(2.0)
    assert tmap.relief_m()[r, c] == pytest.approx(2.0)
    assert tmap.roughness_m()[r, c] == pytest.approx(np.std([1.0, 3.0, 2.0]))
    assert tmap.n_scans == 1
    assert tmap.n_points_integrated == 4


def test_unobserved_cells_are_nan_and_not_landable():
    grid = MapGrid(cell_m=1.0, half_extent_m=5.0)
    tmap = TerrainMap(grid)
    tmap.integrate(np.array([[0.5, 0.5, 1.0]]))
    elev = tmap.elevation_m()
    assert np.isnan(elev).sum() == elev.size - 1
    # a single hit can never satisfy the footpad test, so nothing is landable
    assert not tmap.landable_mask().any()
    # one hit is not enough for a spread/variance either
    assert np.isnan(tmap.relief_m()).all()
    assert np.isnan(tmap.roughness_m()).all()


def test_points_outside_grid_are_dropped_not_clamped():
    grid = MapGrid(cell_m=1.0, half_extent_m=3.0)
    tmap = TerrainMap(grid)
    # a far-off-map "horizon" return that clamping would pile into an edge cell
    n_kept = tmap.integrate(np.array([[500.0, 0.0, 999.0], [0.5, 0.5, 1.0]]))
    assert n_kept == 1
    assert tmap.n_points_dropped == 1
    assert tmap.coverage().sum() == 1
    assert np.nanmax(tmap.elevation_m()) == pytest.approx(1.0)


def test_non_finite_points_are_dropped():
    grid = MapGrid(cell_m=1.0, half_extent_m=3.0)
    tmap = TerrainMap(grid)
    pts = np.array([[np.nan, 0.0, 1.0], [0.0, np.inf, 1.0], [0.5, 0.5, np.nan], [0.5, 0.5, 2.0]])
    assert tmap.integrate(pts) == 1
    assert tmap.n_points_dropped == 3


def test_integrate_rejects_bad_shapes():
    tmap = TerrainMap(MapGrid(cell_m=1.0, half_extent_m=3.0))
    with pytest.raises(ValueError):
        tmap.integrate(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        tmap.integrate(np.zeros((4, 3)), intensity=np.zeros(3))


def test_empty_scan_counts_as_a_scan():
    tmap = TerrainMap(MapGrid(cell_m=1.0, half_extent_m=3.0))
    assert tmap.integrate(np.empty((0, 3))) == 0
    assert tmap.n_scans == 1


def test_last_seen_scan_tracks_the_latest_scan_only():
    grid = MapGrid(cell_m=1.0, half_extent_m=3.0)
    tmap = TerrainMap(grid)
    tmap.integrate(np.array([[0.5, 0.5, 1.0]]), scan_index=7)
    tmap.integrate(np.array([[-0.5, -0.5, 1.0]]), scan_index=9)
    last = tmap.last_seen_scan()
    ix, iy, _ = grid.index_of(np.array([0.5, -0.5]), np.array([0.5, -0.5]))
    assert last[iy[0], ix[0]] == 7
    assert last[iy[1], ix[1]] == 9
    assert (last == -1).sum() == last.size - 2


def test_mean_intensity():
    grid = MapGrid(cell_m=1.0, half_extent_m=2.0)
    tmap = TerrainMap(grid)
    tmap.integrate(np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]), intensity=np.array([0.2, 0.4]))
    ix, iy, _ = grid.index_of(np.array([0.5]), np.array([0.5]))
    assert tmap.intensity()[iy[0], ix[0]] == pytest.approx(0.3)


def test_accumulation_is_order_independent():
    """Two scans integrated in either order must give the same map -- the
    whole point of sum/min/max accumulation is that it has no history."""
    grid = MapGrid(cell_m=2.0, half_extent_m=20.0)
    rng = np.random.default_rng(3)
    a = np.column_stack([rng.uniform(-18, 18, 400), rng.uniform(-18, 18, 400), rng.normal(0, 1, 400)])
    b = np.column_stack([rng.uniform(-18, 18, 400), rng.uniform(-18, 18, 400), rng.normal(0, 1, 400)])

    forward, backward = TerrainMap(grid), TerrainMap(grid)
    forward.integrate(a)
    forward.integrate(b)
    backward.integrate(b)
    backward.integrate(a)

    np.testing.assert_allclose(forward.elevation_m(), backward.elevation_m(), equal_nan=True)
    np.testing.assert_allclose(forward.relief_m(), backward.relief_m(), equal_nan=True)
    np.testing.assert_array_equal(forward.coverage(), backward.coverage())


# ---------------------------------------------------------------------------
# derived layers
# ---------------------------------------------------------------------------
def _fill_plane(tmap, slope_x=0.0, z0=0.0, hits_per_cell=4, jitter=0.0, rng=None):
    """Densely populate every cell of `tmap` from the plane z = z0 + slope_x * x."""
    rng = rng or np.random.default_rng(0)
    grid = tmap.grid
    centers = grid.cell_centers_1d()
    gx, gy = np.meshgrid(centers, centers)
    xs, ys = [], []
    for _ in range(hits_per_cell):
        xs.append(gx.ravel() + rng.uniform(-0.3, 0.3, gx.size) * grid.cell_m)
        ys.append(gy.ravel() + rng.uniform(-0.3, 0.3, gy.size) * grid.cell_m)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = z0 + slope_x * x + (rng.normal(0, jitter, x.size) if jitter else 0.0)
    tmap.integrate(np.column_stack([x, y, z]))
    return tmap


def test_slope_of_a_known_plane():
    grid = MapGrid(cell_m=2.0, half_extent_m=40.0)
    tmap = _fill_plane(TerrainMap(grid), slope_x=np.tan(np.deg2rad(10.0)), hits_per_cell=12)
    slope = tmap.slope_deg()
    interior = slope[3:-3, 3:-3]
    assert np.nanmedian(interior) == pytest.approx(10.0, abs=1.0)


def test_flat_ground_is_landable_and_a_steep_plane_is_not():
    grid = MapGrid(cell_m=2.0, half_extent_m=40.0)
    flat = _fill_plane(TerrainMap(grid), slope_x=0.0, hits_per_cell=12)
    mask = flat.landable_mask()
    # the outer ring can't be evaluated (footpad samples fall off the map)
    assert mask[6:-6, 6:-6].all()

    steep = _fill_plane(TerrainMap(grid), slope_x=np.tan(np.deg2rad(25.0)), hits_per_cell=12)
    assert not steep.landable_mask()[6:-6, 6:-6].any()


def test_footpad_diff_detects_a_step_the_cell_mean_hides():
    """A 1 m step inside the footpad span must be rejected even though each
    individual cell is perfectly flat -- this is the failure mode the
    footpad-span test exists for, and a per-cell slope/roughness check alone
    would pass it."""
    grid = MapGrid(cell_m=2.0, half_extent_m=40.0)
    tmap = TerrainMap(grid)
    rng = np.random.default_rng(1)
    centers = grid.cell_centers_1d()
    gx, gy = np.meshgrid(centers, centers)
    xs = np.concatenate([gx.ravel() + rng.uniform(-0.3, 0.3, gx.size) * grid.cell_m for _ in range(12)])
    ys = np.concatenate([gy.ravel() + rng.uniform(-0.3, 0.3, gy.size) * grid.cell_m for _ in range(12)])
    zs = np.where(xs > 0.0, 1.0, 0.0)  # flat either side, 1 m step at x = 0
    tmap.integrate(np.column_stack([xs, ys, zs]))

    diff = tmap.footpad_height_diff_m()
    radius = FOOTPAD_SPAN_M / 2.0
    cell_x = gx
    near_step = np.abs(cell_x) < radius - grid.cell_m
    far_from_step = np.abs(cell_x) > radius + 2 * grid.cell_m

    assert np.nanmedian(diff[near_step]) == pytest.approx(1.0, abs=0.05)
    assert np.nanmax(diff[far_from_step & np.isfinite(diff)]) < 0.05

    mask = tmap.landable_mask()
    assert not mask[near_step].any()
    assert mask[far_from_step & (np.abs(gy) < 25)].any()


def test_footpad_diff_is_nan_when_any_footpad_is_unobserved():
    """Partial observation must not produce an optimistic (smaller) spread."""
    grid = MapGrid(cell_m=2.0, half_extent_m=40.0)
    tmap = TerrainMap(grid)
    centers = grid.cell_centers_1d()
    gx, gy = np.meshgrid(centers, centers)
    keep = gx.ravel() < 0.0  # map only the x < 0 half
    tmap.integrate(np.column_stack([gx.ravel()[keep], gy.ravel()[keep], np.zeros(keep.sum())]))

    diff = tmap.footpad_height_diff_m()
    radius = FOOTPAD_SPAN_M / 2.0
    # cells whose +x footpad lands in the unmapped half have no answer
    straddling = (gx > -radius * 0.5) & (gx < 0.0)
    assert np.isnan(diff[straddling]).all()
    assert not tmap.landable_mask()[straddling].any()


def test_summary_reports_coverage_and_landable_area():
    grid = MapGrid(cell_m=2.0, half_extent_m=40.0)
    tmap = _fill_plane(TerrainMap(grid), hits_per_cell=12)
    s = tmap.summary()
    assert s["n_scans"] == 1
    assert s["cells_observed"] == grid.n * grid.n
    assert s["coverage_frac"] == pytest.approx(1.0)
    assert s["mapped_area_m2"] == pytest.approx((2 * grid.half_extent_m) ** 2)
    assert s["cells_landable"] > 0
    assert s["landable_area_m2"] == pytest.approx(s["cells_landable"] * grid.cell_m ** 2)


# ---------------------------------------------------------------------------
# spherical -> cartesian -> world
# ---------------------------------------------------------------------------
def test_spherical_to_cartesian_known_angles():
    pts = spherical_to_cartesian([0.0, 90.0, 0.0, 180.0], [0.0, 0.0, 90.0, 0.0], [10.0, 10.0, 10.0, 5.0])
    np.testing.assert_allclose(pts[0], [10.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[1], [0.0, 10.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[2], [0.0, 0.0, 10.0], atol=1e-9)
    np.testing.assert_allclose(pts[3], [-5.0, 0.0, 0.0], atol=1e-9)


def test_spherical_to_cartesian_preserves_range():
    rng = np.random.default_rng(5)
    az = rng.uniform(-180, 180, 500)
    el = rng.uniform(-45, 45, 500)
    r = rng.uniform(1, 300, 500)
    np.testing.assert_allclose(np.linalg.norm(spherical_to_cartesian(az, el, r), axis=1), r, rtol=1e-12)


def test_spherical_matches_the_codebase_ray_direction_convention():
    """`spherical_to_cartesian` must agree with
    `LidarScanPattern.ray_directions`, which is the convention the analytic
    raycaster already uses -- otherwise the two LiDAR paths in this codebase
    would disagree about where a return came from."""
    pattern = LidarScanPattern.spinning(n_channels=6, vertical_fov_deg=(-20, 20), horizontal_res_deg=15.0)
    dirs = pattern.ray_directions()
    r = np.full(len(dirs), 7.0)
    pts = spherical_to_cartesian(
        np.degrees(pattern.azimuth_rad), np.degrees(pattern.elevation_rad), r
    )
    np.testing.assert_allclose(pts, dirs * r[:, None], atol=1e-9)


def test_spherical_to_cartesian_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spherical_to_cartesian([0.0, 1.0], [0.0], [1.0, 2.0])


def test_sensor_to_world_applies_rotation_then_translation():
    # 90 deg yaw: sensor +x becomes world +y
    rot = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2)
    out = sensor_to_world(np.array([[1.0, 0.0, 0.0]]), np.array([10.0, 20.0, 30.0]), rot)
    np.testing.assert_allclose(out[0], [10.0, 21.0, 30.0], atol=1e-9)


def test_sensor_to_world_pitch_sign_points_a_tilted_sensor_down():
    """The mount in the capture scripts is a +60 deg rotation about Y; that
    must aim the sensor's forward axis DOWN, not up."""
    rot = euler_to_rotation_matrix(0.0, np.deg2rad(60.0), 0.0)
    out = sensor_to_world(np.array([[1.0, 0.0, 0.0]]), np.zeros(3), rot)
    assert out[0, 2] < 0.0
    np.testing.assert_allclose(out[0], [0.5, 0.0, -np.sqrt(3) / 2], atol=1e-9)


def test_sensor_to_world_rejects_bad_shapes():
    with pytest.raises(ValueError):
        sensor_to_world(np.zeros((2, 3)), np.zeros(3), np.eye(2))
    with pytest.raises(ValueError):
        sensor_to_world(np.zeros((2, 3)), np.zeros(2), np.eye(3))
    with pytest.raises(ValueError):
        sensor_to_world(np.zeros((2, 2)), np.zeros(3), np.eye(3))


def test_sensor_to_world_handles_empty():
    assert sensor_to_world(np.empty((0, 3)), np.zeros(3), np.eye(3)).shape == (0, 3)


# ---------------------------------------------------------------------------
# end-to-end: known terrain -> scans -> map -> terrain
# ---------------------------------------------------------------------------
def test_map_reconstructs_known_heightfield():
    res_m = 1.0
    n = 161
    half = (n - 1) / 2 * res_m
    xs = (np.arange(n) - (n - 1) / 2) * res_m
    gx, _gy = np.meshgrid(xs, xs)
    # a gentle ramp plus a localized bump: enough structure that a wrong
    # rotation or azimuth sign cannot accidentally still match
    height = 0.05 * gx + 2.0 * np.exp(-((gx - 20.0) ** 2) / 200.0)

    pattern = LidarScanPattern.spinning(n_channels=20, vertical_fov_deg=(-11.0, 11.0), horizontal_res_deg=2.5)
    az_deg = np.degrees(pattern.azimuth_rad)
    el_deg = np.degrees(pattern.elevation_rad)
    dirs_sensor = pattern.ray_directions()

    grid = MapGrid(cell_m=4.0, half_extent_m=60.0)
    tmap = TerrainMap(grid)

    # several poses, each with the real 60 deg nose-down mount tilt plus its
    # own yaw -- the same geometry the capture scripts fly
    poses = [
        (np.array([-30.0, -10.0, 60.0]), 0.0),
        (np.array([-10.0, 20.0, 55.0]), 60.0),
        (np.array([10.0, 25.0, 45.0]), 120.0),
        (np.array([20.0, 5.0, 40.0]), 180.0),
        (np.array([25.0, -20.0, 35.0]), 240.0),
        (np.array([0.0, -25.0, 30.0]), 300.0),
    ]
    for sensor_pos, yaw_deg in poses:
        rot = euler_to_rotation_matrix(0.0, np.deg2rad(60.0), np.deg2rad(yaw_deg))
        pc = raycast_lidar(
            height, res_m, sensor_pos, dirs_sensor @ rot.T, max_range_m=300.0, step_m=0.5
        )
        hits = pc.hit_mask
        assert hits.sum() > 50, "test poses must actually see the ground"

        # re-encode as the RTX sensor reports it, then convert back
        rebuilt = sensor_to_world(
            spherical_to_cartesian(az_deg[hits], el_deg[hits], pc.range_m[hits]), sensor_pos, rot
        )
        np.testing.assert_allclose(rebuilt, pc.points_m[hits], atol=1e-6)
        tmap.integrate(rebuilt, pc.intensity[hits])

    elev = tmap.elevation_m()
    seen = tmap.observed
    assert seen.sum() > 100

    centers = grid.cell_centers_1d()
    cgx, cgy = np.meshgrid(centers, centers)
    truth = _bilinear_height(height, res_m, cgx.ravel(), cgy.ravel()).reshape(cgx.shape)
    in_tile = (np.abs(cgx) < half - 2 * grid.cell_m) & (np.abs(cgy) < half - 2 * grid.cell_m)
    compare = seen & in_tile & (tmap.coverage() >= 3)
    assert compare.sum() > 50, f"only {compare.sum()} cells densely mapped"

    err = elev[compare] - truth[compare]
    # sub-cell relief inside a 4 m cell on a ramp+bump is the error floor here
    assert np.abs(np.median(err)) < 0.15
    assert np.sqrt(np.mean(err ** 2)) < 0.4


def test_map_of_flat_terrain_is_landable_where_a_bump_is_not():
    """Same pipeline as above, but asking the operational question: the
    landability mask must reject the neighbourhood of a boulder-scale bump
    and accept the flat ground away from it."""
    res_m = 1.0
    n = 201
    xs = (np.arange(n) - (n - 1) / 2) * res_m
    gx, gy = np.meshgrid(xs, xs)
    height = 1.5 * np.exp(-(((gx - 25.0) ** 2 + gy ** 2) / 40.0))  # flat, one bump at (25, 0)

    pattern = LidarScanPattern.spinning(n_channels=16, vertical_fov_deg=(-11.0, 11.0), horizontal_res_deg=3.0)
    az_deg = np.degrees(pattern.azimuth_rad)
    el_deg = np.degrees(pattern.elevation_rad)
    dirs_sensor = pattern.ray_directions()

    grid = MapGrid(cell_m=3.0, half_extent_m=60.0)
    tmap = TerrainMap(grid)
    for sensor_pos, yaw_deg in [
        (np.array([0.0, 0.0, 70.0]), 0.0),
        (np.array([15.0, 15.0, 50.0]), 90.0),
        (np.array([15.0, -15.0, 40.0]), 200.0),
    ]:
        rot = euler_to_rotation_matrix(0.0, np.deg2rad(60.0), np.deg2rad(yaw_deg))
        pc = raycast_lidar(height, res_m, sensor_pos, dirs_sensor @ rot.T, max_range_m=300.0, step_m=0.5)
        hits = pc.hit_mask
        tmap.integrate(
            sensor_to_world(spherical_to_cartesian(az_deg[hits], el_deg[hits], pc.range_m[hits]), sensor_pos, rot),
            pc.intensity[hits],
        )

    mask = tmap.landable_mask()
    centers = grid.cell_centers_1d()
    cgx, cgy = np.meshgrid(centers, centers)
    dist_to_bump = np.hypot(cgx - 25.0, cgy)

    on_bump = dist_to_bump < 6.0
    assert not mask[on_bump].any(), "the bump must not be reported landable"
    assert mask[tmap.coverage() >= 3].any(), "some mapped flat ground must be landable"
