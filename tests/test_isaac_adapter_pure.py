"""Only tests the parts of the Isaac adapter that don't touch `pxr`/`omni`
(no Isaac Sim install available in this environment)."""
from dataclasses import dataclass

import numpy as np

from lunarsim.adapters.isaac.dust import dust_event_from_disturbance
from lunarsim.adapters.isaac.rocks import cap_rock_count
from lunarsim.core.terrain.rocks import RockField


@dataclass
class _FakeTile:
    rocks: RockField


def test_cap_rock_count_keeps_largest_and_respects_max():
    rng = np.random.default_rng(0)
    n = 100
    rocks = RockField(
        x_m=rng.uniform(-10, 10, n),
        y_m=rng.uniform(-10, 10, n),
        diameter_m=np.linspace(0.1, 5.0, n),
    )
    tile = _FakeTile(rocks)

    capped = cap_rock_count(tile, max_count=10, rng=rng)
    assert len(capped.rocks.diameter_m) == 10
    # the 10 largest of a linspace(0.1, 5.0, 100) are all > 4.5
    assert capped.rocks.diameter_m.min() > 4.5


def test_cap_rock_count_noop_when_under_limit():
    rng = np.random.default_rng(0)
    rocks = RockField(x_m=np.zeros(5), y_m=np.zeros(5), diameter_m=np.ones(5))
    tile = _FakeTile(rocks)
    capped = cap_rock_count(tile, max_count=10, rng=rng)
    assert capped is tile


def test_dust_event_from_disturbance_scales_with_intensity():
    low = dust_event_from_disturbance(np.array([0.0, 0.0, 0.0]), intensity=0.0)
    high = dust_event_from_disturbance(np.array([0.0, 0.0, 0.0]), intensity=1.0)
    assert low.n_particles < high.n_particles
    assert low.speed_range_m_s[1] < high.speed_range_m_s[1]


def test_dust_event_from_disturbance_clips_out_of_range_intensity():
    event = dust_event_from_disturbance(np.array([0.0, 0.0, 0.0]), intensity=5.0)
    event_clipped = dust_event_from_disturbance(np.array([0.0, 0.0, 0.0]), intensity=1.0)
    assert event.n_particles == event_clipped.n_particles


# ----------------------------------------------------------------------
# Ground-contact geometry (the real bug these were written for: a real
# Isaac Sim ZemZev capture, out/eval_snapshots/orbit_descent_multiseed_44/,
# landed, parked motionless for 16 s at a reported 0.067 m altitude, and
# was still classified `end_reason=max_episode_s` because touchdown was
# tested as "belly centre within 1 mm of the ground under the body origin")
# ----------------------------------------------------------------------
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs
from lunarsim.adapters.isaac.isaac_lander_env import (  # noqa: E402
    TOUCHDOWN_CONTACT_EPS_M, collision_mesh_height_at, contact_clearance_m, out_of_tile,
)
from lunarsim.core.terrain.generate import Tile  # noqa: E402
from lunarsim.core.terrain.rocks import sample_height_at  # noqa: E402


def _mesh_faces(n):
    """The literal triangulation `heightfield._mesh_from_heightfield` emits."""
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            p00, p10 = i * n + j, (i + 1) * n + j
            p01, p11 = i * n + (j + 1), (i + 1) * n + (j + 1)
            faces += [(p00, p10, p11), (p00, p11, p01)]
    return faces


def test_collision_mesh_height_matches_the_actual_triangulation():
    """`collision_mesh_height_at` must reproduce the exact collision mesh
    PhysX is given, not an interpolation that merely looks similar."""
    rng = np.random.default_rng(0)
    n, res = 9, 3.0
    height = rng.normal(size=(n, n)) * 2.0
    ax = (np.arange(n) - (n - 1) / 2) * res
    pts = np.array([(ax[i], ax[j], height[i, j]) for i in range(n) for j in range(n)])

    for _ in range(200):
        x, y = rng.uniform(ax[0], ax[-1], 2)
        brute = None
        for a_i, b_i, c_i in _mesh_faces(n):
            a, b, c = pts[a_i], pts[b_i], pts[c_i]
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det
            l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det
            l3 = 1.0 - l1 - l2
            if min(l1, l2, l3) >= -1e-9:
                brute = l1 * a[2] + l2 * b[2] + l3 * c[2]
                break
        assert brute is not None
        got = float(collision_mesh_height_at(height, res, np.array([x]), np.array([y]))[0])
        assert abs(got - brute) < 1e-9


def test_collision_mesh_height_differs_from_nearest_sample_on_a_slope():
    """The nearest-sample lookup the rest of the codebase uses is NOT the
    surface PhysX collides with -- that difference is what made the 1 mm
    touchdown tolerance unreachable."""
    n, res = 5, 4.0
    ax = (np.arange(n) - (n - 1) / 2) * res
    height = np.tile(ax.reshape(-1, 1) * 0.1, (1, n))  # 10% slope along +x
    q = np.array([1.9])  # just inside the cell centred on ax=0, so NN -> 0.0
    nearest = float(sample_height_at(height, res, q, np.array([0.0]))[0])
    mesh = float(collision_mesh_height_at(height, res, q, np.array([0.0]))[0])
    assert nearest == 0.0
    assert abs(mesh - 0.19) < 1e-9


def test_level_vehicle_resting_on_a_flat_slab_reads_zero_clearance():
    n, res, half_h, r_body = 9, 2.0, 3.52, 2.1
    height = np.zeros((n, n))
    clearance, ground = contact_clearance_m(height, res, 0.0, 0.0, half_h, 0.0, 0.0, half_h, r_body)
    assert abs(ground) < 1e-12
    assert abs(clearance) < 1e-12


def test_tilted_vehicle_on_flat_ground_is_detected_as_touched_down():
    """A 5 deg resting tilt lifts the BELLY CENTRE 0.18 m off the ground
    (body_radius_m * sin(5 deg) = 2.1 * 0.0872), which the old
    `belly_z <= ground + 1e-3` test read as "still flying". The lowest
    point of the collider is what actually touches."""
    n, res, half_h, r_body = 9, 2.0, 3.52, 2.1
    height = np.zeros((n, n))
    tilt = np.deg2rad(5.0)
    z_rest = half_h * np.cos(tilt) + r_body * np.sin(tilt)  # exactly resting
    belly_above_ground = (z_rest - half_h)
    assert belly_above_ground > 0.16  # the old test's blind spot, in metres

    clearance, _ = contact_clearance_m(height, res, 0.0, 0.0, z_rest, tilt, 0.0, half_h, r_body)
    assert abs(clearance) < 1e-9
    assert clearance <= TOUCHDOWN_CONTACT_EPS_M


def test_clearance_ring_is_the_footpad_stance_not_the_body_radius():
    """The vehicle rests on its four footpads (9.4 m stance), not on the
    2.1 m descent-stage cylinder it used to collide as.

    Terrain with a rise that falls OUTSIDE the old 2.1 m disc but under the
    real footpads: the narrow model says the vehicle is still 0.3 m above
    the ground, the footpad model correctly says it is already resting on
    the rise. Getting this wrong is what let the vehicle settle balanced on
    a 2.1 m disc and rock for ~10 s after touchdown.
    """
    specs = ApolloLMSpecs()
    stance = specs.footpad_span_m / 2.0
    half_h = specs.height_m / 2.0
    n, res = 41, 1.0
    ax = (np.arange(n) - (n - 1) / 2) * res
    gx, _gy = np.meshgrid(ax, ax, indexing="ij")
    height = np.where(gx > 3.5, 0.3, 0.0)  # rise beyond the body radius, under a footpad

    narrow, narrow_ground = contact_clearance_m(
        height, res, 0.0, 0.0, half_h, 0.0, 0.0, half_h, specs.body_radius_m)
    wide, wide_ground = contact_clearance_m(
        height, res, 0.0, 0.0, half_h, 0.0, 0.0, half_h, stance)

    assert abs(narrow_ground) < 1e-9, "the rise is outside the 2.1 m disc"
    assert abs(narrow - 0.0) < 1e-9
    assert abs(wide_ground - 0.3) < 1e-9, "the rise is under the footpads"
    assert wide < 0.0, "the footpad model sees contact the narrow one misses"


def test_tilt_penalty_scales_with_the_stance_not_the_body():
    """Tilting lifts the contact ring by `radius * sin(tilt)`, so a wider
    stance reaches the ground sooner. The footpad stance is 2.24x the body
    radius, and the clearance difference has to follow that exactly."""
    specs = ApolloLMSpecs()
    stance = specs.footpad_span_m / 2.0
    half_h = specs.height_m / 2.0
    n, res = 41, 1.0
    height = np.zeros((n, n))
    tilt = np.deg2rad(6.0)
    z = 10.0

    narrow, _ = contact_clearance_m(height, res, 0.0, 0.0, z, tilt, 0.0, half_h, specs.body_radius_m)
    wide, _ = contact_clearance_m(height, res, 0.0, 0.0, z, tilt, 0.0, half_h, stance)
    expected = (stance - specs.body_radius_m) * np.sin(tilt)
    assert abs((narrow - wide) - expected) < 1e-9


def test_clearance_uses_the_highest_ground_under_the_footprint():
    """A level vehicle straddling a ridge rests on the ridge, not on the
    terrain directly under its axis."""
    n, res, half_h, r_body = 11, 1.0, 3.52, 2.1
    height = np.zeros((n, n))
    height[7:, :] = 0.4  # a step up 2 m along +x of the origin, inside the footprint
    clearance, ground = contact_clearance_m(height, res, 0.0, 0.0, half_h, 0.0, 0.0, half_h, r_body)
    assert abs(ground - 0.4) < 1e-9
    assert abs(clearance + 0.4) < 1e-9  # belly is 0.4 m BELOW the ridge top


def _tile(size_m):
    return Tile(height=np.zeros((9, 9)), res_m=size_m / 8.0, size_m=size_m, seed=0,
                craters=None, rocks=None, params=None, mode="fine", coarse_source="procedural")


def test_out_of_tile_flags_positions_past_the_collision_mesh_edge():
    tile = _tile(600.0)
    margin = 4.7
    assert not out_of_tile(tile, 0.0, 0.0, margin)
    assert not out_of_tile(tile, 290.0, 0.0, margin)
    # every real orbit_descent capture on disk ended 277-650 m from the tile
    # centre on this 600 m tile -- past 295.3 m there is no collider at all
    assert out_of_tile(tile, 296.0, 0.0, margin)
    assert out_of_tile(tile, 0.0, -487.0, margin)
