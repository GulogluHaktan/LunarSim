"""Exercises `lunarsim.adapters.isaac.lander.spawn_apollo_lm`'s pure USD
authoring against `usd-core` (real `pxr` bindings, no Isaac Sim/omni
runtime needed -- this only checks that the stage is authored correctly,
not that PhysX simulates it correctly).
"""
import os

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from lunarsim.adapters.isaac.lander import author_physics, spawn_apollo_lm  # noqa: E402
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, moment_of_inertia  # noqa: E402

_ASSET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "models", "apollo_lm", "Apollo_Lunar_Module.usdz"
)


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def test_asset_file_exists():
    assert os.path.isfile(_ASSET_PATH)


def test_root_has_rigid_body_and_real_mass():
    stage = _stage()
    root = spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    prim = root.GetPrim()
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)

    specs = ApolloLMSpecs()
    mass_api = UsdPhysics.MassAPI(prim)
    assert mass_api.GetMassAttr().Get() == pytest.approx(specs.dry_mass_kg + specs.descent_propellant_kg)


def test_custom_fuel_load_changes_mass():
    stage = _stage()
    specs = ApolloLMSpecs()
    root = spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH, fuel_kg=0.0)
    mass_api = UsdPhysics.MassAPI(root.GetPrim())
    assert mass_api.GetMassAttr().Get() == pytest.approx(specs.dry_mass_kg)


def test_inertia_matches_moment_of_inertia_helper():
    stage = _stage()
    specs = ApolloLMSpecs()
    root = spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    mass_api = UsdPhysics.MassAPI(root.GetPrim())
    i_tilt, i_yaw = moment_of_inertia(specs.dry_mass_kg + specs.descent_propellant_kg, specs.body_radius_m, specs.height_m)
    ixx, iyy, izz = mass_api.GetDiagonalInertiaAttr().Get()
    assert ixx == pytest.approx(i_tilt, rel=1e-4)
    assert iyy == pytest.approx(i_tilt, rel=1e-4)
    assert izz == pytest.approx(i_yaw, rel=1e-4)


def test_dps_engine_locator_carries_real_thrust_range():
    stage = _stage()
    specs = ApolloLMSpecs()
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    dps = stage.GetPrimAtPath("/World/LM/DPS_Engine")
    assert dps.IsValid()
    assert dps.GetAttribute("lunarsim:thrustMinN").Get() == pytest.approx(specs.dps_thrust_min_n)
    assert dps.GetAttribute("lunarsim:thrustMaxN").Get() == pytest.approx(specs.dps_thrust_max_n)


def test_all_16_rcs_jets_are_authored_with_real_thrust():
    stage = _stage()
    specs = ApolloLMSpecs()
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    jets = []
    for q in range(1, 5):
        for j in range(1, 5):
            jet = stage.GetPrimAtPath(f"/World/LM/RCS/Quad{q}/Jet{j}")
            assert jet.IsValid()
            assert jet.GetAttribute("lunarsim:thrustN").Get() == pytest.approx(specs.rcs_thruster_thrust_n)
            jets.append(jet)
    assert len(jets) == specs.rcs_thruster_count


def test_visual_reference_is_scaled_to_real_dimensions():
    stage = _stage()
    specs = ApolloLMSpecs()
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/LM/Visual"))
    size = box.ComputeAlignedRange().GetMax() - box.ComputeAlignedRange().GetMin()
    # Z is up in this stage; the visual reference's height must match the
    # real, documented Apollo LM height even though the raw mesh does not.
    assert size[2] == pytest.approx(specs.height_m, rel=1e-3)


def test_visual_footpads_rest_on_the_collision_contact_plane():
    """The lowest visible point of the vehicle must be the same plane the
    collision cylinder makes contact on.

    This is the assertion that was missing while the model floated: the raw
    mesh stands on its own origin instead of being centred on it, so without
    a compensating shift the visual LM sat 3.66 m ABOVE the cylinder face
    that actually touches the ground -- footpads visibly in mid-air, the
    vehicle balanced on nothing.
    """
    stage = _stage()
    specs = ApolloLMSpecs()
    # visual_only: these two tests inspect only the visual reference, and
    # skipping the physics proxy keeps them runnable without PhysxSchema.
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH, visual_only=True)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    visual_min_z = cache.ComputeWorldBound(
        stage.GetPrimAtPath("/World/LM/Visual")
    ).ComputeAlignedRange().GetMin()[2]
    # the collision cylinder is centred on the body origin, so its bottom
    # face -- the surface that rests on terrain -- is at -height/2.
    assert visual_min_z == pytest.approx(-specs.height_m / 2.0, abs=1e-3)


def test_ground_contacting_geometry_spans_the_real_footpad_width():
    """The part of the model that reaches the ground must be as wide as the
    documented footpad span.

    Measured on the prim that owns the model's lowest point (the legs and
    descent stage) rather than on the whole model's bounding box: the asset
    carries a thin boom high on the ascent stage that reaches far out in
    +x, and taking the footprint from the full box is exactly what made the
    horizontal scale 2.24x too small, rendering a 4.19 m wide LM.
    """
    stage = _stage()
    specs = ApolloLMSpecs()
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH, visual_only=True)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    lowest_prim, lowest_z = None, float("inf")
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/LM/Visual")):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        if rng.GetMin()[2] < lowest_z:
            lowest_prim, lowest_z = prim, rng.GetMin()[2]
    assert lowest_prim is not None

    rng = cache.ComputeWorldBound(lowest_prim).ComputeAlignedRange()
    span = rng.GetMax() - rng.GetMin()
    assert span[0] == pytest.approx(specs.footpad_span_m, rel=0.02)
    assert span[1] == pytest.approx(specs.footpad_span_m, rel=0.02)


def test_physics_proxy_collision_is_independent_of_visual_mesh():
    stage = _stage()
    specs = ApolloLMSpecs()
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    proxy = stage.GetPrimAtPath("/World/LM/PhysicsProxy")
    assert proxy.HasAPI(UsdPhysics.CollisionAPI)
    cylinder = UsdGeom.Cylinder(proxy)
    assert cylinder.GetRadiusAttr().Get() == pytest.approx(specs.body_radius_m)
    # the body cylinder no longer spans the whole vehicle: it is the descent
    # stage only, raised clear of the footpads that now carry touchdown (see
    # `test_descent_stage_cylinder_sits_above_the_footpads`).
    assert 0.0 < cylinder.GetHeightAttr().Get() < specs.height_m


def test_centre_of_mass_is_pinned_to_the_body_origin():
    """With several collision shapes instead of one origin-centred cylinder,
    PhysX would derive the centre of mass from the shapes -- the footpads
    would drag it down and silently change every thrust and RCS moment arm,
    in a model whose inertia tensor is authored about the origin."""
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/LM")
    author_physics(stage, "/World/LM")
    com = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/LM")).GetCenterOfMassAttr().Get()
    assert com is not None, "centre of mass must be authored explicitly"
    assert tuple(com) == pytest.approx((0.0, 0.0, 0.0))


def test_footpads_carry_the_touchdown_load_at_the_real_stance():
    """Four footpads at the documented stance, with their undersides on the
    contact plane -- not one narrow cylinder.

    The vehicle used to collide as a single 2.1 m radius cylinder spanning
    the whole height, so it rested on a support polygon 4.5x narrower than
    its real gear and rang for ~10 s after touchdown. The landing-safety
    test already measured terrain spread under exactly these four points,
    so until now the collider and the criterion disagreed about where the
    vehicle touches.
    """
    from lunarsim.adapters.isaac.lander import author_physics

    stage = _stage()
    specs = ApolloLMSpecs()
    UsdGeom.Xform.Define(stage, "/World/LM")
    author_physics(stage, "/World/LM", specs)

    contact_plane_z = -specs.height_m / 2.0
    stance = specs.footpad_span_m / 2.0
    seen = []
    for q in range(1, 5):
        pad = stage.GetPrimAtPath(f"/World/LM/Footpad{q}")
        assert pad.IsValid(), f"Footpad{q} missing"
        assert pad.HasAPI(UsdPhysics.CollisionAPI)
        radius = UsdGeom.Sphere(pad).GetRadiusAttr().Get()
        assert radius == pytest.approx(specs.footpad_radius_m)
        pos = UsdGeom.Xformable(pad).GetOrderedXformOps()[0].Get()
        # the pad's UNDERSIDE, not its centre, is what rests on the ground
        assert pos[2] - radius == pytest.approx(contact_plane_z, abs=1e-9)
        assert pytest.approx(stance, rel=1e-9) == (pos[0] ** 2 + pos[1] ** 2) ** 0.5
        seen.append((round(pos[0], 6), round(pos[1], 6)))
    assert len(set(seen)) == 4, "footpads must be at four distinct azimuths"


def test_descent_stage_cylinder_sits_above_the_footpads():
    """The body collider must not reach the contact plane, or it would take
    the load and the footpads would be decorative."""
    from lunarsim.adapters.isaac.lander import author_physics

    stage = _stage()
    specs = ApolloLMSpecs()
    UsdGeom.Xform.Define(stage, "/World/LM")
    author_physics(stage, "/World/LM", specs)

    proxy = stage.GetPrimAtPath("/World/LM/PhysicsProxy")
    assert proxy.HasAPI(UsdPhysics.CollisionAPI)
    cylinder = UsdGeom.Cylinder(proxy)
    assert cylinder.GetRadiusAttr().Get() == pytest.approx(specs.body_radius_m)
    centre_z = UsdGeom.Xformable(proxy).GetOrderedXformOps()[0].Get()[2]
    underside_z = centre_z - cylinder.GetHeightAttr().Get() / 2.0
    assert underside_z > -specs.height_m / 2.0 + 1.0
    # and its top still reaches the top of the vehicle
    top_z = centre_z + cylinder.GetHeightAttr().Get() / 2.0
    assert top_z == pytest.approx(specs.height_m / 2.0, abs=1e-9)
