"""Exercises `lunarsim.adapters.isaac.lander.spawn_apollo_lm`'s pure USD
authoring against `usd-core` (real `pxr` bindings, no Isaac Sim/omni
runtime needed -- this only checks that the stage is authored correctly,
not that PhysX simulates it correctly).
"""
import os

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from lunarsim.adapters.isaac.lander import spawn_apollo_lm  # noqa: E402
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


def test_physics_proxy_collision_is_independent_of_visual_mesh():
    stage = _stage()
    specs = ApolloLMSpecs()
    spawn_apollo_lm(stage, "/World/LM", _ASSET_PATH)
    proxy = stage.GetPrimAtPath("/World/LM/PhysicsProxy")
    assert proxy.HasAPI(UsdPhysics.CollisionAPI)
    cylinder = UsdGeom.Cylinder(proxy)
    assert cylinder.GetRadiusAttr().Get() == pytest.approx(specs.body_radius_m)
    assert cylinder.GetHeightAttr().Get() == pytest.approx(specs.height_m)
