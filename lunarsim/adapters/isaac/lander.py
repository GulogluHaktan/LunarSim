"""Apollo LM vehicle authoring for Isaac.

The only asset available (`assets/models/apollo_lm/Apollo_Lunar_Module.usdz`,
a free Sketchfab model, "lunarlandernofoil_carbajal_3ds") was inspected
directly (`pxr.Usd.Stage.Open` + `UsdGeom.BBoxCache`) before writing this:
it is pure decorative geometry -- 16 mesh parts, `UsdPreviewSurface`
materials, no `UsdPhysics` schema anywhere on it -- and its own proportions
don't match the real vehicle either (measured bounding box height 3.0 m vs.
the real, documented 7.04 m, at a horizontal footprint that's already wider
than that height would suggest). So this module treats it as render-only
and builds everything physical from scratch, sourced from
`lunarsim.core.vehicle.apollo_lm` (the real/documented Apollo LM numbers),
not from the mesh:

- the visual reference gets a non-uniform scale correction (this is a
  decorative-asset scale hack, explicitly not claimed to be geometrically
  exact -- see `spawn_apollo_lm`);
- mass + collision physics live on a separate simple cylinder proxy sized
  from the real vehicle's documented body radius/height;
- the DPS (descent engine) and all 16 RCS thrusters are authored as
  locator `Xform`s carrying real thrust/Isp numbers as custom attributes,
  for whatever force-application code drives the vehicle in Isaac -- this
  module only authors the USD, it does not apply forces.

Requires a running Isaac Sim / Omniverse Kit process (imports `pxr`).
"""
from __future__ import annotations

import numpy as np

from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, moment_of_inertia

# measured directly off the raw mesh (see module docstring) -- inputs to
# the visual-only scale correction in spawn_apollo_lm, not real-vehicle numbers.
_RAW_MESH_HEIGHT_M = 3.0
_RAW_MESH_FOOTPRINT_M = 8.622


def spawn_apollo_lm(
    stage,
    prim_path: str,
    visual_asset_path: str,
    specs: ApolloLMSpecs | None = None,
    fuel_kg: float | None = None,
    visual_only: bool = False,
):
    """Author the vehicle at `prim_path`: a `RigidBodyAPI`+`MassAPI` root,
    a render-only visual reference, a cylinder collision proxy, one DPS
    locator, and 16 RCS jet locators (4 quads x 4).

    `fuel_kg` sets the DPS descent-propellant load at spawn (defaults to
    the full real load, `specs.descent_propellant_kg`) -- pass a lower
    value to author the vehicle partway through a descent.

    `visual_only=True` skips the RigidBodyAPI/MassAPI/collision proxy
    entirely -- REAL BUG FOUND VIA A RENDER: a script that pose-drives
    this prim kinematically frame-by-frame (e.g. replaying a precomputed
    analytic trajectory, `scripts/isaaclab_landing_render.py`) still had
    it fall and settle under normal PhysX rigid-body dynamics instead of
    following the driven pose, because `physics:kinematicEnabled=True`
    alone isn't enough -- a kinematic body's pose has to be pushed through
    the physics kinematic-target API, not plain `UsdGeom.Xformable` ops,
    or PhysX's own cached transform silently wins. For a pure visual
    replay (no physics simulation wanted at all), the actual fix is to
    not attach rigid-body physics in the first place.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    specs = specs or ApolloLMSpecs()
    fuel_kg = specs.descent_propellant_kg if fuel_kg is None else fuel_kg
    total_mass_kg = float(specs.dry_mass_kg + fuel_kg)

    root = UsdGeom.Xform.Define(stage, prim_path)
    root_prim = root.GetPrim()

    # -- visual mesh: reference-only, no physics on it --
    # USD composition does NOT auto-rescale raw xform values across a
    # differing `metersPerUnit` between a reference and the stage it's
    # referenced into (metersPerUnit is metadata, not an implicit
    # transform) -- verified against this exact asset, which is authored
    # at metersPerUnit=0.01 (cm-scale raw values): referencing it as-is
    # into a metersPerUnit=1.0 stage produced a ~940 m tall vehicle. That
    # source/target ratio has to be folded into the correction scale below.
    visual_path = f"{prim_path}/Visual"
    visual = UsdGeom.Xform.Define(stage, visual_path)
    visual.GetPrim().GetReferences().AddReference(visual_asset_path)
    visual.AddRotateXOp().Set(90.0)  # source mesh is Y-up; this stage is Z-up

    source_mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(visual_asset_path))
    target_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    unit_scale = source_mpu / target_mpu

    scale_vertical = unit_scale * specs.height_m / _RAW_MESH_HEIGHT_M
    scale_horizontal = unit_scale * specs.footpad_span_m / _RAW_MESH_FOOTPRINT_M
    visual.AddScaleOp().Set(Gf.Vec3f(scale_horizontal, scale_vertical, scale_horizontal))

    # -- physics proxy: mass + collision, independent of the visual mesh --
    if not visual_only:
        proxy = UsdGeom.Cylinder.Define(stage, f"{prim_path}/PhysicsProxy")
        proxy.CreateRadiusAttr(specs.body_radius_m)
        proxy.CreateHeightAttr(specs.height_m)
        proxy.CreateAxisAttr("Z")
        UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())

        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr(total_mass_kg)
        i_tilt, i_yaw = moment_of_inertia(total_mass_kg, specs.body_radius_m, specs.height_m)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(i_tilt, i_tilt, i_yaw))

        # REAL BUG FOUND VIA A DOCKER RUN: a real, uncontrolled ballistic
        # descent (released at ~340m, no thrust to slow it) hits the ground
        # at ~33 m/s -- without CCD, PhysX's discrete per-step collision
        # check can miss the terrain's exact triangle mesh entirely at that
        # approach speed (it did: telemetry showed the vehicle passing
        # straight through, in a perfectly straight line with zero rotation
        # or deceleration, down to -180m below the surface -- a clean
        # tunneling miss, not a partial/glitchy contact). Confirmed
        # velocity-dependent: a slower ~11 m/s impact (from a ~40m release)
        # collided fine. Continuous collision detection makes PhysX sweep
        # the body's motion across the step instead of only testing the
        # start/end poses, which is what a small, fast body needs against a
        # comparatively thin static mesh.
        #
        # REAL BUG FOUND FIXING THAT, VIA A LATER DOCKER RUN: full/discrete
        # CCD then produced the opposite failure on a different real
        # descent -- telemetry showed a ~1860 m/s^2 deceleration spike at
        # first contact (consistent -- a ~31 m/s impact stopped in one
        # 1/60s step) immediately followed by the tracked position
        # climbing back UP for many steps instead of settling, a classic
        # symptom of PhysX's discrete-CCD sweep over-resolving/exploding
        # against a non-convex triangle mesh collider (this terrain's
        # collision IS exactly that -- `physxCollision:approximation=
        # "none"`, the real crater/slope shape, not a convex hull) --  a
        # documented bad combination for PhysX CCD in general, not
        # specific to this scene. Speculative CCD uses a contact-margin
        # approach instead of a full discrete sweep+resolve and is the
        # standard mitigation for exactly this concave-mesh failure mode,
        # while still catching the same high-speed tunneling case.
        from pxr import PhysxSchema
        physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_rigid_body.CreateEnableSpeculativeCCDAttr(True)

    # -- DPS (descent engine): throttle-only along body -Z; real thrust range as metadata --
    dps_prim = UsdGeom.Xform.Define(stage, f"{prim_path}/DPS_Engine").GetPrim()
    dps_prim.CreateAttribute("lunarsim:thrustMinN", Sdf.ValueTypeNames.Float).Set(specs.dps_thrust_min_n)
    dps_prim.CreateAttribute("lunarsim:thrustMaxN", Sdf.ValueTypeNames.Float).Set(specs.dps_thrust_max_n)
    dps_prim.CreateAttribute("lunarsim:ispS", Sdf.ValueTypeNames.Float).Set(specs.dps_isp_s)
    dps_prim.CreateAttribute("lunarsim:gimbalMaxDeg", Sdf.ValueTypeNames.Float).Set(specs.dps_gimbal_max_deg)

    # -- RCS: 4 quads x 4 jets, real per-jet thrust, positioned at the real quad radius --
    UsdGeom.Xform.Define(stage, f"{prim_path}/RCS")
    jets_per_quad = specs.rcs_thruster_count // 4
    # quad mount height above CG: not a sourced measurement (see module
    # docstring caveat on non-mesh geometry), just a plausible fraction of
    # vehicle height putting the quads near the ascent-stage top.
    quad_height_offset_m = specs.height_m * 0.35
    for q in range(4):
        az = np.deg2rad(90.0 * q)
        qx = specs.rcs_quad_radius_m * np.cos(az)
        qy = specs.rcs_quad_radius_m * np.sin(az)
        quad_path = f"{prim_path}/RCS/Quad{q + 1}"
        quad = UsdGeom.Xform.Define(stage, quad_path)
        quad.AddTranslateOp().Set(Gf.Vec3d(qx, qy, quad_height_offset_m))
        for j in range(jets_per_quad):
            jet_prim = UsdGeom.Xform.Define(stage, f"{quad_path}/Jet{j + 1}").GetPrim()
            jet_prim.CreateAttribute("lunarsim:thrustN", Sdf.ValueTypeNames.Float).Set(specs.rcs_thruster_thrust_n)
            jet_prim.CreateAttribute("lunarsim:ispS", Sdf.ValueTypeNames.Float).Set(specs.rcs_isp_s)

    return root
