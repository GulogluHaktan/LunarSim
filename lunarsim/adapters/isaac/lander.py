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
#
# Re-measured per-prim with a USD BBoxCache, because the previous footprint
# figure was taken from the asset's WHOLE bounding box and that box is
# dominated by one prim, `Object_8`: a thin boom high on the ascent stage
# (y = 231..294 cm) that reaches x = +669.8 cm while every other prim in
# the model stays inside +/-192.4 cm. It stretched the measured footprint
# to 862.2 cm, so `scale_horizontal` came out 2.24x too small and the LM
# rendered 4.19 m wide instead of its real 9.4 m footpad span -- correct
# height, far too narrow. Excluding that one prim, the body measures
# 384.7 x 300.0 x 384.6 cm, symmetric in both horizontal axes.
#
# The height figure was NOT affected (the body spans the full 300.0 cm in
# y on its own), so only the footprint changes here.
_RAW_MESH_HEIGHT_M = 3.0
_RAW_MESH_FOOTPRINT_M = 3.847
# the mesh stands on its origin rather than being centred on it: its lowest
# point (the footpads) is this far ABOVE y = 0 in raw mesh units.
_RAW_MESH_BOTTOM_M = 0.061

# How far the descent stage's collidable underside sits above the footpad
# contact plane. An engineering estimate of the deployed gear height, not a
# sourced figure -- its only job is to keep the body cylinder out of the way
# so the four pads are what touches down (see the physics proxy below).
_BODY_UNDERSIDE_ABOVE_PADS_M = 1.5


def author_physics(stage, prim_path: str, specs: ApolloLMSpecs | None = None,
                   total_mass_kg: float | None = None) -> None:
    """Author everything physical that plain `UsdPhysics` can express: the
    rigid body, its mass/inertia/centre of mass, four footpads at the real
    stance, and a raised descent-stage cylinder.

    Split out of `spawn_apollo_lm` so all of it can be asserted without an
    Isaac runtime. `spawn_apollo_lm` goes on to import `PhysxSchema` for the
    PhysX-only extras (speculative CCD), and that module ships only with
    Isaac Sim -- so a plain `usd-core` test could not otherwise reach any of
    this.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    specs = specs or ApolloLMSpecs()
    if total_mass_kg is None:
        total_mass_kg = float(specs.dry_mass_kg + specs.descent_propellant_kg)
    contact_plane_z = -specs.height_m / 2.0
    stance_radius = specs.footpad_span_m / 2.0

    # descent-stage body: raised so its underside clears the pads by roughly
    # a real LM's gear height, leaving the pads as first contact.
    body_underside_z = contact_plane_z + _BODY_UNDERSIDE_ABOVE_PADS_M
    body_height = specs.height_m / 2.0 - body_underside_z
    proxy = UsdGeom.Cylinder.Define(stage, f"{prim_path}/PhysicsProxy")
    proxy.CreateRadiusAttr(specs.body_radius_m)
    proxy.CreateHeightAttr(body_height)
    proxy.CreateAxisAttr("Z")
    UsdGeom.Xformable(proxy.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, body_underside_z + body_height / 2.0))
    UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())

    # four footpads, at the azimuths `_footpad_height_diff_m` samples.
    # Spheres rather than thin discs: a sphere is the shape PhysX's
    # speculative CCD handles most reliably against a concave triangle mesh,
    # which this terrain is (see the CCD note in `spawn_apollo_lm`).
    for q, az_deg in enumerate((0.0, 90.0, 180.0, 270.0)):
        az = np.deg2rad(az_deg)
        pad = UsdGeom.Sphere.Define(stage, f"{prim_path}/Footpad{q + 1}")
        pad.CreateRadiusAttr(specs.footpad_radius_m)
        UsdGeom.Xformable(pad.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(
            stance_radius * np.cos(az), stance_radius * np.sin(az),
            contact_plane_z + specs.footpad_radius_m))
        UsdGeom.Imageable(pad.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())

    root_prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(total_mass_kg)
    i_tilt, i_yaw = moment_of_inertia(total_mass_kg, specs.body_radius_m, specs.height_m)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(i_tilt, i_tilt, i_yaw))
    # Pin the centre of mass to the body origin. Without this PhysX derives
    # it from the collision shapes, and the shapes are no longer a single
    # origin-centred cylinder -- the pads would drag it down and silently
    # change every thrust/RCS moment arm in a model whose inertia tensor is
    # authored about the origin.
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))


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

    source_mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(visual_asset_path))
    target_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    unit_scale = source_mpu / target_mpu

    scale_vertical = unit_scale * specs.height_m / _RAW_MESH_HEIGHT_M
    scale_horizontal = unit_scale * specs.footpad_span_m / _RAW_MESH_FOOTPRINT_M

    # REAL BUG FOUND BY WATCHING A LANDING: the vehicle came to rest with
    # its footpads visibly in the air, balanced on nothing. The raw mesh is
    # modelled standing ON its own origin, not centred on it -- its lowest
    # point sits at +6.1 cm (`_RAW_MESH_BOTTOM_M`), so after scaling the
    # visual LM occupies z = +0.14 .. +7.18 m in the body frame while the
    # collision cylinder occupies z = -3.52 .. +3.52. The physics therefore
    # rested on a cylinder face 3.66 m BELOW the footpads anyone could see.
    # Shift the mesh down so its footpad plane coincides with the collision
    # cylinder's bottom, which is the surface that actually makes contact.
    #
    # Ops are authored translate-rotate-scale so the composed transform is
    # T * R * S (scale in the mesh's own frame, then the Y-up -> Z-up
    # rotation, then the shift in the vehicle's frame). USD applies them in
    # the order of `xformOpOrder`, so the translate has to be added FIRST.
    footpad_offset_m = specs.height_m * _RAW_MESH_BOTTOM_M / _RAW_MESH_HEIGHT_M
    visual.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -specs.height_m * 0.5 - footpad_offset_m))
    visual.AddRotateXOp().Set(90.0)  # source mesh is Y-up; this stage is Z-up
    visual.AddScaleOp().Set(Gf.Vec3f(scale_horizontal, scale_vertical, scale_horizontal))

    # -- physics proxy: mass + collision, independent of the visual mesh --
    #
    # REAL BUG FOUND BY WATCHING A LANDING SETTLE: the whole vehicle used to
    # collide as ONE `body_radius_m` (2.1 m) cylinder spanning the full
    # height, so it came to rest balanced on a 2.1 m disc rather than on its
    # 9.4 m leg base. Measured on the successful ZemZev capture
    # (`orbit_descent_controller_isaacsim_seed2001_terrainfix`): first
    # contact at t=44 s produced an angular-rate spike, after which the
    # vehicle ROCKED for ~10 s -- tilt swinging 14.9 -> 6.9 -> 10.4 -> 3.1
    # deg with the engine already off (throttle 0.005) and the reported
    # clearance stuck between 0.10 and 0.26 m, which free fall would have
    # closed in half a second. A 4.5x narrower support polygon than the real
    # gear is exactly the kind of thing that rings like that.
    #
    # Modelled now the way the vehicle is actually built: four footpads at
    # the real `footpad_span_m` stance carry the touchdown load, and the
    # descent-stage body is a shorter cylinder sitting ABOVE them so it only
    # participates if the vehicle bottoms out on something tall. Note the
    # landing-safety test already assumed this geometry -- it measures
    # terrain height spread under exactly these four points
    # (`_footpad_height_diff_m`) -- so the collider and the criterion only
    # now agree with each other.
    if not visual_only:
        author_physics(stage, prim_path, specs, total_mass_kg)

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
