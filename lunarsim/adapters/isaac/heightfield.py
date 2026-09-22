"""Heightfield collision + a separate, finer render mesh from a lunarsim Tile.

Isaac Sim has no dedicated "heightfield collider" API in the 6.0.1 line (this
was verified against a working prior Isaac Lab integration, see
lunarsim/adapters/isaac/README.md) -- collision is authored as an ordinary
`UsdGeom.Mesh` with `UsdPhysics.CollisionAPI` + `physxCollision:approximation
= "none"` (exact triangle mesh, not a convex hull -- required so the true
crater/slope shape is actually collided against, not a convex approximation
of it). The render mesh is a second, separate, denser mesh built at up to
`render_mesh_lod` extra subdivision steps so LiDAR/camera don't see
collision-grid aliasing (plan section 3 + section 8: "Render mesh, LiDAR
ışın izinden ve kaya boyutundan ince olmalı").

Requires a running Isaac Sim / Omniverse Kit process (imports `pxr`).
"""
from __future__ import annotations

import numpy as np

from lunarsim.core.terrain.generate import Tile


def _mesh_from_heightfield(stage, prim_path: str, height: np.ndarray, res_m: float, uv_tile_size_m: float | None = None,
                            center_x_m: float = 0.0, center_y_m: float = 0.0, z_offset_m: float = 0.0):
    from pxr import Sdf, UsdGeom

    n = height.shape[0]
    ax = (np.arange(n) - (n - 1) / 2) * res_m
    xx, yy = np.meshgrid(center_x_m + ax, center_y_m + ax, indexing="ij")
    points = np.stack([xx, yy, height + z_offset_m], axis=-1).reshape(-1, 3)

    face_counts = []
    face_indices = []
    for i in range(n - 1):
        for j in range(n - 1):
            p00 = i * n + j
            p10 = (i + 1) * n + j
            p01 = i * n + (j + 1)
            p11 = (i + 1) * n + (j + 1)
            face_counts += [3, 3]
            face_indices += [p00, p10, p11, p00, p11, p01]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)

    if uv_tile_size_m is not None:
        # world-space planar UVs, tiled every `uv_tile_size_m` meters so a
        # small micro-bump texture (see core.lighting.regolith_texture)
        # repeats at a consistent real-world scale regardless of tile size --
        # sampled with wrap="repeat" on the texture reader in materials.py.
        uv = np.stack([xx / uv_tile_size_m, yy / uv_tile_size_m], axis=-1).reshape(-1, 2)
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st_attr = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st_attr.Set(uv.tolist())

    return mesh


def add_heightfield_collision(stage, prim_path: str, tile: Tile, hide_from_render: bool = True):
    """Author an exact-triangle-mesh PhysX collider from `tile.height`.

    Uses the tile's native resolution directly (no extra decimation) --
    callers that want a cheaper collision mesh should downsample `tile`
    before calling this, since the collider intentionally matches whatever
    grid was passed in.

    REAL BUG FIXED HERE: this collider mesh and `build_render_mesh`'s output
    occupy nearly the same 3D space (same macro heightfield, different
    resolutions) -- when both were left visible, the renderer sees two
    overlapping, non-identical surfaces, which self-shadow/z-fight against
    each other and produced a persistent grid-aligned checkerboard/dither
    artifact in shadow transition zones. Reproduced identically under both
    real-time raster and path tracing, and unaffected by the render mesh's
    upsampling order (bilinear vs. cubic) -- ruling out a renderer-setting
    or geometry-smoothness cause and pointing at the two-overlapping-meshes
    setup itself. Fixed by hiding the collision mesh from the renderer
    (`UsdGeom.Imageable.MakeInvisible()`) by default -- it was only ever
    meant to be a physics collider, per plan section 3 ("Collision =
    heightfield, render mesh ayrı ve daha ince"). Pass `hide_from_render=False`
    only for debugging (e.g. to visually inspect the collider itself).
    """
    from pxr import Sdf, UsdGeom, UsdPhysics

    mesh = _mesh_from_heightfield(stage, prim_path, tile.height, tile.res_m)

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("none")
    mesh.GetPrim().CreateAttribute("physxCollision:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    if hide_from_render:
        UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()

    return mesh


def apply_regolith_physics_material(stage, prim_path: str, static_friction: float = 0.8, dynamic_friction: float = 0.6, restitution: float = 0.0):
    """Author a `UsdPhysics.MaterialAPI` physics material (friction/restitution,
    not to be confused with the visual/render material in `materials.py`) and
    return it so the caller can bind it to a collider prim via
    `UsdShade.MaterialBindingAPI(prim).Bind(mat, materialPurpose="physics")`.
    """
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, prim_path)
    physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_api.CreateStaticFrictionAttr(static_friction)
    physics_api.CreateDynamicFrictionAttr(dynamic_friction)
    physics_api.CreateRestitutionAttr(restitution)
    return material


def build_render_mesh(stage, prim_path: str, tile: Tile, lod: int = 0, uv_tile_size_m: float | None = 2.0):
    """Build a non-colliding `UsdGeom.Mesh` render surface from `tile.height`,
    at `res_m / (2**lod)` effective density via cubic upsampling (finer
    than the collision mesh -- see module docstring).

    REAL BUG FIXED HERE: this used to upsample with `order=1` (bilinear).
    Bilinear upsampling of a heightfield produces a *piecewise-bilinear*
    surface -- locally near-planar micro-facets aligned to the original
    coarse grid. At grazing sun angles that surface self-shadows across
    those facet boundaries, producing a visible grid-aligned checkerboard/
    dither pattern in the shadow transition zones -- confirmed to be a
    geometry problem, not a renderer setting, by reproducing it identically
    under both real-time raster AND path tracing. `order=3` (cubic) keeps
    the surface curvature-continuous across the original grid, removing the
    facet boundaries that caused it.

    `uv_tile_size_m` (default 2 m) sets world-space planar UV tiling so a
    micro-bump normal map (see `core.lighting.regolith_texture` +
    `materials.create_regolith_material(..., normal_map_path=...)`) repeats
    at a consistent real-world scale; pass `None` to skip UV authoring.
    """
    height = tile.height
    res_m = tile.res_m
    if lod > 0:
        from scipy.ndimage import zoom

        height = zoom(height, 2**lod, order=3)
        res_m = tile.res_m / (2**lod)

    return _mesh_from_heightfield(stage, prim_path, height, res_m, uv_tile_size_m=uv_tile_size_m)


def build_hires_patch_mesh(stage, prim_path: str, patch, uv_tile_size_m: float | None = 2.0, z_offset_m: float = 0.003):
    """Build a render-only mesh from a `core.terrain.generate_hires_patch`
    result (down to ~2.5 cm/px around one point of interest -- see that
    function's docstring for why this is a separate small patch rather
    than the whole tile at that resolution).

    `z_offset_m` lifts the patch a few mm above the coarser parent render
    mesh underneath it -- the patch's low-frequency shape is resampled
    from that same parent surface so the two are almost, but not exactly,
    coincident (bicubic resampling vs. the parent's own vertices); a tiny
    lift avoids z-fighting between the two without being visible at any
    normal viewing distance.
    """
    return _mesh_from_heightfield(
        stage, prim_path, patch.height, patch.res_m, uv_tile_size_m=uv_tile_size_m,
        center_x_m=patch.center_x_m, center_y_m=patch.center_y_m, z_offset_m=z_offset_m,
    )
