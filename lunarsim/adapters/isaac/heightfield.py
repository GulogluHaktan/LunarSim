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


def _mesh_from_heightfield(stage, prim_path: str, height: np.ndarray, res_m: float):
    from pxr import UsdGeom

    n = height.shape[0]
    ax = (np.arange(n) - (n - 1) / 2) * res_m
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    points = np.stack([xx, yy, height], axis=-1).reshape(-1, 3)

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
    return mesh


def add_heightfield_collision(stage, prim_path: str, tile: Tile):
    """Author an exact-triangle-mesh PhysX collider from `tile.height`.

    Uses the tile's native resolution directly (no extra decimation) --
    callers that want a cheaper collision mesh should downsample `tile`
    before calling this, since the collider intentionally matches whatever
    grid was passed in.
    """
    from pxr import Sdf, UsdPhysics

    mesh = _mesh_from_heightfield(stage, prim_path, tile.height, tile.res_m)

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("none")
    mesh.GetPrim().CreateAttribute("physxCollision:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

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


def build_render_mesh(stage, prim_path: str, tile: Tile, lod: int = 0):
    """Build a non-colliding `UsdGeom.Mesh` render surface from `tile.height`,
    at `res_m / (2**lod)` effective density via bilinear upsampling (finer
    than the collision mesh -- see module docstring).
    """
    height = tile.height
    res_m = tile.res_m
    if lod > 0:
        from scipy.ndimage import zoom

        height = zoom(height, 2**lod, order=1)
        res_m = tile.res_m / (2**lod)

    return _mesh_from_heightfield(stage, prim_path, height, res_m)
