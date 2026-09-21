"""Individual rock prims from a `RockField`, flush with the tile's terrain
height and scaled per-rock diameter.

Each rock is spawned as its own prim referencing a prototype rock USD asset
(caller-supplied, e.g. a small library of pre-modeled rock meshes) with its
own transform, rather than a `UsdGeom.PointInstancer`. A prior working Isaac
integration on this same task moved away from PointInstancer specifically
because prototype-based physics colliders under it were unreliable
(instance-level collision didn't consistently register); per-prim
references pay a higher prim count for correct, per-rock collision instead.
For counts above a few thousand where prim overhead starts to matter, this
is a known cost -- `cap_rock_count` (below) is the mitigation the quality
profile's `rocks.max_count` maps to.
"""
from __future__ import annotations

import numpy as np

from lunarsim.core.terrain.generate import Tile
from lunarsim.core.terrain.rocks import sample_height_at


def spawn_rocks(
    stage,
    parent_prim_path: str,
    tile: Tile,
    prototype_usd_paths: list[str],
    rng: np.random.Generator,
    add_collision: bool = True,
):
    """Reference `tile.rocks` onto the stage under `parent_prim_path/rock_{i}`,
    each an instance of a randomly chosen prototype in `prototype_usd_paths`,
    positioned at (x, y, local terrain height) and scaled to the sampled
    diameter (prototypes are assumed unit-diameter along their longest axis).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    z = sample_height_at(tile.height, tile.res_m, tile.rocks.x_m, tile.rocks.y_m)
    proto_choice = rng.integers(0, len(prototype_usd_paths), size=len(tile.rocks.x_m))

    prims = []
    for i in range(len(tile.rocks.x_m)):
        prim_path = f"{parent_prim_path}/rock_{i}"
        # No explicit typeName here: USD type-resolution is local-strongest,
        # so authoring "Xform" before the reference would shadow the
        # referenced asset's real type (e.g. Mesh/Sphere) and leave the prim
        # with no visible/collidable geometry despite the reference existing.
        prim = stage.DefinePrim(prim_path)
        prim.GetReferences().AddReference(prototype_usd_paths[proto_choice[i]])

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(tile.rocks.x_m[i]), float(tile.rocks.y_m[i]), float(z[i])))
        scale = float(tile.rocks.diameter_m[i])
        xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

        if add_collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")

        prims.append(prim)

    return prims


def cap_rock_count(tile: Tile, max_count: int, rng: np.random.Generator) -> Tile:
    """Subsample `tile.rocks` down to `max_count` (largest rocks kept
    preferentially) to respect a quality profile's `rocks.max_count`."""
    n = len(tile.rocks.diameter_m)
    if n <= max_count:
        return tile

    order = np.argsort(-tile.rocks.diameter_m)
    keep = order[:max_count]
    rng.shuffle(keep)  # avoid a purely size-sorted spatial bias

    from dataclasses import replace

    from lunarsim.core.terrain.rocks import RockField

    capped = RockField(
        x_m=tile.rocks.x_m[keep],
        y_m=tile.rocks.y_m[keep],
        diameter_m=tile.rocks.diameter_m[keep],
    )
    return replace(tile, rocks=capped)
