"""Regolith visual material authoring: maps `core.lighting.materials` BRDF
choice to a USD material.

`UsdPreviewSurface` is used as the reliable, renderer-agnostic fallback
(works under both the raster and RTX/path-trace render modes without extra
setup) -- verified pattern from a prior working Isaac integration. When a
real regolith MDL asset is available (e.g. a Hapke-like shader such as the
one shipped by NVIDIA's OmniLRS lunar-surface project), pass its path via
`mdl_asset_path` to bind that instead; no such shader is bundled with
lunarsim itself.

Note this is strictly the *visual* material. Collider friction/restitution
is a separate `UsdPhysics.MaterialAPI` material -- see
`heightfield.apply_regolith_physics_material`.

`albedo`/`lommel_seeliger` degrade to a plain diffuse-albedo/roughness
approximation; this is a visual approximation, not a physical match (in
particular Lommel-Seeliger's limb-darkening isn't reproducible by
UsdPreviewSurface's roughness model) -- use
`core.lighting.materials.reflectance` directly for any quantitative work,
e.g. camera radiometric ground truth.
"""
from __future__ import annotations


def create_regolith_material(
    stage,
    prim_path: str,
    albedo: float,
    brdf: str,
    roughness: float = 0.95,
    mdl_asset_path: str | None = None,
    mdl_subidentifier: str = "LunarRegolith",
):
    """Author a `UsdPreviewSurface` material (or bind a real MDL shader if
    `mdl_asset_path` is given) under `prim_path`.
    """
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, prim_path)

    if mdl_asset_path is not None:
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/MdlShader")
        shader.CreateIdAttr(mdl_subidentifier)
        shader.SetSourceAsset(mdl_asset_path, "mdl")
        shader.SetSourceAssetSubIdentifier(mdl_subidentifier, "mdl")
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        return material

    shader = UsdShade.Shader.Define(stage, f"{prim_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((albedo, albedo, albedo))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if brdf in ("hapke_approx", "hapke"):
        import warnings

        warnings.warn(
            "hapke/hapke_approx BRDF requested but no MDL shader path was given -- "
            "falling back to a flat UsdPreviewSurface approximation (no limb-darkening "
            "or opposition surge). Use core.lighting.materials.reflectance() directly "
            "for physically accurate radiometry instead of relying on the render.",
            stacklevel=2,
        )

    return material
