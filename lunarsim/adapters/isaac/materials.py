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

REAL BUG FIXED HERE: an earlier version of this function set a `"specular"`
Float input on the shader intending to zero out specular highlights --
that input doesn't exist on `UsdPreviewSurface` (it's not part of the
schema), so it was silently ignored and the shader kept its default
metallic-workflow Fresnel term (`ior=1.5`), which is exactly what produced
the glossy/plastic-looking blown-out highlight seen in early renders. Fixed
by switching to the specular workflow with `specularColor=(0,0,0)`, which
*is* a real, honored input and reliably kills the highlight.
"""
from __future__ import annotations


def create_regolith_material(
    stage,
    prim_path: str,
    albedo: float,
    brdf: str,
    roughness: float = 1.0,
    mdl_asset_path: str | None = None,
    mdl_subidentifier: str = "LunarRegolith",
    normal_map_path: str | None = None,
):
    """Author a `UsdPreviewSurface` material (or bind a real MDL shader if
    `mdl_asset_path` is given) under `prim_path`.

    `normal_map_path` (a PNG/PNG-like tangent-space normal map, e.g. from
    `core.lighting.regolith_texture.bake_regolith_normal_map`) adds
    real high-frequency micro-bump detail no terrain mesh alone can carry --
    without it, even a geometrically-accurate terrain mesh reads as smooth/
    plastic at any reasonable mesh resolution. Requires the mesh this
    material is bound to have `st` (UV) primvars, e.g.
    `heightfield.build_render_mesh(..., uv_tile_size_m=...)`.
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
    # Belt-and-suspenders against the glossy Fresnel highlight (see module
    # docstring): specular workflow with specularColor=0 zeroes the F0 term
    # in the schema's own model, and ior=1.0 additionally zeroes the
    # metallic-workflow Fresnel term in case the installed renderer doesn't
    # fully honor useSpecularWorkflow. A plain "specular" Float input does
    # NOT exist on this schema and silently no-ops -- do not use it.
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    if normal_map_path is not None:
        _bind_normal_map(stage, prim_path, material, shader, normal_map_path)

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


def _bind_normal_map(stage, prim_path: str, material, shader, normal_map_path: str):
    from pxr import Sdf, UsdShade

    st_reader = UsdShade.Shader.Define(stage, f"{prim_path}/StReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    normal_tex = UsdShade.Shader.Define(stage, f"{prim_path}/NormalTexture")
    normal_tex.CreateIdAttr("UsdUVTexture")
    normal_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_map_path)
    normal_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    normal_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    normal_tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    normal_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
    # normal maps store [0,1]-encoded vectors; scale/bias back to [-1,1]
    normal_tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
    normal_tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
    rgb_output = normal_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(rgb_output)
