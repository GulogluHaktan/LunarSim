"""Procedural regolith micro-bump texture: a small-scale (centimeter-to-decimeter)
fBm height field, independent of the terrain's own (meter-scale) heightmap,
encoded as a tangent-space normal map. This is what actually reads as
"grainy regolith" up close in a render -- the terrain mesh's own geometry
only carries detail down to its grid resolution (typically 0.1-1 m/px even
at `reference` quality), which is much coarser than real regolith grain/
texture. A meters-scale mesh alone renders as smooth/plastic no matter how
good the macro-shape (craters, slopes) is; this micro-bump layer is the
missing high-frequency detail.
"""
from __future__ import annotations

import numpy as np

from lunarsim.core.terrain.fbm import fbm_heightfield


def bake_regolith_normal_map(
    resolution: int = 1024,
    physical_size_m: float = 2.0,
    amplitude_m: float = 0.015,
    wavelength_m: float = 0.15,
    hurst: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Return an (resolution, resolution, 3) uint8 tangent-space normal map
    from a small synthetic fBm micro-height-field (independent RNG stream
    from the terrain's own noise -- this is a texture, not terrain geometry).

    `physical_size_m` is the real-world footprint this one texture tile
    covers when mapped at 1:1 texel density; the caller tiles it across the
    mesh via repeating UVs (see `terrain_uv_scale_m` in
    `adapters.isaac.heightfield.build_render_mesh`).
    """
    rng = np.random.default_rng(seed)
    height = fbm_heightfield(
        resolution,
        amplitude_m=amplitude_m,
        wavelength_m=wavelength_m,
        res_m=physical_size_m / resolution,
        hurst=hurst,
        rng=rng,
        octaves=5,
    )

    res_m = physical_size_m / resolution
    dzdy, dzdx = np.gradient(height, res_m)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    # tangent-space normal -> [0, 255] RGB (standard OpenGL-style normal map encoding)
    rgb = ((normal * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return rgb
