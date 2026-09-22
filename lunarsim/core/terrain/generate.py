"""Tile generation: coarse elevation (real DEM or procedural fBm) + procedural
craters/detail, coarse/fine/blend mode selection.

h = coarse elevation (DEM or fBm hills) + craters + high-frequency detail (fine only)
    [+ curvature, procedural coarse source only -- a real DEM is already sphere-relative]

Craters and rocks below DEM resolution aren't resolved by any real raster we
load, so they're always added as a synthetic augmentation layer on top of
whichever coarse source is in use. This mirrors standard practice for
simulation-realism datasets built on real DEMs; it is never meant to imply
those specific craters/rocks were surveyed at that real site.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunarsim import __version__
from lunarsim.core.terrain.blend import RoiSpec, blend, roi_weight
from lunarsim.core.terrain.config import SampledParams, TerrainConfig, sample_params
from lunarsim.core.terrain.craters import CraterField, rasterize_craters, sample_crater_field
from lunarsim.core.terrain.curvature import add_curvature
from lunarsim.core.terrain.fbm import fbm_heightfield
from lunarsim.core.terrain.rocks import RockField, sample_rock_field


@dataclass
class Tile:
    height: np.ndarray  # (n, n) meters
    res_m: float
    size_m: float
    seed: int
    craters: CraterField
    rocks: RockField
    params: SampledParams
    mode: str
    coarse_source: str


def _coarse_elevation(cfg: TerrainConfig, n: int, res_m: float, hills: dict, rng: np.random.Generator) -> np.ndarray:
    if cfg.coarse_source == "dem":
        if not cfg.dem_path:
            raise ValueError(
                "coarse_source='dem' requires terrain.dem_path pointing at a LOLA/LRO "
                "GeoTIFF (see lunarsim/core/terrain/dem.py for supported products)."
            )
        from lunarsim.core.terrain.dem import DemSource

        dem = DemSource(cfg.dem_path)
        return dem.read_tile(cfg.site_lat_deg, cfg.site_lon_deg, n * res_m, res_m)

    if cfg.coarse_source == "procedural":
        return fbm_heightfield(
            n,
            amplitude_m=hills.get("amplitude_m", 1.0),
            wavelength_m=hills.get("wavelength_m", 100.0),
            res_m=res_m,
            hurst=hills.get("hurst", 0.75),
            rng=rng,
        )

    raise ValueError(f"unknown coarse_source: {cfg.coarse_source!r} (expected 'dem' or 'procedural')")


def _add_craters(height: np.ndarray, res_m: float, size_m: float, params: SampledParams, rng: np.random.Generator) -> tuple[np.ndarray, CraterField]:
    craters = params.craters
    field = sample_crater_field(
        size_m=size_m,
        d_min_m=craters["d_min_m"],
        d_max_m=craters["d_max_m"],
        b=craters["b"],
        count_scale=craters["count_scale"],
        depth_ratio_range=craters["depth_ratio_range"],
        age_range=craters["age_range"],
        rng=rng,
    )
    return rasterize_craters(height, res_m, field), field


def generate_tile(cfg: TerrainConfig) -> Tile:
    rng = np.random.default_rng(cfg.seed)
    params = sample_params(cfg, rng)

    n_fine = int(round(cfg.size_m / cfg.res_m))
    apply_curvature = cfg.curvature and cfg.coarse_source == "procedural"

    if cfg.mode == "coarse":
        n_coarse = int(round(cfg.size_m / cfg.coarse_res_m))
        coarse_h = _coarse_elevation(cfg, n_coarse, cfg.coarse_res_m, params.hills, rng)
        coarse_h, field = _add_craters(coarse_h, cfg.coarse_res_m, cfg.size_m, params, rng)
        from scipy.ndimage import zoom

        height = zoom(coarse_h, n_fine / n_coarse, order=3)[:n_fine, :n_fine]

    elif cfg.mode == "fine":
        fine_h = _coarse_elevation(cfg, n_fine, cfg.res_m, params.hills, rng)
        height, field = _add_craters(fine_h, cfg.res_m, cfg.size_m, params, rng)

    elif cfg.mode == "blend":
        n_coarse = int(round(cfg.size_m / cfg.coarse_res_m))
        coarse_h = _coarse_elevation(cfg, n_coarse, cfg.coarse_res_m, params.hills, rng)

        fine_h = fbm_heightfield(
            n_fine,
            amplitude_m=params.hills.get("amplitude_m", 1.0) * 0.15,  # fine layer only contributes high-freq detail
            wavelength_m=params.hills.get("wavelength_m", 100.0),
            res_m=cfg.res_m,
            hurst=params.hills.get("hurst", 0.75),
            rng=rng,
        )
        fine_h, field = _add_craters(fine_h, cfg.res_m, cfg.size_m, params, rng)

        roi = params.roi
        regions = roi.get("regions", None)
        if regions:
            # explicit per-region detail control: world-meter (x_m, y_m) center,
            # each region's own sigma_m (falloff radius) and weight (peak detail
            # strength, 1.0 = full fine detail) -- this is how to make one area
            # noticeably more detailed than the rest of the tile, or give several
            # areas different detail levels from each other in the same tile.
            px_center = (n_fine - 1) / 2
            roi_specs = [
                RoiSpec(
                    center_px=(r["y_m"] / cfg.res_m + px_center, r["x_m"] / cfg.res_m + px_center),
                    sigma_m=r.get("sigma_m", 50.0),
                    weight=r.get("weight", 1.0),
                )
                for r in regions
            ]
            w = roi_weight(n_fine, cfg.res_m, roi_specs)
        else:
            centers = roi.get("centers", None)
            if centers == "random_16" or centers is None:
                n_centers = 16 if centers == "random_16" else 1
                cy = rng.uniform(0, n_fine, n_centers)
                cx = rng.uniform(0, n_fine, n_centers)
                centers_px = list(zip(cy, cx))
            else:
                centers_px = [(n_fine / 2, n_fine / 2)]

            w = roi_weight(n_fine, cfg.res_m, centers_px, roi.get("sigma_m", 200.0))
        height = blend(coarse_h, fine_h, cfg.res_m, cfg.split_m, weight=w)

    else:
        raise ValueError(f"unknown terrain mode: {cfg.mode}")

    if apply_curvature:
        height = add_curvature(height, cfg.res_m)

    rock_field = sample_rock_field(
        size_m=cfg.size_m,
        density_scale=params.rocks["density_scale"],
        d_max_m=params.rocks["d_max_m"],
        rng=rng,
    )

    return Tile(
        height=height,
        res_m=cfg.res_m,
        size_m=cfg.size_m,
        seed=cfg.seed,
        craters=field,
        rocks=rock_field,
        params=params,
        mode=cfg.mode,
        coarse_source=cfg.coarse_source,
    )


@dataclass
class HiresPatch:
    height: np.ndarray  # (n, n) meters, world-aligned to the parent tile
    res_m: float
    center_x_m: float
    center_y_m: float
    size_m: float


def generate_hires_patch(
    tile: Tile,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    size_m: float = 40.0,
    res_m: float = 0.025,
    micro_amplitude_m: float = 0.03,
    micro_wavelength_m: float = 1.2,
    seed_offset: int = 9001,
) -> HiresPatch:
    """A small, OmniLRS-resolution-class (down to ~2.5 cm/px) render-only
    patch around one point of interest -- e.g. where a rover sits or a
    camera orbits -- instead of the whole tile, which is the only way to
    reach that resolution without an intractable grid (a 150 m tile at
    2.5 cm/px would be 6000x6000; a 40 m patch is a manageable 1600x1600).

    The patch's low-frequency shape comes from bicubic-resampling the
    parent tile's own (coarse+fine, already-generated) heightfield, so it
    stays continuous with the surrounding terrain -- then a genuinely new,
    even-finer fBm layer is added on top (`micro_amplitude_m`/
    `micro_wavelength_m`), injecting real sub-25cm surface texture that
    was never present at the parent tile's native resolution at all.
    """
    from scipy.ndimage import map_coordinates

    n = int(round(size_m / res_m))
    ax = (np.arange(n) - (n - 1) / 2) * res_m
    xs = center_x_m + ax
    ys = center_y_m + ax
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    n_parent = tile.height.shape[0]
    px = xx / tile.res_m + (n_parent - 1) / 2
    py = yy / tile.res_m + (n_parent - 1) / 2
    base = map_coordinates(tile.height, [px, py], order=3, mode="nearest")

    rng = np.random.default_rng(tile.seed + seed_offset)
    micro = fbm_heightfield(
        n, amplitude_m=micro_amplitude_m, wavelength_m=micro_wavelength_m,
        res_m=res_m, hurst=0.8, rng=rng,
    )

    return HiresPatch(
        height=base + micro, res_m=res_m,
        center_x_m=center_x_m, center_y_m=center_y_m, size_m=size_m,
    )


def save_tile(tile: Tile, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "heightmap.npy", tile.height.astype(np.float32))

    ground_truth = {
        "craters": {
            "x_m": tile.craters.x_m.tolist(),
            "y_m": tile.craters.y_m.tolist(),
            "diameter_m": tile.craters.diameter_m.tolist(),
            "depth_ratio": tile.craters.depth_ratio.tolist(),
            "age": tile.craters.age.tolist(),
        },
        "rocks": {
            "x_m": tile.rocks.x_m.tolist(),
            "y_m": tile.rocks.y_m.tolist(),
            "diameter_m": tile.rocks.diameter_m.tolist(),
        },
    }
    with open(out / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f)

    meta = {
        "version": __version__,
        "seed": tile.seed,
        "mode": tile.mode,
        "coarse_source": tile.coarse_source,
        "size_m": tile.size_m,
        "res_m": tile.res_m,
        "shape": list(tile.height.shape),
        "params": {
            "hills": tile.params.hills,
            "craters": tile.params.craters,
            "rocks": tile.params.rocks,
            "roi": tile.params.roi,
        },
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
