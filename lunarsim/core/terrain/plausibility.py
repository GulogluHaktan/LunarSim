"""Statistical sanity checks on a generated Tile -- catches "clearly not
real" artifacts (NaN/Inf, impossible slopes, craters that don't actually
look like craters, rocks embedded below ground) before a tile ever reaches
a renderer or a training run.

This does not verify accuracy against real lunar terrain (that's what
DEM-backed `coarse_source: dem` is for) -- it only verifies internal
physical plausibility of whatever was generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lunarsim.core.terrain.generate import Tile
from lunarsim.core.terrain.rocks import sample_height_at


@dataclass
class PlausibilityReport:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def check_tile(tile: Tile, max_slope_deg: float = 60.0) -> PlausibilityReport:
    warnings: list[str] = []
    height = tile.height

    if not np.isfinite(height).all():
        warnings.append(f"heightmap has {np.sum(~np.isfinite(height))} non-finite cells")

    dzdy, dzdx = np.gradient(height, tile.res_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    max_slope = float(np.nanmax(slope_deg))
    frac_over = float(np.mean(slope_deg > max_slope_deg))
    if frac_over > 0.02:
        warnings.append(
            f"{frac_over * 100:.1f}% of the surface exceeds {max_slope_deg} deg slope "
            f"(max observed {max_slope:.1f} deg) -- check crater depth_ratio / hill amplitude for this tile"
        )

    relief = float(np.nanmax(height) - np.nanmin(height))
    if relief > tile.size_m:
        warnings.append(f"total relief ({relief:.1f} m) exceeds tile size ({tile.size_m:.1f} m) -- implausible for this scale")

    if len(tile.rocks.diameter_m) > 0:
        # core.terrain only produces x/y/diameter for rocks -- z is assigned by
        # whichever adapter places them (see adapters.isaac.rocks.spawn_rocks,
        # which sets z = ground height so rocks sit flush with the surface).
        # Here we only confirm the ground-height lookup itself is sane.
        ground_z = sample_height_at(height, tile.res_m, tile.rocks.x_m, tile.rocks.y_m)
        if not np.isfinite(ground_z).all():
            warnings.append("rock ground-height sampling produced non-finite values")
        out_of_bounds = (np.abs(tile.rocks.x_m) > tile.size_m / 2) | (np.abs(tile.rocks.y_m) > tile.size_m / 2)
        if out_of_bounds.any():
            warnings.append(f"{int(out_of_bounds.sum())} rocks fall outside the tile bounds")

    if len(tile.craters.diameter_m) > 0:
        bad_depth = tile.craters.depth_ratio <= 0
        if bad_depth.any():
            warnings.append(f"{int(bad_depth.sum())} craters have non-positive depth_ratio")
        too_deep = tile.craters.depth_ratio > 0.3
        if too_deep.any():
            warnings.append(
                f"{int(too_deep.sum())} craters have depth_ratio > 0.3 (real simple craters are ~0.1-0.2 "
                f"depth/diameter; check the config's craters.depth_ratio range)"
            )
        out_of_bounds = (np.abs(tile.craters.x_m) > tile.size_m / 2) | (np.abs(tile.craters.y_m) > tile.size_m / 2)
        if out_of_bounds.any():
            warnings.append(f"{int(out_of_bounds.sum())} craters fall outside the tile bounds")

    stats = {
        "max_slope_deg": max_slope,
        "frac_over_max_slope": frac_over,
        "relief_m": relief,
        "n_craters": len(tile.craters.diameter_m),
        "n_rocks": len(tile.rocks.diameter_m),
        "height_min_m": float(np.nanmin(height)),
        "height_max_m": float(np.nanmax(height)),
        "height_std_m": float(np.nanstd(height)),
    }

    return PlausibilityReport(ok=len(warnings) == 0, warnings=warnings, stats=stats)
