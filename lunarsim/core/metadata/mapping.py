"""Incremental terrain mapping from a sequence of LiDAR scans.

A descending lander does not get one good look at its landing site; it gets
a few hundred partial, oblique, range-limited looks on the way down. This
module is the thing that turns that sequence into a single artifact: a
world-frame grid that every scan is folded into, plus the derived layers a
real hazard-detection-and-avoidance (HDA) system reads off such a map --
elevation, coverage, within-cell relief, slope, and a landability mask.

The landability mask deliberately uses the SAME criterion the simulator's
own `landed_safely` test uses (`_footpad_height_diff_m` in
`adapters.isaac.isaac_lander_env`: the spread of ground height under the
four deployed footpads, against
`safe_landing_max_leg_height_diff_m` = 0.16 m), evaluated here from
LiDAR-derived elevation instead of from the ground-truth heightfield. That
makes the map answer an operational question -- "where, of what the sensor
has actually seen so far, could this vehicle put its legs down?" -- and it
makes the map checkable: run it over scans of a known tile and the mask
should agree with the heightfield's own answer.

Everything here is pure numpy: no Isaac, no renderer, no plotting. Scans
arrive as world-frame (n, 3) hit points, whatever produced them.

Accumulation is done in sum/sum-of-squares/min/max form rather than by
keeping every point, so memory is fixed by the grid size, not by how long
the descent lasted (a 253-scan capture at ~20k hits/scan is ~5M points --
a per-cell point list is both far larger and unnecessary for every layer
below).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Footpad-span diagonal of the Apollo LM with the gear deployed, and the
# leg-height-difference limit the landing test applies across it. Duplicated
# as defaults here rather than imported from `core.vehicle.apollo_lm` /
# `rl.analytic_lander_env` so this module stays dependency-free; both are
# constructor arguments, so a caller with the real specs should pass them.
FOOTPAD_SPAN_M = 9.4
SAFE_LEG_HEIGHT_DIFF_M = 0.16


@dataclass(frozen=True)
class MapGrid:
    """A square, world-axis-aligned, cell-centered grid covering
    `[-half_extent_m, +half_extent_m]` in both x and y.

    Centered on the world origin because that is where every tile in this
    codebase is centered (`core.terrain.generate.generate_tile` puts the
    tile center at the origin), so a map built in these coordinates can be
    compared cell-for-cell against the ground-truth heightfield.
    """

    cell_m: float
    half_extent_m: float

    def __post_init__(self) -> None:
        if self.cell_m <= 0:
            raise ValueError(f"cell_m must be positive, got {self.cell_m}")
        if self.half_extent_m <= 0:
            raise ValueError(f"half_extent_m must be positive, got {self.half_extent_m}")
        if self.n < 2:
            raise ValueError(
                f"grid too coarse: cell_m={self.cell_m} over half_extent_m="
                f"{self.half_extent_m} gives only {self.n} cells per side"
            )

    @property
    def n(self) -> int:
        """Cells per side."""
        return int(round(2.0 * self.half_extent_m / self.cell_m))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """`(x_min, x_max, y_min, y_max)`, ready for `imshow(extent=...)`."""
        h = self.half_extent_m
        return (-h, h, -h, h)

    def index_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World x/y -> `(ix, iy, inside)` column/row indices plus an
        in-bounds mask. Indices are only meaningful where `inside`."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # non-finite coordinates are forced to a definitely-outside index
        # before the integer cast: casting NaN/inf to int64 is undefined
        # (it warns and yields a garbage index that could land inside the
        # grid), and such a point must read as "not on the map", not as a
        # hit on whatever cell the garbage happened to name.
        finite = np.isfinite(x) & np.isfinite(y)
        sentinel = -1.0 - self.half_extent_m
        xs = np.where(finite, x, sentinel)
        ys = np.where(finite, y, sentinel)
        ix = np.floor((xs + self.half_extent_m) / self.cell_m).astype(np.int64)
        iy = np.floor((ys + self.half_extent_m) / self.cell_m).astype(np.int64)
        inside = finite & (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        return ix, iy, inside

    def cell_centers_1d(self) -> np.ndarray:
        """The `n` cell-center coordinates along one axis."""
        return -self.half_extent_m + (np.arange(self.n) + 0.5) * self.cell_m


def _bilinear_sample(values: np.ndarray, grid: MapGrid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a cell-centered `(n, n)` layer (indexed
    `[row=y, col=x]`) at world `x`/`y`.

    Returns NaN where the point falls outside the grid or where any of the
    four contributing cells is NaN -- "unobserved" must not silently become
    "interpolated from one neighbour", since every layer derived from this
    feeds a landability decision.
    """
    n = grid.n
    fx = (np.asarray(x, dtype=float) + grid.half_extent_m) / grid.cell_m - 0.5
    fy = (np.asarray(y, dtype=float) + grid.half_extent_m) / grid.cell_m - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = fx - x0
    ty = fy - y0

    out = np.full(np.broadcast(fx, fy).shape, np.nan)
    ok = (x0 >= 0) & (x0 + 1 < n) & (y0 >= 0) & (y0 + 1 < n)
    if not np.any(ok):
        return out

    xi, yi = x0[ok], y0[ok]
    txi, tyi = tx[ok], ty[ok]
    v00 = values[yi, xi]
    v01 = values[yi, xi + 1]
    v10 = values[yi + 1, xi]
    v11 = values[yi + 1, xi + 1]
    out[ok] = (
        v00 * (1 - txi) * (1 - tyi)
        + v01 * txi * (1 - tyi)
        + v10 * (1 - txi) * tyi
        + v11 * txi * tyi
    )
    return out


class TerrainMap:
    """A grid that successive world-frame LiDAR scans are accumulated into.

    Layer accessors return `(n, n)` float arrays indexed `[row=y, col=x]`
    (so `imshow(..., origin="lower", extent=grid.extent)` draws them in
    world orientation), NaN where nothing has been observed yet.
    """

    def __init__(self, grid: MapGrid) -> None:
        self.grid = grid
        n = grid.n
        self._count = np.zeros((n, n), dtype=np.int64)
        self._z_sum = np.zeros((n, n), dtype=float)
        self._z_sq_sum = np.zeros((n, n), dtype=float)
        self._z_min = np.full((n, n), np.inf)
        self._z_max = np.full((n, n), -np.inf)
        self._intensity_sum = np.zeros((n, n), dtype=float)
        self._last_scan = np.full((n, n), -1, dtype=np.int64)
        self.n_scans = 0
        self.n_points_integrated = 0
        self.n_points_dropped = 0

    # ------------------------------------------------------------------
    # accumulation
    # ------------------------------------------------------------------
    def integrate(
        self,
        points_m: np.ndarray,
        intensity: np.ndarray | None = None,
        scan_index: int | None = None,
    ) -> int:
        """Fold one scan's world-frame hit points into the map.

        `points_m` is `(n_hits, 3)`. Points outside the grid, and any
        non-finite point, are counted into `n_points_dropped` and discarded
        rather than clamped to the edge -- clamping would pile a whole
        off-map horizon into the boundary cells and corrupt exactly the
        cells a descending vehicle is about to fly over. Returns the number
        of points integrated.
        """
        points_m = np.asarray(points_m, dtype=float)
        if points_m.size == 0:
            self.n_scans += 1
            return 0
        if points_m.ndim != 2 or points_m.shape[1] != 3:
            raise ValueError(f"points_m must be (n, 3), got {points_m.shape}")

        if intensity is None:
            intensity = np.zeros(len(points_m))
        else:
            intensity = np.asarray(intensity, dtype=float).ravel()
            if len(intensity) != len(points_m):
                raise ValueError(
                    f"intensity has {len(intensity)} entries for {len(points_m)} points"
                )

        finite = np.isfinite(points_m).all(axis=1) & np.isfinite(intensity)
        ix, iy, inside = self.grid.index_of(points_m[:, 0], points_m[:, 1])
        keep = finite & inside
        self.n_points_dropped += int((~keep).sum())
        if not np.any(keep):
            self.n_scans += 1
            return 0

        ix, iy = ix[keep], iy[keep]
        z = points_m[keep, 2]
        inten = intensity[keep]
        n = self.grid.n
        flat = iy * n + ix

        # bincount over flattened cell ids: one pass per statistic, rather
        # than a Python loop over hits (a single capture is millions of
        # points, so the loop version is minutes instead of milliseconds).
        size = n * n
        delta_count = np.bincount(flat, minlength=size).reshape(n, n)
        self._count += delta_count
        self._z_sum += np.bincount(flat, weights=z, minlength=size).reshape(n, n)
        self._z_sq_sum += np.bincount(flat, weights=z * z, minlength=size).reshape(n, n)
        self._intensity_sum += np.bincount(flat, weights=inten, minlength=size).reshape(n, n)
        # indexed with a (row, col) tuple rather than against a flattened
        # view: `reshape(-1)` only aliases the original storage while the
        # array stays C-contiguous, so an in-place `ufunc.at` through it
        # would silently write to a throwaway copy if that ever changed.
        np.minimum.at(self._z_min, (iy, ix), z)
        np.maximum.at(self._z_max, (iy, ix), z)

        scan_id = self.n_scans if scan_index is None else int(scan_index)
        self._last_scan[delta_count > 0] = scan_id

        self.n_scans += 1
        self.n_points_integrated += int(len(z))
        return int(len(z))

    # ------------------------------------------------------------------
    # layers
    # ------------------------------------------------------------------
    @property
    def observed(self) -> np.ndarray:
        """Boolean mask of cells with at least one hit."""
        return self._count > 0

    def coverage(self) -> np.ndarray:
        """Hits per cell (integer, 0 where unobserved)."""
        return self._count.copy()

    def elevation_m(self) -> np.ndarray:
        """Mean hit height per cell -- the map's primary layer."""
        out = np.full(self._count.shape, np.nan)
        seen = self.observed
        out[seen] = self._z_sum[seen] / self._count[seen]
        return out

    def relief_m(self) -> np.ndarray:
        """Within-cell height spread (`max - min`), NaN where fewer than two
        hits. This is sub-cell structure -- a boulder or a crater rim that
        the cell's mean elevation averages away."""
        out = np.full(self._count.shape, np.nan)
        enough = self._count >= 2
        out[enough] = self._z_max[enough] - self._z_min[enough]
        return out

    def roughness_m(self) -> np.ndarray:
        """Per-cell standard deviation of hit heights, NaN where fewer than
        two hits. Less outlier-driven than `relief_m`."""
        out = np.full(self._count.shape, np.nan)
        enough = self._count >= 2
        c = self._count[enough]
        mean = self._z_sum[enough] / c
        var = np.maximum(self._z_sq_sum[enough] / c - mean * mean, 0.0)
        out[enough] = np.sqrt(var)
        return out

    def intensity(self) -> np.ndarray:
        """Mean return intensity per cell, NaN where unobserved."""
        out = np.full(self._count.shape, np.nan)
        seen = self.observed
        out[seen] = self._intensity_sum[seen] / self._count[seen]
        return out

    def last_seen_scan(self) -> np.ndarray:
        """Index of the most recent scan that hit each cell, -1 if never."""
        return self._last_scan.copy()

    def slope_deg(self) -> np.ndarray:
        """Surface slope from the elevation layer's gradient, in degrees.

        NaN wherever the central difference would have to cross an
        unobserved cell, so a slope number always rests on real
        observations on both sides.
        """
        elev = self.elevation_m()
        dz_dy, dz_dx = np.gradient(elev, self.grid.cell_m)
        return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    def footpad_height_diff_m(self, footpad_span_m: float = FOOTPAD_SPAN_M) -> np.ndarray:
        """For every cell, the spread of elevation under the four deployed
        footpads if the vehicle set down centered on that cell.

        Same construction as the simulator's `_footpad_height_diff_m`: four
        samples on a circle of radius `footpad_span_m / 2` at 0/90/180/270
        degrees, `max - min`. Sampled bilinearly from the elevation layer,
        so the radius is the real radius rather than one snapped to the
        cell size. NaN where any of the four samples is unobserved.
        """
        radius = footpad_span_m / 2.0
        elev = self.elevation_m()
        gx, gy = np.meshgrid(self.grid.cell_centers_1d(), self.grid.cell_centers_1d())
        samples = [
            _bilinear_sample(elev, self.grid, gx + radius * np.cos(a), gy + radius * np.sin(a))
            for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])
        ]
        stack = np.stack(samples, axis=0)
        # all four footpads must be observed: a spread taken over only the
        # two or three that happen to be mapped UNDERSTATES the real step
        # the gear would meet, which would mark unseen rough ground
        # landable -- the one error this layer must not make.
        valid = np.isfinite(stack).all(axis=0)
        out = np.full(stack.shape[1:], np.nan)
        out[valid] = stack[:, valid].max(axis=0) - stack[:, valid].min(axis=0)
        return out

    def landable_mask(
        self,
        max_leg_height_diff_m: float = SAFE_LEG_HEIGHT_DIFF_M,
        max_slope_deg: float = 15.0,
        min_hits: int = 3,
        footpad_span_m: float = FOOTPAD_SPAN_M,
    ) -> np.ndarray:
        """Cells the map says the vehicle could land on.

        A cell is landable when it has been seen at least `min_hits` times,
        its footpad-span height difference is within
        `max_leg_height_diff_m`, and its slope is within `max_slope_deg`
        (the default matches `safe_landing_tilt_rad` = 15 deg -- a vehicle
        that lands level on a steeper slope exceeds the tilt limit by the
        slope alone).

        Unobserved is NOT landable: the mask is "known good", not "not
        known bad", which is the only reading that is safe to act on.
        """
        diff = self.footpad_height_diff_m(footpad_span_m)
        slope = self.slope_deg()
        with np.errstate(invalid="ignore"):
            return (
                (self._count >= min_hits)
                & np.isfinite(diff)
                & np.isfinite(slope)
                & (diff <= max_leg_height_diff_m)
                & (slope <= max_slope_deg)
            )

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Scalar map statistics, for logging or an on-screen readout."""
        seen = self.observed
        n_seen = int(seen.sum())
        elev = self.elevation_m()
        landable = self.landable_mask()
        cell_area = self.grid.cell_m ** 2
        out = {
            "n_scans": self.n_scans,
            "n_points": self.n_points_integrated,
            "n_points_dropped": self.n_points_dropped,
            "cells_total": int(self._count.size),
            "cells_observed": n_seen,
            "coverage_frac": n_seen / self._count.size,
            "mapped_area_m2": n_seen * cell_area,
            "cells_landable": int(landable.sum()),
            "landable_area_m2": float(landable.sum()) * cell_area,
        }
        if n_seen:
            out["elevation_min_m"] = float(np.nanmin(elev))
            out["elevation_max_m"] = float(np.nanmax(elev))
            out["elevation_p50_m"] = float(np.nanmedian(elev))
        return out
