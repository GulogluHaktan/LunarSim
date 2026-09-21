"""Real DEM (LOLA/LRO) ingestion: crop + resample a georeferenced lunar elevation
raster into a local (size_m x size_m, res_m/px) tile centered on a selenographic
(lat, lon) site.

Supported sources (anything GDAL/rasterio can open with a lunar CRS):
- LOLA global GDR mosaics (118 ppd / 512 ppd), equirectangular, PDS Geosciences Node
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/
- LOLA+Kaguya TC merged DEM (59 m/px), USGS Astropedia
  https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m
- LRO NAC DTMs (0.5-5 m/px, local sites only), PDS Cartography and Imaging
  Sciences Node / USGS Astropedia per-site products

Datum note: LOLA/NAC DEM pixel values are typically "radius" (distance from
the Moon's center of mass) or already-reduced "elevation relative to a
1737.4 km reference sphere". This loader assumes the latter (elevation, not
raw radius) — if you're loading a raw *radius* product, subtract
`lunarsim.core.terrain.curvature.MOON_RADIUS_M` from it first, and make sure
you know which reference radius your specific product used. Because the
elevation is already sphere-relative, a small local window cut from it
already encodes the sphere curvature falloff — do NOT call
`add_curvature()` again on a DEM-sourced tile (see `generate.py`,
`coarse_source="dem"`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarsim.core.terrain.curvature import MOON_RADIUS_M


@dataclass
class DemSource:
    path: str

    def __post_init__(self):
        import rasterio

        self._rasterio = rasterio
        self._ds = rasterio.open(self.path)

    @property
    def crs(self):
        return self._ds.crs

    @property
    def is_geographic(self) -> bool:
        return self._ds.crs is not None and self._ds.crs.is_geographic

    def read_tile(
        self,
        center_lat_deg: float,
        center_lon_deg: float,
        size_m: float,
        res_m: float,
        radius_m: float = MOON_RADIUS_M,
    ) -> np.ndarray:
        """Crop + resample to an (n, n) heightmap, n = round(size_m / res_m), in meters."""
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        n = int(round(size_m / res_m))

        if self.is_geographic:
            cx, cy = center_lon_deg, center_lat_deg
            half_deg_lat = np.rad2deg((size_m / 2) / radius_m)
            half_deg_lon = half_deg_lat / max(np.cos(np.deg2rad(center_lat_deg)), 1e-6)
            left, right = cx - half_deg_lon, cx + half_deg_lon
            bottom, top = cy - half_deg_lat, cy + half_deg_lat
        else:
            cx, cy = self._lonlat_to_dem_xy(center_lon_deg, center_lat_deg)
            left, right = cx - size_m / 2, cx + size_m / 2
            bottom, top = cy - size_m / 2, cy + size_m / 2

        window = from_bounds(left, bottom, right, top, transform=self._ds.transform)
        data = self._ds.read(
            1,
            window=window,
            out_shape=(n, n),
            resampling=Resampling.bilinear,
        ).astype(np.float64)

        if self._ds.nodata is not None:
            data = np.where(data == self._ds.nodata, np.nan, data)
            if np.isnan(data).any():
                data = _fill_nodata(data)

        return data

    def _lonlat_to_dem_xy(self, lon_deg: float, lat_deg: float) -> tuple[float, float]:
        """Project a selenographic lon/lat into the DEM's own (projected, meters) CRS."""
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform(_lunar_geographic_crs(), self._ds.crs, [lon_deg], [lat_deg])
        return xs[0], ys[0]


def _lunar_geographic_crs():
    from rasterio.crs import CRS

    # IAU Moon 2000 geographic CRS (sphere, GDAL/rasterio understand this WKT string).
    return CRS.from_proj4(
        "+proj=longlat +a=1737400 +b=1737400 +no_defs"
    )


def _fill_nodata(data: np.ndarray) -> np.ndarray:
    """Nearest-neighbor fill for any NaN gaps left by a nodata mask, so the tile stays usable."""
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(data)
    if not mask.any():
        return data
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return data[tuple(idx)]
