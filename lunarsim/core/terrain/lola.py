"""Real NASA LOLA elevation (PDS3 `LDEM_*` .IMG tiles) -> local heightmap.

The PDS tiles are raw int16 rasters (HEIGHT = DN * 0.5 m relative to a 1737.4 km reference sphere),
simple-cylindrical, pixel-registered, row 0 at the tile's northern edge, column 0 at its western
edge. No GDAL needed: the file is memory-mapped and only the window around the site is touched.

File name encodes the footprint, e.g. ``ldem_1024_00n_15n_000_030.img`` = 1024 px/deg,
0..15 deg N, 0..30 deg E.
"""
from __future__ import annotations

import os
import re

import numpy as np

from lunarsim.core.terrain.curvature import MOON_RADIUS_M

_NAME = re.compile(r"ldem_(\d+)_(\d+)([ns])_(\d+)([ns])_(\d+)_(\d+)", re.IGNORECASE)


class LolaImg:
    def __init__(self, path: str):
        m = _NAME.search(os.path.basename(path))
        if m is None:
            raise ValueError(f"cannot parse LDEM footprint from file name: {path}")
        self.ppd = int(m.group(1))
        la, sa, lb, sb = int(m.group(2)), m.group(3).lower(), int(m.group(4)), m.group(5).lower()
        la = la if sa == "n" else -la
        lb = lb if sb == "n" else -lb
        self.lat_top, self.lat_bot = max(la, lb), min(la, lb)
        self.lon_w, self.lon_e = int(m.group(6)), int(m.group(7))
        rows = int(round((self.lat_top - self.lat_bot) * self.ppd))
        cols = int(round((self.lon_e - self.lon_w) * self.ppd))
        self.data = np.memmap(path, dtype="<i2", mode="r", shape=(rows, cols))
        self.rows, self.cols = rows, cols

    @property
    def m_per_px(self) -> float:
        return float(np.deg2rad(1.0 / self.ppd) * MOON_RADIUS_M)

    def contains(self, lat: float, lon: float, half_size_m: float) -> bool:
        half_deg = np.rad2deg(half_size_m / MOON_RADIUS_M) + 2.0 / self.ppd
        hl = half_deg / max(np.cos(np.deg2rad(lat)), 1e-6)
        return (self.lat_bot + half_deg < lat < self.lat_top - half_deg
                and self.lon_w + hl < lon < self.lon_e - hl)

    def read_tile(self, lat_deg: float, lon_deg: float, size_m: float, res_m: float) -> np.ndarray:
        """(n, n) heights [m]; ``h[i, j]`` sits at local (x = east, y = north) = (ax[i], ax[j])."""
        from scipy.ndimage import map_coordinates

        n = int(round(size_m / res_m))
        ax = (np.arange(n) - (n - 1) / 2.0) * res_m
        x, y = np.meshgrid(ax, ax, indexing="ij")             # x east, y north
        dlat = np.rad2deg(y / MOON_RADIUS_M)
        dlon = np.rad2deg(x / (MOON_RADIUS_M * max(np.cos(np.deg2rad(lat_deg)), 1e-6)))
        lat = lat_deg + dlat
        lon = lon_deg + dlon
        # pixel-registered: pixel centres sit at integer + 0.5
        row = (self.lat_top - lat) * self.ppd - 0.5
        col = (lon - self.lon_w) * self.ppd - 0.5
        r0, r1 = int(np.floor(row.min())) - 3, int(np.ceil(row.max())) + 4
        c0, c1 = int(np.floor(col.min())) - 3, int(np.ceil(col.max())) + 4
        if r0 < 0 or c0 < 0 or r1 > self.rows or c1 > self.cols:
            raise ValueError("requested window falls outside the LDEM tile")
        win = np.asarray(self.data[r0:r1, c0:c1], dtype=np.float64) * 0.5
        h = map_coordinates(win, [row - r0, col - c0], order=3, mode="nearest")
        return h - h[n // 2, n // 2]


class NacDtm:
    """LROC NAC stereo DTM (GeoTIFF, float32 metres, equirectangular, ~2 m/px), read without GDAL.

    The georeference is taken from the TIFF tags (ModelPixelScale 33550, ModelTiepoint 33922): the
    projection is simple-cylindrical about lon 180 with standard parallel ``LAT_TS`` (degrees).
    A float32 .npy cache of the pixel array is written next to the TIFF on first use.
    """

    LON0_DEG = 180.0
    LAT_TS_DEG = 1.0
    NODATA = -1.0e6

    def __init__(self, path: str):
        from PIL import Image

        cache = os.path.splitext(path)[0] + ".f32.npy"
        im = None
        if not os.path.exists(cache):
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(path)
            np.save(cache, np.asarray(im, dtype=np.float32))
        else:
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(path)
        self.data = np.load(cache, mmap_mode="r")
        tags = im.tag_v2
        self.px = float(tags[33550][0])
        tp = tags[33922]
        self.x0, self.y0 = float(tp[3]), float(tp[4])          # projected metres of pixel (0, 0) corner
        self.rows, self.cols = self.data.shape
        # projection parameters from the GeoKey directory (the standard parallel differs per DTM)
        lat_ts, lon0 = self.LAT_TS_DEG, self.LON0_DEG
        gk, dbl = tags.get(34735), tags.get(34736)
        if gk is not None and dbl is not None:
            for i in range(int(gk[3])):
                key, loc, _cnt, off = gk[4 + 4 * i: 8 + 4 * i]
                if loc == 34736 and key == 3078:
                    lat_ts = float(dbl[off])
                if loc == 34736 and key == 3088:
                    lon0 = float(dbl[off])
        self.LAT_TS_DEG, self.LON0_DEG = lat_ts, lon0
        self.R = MOON_RADIUS_M
        self._kx = self.R * np.cos(np.deg2rad(self.LAT_TS_DEG))

    # lat/lon <-> projected metres
    def to_xy(self, lat, lon):
        return self._kx * np.deg2rad(np.asarray(lon) - self.LON0_DEG), self.R * np.deg2rad(np.asarray(lat))

    def bounds(self, size_m: float):
        """(lat_min, lat_max, lon_min, lon_max) of centres whose size_m window is fully inside."""
        h = size_m / 2.0 + 4 * self.px
        x_lo, x_hi = self.x0 + h, self.x0 + self.cols * self.px - h
        y_hi, y_lo = self.y0 - h, self.y0 - self.rows * self.px + h
        lon = lambda x: self.LON0_DEG + np.rad2deg(x / self._kx)  # noqa: E731
        lat = lambda y: np.rad2deg(y / self.R)                    # noqa: E731
        return float(lat(y_lo)), float(lat(y_hi)), float(lon(x_lo)), float(lon(x_hi))

    def valid_fraction(self, lat_deg: float, lon_deg: float, size_m: float) -> float:
        """Fraction of valid (finite, |h| < 10 km) DTM pixels in the size_m window (subsampled)."""
        xc, yc = self.to_xy(lat_deg, lon_deg)
        c0 = int((xc - size_m / 2 - self.x0) / self.px); c1 = int((xc + size_m / 2 - self.x0) / self.px)
        r0 = int((self.y0 - yc - size_m / 2) / self.px); r1 = int((self.y0 - yc + size_m / 2) / self.px)
        if r0 < 0 or c0 < 0 or r1 > self.rows or c1 > self.cols:
            return 0.0
        w = np.asarray(self.data[r0:r1:6, c0:c1:6], dtype=np.float64)
        return float(np.mean(np.isfinite(w) & (w > -1.0e4) & (w < 1.0e4)))

    def read_tile(self, lat_deg: float, lon_deg: float, size_m: float, res_m: float) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

        n = int(round(size_m / res_m))
        ax = (np.arange(n) - (n - 1) / 2.0) * res_m
        xc, yc = self.to_xy(lat_deg, lon_deg)
        X, Y = np.meshgrid(xc + ax, yc + ax, indexing="ij")     # x east (axis 0), y north (axis 1)
        col = (X - self.x0) / self.px - 0.5
        row = (self.y0 - Y) / self.px - 0.5
        r0, r1 = int(np.floor(row.min())) - 3, int(np.ceil(row.max())) + 4
        c0, c1 = int(np.floor(col.min())) - 3, int(np.ceil(col.max())) + 4
        if r0 < 0 or c0 < 0 or r1 > self.rows or c1 > self.cols:
            raise ValueError("requested window falls outside the NAC DTM")
        win = np.array(self.data[r0:r1, c0:c1], dtype=np.float64)
        # nodata is not only -3.4e38: stray positive outliers up to ~1e22 m exist in the products.
        # Lunar elevations (vs the reference sphere) stay within +-10 km, so anything outside is void;
        # then single-pixel spikes (>40 m off the 5x5 median) are voided as well.
        from scipy.ndimage import median_filter
        bad = ~np.isfinite(win) | (win < -1.0e4) | (win > 1.0e4)
        if bad.all():
            raise ValueError("window holds no valid DTM pixels")
        if bad.any():
            filled = win.copy(); filled[bad] = np.nanmedian(np.where(bad, np.nan, win))
        else:
            filled = win
        spikes = np.abs(filled - median_filter(filled, size=5)) > 40.0
        bad |= spikes
        if bad.any():
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            win = win[tuple(idx)]
        if res_m > self.px * 1.2:                               # anti-alias before decimating
            win = gaussian_filter(win, 0.5 * res_m / self.px)
        h = map_coordinates(win, [row - r0, col - c0], order=1, mode="nearest")
        return h - h[n // 2, n // 2]


def open_dem(path: str):
    """LOLA PDS3 LDEM tile or LROC NAC DTM GeoTIFF, chosen by file name."""
    low = os.path.basename(path).lower()
    return NacDtm(path) if low.endswith((".tif", ".tiff")) else LolaImg(path)


def sample_site(path: str, size_m: float, rng: np.random.Generator, box=None, split: str = ""):
    """Random (lat, lon) whose size_m tile is fully covered. ``box=(lat0, lon0, half_deg)`` for LDEM.

    ``split`` = "train" / "val" cuts the DTM at its middle longitude (west / east half, with a
    one-tile gap) so evaluation tiles never overlap any training tile.
    """
    dem = open_dem(path)
    if isinstance(dem, NacDtm):
        la0, la1, lo0, lo1 = dem.bounds(size_m)
        if split in ("train", "val"):
            mid = 0.5 * (lo0 + lo1)
            gap = np.rad2deg(size_m / (dem._kx))            # one tile width, in degrees of longitude
            lo0, lo1 = (lo0, mid - gap / 2) if split == "train" else (mid + gap / 2, lo1)
        for _ in range(200):                                 # the footprint has voids: redraw until covered
            la, lo = float(rng.uniform(la0, la1)), float(rng.uniform(lo0, lo1))
            if dem.valid_fraction(la, lo, size_m) >= 0.995:
                return la, lo
        raise RuntimeError("no fully covered site found in the requested region")
    lat0, lon0, half = box
    return lat0 + float(rng.uniform(-half, half)), lon0 + float(rng.uniform(-half, half))
