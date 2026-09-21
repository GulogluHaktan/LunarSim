import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def synthetic_lunar_dem(tmp_path):
    """A small geographic (equirectangular, lunar sphere) GeoTIFF with a known
    elevation gradient, standing in for a real LOLA GDR mosaic tile in tests.
    """
    path = tmp_path / "fake_lola.tif"
    n = 200
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, y)
    elevation = (500.0 * xx + 200.0 * yy).astype(np.float64)

    transform = from_origin(-1.0, 1.0, 2.0 / n, 2.0 / n)
    crs = rasterio.crs.CRS.from_proj4("+proj=longlat +a=1737400 +b=1737400 +no_defs")

    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1,
        dtype="float64", crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)

    return path, elevation
