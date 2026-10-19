"""Fusing footprints: equivalence with 2019 products, and L − H subtraction."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from vifdahl.aggregate import fuse

REFDIR = Path(__file__).parents[1] / "data" / "reference" / "2019_shovelcreek"
SLICES = {
    "20190708_0210": ["20190708_1104222", "20190708_1245224", "20190708_1424572"],
    "20190707_1018": ["20190707_1939446", "20190707_2119194", "20190707_2258542"],
}


def load_overpasses(stamps: list[str]) -> gpd.GeoDataFrame:
    """Read 2019 per-overpass footprints, adding the columns ``fuse`` needs."""
    frames = []
    for stamp in stamps:
        df = gpd.read_file(REFDIR / f"{stamp}_VIFDAHL305_polygons.shp")
        frames.append(df.assign(Granule=stamp, row=df.index, col=0, Orbit=0))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


@pytest.mark.parametrize("tslice", SLICES)
def test_matches_2019_product(tslice: str) -> None:
    expected = gpd.read_file(REFDIR / f"shovelcreek_VIFDAHL305_{tslice}_AlaskaAlbers.gpkg")
    fused = fuse(load_overpasses(SLICES[tslice]), crs="EPSG:3338", subtract=False)
    result = fused.dissolve(by="Class")
    for cls in ("H", "L"):
        want = expected.loc[expected["Type"] == cls].geometry.iloc[0]
        got = result.geometry.loc[cls]
        assert got.symmetric_difference(want).area / want.area < 1e-6


def square(x: float, y: float, size: float = 375.0) -> object:
    return box(x, y, x + size, y + size)


def synthetic(types: list[str], geoms: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"Type": types, "Granule": "g", "row": range(len(types)), "col": 0,
         "Platform": "NPP", "Orbit": 1, "TB4": 330.0},
        geometry=geoms, crs="EPSG:3338",
    )


def test_subtract_low_from_high() -> None:
    gdf = synthetic(["H", "L", "A"], [square(0, 0), square(375, 0), square(0, 375)])
    fused = fuse(gdf, crs="EPSG:3338")
    high = fused.loc[fused["Class"] == "H"]
    low = fused.loc[fused["Class"] == "L"]
    assert len(high) == 1 and high["n_pixels"].iloc[0] == 2
    assert len(low) == 1
    assert low.geometry.iloc[0].intersection(high.geometry.iloc[0]).area < 1.0


def test_separate_areas() -> None:
    gdf = synthetic(["H", "H", "H"], [square(0, 0), square(375, 0), square(5000, 0)])
    fused = fuse(gdf, crs="EPSG:3338")
    assert sorted(fused["n_pixels"]) == [1, 2]


def test_low_fully_covered_by_high() -> None:
    gdf = synthetic(["H", "L"], [square(0, 0), square(0, 0)])
    fused = fuse(gdf, crs="EPSG:3338")
    assert list(fused["Class"]) == ["H"]
