"""End-to-end check on a synthetic swath: detect, tabulate, write daily files."""

from pathlib import Path

import geopandas as gpd
import pyogrio
import xarray as xr

from vifdahl.detect import detect, detections_table
from vifdahl.output import write_daily


def test_detect_and_write(swath: xr.Dataset, tmp_path: Path) -> None:
    masks = detect(swath)
    assert masks.hot.sum() == 5
    table = detections_table(swath, masks)
    assert set(table["Type"]) == {"H"}
    assert (table["TB4"] >= 330.0).all()
    # 11:35 UTC at ~147°W is ~01:45 local solar time: night of the previous date
    assert set(table["LocalDate"]) == {"2015-06-14"}
    assert set(table["Period"]) == {"night"}

    paths = write_daily(table, tmp_path)
    assert [p.name for p in paths] == ["VIFDAHL_20150614.gpkg"]
    layers = {name for name, _ in pyogrio.list_layers(paths[0])}
    assert layers == {"points", "footprints", "fused"}
    footprints = gpd.read_file(paths[0], layer="footprints")
    fused = gpd.read_file(paths[0], layer="fused")
    assert len(footprints) == 5 and footprints.geometry.is_valid.all()
    assert sorted(fused["n_pixels"]) == [1, 4]


def test_empty_swath(swath: xr.Dataset) -> None:
    swath["i4_bt"][:] = 270.0
    table = detections_table(swath, detect(swath))
    assert table.empty and "TB4" in table.columns
