"""Run config parsing: AOIs from a vector file."""

import datetime as dt
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from vifdahl.runconfig import load_aoi_file, resolve_aois


def square_z(x: float, y: float) -> Polygon:
    return Polygon([(x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)])


@pytest.fixture
def fires(tmp_path: Path) -> Path:
    path = tmp_path / "fires.geojson"
    gpd.GeoDataFrame({"Name": ["Fire A", "Fire/B"]},
                     geometry=[square_z(-150, 64), square_z(-145, 65)],
                     crs="EPSG:4326").to_file(path)
    return path


def test_load_aoi_file(fires: Path) -> None:
    aois = load_aoi_file(fires, id_column="Name")
    assert list(aois) == ["Fire_A", "Fire_B"]
    assert not aois["Fire_A"].has_z
    assert aois["Fire_A"].bounds == pytest.approx((-150, 64, -149, 65))
    grown = load_aoi_file(fires, id_column="Name", buffer_km=10)["Fire_A"]
    assert grown.contains(aois["Fire_A"]) and grown.bounds[1] < 64 - 0.08
    with pytest.raises(ValueError, match="id_column"):
        load_aoi_file(fires)


def test_resolve_file_aois(fires: Path) -> None:
    run = {"name": "two fires", "aoi": {"file": str(fires), "id_column": "Name"},
           "dates": {"start": dt.date(2022, 6, 1), "end": dt.date(2022, 6, 30)}}
    aois, start, end = resolve_aois(run)
    assert list(aois) == ["Fire_A", "Fire_B"] and start == dt.date(2022, 6, 1)
    single = gpd.read_file(fires).iloc[:1]
    single.to_file(fires.with_name("one.geojson"))
    run["aoi"] = {"file": str(fires.with_name("one.geojson"))}
    assert list(resolve_aois(run)[0]) == ["two_fires"]
    del run["dates"]
    with pytest.raises(ValueError, match="dates"):
        resolve_aois(run)
