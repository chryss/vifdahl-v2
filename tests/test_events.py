"""Event registry and AOI geometry."""

import datetime as dt
from pathlib import Path

import geopandas as gpd
import pytest

from conftest import EVENTS_TEMPLATE
from vifdahl.events import load_events


def test_load_registry() -> None:
    shovel = load_events(EVENTS_TEMPLATE)["shovelcreek"]
    assert shovel.name == "Shovel Creek Fire"
    assert (shovel.west_km, shovel.east_km) == (5, 20)
    assert shovel.start == dt.date(2019, 6, 24)


def test_defaults_and_optional_dates(tmp_path: Path) -> None:
    path = tmp_path / "events.toml"
    path.write_text(
        '[defaults]\nextent_km = { north = 10, south = 10, west = 10, east = 10 }\n'
        '[events.x]\nname = "X"\ncenter = [-150.0, 65.0]\nextent_km = { west = 3 }\n'
    )
    x = load_events(path)["x"]
    assert (x.north_km, x.west_km) == (10, 3)
    assert x.start is None and x.end is None


def test_aoi_size_and_center() -> None:
    shovel = load_events(EVENTS_TEMPLATE)["shovelcreek"]
    aoi = gpd.GeoSeries([shovel.aoi()], crs="EPSG:4326").to_crs("EPSG:6931")
    assert aoi.area.iloc[0] / 1e6 == pytest.approx(25 * 20, rel=1e-3)
    w, s, e, n = shovel.bbox()
    assert w < shovel.center_lon < e and s < shovel.center_lat < n
