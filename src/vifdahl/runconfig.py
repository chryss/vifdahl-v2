"""Shared parsing of run config TOMLs (see ``config/run_template.toml``)."""

import datetime as dt
import re
import tomllib
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from vifdahl.aggregate import POLAR_CRS, WGS84
from vifdahl.events import EVENTS_FILE, load_events


def load_run(path: str | Path) -> dict:
    """Read a run config TOML."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def safe_id(value: object) -> str:
    """Folder-safe AOI ID: runs of characters other than letters, digits, ``.-_`` → ``_``."""
    return re.sub(r"[^\w.-]+", "_", str(value)).strip("_")


def load_aoi_file(
    path: str | Path,
    id_column: str | None = None,
    buffer_km: float = 0.0,
    default_id: str = "aoi",
) -> dict[str, BaseGeometry]:
    """Read a vector file as AOIs, one per feature.

    Parameters
    ----------
    path : str or pathlib.Path
        Any file geopandas reads (GeoJSON, GeoPackage, shapefile, ...).
    id_column : str, optional
        Column holding each feature's ID; required with more than one feature.
    buffer_km : float, optional
        Outward buffer, applied in EPSG:6931.
    default_id : str, optional
        ID of a single feature when ``id_column`` is not given.

    Returns
    -------
    dict of str to shapely.geometry.base.BaseGeometry
        AOI ID → 2-D geometry in EPSG:4326.
    """
    gdf = gpd.read_file(path)
    if id_column is None and len(gdf) > 1:
        raise ValueError(f"{path}: {len(gdf)} features; set id_column")
    ids = [safe_id(v) for v in gdf[id_column]] if id_column else [default_id]
    if len(set(ids)) < len(ids):
        raise ValueError(f"{path}: duplicate AOI IDs in {id_column!r}")
    geoms = gdf.geometry.to_crs(POLAR_CRS)
    if buffer_km:
        geoms = geoms.buffer(buffer_km * 1e3)
    geoms = shapely.force_2d(geoms.to_crs(WGS84).to_numpy())
    return dict(zip(ids, geoms))


def resolve_aois(run: dict) -> tuple[dict[str, BaseGeometry], dt.date | None, dt.date | None]:
    """Return the AOIs (ID → EPSG:4326 geometry) and the run's local start and end dates.

    Parameters
    ----------
    run : dict
        Parsed run config.

    Returns
    -------
    tuple
        AOIs, start date, end date. An event is keyed by its events-file key,
        a bbox by the run name, file features by ``id_column`` (a single
        feature without one: the run name). ``[dates]`` overrides an event's
        dates and is required otherwise.
    """
    aoi = run["aoi"]
    if "event" in aoi:
        event = load_events(aoi.get("events_file", EVENTS_FILE))[aoi["event"]]
        aois, start, end = {aoi["event"]: event.aoi()}, event.start, event.end
    elif "bbox" in aoi:
        aois, start, end = {run["name"]: box(*aoi["bbox"])}, None, None
    elif "file" in aoi:
        aois = load_aoi_file(aoi["file"], aoi.get("id_column"), aoi.get("buffer_km", 0.0),
                             default_id=safe_id(run["name"]))
        start = end = None
    else:
        raise ValueError("[aoi] needs one of event, bbox, file")
    dates = run.get("dates", {})
    start, end = dates.get("start", start), dates.get("end", end)
    if start is None or end is None:
        raise ValueError("[dates] start and end are required unless the AOI is an event")
    return aois, start, end
