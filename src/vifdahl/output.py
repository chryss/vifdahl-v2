"""GIS output: points, pixel footprints, and fused areas; one GeoPackage per local date."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from vifdahl.aggregate import aggregate

WGS84 = "EPSG:4326"


def points_layer(detections: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return detections as pixel-center points (EPSG:4326)."""
    geometry = [Point(lon, lat) for lon, lat in zip(detections["LON"], detections["LAT"])]
    return gpd.GeoDataFrame(
        detections.drop(columns="footprint"), geometry=geometry, crs=WGS84
    )


def footprints_layer(detections: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return detections as per-pixel footprint polygons (EPSG:4326)."""
    return gpd.GeoDataFrame(
        detections.drop(columns="footprint"),
        geometry=list(detections["footprint"]),
        crs=WGS84,
    )


def write_geopackage(detections: pd.DataFrame, path: str | Path, **fuse_kwargs) -> None:
    """Write ``points``, ``footprints`` and ``fused`` layers to one GeoPackage.

    Parameters
    ----------
    detections : pandas.DataFrame
        Output of ``vifdahl.detect.detections_table`` (concatenated).
    path : str or pathlib.Path
        Output ``.gpkg``; existing layers are overwritten.
    **fuse_kwargs
        Passed to ``vifdahl.aggregate.fuse``.
    """
    footprints = footprints_layer(detections)
    points_layer(detections).to_file(path, layer="points", driver="GPKG")
    footprints.to_file(path, layer="footprints", driver="GPKG")
    aggregate(footprints, **fuse_kwargs).to_file(path, layer="fused", driver="GPKG")


def write_daily(
    detections: pd.DataFrame, outdir: str | Path, prefix: str = "VIFDAHL", **fuse_kwargs
) -> list[Path]:
    """Write one GeoPackage per local solar date: ``{prefix}_{YYYYMMDD}.gpkg``.

    Parameters
    ----------
    detections : pandas.DataFrame
        Output of ``vifdahl.detect.detections_table`` (concatenated).
    outdir : str or pathlib.Path
        Output directory (created if missing).
    prefix : str, optional
        File name prefix, e.g. including the threshold variant.
    **fuse_kwargs
        Passed to ``vifdahl.aggregate.fuse``.

    Returns
    -------
    list of pathlib.Path
        Files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for date, day in detections.groupby("LocalDate"):
        path = outdir / f"{prefix}_{date.replace('-', '')}.gpkg"
        path.unlink(missing_ok=True)
        write_geopackage(day, path, **fuse_kwargs)
        paths.append(path)
    return paths
