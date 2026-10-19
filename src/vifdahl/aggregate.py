"""
Fuse pixel footprints into high (H, incl. A) and low (L) fire areas.

Ported from the 2019 aggregation notebooks
(legacy/2019_interior_fires_long_version/code/notebooks/*_agg*.ipynb).
"""

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

POLAR_CRS = "EPSG:6931"
WGS84 = "EPSG:4326"
GROW_M = 200.0
SHRINK_M = 120.0
HIGH_TYPES = ("A", "H")
MIN_OVERLAP_M2 = 1.0


def _summarize(pixels: pd.DataFrame) -> dict:
    """Attributes for one fused area from its member pixels."""
    unique = pixels.drop_duplicates(["Granule", "row", "col"])
    return {
        "n_pixels": len(unique),
        "n_overpasses": len(pixels.drop_duplicates(["Platform", "Orbit"])),
        "platforms": ",".join(sorted(pixels["Platform"].unique())),
        "first_utc": pixels["Granule"].min(),
        "last_utc": pixels["Granule"].max(),
        "TB4_max": pixels["TB4"].max(),
    }


def fuse(
    footprints: gpd.GeoDataFrame,
    crs: str = POLAR_CRS,
    grow_m: float = GROW_M,
    shrink_m: float = SHRINK_M,
    subtract: bool = True,
) -> gpd.GeoDataFrame:
    """Fuse footprints into rounded H and L areas, one feature per area.

    Each footprint is buffered by ``+grow_m`` then ``-shrink_m`` (rounded
    corners), then unioned per class. With ``subtract``, L = L − H.

    Parameters
    ----------
    footprints : geopandas.GeoDataFrame
        Pixel footprints with columns ``Type``, ``Granule``, ``row``,
        ``col``, ``Platform``, ``Orbit``, ``TB4``.
    crs : str, optional
        Projected CRS for buffering (meters).
    grow_m, shrink_m : float, optional
        Buffer distances in meters.
    subtract : bool, optional
        Remove H areas from L areas.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per contiguous area, in ``crs``, with column ``Class``.
    """
    proj = footprints.to_crs(crs)
    rounded = proj.assign(
        Class=np.where(proj["Type"].isin(HIGH_TYPES), "H", "L"),
        geometry=proj.buffer(grow_m).buffer(-shrink_m),
    )
    areas = {c: rounded.loc[rounded["Class"] == c].union_all() for c in ("H", "L")}
    if subtract and not areas["H"].is_empty:
        areas["L"] = areas["L"].difference(areas["H"])

    rows = []
    for cls, geom in areas.items():
        if geom.is_empty:
            continue
        parts = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([geom], crs=crs).explode(index_parts=False),
            crs=crs,
        ).reset_index(drop=True)
        members = rounded.loc[rounded["Class"] == cls]
        pairs = gpd.sjoin(members, parts, predicate="intersects")
        part_geoms = parts.geometry.loc[pairs["index_right"]].to_numpy()
        overlap = shapely.area(shapely.intersection(pairs.geometry.to_numpy(), part_geoms))
        pairs = pairs.loc[overlap > MIN_OVERLAP_M2]
        for part_id, pixels in pairs.groupby("index_right"):
            rows.append({"Class": cls, **_summarize(pixels),
                         "geometry": parts.geometry.loc[part_id]})
    if not rows:
        return gpd.GeoDataFrame({"Class": []}, geometry=[], crs=crs)
    fused = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    fused["area_km2"] = fused.area / 1e6
    return fused


def aggregate(
    footprints: gpd.GeoDataFrame,
    by: Sequence[str] = ("LocalDate", "Period"),
    **fuse_kwargs,
) -> gpd.GeoDataFrame:
    """Fuse footprints separately for each group (default: date and period).

    Parameters
    ----------
    footprints : geopandas.GeoDataFrame
        Pixel footprints (see ``fuse``) with the ``by`` columns.
    by : sequence of str, optional
        Grouping columns.
    **fuse_kwargs
        Passed to ``fuse``.

    Returns
    -------
    geopandas.GeoDataFrame
        Fused areas in EPSG:4326 with the ``by`` columns.
    """
    groups = []
    for keys, group in footprints.groupby(list(by)):
        fused = fuse(group, **fuse_kwargs)
        for col, key in zip(by, keys):
            fused.insert(0, col, key)
        groups.append(fused.to_crs(WGS84))
    if not groups:
        return gpd.GeoDataFrame(columns=[*by, "Class"], geometry=[], crs=WGS84)
    return gpd.GeoDataFrame(pd.concat(groups, ignore_index=True), crs=WGS84)
