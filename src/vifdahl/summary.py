"""
Run summary: one row per AOI × local date × period × platform, plus ``ALL``.

Counts are empty (not 0) where no overpass covered the AOI, so "observed, no
fire" and "not observed" stay distinct.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from vifdahl.aggregate import FUSE_BY, POLAR_CRS
from vifdahl.periods import assign_period

KEYS = ["aoi_id", "LocalDate", "Period", "Platform"]
TYPES = ["A", "H", "L"]
COUNTS = ["n_A", "n_H", "n_L", "n_detections"]
AREAS = ["area_H_km2", "area_L_km2"]
GAP = pd.Timedelta(minutes=10)  # same-platform granules closer than this: one overpass


def granule_periods(manifest: pd.DataFrame, aois: dict[str, BaseGeometry]) -> pd.DataFrame:
    """Add each granule's start ``utc``, ``LocalDate`` and ``Period``.

    The period is taken at the AOI centroid longitude. For AOIs spanning
    several hours of solar time a granule may straddle 06:00/18:00; it is
    still counted once, in the period of its start time at the centroid.
    """
    utc = pd.to_datetime(manifest["time_key"], format="A%Y%j.%H%M")
    lon = manifest["aoi_id"].map(lambda a: aois[a].centroid.x)
    periods = assign_period(utc, lon)
    return manifest.assign(utc=utc.to_numpy(), LocalDate=periods["LocalDate"].to_numpy(),
                           Period=periods["Period"].to_numpy(), Platform=manifest["platform"])


def observations(granules: pd.DataFrame) -> pd.DataFrame:
    """Overpass and failure counts per AOI, date, period and platform.

    Overpasses are distinct orbits; granules without an orbit fall back to
    grouping by time gap (``overpass_source`` = ``"time_gap"``).
    """
    done = granules.loc[granules["status"] == "done"].sort_values("utc")
    gap_id = done.groupby(["aoi_id", "Platform"])["utc"].transform(
        lambda t: (t.diff() > GAP).cumsum())
    no_orbit = done["orbit"].isna()
    done = done.assign(overpass=np.where(no_orbit, "t" + gap_id.astype(str),
                                         done["orbit"].astype("Int64").astype(str)),
                       no_orbit=no_orbit)
    obs = done.groupby(KEYS).agg(n_overpasses=("overpass", "nunique"), gap=("no_orbit", "any"))
    obs["overpass_source"] = np.where(obs.pop("gap"), "time_gap", "orbit")
    failed = granules.loc[granules["status"] == "failed"].groupby(KEYS).size()
    return obs.join(failed.rename("n_failed"), how="outer")


def detection_stats(detections: pd.DataFrame) -> pd.DataFrame:
    """Counts by type and hottest TB4 per AOI, date, period and platform."""
    if detections.empty:
        return pd.DataFrame(columns=[*COUNTS, "TB4_max"],
                            index=pd.MultiIndex.from_tuples([], names=KEYS))
    grouped = detections.groupby(KEYS)
    stats = (grouped["Type"].value_counts().unstack(fill_value=0)
             .reindex(columns=TYPES, fill_value=0).add_prefix("n_"))
    stats["n_detections"] = stats.sum(axis=1)
    return stats.join(grouped["TB4"].max().rename("TB4_max"))


def fused_areas(fused: gpd.GeoDataFrame, union: bool = False) -> pd.DataFrame:
    """H and L area (km², EPSG:6931) per AOI, date, period and ``Platform``.

    With ``union``, per-platform areas are merged into one ``ALL`` row:
    H = union of H, L = union of L minus H.
    """
    if fused.empty:
        return pd.DataFrame(columns=AREAS, index=pd.MultiIndex.from_tuples([], names=KEYS))
    if not union:
        areas = fused.groupby([*KEYS, "Class"])["area_km2"].sum().unstack("Class")
        return areas.reindex(columns=["H", "L"]).set_axis(AREAS, axis=1)
    rows = {}
    for key, group in fused.to_crs(POLAR_CRS).groupby(KEYS[:3]):
        high = group.loc[group["Class"] == "H"].union_all()
        low = group.loc[group["Class"] == "L"].union_all().difference(high)
        rows[(*key, "ALL")] = (high.area / 1e6, low.area / 1e6)
    return pd.DataFrame(list(rows.values()), columns=AREAS,
                        index=pd.MultiIndex.from_tuples(list(rows), names=KEYS))


def summarize(
    manifest: pd.DataFrame,
    detections: pd.DataFrame,
    fused: gpd.GeoDataFrame,
    aois: dict[str, BaseGeometry],
    fuse_by: str = "all",
    dates: pd.DatetimeIndex | None = None,
    platforms: list[str] | None = None,
) -> pd.DataFrame:
    """Build the summary table for a run.

    Parameters
    ----------
    manifest : pandas.DataFrame
        From ``vifdahl.manifest.Manifest.to_frame``.
    detections : pandas.DataFrame
        All AOIs' detections, with an ``aoi_id`` column.
    fused : geopandas.GeoDataFrame
        All AOIs' fused areas (``vifdahl.aggregate.aggregate``), with ``aoi_id``.
    aois : dict of str to shapely.geometry.base.BaseGeometry
        AOI ID → geometry (EPSG:4326).
    fuse_by : {"all", "platform"}, optional
        How ``fused`` was made; sets which rows carry areas.
    dates : pandas.DatetimeIndex, optional
        Local dates to cover; default from the first to the last granule.
    platforms : list of str, optional
        Default: those in the manifest.

    Returns
    -------
    pandas.DataFrame
        One row per AOI, date, period and platform (incl. ``ALL``).
    """
    if fuse_by not in FUSE_BY:
        raise ValueError(f"fuse_by must be one of {FUSE_BY}, not {fuse_by!r}")
    granules = granule_periods(manifest, aois)
    if dates is None:
        dates = pd.date_range(granules["LocalDate"].min(), granules["LocalDate"].max())
    grid = pd.MultiIndex.from_product(
        [sorted(aois), dates.strftime("%Y-%m-%d"), ["day", "night"],
         platforms or sorted(granules["Platform"].unique())], names=KEYS)
    obs = observations(granules)
    stats = detection_stats(detections)
    if len(unseen := stats.index.difference(obs.index[obs["n_overpasses"] > 0])):
        warnings.warn(f"{len(unseen)} groups have detections but no overpass, e.g. {unseen[0]}")
    table = obs.reindex(grid).join(stats)
    table[["n_overpasses", "n_failed"]] = table[["n_overpasses", "n_failed"]].fillna(0)
    if fuse_by == "platform":
        table = table.join(fused_areas(fused))

    combined = table.groupby(level=KEYS[:3]).agg(
        n_overpasses=("n_overpasses", "sum"), n_failed=("n_failed", "sum"),
        **{c: (c, "sum") for c in COUNTS}, TB4_max=("TB4_max", "max"),
        overpass_source=("overpass_source", lambda s: s.dropna().max()))  # time_gap > orbit
    combined = combined.assign(Platform="ALL").set_index("Platform", append=True)
    combined = combined.join(fused_areas(fused, union=fuse_by == "platform"))

    out = pd.concat([table, combined]).sort_index().reindex(columns=[
        "n_overpasses", "overpass_source", "n_failed", *COUNTS, *AREAS, "TB4_max"])
    observed = out["n_overpasses"] > 0
    has_area = (out.index.get_level_values("Platform") == "ALL") | (fuse_by == "platform")
    for col in COUNTS:
        out[col] = out[col].fillna(0).where(observed).astype("Int64")
    for col in AREAS:
        out[col] = out[col].astype(float).fillna(0.0).where(observed & has_area)
    out[["n_overpasses", "n_failed"]] = out[["n_overpasses", "n_failed"]].astype(int)
    return out.reset_index()
