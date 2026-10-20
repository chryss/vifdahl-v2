"""
Stage 2: stage-1 detection files → one GeoPackage per AOI and local solar date
(points, footprints, fused layers), plus ``summary.csv`` for the run.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vifdahl.detect import COLUMNS
from vifdahl.manifest import Manifest
from vifdahl.output import write_daily
from vifdahl.summary import summarize


def read_aois(workdir: Path) -> dict[str, BaseGeometry]:
    """AOI ID → geometry from a run's ``stage1_config.json``."""
    info = json.loads((workdir / "stage1_config.json").read_text())
    return {k: shapely.from_wkt(v) for k, v in info["aois_wkt"].items()}


def load_detections(manifest: pd.DataFrame, aoi_id: str) -> pd.DataFrame:
    """Concatenate one AOI's detections from the granules listed as done.

    Parameters
    ----------
    manifest : pandas.DataFrame
        From ``vifdahl.manifest.Manifest.to_frame``.
    aoi_id : str
        AOI to load.

    Returns
    -------
    pandas.DataFrame
        Columns as ``vifdahl.detect.detections_table``, with footprint
        polygons in ``footprint``.
    """
    # empty detection files carry no dtypes; skip them
    done = manifest.loc[(manifest["aoi_id"] == aoi_id) & (manifest["status"] == "done")
                        & (manifest["n_detections"] > 0)]
    if done.empty:
        return pd.DataFrame(columns=COLUMNS)
    frames = []
    for path in done["detections_path"]:
        gdf = gpd.read_parquet(path)
        frames.append(pd.DataFrame(gdf.drop(columns="geometry"))
                      .assign(footprint=gdf.geometry.to_numpy()))
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def run_stage2(
    workdir: str | Path,
    outdir: str | Path | None = None,
    prefix: str = "VIFDAHL",
    aoi_ids: list[str] | None = None,
    fuse_by: str = "all",
    dates: pd.DatetimeIndex | None = None,
    platforms: list[str] | None = None,
    **fuse_kwargs,
) -> dict[str, list[Path]]:
    """Write daily GeoPackages for each AOI of a run, and ``outdir/summary.csv``.

    Parameters
    ----------
    workdir : str or pathlib.Path
        Run folder from stage 1.
    outdir : str or pathlib.Path, optional
        Output folder; default ``workdir/gpkg``. Files go in ``outdir/<aoi_id>/``.
    prefix : str, optional
        File name prefix; files are ``{prefix}_{aoi_id}_{YYYYMMDD}.gpkg``.
    aoi_ids : list of str, optional
        AOIs to process; default all in the manifest.
    fuse_by : {"all", "platform"}, optional
        See ``vifdahl.aggregate.aggregate``.
    dates : pandas.DatetimeIndex, optional
        Local dates for the summary; default from the first to the last granule.
    platforms : list of str, optional
        Platforms for the summary; default those in the manifest.
    **fuse_kwargs
        Passed to ``vifdahl.aggregate.fuse``.

    Returns
    -------
    dict of str to list of pathlib.Path
        Files written per AOI, one per local date with detections.
    """
    workdir = Path(workdir)
    outdir = Path(outdir or workdir / "gpkg")
    manifest = Manifest(workdir / "manifest.sqlite").to_frame()
    aois = read_aois(workdir)
    aoi_ids = aoi_ids or sorted(manifest["aoi_id"].unique())
    written, detections, fused = {}, [], []
    for aoi_id in aoi_ids:
        dets = load_detections(manifest, aoi_id)
        written[aoi_id] = write_daily(dets, outdir / aoi_id, prefix=f"{prefix}_{aoi_id}",
                                      fuse_by=fuse_by, **fuse_kwargs)
        detections.append(dets.assign(aoi_id=aoi_id))
        fused += [gpd.read_file(p, layer="fused").assign(aoi_id=aoi_id) for p in written[aoi_id]]
    fused = (gpd.GeoDataFrame(pd.concat(fused, ignore_index=True)) if fused
             else gpd.GeoDataFrame({"Class": []}, geometry=[], crs="EPSG:4326"))
    summary = summarize(manifest[manifest["aoi_id"].isin(aoi_ids)],
                        pd.concat(detections, ignore_index=True), fused,
                        {a: aois[a] for a in aoi_ids}, fuse_by, dates, platforms)
    summary.to_csv(outdir / "summary.csv", index=False)
    return written
