"""
Stage 1: per granule pair, download → crop to AOI → detect → save crop and
detections → delete raw files. Granules run in parallel (joblib).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import earthaccess
import geopandas as gpd
import pandas as pd
import shapely
import xarray as xr
from joblib import Parallel, delayed
from shapely.geometry.base import BaseGeometry

from vifdahl.algorithm import DEFAULT_THRESHOLDS, Thresholds
from vifdahl.detect import detect, detections_table
from vifdahl.granules import GranulePair, download_pair
from vifdahl.manifest import Manifest
from vifdahl.readers import save_swath
from vifdahl.readers.l1b import read_l1b

_LOGGED_IN = False


@dataclass
class Stage1Config:
    """Settings shared by all granules of a run; ``aois`` maps AOI ID → geometry (EPSG:4326)."""

    aois: dict[str, BaseGeometry]
    workdir: Path
    thresholds: Thresholds = field(default_factory=lambda: DEFAULT_THRESHOLDS)
    daynight: str = "scene"
    keep_raw: bool = False
    keep_crops: bool = True

    @property
    def raw_dir(self) -> Path:
        return self.workdir / "raw"

    def write(self) -> None:
        """Record the settings in ``workdir/stage1_config.json``."""
        info = {"aois_wkt": {k: v.wkt for k, v in self.aois.items()},
                "thresholds": asdict(self.thresholds),
                "daynight": self.daynight,
                "keep_raw": self.keep_raw, "keep_crops": self.keep_crops}
        (self.workdir / "stage1_config.json").write_text(json.dumps(info, indent=2))


def process_pair(obs: Path, geo: Path, granule_id: str, aoi_id: str, cfg: Stage1Config) -> dict:
    """Crop, detect and save one granule pair for one AOI from local files.

    The crop is saved only with ``cfg.keep_crops``.

    Returns
    -------
    dict
        Manifest record fields for this AOI (without raw-file fields).
    """
    aoi = cfg.aois[aoi_id]
    swath = read_l1b(obs, geo, aoi=aoi)
    if swath is None:
        return {"status": "no_coverage", "n_detections": 0}
    crop_path = None
    if cfg.keep_crops:
        crop_path = cfg.workdir / "crops" / aoi_id / f"{granule_id}.nc"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        save_swath(swath, crop_path)
    det_path = cfg.workdir / "detections" / aoi_id / f"{granule_id}.parquet"
    n_detections = write_detections(swath, det_path, aoi, cfg)
    return {"status": "done", "orbit": swath.attrs["orbit"], "n_detections": n_detections,
            "crop_path": crop_path and str(crop_path), "detections_path": str(det_path)}


def write_detections(swath: xr.Dataset, det_path: Path, aoi: BaseGeometry,
                     cfg: Stage1Config) -> int:
    """Detect fire pixels in a cropped swath and save those inside the AOI.

    Parameters
    ----------
    swath : xarray.Dataset
        Cropped swath, as from ``read_l1b`` or ``load_swath``.
    det_path : pathlib.Path
        Output GeoParquet file.
    aoi : shapely.geometry.base.BaseGeometry
        Detections outside it are dropped.
    cfg : Stage1Config
        Supplies thresholds and day/night mode.

    Returns
    -------
    int
        Number of detections written.
    """
    det_path.parent.mkdir(parents=True, exist_ok=True)
    table = detections_table(swath, detect(swath, cfg.thresholds, cfg.daynight))
    inside = shapely.contains_xy(aoi, table["LON"].to_numpy(float), table["LAT"].to_numpy(float))
    table = table[inside]  # drop scan padding
    gpd.GeoDataFrame(table.drop(columns="footprint"), geometry=list(table["footprint"]),
                     crs="EPSG:4326").to_parquet(det_path)
    return len(table)


def failed_record(err: Exception) -> dict:
    """Manifest fields for a failure, kept so a rerun retries it."""
    return {"status": "failed", "error": f"{type(err).__name__}: {err}"}


def _run_one(pair: GranulePair, aoi_ids: list[str], cfg: Stage1Config) -> list[dict]:
    """Worker: download once, process each AOI, delete raw files; never raises."""
    global _LOGGED_IN
    base = {"granule_id": pair.granule_id, "platform": pair.platform,
            "time_key": pair.time_key}
    try:
        if not _LOGGED_IN:
            earthaccess.login(strategy="netrc")
            _LOGGED_IN = True
        obs, geo = download_pair(pair, cfg.raw_dir)
    except Exception as err:  # recorded in the manifest; rerun retries it
        return [base | {"aoi_id": a} | failed_record(err) for a in aoi_ids]
    base |= {"obs_file": str(obs), "geo_file": str(geo), "raw_on_disk": int(cfg.keep_raw)}
    records = []
    for aoi_id in aoi_ids:
        record = base | {"aoi_id": aoi_id}
        try:
            record |= process_pair(obs, geo, pair.granule_id, aoi_id, cfg)
        except Exception as err:
            record |= failed_record(err)
        records.append(record)
    if not cfg.keep_raw:
        obs.unlink(missing_ok=True)
        geo.unlink(missing_ok=True)
    return records


def run_stage1(pairs: list[GranulePair], cfg: Stage1Config, n_jobs: int = 4) -> pd.DataFrame:
    """Process granule pairs in parallel, skipping AOIs already finished for a granule.

    Parameters
    ----------
    pairs : list of GranulePair
        From ``vifdahl.granules.search_pairs``.
    cfg : Stage1Config
        Shared settings.
    n_jobs : int, optional
        Parallel workers (each holds up to one granule pair, ~0.4 GB, on disk).

    Returns
    -------
    pandas.DataFrame
        The manifest after this run.
    """
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.write()
    manifest = Manifest(cfg.workdir / "manifest.sqlite")
    finished = manifest.finished()
    todo = [(p, [a for a in cfg.aois if (p.granule_id, a) not in finished]) for p in pairs]
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_run_one)(p, aoi_ids, cfg) for p, aoi_ids in todo if aoi_ids)
    for records in results:
        for record in records:
            manifest.record(record)
            print(record["granule_id"], record["aoi_id"], record["status"],
                  record.get("n_detections", ""), record.get("error") or "", flush=True)
    return manifest.to_frame()
