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
    """Settings shared by all granules of a run or event."""

    aoi: BaseGeometry
    workdir: Path
    thresholds: Thresholds = field(default_factory=lambda: DEFAULT_THRESHOLDS)
    per_pixel_daynight: bool = False
    keep_raw: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.workdir / "raw"

    def write(self) -> None:
        """Record the settings in ``workdir/stage1_config.json``."""
        info = {"aoi_wkt": self.aoi.wkt, "thresholds": asdict(self.thresholds),
                "per_pixel_daynight": self.per_pixel_daynight, "keep_raw": self.keep_raw}
        (self.workdir / "stage1_config.json").write_text(json.dumps(info, indent=2))


def process_pair(obs: Path, geo: Path, granule_id: str, cfg: Stage1Config) -> dict:
    """Crop, detect and save one granule pair from local files.

    Returns
    -------
    dict
        Manifest record (without raw-file fields).
    """
    swath = read_l1b(obs, geo, aoi=cfg.aoi)
    if swath is None:
        return {"status": "no_coverage", "n_detections": 0}
    crop_path = cfg.workdir / "crops" / f"{granule_id}.nc"
    det_path = cfg.workdir / "detections" / f"{granule_id}.parquet"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    det_path.parent.mkdir(parents=True, exist_ok=True)
    save_swath(swath, crop_path)
    table = detections_table(swath, detect(swath, cfg.thresholds, cfg.per_pixel_daynight))
    inside = shapely.contains_xy(cfg.aoi, table["LON"].to_numpy(float), table["LAT"].to_numpy(float))
    table = table[inside]  # drop scan padding
    gpd.GeoDataFrame(table.drop(columns="footprint"), geometry=list(table["footprint"]),
                     crs="EPSG:4326").to_parquet(det_path)
    return {"status": "done", "n_detections": len(table),
            "crop_path": str(crop_path), "detections_path": str(det_path)}


def _run_one(pair: GranulePair, cfg: Stage1Config) -> dict:
    """Worker: download, process, delete raw files; never raises."""
    global _LOGGED_IN
    record = {"granule_id": pair.granule_id, "platform": pair.platform,
              "time_key": pair.time_key}
    try:
        if not _LOGGED_IN:
            earthaccess.login(strategy="netrc")
            _LOGGED_IN = True
        obs, geo = download_pair(pair, cfg.raw_dir)
        record |= {"obs_file": str(obs), "geo_file": str(geo)}
        record |= process_pair(obs, geo, pair.granule_id, cfg)
        if not cfg.keep_raw:
            obs.unlink(missing_ok=True)
            geo.unlink(missing_ok=True)
        record["raw_on_disk"] = int(cfg.keep_raw)
    except Exception as err:  # recorded in the manifest; rerun retries it
        record |= {"status": "failed", "error": f"{type(err).__name__}: {err}"}
    return record


def run_stage1(pairs: list[GranulePair], cfg: Stage1Config, n_jobs: int = 4) -> pd.DataFrame:
    """Process granule pairs in parallel, skipping those already finished.

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
    todo = [p for p in pairs if p.granule_id not in finished]
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_run_one)(p, cfg) for p in todo)
    for record in results:
        manifest.record(record)
        print(record["granule_id"], record["status"], record.get("n_detections", ""),
              record.get("error") or "", flush=True)
    return manifest.to_frame()
