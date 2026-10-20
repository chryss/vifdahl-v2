"""Rerun stage 1 detection on a finished run from saved crops or raw files; no downloads."""

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from vifdahl.manifest import Manifest
from vifdahl.readers import load_swath
from vifdahl.stage1 import Stage1Config, failed_record, process_pair, write_detections


def _exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _source(row: dict) -> str | None:
    """``"crop"``, ``"raw"`` or None: what this granule-AOI can be redetected from."""
    if _exists(row["crop_path"]):
        return "crop"
    if _exists(row["obs_file"]) and _exists(row["geo_file"]):
        return "raw"
    return None


def _redetect_one(row: dict, cfg: Stage1Config) -> dict:
    """Worker: rebuild one granule-AOI's detections from crop or raw files; never raises."""
    try:
        if _source(row) == "raw":
            return row | process_pair(Path(row["obs_file"]), Path(row["geo_file"]),
                                      row["granule_id"], row["aoi_id"], cfg) | {"error": None}
        swath = load_swath(row["crop_path"])
        n_detections = write_detections(swath, Path(row["detections_path"]),
                                        cfg.aois[row["aoi_id"]], cfg)
        return row | {"n_detections": n_detections, "error": None}
    except Exception as err:  # recorded in the manifest; rerun retries it
        return row | failed_record(err)


def redetect(cfg: Stage1Config, n_jobs: int = 4) -> pd.DataFrame:
    """Rerun detection on a finished run; no downloads.

    Each granule-AOI uses its saved crop if present, else the whole raw files
    if still on disk. Rows with neither are left unchanged and reported.

    Parameters
    ----------
    cfg : Stage1Config
        Settings for the rerun; ``workdir`` must hold an earlier stage 1 run.
    n_jobs : int, optional
        Parallel workers.

    Returns
    -------
    pandas.DataFrame
        The manifest after this run.
    """
    manifest = Manifest(cfg.workdir / "manifest.sqlite")
    frame = manifest.to_frame()
    frame = frame.astype(object).where(frame.notna(), None)  # plain Python values for sqlite
    candidates = [r for r in frame.to_dict("records")
                  if r["aoi_id"] in cfg.aois and r["status"] != "no_coverage"]
    rows = [r for r in candidates if _source(r)]
    print(f"redetect: {len(rows)} granule-AOI pairs from crops or raw files; "
          f"{len(candidates) - len(rows)} without either, left unchanged", flush=True)
    if not rows:
        return manifest.to_frame()
    cfg.write()
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_redetect_one)(r, cfg) for r in rows)
    for record in results:
        manifest.record(record)
        print(record["granule_id"], record["aoi_id"], record["status"],
              record["n_detections"], record.get("error") or "", flush=True)
    return manifest.to_frame()
