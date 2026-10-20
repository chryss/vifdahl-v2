"""Run stage 1 (granule → crop + detections) for a run config TOML."""

import argparse
from dataclasses import fields
from pathlib import Path

import earthaccess
import pandas as pd

from vifdahl.algorithm import Thresholds
from vifdahl.granules import search_aois
from vifdahl.periods import utc_window
from vifdahl.redetect import redetect
from vifdahl.runconfig import load_run, resolve_aois
from vifdahl.stage1 import Stage1Config, run_stage1


def main(config_path: Path, dry_run: bool = False, redetect_only: bool = False) -> None:
    """Search granules for the run and process them.

    Parameters
    ----------
    config_path : pathlib.Path
        Run config TOML (see ``config/run_template.toml``).
    dry_run : bool, optional
        Only search and report the granule count and download size.
    redetect_only : bool, optional
        Skip search and download; rerun detection on the run's saved crops.
    """
    run = load_run(config_path)
    aois, start, end = resolve_aois(run)
    data = run["data"]
    algo = run.get("algorithm", {})
    names = {f.name for f in fields(Thresholds)}
    thresholds = Thresholds(**{k: v for k, v in algo.items() if k in names})
    cfg = Stage1Config(aois=aois, workdir=Path(data["workdir"]) / run["name"],
                       thresholds=thresholds,
                       daynight=algo.get("daynight", "scene"),
                       keep_raw=data.get("keep_raw", False),
                       keep_crops="event" in run["aoi"] or data.get("keep_raw", False))
    if redetect_only:
        report(redetect(cfg, n_jobs=data.get("n_jobs", 4)))
        return

    w = min(g.bounds[0] for g in aois.values())
    e = max(g.bounds[2] for g in aois.values())
    t0, t1 = utc_window(start, end, w, e)
    earthaccess.login(strategy="netrc")
    pairs = search_aois(aois, t0, t1, tuple(data["platforms"]))
    size_gb = sum(p.size_mb for p in pairs) / 1024
    print(f"{run['name']}: {start} to {end} local, UTC {t0:%Y-%m-%d %H:%M} to "
          f"{t1:%Y-%m-%d %H:%M}; {len(pairs)} granule pairs, {size_gb:.1f} GB", flush=True)
    if dry_run:
        return

    report(run_stage1(pairs, cfg, n_jobs=data.get("n_jobs", 4)))


def report(manifest: pd.DataFrame) -> None:
    """Print status counts and total detections from a manifest."""
    print(manifest["status"].value_counts().to_string())
    print("detections:", int(manifest["n_detections"].fillna(0).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="run config TOML")
    parser.add_argument("--dry-run", action="store_true",
                        help="search only; report granule count and size")
    parser.add_argument("--redetect", action="store_true",
                        help="rerun detection on saved crops; no search or download")
    args = parser.parse_args()
    main(args.config, args.dry_run, args.redetect)
