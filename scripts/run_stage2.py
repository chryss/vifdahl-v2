"""Run stage 2 (detections → daily GeoPackages + summary.csv) for a run config TOML."""

import argparse
from pathlib import Path

import pandas as pd

from vifdahl.runconfig import load_run, resolve_aois
from vifdahl.stage2 import run_stage2


def main(config_path: Path) -> None:
    """Write a run's daily GeoPackages and summary table.

    Parameters
    ----------
    config_path : pathlib.Path
        Run config TOML (see ``config/run_template.toml``); the same one used
        for stage 1.
    """
    run = load_run(config_path)
    _, start, end = resolve_aois(run)
    dates = pd.date_range(start, end) if start and end else None
    workdir = Path(run["data"]["workdir"]) / run["name"]
    written = run_stage2(workdir, fuse_by=run.get("output", {}).get("fuse_by", "all"),
                         dates=dates, platforms=run["data"]["platforms"])
    for aoi_id, paths in written.items():
        print(f"{aoi_id}: {len(paths)} daily files")
    print("summary:", workdir / "gpkg" / "summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="run config TOML")
    main(parser.parse_args().config)
