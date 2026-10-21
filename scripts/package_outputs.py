"""
Zip each run's daily GeoPackages and summary.csv for distribution.

One archive per run: ``<out-dir>/VIFDAHL_<run>.zip`` containing
``<run>/summary.csv`` and ``<run>/<aoi_id>/*.gpkg``. Prints a Markdown table
row per archive (size, AOIs, files, dates) for docs/index.md.

Usage: pixi run python scripts/package_outputs.py [RUN ...] [--out-dir DIR]
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path("output")
RUNS = ("eastsiberia_2020", "AK_boreal_2022", "mukluk_fullrun")
OUT_DIR = OUTPUT_ROOT / "packages"


def package_run(run: str, out_dir: Path) -> dict:
    """Write one run's archive and return a description of it.

    Parameters
    ----------
    run : str
        Folder name under ``output/`` with a ``gpkg/`` subfolder.
    out_dir : pathlib.Path
        Where the archive goes.

    Returns
    -------
    dict
        Archive path, size in MB, AOI IDs, file count, first and last date.
    """
    gpkg_root = OUTPUT_ROOT / run / "gpkg"
    files = sorted(gpkg_root.glob("*/*.gpkg"))
    summary = gpkg_root / "summary.csv"
    archive = out_dir / f"VIFDAHL_{run}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary, f"{run}/summary.csv")
        for f in files:
            zf.write(f, f"{run}/{f.parent.name}/{f.name}")
    dates = pd.read_csv(summary)["LocalDate"]
    return {"archive": archive, "size_mb": archive.stat().st_size / 1e6,
            "aois": sorted({f.parent.name for f in files}), "n_files": len(files),
            "first": dates.min(), "last": dates.max()}


def main() -> None:
    """Parse arguments and package the runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("runs", nargs="*", default=RUNS, help=f"default: {' '.join(RUNS)}")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        info = package_run(run, args.out_dir)
        print(f"| `{info['archive'].name}` | {info['first']} – {info['last']} | "
              f"{', '.join(info['aois'])} | {info['n_files']} | {info['size_mb']:.0f} MB |")


if __name__ == "__main__":
    main()
