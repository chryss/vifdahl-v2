"""
Quick look at the daily GeoPackages of one event from config/events.toml.

One panel per local date (columns) and period (rows: day, night). Fused H/L
areas are filled, A pixels are marked from the points layer. Axes are km from
the event center, in the same CRS used to build the AOI, so the AOI is an
upright rectangle.

Usage: pixi run python scripts/plot_event.py mukluk [--gpkg-dir DIR] [--out PNG]
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from vifdahl.events import FireEvent, load_events

OUTPUT_ROOT = Path("output")
PERIODS = ("day", "night")
COLORS = {"H": "gold", "L": "orangered", "A": "cyan"}
DATE_RE = re.compile(r"_(\d{8})\.gpkg$")
PANEL_SIZE = (4.5, 3.8)  # inches per panel


def read_layers(gpkg_dir: Path, event: FireEvent) -> dict[str, gpd.GeoDataFrame]:
    """Read the fused and points layers of all daily GeoPackages, in the event CRS.

    Parameters
    ----------
    gpkg_dir : pathlib.Path
        Directory holding ``*_YYYYMMDD.gpkg`` files.
    event : FireEvent
        Event whose CRS is used for reprojection.

    Returns
    -------
    dict of str to geopandas.GeoDataFrame
        ``fused`` and ``points``, each with a ``LocalDate`` column.
    """
    files = sorted(p for p in gpkg_dir.glob("*.gpkg") if DATE_RE.search(p.name))
    if not files:
        raise FileNotFoundError(f"no *_YYYYMMDD.gpkg files in {gpkg_dir}")
    layers = {}
    for layer in ("fused", "points"):
        frames = [gpd.read_file(f, layer=layer) for f in files]
        layers[layer] = pd.concat(frames, ignore_index=True).to_crs(event.crs)
    return layers


def to_km(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Scale projected coordinates from meters to km."""
    return gdf.set_geometry(gdf.geometry.scale(1e-3, 1e-3, origin=(0, 0)))


def plot_event(event: FireEvent, layers: dict[str, gpd.GeoDataFrame]) -> plt.Figure:
    """Draw the day/night by local date panel grid.

    Parameters
    ----------
    event : FireEvent
        Event, for the AOI outline and title.
    layers : dict of str to geopandas.GeoDataFrame
        Output of ``read_layers``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fused, points = to_km(layers["fused"]), to_km(layers["points"])
    aoi = to_km(gpd.GeoDataFrame(geometry=[event.aoi()], crs="EPSG:4326").to_crs(event.crs))
    dates = sorted(set(fused["LocalDate"]) | set(points["LocalDate"]))
    fig, axes = plt.subplots(
        len(PERIODS), len(dates), squeeze=False, sharex=True, sharey=True,
        figsize=(PANEL_SIZE[0] * len(dates), PANEL_SIZE[1] * len(PERIODS)),
    )
    for row, period in enumerate(PERIODS):
        for col, date in enumerate(dates):
            ax = axes[row, col]
            aoi.boundary.plot(ax=ax, color="black", linewidth=0.8)
            f = fused[(fused["LocalDate"] == date) & (fused["Period"] == period)]
            for cls in ("L", "H"):
                if (f["Class"] == cls).any():
                    f[f["Class"] == cls].plot(ax=ax, color=COLORS[cls], linewidth=0)
            p = points[(points["LocalDate"] == date) & (points["Period"] == period)]
            if (p["Type"] == "A").any():
                p[p["Type"] == "A"].plot(ax=ax, color=COLORS["A"], marker="x", markersize=12)
            counts = p["Type"].value_counts().reindex(["H", "L", "A"], fill_value=0)
            ax.set_title(f"{date} {period}  H {counts.H}  L {counts.L}  A {counts.A}",
                         fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlabel("km east of center" if row == len(PERIODS) - 1 else "")
            ax.set_ylabel("km north of center" if col == 0 else "")
    fig.suptitle(f"{event.name}: fused H (gold) / L (red), A = cyan x; "
                 "pixel counts from points layer")
    fig.tight_layout()
    return fig


def main() -> None:
    """Parse arguments, plot, and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("event", help="event key in config/events.toml")
    parser.add_argument("--gpkg-dir", type=Path,
                        help="default: output/<event>_quicklook")
    parser.add_argument("--out", type=Path, help="default: <gpkg-dir>/quicklook.png")
    args = parser.parse_args()

    event = load_events()[args.event]
    gpkg_dir = args.gpkg_dir or OUTPUT_ROOT / f"{event.key}_quicklook"
    out = args.out or gpkg_dir / "quicklook.png"
    fig = plot_event(event, read_layers(gpkg_dir, event))
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
