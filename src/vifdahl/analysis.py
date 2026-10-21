"""Load daily GeoPackage layers of a run and plot brightness temperature summaries."""

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.lines import Line2D

from vifdahl.aggregate import POLAR_CRS
from vifdahl.algorithm import normalized_difference
from vifdahl.style import CLASS_COLORS, CLASS_LABELS, GROUP_LINESTYLES, SLIDE_RC, SLIDE_SIZE

POINT_COLUMNS = ["Platform", "Granule", "Type", "LAT", "LON", "TB4", "TB5",
                 "SolZenithAng", "SatZenithAng", "LocalDate", "Period"]
SUN = ("sun up", "sun down")  # pixel solar zenith angle <= 90° / > 90°
BINS = {"TB4": np.arange(270, 370, 1.0), "TB5": np.arange(240, 345, 1.0),
        "ndiff": np.arange(0.0, 0.2, 0.002)}
MIN_HIST_N = 50  # smaller series are left out of histograms
LABELS = {"TB4": "TB4 (K)", "TB5": "TB5 (K)", "ndiff": "(TB4 − TB5) / (TB4 + TB5)"}


def load_points(gpkg_root: str | Path, aoi_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Read the ``points`` layer (attributes only) of all daily GeoPackages of a run.

    Parameters
    ----------
    gpkg_root : str or pathlib.Path
        Run output folder with one subfolder per AOI (``<run>/gpkg``).
    aoi_ids : sequence of str, optional
        AOI subfolders to read; default all.

    Returns
    -------
    pandas.DataFrame
        One row per detection, with added ``aoi_id``, ``date`` (local solar
        date as datetime), ``ndiff`` and ``sun`` (see ``SUN``).
    """
    root = Path(gpkg_root)
    dirs = [root / a for a in aoi_ids] if aoi_ids else sorted(p for p in root.iterdir() if p.is_dir())
    frames = []
    for d in dirs:
        for f in sorted(d.glob("*.gpkg")):
            df = pyogrio.read_dataframe(f, layer="points", columns=POINT_COLUMNS,
                                        read_geometry=False)
            frames.append(df.assign(aoi_id=d.name))
    points = pd.concat(frames, ignore_index=True)
    return points.assign(
        date=pd.to_datetime(points["LocalDate"]),
        ndiff=normalized_difference(points["TB4"], points["TB5"]),
        sun=np.where(points["SolZenithAng"] <= 90, SUN[0], SUN[1]),
    )


def load_fused(gpkg_root: str | Path, aoi_ids: Sequence[str] | None = None) -> gpd.GeoDataFrame:
    """Read the ``fused`` layer of all daily GeoPackages of a run.

    Parameters
    ----------
    gpkg_root : str or pathlib.Path
        Run output folder with one subfolder per AOI (``<run>/gpkg``).
    aoi_ids : sequence of str, optional
        AOI subfolders to read; default all.

    Returns
    -------
    geopandas.GeoDataFrame
        Fused areas (EPSG:4326) with added ``aoi_id``, ``date``, and centroid
        ``lon``/``lat`` (computed in EPSG:6931).
    """
    root = Path(gpkg_root)
    dirs = [root / a for a in aoi_ids] if aoi_ids else sorted(p for p in root.iterdir() if p.is_dir())
    frames = [gpd.read_file(f, layer="fused").assign(aoi_id=d.name)
              for d in dirs for f in sorted(d.glob("*.gpkg"))]
    fused = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    centroids = fused.to_crs(POLAR_CRS).centroid.to_crs(fused.crs)
    return fused.assign(date=pd.to_datetime(fused["LocalDate"]),
                        lon=centroids.x.to_numpy(), lat=centroids.y.to_numpy())


def _label(value: str) -> str:
    """Display form of an AOI or group name."""
    return str(value).replace("_", " ")


def plot_histograms(
    points: pd.DataFrame,
    group: str,
    variables: Sequence[str] = ("TB4", "ndiff"),
    types: Sequence[str] = ("H", "L"),
) -> plt.Figure:
    """Normalized histograms: fire type by color, group by line style.

    Rows split by ``sun``; A pixels (unusable TB4) and series with fewer than
    ``MIN_HIST_N`` values are left out.

    Parameters
    ----------
    points : pandas.DataFrame
        Output of ``load_points``.
    group : str
        Column whose values get one line style each (e.g. biome); at most four.
    variables : sequence of str, optional
        Columns to histogram; bins from ``BINS``.
    types : sequence of str, optional
        Fire types to show.

    Returns
    -------
    matplotlib.figure.Figure
    """
    groups = sorted(points[group].unique())
    styles = dict(zip(groups, GROUP_LINESTYLES))
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(len(SUN), len(variables), figsize=SLIDE_SIZE,
                                 squeeze=False, sharex="col", layout="constrained")
        for row, sun in enumerate(SUN):
            for col, var in enumerate(variables):
                ax = axes[row, col]
                for g in groups:
                    for t in types:
                        vals = points.loc[(points[group] == g) & (points["Type"] == t)
                                          & (points["sun"] == sun), var].dropna()
                        if len(vals) >= MIN_HIST_N:
                            ax.hist(vals, bins=BINS.get(var, 50), density=True,
                                    histtype="step", linewidth=2, color=CLASS_COLORS[t],
                                    linestyle=styles[g])
                ax.set_title(sun)
                ax.set_yticks([])
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                if row == len(SUN) - 1:
                    ax.set_xlabel(LABELS.get(var, var))
            axes[row, 0].set_ylabel("density")
        handles = ([Line2D([], [], color=CLASS_COLORS[t], linewidth=3, label=CLASS_LABELS[t])
                    for t in types]
                   + [Line2D([], [], color="0.3", linewidth=2, linestyle=styles[g],
                             label=_label(g)) for g in groups])
        fig.legend(handles=handles, loc="outside upper center", ncols=len(handles),
                   frameon=False)
    return fig


def plot_tb_scatter(points: pd.DataFrame, group: str, max_points: int = 200_000,
                    seed: int = 0) -> plt.Figure:
    """TB5 vs TB4 colored by local date; rows = groups, columns = ``sun``.

    Groups with more than ``max_points`` H/L detections are randomly subsampled.
    """
    groups = sorted(points[group].unique())
    hl = points.loc[points["Type"].isin(["H", "L"])]
    norm = plt.Normalize(mdates.date2num(hl["date"].min()), mdates.date2num(hl["date"].max()))
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(len(groups), len(SUN), figsize=(13.33, 3.9 * len(groups)),
                                 squeeze=False, sharex=True, sharey=True, layout="constrained")
        for row, g in enumerate(groups):
            for col, sun in enumerate(SUN):
                sel = hl.loc[(hl[group] == g) & (hl["sun"] == sun)]
                n = len(sel)
                if n > max_points:
                    sel = sel.sample(max_points, random_state=seed)
                sc = axes[row, col].scatter(sel["TB4"], sel["TB5"], s=3, alpha=0.5,
                                            c=mdates.date2num(sel["date"]), cmap="viridis",
                                            norm=norm, rasterized=True)
                axes[row, col].set_title(f"{_label(g)} · {sun}  (n = {n:,})")
            axes[row, 0].set_ylabel("TB5 (K)")
        for ax in axes[-1]:
            ax.set_xlabel("TB4 (K)")
        cbar = fig.colorbar(sc, ax=axes, shrink=0.8, label="local date")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    return fig


def plot_date_scatter(points: pd.DataFrame, group: str, var: str = "TB4",
                      seed: int = 0) -> plt.Figure:
    """``var`` against local date, H and L colored; one panel per group (jittered by ±0.4 d)."""
    groups = sorted(points[group].unique())
    rng = np.random.default_rng(seed)
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(len(groups), 1, figsize=(13.33, 3.75 * len(groups)),
                                 squeeze=False, sharex=True, layout="constrained")
        for ax, g in zip(axes[:, 0], groups):
            for t in ("H", "L"):  # L on top, it is sparser
                sel = points.loc[(points[group] == g) & (points["Type"] == t)]
                x = mdates.date2num(sel["date"]) + rng.uniform(-0.4, 0.4, len(sel))
                ax.scatter(x, sel[var], s=2, alpha=0.4, color=CLASS_COLORS[t],
                           rasterized=True)
            ax.set_title(_label(g))
            ax.set_ylabel(LABELS.get(var, var))
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[-1, 0].set_xlabel("local solar date")
        handles = [Line2D([], [], marker="o", linestyle="", markersize=10,
                          color=CLASS_COLORS[t], label=CLASS_LABELS[t]) for t in ("H", "L")]
        fig.legend(handles=handles, loc="outside upper right", ncols=2, frameon=False)
    return fig
