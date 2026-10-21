"""
CFFDRS fire weather indices (ERA5 0.25° daily grid) at fire locations, and
slide-format scatter plots.

The grid files (``data/aux/arcticfire_<year>.nc``) have longitudes 0–360 and
one field per day; detections are matched by local solar date.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from vifdahl.style import CLASS_COLORS, CLASS_LABELS, SLIDE_RC, SLIDE_SIZE

FWI_VARS = ("bui", "dmc", "isi", "dc")
FWI_LABELS = {"bui": "Buildup Index (BUI)", "dmc": "Duff Moisture Code (DMC)",
              "isi": "Initial Spread Index (ISI)", "dc": "Drought Code (DC)",
              "area_km2": "fused area (km²)", "TB4_max": "max TB4 (K)"}


def sample_nearest(table: pd.DataFrame, nc_path: str | Path,
                   variables: Sequence[str] = FWI_VARS) -> pd.DataFrame:
    """Nearest grid cell values for each row's ``lon``, ``lat`` and ``date``.

    Parameters
    ----------
    table : pandas.DataFrame
        Columns ``lon``, ``lat`` (degrees, −180–180 or 0–360) and ``date``.
    nc_path : str or pathlib.Path
        Fire weather NetCDF with dimensions ``time``, ``latitude``, ``longitude``.
    variables : sequence of str, optional
        Variables to sample.

    Returns
    -------
    pandas.DataFrame
        ``table`` with one added column per variable (NaN where ``date`` is
        outside the file's time range).
    """
    with xr.open_dataset(nc_path) as ds:
        times = pd.DatetimeIndex(table["date"])
        inside = times.isin(ds.indexes["time"])
        idx = {"time": xr.DataArray(times[inside], dims="p"),
               "latitude": xr.DataArray(table["lat"].to_numpy()[inside], dims="p"),
               "longitude": xr.DataArray(table["lon"].to_numpy()[inside] % 360, dims="p")}
        values = ds[list(variables)].sel(idx, method="nearest").load()
    out = table.copy()
    for v in variables:
        out[v] = np.nan
        out.loc[inside, v] = values[v].to_numpy()
    return out


def plot_slide_hist(table: pd.DataFrame, var: str = "bui", rows: str | None = None,
                    cols: str | None = "Period", bin_width: float = 5.0,
                    title: str = "") -> plt.Figure:
    """16:9 grid of normalized ``var`` histograms, H and L overlaid, with median lines.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows with ``Class`` (H or L), ``var`` and the facet columns.
    var : str, optional
        Column to histogram.
    rows, cols : str, optional
        Facet columns; ``None`` for a single row or column.
    bin_width : float, optional
        Histogram bin width.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    row_vals = sorted(table[rows].unique(), reverse=True) if rows else [None]
    col_vals = sorted(table[cols].unique()) if cols else [None]
    bins = np.arange(0, table[var].max() + bin_width, bin_width)
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(len(row_vals), len(col_vals), figsize=SLIDE_SIZE,
                                 squeeze=False, sharex=True, layout="constrained")
        for i, r in enumerate(row_vals):
            for j, c in enumerate(col_vals):
                ax = axes[i, j]
                sel = table[((table[rows] == r) if rows else True)
                            & ((table[cols] == c) if cols else True)]
                for cls, color in CLASS_COLORS.items():
                    vals = sel.loc[sel["Class"] == cls, var].dropna()
                    if vals.empty:
                        continue
                    ax.hist(vals, bins=bins, density=True, color=color, alpha=0.45)
                    ax.hist(vals, bins=bins, density=True, histtype="step", color=color,
                            linewidth=2)
                    ax.axvline(vals.median(), color=color, linestyle="--", linewidth=2.5)
                counts = sel["Class"].value_counts()
                ax.set_title(" · ".join(str(v) for v in (r, c) if v is not None)
                             + f"  (H {counts.get('H', 0):,}, L {counts.get('L', 0):,})")
                ax.set_yticks([])
        for ax in axes[-1]:
            ax.set_xlabel(FWI_LABELS.get(var, var))
        for ax in axes[:, 0]:
            ax.set_ylabel("density")
        handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[k], label=CLASS_LABELS[k])
                   for k in CLASS_COLORS]
        handles.append(plt.Line2D([], [], color="0.3", linestyle="--", linewidth=2.5,
                                  label="median"))
        fig.legend(handles=handles, loc="outside upper right", frameon=False, ncols=3)
        fig.suptitle(title, x=0.01, ha="left")
    return fig


def _scatter_classes(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, seed: int = 0) -> None:
    """H and L in one call, in random order, so neither class hides the other."""
    df = df.sample(frac=1, random_state=seed)
    xv = mdates.date2num(df[x]) if x == "date" else df[x]
    ax.scatter(xv, df[y], c=df["Class"].map(CLASS_COLORS), s=6, alpha=0.5,
               linewidths=0, rasterized=True)


def plot_slide_grid(fused: pd.DataFrame, x: str, y: str, rows: str | None = None,
                    cols: str | None = "Period", logy: bool = False,
                    title: str = "") -> plt.Figure:
    """16:9 scatter grid of fused areas, H/L colored, faceted by ``rows`` × ``cols``.

    Parameters
    ----------
    fused : pandas.DataFrame
        Fused areas with ``Class`` and the plotted columns (``date`` allowed for x).
    x, y : str
        Columns to plot.
    rows, cols : str, optional
        Facet columns; ``None`` for a single row or column.
    logy : bool, optional
        Log-scaled y axis.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    row_vals = sorted(fused[rows].unique(), reverse=True) if rows else [None]
    col_vals = sorted(fused[cols].unique()) if cols else [None]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(len(row_vals), len(col_vals), figsize=SLIDE_SIZE,
                                 squeeze=False, sharex=True, sharey=True, layout="constrained")
        for i, r in enumerate(row_vals):
            for j, c in enumerate(col_vals):
                ax = axes[i, j]
                sel = fused[((fused[rows] == r) if rows else True)
                            & ((fused[cols] == c) if cols else True)]
                _scatter_classes(ax, sel, x, y)
                ax.set_title(" · ".join(str(v) for v in (r, c) if v is not None)
                             + f"  (n = {len(sel):,})")
                if logy:
                    ax.set_yscale("log")
        for ax in axes[-1]:
            ax.set_xlabel("local solar date" if x == "date" else FWI_LABELS.get(x, x))
            if x == "date":
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for ax in axes[:, 0]:
            ax.set_ylabel(FWI_LABELS.get(y, y))
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                              color=color, label=CLASS_LABELS[k])
                   for k, color in CLASS_COLORS.items()]
        fig.legend(handles=handles, loc="outside upper right", frameon=False, ncols=2)
        fig.suptitle(title, x=0.01, ha="left")
    return fig
