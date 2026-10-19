"""
Run VIFDAHL on a swath Dataset and tabulate the detected fire pixels.

Ported from the 2019 product-generation notebook
(legacy/2019_pipeline/VIIRSAlaskaFires-ProdGen.ipynb).
"""

import numpy as np
import pandas as pd
import xarray as xr

from vifdahl.algorithm import DEFAULT_THRESHOLDS, FireMasks, Thresholds, classify
from vifdahl.periods import assign_period
from vifdahl.readers import validate_swath
from vifdahl.swath import duplicate_pixels, is_day, pixel_footprints, scene_is_day

SENSOR = "VIIRS"
BANDS = "I-Bands"
TYPE_CODES = {"anomalous": "A", "hot": "H", "warm": "L"}
COLUMNS = [
    "Platform", "Sensor", "Bands", "Orbit", "Granule", "DATE", "GMT", "Type",
    "AscNode", "LAT", "LON", "TB4", "TB5", "SolZenithAng", "SatZenithAng",
    "LocalDate", "Period", "row", "col", "footprint",
]


def detect(
    swath: xr.Dataset,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    per_pixel_daynight: bool = False,
) -> FireMasks:
    """Classify fire pixels in one granule, excluding scan-overlap duplicates.

    Parameters
    ----------
    swath : xarray.Dataset
        Granule (or crop) following the schema in ``vifdahl.readers``.
    thresholds : Thresholds, optional
        Threshold set.
    per_pixel_daynight : bool, optional
        If False (default, as published), the whole scene is classified as
        night unless every pixel has a solar zenith angle <= 90°. If True,
        day/night is decided per pixel.

    Returns
    -------
    FireMasks
        Boolean rasters with duplicated pixels set to False.
    """
    validate_swath(swath)
    sza = swath["sza"].values
    day = is_day(sza) if per_pixel_daynight else scene_is_day(sza)
    masks = classify(
        swath["i4_bt"].values,
        swath["i5_bt"].values,
        swath["anomalous"].values,
        day,
        thresholds,
    )
    dup = duplicate_pixels(swath["lat"].values, bool(swath.attrs["ascending_node"]))
    return FireMasks(*(m & ~dup for m in masks))


def detections_table(swath: xr.Dataset, masks: FireMasks) -> pd.DataFrame:
    """Tabulate detected pixels with attributes and footprint polygons.

    Column names follow the original VIFDAHL product files.

    Parameters
    ----------
    swath : xarray.Dataset
        The granule the masks were computed from.
    masks : FireMasks
        Output of ``detect``.

    Returns
    -------
    pandas.DataFrame
        One row per detected pixel and fire type; ``footprint`` holds a
        shapely polygon in EPSG:4326.
    """
    start = pd.Timestamp(swath.attrs["start_time"])
    lons, lats = swath["lon"].values, swath["lat"].values
    frames = []
    for name, code in TYPE_CODES.items():
        rows, cols = np.nonzero(getattr(masks, name))
        if len(rows) == 0:
            continue
        frames.append(pd.DataFrame({
            "Platform": swath.attrs["platform"],
            "Sensor": SENSOR,
            "Bands": BANDS,
            "Orbit": int(swath.attrs["orbit"]),
            "Granule": start.isoformat(),
            "DATE": start.strftime("%Y-%m-%d"),
            "GMT": start.strftime("%H%M"),
            "Type": code,
            "AscNode": "Y" if swath.attrs["ascending_node"] else "N",
            "LAT": lats[rows, cols],
            "LON": lons[rows, cols],
            "TB4": swath["i4_bt"].values[rows, cols],
            "TB5": swath["i5_bt"].values[rows, cols],
            "SolZenithAng": swath["sza"].values[rows, cols],
            "SatZenithAng": swath["vza"].values[rows, cols],
            "row": rows,
            "col": cols,
            "footprint": pixel_footprints(lons, lats, rows, cols),
        }))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    periods = assign_period(np.repeat(start, len(table)), table["LON"])
    table[["LocalDate", "Period"]] = periods.to_numpy()
    return table[COLUMNS]
