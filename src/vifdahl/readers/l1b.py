"""
Reader for VIIRS I-band L1B NetCDF: VNP02IMG/VNP03IMG (SNPP), VJ102IMG/VJ103IMG
(NOAA-20), VJ202IMG/VJ203IMG (NOAA-21).
"""

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from vifdahl.readers import validate_swath

SCAN_LINES = 32
FLAG_MIN_DN = 65532  # 65532-65534: Missing_EV, Bowtie_Deleted, Cal_Fail; 65535: fill
QF_OUT_OF_RANGE = 2
QF_SATURATION = 4
# Out_of_Range also flags cold I4; only count it as anomalous if I5 is warm.
# Provisional: to be checked against the SDR rule (QF1 == 193).
ANOMALY_MIN_I5_BT = 300.0
PLATFORMS = {"Suomi-NPP": "NPP", "NOAA-20": "N20", "JPSS-1": "N20",
             "NOAA-21": "N21", "JPSS-2": "N21"}


def to_bt(dn: NDArray[np.integer], lut: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert raw I4/I5 values to brightness temperature (K) via lookup table.

    Parameters
    ----------
    dn : numpy.ndarray
        Raw uint16 values.
    lut : numpy.ndarray
        ``I0x_brightness_temperature_lut``.

    Returns
    -------
    numpy.ndarray
        Brightness temperature; NaN for fill and flag values.
    """
    dn = np.asarray(dn, dtype=np.int64)
    bt = np.asarray(lut, dtype=float)[np.clip(dn, 0, len(lut) - 1)]
    bt[(dn >= FLAG_MIN_DN) | (bt < 0)] = np.nan
    return bt


def anomaly_mask(qf4: NDArray[np.integer], i5_bt: NDArray[np.floating]) -> NDArray[np.bool_]:
    """I4 saturated, or I4 out of range with warm I5 (wrap-around on hot pixels).

    Parameters
    ----------
    qf4 : numpy.ndarray
        ``I04_quality_flags``.
    i5_bt : numpy.ndarray
        I5 brightness temperature (K).

    Returns
    -------
    numpy.ndarray of bool
    """
    saturated = (qf4 & QF_SATURATION) > 0
    out_of_range = (qf4 & QF_OUT_OF_RANGE) > 0
    with np.errstate(invalid="ignore"):
        return saturated | (out_of_range & (i5_bt > ANOMALY_MIN_I5_BT))


def _crop_slices(
    lat: NDArray, lon: NDArray, aoi: BaseGeometry, pad_scans: int
) -> dict[str, slice] | None:
    """Row/column slices covering the AOI bounds, rows extended to whole scans."""
    w, s, e, n = aoi.bounds
    rows, cols = np.nonzero((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n))
    if len(rows) == 0:
        return None
    r0 = max((rows.min() // SCAN_LINES - pad_scans) * SCAN_LINES, 0)
    r1 = min((rows.max() // SCAN_LINES + 1 + pad_scans) * SCAN_LINES, lat.shape[0])
    return {"number_of_lines": slice(r0, r1),
            "number_of_pixels": slice(cols.min(), cols.max() + 1)}


def read_l1b(
    obs_path: str | Path,
    geo_path: str | Path,
    aoi: BaseGeometry | None = None,
    pad_scans: int = 1,
) -> xr.Dataset | None:
    """Read an L1B granule pair into the common swath Dataset.

    Parameters
    ----------
    obs_path : str or pathlib.Path
        ``*02IMG`` file (observations).
    geo_path : str or pathlib.Path
        Matching ``*03IMG`` file (geolocation).
    aoi : shapely geometry, optional
        Crop to the bounding box of this geometry (EPSG:4326).
    pad_scans : int, optional
        Extra scans kept above and below the AOI, for duplicate detection.

    Returns
    -------
    xarray.Dataset or None
        Swath Dataset; None if the granule does not cover the AOI.
    """
    obs = xr.open_dataset(obs_path, group="observation_data", engine="h5netcdf",
                          mask_and_scale=False)
    geo = xr.open_dataset(geo_path, group="geolocation_data", engine="h5netcdf")
    attrs = xr.open_dataset(geo_path, engine="h5netcdf").attrs

    if aoi is not None:
        crop = _crop_slices(geo["latitude"].values, geo["longitude"].values, aoi, pad_scans)
        if crop is None:
            return None
        obs_c, geo_c = obs.isel(crop), geo.isel(crop)
    else:
        obs_c, geo_c = obs, geo

    qf4 = obs_c["I04_quality_flags"].values
    i5_bt = to_bt(obs_c["I05"].values, obs["I05_brightness_temperature_lut"].values)
    data = {
        "i4_bt": to_bt(obs_c["I04"].values, obs["I04_brightness_temperature_lut"].values),
        "i5_bt": i5_bt,
        "anomalous": anomaly_mask(qf4, i5_bt),
        "lat": geo_c["latitude"].values,
        "lon": geo_c["longitude"].values,
        "sza": geo_c["solar_zenith"].values,
        "vza": geo_c["sensor_zenith"].values,
        "i4_qf": qf4,
        "i5_qf": obs_c["I05_quality_flags"].values,
    }
    swath = xr.Dataset(
        {k: (("y", "x"), v) for k, v in data.items()},
        attrs={
            "platform": PLATFORMS.get(attrs["platform"], attrs["platform"]),
            "start_time": attrs["time_coverage_start"],
            "orbit": int(np.asarray(attrs["OrbitNumber"]).item()),
            # TODO: granules that change direction (near the pole) use the start direction
            "ascending_node": attrs["startDirection"] == "Ascending",
            "source": "l1b",
            "granule": Path(obs_path).name,
        },
    )
    validate_swath(swath)
    return swath
