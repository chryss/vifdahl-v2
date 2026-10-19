"""
Readers that turn VIIRS I-band granules into a common swath Dataset.

Every reader returns an ``xarray.Dataset`` with dimensions ``("y", "x")``
(along-track, cross-track) and the variables and attributes listed in
``SWATH_VARIABLES`` and ``SWATH_ATTRS``. Brightness temperatures are in K
with fill values as NaN; ``anomalous`` is built from format-specific
quality flags so that ``vifdahl.algorithm`` stays format-agnostic.
"""

from pathlib import Path

import xarray as xr

SWATH_VARIABLES = {
    "i4_bt": "I4 brightness temperature (K)",
    "i5_bt": "I5 brightness temperature (K)",
    "anomalous": "anomalous I4 pixel (bool)",
    "lat": "latitude (degrees north)",
    "lon": "longitude (degrees east)",
    "sza": "solar zenith angle (degrees)",
    "vza": "sensor zenith angle (degrees)",
}
SWATH_ATTRS = {
    "platform": "satellite short name, e.g. 'NPP', 'N20', 'N21'",
    "start_time": "granule start time, ISO 8601 UTC",
    "orbit": "orbit number (groups granules into overpasses)",
    "ascending_node": "True for a northbound pass",
    "source": "'sdr' or 'l1b'",
}


def save_swath(swath: xr.Dataset, path: str | Path) -> None:
    """Write a swath Dataset to NetCDF (bool attributes stored as 0/1)."""
    out = swath.copy()
    out.attrs = {k: int(v) if isinstance(v, bool) else v for k, v in swath.attrs.items()}
    out.to_netcdf(path, engine="h5netcdf")


def load_swath(path: str | Path) -> xr.Dataset:
    """Read a swath Dataset written by ``save_swath``."""
    swath = xr.load_dataset(path, engine="h5netcdf")
    swath.attrs["ascending_node"] = bool(swath.attrs["ascending_node"])
    swath["anomalous"] = swath["anomalous"].astype(bool)
    return swath


def validate_swath(swath: xr.Dataset) -> None:
    """Raise ``ValueError`` if ``swath`` does not follow the common schema.

    Parameters
    ----------
    swath : xarray.Dataset
        Dataset produced by a reader.
    """
    missing = [v for v in SWATH_VARIABLES if v not in swath.data_vars]
    missing += [a for a in SWATH_ATTRS if a not in swath.attrs]
    if missing:
        raise ValueError(f"swath is missing required fields: {missing}")
    for name in SWATH_VARIABLES:
        if swath[name].dims != ("y", "x"):
            raise ValueError(f"{name} has dims {swath[name].dims}, expected ('y', 'x')")
