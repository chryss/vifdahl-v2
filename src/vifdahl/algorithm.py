"""
VIFDAHL fire-pixel classification from VIIRS I4/I5 brightness temperatures.

The classification is format-agnostic: it takes brightness temperatures in
kelvin (with fill values already converted to NaN) and a precomputed
"anomalous" mask. Building that mask depends on the quality flags of the
input format (SDR HDF5 vs. L1B NetCDF) and lives in ``vifdahl.readers``.

Ported from satfire/viirsifire.py (Waigl, 2016; 2019).
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Thresholds:
    """Brightness temperature (K) and normalized-difference thresholds.

    Defaults: 2019 operational setting (``min_i5_special=305``).
    """

    min_i4_day: float = 320.0
    min_i4_night: float = 290.0
    min_i5_day: float = 285.0
    min_i5_special: float = 305.0
    cloud_i5_night: float = 265.0
    ndiff_hot: float = 0.05
    ndiff_warm: float = 0.02
    ndiff_warm_special: float = 0.015


DEFAULT_THRESHOLDS = Thresholds()
PUBLISHED_THRESHOLDS = Thresholds(min_i5_special=312.0)


class FireMasks(NamedTuple):
    """Boolean fire-condition rasters.

    ``hot`` and ``warm`` are mutually exclusive. ``anomalous`` excludes
    ``hot`` but, as in the original algorithm, may overlap with ``warm``.
    """

    anomalous: NDArray[np.bool_]
    hot: NDArray[np.bool_]
    warm: NDArray[np.bool_]


def normalized_difference(i4_bt: ArrayLike, i5_bt: ArrayLike) -> NDArray[np.floating]:
    """Return the normalized I4/I5 brightness temperature difference.

    Parameters
    ----------
    i4_bt, i5_bt : array_like
        I4 and I5 brightness temperatures in K.

    Returns
    -------
    numpy.ndarray
        ``(I4 - I5) / (I4 + I5)``.
    """
    i4_bt = np.asarray(i4_bt, dtype=float)
    i5_bt = np.asarray(i5_bt, dtype=float)
    return (i4_bt - i5_bt) / (i4_bt + i5_bt)


def classify(
    i4_bt: ArrayLike,
    i5_bt: ArrayLike,
    anomalous: ArrayLike,
    is_day: bool | ArrayLike,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> FireMasks:
    """Classify pixels into anomalous, hot and warm fire conditions.

    Parameters
    ----------
    i4_bt, i5_bt : array_like
        I4 and I5 brightness temperatures in K. Fill values must be NaN.
    anomalous : array_like of bool
        Pixels with anomalous I4 values (e.g. saturation or folding),
        derived from format-specific quality flags.
    is_day : bool or array_like of bool
        Day/night switch. A scalar applies one regime to the whole scene
        (the original per-scene behavior); an array applies it per pixel.
    thresholds : Thresholds, optional
        Threshold set.

    Returns
    -------
    FireMasks
        Boolean rasters ``anomalous``, ``hot`` and ``warm``.
    """
    th = thresholds
    i4_bt = np.asarray(i4_bt, dtype=float)
    i5_bt = np.asarray(i5_bt, dtype=float)
    anomalous = np.asarray(anomalous, dtype=bool)
    is_day = np.asarray(is_day, dtype=bool)
    ndiff = normalized_difference(i4_bt, i5_bt)

    hot_day = (
        (ndiff >= th.ndiff_hot)
        & (i4_bt > th.min_i4_day)
        & (i5_bt > th.min_i5_day)
        & ~anomalous
    )
    warm_day = (
        ((ndiff >= th.ndiff_warm)
         | ((ndiff >= th.ndiff_warm_special) & (i5_bt >= th.min_i5_special)))
        & (i5_bt > th.min_i5_day)
        & (i4_bt > th.min_i4_day)
        & ~hot_day
    )

    cloud_night = i5_bt < th.cloud_i5_night
    hot_night = (ndiff >= th.ndiff_hot) & ~cloud_night & ~anomalous
    warm_night = (
        (ndiff >= th.ndiff_warm_special)
        & (i4_bt > th.min_i4_night)
        & ~cloud_night
        & ~hot_night
    )

    hot = np.where(is_day, hot_day, hot_night)
    warm = np.where(is_day, warm_day, warm_night)
    return FireMasks(np.broadcast_to(anomalous, hot.shape).copy(), hot, warm)
