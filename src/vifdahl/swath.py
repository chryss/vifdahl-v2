"""
Swath geometry helpers for VIIRS I-band granules.

Detects duplicated pixels in the scan-overlap ("bow-tie") zone from jumps
in along-track latitude, determines day/night, and constructs approximate
pixel footprints. All functions take plain arrays, independent of the
input file format.

Ported from satfire/viirsswathtools.py (Waigl, 2016).
"""

from functools import reduce

import numpy as np
import pyproj
from numpy.typing import NDArray
from shapely.geometry import Polygon

GEOD = pyproj.Geod(ellps="WGS84")
NIGHT_SZA = 90.0


def _edge_and_center_rows(mask: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Return row indices where ``mask`` is set in the first, last or center column."""
    cols = (0, -1, mask.shape[1] // 2)
    return reduce(np.union1d, [np.where(mask[:, c])[0] for c in cols])


def latitude_reversals(lats: NDArray[np.floating], ascending: bool) -> NDArray[np.intp]:
    """Find rows where along-track latitude stops progressing.

    Parameters
    ----------
    lats : numpy.ndarray
        2-D latitude array (along-track, cross-track).
    ascending : bool
        True for an ascending-node (northbound) pass.

    Returns
    -------
    numpy.ndarray
        Row indices where the latitude step is zero or reversed.
    """
    deltas = lats[:-1, :] - lats[1:, :]
    if ascending:
        deltas[deltas > 0] = 0
    else:
        deltas[deltas < 0] = 0
    return _edge_and_center_rows(deltas == 0)


def _skips_for_column(
    col: NDArray[np.floating], reversals: NDArray[np.intp], ascending: bool
) -> NDArray[np.floating]:
    """Flag rows of one column that duplicate earlier rows (1.0) or not (0.0)."""
    length = col.shape[0]
    flags = np.zeros(length)
    sign = 1.0 if ascending else -1.0
    for idx in reversals:
        step = np.argmax(sign * col[idx:] > sign * col[idx])
        if step == 0:
            flags[idx + 1:] = 1.0
            continue
        half_left = min(step // 2, idx)
        flags[idx - half_left + 1: idx + step - half_left] = 1.0
    return flags


def duplicate_pixels(lats: NDArray[np.floating], ascending: bool) -> NDArray[np.bool_]:
    """Flag pixels duplicated by scan overlap.

    Parameters
    ----------
    lats : numpy.ndarray
        2-D latitude array (along-track, cross-track).
    ascending : bool
        True for an ascending-node (northbound) pass.

    Returns
    -------
    numpy.ndarray of bool
        True where a pixel repeats ground already covered by an earlier scan.
    """
    reversals = latitude_reversals(lats, ascending)
    if len(reversals) == 0:
        return np.zeros(lats.shape, dtype=bool)
    flags = np.apply_along_axis(_skips_for_column, 0, lats, reversals, ascending)
    return flags.astype(bool)


def is_day(solar_zenith: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Return a per-pixel day mask (solar zenith angle <= 90°)."""
    return np.asarray(solar_zenith) <= NIGHT_SZA


def scene_is_day(solar_zenith: NDArray[np.floating]) -> bool:
    """Return the per-scene day flag used by the original algorithm.

    A scene counts as day only if every pixel has a solar zenith angle
    <= 90°; mixed scenes are treated as night.
    """
    return bool(np.all(is_day(solar_zenith)))


def nadir_daynight(
    solar_zenith: NDArray[np.floating], sensor_zenith: NDArray[np.floating]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Per-pixel day flag, and a mask of pixels matching their row's most-nadir pixel.

    Each row takes day/night from its pixel with the smallest sensor zenith
    angle; pixels of the other class (off-nadir, across the terminator) are
    masked out.

    Returns
    -------
    tuple of numpy.ndarray
        ``day`` and ``keep``, both boolean with the input shape.
    """
    day = is_day(solar_zenith)
    vza = np.where(np.isfinite(sensor_zenith), sensor_zenith, np.inf)
    ref = day[np.arange(day.shape[0]), np.argmin(vza, axis=1)]
    return day, day == ref[:, None]


def pixel_footprints(
    lons: NDArray[np.floating],
    lats: NDArray[np.floating],
    rows: NDArray[np.intp],
    cols: NDArray[np.intp],
) -> list[Polygon]:
    """Build approximate footprints for selected pixels.

    Each corner is the geodesic midpoint between the pixel center and its
    diagonal neighbor. At swath edges the missing neighbor is replaced by
    the pixel itself, which yields a degenerate corner (as in the original).

    Parameters
    ----------
    lons, lats : numpy.ndarray
        2-D geolocation arrays in degrees.
    rows, cols : numpy.ndarray
        Indices of the pixels to build footprints for.

    Returns
    -------
    list of shapely.geometry.Polygon
        One footprint per selected pixel, in EPSG:4326.
    """
    rows, cols = np.asarray(rows), np.asarray(cols)
    nrow, ncol = lons.shape
    corners = []
    for di, dj in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        ni = np.clip(rows + di, 0, nrow - 1)
        nj = np.clip(cols + dj, 0, ncol - 1)
        lon0, lat0 = lons[rows, cols], lats[rows, cols]
        az, _, dist = GEOD.inv(lon0, lat0, lons[ni, nj], lats[ni, nj])
        clon, clat, _ = GEOD.fwd(lon0, lat0, az, np.asarray(dist) / 2.0)
        corners.append(np.column_stack([clon, clat]))
    stacked = np.stack(corners, axis=1)
    return [Polygon(c) for c in stacked]
