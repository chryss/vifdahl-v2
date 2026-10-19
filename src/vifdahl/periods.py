"""Local solar date and day/night period for detections.

TODO: date jumps at 180°; fires straddling it (Chukotka) split across days.
"""

import datetime as dt

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

DAY_START_H = 6.0
NIGHT_START_H = 18.0


def local_solar_time(utc: ArrayLike, lon: ArrayLike) -> pd.DatetimeIndex:
    """Return local mean solar time (UTC + lon/15 h), tz-naive.

    Parameters
    ----------
    utc : array_like of datetime-like
        Observation times in UTC.
    lon : array_like
        Longitudes in degrees east.

    Returns
    -------
    pandas.DatetimeIndex
    """
    utc = pd.DatetimeIndex(pd.to_datetime(utc, utc=True)).tz_localize(None)
    offset = pd.to_timedelta(np.asarray(lon, dtype=float) / 15.0, unit="h")
    return utc + offset


def utc_window(
    start: dt.date, end: dt.date, lon_min: float, lon_max: float
) -> tuple[dt.datetime, dt.datetime]:
    """UTC search window covering local solar dates ``start``–``end`` (inclusive).

    A local date runs from 06:00 to 06:00 the next day (local solar time).

    Parameters
    ----------
    start, end : datetime.date
        First and last local date.
    lon_min, lon_max : float
        Longitude range of the AOI (degrees east).

    Returns
    -------
    tuple of datetime.datetime
        UTC start and end, tz-naive.
    """
    first = dt.datetime.combine(start, dt.time()) + dt.timedelta(hours=DAY_START_H)
    last = dt.datetime.combine(end + dt.timedelta(days=1), dt.time()) + dt.timedelta(hours=DAY_START_H)
    return (first - dt.timedelta(hours=lon_max / 15.0),
            last - dt.timedelta(hours=lon_min / 15.0))


def assign_period(utc: ArrayLike, lon: ArrayLike) -> pd.DataFrame:
    """Assign local solar date and day/night period.

    Day: 06:00–18:00 local solar time. Night: 18:00–06:00, labeled with
    the date on which it starts.

    Parameters
    ----------
    utc : array_like of datetime-like
        Observation times in UTC.
    lon : array_like
        Longitudes in degrees east.

    Returns
    -------
    pandas.DataFrame
        Columns ``LocalDate`` (YYYY-MM-DD) and ``Period`` ('day'/'night').
    """
    lst = local_solar_time(utc, lon)
    hour = lst.hour + lst.minute / 60.0 + lst.second / 3600.0
    is_day = (hour >= DAY_START_H) & (hour < NIGHT_START_H)
    date = lst.normalize() - pd.to_timedelta(np.where(hour < DAY_START_H, 1, 0), unit="D")
    return pd.DataFrame({
        "LocalDate": date.strftime("%Y-%m-%d"),
        "Period": np.where(is_day, "day", "night"),
    })
