"""Search and download L1B granule pairs (observations + geolocation) via earthaccess."""

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import earthaccess

PRODUCTS = {
    "NPP": ("VNP02IMG", "VNP03IMG"),
    "N20": ("VJ102IMG", "VJ103IMG"),
    "N21": ("VJ202IMG", "VJ203IMG"),
}
TIME_KEY = re.compile(r"\.(A\d{7}\.\d{4})\.")


@dataclass
class GranulePair:
    """Matching observation and geolocation granules (earthaccess results)."""

    platform: str
    time_key: str
    obs: earthaccess.DataGranule
    geo: earthaccess.DataGranule

    @property
    def granule_id(self) -> str:
        """Unique ID, e.g. ``NPP.A2015166.1130``."""
        return f"{self.platform}.{self.time_key}"

    @property
    def size_mb(self) -> float:
        """Download size of both files (MB)."""
        return self.obs.size() + self.geo.size()


def _time_key(granule: earthaccess.DataGranule) -> str:
    """Acquisition key ``AYYYYDDD.HHMM`` from the granule's file name."""
    return TIME_KEY.search(Path(granule.data_links()[0]).name).group(1)


def search_pairs(
    bbox: tuple[float, float, float, float],
    start: dt.datetime,
    end: dt.datetime,
    platforms: tuple[str, ...] = ("NPP", "N20", "N21"),
) -> list[GranulePair]:
    """Find observation/geolocation granule pairs intersecting ``bbox``.

    Parameters
    ----------
    bbox : tuple of float
        West, south, east, north (degrees).
    start, end : datetime.datetime
        UTC search window.
    platforms : tuple of str, optional
        Keys of ``PRODUCTS``.

    Returns
    -------
    list of GranulePair
        Sorted by time; unmatched granules are dropped.
    """
    temporal = (start.isoformat(), end.isoformat())
    pairs = []
    for platform in platforms:
        obs_name, geo_name = PRODUCTS[platform]
        found = {
            name: {_time_key(g): g for g in earthaccess.search_data(
                short_name=name, bounding_box=bbox, temporal=temporal)}
            for name in (obs_name, geo_name)
        }
        for key in sorted(found[obs_name].keys() & found[geo_name].keys()):
            pairs.append(GranulePair(platform, key, found[obs_name][key], found[geo_name][key]))
    return sorted(pairs, key=lambda p: (p.time_key, p.platform))


def download_pair(pair: GranulePair, raw_dir: Path) -> tuple[Path, Path]:
    """Download both files of a pair; existing files are not fetched again.

    Returns
    -------
    tuple of pathlib.Path
        Observation and geolocation file paths.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = earthaccess.download([pair.obs, pair.geo], local_path=raw_dir,
                                 show_progress=False)
    obs = next(Path(p) for p in paths if PRODUCTS[pair.platform][0] in Path(p).name)
    geo = next(Path(p) for p in paths if PRODUCTS[pair.platform][1] in Path(p).name)
    return obs, geo
