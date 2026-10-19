"""Fire event registry (config/events.toml) and AOI geometry."""

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
import shapely
from shapely.geometry import Polygon, box

EVENTS_FILE = Path(__file__).parents[2] / "config" / "events.toml"
DENSIFY_M = 1000.0


@dataclass(frozen=True)
class FireEvent:
    """One fire event: AOI box around a center point, optional date range."""

    key: str
    name: str
    center_lon: float
    center_lat: float
    north_km: float
    south_km: float
    west_km: float
    east_km: float
    start: dt.date | None = None
    end: dt.date | None = None

    @property
    def crs(self) -> str:
        """Lambert azimuthal equal-area CRS centered on the event (north-up, meters)."""
        return f"+proj=laea +lat_0={self.center_lat} +lon_0={self.center_lon} +datum=WGS84"

    def aoi(self, crs: str | None = None) -> Polygon:
        """Return the AOI in EPSG:4326, built as a box in ``crs``.

        Parameters
        ----------
        crs : str, optional
            Projected CRS (meters) in which the box is rectangular. Default:
            ``self.crs``.

        Returns
        -------
        shapely.geometry.Polygon
        """
        crs = crs or self.crs
        to_proj = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        to_geo = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x, y = to_proj.transform(self.center_lon, self.center_lat)
        rect = box(x - self.west_km * 1e3, y - self.south_km * 1e3,
                   x + self.east_km * 1e3, y + self.north_km * 1e3)
        rect = shapely.segmentize(rect, DENSIFY_M)
        return shapely.transform(rect, lambda xy: np.column_stack(to_geo.transform(*xy.T)))

    def bbox(self) -> tuple[float, float, float, float]:
        """Return the AOI bounding box (west, south, east, north) in degrees."""
        return self.aoi().bounds


def load_events(path: str | Path = EVENTS_FILE) -> dict[str, FireEvent]:
    """Read the event registry.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        TOML file; defaults to ``config/events.toml`` in the repository.

    Returns
    -------
    dict of str to FireEvent
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)
    default_extent = config.get("defaults", {}).get("extent_km", {})
    events = {}
    for key, entry in config["events"].items():
        extent = {**default_extent, **entry.get("extent_km", {})}
        events[key] = FireEvent(
            key=key,
            name=entry["name"],
            center_lon=entry["center"][0],
            center_lat=entry["center"][1],
            north_km=extent["north"],
            south_km=extent["south"],
            west_km=extent["west"],
            east_km=extent["east"],
            start=entry.get("start"),
            end=entry.get("end"),
        )
    return events
