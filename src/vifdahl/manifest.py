"""SQLite record of processed granules, for resuming runs and cleaning up files."""

import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd

COLUMNS = {
    "granule_id": "TEXT NOT NULL",
    "aoi_id": "TEXT NOT NULL",
    "platform": "TEXT",
    "time_key": "TEXT",
    "orbit": "INTEGER",        # NULL when unknown (failed before reading)
    "status": "TEXT",          # done | no_coverage | failed
    "n_detections": "INTEGER",
    "crop_path": "TEXT",
    "detections_path": "TEXT",
    "obs_file": "TEXT",
    "geo_file": "TEXT",
    "raw_on_disk": "INTEGER",
    "error": "TEXT",
    "updated": "TEXT",
}
FINISHED = ("done", "no_coverage")


class Manifest:
    """One row per granule pair and AOI, keyed by ``(granule_id, aoi_id)``.

    Parameters
    ----------
    path : str or pathlib.Path
        SQLite file; created if missing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cols = ", ".join(f"{k} {v}" for k, v in COLUMNS.items())
        with self._connect() as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS granules "
                        f"({cols}, PRIMARY KEY (granule_id, aoi_id))")
            existing = {r[1] for r in con.execute("PRAGMA table_info(granules)")}
        if missing := set(COLUMNS) - existing:
            raise RuntimeError(f"{self.path}: old manifest schema, missing {sorted(missing)}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, row: dict) -> None:
        """Insert or replace one granule-AOI record."""
        row = {k: row.get(k) for k in COLUMNS} | {"updated": dt.datetime.now(dt.UTC).isoformat()}
        keys = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with self._connect() as con:
            con.execute(f"INSERT OR REPLACE INTO granules ({keys}) VALUES ({marks})",
                        tuple(row.values()))

    def finished(self) -> set[tuple[str, str]]:
        """``(granule_id, aoi_id)`` pairs that need no further processing."""
        marks = ", ".join("?" * len(FINISHED))
        with self._connect() as con:
            rows = con.execute(
                f"SELECT granule_id, aoi_id FROM granules WHERE status IN ({marks})", FINISHED)
            return {(r[0], r[1]) for r in rows}

    def to_frame(self) -> pd.DataFrame:
        """Return the whole manifest as a DataFrame."""
        with self._connect() as con:
            return pd.read_sql("SELECT * FROM granules ORDER BY aoi_id, time_key, platform", con)
