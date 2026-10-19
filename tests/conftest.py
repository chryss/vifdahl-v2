"""Shared fixtures: legacy-module loader and a synthetic swath."""

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

LEGACY_DIR = Path(__file__).parents[1] / "legacy" / "satfire"
EVENTS_TEMPLATE = Path(__file__).parents[1] / "config" / "events_template.toml"
SCAN_ROWS = 32
OVERLAP_ROWS = 2


def load_legacy(name: str) -> ModuleType:
    """Import a module from legacy/satfire with ``pygaarst`` stubbed out."""
    if not (LEGACY_DIR / f"{name}.py").exists():
        pytest.skip("legacy/ not present")
    if "pygaarst" not in sys.modules:
        stub = types.ModuleType("pygaarst")
        stub.raster = types.ModuleType("pygaarst.raster")
        sys.modules["pygaarst"] = stub
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", LEGACY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bowtie_lats(nscans: int = 4, ncols: int = 20, lat0: float = 64.0) -> np.ndarray:
    """Ascending-pass latitudes where each scan overlaps the previous one."""
    step = 0.0034
    rows = []
    for scan in range(nscans):
        start = lat0 + scan * (SCAN_ROWS - OVERLAP_ROWS) * step
        rows.append(start + step * np.arange(SCAN_ROWS))
    col = np.concatenate(rows)
    return np.repeat(col[:, None], ncols, axis=1) + 0.0001 * np.arange(ncols)


@pytest.fixture
def swath() -> xr.Dataset:
    """A small cool, nighttime swath with two separate hot clusters."""
    nrow, ncol = 40, 30
    lat = 64.0 + 0.0034 * np.arange(nrow)[:, None] + 0.0 * np.arange(ncol)
    lon = -147.0 + 0.0078 * np.arange(ncol)[None, :] + 0.0 * lat
    i4 = np.full((nrow, ncol), 270.0)
    i5 = np.full((nrow, ncol), 268.0)
    i4[10:12, 10:12] = 340.0           # 2x2 hot cluster
    i4[30, 25] = 330.0                  # isolated hot pixel
    data = {
        "i4_bt": i4, "i5_bt": i5, "anomalous": np.zeros_like(i4, dtype=bool),
        "lat": lat, "lon": lon,
        "sza": np.full_like(i4, 100.0), "vza": np.full_like(i4, 20.0),
    }
    return xr.Dataset(
        {k: (("y", "x"), v) for k, v in data.items()},
        attrs={"platform": "NPP", "start_time": "2015-06-15T11:35:00Z",
               "orbit": 18760, "ascending_node": True, "source": "synthetic"},
    )
