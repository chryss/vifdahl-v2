"""Run summary: overpass counting, observed-vs-unobserved, areas per fusing mode."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from vifdahl.summary import summarize

AOI = {"fire": box(-147.0, 64.0, -146.0, 65.0)}  # centroid 146.5°W: LST = UTC − 9 h 46 min
DATES = pd.date_range("2026-08-17", "2026-08-19")


def manifest() -> pd.DataFrame:
    rows = [  # day 230 = 2026-08-18; 21:00 UTC ≈ 11:14 LST
        ("N20", "A2026230.2100", 100, "done"),
        ("N20", "A2026230.2106", 100, "done"),     # same orbit: one overpass
        ("NPP", "A2026230.2130", None, "done"),    # no orbit: time-gap fallback
        ("N20", "A2026231.1200", None, "failed"),  # 02:14 LST: night of 08-18
    ]
    return pd.DataFrame(
        [{"granule_id": f"{p}.{t}", "aoi_id": "fire", "platform": p, "time_key": t,
          "orbit": o, "status": s} for p, t, o, s in rows])


def detections() -> pd.DataFrame:
    return pd.DataFrame({"aoi_id": "fire", "LocalDate": "2026-08-18", "Period": "day",
                         "Platform": ["N20", "N20", "NPP"], "Type": ["H", "H", "L"],
                         "TB4": [340.0, 350.0, 310.0]})


def fused(platforms: list[str]) -> gpd.GeoDataFrame:
    geoms = [box(-146.6, 64.5, -146.5, 64.55), box(-146.55, 64.5, -146.45, 64.55)]
    return gpd.GeoDataFrame({"aoi_id": "fire", "LocalDate": "2026-08-18", "Period": "day",
                             "Platform": platforms, "Class": "H", "area_km2": [20.0, 20.0]},
                            geometry=geoms, crs="EPSG:4326")


def row(table: pd.DataFrame, date: str, period: str, platform: str) -> pd.Series:
    sel = ((table["LocalDate"] == date) & (table["Period"] == period)
           & (table["Platform"] == platform))
    assert sel.sum() == 1
    return table.loc[sel].iloc[0]


def test_summarize_fused_all() -> None:
    table = summarize(manifest(), detections(), fused(["ALL", "ALL"]), AOI,
                      dates=DATES, platforms=["N20", "NPP"])
    assert len(table) == 3 * 2 * 3
    n20 = row(table, "2026-08-18", "day", "N20")
    assert (n20["n_overpasses"], n20["overpass_source"], n20["n_H"]) == (1, "orbit", 2)
    assert pd.isna(n20["area_H_km2"])
    npp = row(table, "2026-08-18", "day", "NPP")
    assert (npp["n_overpasses"], npp["overpass_source"], npp["n_L"]) == (1, "time_gap", 1)
    combined = row(table, "2026-08-18", "day", "ALL")
    assert (combined["n_overpasses"], combined["n_detections"]) == (2, 3)
    assert combined["overpass_source"] == "time_gap" and combined["TB4_max"] == 350.0
    assert combined["area_H_km2"] == 40.0 and combined["area_L_km2"] == 0.0
    night = row(table, "2026-08-18", "night", "N20")
    assert night["n_overpasses"] == 0 and night["n_failed"] == 1 and pd.isna(night["n_H"])
    unseen = row(table, "2026-08-17", "day", "ALL")
    assert unseen["n_overpasses"] == 0 and pd.isna(unseen["n_detections"])
    assert pd.isna(unseen["area_H_km2"])


def test_summarize_fused_by_platform() -> None:
    table = summarize(manifest(), detections(), fused(["N20", "NPP"]), AOI,
                      fuse_by="platform", dates=DATES, platforms=["N20", "NPP"])
    assert row(table, "2026-08-18", "day", "N20")["area_H_km2"] == 20.0
    assert row(table, "2026-08-18", "day", "NPP")["area_H_km2"] == 20.0
    union = row(table, "2026-08-18", "day", "ALL")["area_H_km2"]
    single = fused(["N20", "NPP"]).to_crs("EPSG:6931").area.iloc[0] / 1e6
    assert single < union < 2 * single  # overlap counted once
    with pytest.raises(ValueError):
        summarize(manifest(), detections(), fused(["N20", "NPP"]), AOI, fuse_by="orbit")
