"""Stage 1: search window, manifest, per-granule processing, resume."""

import datetime as dt
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest

from vifdahl import stage1
from vifdahl.algorithm import DEFAULT_THRESHOLDS
from conftest import EVENTS_TEMPLATE
from vifdahl.events import load_events
from vifdahl.manifest import Manifest
from vifdahl.periods import utc_window
from vifdahl.readers import load_swath
from vifdahl.stage1 import Stage1Config, process_pair, run_stage1

RAW = Path(__file__).parents[1] / "data" / "raw" / "l1b"


def test_utc_window() -> None:
    start, end = utc_window(dt.date(2026, 8, 17), dt.date(2026, 8, 19), -150.0, -135.0)
    assert start == dt.datetime(2026, 8, 17, 15, 0)   # 06:00 LST at 135°W
    assert end == dt.datetime(2026, 8, 20, 16, 0)     # 06:00 LST at 150°W


def test_manifest(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "m.sqlite")
    m.record({"granule_id": "NPP.A1", "status": "done"})
    m.record({"granule_id": "NPP.A2", "status": "failed", "error": "x"})
    m.record({"granule_id": "NPP.A2", "status": "no_coverage"})
    assert m.finished() == {"NPP.A1", "NPP.A2"}
    assert len(m.to_frame()) == 2


def test_process_pair_sockeye(tmp_path: Path) -> None:
    obs = sorted(RAW.glob("VNP02IMG.A2015166.1130.*.nc"))
    geo = sorted(RAW.glob("VNP03IMG.A2015166.1130.*.nc"))
    if not (obs and geo):
        pytest.skip("Sockeye 2015 granule not downloaded")
    cfg = Stage1Config(aoi=load_events(EVENTS_TEMPLATE)["sockeye"].aoi(), workdir=tmp_path)
    record = process_pair(obs[0], geo[0], "NPP.A2015166.1130", cfg)
    assert record["status"] == "done" and record["n_detections"] == 53
    crop = load_swath(record["crop_path"])
    assert crop.attrs["ascending_node"] is False and crop["anomalous"].dtype == bool
    assert len(gpd.read_parquet(record["detections_path"])) == 53

    tiny = Stage1Config(aoi=load_events(EVENTS_TEMPLATE)["sockeye"].aoi().centroid.buffer(1e-4),
                        workdir=tmp_path / "tiny")
    assert process_pair(obs[0], geo[0], "NPP.A2015166.1130", tiny)["n_detections"] == 0

    cold = Stage1Config(aoi=load_events(EVENTS_TEMPLATE)["sockeye"].aoi(), workdir=tmp_path / "cold",
                        thresholds=replace(DEFAULT_THRESHOLDS, ndiff_hot=9.0,
                                           ndiff_warm=9.0, ndiff_warm_special=9.0))
    record = process_pair(obs[0], geo[0], "NPP.A2015166.1130", cold)
    assert record["n_detections"] == 0
    assert len(gpd.read_parquet(record["detections_path"])) == 0


def test_failure_recorded_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_download(pair, raw_dir):
        raise OSError("network down")

    monkeypatch.setattr(stage1, "_LOGGED_IN", True)
    monkeypatch.setattr(stage1, "download_pair", broken_download)
    pairs = [SimpleNamespace(granule_id=f"NPP.A{i}", platform="NPP", time_key=f"A{i}")
             for i in range(3)]
    cfg = Stage1Config(aoi=load_events(EVENTS_TEMPLATE)["sockeye"].aoi(), workdir=tmp_path)
    result = run_stage1(pairs, cfg, n_jobs=1)
    assert set(result["status"]) == {"failed"}
    assert "network down" in result["error"].iloc[0]

    monkeypatch.setattr(stage1, "download_pair", lambda pair, raw_dir: (_ for _ in ()).throw(
        AssertionError("finished granules must be skipped")))
    Manifest(tmp_path / "manifest.sqlite").record({"granule_id": "NPP.A0", "status": "done"})
    result = run_stage1(pairs[:1], cfg, n_jobs=1)
    assert result.set_index("granule_id").loc["NPP.A0", "status"] == "done"
