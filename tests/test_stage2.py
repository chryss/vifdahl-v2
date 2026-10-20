"""Stage 2: per-AOI daily GeoPackages from a stage 1 run."""

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from conftest import EVENTS_TEMPLATE
from vifdahl import stage1
from vifdahl.events import load_events
from vifdahl.stage1 import Stage1Config, run_stage1
from vifdahl.stage2 import run_stage2

RAW = Path(__file__).parents[1] / "data" / "raw" / "l1b"


def test_run_stage2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    obs = sorted(RAW.glob("VNP02IMG.A2015166.1130.*.nc"))
    geo = sorted(RAW.glob("VNP03IMG.A2015166.1130.*.nc"))
    if not (obs and geo):
        pytest.skip("Sockeye 2015 granule not downloaded")
    monkeypatch.setattr(stage1, "_LOGGED_IN", True)
    monkeypatch.setattr(stage1, "download_pair", lambda pair, raw_dir: (obs[0], geo[0]))
    aois = {"sockeye": load_events(EVENTS_TEMPLATE)["sockeye"].aoi(), "nowhere": box(0, 0, 1, 1)}
    cfg = Stage1Config(aois=aois, workdir=tmp_path, keep_raw=True)  # keep reference granule
    pair = SimpleNamespace(granule_id="NPP.A2015166.1130", platform="NPP",
                           time_key="A2015166.1130")
    run_stage1([pair], cfg, n_jobs=1)

    written = run_stage2(tmp_path)
    assert written["nowhere"] == []
    assert [p.relative_to(tmp_path).as_posix() for p in written["sockeye"]] == [
        "gpkg/sockeye/VIFDAHL_sockeye_20150614.gpkg"]
    assert set(gpd.read_file(written["sockeye"][0], layer="points")["Platform"]) == {"NPP"}
    assert set(gpd.read_file(written["sockeye"][0], layer="fused")["Platform"]) == {"ALL"}
    summary = pd.read_csv(tmp_path / "gpkg" / "summary.csv")
    seen = summary.loc[(summary["aoi_id"] == "sockeye") & (summary["n_overpasses"] > 0)]
    assert list(seen["Platform"]) == ["ALL", "NPP"]
    assert set(seen["LocalDate"] + " " + seen["Period"]) == {"2015-06-14 night"}
    assert list(seen["n_detections"]) == [53, 53]
    assert seen.set_index("Platform").loc["ALL", "area_H_km2"] > 0
    assert (summary.loc[summary["aoi_id"] == "nowhere", "n_overpasses"] == 0).all()

    written = run_stage2(tmp_path, aoi_ids=["sockeye"], fuse_by="platform")
    assert set(gpd.read_file(written["sockeye"][0], layer="fused")["Platform"]) == {"NPP"}
