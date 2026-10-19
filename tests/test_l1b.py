"""L1B reader: unit tests, and Sockeye 2015 check if the granule is downloaded."""

from pathlib import Path

import numpy as np
import pytest

from vifdahl.detect import detect
from conftest import EVENTS_TEMPLATE
from vifdahl.events import load_events
from vifdahl.readers.l1b import anomaly_mask, read_l1b, to_bt

RAW = Path(__file__).parents[1] / "data" / "raw" / "l1b"
SOCKEYE = "A2015166.1130"


def test_to_bt() -> None:
    lut = np.linspace(200.0, 400.0, 65536)
    dn = np.array([0, 100, 65532, 65533, 65535])
    bt = to_bt(dn, lut)
    assert bt[0] == 200.0 and bt[1] == pytest.approx(lut[100])
    assert np.isnan(bt[2:]).all()


def test_anomaly_mask() -> None:
    qf4 = np.array([0, 4, 2, 2, 6])
    i5 = np.array([290.0, 280.0, 220.0, 343.0, np.nan])
    np.testing.assert_array_equal(anomaly_mask(qf4, i5), [False, True, False, True, True])


def test_sockeye_2015() -> None:
    obs = sorted(RAW.glob(f"VNP02IMG.{SOCKEYE}.*.nc"))
    geo = sorted(RAW.glob(f"VNP03IMG.{SOCKEYE}.*.nc"))
    if not (obs and geo):
        pytest.skip("Sockeye 2015 granule not downloaded")
    swath = read_l1b(obs[0], geo[0], aoi=load_events(EVENTS_TEMPLATE)["sockeye"].aoi())
    assert swath.attrs["platform"] == "NPP"
    assert swath.attrs["orbit"] == 18814
    assert not swath.attrs["ascending_node"]
    masks = detect(swath)
    assert (int(masks.hot.sum()), int(masks.warm.sum())) == (4, 49)
