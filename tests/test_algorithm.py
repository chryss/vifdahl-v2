"""Equivalence tests: ported classification vs. legacy implementation."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import load_legacy
from vifdahl.algorithm import PUBLISHED_THRESHOLDS, classify
from vifdahl.readers.sdr import anomaly_mask

RNG = np.random.default_rng(42)
SHAPE = (200, 300)


def random_scene() -> tuple[np.ndarray, ...]:
    """Random I4/I5 BTs and QF1 values spanning all threshold regimes."""
    i5 = RNG.uniform(250.0, 320.0, SHAPE)
    i4 = i5 * RNG.uniform(0.98, 1.2, SHAPE)
    i4[RNG.random(SHAPE) < 0.02] = 367.0
    pixq4 = np.where(RNG.random(SHAPE) < 0.05, 193, 0)
    pixq5 = np.where(RNG.random(SHAPE) < 0.1, 1, 0)
    return i4, i5, pixq4, pixq5


@pytest.mark.parametrize("daytime", [True, False])
@pytest.mark.parametrize("min_i5_special", [312.0, 305.0])
def test_matches_legacy(daytime: bool, min_i5_special: float) -> None:
    legacy = load_legacy("viirsifire")
    i4, i5, pixq4, pixq5 = random_scene()
    expected = legacy.getfireconditions_fromrasters(
        i4, i5, pixq4, pixq5, daytime=daytime, mint5_special=min_i5_special)
    thresholds = replace(PUBLISHED_THRESHOLDS, min_i5_special=min_i5_special)
    # removing the legacy_anom from legacy_warm as we changed the algorithm
    # to make these two conditions mutually exclusive
    legacy_anom, legacy_hot, legacy_warm = expected
    expected = (legacy_anom, legacy_hot, legacy_warm & ~legacy_anom)
    result = classify(i4, i5, anomaly_mask(i4, pixq4, pixq5), daytime, thresholds)
    for got, want in zip(result, expected):
        np.testing.assert_array_equal(got, want)
    assert any(m.any() for m in result)


def test_per_pixel_daynight() -> None:
    i4, i5, pixq4, pixq5 = random_scene()
    anom = anomaly_mask(i4, pixq4, pixq5)
    day = np.zeros(SHAPE, dtype=bool)
    day[:, :150] = True
    mixed = classify(i4, i5, anom, day)
    all_day = classify(i4, i5, anom, True)
    all_night = classify(i4, i5, anom, False)
    np.testing.assert_array_equal(mixed.hot[:, :150], all_day.hot[:, :150])
    np.testing.assert_array_equal(mixed.warm[:, 150:], all_night.warm[:, 150:])


def test_hot_and_warm_exclusive() -> None:
    i4, i5, pixq4, pixq5 = random_scene()
    for daytime in (True, False):
        masks = classify(i4, i5, anomaly_mask(i4, pixq4, pixq5), daytime)
        assert not (masks.warm & masks.anomalous).any()
        assert not (masks.hot & masks.warm).any()
        assert not (masks.hot & masks.anomalous).any()


def test_nan_is_never_fire() -> None:
    i4 = np.array([np.nan, 340.0])
    i5 = np.array([280.0, np.nan])
    masks = classify(i4, i5, np.zeros(2, dtype=bool), True)
    assert not masks.hot.any() and not masks.warm.any()
