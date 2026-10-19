"""Check swath helpers against the legacy implementation."""

from types import SimpleNamespace

import numpy as np

from conftest import OVERLAP_ROWS, SCAN_ROWS, bowtie_lats, load_legacy
from vifdahl.swath import duplicate_pixels, pixel_footprints, scene_is_day


def test_duplicates_match_legacy() -> None:
    legacy = load_legacy("viirsswathtools")
    lats = bowtie_lats()
    scene = SimpleNamespace(lats=lats, ascending_node=True)
    expected = legacy.get_skips(scene).astype(bool)
    result = duplicate_pixels(lats, ascending=True)
    np.testing.assert_array_equal(result, expected)
    assert result.any()


def test_duplicates_descending_match_legacy() -> None:
    legacy = load_legacy("viirsswathtools")
    lats = bowtie_lats(lat0=66.0)[::-1].copy()
    scene = SimpleNamespace(lats=lats, ascending_node=False)
    expected = legacy.get_skips(scene).astype(bool)
    result = duplicate_pixels(lats, ascending=False)
    np.testing.assert_array_equal(result, expected)
    assert result.sum() >= OVERLAP_ROWS * (lats.shape[0] // SCAN_ROWS - 1)


def test_no_overlap_no_duplicates() -> None:
    lats = 64.0 + 0.0034 * np.arange(64)[:, None] + np.zeros((1, 10))
    assert not duplicate_pixels(lats, ascending=True).any()


def test_early_reversal_stays_in_bounds() -> None:
    """Legacy code wrapped to the end of the array here (``==`` typo)."""
    col = np.array([0.0, 1.0, 0.5, 0.6, 0.7, 1.5, 2.0, 3.0, 4.0, 5.0])
    lats = np.repeat(col[:, None], 3, axis=1)
    dup = duplicate_pixels(lats, ascending=True)
    assert not dup[-3:].any()


def test_footprints_match_legacy() -> None:
    legacy = load_legacy("viirsswathtools")
    lats = bowtie_lats(nscans=1, ncols=8)
    lons = -147.0 + 0.0078 * np.arange(8)[None, :] + 0.0 * lats
    rows, cols = np.array([0, 5, 31]), np.array([0, 3, 7])
    result = pixel_footprints(lons, lats, rows, cols)
    for poly, i, j in zip(result, rows, cols):
        expected = legacy.get_corners_wgs_from_latlon(i, j, lons, lats)
        np.testing.assert_allclose(np.asarray(poly.exterior.coords)[:4], expected)


def test_scene_is_day() -> None:
    assert scene_is_day(np.array([[30.0, 89.0]]))
    assert not scene_is_day(np.array([[30.0, 95.0]]))
