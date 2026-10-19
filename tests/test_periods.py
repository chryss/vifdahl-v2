"""Local solar date and day/night assignment."""

import pytest

from vifdahl.periods import assign_period


@pytest.mark.parametrize(
    ("utc", "lon", "date", "period"),
    [
        ("2019-07-08T22:00Z", -150.0, "2019-07-08", "day"),    # 12:00 LST
        ("2019-07-08T12:00Z", -150.0, "2019-07-07", "night"),  # 02:00 LST
        ("2019-07-09T05:00Z", -150.0, "2019-07-08", "night"),  # 19:00 LST
        ("2019-07-08T16:00Z", -150.0, "2019-07-08", "day"),    # 06:00 LST
        ("2019-07-08T01:00Z", 165.0, "2019-07-08", "day"),     # 12:00 LST
        ("2019-07-08T01:00Z", -165.0, "2019-07-07", "day"),    # 14:00 LST
    ],
)
def test_assign_period(utc: str, lon: float, date: str, period: str) -> None:
    result = assign_period([utc], [lon])
    assert result.loc[0, "LocalDate"] == date
    assert result.loc[0, "Period"] == period
