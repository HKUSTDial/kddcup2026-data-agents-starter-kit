from datetime import datetime, timezone
from decimal import Decimal

import pytest

from data_agent_baseline.evaluation.normalization import (
    column_signature,
    normalize_cell,
)


@pytest.mark.parametrize("value", [None, "", " null ", "NONE", "NaN", "NaT", "<NA>"])
def test_null_values_normalize_to_empty_string(value) -> None:
    assert normalize_cell(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("18", "18.00"),
        ("0.005", "0.01"),
        ("-0.001", "0.00"),
        ("1,234.5", "1234.50"),
        (Decimal("2.345"), "2.35"),
    ],
)
def test_numbers_use_two_decimal_half_up_normalization(value, expected: str) -> None:
    assert normalize_cell(value) == expected


def test_strict_commas_preserves_grouped_numeric_text() -> None:
    assert normalize_cell("1,234.5", strict_commas=True) == "1,234.5"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026/07/23", "2026-07-23"),
        ("23/07/2026", "2026-07-23"),
        ("2026-07-23 09:10:11", "2026-07-23T09:10:11"),
        ("2026-07-23T09:10:11+0800", "2026-07-23T01:10:11Z"),
        (
            datetime(2026, 7, 23, 9, 10, 11, tzinfo=timezone.utc),
            "2026-07-23T09:10:11Z",
        ),
    ],
)
def test_dates_and_timestamps_are_canonicalized(value, expected: str) -> None:
    assert normalize_cell(value) == expected


def test_column_signature_ignores_value_order_but_preserves_multiplicity() -> None:
    assert column_signature(["2", "1", "1"]) == column_signature(["1", "2", "1"])
    assert column_signature(["2", "1", "1"]) != column_signature(["2", "1"])
