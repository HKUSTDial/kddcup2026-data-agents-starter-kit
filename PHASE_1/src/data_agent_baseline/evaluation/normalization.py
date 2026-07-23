from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

NULL_TOKENS = frozenset({"", "null", "none", "nan", "nat", "<na>"})
TWO_DECIMAL_PLACES = Decimal("0.01")

NUMBER_WITH_GROUPING = re.compile(r"-?[\d,]+(?:\.\d+)?")

DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
)


def _normalize_decimal(text: str) -> str | None:
    try:
        value = Decimal(text)
        if not value.is_finite():
            return None
        normalized = value.quantize(TWO_DECIMAL_PLACES, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None

    if normalized == 0:
        return "0.00"
    return format(normalized, "f")


def _normalize_datetime(text: str) -> str | None:
    for value_format in DATETIME_FORMATS:
        try:
            value = datetime.strptime(text, value_format)
        except ValueError:
            continue
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return value.isoformat()

    for value_format in DATE_FORMATS:
        try:
            value = datetime.strptime(text, value_format)
        except ValueError:
            continue
        return value.strftime("%Y-%m-%d")
    return None


def normalize_cell(value: Any, *, strict_commas: bool = False) -> str:
    """Normalize one answer cell before column-level comparison."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float, Decimal)):
        normalized_number = _normalize_decimal(str(value))
        if normalized_number is not None:
            return normalized_number

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip().replace("\r\n", "\n")
    numeric_candidate = text
    if not strict_commas and NUMBER_WITH_GROUPING.fullmatch(text):
        numeric_candidate = text.replace(",", "")

    normalized_number = _normalize_decimal(numeric_candidate)
    if normalized_number is not None:
        return normalized_number

    normalized_datetime = _normalize_datetime(text)
    if normalized_datetime is not None:
        return normalized_datetime
    return text


def column_signature(values: list[Any], *, strict_commas: bool = False) -> str:
    """Create an order-independent fingerprint for a single answer column."""

    normalized_values = sorted(
        normalize_cell(value, strict_commas=strict_commas) for value in values
    )
    payload = "\x1f".join(normalized_values).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
