"""Core value types and missing value semantics for SASLite."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any


class DType(str, Enum):
    """Variable data types."""
    NUMERIC = "numeric"
    CHARACTER = "character"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"


# SAS missing value sentinel — maps to float('nan') / pd.NA
MISSING_NUMERIC = float("nan")
MISSING_CHARACTER = ""

# Special missing values .A-.Z (Phase 4)
SPECIAL_MISSING: dict[str, float] = {}


def is_missing(value: Any) -> bool:
    """Check if a value is SAS-missing.

    Note: Empty string "" is NOT considered missing in SAS.
    Only None, NaN, and pd.NA are missing values.
    """
    if value is None:
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def coerce_to_numeric(value: Any) -> float:
    """Coerce a value to numeric, returning NaN on failure."""
    if value is None:
        return MISSING_NUMERIC
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value == ".":
            return MISSING_NUMERIC
        try:
            return float(value)
        except ValueError:
            return MISSING_NUMERIC
    return MISSING_NUMERIC


def coerce_to_str(value: Any) -> str:
    """Coerce a value to string."""
    if value is None:
        return MISSING_CHARACTER
    if isinstance(value, float) and math.isnan(value):
        return "."
    return str(value)


def sas_bool(value: Any) -> bool:
    """Evaluate a value as SAS boolean (0/missing = False, nonzero = True)."""
    if is_missing(value):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return len(value) > 0
    return bool(value)
