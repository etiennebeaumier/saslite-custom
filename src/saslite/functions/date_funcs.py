"""SAS date/time functions."""

from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Any

from saslite.runtime.types import is_missing


# SAS date value = days since 1960-01-01
_SAS_EPOCH = date(1960, 1, 1)


def today() -> float:
    """TODAY() / DATE() — current SAS date value."""
    delta = date.today() - _SAS_EPOCH
    return float(delta.days)


def date_val() -> float:
    """DATE() — alias for TODAY()."""
    return today()


_cached_datetime: float | None = None

def datetime_val() -> float:
    """DATETIME() — current SAS datetime value (seconds since epoch)."""
    global _cached_datetime
    if _cached_datetime is not None:
        return _cached_datetime
    fixed = os.environ.get("SASLITE_FIXED_DATETIME")
    if fixed:
        _cached_datetime = _parse_fixed_datetime(fixed)
        return _cached_datetime
    delta = datetime.now() - datetime(1960, 1, 1)
    _cached_datetime = delta.total_seconds()
    return _cached_datetime


def _reset_datetime_cache() -> None:
    """Reset datetime cache (called at start of each query execution)."""
    global _cached_datetime
    _cached_datetime = None


def _parse_fixed_datetime(value: str) -> float:
    """Parse a deterministic datetime override for tests/reproducible runs."""
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return (dt - datetime(1960, 1, 1)).total_seconds()
        except ValueError:
            continue

    raise ValueError(
        "SASLITE_FIXED_DATETIME must be a SAS datetime value or "
        "YYYY-MM-DDTHH:MM:SS"
    )


def mdy(month: Any, day: Any, year: Any) -> float:
    """MDY(month, day, year) — create date from components."""
    if is_missing(month) or is_missing(day) or is_missing(year):
        return float("nan")
    try:
        d = date(int(year), int(month), int(day))
        return float((d - _SAS_EPOCH).days)
    except (ValueError, TypeError):
        return float("nan")


def year(sas_date: Any) -> float:
    """YEAR(date) — extract year."""
    if is_missing(sas_date):
        return float("nan")
    d = _sas_to_date(float(sas_date))
    return float(d.year)


def month(sas_date: Any) -> float:
    """MONTH(date) — extract month (1-12)."""
    if is_missing(sas_date):
        return float("nan")
    d = _sas_to_date(float(sas_date))
    return float(d.month)


def day(sas_date: Any) -> float:
    """DAY(date) — extract day (1-31)."""
    if is_missing(sas_date):
        return float("nan")
    d = _sas_to_date(float(sas_date))
    return float(d.day)


def weekday(sas_date: Any) -> float:
    """WEEKDAY(date) — day of week (1=Sunday)."""
    if is_missing(sas_date):
        return float("nan")
    d = _sas_to_date(float(sas_date))
    # Python weekday: 0=Monday, SAS: 1=Sunday
    return float((d.weekday() + 1) % 7 + 1)


def qtr(sas_date: Any) -> float:
    """QTR(date) — quarter (1-4)."""
    if is_missing(sas_date):
        return float("nan")
    d = _sas_to_date(float(sas_date))
    return float((d.month - 1) // 3 + 1)


def intnx(interval: str, start_date: Any, increment: Any, alignment: str = "S") -> float:
    """INTNX(interval, date, increment [, 'alignment']) — date increment."""
    if is_missing(start_date):
        return float("nan")
    d = _sas_to_date(float(start_date))
    inc = int(increment)
    interval = interval.upper()
    alignment = alignment.upper() if alignment else "S"

    if interval in ("DAY", "DAYS"):
        result = d + timedelta(days=inc)
    elif interval in ("WEEK", "WEEKS"):
        result = d + timedelta(weeks=inc)
    elif interval in ("MONTH", "MONTHS"):
        result = _add_months(d, inc)
    elif interval in ("QTR", "QUARTER", "QTRS"):
        result = _add_months(d, inc * 3)
    elif interval in ("YEAR", "YEARS"):
        result = _add_months(d, inc * 12)
    elif interval in ("HOUR", "HOURS"):
        result = d + timedelta(hours=inc)
    elif interval in ("MINUTE", "MINUTES"):
        result = d + timedelta(minutes=inc)
    elif interval in ("SECOND", "SECONDS"):
        result = d + timedelta(seconds=inc)
    else:
        result = d + timedelta(days=inc)

    # Apply alignment
    if alignment == "B":
        result = _beginning_of_interval(result, interval)
    elif alignment == "E":
        result = _end_of_interval(result, interval)
    elif alignment == "M":
        result = _middle_of_interval(result, interval)

    return float((result - _SAS_EPOCH).days) if isinstance(result, date) else float(result)


def intck(interval: str, start_date: Any, end_date: Any) -> float:
    """INTCK(interval, start, end) — count intervals between dates."""
    if is_missing(start_date) or is_missing(end_date):
        return float("nan")
    d1 = _sas_to_date(float(start_date))
    d2 = _sas_to_date(float(end_date))
    interval = interval.upper()

    if interval in ("DAY", "DAYS"):
        return float((d2 - d1).days)
    if interval in ("WEEK", "WEEKS"):
        return float((d2 - d1).days // 7)
    if interval in ("MONTH", "MONTHS"):
        return float((d2.year - d1.year) * 12 + (d2.month - d1.month))
    if interval in ("QTR", "QUARTER"):
        return float((d2.year - d1.year) * 4 + (d2.month - d1.month) // 3)
    if interval in ("YEAR", "YEARS"):
        return float(d2.year - d1.year)

    return float((d2 - d1).days)


def datepart(dt: Any) -> float:
    """DATEPART(datetime) — extract date from datetime."""
    import pandas as pd

    if is_missing(dt):
        return float("nan")

    # Handle pandas Timestamp (from datetime64 columns)
    if isinstance(dt, pd.Timestamp):
        if pd.isna(dt):
            return float("nan")
        # Convert Timestamp to SAS date (days since 1960-01-01)
        delta = dt.date() - _SAS_EPOCH
        return float(delta.days)

    # Handle SAS datetime value (seconds since 1960-01-01 00:00:00)
    try:
        return float(int(float(dt) / 86400))
    except (ValueError, TypeError):
        return float("nan")


def timepart(dt: Any) -> float:
    """TIMEPART(datetime) — extract time from datetime."""
    if is_missing(dt):
        return float("nan")
    return float(float(dt) % 86400)


def datedif(start_date: Any, end_date: Any, unit: str) -> float:
    """DATEDIF(start, end, unit) — difference between two dates in the given unit.

    Units: DAY/DAYS, WEEK/WEEKS, MONTH/MONTHS, YEAR/YEARS, QTR/QUARTER.
    """
    if is_missing(start_date) or is_missing(end_date) or is_missing(unit):
        return float("nan")
    d1 = _sas_to_date(float(start_date))
    d2 = _sas_to_date(float(end_date))
    unit_u = str(unit).strip().upper().strip("'\"")

    if unit_u in ("DAY", "DAYS"):
        return float((d2 - d1).days)
    if unit_u in ("WEEK", "WEEKS"):
        return float((d2 - d1).days // 7)
    if unit_u in ("MONTH", "MONTHS"):
        return float((d2.year - d1.year) * 12 + (d2.month - d1.month))
    if unit_u in ("QTR", "QUARTER", "QTRS"):
        return float((d2.year - d1.year) * 4 + (d2.month - d1.month) // 3)
    if unit_u in ("YEAR", "YEARS"):
        return float(d2.year - d1.year)
    # Default: days
    return float((d2 - d1).days)


def _sas_to_date(sas_val: float) -> date:
    """Convert SAS date value to Python date."""
    return _SAS_EPOCH + timedelta(days=int(sas_val))


def _add_months(d: date, months: int) -> date:
    """Add months to a date."""
    total_month = d.month + months
    year = d.year + (total_month - 1) // 12
    month = ((total_month - 1) % 12) + 1
    # Clamp day
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def _beginning_of_interval(d: date, interval: str) -> date:
    if interval in ("MONTH", "MONTHS"):
        return d.replace(day=1)
    if interval in ("QTR", "QUARTER"):
        q = (d.month - 1) // 3
        return d.replace(month=q * 3 + 1, day=1)
    if interval in ("YEAR", "YEARS"):
        return d.replace(month=1, day=1)
    if interval in ("WEEK", "WEEKS"):
        return d - timedelta(days=d.weekday())
    return d


def _end_of_interval(d: date, interval: str) -> date:
    import calendar
    if interval in ("MONTH", "MONTHS"):
        max_day = calendar.monthrange(d.year, d.month)[1]
        return d.replace(day=max_day)
    if interval in ("QTR", "QUARTER"):
        q = (d.month - 1) // 3
        end_month = q * 3 + 3
        max_day = calendar.monthrange(d.year, end_month)[1]
        return d.replace(month=end_month, day=max_day)
    if interval in ("YEAR", "YEARS"):
        return d.replace(month=12, day=31)
    if interval in ("WEEK", "WEEKS"):
        return d + timedelta(days=(6 - d.weekday()))
    return d


def _middle_of_interval(d: date, interval: str) -> date:
    b = _beginning_of_interval(d, interval)
    e = _end_of_interval(d, interval)
    mid = b + (e - b) // 2
    return mid


def hour(dt: Any) -> float:
    """HOUR(datetime) — extract hour from datetime value (0-23)."""
    if is_missing(dt):
        return float("nan")
    # SAS datetime is seconds since 1960-01-01 00:00:00
    seconds = float(dt)
    hours = int(seconds // 3600) % 24
    return float(hours)


def minute(dt: Any) -> float:
    """MINUTE(datetime) — extract minute from datetime value (0-59)."""
    if is_missing(dt):
        return float("nan")
    # SAS datetime is seconds since 1960-01-01 00:00:00
    seconds = float(dt)
    minutes = int((seconds % 3600) // 60)
    return float(minutes)
