"""SAS conversion functions: INPUT and PUT."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from saslite.runtime.types import is_missing

_SAS_EPOCH = date(1960, 1, 1)


def input_sas(source: Any, informat: str) -> Any:
    """INPUT(source, informat.) — read character as numeric/date."""
    if is_missing(source):
        return float("nan")
    s = str(source).strip()
    if not s or s == ".":
        return float("nan")

    inf = informat.upper().rstrip(".")

    # Numeric informats
    if inf in ("BEST", "BEST32", "BEST12", "F", "COMMA", "DOLLAR", "NUMERIC") or inf.startswith("COMMA") or inf.startswith("DOLLAR"):
        s = re.sub(r"[$,]", "", s)
        try:
            return float(s)
        except ValueError:
            return float("nan")

    # Integer
    if re.match(r"^(\d+)$", inf):
        try:
            return float(s)
        except ValueError:
            return float("nan")

    # ISO 8601 date informat: e8601da -> yyyymmdd
    if inf in ("E8601DA", "E8601DA10", "YYMMDD10", "YYMMDD"):
        return _parse_iso_date(s)

    # ISO 8601 datetime informat: e8601dt -> yyyymmddThh:mm:ss
    if inf in ("E8601DT", "E8601DT25", "E8601DT19", "IS8601DT", "IS8601DT25"):
        return _parse_iso_datetime(s)

    # Date informats
    if inf in ("DATE", "DATE9", "DDMMYY", "MMDDYY", "YYMMDD", "MONYY") or inf.startswith(("DDMMYY", "MMDDYY", "YYMMDD", "DATE")):
        return _parse_date_informat(s, inf)

    if inf in ("DATETIME", "DATETIME20", "DTDATE9"):
        return _parse_datetime_informat(s, inf)

    # Fallback: try as number
    try:
        return float(s)
    except ValueError:
        return float("nan")


def put_sas(source: Any, format_spec: str) -> str:
    """PUT(source, format.) — convert numeric/date to character."""
    if is_missing(source):
        return "."

    fmt = format_spec.upper().rstrip(".")

    # ISO 8601 date format: e8601da. -> yyyymmdd
    if fmt in ("E8601DA", "E8601DA10", "YYMMDD10", "B8601DA", "B8601DA10"):
        try:
            d = _SAS_EPOCH + timedelta(days=int(float(source)))
            return d.strftime("%Y-%m-%d")
        except Exception:
            return "."

    # ISO 8601 datetime format: e8601dt25. -> yyyymmddThh:mm:ss
    if fmt in ("E8601DT", "E8601DT25", "E8601DT19", "IS8601DT", "IS8601DT25"):
        try:
            total_seconds = float(source)
            dt = datetime(1960, 1, 1) + timedelta(seconds=total_seconds)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return "."

    # BEST format
    if fmt.startswith("BEST"):
        try:
            val = float(source)
            return f"{val:g}"
        except (ValueError, TypeError):
            return "."

    # Numeric formats
    if re.match(r"^(\d+)\.(\d+)$", fmt):
        m = re.match(r"^(\d+)\.(\d+)$", fmt)
        width = int(m.group(1))
        decimals = int(m.group(2))
        try:
            val = float(source)
            return f"{val:>{width}.{decimals}f}"
        except (ValueError, TypeError):
            return "."

    if re.match(r"^(\d+)\.$", fmt):
        m = re.match(r"^(\d+)\.$", fmt)
        width = int(m.group(1))
        try:
            val = float(source)
            if val == int(val):
                return f"{int(val):>{width}}"
            return f"{val:>{width}.2f}"
        except (ValueError, TypeError):
            return "."

    # Date formats
    if fmt in ("DATE9", "DATE"):
        try:
            d = _SAS_EPOCH + timedelta(days=int(float(source)))
            return d.strftime("%d%b%Y").upper()
        except Exception:
            return "."

    if fmt == "MMDDYY10":
        try:
            d = _SAS_EPOCH + timedelta(days=int(float(source)))
            return d.strftime("%m/%d/%Y")
        except Exception:
            return "."

    if fmt == "YYMMDD10":
        try:
            d = _SAS_EPOCH + timedelta(days=int(float(source)))
            return d.strftime("%Y-%m-%d")
        except Exception:
            return "."

    # Fallback
    return str(source)


def _parse_iso_date(s: str) -> float:
    """Parse ISO 8601 date string (yyyy-mm-dd) to SAS date value."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            d = datetime.strptime(s.strip(), fmt).date()
            return float((d - _SAS_EPOCH).days)
        except ValueError:
            continue
    return float("nan")


def _parse_iso_datetime(s: str) -> float:
    """Parse ISO 8601 datetime string to SAS datetime value (seconds since 1960-01-01)."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            delta = dt - datetime(1960, 1, 1)
            return delta.total_seconds()
        except ValueError:
            continue
    return float("nan")


def _parse_date_informat(s: str, informat: str) -> float:
    """Parse a date string according to the informat."""
    formats = {
        "DATE9": "%d%b%Y",
        "DATE": "%d%b%Y",
        "MMDDYY": "%m/%d/%y",
        "MMDDYY8": "%m/%d/%y",
        "MMDDYY10": "%m/%d/%Y",
        "DDMMYY": "%d/%m/%y",
        "DDMMYY8": "%d/%m/%y",
        "DDMMYY10": "%d/%m/%Y",
        "YYMMDD": "%Y-%m-%d",
        "YYMMDD10": "%Y-%m-%d",
    }
    fmt = formats.get(informat, "%d%b%Y")
    try:
        d = datetime.strptime(s, fmt).date()
        return float((d - _SAS_EPOCH).days)
    except ValueError:
        # Try common variations
        for f in formats.values():
            try:
                d = datetime.strptime(s, f).date()
                return float((d - _SAS_EPOCH).days)
            except ValueError:
                continue
    return float("nan")


def _parse_datetime_informat(s: str, informat: str) -> float:
    """Parse a datetime string."""
    try:
        dt = datetime.strptime(s, "%d%b%Y:%H:%M:%S")
        delta = dt - datetime(1960, 1, 1)
        return delta.total_seconds()
    except ValueError:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            delta = dt - datetime(1960, 1, 1)
            return delta.total_seconds()
        except ValueError:
            return float("nan")


from datetime import timedelta  # noqa: E402


def putn(source: Any, format_name: Any) -> str:
    """PUTN(value, format) — apply a numeric format known at run time."""
    fmt = str(format_name).strip().strip("'\"")
    if not fmt.endswith("."):
        # put_sas strips a single trailing dot; w.d formats already contain one
        if not re.search(r"\.\d*$", fmt):
            fmt += "."
    return put_sas(source, fmt)


def putc(source: Any, format_name: Any) -> str:
    """PUTC(value, format) — apply a character format known at run time."""
    if is_missing(source):
        return ""
    fmt = str(format_name).strip().strip("'\"").upper().rstrip(".")
    s = str(source)
    m = re.match(r"^\$?(\d+)$", fmt.lstrip("$") if fmt.startswith("$") else fmt)
    if fmt.startswith("$"):
        m = re.match(r"^(\d+)$", fmt[1:])
    if m:
        width = int(m.group(1))
        return s[:width].ljust(width)
    return s


def inputn(source: Any, informat_name: Any) -> Any:
    """INPUTN(value, informat) — apply a numeric informat known at run time."""
    fmt = str(informat_name).strip().strip("'\"")
    if not fmt.endswith("."):
        if not re.search(r"\.\d*$", fmt):
            fmt += "."
    return input_sas(source, fmt)
