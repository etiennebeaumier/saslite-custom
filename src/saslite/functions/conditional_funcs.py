"""SAS conditional functions: IFC, IFN, COALESCE (COALESCEC is in char_funcs)."""

from __future__ import annotations

import math
from typing import Any

from saslite.runtime.types import is_missing, sas_bool


def ifc(condition: Any, true_val: str, false_val: str = "", missing_val: str = "") -> str:
    """IFC(condition, true-value, false-value [, missing-value]) — character result."""
    if is_missing(condition):
        return missing_val
    if sas_bool(condition):
        return str(true_val)
    return str(false_val)


def ifn(condition: Any, true_val: Any, false_val: Any = 0, missing_val: Any = None) -> float:
    """IFN(condition, true-value, false-value [, missing-value]) — numeric result."""
    if is_missing(condition):
        if missing_val is None:
            return float("nan")
        return float(missing_val)
    if sas_bool(condition):
        return float(true_val)
    return float(false_val)


def coalesce_num(*args: Any) -> float:
    """COALESCE(n1, n2, ...) — first non-missing numeric."""
    for a in args:
        if not is_missing(a):
            return float(a)
    return float("nan")
