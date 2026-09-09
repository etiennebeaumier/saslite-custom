"""Helpers for rendering SAS datasets with variable formats."""

from __future__ import annotations

from typing import Any

import pandas as pd

from saslite.functions.convert_funcs import put_sas
from saslite.runtime.dataset import Dataset
from saslite.runtime.types import is_missing


def formatted_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Return a copy of a dataset with supported SAS formats applied."""
    df = dataset.data.copy()
    for col in df.columns:
        var = dataset.metadata.get_variable(str(col))
        if var is None or not var.format:
            continue
        fmt = var.format
        fmt_u = fmt.upper().rstrip(".")
        if _is_supported_output_format(fmt_u):
            df[col] = df[col].map(lambda value, fmt=fmt: _format_value(value, fmt))
    return df


def csv_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Return dataset values rendered for CSV export."""
    df = formatted_dataframe(dataset)
    for col in df.columns:
        var = dataset.metadata.get_variable(str(col))
        if var is not None and var.format and _is_supported_output_format(var.format.upper().rstrip(".")):
            continue
        df[col] = df[col].map(_csv_value)
    return df


def _is_supported_output_format(fmt: str) -> bool:
    return fmt in {
        "E8601DA",
        "E8601DA10",
        "B8601DA",
        "B8601DA10",
        "YYMMDD10",
        "E8601DT",
        "E8601DT19",
        "E8601DT25",
        "IS8601DT",
        "IS8601DT25",
        "DATE",
        "DATE9",
    }


def _format_value(value: Any, fmt: str) -> str:
    if is_missing(value):
        return ""
    rendered = put_sas(value, fmt)
    return "" if rendered == "." else rendered


def _csv_value(value: Any) -> Any:
    if is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return value
