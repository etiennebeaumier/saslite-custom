"""Dataset abstraction wrapping pandas DataFrame with SAS metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from saslite.runtime.metadata import DatasetMetadata, VariableMetadata, make_variable


@dataclass
class Dataset:
    """A dataset: DataFrame + metadata."""
    name: str
    data: pd.DataFrame
    metadata: DatasetMetadata

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        libref: str = "WORK",
    ) -> Dataset:
        """Create Dataset from a pandas DataFrame, auto-inferring metadata."""
        variables: dict[str, VariableMetadata] = {}
        for col in df.columns:
            col_str = str(col)
            sas_dtype = _infer_sas_dtype(df[col])
            var = make_variable(col_str, dtype=sas_dtype)
            variables[var.logical_name] = var

        meta = DatasetMetadata(
            libref=libref,
            member_name=name.upper(),
            row_count=len(df),
            variables=variables,
        )
        return cls(name=name, data=df, metadata=meta)

    @classmethod
    def empty(cls, name: str, libref: str = "WORK") -> Dataset:
        """Create an empty dataset."""
        return cls(
            name=name,
            data=pd.DataFrame(),
            metadata=DatasetMetadata(libref=libref, member_name=name.upper()),
        )

    @property
    def nrow(self) -> int:
        return len(self.data)

    @property
    def ncol(self) -> int:
        return len(self.data.columns)

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    def copy(self, deep: bool = True) -> Dataset:
        return Dataset(
            name=self.name,
            data=self.data.copy(deep=deep),
            metadata=self.metadata.copy(),
        )

    def select_columns(self, columns: list[str]) -> Dataset:
        """Return new dataset with only the specified columns."""
        cols = [c for c in columns if c in self.data.columns]
        return Dataset(
            name=self.name,
            data=self.data[cols].copy(),
            metadata=self.metadata.copy(),
        )

    def rename_columns(self, mapping: dict[str, str]) -> Dataset:
        """Return new dataset with renamed columns."""
        new_meta = self.metadata.copy()
        new_vars: dict[str, VariableMetadata] = {}
        for logical_name, var in new_meta.variables.items():
            if var.name in mapping:
                new_name = mapping[var.name]
                var = VariableMetadata(
                    name=new_name,
                    logical_name=new_name.upper(),
                    dtype=var.dtype,
                    length=var.length,
                    format=var.format,
                    informat=var.informat,
                    label=var.label,
                    retained=var.retained,
                )
            new_vars[var.logical_name] = var
        new_meta.variables = new_vars
        return Dataset(
            name=self.name,
            data=self.data.rename(columns=mapping),
            metadata=new_meta,
        )


def _infer_sas_dtype(series: pd.Series) -> str:
    """Infer SAS type from pandas dtype."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "character"
