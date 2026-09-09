"""Variable and dataset metadata for SASLite."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(slots=True)
class VariableMetadata:
    """Metadata for a single variable."""
    name: str
    logical_name: str  # uppercase normalized
    dtype: Literal["numeric", "character", "date", "datetime", "time"]
    length: int | None = None
    format: str | None = None
    informat: str | None = None
    label: str | None = None
    retained: bool = False

    def __post_init__(self) -> None:
        self.logical_name = self.logical_name.upper()


@dataclass(slots=True)
class DatasetMetadata:
    """Metadata for a dataset."""
    libref: str
    member_name: str
    engine: Literal["memory", "csv", "parquet"] = "memory"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    row_count: int = 0
    sort_keys: list[str] = field(default_factory=list)
    variables: dict[str, VariableMetadata] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.libref}.{self.member_name}"

    def variable_names(self) -> list[str]:
        """Return variable names in creation order."""
        return [v.name for v in self.variables.values()]

    def logical_variable_names(self) -> list[str]:
        """Return logical (uppercase) variable names."""
        return [v.logical_name for v in self.variables.values()]

    def add_variable(self, var: VariableMetadata) -> None:
        self.variables[var.logical_name] = var

    def get_variable(self, name: str) -> VariableMetadata | None:
        return self.variables.get(name.upper())

    def copy(self) -> DatasetMetadata:
        import copy
        return copy.deepcopy(self)


def make_variable(
    name: str,
    dtype: str = "numeric",
    length: int | None = None,
    label: str | None = None,
    format: str | None = None,
) -> VariableMetadata:
    """Convenience factory for VariableMetadata."""
    return VariableMetadata(
        name=name,
        logical_name=name.upper(),
        dtype=dtype,  # type: ignore[arg-type]
        length=length,
        label=label,
        format=format,
    )
