"""CSV storage backend."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from saslite.runtime.dataset import Dataset
from saslite.runtime.formatting import csv_dataframe


_MEMBER_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,31}$")


def _member_name(name: str) -> str:
    member = str(name).upper()
    if not _MEMBER_NAME_RE.fullmatch(member):
        raise ValueError(f"Invalid SAS member name: {name}")
    return member


class CsvBackend:
    """CSV file-based dataset storage."""

    def __init__(self, base_dir: str | Path, libref: str = "DISK") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._libref = libref
        self.engine = "CSV"
        self.path = self._base

    def _find_existing_path(self, name: str) -> Path:
        """Find a CSV file using SAS-style case-insensitive member names."""
        member = _member_name(name)
        exact = self._base / f"{member}.csv"
        if exact.exists():
            return exact

        if not self._base.exists():
            return exact

        for candidate in self._base.iterdir():
            if (
                candidate.is_file()
                and candidate.suffix.lower() == ".csv"
                and candidate.stem.upper() == member
            ):
                return candidate

        return exact

    def _path(self, name: str) -> Path:
        return self._find_existing_path(name)

    def read(self, name: str) -> Dataset | None:
        member = _member_name(name)
        p = self._path(name)
        if not p.exists():
            return None
        df = pd.read_csv(p)
        return Dataset.from_dataframe(df, name=member, libref=self._libref)

    def write(self, name: str, dataset: Dataset) -> None:
        member = _member_name(name)
        p = self._base / f"{member}.csv"
        csv_dataframe(dataset).to_csv(p, index=False)

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_datasets(self) -> list[str]:
        if not self._base.exists():
            return []
        return sorted(
            p.stem.upper()
            for p in self._base.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() == ".csv"
                and _MEMBER_NAME_RE.fullmatch(p.stem.upper())
            )
        )
