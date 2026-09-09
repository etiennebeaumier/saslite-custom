"""In-memory storage backend."""

from __future__ import annotations

from saslite.runtime.dataset import Dataset


class MemoryBackend:
    """In-memory dataset storage."""

    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}

    def read(self, name: str) -> Dataset | None:
        return self._datasets.get(name.upper())

    def write(self, name: str, dataset: Dataset) -> None:
        self._datasets[name.upper()] = dataset

    def exists(self, name: str) -> bool:
        return name.upper() in self._datasets

    def delete(self, name: str) -> bool:
        key = name.upper()
        if key in self._datasets:
            del self._datasets[key]
            return True
        return False

    def list_datasets(self) -> list[str]:
        return list(self._datasets.keys())
