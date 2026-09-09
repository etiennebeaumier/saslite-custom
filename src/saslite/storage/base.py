"""Storage backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from saslite.runtime.dataset import Dataset


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for dataset storage backends."""

    def read(self, name: str) -> Dataset | None:
        """Read a dataset by name. Returns None if not found."""
        ...

    def write(self, name: str, dataset: Dataset) -> None:
        """Write a dataset."""
        ...

    def exists(self, name: str) -> bool:
        """Check if a dataset exists."""
        ...

    def delete(self, name: str) -> bool:
        """Delete a dataset. Returns True if existed."""
        ...

    def list_datasets(self) -> list[str]:
        """List all dataset names."""
        ...
