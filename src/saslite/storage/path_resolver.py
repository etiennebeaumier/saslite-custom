"""Resolve libref to storage backend."""

from __future__ import annotations

from saslite.storage.base import StorageBackend
from saslite.storage.memory import MemoryBackend
from saslite.storage.sas_backend import SasBackend


class StorageRouter:
    """Routes dataset operations to the correct storage backend."""

    def __init__(self, work_dir: str | None = None, sas_format: str = "sas7bdat") -> None:
        self._backends: dict[str, StorageBackend] = {}
        # Default WORK library = memory
        self._backends["WORK"] = MemoryBackend()
        if work_dir:
            self._backends["DISK"] = SasBackend(work_dir, libref="DISK", format=sas_format)

    def register(self, libref: str, backend: StorageBackend) -> None:
        self._backends[libref.upper()] = backend

    def get_backend(self, libref: str) -> StorageBackend | None:
        return self._backends.get(libref.upper())

    def resolve(self, libref: str, name: str) -> tuple[StorageBackend, str]:
        """Resolve a libref.name reference to (backend, name)."""
        libref_upper = libref.upper()
        backend = self._backends.get(libref_upper)
        if backend is None:
            raise KeyError(f"Library {libref_upper} is not defined")
        return backend, name.upper()
