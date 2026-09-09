"""Session: the central runtime state object."""

from __future__ import annotations

from typing import Any

from saslite.runtime.dataset import Dataset
from saslite.storage.path_resolver import StorageRouter
from saslite.storage.memory import MemoryBackend


class MacroScope:
    """Scope for macro variables and macro definitions."""

    def __init__(self, name: str = "_GLOBAL_", parent: MacroScope | None = None) -> None:
        self.name = name
        self.parent = parent
        self.variables: dict[str, str] = {}
        self.macros: dict[str, Any] = {}

    def define_var(self, name: str, value: str) -> None:
        self.variables[name.upper()] = value

    def resolve_var(self, name: str) -> str | None:
        key = name.upper()
        if key in self.variables:
            return self.variables[key]
        if self.parent:
            return self.parent.resolve_var(key)
        return None

    def has_var(self, name: str) -> bool:
        return self.resolve_var(name) is not None


class Session:
    """Central session object holding all runtime state."""

    def __init__(self, storage: StorageRouter | None = None) -> None:
        self.storage = storage or StorageRouter()
        self._macro_stack: list[MacroScope] = [MacroScope("_GLOBAL_")]
        self._formats: dict[str, Any] = {}
        self._options: dict[str, Any] = {
            "LINESIZE": 80,
            "PAGESIZE": 60,
            "FIRSTOBS": 1,
            "OBS": None,
            "MISSING": ".",
            "NODATE": False,
            "NONUMBER": False,
            "MPRINT": False,
            "SYMBOLGEN": False,
            "MLOGIC": False,
        }
        self._debug_output: list[str] = []
        # Automatic macro variables
        global_scope = self._macro_stack[0]
        global_scope.define_var("SYSLAST", "")
        global_scope.define_var("SYSERR", "0")

    @property
    def macro_scope(self) -> MacroScope:
        return self._macro_stack[-1]

    @property
    def global_scope(self) -> MacroScope:
        return self._macro_stack[0]

    def push_macro_scope(self, name: str) -> MacroScope:
        scope = MacroScope(name, parent=self.macro_scope)
        self._macro_stack.append(scope)
        return scope

    def pop_macro_scope(self) -> None:
        if len(self._macro_stack) > 1:
            self._macro_stack.pop()

    def set_macro_var(self, name: str, value: str) -> None:
        self.macro_scope.define_var(name, value)

    def get_macro_var(self, name: str) -> str | None:
        return self.macro_scope.resolve_var(name)

    def get_option(self, name: str, default: Any = None) -> Any:
        return self._options.get(name.upper(), default)

    def set_option(self, name: str, value: Any) -> None:
        self._options[name.upper()] = value

    def get_dataset(self, libref: str, name: str) -> Dataset:
        backend, ds_name = self.storage.resolve(libref, name)
        ds = backend.read(ds_name)
        if ds is None:
            raise KeyError(f"Dataset {libref.upper()}.{ds_name} does not exist")
        return ds

    def put_dataset(self, libref: str, name: str, dataset: Dataset) -> None:
        backend, ds_name = self.storage.resolve(libref, name)
        backend.write(ds_name, dataset)
        self.global_scope.define_var("SYSLAST", f"{libref.upper()}.{ds_name}")

    def dataset_exists(self, libref: str, name: str) -> bool:
        backend, ds_name = self.storage.resolve(libref, name)
        return backend.exists(ds_name)

    def add_debug_output(self, message: str) -> None:
        """Add a debug message to the output log."""
        self._debug_output.append(message)

    def get_debug_output(self) -> list[str]:
        """Get all debug output messages."""
        return self._debug_output.copy()

    def clear_debug_output(self) -> None:
        """Clear debug output messages."""
        self._debug_output.clear()
