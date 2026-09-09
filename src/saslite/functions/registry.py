"""Function registry — registration and dispatch for built-in SAS functions."""

from __future__ import annotations

from typing import Any, Callable


class FunctionRegistry:
    """Registry for built-in SAS functions."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._functions[name.upper()] = fn

    def call(self, name: str, args: list[Any]) -> Any:
        fn = self._functions.get(name.upper())
        if fn is None:
            raise NameError(f"Unknown function: {name}")
        return fn(*args)

    def exists(self, name: str) -> bool:
        return name.upper() in self._functions

    def get(self, name: str) -> Callable | None:
        return self._functions.get(name.upper())

    @property
    def names(self) -> list[str]:
        return sorted(self._functions.keys())
