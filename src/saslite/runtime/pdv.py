"""Program Data Vector (PDV) — the core of DATA step execution."""

from __future__ import annotations

import math
from typing import Any

from saslite.runtime.metadata import VariableMetadata
from saslite.runtime.types import MISSING_NUMERIC, is_missing


class LagState:
    """Centralized LAG/DIF state management."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, int], list[Any]] = {}

    def lag(self, var_name: str, value: Any, n: int = 1) -> Any:
        """LAGn(var) — return value from n observations ago."""
        key = (var_name.upper(), n)
        buf = self._buffers.setdefault(key, [])
        if len(buf) >= n:
            result = buf.pop(0)
        else:
            result = None
        buf.append(value)
        return result


class ByState:
    """Centralized BY-group state management."""

    def __init__(self) -> None:
        self._by_vars: list[str] = []
        self._first_flags: dict[str, bool] = {}
        self._last_flags: dict[str, bool] = {}
        self._prev_by_values: dict[str, Any] = {}

    def set_by_vars(self, by_vars: list[str]) -> None:
        """Set BY variables for FIRST./LAST. tracking."""
        self._by_vars = [v.upper() for v in by_vars]
        self._first_flags = {v: False for v in self._by_vars}
        self._last_flags = {v: False for v in self._by_vars}
        self._prev_by_values = {v: None for v in self._by_vars}

    def update_flags(self, variables: dict[str, Any], next_row: dict[str, Any] | None) -> None:
        """Update FIRST./LAST. flags based on current and next row."""
        for bv in self._by_vars:
            cur_val = variables.get(bv)
            prev_val = self._prev_by_values.get(bv)
            self._first_flags[bv] = (prev_val is None) or (prev_val != cur_val)

            if next_row is not None:
                next_val = None
                for k, v in next_row.items():
                    if k.upper() == bv:
                        next_val = v
                        break
                self._last_flags[bv] = (next_val is None) or (next_val != cur_val)
            else:
                self._last_flags[bv] = True

            self._prev_by_values[bv] = cur_val

    def get_first(self, var: str) -> bool:
        return self._first_flags.get(var.upper(), False)

    def get_last(self, var: str) -> bool:
        return self._last_flags.get(var.upper(), False)


class PDVVariable:
    """A variable slot in the PDV."""

    __slots__ = ("name", "value", "metadata", "retained", "initialized")

    def __init__(self, name: str, metadata: VariableMetadata) -> None:
        self.name = name
        self.metadata = metadata
        self.retained = metadata.retained
        self.initialized = False
        # Initialize based on type
        if metadata.dtype == "character":
            self.value: Any = ""
        else:
            self.value: Any = MISSING_NUMERIC


class PDV:
    """Program Data Vector — holds current observation state."""

    def __init__(self) -> None:
        self.variables: dict[str, PDVVariable] = {}
        self._n: int = 0
        self._error: int = 0
        self.output_flag: bool = False
        self.delete_flag: bool = False
        self.stop_flag: bool = False
        # Centralized state management
        self._retain_state: dict[str, Any] = {}
        self._lag_state: LagState = LagState()
        self._by_state: ByState = ByState()

    def add_variable(self, name: str, metadata: VariableMetadata) -> PDVVariable:
        """Add a variable to the PDV."""
        var = PDVVariable(name, metadata)
        self.variables[name.upper()] = var
        return var

    def ensure_variable(self, name: str, dtype: str = "numeric") -> PDVVariable:
        """Ensure a variable exists, creating it if needed."""
        key = name.upper()
        if key not in self.variables:
            meta = VariableMetadata(
                name=name,
                logical_name=key,
                dtype=dtype,  # type: ignore[arg-type]
            )
            self.add_variable(key, meta)
        return self.variables[key]

    def set_by_vars(self, by_vars: list[str]) -> None:
        """Set BY variables for FIRST./LAST. tracking."""
        self._by_state.set_by_vars(by_vars)

    def update_by_flags(self, next_row: dict[str, Any] | None = None) -> None:
        """Update FIRST./LAST. flags based on current and next row."""
        current_values = {k: v.value for k, v in self.variables.items()}
        self._by_state.update_flags(current_values, next_row)

    def get(self, name: str) -> Any:
        """Get variable value."""
        key = name.upper()
        # Automatic variables
        if key == "_N_":
            return self._n
        if key == "_ERROR_":
            return self._error
        if key == "_NULL_":
            return None

        # FIRST./LAST. auto-variables
        if key.startswith("FIRST."):
            bv = key[6:]
            return self._by_state.get_first(bv)
        if key.startswith("LAST."):
            bv = key[5:]
            return self._by_state.get_last(bv)

        var = self.variables.get(key)
        if var is None:
            return MISSING_NUMERIC
        return var.value

    def set(self, name: str, value: Any) -> None:
        """Set variable value."""
        key = name.upper()
        if key in ("_N_", "_ERROR_", "_NULL_"):
            return  # Cannot set automatic variables

        var = self.variables.get(key)
        if var is None:
            # Auto-create variable — infer type from value
            if isinstance(value, str):
                dtype = "character"
            else:
                dtype = "numeric"
            var = self.ensure_variable(name, dtype)
        var.value = value
        var.initialized = True

    def reset_for_iteration(self) -> None:
        """Reset PDV for a new iteration (non-retained variables)."""
        self.output_flag = False
        self.delete_flag = False
        self.stop_flag = False
        self._error = 0

        for var in self.variables.values():
            if not var.retained:
                if var.metadata.dtype == "character":
                    var.value = ""
                else:
                    var.value = MISSING_NUMERIC
                var.initialized = False

    def increment_n(self) -> None:
        self._n += 1

    def snapshot_output_row(self) -> dict[str, Any]:
        """Capture current PDV state as an output row."""
        row: dict[str, Any] = {}
        for name, var in self.variables.items():
            row[var.metadata.name] = var.value
        return row

    def load_row(self, row: dict[str, Any]) -> None:
        """Load a row into the PDV."""
        for col_name, value in row.items():
            key = col_name.upper()
            if key in self.variables:
                self.variables[key].value = value
                self.variables[key].initialized = True
