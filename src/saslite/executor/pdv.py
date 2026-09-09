"""PDV wrapper for test compatibility — delegates to runtime.pdv."""

from typing import Any
from saslite.runtime.pdv import PDV as RuntimePDV, LagState, ByState


class PDV:
    """Test-compatible PDV wrapper with simplified API."""

    def __init__(self) -> None:
        self._pdv = RuntimePDV()
        self._retain_vars: set[str] = set()
        self._lag_history: dict[str, list[Any]] = {}

    def get_all_variables(self) -> dict[str, Any]:
        """Return all variable values."""
        return {k: v.value for k, v in self._pdv.variables.items()}

    def get_variable(self, name: str) -> Any:
        """Get variable value (None if not set)."""
        val = self._pdv.get(name)
        from saslite.runtime.types import is_missing
        if is_missing(val):
            return None
        return val

    def set_variable(self, name: str, value: Any, retain: bool = False) -> None:
        """Set variable value with optional RETAIN."""
        self._pdv.set(name, value)
        key = name.upper()
        if retain:
            self._retain_vars.add(key)
            if key in self._pdv.variables:
                self._pdv.variables[key].retained = True

    def reset_for_iteration(self) -> None:
        """Reset non-RETAIN variables (capture LAG state before reset)."""
        # Capture current values for LAG history
        for name, var in self._pdv.variables.items():
            key = name.upper()
            if key not in self._lag_history:
                self._lag_history[key] = []
            self._lag_history[key].append(var.value)
        self._pdv.reset_for_iteration()

    def lag(self, var_name: str, n: int = 1) -> Any:
        """LAG function — return value from n iterations ago."""
        key = var_name.upper()
        history = self._lag_history.get(key, [])
        if len(history) >= n:
            return history[-n]
        return None

    def set_by_group_markers(self, by_vars: list[str], is_first: bool, is_last: bool) -> None:
        """Set FIRST./LAST. markers for BY groups."""
        self._pdv._by_state.set_by_vars(by_vars)
        for var in by_vars:
            key = var.upper()
            self._pdv._by_state._first_flags[key] = is_first
            self._pdv._by_state._last_flags[key] = is_last

    def snapshot_output_row(self, exclude_automatic: bool = False) -> dict[str, Any]:
        """Capture current variable state."""
        row = {}
        for name, var in self._pdv.variables.items():
            if exclude_automatic and name.startswith("_"):
                continue
            row[var.metadata.name] = var.value
        return row

    def register_array(self, name: str, size: int) -> None:
        """Register an array (creates individual variables)."""
        for i in range(1, size + 1):
            var_name = f"{name}_{i}"
            self._pdv.ensure_variable(var_name)

    def set_array_element(self, array_name: str, index: int, value: Any) -> None:
        """Set array element."""
        var_name = f"{array_name}_{index}"
        self._pdv.set(var_name, value)

    def get_array_element(self, array_name: str, index: int) -> Any:
        """Get array element."""
        var_name = f"{array_name}_{index}"
        return self.get_variable(var_name)
