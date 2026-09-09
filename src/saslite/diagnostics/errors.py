"""Unified exception hierarchy for SASLite."""

from __future__ import annotations


class SASLiteError(Exception):
    """Base error for SASLite."""

    def __init__(self, message: str, source_name: str = "", line: int | None = None) -> None:
        self.source_name = source_name
        self.line = line
        parts = []
        if source_name:
            parts.append(source_name)
        if line is not None:
            parts.append(f"line {line}")
        prefix = f"[{' '.join(parts)}] " if parts else ""
        super().__init__(f"{prefix}{message}")


class ParseError(SASLiteError):
    """Syntax error during parsing."""


class MacroError(SASLiteError):
    """Error during macro expansion."""


class ExecutionError(SASLiteError):
    """Runtime execution error."""


class StorageError(SASLiteError):
    """Storage I/O error."""


class FunctionError(SASLiteError):
    """Error in built-in function call."""


class DataStepError(ExecutionError):
    """DATA step specific error."""


class SqlError(ExecutionError):
    """PROC SQL specific error."""


class ProcError(ExecutionError):
    """PROC specific error."""
