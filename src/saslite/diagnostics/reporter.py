"""Diagnostic reporter for SAS-style logging."""

from __future__ import annotations

import sys
from typing import TextIO


class Reporter:
    """SAS-style diagnostic reporter (NOTE/WARNING/ERROR)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stderr
        self._notes: list[str] = []
        self._warnings: list[str] = []
        self._errors: list[str] = []

    def note(self, message: str) -> None:
        line = f"NOTE: {message}"
        self._notes.append(line)
        print(line, file=self._stream)

    def warning(self, message: str) -> None:
        line = f"WARNING: {message}"
        self._warnings.append(line)
        print(line, file=self._stream)

    def error(self, message: str) -> None:
        line = f"ERROR: {message}"
        self._errors.append(line)
        print(line, file=self._stream)

    def log(self, message: str) -> None:
        print(message, file=self._stream)

    @property
    def has_errors(self) -> bool:
        return len(self._errors) > 0

    @property
    def error_count(self) -> int:
        return len(self._errors)

    @property
    def warning_count(self) -> int:
        return len(self._warnings)

    def summary(self) -> str:
        parts = []
        if self._errors:
            parts.append(f"{len(self._errors)} error(s)")
        if self._warnings:
            parts.append(f"{len(self._warnings)} warning(s)")
        if not parts:
            return "No errors or warnings."
        return ", ".join(parts)
