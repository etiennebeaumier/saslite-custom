"""Execution result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Result of executing a single SAS step (DATA/PROC)."""
    success: bool = True
    dataset_name: str | None = None
    rows_affected: int = 0
    error: str | None = None
    output_messages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    """Summary of a full script execution."""
    success: bool = True
    steps: list[StepResult] = field(default_factory=list)
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def add_step(self, result: StepResult) -> None:
        self.steps.append(result)
        if not result.success:
            self.success = False
