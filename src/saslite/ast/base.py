"""Base AST node types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """Source location information."""
    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0
    source: str = ""


@dataclass
class Node:
    """Base AST node."""
    span: Span = field(default_factory=Span)
