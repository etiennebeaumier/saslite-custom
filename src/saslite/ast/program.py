"""Top-level program AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saslite.ast.base import Node


@dataclass
class ProgramNode(Node):
    """Root node containing all steps."""
    steps: list[Any] = field(default_factory=list)


@dataclass
class LibnameNode(Node):
    """LIBNAME statement."""
    libref: str = ""
    engine: str = ""
    path: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionsNode(Node):
    """OPTIONS statement."""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilenameNode(Node):
    """FILENAME statement."""
    fileref: str = ""
    filepath: str = ""
