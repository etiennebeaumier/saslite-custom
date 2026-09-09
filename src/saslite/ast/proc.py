"""PROC (non-SQL) AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saslite.ast.base import Node


@dataclass
class ProcNode(Node):
    """Generic PROC statement."""
    proc_name: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    statements: list[Any] = field(default_factory=list)


@dataclass
class VarListNode(Node):
    """VAR statement."""
    variables: list[str] = field(default_factory=list)


@dataclass
class ByNode(Node):
    """BY statement."""
    variables: list[str] = field(default_factory=list)
    descending: list[bool] = field(default_factory=list)


@dataclass
class IdNode(Node):
    """ID statement."""
    variables: list[str] = field(default_factory=list)


@dataclass
class SumNode(Node):
    """SUM statement (for PROC PRINT)."""
    variables: list[str] = field(default_factory=list)


@dataclass
class ClassNode(Node):
    """CLASS statement (for PROC MEANS/FREQ)."""
    variables: list[str] = field(default_factory=list)


@dataclass
class OutputNode(Node):
    """OUTPUT OUT= statement (for PROC MEANS/FREQ)."""
    out: str = ""
    out_libref: str = "WORK"
    stats: dict[str, str] = field(default_factory=dict)


@dataclass
class TablesNode(Node):
    """TABLES statement (for PROC FREQ)."""
    table_specs: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class FreqTableSpec(Node):
    """Single table specification in PROC FREQ TABLES statement.
    e.g., a * b / norow nocol means: cross-tab of a and b, suppress row/col percents.
    """
    var_names: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcImportNode(Node):
    """PROC IMPORT."""
    datafile: str = ""
    out: str = ""
    out_libref: str = "WORK"
    dbms: str = "csv"
    getnames: bool = True
    delimiter: str = ","


@dataclass
class ProcExportNode(Node):
    """PROC EXPORT."""
    data: str = ""
    data_libref: str = "WORK"
    outfile: str = ""
    dbms: str = "csv"
    delimiter: str = ","


@dataclass
class PairedNode(Node):
    """PAIRED statement for PROC TTEST."""
    var1: str = ""
    var2: str = ""
