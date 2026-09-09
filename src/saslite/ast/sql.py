"""PROC SQL AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saslite.ast.base import Node


@dataclass
class ProcSqlNode(Node):
    """PROC SQL block."""
    statements: list[Any] = field(default_factory=list)


@dataclass
class SelectNode(Node):
    """SELECT statement."""
    distinct: bool = False
    columns: list[Any] = field(default_factory=list)
    from_clause: Any = None
    where_clause: Any = None
    group_by: list[Any] = field(default_factory=list)
    having_clause: Any = None
    order_by: list[Any] = field(default_factory=list)
    into_vars: list[str] = field(default_factory=list)


@dataclass
class SelectColumnNode(Node):
    """A column in SELECT list."""
    expr: Any = None
    alias: str = ""
    col_length: int | None = None
    col_format: str = ""
    col_label: str = ""


@dataclass
class FromTableNode(Node):
    """Table reference in FROM."""
    name: str = ""
    libref: str = "WORK"
    alias: str = ""
    ds_options: list[Any] = field(default_factory=list)
    select: Any = None


@dataclass
class JoinNode(Node):
    """JOIN clause."""
    join_type: str = "INNER"
    table: Any = None
    on_condition: Any = None


@dataclass
class CreateTableNode(Node):
    """CREATE TABLE AS SELECT."""
    name: str = ""
    libref: str = "WORK"
    select: Any = None


@dataclass
class InsertNode(Node):
    """INSERT INTO ... VALUES/SELECT."""
    name: str = ""
    libref: str = "WORK"
    columns: list[str] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    select: Any = None


@dataclass
class UpdateSqlNode(Node):
    """UPDATE ... SET ... WHERE ..."""
    name: str = ""
    libref: str = "WORK"
    assignments: list[Any] = field(default_factory=list)
    where_clause: Any = None


@dataclass
class DeleteSqlNode(Node):
    """DELETE FROM ... WHERE ..."""
    name: str = ""
    libref: str = "WORK"
    where_clause: Any = None


@dataclass
class SetOperationNode(Node):
    """UNION/INTERSECT/EXCEPT."""
    op: str = ""
    left: Any = None
    right: Any = None
    all: bool = False


@dataclass
class OrderItemNode(Node):
    """ORDER BY item."""
    expr: Any = None
    ascending: bool = True
