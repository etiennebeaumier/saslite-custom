"""Expression AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saslite.ast.base import Node


@dataclass
class LiteralNode(Node):
    """Literal value."""
    value: Any = None
    literal_type: str = "string"  # string, number, missing


@dataclass
class VariableNode(Node):
    """Variable reference."""
    name: str = ""


@dataclass
class BinaryOpNode(Node):
    """Binary operation."""
    op: str = ""
    left: Any = None
    right: Any = None


@dataclass
class UnaryOpNode(Node):
    """Unary operation."""
    op: str = ""
    operand: Any = None


@dataclass
class FunctionCallNode(Node):
    """Function call."""
    name: str = ""
    args: list[Any] = field(default_factory=list)


@dataclass
class InListNode(Node):
    """IN (list) expression."""
    expr: Any = None
    values: list[Any] = field(default_factory=list)


@dataclass
class BetweenNode(Node):
    """BETWEEN expression."""
    expr: Any = None
    low: Any = None
    high: Any = None


@dataclass
class CaseNode(Node):
    """CASE WHEN expression."""
    conditions: list[Any] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    else_result: Any = None


@dataclass
class SubqueryNode(Node):
    """Subquery expression."""
    select_node: Any = None


@dataclass
class LikeNode(Node):
    """LIKE expression."""
    expr: Any = None
    pattern: Any = None
    negated: bool = False


@dataclass
class ExistsNode(Node):
    """EXISTS (subquery) expression."""
    select_node: Any = None


@dataclass
class CalculatedNode(Node):
    """CALCULATED column reference in PROC SQL."""
    name: str = ""


@dataclass
class ScalarSubqueryNode(Node):
    """Scalar subquery: (SELECT ... ) used as an expression."""
    select_node: Any = None


@dataclass
class ArrayRefNode(Node):
    """Array subscript reference: arr[i]."""
    name: str = ""
    index: Any = None


@dataclass
class WindowFuncNode(Node):
    """Window function: ROW_NUMBER() / RANK() / DENSE_RANK() / SUM/AVG/COUNT / LAG/LEAD
    OVER (PARTITION BY ... ORDER BY ...)."""
    func_name: str = ""
    args: list[Any] = field(default_factory=list)
    partition_by: list = None
    order_by: list = None  # list of (name, ascending)

    def __post_init__(self) -> None:
        if self.partition_by is None:
            self.partition_by = []
        if self.order_by is None:
            self.order_by = []
