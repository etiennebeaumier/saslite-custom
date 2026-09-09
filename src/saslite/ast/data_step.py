"""DATA step AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saslite.ast.base import Node


@dataclass
class DataStepNode(Node):
    """DATA step."""
    target: str = ""
    target_options: dict[str, Any] = field(default_factory=dict)
    statements: list[Any] = field(default_factory=list)
    extra_targets: list[str] = field(default_factory=list)  # DATA a b c;


@dataclass
class SetNode(Node):
    """SET statement."""
    datasets: list[Any] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeNode(Node):
    """MERGE statement."""
    datasets: list[Any] = field(default_factory=list)
    by_vars: list[str] = field(default_factory=list)


@dataclass
class DatasetRefNode(Node):
    """Dataset reference with options."""
    name: str = ""
    libref: str = "WORK"
    options: list[Any] = field(default_factory=list)


@dataclass
class AssignNode(Node):
    """Variable assignment."""
    target: str = ""
    expr: Any = None


@dataclass
class IfNode(Node):
    """IF/THEN/ELSE statement."""
    condition: Any = None
    then_stmt: Any = None
    else_stmt: Any = None


@dataclass
class IfBlockNode(Node):
    """IF condition THEN DO; ... END; ELSE DO; ... END;"""
    condition: Any = None
    then_body: list[Any] = field(default_factory=list)
    else_body: list[Any] = field(default_factory=list)


@dataclass
class DoNode(Node):
    """DO block or iterative DO loop."""
    body: list[Any] = field(default_factory=list)
    # Iterative DO
    var: str = ""
    start: Any = None
    end: Any = None
    by: Any = None
    # WHILE/UNTIL
    while_cond: Any = None
    until_cond: Any = None


@dataclass
class OutputNode(Node):
    """OUTPUT statement."""
    target: str = ""


@dataclass
class DeleteNode(Node):
    """DELETE statement."""
    pass


@dataclass
class StopNode(Node):
    """STOP statement."""
    pass


@dataclass
class RetainNode(Node):
    """RETAIN statement."""
    items: list[tuple[str, Any | None]] = field(default_factory=list)


@dataclass
class WhereNode(Node):
    """WHERE statement."""
    condition: Any = None


@dataclass
class KeepNode(Node):
    """KEEP statement (in-step, not option)."""
    variables: list[str] = field(default_factory=list)


@dataclass
class DropNode(Node):
    """DROP statement (in-step, not option)."""
    variables: list[str] = field(default_factory=list)


@dataclass
class RenameNode(Node):
    """RENAME statement."""
    mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class ArrayNode(Node):
    """ARRAY statement."""
    name: str = ""
    bounds: Any = None
    variables: list[str] = field(default_factory=list)
    is_character: bool = False


@dataclass
class FormatNode(Node):
    """FORMAT statement."""
    items: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class LabelNode(Node):
    """LABEL statement."""
    items: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class InfileNode(Node):
    """INFILE statement — specify input source and options."""
    source: str = ""  # 'datalines', 'cards', or file path
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputNode(Node):
    """INPUT statement — variable list for reading raw data."""
    variables: list[str] = field(default_factory=list)
    is_character: dict[str, bool] = field(default_factory=dict)
    formats: dict[str, str] = field(default_factory=dict)
    datalines_data: str = ""
    # Column-mode positions: var -> (start, end), 1-based inclusive
    col_positions: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class SubstrAssignNode(Node):
    """SUBSTR(target, start [, length]) = expression;"""
    target: str = ""
    start: Any = None
    length: Any = None
    expr: Any = None


@dataclass
class LengthNode(Node):
    """LENGTH statement — set variable lengths."""
    items: list[tuple[str, int | None]] = field(default_factory=list)


@dataclass
class AttribNode(Node):
    """ATTRIB statement — set variable attributes."""
    items: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class PutNode(Node):
    """PUT statement — write to log."""
    items: list[Any] = field(default_factory=list)


@dataclass
class PutItemNode(Node):
    """A single item in a PUT list: variable, literal, or formatting."""
    expr: Any = None
    format_spec: str = ""


@dataclass
class UpdateDataNode(Node):
    """DATA step UPDATE statement (not SQL UPDATE)."""
    datasets: list[Any] = field(default_factory=list)


@dataclass
class CallSymputNode(Node):
    """CALL SYMPUT('macro_var', value)."""
    macro_var: Any = None
    value: Any = None
    trim: bool = False  # SYMPUTX trims leading/trailing blanks


@dataclass
class CallMissingNode(Node):
    """CALL MISSING(var1, var2, ...)."""
    variables: list[Any] = field(default_factory=list)


@dataclass
class ArrayAssignNode(Node):
    """arr[index] = expression;"""
    array_name: str = ""
    index: Any = None
    expr: Any = None


@dataclass
class SumStatementNode(Node):
    """SAS sum statement: var + expr; (implicit RETAIN, missing treated as 0)."""
    target: str = ""
    expr: Any = None
