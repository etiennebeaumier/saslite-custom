"""Lark parse tree transformer — converts parse tree to SASLite AST nodes."""

from __future__ import annotations

from typing import Any

from lark import Transformer, Token, Tree

from saslite.ast.base import Span
from saslite.ast.expressions import (
    LiteralNode, VariableNode, BinaryOpNode, UnaryOpNode, FunctionCallNode,
    CaseNode, BetweenNode, LikeNode, ExistsNode, ArrayRefNode,
    CalculatedNode, ScalarSubqueryNode,
)
from saslite.ast.program import ProgramNode, FilenameNode, LibnameNode, OptionsNode
from saslite.ast.data_step import (
    DataStepNode, SetNode, DatasetRefNode, AssignNode, IfNode, DoNode,
    OutputNode, DeleteNode, StopNode, RetainNode, WhereNode, KeepNode,
    DropNode, RenameNode, FormatNode, LabelNode, MergeNode, ArrayNode,
    InputNode, InfileNode, SubstrAssignNode, LengthNode, AttribNode, PutNode,
    UpdateDataNode, CallSymputNode,
)
from saslite.ast.sql import (
    ProcSqlNode, SelectNode, SelectColumnNode, FromTableNode, JoinNode,
    CreateTableNode, InsertNode, UpdateSqlNode, DeleteSqlNode, OrderItemNode,
    SetOperationNode,
)
from saslite.ast.proc import ProcNode, VarListNode, ByNode, ClassNode, FreqTableSpec


def _non_tokens(items: list[Any]) -> list[Any]:
    """Filter out Token instances from a list of items."""
    return [item for item in items if not isinstance(item, Token) and item is not None]


def _span_from_tree(tree: Tree | Token) -> Span:
    """Extract span info from a Lark tree/token."""
    if hasattr(tree, "line") and hasattr(tree, "column"):
        return Span(start_line=tree.line, start_col=tree.column)
    if hasattr(tree, "meta") and hasattr(tree.meta, "line"):
        return Span(
            start_line=tree.meta.line,
            start_col=tree.meta.column,
            end_line=getattr(tree.meta, "end_line", 0),
            end_col=getattr(tree.meta, "end_column", 0),
        )
    return Span()


def _get_text(node: Any) -> str:
    """Extract text from a Lark token or tree."""
    if isinstance(node, Token):
        return str(node)
    # Handle already-transformed AST nodes
    if hasattr(node, "name"):
        return str(node.name)
    if isinstance(node, Tree):
        if node.data == "name" and node.children:
            return str(node.children[0])
        if node.data == "qualified_name" and node.children:
            parts = [str(c) for c in node.children if isinstance(c, Token)]
            return ".".join(parts)
    if isinstance(node, str):
        return node
    return str(node)


def _get_name(node: Any) -> str:
    """Get a NAME token value."""
    if isinstance(node, Token):
        return str(node)
    if hasattr(node, "name"):
        return str(node.name)
    if isinstance(node, str):
        return node
    return str(node).upper() if hasattr(node, "__str__") else ""


def _clean_token_value(value: Any) -> Any:
    """Convert a Lark token or transformed value into a Python option value."""
    if isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, Token):
        text = str(value)
        token_type = getattr(value, "type", "")
        if token_type == "NUMBER":
            try:
                number = float(text)
                return int(number) if number.is_integer() else number
            except ValueError:
                return text
    else:
        text = _get_text(value)

    if isinstance(text, str) and len(text) >= 2 and text[0] in ("'", '"') and text[-1] == text[0]:
        return text[1:-1]
    return text


class SasTransformer(Transformer):
    """Transform Lark parse tree into SASLite AST."""

    def __init__(self) -> None:
        super().__init__()

    # ── Top level ──────────────────────────────────

    def start(self, items: list[Any]) -> ProgramNode:
        steps = [item for item in items if item is not None]
        return ProgramNode(steps=steps)

    def statement(self, items: list[Any]) -> Any:
        return items[0] if items else None

    def empty_stmt(self, items: list[Any]) -> None:
        return None

    def data_stmt(self, items: list[Any]) -> Any:
        """Unwrap data_stmt — return the inner statement."""
        return items[0] if items else None

    def macro_let(self, items: list[Any]) -> None:
        # Handled by MacroExpander, not by parser
        return None

    def primary(self, items: list[Any]) -> Any:
        """Handle parenthesized expressions: ( expr ).
        Unwrap the parentheses and return the inner expression."""
        non_tok = _non_tokens(items)
        return non_tok[0] if non_tok else None

    # ── Parenthesized expressions ────────────────────

    # Lark with Earley parser may produce intermediate Tree nodes for
    # "( expr )" rules. The default transformer just passes them through.
    # We need to unwrap them so the inner expression is returned directly.
    # This is handled by the ? prefix on expr/or_expr/etc. in the grammar,
    # but sometimes Earley still produces a Tree for the "(" or_expr ")" rule.
    # The default Transformer behavior for unknown rules is to return a Tree,
    # so we override the default to recursively unwrap.

    # ── Expressions ────────────────────────────────

    def binop(self, items: list[Any]) -> BinaryOpNode:
        left = items[0]
        op_token = items[1] if len(items) > 1 else ""
        right = items[2] if len(items) > 2 else items[1] if len(items) > 1 else None
        op = _get_text(op_token).upper() if op_token else ""
        return BinaryOpNode(op=op, left=left, right=right)

    def unaryop(self, items: list[Any]) -> UnaryOpNode:
        if len(items) == 1:
            return UnaryOpNode(op="NOT", operand=items[0])
        op = _get_text(items[0])
        return UnaryOpNode(op=op, operand=items[1])

    def number(self, items: list[Any]) -> LiteralNode:
        text = _get_text(items[0])
        # Always use float for scientific notation, decimals, or leading dot
        if "." in text or "e" in text.lower():
            return LiteralNode(value=float(text), literal_type="number")
        return LiteralNode(value=int(text), literal_type="number")

    def string(self, items: list[Any]) -> LiteralNode:
        text = _get_text(items[0])
        # Strip quotes
        if (text.startswith("'") and text.endswith("'")) or (
            text.startswith('"') and text.endswith('"')
        ):
            text = text[1:-1]
        return LiteralNode(value=text, literal_type="string")

    def null_val(self, items: list[Any]) -> LiteralNode:
        return LiteralNode(value=None, literal_type="missing")

    def date_literal(self, items: list[Any]) -> LiteralNode:
        """SAS date/time/datetime literal: '01JAN2020'd, '12:00't, ...dt."""
        text = _get_text(items[0])
        m = None
        import re as _re
        m = _re.match(r"^(['\"])(.*)\1(dt|d|t)$", text, flags=_re.IGNORECASE)
        if not m:
            return LiteralNode(value=text, literal_type="string")
        inner, kind = m.group(2), m.group(3).lower()
        from saslite.functions.convert_funcs import input_sas
        if kind == "d":
            val = input_sas(inner, "DATE9.")
        elif kind == "dt":
            val = input_sas(inner, "DATETIME.")
        else:  # 't' — time literal hh:mm[:ss]
            parts = inner.split(":")
            try:
                h = int(parts[0])
                mi = int(parts[1]) if len(parts) > 1 else 0
                se = float(parts[2]) if len(parts) > 2 else 0.0
                val = h * 3600 + mi * 60 + se
            except (ValueError, IndexError):
                val = float("nan")
        return LiteralNode(value=val, literal_type="number")

    def name(self, items: list[Any]) -> VariableNode:
        return VariableNode(name=_get_name(items[0]))

    def qualified_name(self, items: list[Any]) -> VariableNode:
        parts = [_get_name(c) for c in items if c is not None and _get_text(c) != "."]
        return VariableNode(name=".".join(parts))

    def func_call(self, items: list[Any]) -> FunctionCallNode:
        name = _get_name(items[0])
        # With keep_all_tokens: [NAME, '(', func_args?, ')']
        # Find the args list
        args = []
        for item in items[1:]:
            if isinstance(item, list):
                args = item
                break
        return FunctionCallNode(name=name.upper(), args=args)

    def func_args(self, items: list[Any]) -> list[Any]:
        return [item for item in items
                if item is not None and not isinstance(item, Token)]

    def star_arg(self, items: list[Any]) -> LiteralNode:
        return LiteralNode(value="*", literal_type="string")

    def of_array_arg(self, items: list[Any]) -> FunctionCallNode:
        """OF arr[*] — expand to all array elements at evaluation time."""
        name = ""
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t.upper() == "OF" or t in ("[", "]", "*"):
                    continue
                if not name:
                    name = t.upper()
        return FunctionCallNode(name="_OF_ARRAY_", args=[LiteralNode(value=name, literal_type="string")])

    def expr_arg(self, items: list[Any]) -> Any:
        return items[0] if items else None

    def expr_list(self, items: list[Any]) -> list[Any]:
        return _non_tokens(items)

    def name_list(self, items: list[Any]) -> list[str]:
        return [_get_name(item) for item in items if item is not None]

    def in_list(self, items: list[Any]) -> FunctionCallNode:
        # Rewrite as IN function call
        non_tok = _non_tokens(items)
        expr = non_tok[0] if non_tok else None
        values = non_tok[1] if len(non_tok) > 1 else []
        return FunctionCallNode(name="IN", args=[expr] + (values if isinstance(values, list) else [values]))

    def not_in_list(self, items: list[Any]) -> UnaryOpNode:
        non_tok = _non_tokens(items)
        expr = non_tok[0] if non_tok else None
        values = non_tok[1] if len(non_tok) > 1 else []
        return UnaryOpNode(
            op="NOT",
            operand=FunctionCallNode(
                name="IN",
                args=[expr] + (values if isinstance(values, list) else [values]),
            ),
        )

    def like_op(self, items: list[Any]) -> LikeNode:
        non_tok = _non_tokens(items)
        return LikeNode(expr=non_tok[0], pattern=non_tok[1])

    def not_like_op(self, items: list[Any]) -> LikeNode:
        non_tok = _non_tokens(items)
        return LikeNode(expr=non_tok[0], pattern=non_tok[1], negated=True)

    def between_op(self, items: list[Any]) -> BetweenNode:
        non_tok = _non_tokens(items)
        return BetweenNode(expr=non_tok[0], low=non_tok[1], high=non_tok[2])

    def not_between_op(self, items: list[Any]) -> UnaryOpNode:
        non_tok = _non_tokens(items)
        return UnaryOpNode(op="NOT", operand=BetweenNode(expr=non_tok[0], low=non_tok[1], high=non_tok[2]))

    def is_null(self, items: list[Any]) -> BinaryOpNode:
        return BinaryOpNode(op="IS NULL", left=items[0], right=LiteralNode(value=None, literal_type="missing"))

    def is_not_null(self, items: list[Any]) -> BinaryOpNode:
        return BinaryOpNode(op="IS NOT NULL", left=items[0], right=LiteralNode(value=None, literal_type="missing"))

    def case_expr(self, items: list[Any]) -> CaseNode:
        # items: [CASE_token, expr?, (cond, result)..., ELSE?, else_result, END_token]
        conditions = []
        results = []
        else_result = None
        case_expr_val = None  # For simple CASE: the expression after CASE

        # Collect non-token, non-tuple items to detect the case expression
        non_tok_items = [it for it in items if it is not None and not isinstance(it, Token) and not (isinstance(it, tuple) and len(it) == 2)]

        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                conditions.append(item[0])
                results.append(item[1])
            elif item is not None and not isinstance(item, Token):
                # Could be case_expr or else_result
                if case_expr_val is None and not any(isinstance(it, tuple) for it in items if it is item):
                    pass  # handled below

        # Determine if simple CASE: expression appears before any WHEN tuple
        for item in items:
            if item is None or isinstance(item, Token):
                continue
            if isinstance(item, tuple) and len(item) == 2:
                break  # first WHEN found, no case_expr
            # First non-token, non-tuple item before any WHEN = case_expr
            case_expr_val = item
            break

        if case_expr_val is not None:
            # Simple CASE: rewrite WHEN values as comparisons
            new_conditions = []
            for cond in conditions:
                new_conditions.append(BinaryOpNode(op="=", left=case_expr_val, right=cond))
            conditions = new_conditions

        # ELSE result is the last non-token, non-tuple item after all WHENs
        for item in reversed(items):
            if item is None or isinstance(item, Token):
                continue
            if isinstance(item, tuple) and len(item) == 2:
                break
            else_result = item
            break

        return CaseNode(conditions=conditions, results=results, else_result=else_result)

    def case_when(self, items: list[Any]) -> tuple[Any, Any]:
        non_tok = _non_tokens(items)
        # items[0] = WHEN condition, items[1] = THEN result
        return (non_tok[0], non_tok[1]) if len(non_tok) >= 2 else (non_tok[0], None)

    # ── DATA step ──────────────────────────────────

    def data_step(self, items: list[Any]) -> DataStepNode:
        # With keep_all_tokens, items include DATA, dataset_target+, ;, data_stmt*, RUN, ;
        targets: list[str] = []
        statements = []
        for item in items:
            if isinstance(item, Token):
                continue  # Skip keyword/punctuation tokens
            if isinstance(item, VariableNode) and not statements:
                targets.append(item.name)
            elif hasattr(item, '__class__') and item.__class__.__name__ == 'DataStepNode':
                # Shouldn't happen, but handle it
                pass
            elif item is not None:
                statements.append(item)
        target = targets[0] if targets else "_NULL_"
        return DataStepNode(target=target, statements=statements,
                            extra_targets=targets[1:])

    def dataset_target(self, items: list[Any]) -> Any:
        return items[0]

    def set_stmt(self, items: list[Any]) -> SetNode:
        datasets = [item for item in items if item is not None and not isinstance(item, Token)]
        return SetNode(datasets=datasets)

    def merge_stmt(self, items: list[Any]) -> MergeNode:
        datasets = _non_tokens(items)
        return MergeNode(datasets=datasets)

    def by_stmt(self, items: list[Any]) -> ByNode:
        names = []
        for item in items:
            if isinstance(item, Token):
                kw = str(item).upper()
                if kw in ("BY", "DESCENDING"):
                    continue
            names.append(_get_name(item))
        return ByNode(variables=names)

    def array_stmt(self, items: list[Any]) -> ArrayNode:
        # ARRAY name[size] [var_list] [(init_values)]
        skip = {"[", "]", "(", ")"}
        # Filter out brackets, parens, and the ARRAY keyword itself
        filtered = []
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in skip or t.upper() == "ARRAY":
                    continue
            filtered.append(item)
        # items: [NAME, NUMBER, name_list?, expr_list?]
        name = _get_name(filtered[0]) if filtered else ""
        size = None
        variables = []
        for item in filtered[1:]:
            if isinstance(item, Token):
                t = str(item)
                if t.isdigit():
                    size = int(t)
            elif isinstance(item, list):
                if item and all(isinstance(v, str) for v in item):
                    variables = item
                elif item:
                    # Could be name_list or expr_list
                    variables = [_get_name(v) for v in item]
            elif isinstance(item, (int, float)):
                size = int(item)
        return ArrayNode(name=name, bounds=size, variables=variables)

    def input_stmt(self, items: list[Any]) -> InputNode:
        """Handle INPUT statement.

        In SAS, $ marks the PREVIOUS variable as character.
        INPUT name $ salary dept $  →  name=char, salary=numeric, dept=char
        Column mode entries (NAME [$] start-end) arrive as 4-tuples.
        """
        variables = []
        is_character: dict[str, bool] = {}
        formats: dict[str, str] = {}
        col_positions: dict[str, tuple[int, int]] = {}

        # Check if last token is a standalone trailing '$'
        trailing_dollar = False
        if items and isinstance(items[-1], Token) and str(items[-1]) == "$":
            trailing_dollar = True
            items = items[:-1]

        pending_dollar = False
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "$":
                    pending_dollar = True
                continue
            if isinstance(item, tuple) and len(item) == 5:
                # Column-mode with format: (var_name, is_char, start, end, format_spec)
                var_name, is_char, start, end, format_spec = item
                if pending_dollar and variables:
                    is_character[variables[-1].upper()] = True
                pending_dollar = False
                variables.append(var_name)
                if is_char:
                    is_character[var_name.upper()] = True
                col_positions[var_name.upper()] = (start, end)
                if format_spec:
                    formats[var_name.upper()] = format_spec
                continue
            if isinstance(item, tuple) and len(item) == 4:
                # Column-mode: (var_name, is_char, start, end)
                var_name, is_char, start, end = item
                if pending_dollar and variables:
                    is_character[variables[-1].upper()] = True
                pending_dollar = False
                variables.append(var_name)
                if is_char:
                    is_character[var_name.upper()] = True
                col_positions[var_name.upper()] = (start, end)
                continue
            if isinstance(item, dict):
                # input_var returns dict with 'is_char', 'name', 'format', 'has_leading_dollar'
                var_name = item['name']
                if var_name:
                    # If has_leading_dollar, mark PREVIOUS variable as char
                    if item.get('has_leading_dollar') and variables:
                        is_character[variables[-1].upper()] = True
                    # $ applies to the PREVIOUS variable (pending_dollar from standalone $ token)
                    if pending_dollar and variables:
                        is_character[variables[-1].upper()] = True
                    pending_dollar = False
                    variables.append(var_name)
                    if item.get('format'):
                        formats[var_name.upper()] = item['format']
                    # If is_char is True, mark this variable as char (e.g., name :$20.)
                    if item.get('is_char'):
                        is_character[var_name.upper()] = True
            elif isinstance(item, tuple):
                # Legacy tuple handling for column-mode and input_var_charfmt
                if len(item) == 5:
                    # Column-mode with format: (var_name, is_char, start, end, format_spec)
                    var_name, is_char, start, end, format_spec = item
                    if pending_dollar and variables:
                        is_character[variables[-1].upper()] = True
                    pending_dollar = False
                    variables.append(var_name)
                    if is_char:
                        is_character[var_name.upper()] = True
                    col_positions[var_name.upper()] = (start, end)
                    if format_spec:
                        formats[var_name.upper()] = format_spec
                elif len(item) == 4:
                    # Column-mode: (var_name, is_char, start, end)
                    var_name, is_char, start, end = item
                    if pending_dollar and variables:
                        is_character[variables[-1].upper()] = True
                    pending_dollar = False
                    variables.append(var_name)
                    if is_char:
                        is_character[var_name.upper()] = True
                    col_positions[var_name.upper()] = (start, end)
                elif len(item) == 3:
                    # input_var_charfmt: (is_char, var_name, format_spec)
                    is_char, var_name, format_spec = item
                    if var_name:
                        if pending_dollar and variables:
                            is_character[variables[-1].upper()] = True
                        pending_dollar = False
                        variables.append(var_name)
                        if format_spec:
                            formats[var_name.upper()] = format_spec
                        if is_char:
                            is_character[var_name.upper()] = True

        # Apply trailing $ to last variable
        if trailing_dollar and variables:
            is_character[variables[-1].upper()] = True
        # Or apply pending $ (from last input_var having $) to previous variable
        elif pending_dollar and variables:
            is_character[variables[-1].upper()] = True

        return InputNode(variables=variables, is_character=is_character,
                         formats=formats, col_positions=col_positions)

    def input_var_colrange(self, items: list[Any]) -> tuple[str, bool, int, int]:
        """Column-mode input: NAME [$] start-end."""
        name = ""
        is_char = False
        nums: list[int] = []
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "$":
                    is_char = True
                elif t == "-":
                    continue
                elif t.replace(".", "").isdigit():
                    nums.append(int(float(t)))
                elif not name:
                    name = t.upper()
        start = nums[0] if nums else 1
        end = nums[1] if len(nums) > 1 else start
        return (name, is_char, start, end)

    def input_var_colsingle(self, items: list[Any]) -> tuple[str, bool, int, int]:
        """Column-mode single position: NAME [$] pos (reads single character)."""
        name = ""
        is_char = False
        pos = 1
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "$":
                    is_char = True
                elif t.replace(".", "").isdigit():
                    pos = int(float(t))
                elif not name:
                    name = t.upper()
        return (name, is_char, pos, pos)

    def input_var_colrange_fmt(self, items: list[Any]) -> tuple[str, bool, int, int, str]:
        """Column-mode with format: NAME format start-end."""
        name = ""
        format_spec = ""
        nums: list[int] = []
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "-":
                    continue
                elif t.replace(".", "").isdigit():
                    nums.append(int(float(t)))
                elif not name:
                    name = t.upper()
            elif isinstance(item, str):
                # Format spec from input_format rule
                format_spec = item
        start = nums[0] if nums else 1
        end = nums[1] if len(nums) > 1 else start
        # Determine if char from format
        is_char = format_spec.startswith("$") if format_spec else False
        return (name, is_char, start, end, format_spec)

    def input_var_charfmt(self, items: list[Any]) -> tuple[bool, str, str]:
        """Formatted char input: NAME $ w. — treated as character marker."""
        name = ""
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in ("$", ".") or t.replace(".", "").isdigit():
                    continue
                if not name:
                    name = t.upper()
        # Return same shape as input_var with explicit char format
        return (False, name, "$")

    def input_var(self, items: list[Any]) -> dict[str, Any]:
        """Handle a single input variable: [$] NAME [: format].

        Returns dict with keys:
        - 'is_char': bool - whether variable is character type
        - 'name': str - variable name
        - 'format': str - format specification
        - 'has_leading_dollar': bool - whether $ appeared before NAME

        When $ appears BEFORE the NAME (e.g., "$ code" in "name $ code $"),
        the $ is a modifier for the PREVIOUS variable, indicated by has_leading_dollar=True.
        """
        is_char = False
        name = ""
        format_spec = ""
        has_leading_dollar = False

        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "$":
                    if not name:
                        # $ appears before NAME (e.g., "$ code")
                        # This $ modifies the previous variable
                        has_leading_dollar = True
                    else:
                        # $ appears after NAME (e.g., "name $")
                        is_char = True
                elif t == ":":
                    # Format follows, handled by input_format rule
                    pass
                else:
                    name = t.upper()
            elif isinstance(item, str):
                # This is the format_spec from input_format rule
                format_spec = item
                # Check if format starts with $ (character format like :$20.)
                if format_spec.startswith("$"):
                    is_char = True

        return {
            'is_char': is_char,
            'name': name,
            'format': format_spec,
            'has_leading_dollar': has_leading_dollar
        }

    def input_format(self, items: list[Any]) -> str:
        """Handle input format specification: [$] (NAME|NUMBER) [. (NAME|NUMBER)]

        Returns format string like "$200." or "E8601DA."
        """
        format_parts = []
        for item in items:
            if isinstance(item, Token):
                format_parts.append(str(item))
        return "".join(format_parts).upper()

    def infile_stmt(self, items: list[Any]) -> InfileNode:
        """Handle INFILE statement.

        INFILE datalines DLM='|' DSD TRUNCOVER;
        """
        source = ""
        options: dict[str, Any] = {}
        source_seen = False

        for item in items:
            if isinstance(item, Token):
                text = str(item)
                if text.upper() == "INFILE":
                    continue
                if not source_seen:
                    source = _clean_token_value(item)
                    if getattr(item, "type", "") == "NAME":
                        source = str(source).upper()
                    source_seen = True
                continue
            if isinstance(item, str) and not source_seen:
                source = _clean_token_value(item)
                source_seen = True
                continue
            if isinstance(item, tuple):
                opt_name, opt_value = item
                options[opt_name.upper()] = opt_value

        return InfileNode(source=source, options=options)

    def infile_option(self, items: list[Any]) -> tuple[str, Any]:
        """Handle INFILE option: NAME [= value]."""
        opt_name = ""
        opt_value = True  # Flag option (no value)

        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if not opt_name:
                    opt_name = t
                else:
                    # This is the value
                    opt_value = t
            elif isinstance(item, str):
                opt_value = item
            elif isinstance(item, (int, float)):
                opt_value = item

        return (opt_name, opt_value)

    def dataset_ref(self, items: list[Any]) -> DatasetRefNode:
        name_node = items[0] if items else VariableNode(name="")
        name = _get_text(name_node)
        # Collect dataset options from remaining items (skip name and paren tokens)
        options: list[dict] = []
        for item in items[1:]:
            if isinstance(item, dict):
                options.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        options.append(sub)
        if "." in name:
            parts = name.split(".", 1)
            return DatasetRefNode(name=parts[1].upper(), libref=parts[0].upper(), options=options)
        return DatasetRefNode(name=name.upper(), options=options)

    def assign_stmt(self, items: list[Any]) -> AssignNode:
        # With keep_all_tokens: [NAME, '=', expr]
        target = _get_name(items[0])
        expr = items[2] if len(items) > 2 else items[1] if len(items) > 1 else None
        return AssignNode(target=target, expr=expr)

    def arr_assign_stmt(self, items: list[Any]) -> Any:
        """arr[index] = expr;"""
        from saslite.ast.data_step import ArrayAssignNode
        name = ""
        exprs: list[Any] = []
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in ("[", "]", "="):
                    continue
                if not name:
                    name = t.upper()
            elif item is not None:
                exprs.append(item)
        index = exprs[0] if exprs else None
        expr = exprs[1] if len(exprs) > 1 else None
        return ArrayAssignNode(array_name=name, index=index, expr=expr)

    def sum_stmt(self, items: list[Any]) -> Any:
        """SAS sum statement: var + expr; (implicit retain, missing → 0)."""
        from saslite.ast.data_step import SumStatementNode
        name = ""
        expr = None
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "+":
                    continue
                if not name:
                    name = t.upper()
            elif item is not None:
                expr = item
        return SumStatementNode(target=name, expr=expr)

    def select_when_block(self, items: list[Any]) -> IfNode:
        """SELECT (expr); WHEN (v) stmt; ... OTHERWISE stmt; END;
        Lowered into a chain of IF/ELSE nodes."""
        select_expr = None
        whens: list[tuple[list, Any]] = []
        otherwise = None
        for item in items:
            if isinstance(item, Token):
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list):
                whens.append(item)
            elif isinstance(item, dict) and "otherwise" in item:
                otherwise = item["otherwise"]
            elif select_expr is None and not whens:
                select_expr = item

        # Build nested IF/ELSE from the last WHEN backwards
        node: Any = otherwise
        for values, stmt in reversed(whens):
            conds = []
            for v in values:
                if select_expr is not None:
                    conds.append(BinaryOpNode(op="=", left=select_expr, right=v))
                else:
                    conds.append(v)
            cond = conds[0]
            for extra in conds[1:]:
                cond = BinaryOpNode(op="OR", left=cond, right=extra)
            node = IfNode(condition=cond, then_stmt=stmt, else_stmt=node)
        return node if isinstance(node, IfNode) else IfNode(
            condition=LiteralNode(value=1, literal_type="number"), then_stmt=node)

    def when_clause(self, items: list[Any]) -> tuple[list, Any]:
        values: list = []
        stmt = None
        for item in items:
            if isinstance(item, Token):
                continue
            if isinstance(item, list):
                values = item
            else:
                stmt = item
        return (values, stmt)

    def otherwise_clause(self, items: list[Any]) -> dict[str, Any]:
        stmt = None
        for item in items:
            if not isinstance(item, Token):
                stmt = item
        return {"otherwise": stmt}

    def in_subquery(self, items: list[Any]) -> Any:
        """expr IN (SELECT ...) — represented as a special function call."""
        from saslite.ast.sql import SelectNode
        expr = None
        sel = None
        for item in items:
            if isinstance(item, SelectNode):
                sel = item
            elif not isinstance(item, Token) and expr is None:
                expr = item
        return FunctionCallNode(name="_IN_SUBQUERY_",
                                args=[expr, ScalarSubqueryNode(select_node=sel)])

    def not_in_subquery(self, items: list[Any]) -> UnaryOpNode:
        return UnaryOpNode(op="NOT", operand=self.in_subquery(items))

    def delete_stmt(self, items: list[Any]) -> DeleteNode:
        return DeleteNode()

    def stop_stmt(self, items: list[Any]) -> StopNode:
        return StopNode()

    def if_stmt(self, items: list[Any]) -> IfNode:
        non_tok = _non_tokens(items)
        condition = non_tok[0] if non_tok else None
        then_stmt = non_tok[1] if len(non_tok) > 1 else None
        else_stmt = non_tok[2] if len(non_tok) > 2 else None
        return IfNode(condition=condition, then_stmt=then_stmt, else_stmt=else_stmt)

    def if_subsetting(self, items: list[Any]) -> IfNode:
        """Handle standalone IF expr; — SAS subsetting IF."""
        non_tok = _non_tokens(items)
        condition = non_tok[0] if non_tok else None
        return IfNode(condition=condition, then_stmt=None, else_stmt=None)

    def do_block(self, items: list[Any]) -> DoNode:
        # Parse iterative/while/until/simple DO blocks
        # Filter out keyword/punctuation tokens from body
        skip_keywords = {"DO", "END", "TO", "BY", "WHILE", "UNTIL", ";", "(", ")"}

        do = DoNode()

        # Find TO keyword for iterative DO
        to_idx = None
        for i, item in enumerate(items):
            if isinstance(item, Token) and str(item).upper() == "TO":
                to_idx = i
                break

        if to_idx is not None:
            # Iterative DO: DO var = start TO end [BY by]
            # items: [DO, var, =, start, TO, end, (; | BY, by, ;), body..., END, ;]
            do.var = _get_name(items[1])
            do.start = items[3]  # start expr
            do.end = items[to_idx + 1]  # end expr

            # Find BY
            for j in range(to_idx + 2, len(items)):
                if isinstance(items[j], Token) and str(items[j]).upper() == "BY":
                    if j + 1 < len(items):
                        do.by = items[j + 1]
                    break

            # Body: everything after the header that isn't a keyword/token/semicolon
            body = []
            for s in items:
                if s is None or isinstance(s, Token):
                    continue
                if isinstance(s, str) and s.upper() in skip_keywords:
                    continue
                if isinstance(s, DoNode):
                    continue
                body.append(s)
            do.body = body
        else:
            # Check for WHILE/UNTIL — with keep_all_tokens the grammar yields
            # WHILE/UNTIL token, then '(' token, then the condition expr node.
            # Take the FIRST non-Token item after the keyword as the condition
            # (not items[i+1], which is the '(' token).
            while_cond = None
            until_cond = None
            for i, item in enumerate(items):
                if isinstance(item, Token):
                    keyword = str(item).upper()
                    if keyword in ("WHILE", "UNTIL"):
                        cond = None
                        for j in range(i + 1, len(items)):
                            if not isinstance(items[j], Token):
                                cond = items[j]
                                break
                        if keyword == "WHILE":
                            while_cond = cond
                        else:
                            until_cond = cond

            do.while_cond = while_cond
            do.until_cond = until_cond

            # Body for simple/while/until DO — exclude the condition node itself
            # (use identity comparison so an equal-valued body stmt is not dropped)
            body = []
            for s in items:
                if s is None or isinstance(s, Token):
                    continue
                if isinstance(s, str) and s.upper() in skip_keywords:
                    continue
                if s is while_cond or s is until_cond:
                    continue
                body.append(s)
            do.body = body

        return do

    def output_stmt(self, items: list[Any]) -> OutputNode:
        # Filter out the OUTPUT keyword token, only keep the dataset name if provided
        non_tok = _non_tokens(items)
        target = _get_text(non_tok[0]) if non_tok else ""
        return OutputNode(target=target)

    def delete_stmt(self, items: list[Any]) -> DeleteNode:
        return DeleteNode()

    def stop_stmt(self, items: list[Any]) -> StopNode:
        return StopNode()

    def retain_stmt(self, items: list[Any]) -> RetainNode:
        items_list = _non_tokens(items)
        result_items = []
        for item in items_list:
            if isinstance(item, tuple):
                result_items.append(item)
            elif isinstance(item, VariableNode):
                result_items.append((item.name, None))
            elif isinstance(item, str):
                result_items.append((item, None))
        return RetainNode(items=result_items)

    def retain_item(self, items: list[Any]) -> tuple[str, Any | None]:
        name = _get_name(items[0])
        value = items[1] if len(items) > 1 else None
        return (name, value)

    def where_stmt(self, items: list[Any]) -> WhereNode:
        non_tok = _non_tokens(items)
        return WhereNode(condition=non_tok[0] if non_tok else None)

    def keep_stmt(self, items: list[Any]) -> KeepNode:
        non_tok = _non_tokens(items)
        names = non_tok[0] if non_tok else []
        return KeepNode(variables=names if isinstance(names, list) else [names])

    def drop_stmt(self, items: list[Any]) -> DropNode:
        non_tok = _non_tokens(items)
        names = non_tok[0] if non_tok else []
        return DropNode(variables=names if isinstance(names, list) else [names])

    def rename_stmt(self, items: list[Any]) -> RenameNode:
        mapping = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                mapping[item[0]] = item[1]
        return RenameNode(mapping=mapping)

    def rename_pair(self, items: list[Any]) -> tuple[str, str]:
        return (_get_name(items[0]), _get_name(items[2]))

    def format_stmt(self, items: list[Any]) -> FormatNode:
        result_items = []
        for item in items:
            if isinstance(item, list):
                result_items.extend(t for t in item if isinstance(t, tuple))
            elif isinstance(item, tuple):
                result_items.append(item)
        return FormatNode(items=result_items)

    def format_group(self, items: list[Any]) -> list[tuple[str, str]]:
        names: list[str] = []
        fmt = ""
        for item in items:
            if isinstance(item, Token):
                names.append(_get_name(item))
            elif isinstance(item, str):
                if item.endswith(".") or item.startswith("$") or item.replace(".", "", 1).isdigit():
                    fmt = item
                else:
                    names.append(item)
        return [(name, fmt) for name in names if fmt]

    def format_item(self, items: list[Any]) -> tuple[str, str]:
        var_name = _get_name(items[0])
        fmt_name = _get_name(items[1])
        # Reconstruct format string: NAME + optional .NUMBER parts
        fmt_parts = [fmt_name]
        for item in items[2:]:
            if isinstance(item, Token):
                t = str(item)
                if t == ".":
                    fmt_parts.append(".")
                elif t.replace(".", "").isdigit():
                    fmt_parts.append(t)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, Token):
                        t = str(sub)
                        if t == ".":
                            fmt_parts.append(".")
                        elif t.replace(".", "").isdigit():
                            fmt_parts.append(t)
                    elif isinstance(sub, (int, float)):
                        fmt_parts.append(str(sub))
        fmt_str = "".join(fmt_parts)
        # Ensure format ends with dot (SAS convention)
        if not fmt_str.endswith("."):
            fmt_str += "."
        return (var_name, fmt_str)

    def label_stmt(self, items: list[Any]) -> LabelNode:
        result_items = []
        for item in items:
            if isinstance(item, tuple):
                result_items.append(item)
        return LabelNode(items=result_items)

    def label_item(self, items: list[Any]) -> tuple[str, str]:
        name = next((str(i) for i in items if isinstance(i, Token) and i.type == "NAME"), "")
        label = next((str(i) for i in items if isinstance(i, Token) and i.type == "STRING"), "")
        if (label.startswith("'") and label.endswith("'")) or (
            label.startswith('"') and label.endswith('"')
        ):
            label = label[1:-1]
        return (name, label)

    # ── PROC SQL ────────────────────────────────────

    def set_op_stmt(self, items: list[Any]) -> SetOperationNode:
        """Build a left-associative chain of set operations."""
        non_tok = _non_tokens(items)
        if not non_tok:
            return SetOperationNode()
        # non_tok = [select_stmt, (op_str, select_stmt), (op_str, select_stmt), ...]
        # But set_op returns a tuple (op, all), so it's actually:
        # [select, (op, all, select), (op, all, select), ...]
        # Let me handle both patterns
        result = non_tok[0]
        i = 1
        while i < len(non_tok):
            item = non_tok[i]
            if isinstance(item, tuple) and len(item) == 2:
                op, all_flag = item
                right = non_tok[i + 1] if i + 1 < len(non_tok) else None
                result = SetOperationNode(op=op, left=result, right=right, all=all_flag)
                i += 2
            elif isinstance(item, str):
                op = item
                all_flag = False
                right = non_tok[i + 1] if i + 1 < len(non_tok) else None
                result = SetOperationNode(op=op, left=result, right=right, all=all_flag)
                i += 2
            else:
                i += 1
        return result

    def union(self, items: list[Any]) -> tuple[str, bool]:
        return ("UNION", False)

    def union_all(self, items: list[Any]) -> tuple[str, bool]:
        return ("UNION", True)

    def intersect(self, items: list[Any]) -> tuple[str, bool]:
        return ("INTERSECT", False)

    def intersect_all(self, items: list[Any]) -> tuple[str, bool]:
        return ("INTERSECT", True)

    def except_op(self, items: list[Any]) -> tuple[str, bool]:
        return ("EXCEPT", False)

    def except_all(self, items: list[Any]) -> tuple[str, bool]:
        return ("EXCEPT", True)

    def proc_sql(self, items: list[Any]) -> ProcSqlNode:
        stmts = []
        options: dict[str, Any] = {}
        for item in _non_tokens(items):
            if isinstance(item, dict):
                options.update(item)
            elif item is not None:
                stmts.append(item)
        node = ProcSqlNode(statements=stmts)
        node.options = options
        return node

    def sql_opt(self, items: list[Any]) -> dict[str, Any]:
        for t in items:
            if isinstance(t, Token):
                return {str(t).upper(): True}
        return {}

    def into_clause(self, items: list[Any]) -> dict[str, Any]:
        names = [item for item in items
                 if isinstance(item, str) and not isinstance(item, Token)]
        return {"_INTO": names}

    def into_var(self, items: list[Any]) -> str:
        for t in items:
            if isinstance(t, Token) and str(t) != ":":
                return str(t).upper()
        return ""

    def select_stmt(self, items: list[Any]) -> SelectNode:
        sel = SelectNode()
        non_tok = _non_tokens(items)
        # Check for DISTINCT token
        has_distinct = any(isinstance(i, Token) and str(i).upper() == "DISTINCT" for i in items)
        sel.distinct = has_distinct

        # Track clause order to distinguish WHERE from HAVING
        # Grammar order: columns, from, where, group_by, having, order_by
        where_assigned = False
        for item in non_tok:
            if isinstance(item, dict) and "_INTO" in item:
                sel.into_vars = item["_INTO"]
            elif isinstance(item, list) and item:
                if isinstance(item[0], SelectColumnNode):
                    sel.columns = item
                elif isinstance(item[0], (FromTableNode, JoinNode)):
                    sel.from_clause = item
                elif isinstance(item[0], OrderItemNode):
                    sel.order_by = item
                elif isinstance(item[0], (VariableNode, BinaryOpNode, FunctionCallNode, LiteralNode, CaseNode, CalculatedNode)):
                    sel.group_by = item
                    # After group_by, next WhereNode is HAVING
                    where_assigned = True
            elif isinstance(item, WhereNode):
                if not where_assigned:
                    sel.where_clause = item
                    where_assigned = True
                else:
                    sel.having_clause = item
        return sel

    def where_clause(self, items: list[Any]) -> WhereNode:
        non_tok = _non_tokens(items)
        return WhereNode(condition=non_tok[0] if non_tok else None)

    def group_by_clause(self, items: list[Any]) -> list[Any]:
        return _non_tokens(items)

    def having_clause(self, items: list[Any]) -> WhereNode:
        non_tok = _non_tokens(items)
        return WhereNode(condition=non_tok[0] if non_tok else None)

    def order_by_clause(self, items: list[Any]) -> list[Any]:
        return [item for item in _non_tokens(items) if isinstance(item, OrderItemNode)]

    def select_list(self, items: list[Any]) -> list[SelectColumnNode]:
        return [item for item in items if isinstance(item, SelectColumnNode)]

    def select_aliased(self, items: list[Any]) -> SelectColumnNode:
        # With keep_all_tokens: [expr, AS, NAME, col_attr*, ...]
        expr = items[0]
        alias = ""
        col_length = None
        col_format = ""
        col_label = ""
        for item in items[1:]:
            if isinstance(item, Token):
                if str(item).upper() == "AS":
                    continue
                if not alias:
                    alias = _get_name(item)
            elif isinstance(item, tuple) and len(item) == 2:
                key, val = item
                key_u = key.upper() if isinstance(key, str) else ""
                if key_u == "LENGTH":
                    col_length = int(val) if val else None
                elif key_u == "FORMAT":
                    col_format = str(val)
                elif key_u == "LABEL":
                    col_label = str(val)
            elif isinstance(item, str):
                if not alias:
                    alias = item
        return SelectColumnNode(expr=expr, alias=alias, col_length=col_length, col_format=col_format, col_label=col_label)

    def select_star(self, items: list[Any]) -> SelectColumnNode:
        return SelectColumnNode(expr=VariableNode(name="*"), alias="")

    def col_attr(self, items: list[Any]) -> tuple[str, Any]:
        """Transform col_attr: LENGTH=40, FORMAT=date9., LABEL='foo'."""
        non_tok = _non_tokens(items)
        # First non-token should be key-value pair
        key = ""
        val = None
        for item in items:
            if isinstance(item, Token):
                t = str(item).upper()
                if t in ("=", ";"):
                    continue
                if not key and t in ("LENGTH", "FORMAT", "LABEL"):
                    key = t
        for item in non_tok:
            if isinstance(item, str):
                val = item
            elif isinstance(item, (int, float)):
                val = item
            elif item is not None:
                val = item
        if not key and non_tok:
            key = str(non_tok[0]).upper() if non_tok else ""
        return (key, val)

    def from_clause(self, items: list[Any]) -> list[Any]:
        # Flatten: table_expr returns a list, so we get nested lists
        result = []
        for item in _non_tokens(items):
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def table_expr(self, items: list[Any]) -> list[Any]:
        """Pass through table_expr children."""
        return [item for item in items if item is not None]

    def table_factor(self, items: list[Any]) -> Any:
        """Pass through table_factor (from_table or from_subquery)."""
        return items[0] if items else None

    def from_table(self, items: list[Any]) -> FromTableNode:
        # With keep_all_tokens: [qualified_name, (, dataset_options, ), AS?, NAME?]
        # Or: [qualified_name, AS?, NAME?]
        # Or: [qualified_name, (, dataset_options, )]
        # SAS allows: ECG1(keep=x) PDOMAIN  (options before alias)
        name_node = items[0] if items else VariableNode(name="")
        name = name_node.name if hasattr(name_node, 'name') else _get_text(name_node)
        alias = ""
        ds_options: list[Any] = []
        skip_keywords = {"AS", "ON", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS"}
        join_keywords = {"LEFT", "RIGHT", "FULL", "CROSS", "INNER"}
        found_close_paren = False
        for item in items[1:]:
            if isinstance(item, Token):
                t = str(item)
                if t == "(":
                    continue
                if t == ")":
                    found_close_paren = True
                    continue
                if t in ("=", ";", ","):
                    continue
                if t.upper() in skip_keywords:
                    # If it's a join keyword, store it for the join_clause to consume
                    if t.upper() in join_keywords:
                        SasTransformer._pending_join_type = t.upper()
                    continue
                # After close paren (or no parens), next NAME token is the alias
                if not alias:
                    alias = _get_name(item)
            elif isinstance(item, list):
                # dataset_option list from inside parens
                ds_options.extend(item)
            elif isinstance(item, dict):
                ds_options.append(item)
            elif not found_close_paren and not alias and item is not None:
                # Could be alias before options (rare but valid SAS)
                alias = _get_name(item)
            elif found_close_paren and not alias and item is not None:
                alias = _get_name(item)
        if "." in name:
            parts = name.split(".", 1)
            return FromTableNode(name=parts[1], libref=parts[0], alias=alias, ds_options=ds_options)
        return FromTableNode(name=name, alias=alias, ds_options=ds_options)

    def from_subquery(self, items: list[Any]) -> Any:
        select = None
        alias = ""
        for item in items:
            if isinstance(item, SelectNode):
                select = item
            elif isinstance(item, Token):
                text = str(item)
                if text.upper() in {"AS", "(", ")"}:
                    continue
                if not alias:
                    alias = _get_name(item)
            elif isinstance(item, str) and item.upper() != "AS":
                if not alias:
                    alias = item
        return FromTableNode(name=alias or "_SUBQUERY_", alias=alias, select=select)

    def _make_join_node(self, items: list[Any], join_type: str) -> JoinNode:
        """Create a JoinNode from join_clause items."""
        # Check for pending join type (set by from_table when it skips a join keyword)
        pending = SasTransformer._consume_pending_join_type()
        if pending and join_type == "INNER":
            join_type = pending

        non_tok = _non_tokens(items)
        table = None
        condition = None
        for item in non_tok:
            if isinstance(item, (FromTableNode,)):
                table = item
            elif item is not None and table is not None and condition is None:
                condition = item
        return JoinNode(join_type=join_type, table=table, on_condition=condition)

    def join_left(self, items: list[Any]) -> JoinNode:
        return self._make_join_node(items, "LEFT")

    def join_right(self, items: list[Any]) -> JoinNode:
        return self._make_join_node(items, "RIGHT")

    def join_full(self, items: list[Any]) -> JoinNode:
        return self._make_join_node(items, "FULL")

    def join_cross(self, items: list[Any]) -> JoinNode:
        return self._make_join_node(items, "CROSS")

    def join_inner(self, items: list[Any]) -> JoinNode:
        return self._make_join_node(items, "INNER")

    # Global storage for last join type detected from from_table
    _pending_join_type: str = ""

    @classmethod
    def _consume_pending_join_type(cls) -> str:
        jt = cls._pending_join_type
        cls._pending_join_type = ""
        return jt

    def join_type(self, items: list[Any]) -> str:
        """Unused — kept for compatibility."""
        if not items:
            return "INNER"
        return _get_text(items[0])

    def join_type(self, items: list[Any]) -> str:
        if not items:
            return "INNER"
        return _get_text(items[0])

    def order_item(self, items: list[Any]) -> OrderItemNode:
        expr = items[0] if items else None
        ascending = True
        for item in items[1:]:
            if _get_text(item).upper() == "DESC":
                ascending = False
        return OrderItemNode(expr=expr, ascending=ascending)

    def create_table_stmt(self, items: list[Any]) -> CreateTableNode:
        non_tok = _non_tokens(items)
        # non_tok should be [VariableNode(qualified_name), SelectNode]
        name_node = non_tok[0] if non_tok else VariableNode(name="")
        name = name_node.name if hasattr(name_node, 'name') else _get_text(name_node)
        select = non_tok[1] if len(non_tok) > 1 else None
        if "." in name:
            parts = name.split(".", 1)
            return CreateTableNode(name=parts[1], libref=parts[0], select=select)
        return CreateTableNode(name=name, select=select)

    def insert_col_list(self, items: list[Any]) -> list[str]:
        """Parse comma-separated column list for INSERT."""
        return [_get_name(item) for item in items if not isinstance(item, Token) or str(item) != ","]

    def insert_stmt(self, items: list[Any]) -> InsertNode:
        non_tok = _non_tokens(items)
        name_node = non_tok[0] if non_tok else VariableNode(name="")
        name = name_node.name if hasattr(name_node, 'name') else _get_text(name_node)
        libref = "WORK"
        if "." in name:
            parts = name.split(".", 1)
            name = parts[1]
            libref = parts[0]
        columns = []
        values = []
        select = None
        for item in non_tok[1:]:
            if isinstance(item, SelectNode):
                select = item
            elif isinstance(item, list):
                # Could be column list (list of strings) or expr list
                if item and all(isinstance(v, str) for v in item):
                    if not columns:
                        columns = item
                    else:
                        values = item
                else:
                    values = item
        return InsertNode(name=name, libref=libref, columns=columns, values=values, select=select)

    def update_sql_stmt(self, items: list[Any]) -> UpdateSqlNode:
        non_tok = _non_tokens(items)
        name_node = non_tok[0] if non_tok else VariableNode(name="")
        name = name_node.name if hasattr(name_node, 'name') else _get_text(name_node)
        libref = "WORK"
        if "." in name:
            parts = name.split(".", 1)
            name = parts[1]
            libref = parts[0]
        assignments = []
        where_clause = None
        for item in non_tok[1:]:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, AssignNode):
                        assignments.append(sub)
            elif isinstance(item, AssignNode):
                assignments.append(item)
            elif isinstance(item, WhereNode):
                where_clause = item.condition
        return UpdateSqlNode(name=name, libref=libref, assignments=assignments, where_clause=where_clause)

    def sql_assign(self, items: list[Any]) -> AssignNode:
        # items: [NAME, '=', expr] with keep_all_tokens
        target = ""
        expr = None
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t != "=":
                    target = t
            else:
                expr = item
        return AssignNode(target=target, expr=expr)

    def delete_sql_stmt(self, items: list[Any]) -> DeleteSqlNode:
        non_tok = _non_tokens(items)
        # First non-token should be the qualified_name
        name = ""
        condition = None
        for item in non_tok:
            if isinstance(item, WhereNode):
                condition = item.condition
            elif not name:
                name = _get_text(item)
        libref = "WORK"
        if "." in name:
            parts = name.split(".", 1)
            name = parts[1]
            libref = parts[0]
        return DeleteSqlNode(name=name, libref=libref, where_clause=condition)

    # ── PROC PRINT ──────────────────────────────────

    def proc_print(self, items: list[Any]) -> ProcNode:
        data_name = ""
        body_statements = []
        for item in _non_tokens(items):
            if isinstance(item, VariableNode):
                data_name = item.name
            elif isinstance(item, list):
                body_statements.extend(item)
            elif item is not None:
                body_statements.append(item)
        return ProcNode(proc_name="PRINT", options={"DATA": data_name}, statements=body_statements)

    def proc_print_body(self, items: list[Any]) -> list[Any]:
        return [item for item in items if item is not None]

    def print_stmt(self, items: list[Any]) -> Any:
        if not items:
            return None
        first = items[0]
        if isinstance(first, Token):
            keyword = str(first).upper()
            names = items[1] if len(items) > 1 else []
            if keyword == "VAR":
                return VarListNode(variables=names)
            elif keyword == "ID":
                return ByNode(variables=names)
            elif keyword == "SUM":
                return VarListNode(variables=names)  # Simplified
            elif keyword == "BY":
                return ByNode(variables=names)
        return first

    # ── PROC SORT ───────────────────────────────────

    def proc_sort(self, items: list[Any]) -> ProcNode:
        options: dict[str, Any] = {}
        by_vars = []
        ascending = []
        for item in _non_tokens(items):
            if isinstance(item, tuple) and len(item) == 2:
                by_vars, ascending = item
            elif isinstance(item, ByNode):
                by_vars = item.variables
            elif isinstance(item, dict):
                options.update(item)
        if ascending:
            options["_ascending"] = ascending
        return ProcNode(proc_name="SORT", options=options, statements=[ByNode(variables=by_vars)])

    def sort_options(self, items: list[Any]) -> dict[str, Any]:
        result = {}
        for item in items:
            if isinstance(item, dict):
                result.update(item)
        return result

    def sort_opt(self, items: list[Any]) -> dict[str, Any]:
        if not items:
            return {}
        key_token = items[0]
        key = _get_text(key_token).upper()
        if key in ("DATA", "OUT"):
            val_items = _non_tokens(items)
            if val_items:
                val = val_items[0]
                if hasattr(val, 'name'):
                    return {key: val.name}
                return {key: _get_text(val)}
        return {key: True}

    def sort_body(self, items: list[Any]) -> tuple[list[str], list[bool]]:
        # Filter out keyword tokens (BY, ;) but track DESCENDING
        names = []
        ascending = []
        is_next_descending = False
        for item in items:
            if item is None:
                continue
            if isinstance(item, Token):
                keyword = str(item).upper()
                if keyword == "BY" or keyword == ";":
                    continue
                if keyword == "DESCENDING":
                    is_next_descending = True
                    continue
                # NAME token
                names.append(_get_name(item))
                ascending.append(not is_next_descending)
                is_next_descending = False
            else:
                names.append(_get_name(item))
                ascending.append(not is_next_descending)
                is_next_descending = False
        return (names, ascending)

    # ── PROC CONTENTS ───────────────────────────────

    def proc_contents(self, items: list[Any]) -> ProcNode:
        options = {}
        for item in _non_tokens(items):
            if isinstance(item, dict):
                options.update(item)
        return ProcNode(proc_name="CONTENTS", options=options)

    def contents_options(self, items: list[Any]) -> dict[str, Any]:
        result = {}
        for item in items:
            if isinstance(item, dict):
                result.update(item)
        return result

    def contents_opt(self, items: list[Any]) -> dict[str, Any]:
        if not items:
            return {}
        key = _get_text(items[0]).upper()
        if key in ("DATA", "OUT"):
            val_items = _non_tokens(items)
            if val_items:
                val = val_items[0]
                if hasattr(val, 'name'):
                    return {key: val.name}
                return {key: _get_text(val)}
        return {key: True}

    # ── PROC MEANS ──────────────────────────────────


    def means_options(self, items: list[Any]) -> dict[str, Any]:
        result = {}
        for item in items:
            if isinstance(item, dict):
                result.update(item)
        return result


    def stat_name(self, items: list[Any]) -> str:
        return _get_text(items[0]).upper() if items else ""

    def means_body(self, items: list[Any]) -> list[Any]:
        return [item for item in items if item is not None]


    # ── PROC FREQ ───────────────────────────────────

    def proc_freq(self, items: list[Any]) -> ProcNode:
        options = {}
        body = []
        for item in _non_tokens(items):
            if isinstance(item, dict):
                options.update(item)
            elif isinstance(item, list):
                body.extend(item)
            elif item is not None:
                body.append(item)
        return ProcNode(proc_name="FREQ", options=options, statements=body)

    def freq_options(self, items: list[Any]) -> dict[str, Any]:
        """Merge multiple freq_opt results into a single dict."""
        result = {}
        for item in items:
            if isinstance(item, dict):
                result.update(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        result.update(sub)
        return result

    def freq_opt_data(self, items: list[Any]) -> dict[str, Any]:
        """Handle DATA= option for PROC FREQ."""
        val_items = _non_tokens(items)
        if val_items:
            val = val_items[0]
            if hasattr(val, 'name'):
                return {"DATA": val.name}
            return {"DATA": _get_text(val)}
        return {}

    def freq_opt_flag(self, items: list[Any]) -> dict[str, Any]:
        """Handle boolean flag options like NOPRINT, NOROW, NOCOL, NOPERCENT, MISSING."""
        if not items:
            return {}
        key = _get_text(items[0]).upper()
        return {key: True}

    def freq_body(self, items: list[Any]) -> list[Any]:
        return [item for item in items if item is not None]

    def freq_tables(self, items: list[Any]) -> list[Any]:
        """Handle freq_tables — returns list of FreqTableSpec."""
        specs = []
        for item in items:
            if isinstance(item, FreqTableSpec):
                specs.append(item)
            elif isinstance(item, list):
                specs.extend([s for s in item if isinstance(s, FreqTableSpec)])
        return specs

    def freq_table_spec(self, items: list[Any]) -> FreqTableSpec:
        """Handle freq_table_spec — single table like 'a * b / norow'."""
        var_names = []
        options = {}
        for item in items:
            if isinstance(item, Token):
                continue  # skip *, ., / tokens
            if isinstance(item, dict):
                options.update(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        options.update(sub)
                    elif sub is not None:
                        name = _get_name(sub)
                        if name:
                            var_names.append(name)
            else:
                name = _get_name(item)
                if name:
                    var_names.append(name)
        return FreqTableSpec(var_names=var_names, options=options)

    def freq_stmt(self, items: list[Any]) -> Any:
        return items[0] if items else None

    # ── PROC IMPORT ─────────────────────────────────

    def _extract_option(self, items: list[Any]) -> tuple[str, Any]:
        """Extract key-value pair from option items (handles both tokens and AST nodes)."""
        # Filter out '=' and ';' tokens, keep everything else
        skip = {"=", ";"}
        filtered = [i for i in items if not (isinstance(i, Token) and str(i) in skip)]
        # Also check for non-token items
        non_tok = _non_tokens(items)
        # Use whichever has more items
        candidates = filtered if len(filtered) >= len(non_tok) else non_tok
        if len(candidates) >= 2:
            key = _get_name(candidates[0]).upper()
            val = candidates[1]
            if hasattr(val, 'name'):
                val = val.name
            elif hasattr(val, 'value'):
                val = val.value
            else:
                val = _get_text(val)
            if isinstance(val, str) and len(val) >= 2 and val[0] in ("'", '"') and val[-1] == val[0]:
                val = val[1:-1]
            return (key, val)
        return ("", "")

    def proc_import(self, items: list[Any]) -> ProcNode:
        options: dict[str, Any] = {}
        non_tok = _non_tokens(items)
        for item in non_tok:
            if isinstance(item, tuple) and len(item) == 2:
                options[item[0].upper()] = item[1]
        return ProcNode(proc_name="IMPORT", options=options, statements=[])

    def import_opt(self, items: list[Any]) -> tuple[str, Any]:
        return self._extract_option(items)

    def import_stmt(self, items: list[Any]) -> tuple[str, Any]:
        return self._extract_option(items)

    # ── PROC EXPORT ─────────────────────────────────

    def proc_export(self, items: list[Any]) -> ProcNode:
        options: dict[str, Any] = {}
        non_tok = _non_tokens(items)
        for item in non_tok:
            if isinstance(item, tuple) and len(item) == 2:
                options[item[0].upper()] = item[1]
        return ProcNode(proc_name="EXPORT", options=options, statements=[])

    def export_opt(self, items: list[Any]) -> tuple[str, Any]:
        return self._extract_option(items)

    def export_stmt(self, items: list[Any]) -> tuple[str, Any]:
        return self._extract_option(items)

    # ── Global statements ───────────────────────────

    def option_value(self, items: list[Any]) -> Any:
        return _clean_token_value(items[0]) if items else ""

    def option_item(self, items: list[Any]) -> tuple[str, Any]:
        filtered = [
            item for item in items
            if not (isinstance(item, Token) and str(item) in ("=", ",", ";"))
        ]
        if not filtered:
            return ("", True)

        key = _get_name(filtered[0]).upper()
        if len(filtered) == 1:
            return (key, True)

        return (key, _clean_token_value(filtered[1]))

    def libname_option(self, items: list[Any]) -> tuple[str, Any]:
        return self.option_item(items)

    def libname_stmt(self, items: list[Any]) -> Any:
        """Handle LIBNAME libref [engine] 'path' [options...];"""
        names: list[str] = []
        path = ""
        options: dict[str, Any] = {}

        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                if key:
                    options[key.upper()] = value
                continue

            if not isinstance(item, Token):
                continue

            t = str(item)
            if t.upper() == "LIBNAME" or t == ";":
                continue

            if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                path = t[1:-1]
            else:
                names.append(t.upper())

        libref = names[0] if len(names) >= 1 else ""
        engine = names[1] if len(names) >= 2 else ""

        return LibnameNode(libref=libref, engine=engine, path=path, options=options)


    def options_stmt(self, items: list[Any]) -> Any:
        options: dict[str, Any] = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                if key:
                    options[key.upper()] = value
        return OptionsNode(options=options)

    def dataset_option_list(self, items: list[Any]) -> list[Any]:
        return [item for item in items if item is not None]

    def dataset_option(self, items: list[Any]) -> Any:
        """Parse dataset options: KEEP=, DROP=, WHERE=, RENAME="""
        if not items:
            return None

        # First item is the keyword token (KEEP, DROP, WHERE, RENAME)
        keyword = None
        for item in items:
            if isinstance(item, Token):
                kw = str(item).upper()
                if kw in ("KEEP", "DROP", "WHERE", "RENAME"):
                    keyword = kw
                    break

        if not keyword:
            return None

        if keyword == "KEEP":
            # KEEP = name_list
            names = []
            for item in items:
                if isinstance(item, list):
                    names.extend([_get_name(n) for n in item if n is not None])
            return {"KEEP": names}

        elif keyword == "DROP":
            # DROP = name_list
            names = []
            for item in items:
                if isinstance(item, list):
                    names.extend([_get_name(n) for n in item if n is not None])
            return {"DROP": names}

        elif keyword == "WHERE":
            # WHERE = ( expr )
            for item in items:
                if not isinstance(item, Token) and item is not None:
                    return {"WHERE": item}
            return None

        elif keyword == "RENAME":
            # RENAME = ( rename_pair+ )
            renames = {}
            for item in items:
                if isinstance(item, tuple) and len(item) == 2:
                    old, new = item
                    renames[old] = new
            return {"RENAME": renames}

        return None

    def sql_stmt(self, items: list[Any]) -> Any:
        """Unwrap sql_stmt."""
        return items[0] if items else None

    def print_stmt(self, items: list[Any]) -> Any:
        """Handle print_stmt — VAR, ID, SUM, BY, WHERE, FORMAT."""
        if not items:
            return None
        first = items[0]
        if isinstance(first, Token):
            keyword = str(first).upper()
            names = items[1] if len(items) > 1 else []
            if isinstance(names, list) and names and not isinstance(names[0], str):
                # Already transformed
                pass
            if keyword == "VAR":
                return VarListNode(variables=names if isinstance(names, list) else [names])
            elif keyword == "ID":
                return ByNode(variables=names if isinstance(names, list) else [names])
            elif keyword == "SUM":
                return VarListNode(variables=names if isinstance(names, list) else [names])
            elif keyword == "BY":
                return ByNode(variables=names if isinstance(names, list) else [names])
        return first if not isinstance(first, Token) else None

    def freq_stmt(self, items: list[Any]) -> Any:
        """Handle freq_stmt — TABLES, BY."""
        if not items:
            return None
        first = items[0]
        if isinstance(first, Token):
            keyword = str(first).upper()
            if keyword == "TABLES":
                # items[0]=TABLES, items[1:]=result of freq_tables (list of FreqTableSpec)
                specs = []
                for item in items[1:]:
                    if isinstance(item, FreqTableSpec):
                        specs.append(item)
                    elif isinstance(item, list):
                        specs.extend([s for s in item if isinstance(s, FreqTableSpec)])
                return specs  # Return list of FreqTableSpec directly
            elif keyword == "BY":
                names = items[1] if len(items) > 1 else []
                return ByNode(variables=names if isinstance(names, list) else [names])
        return first if not isinstance(first, Token) else None



    def distinct_arg(self, items: list[Any]) -> Any:
        """DISTINCT expr inside an aggregate — mark with a wrapper call."""
        inner = None
        for item in items:
            if not isinstance(item, Token):
                inner = item
        return FunctionCallNode(name="_DISTINCT_", args=[inner])

    def import_stmt(self, items: list[Any]) -> Any:
        return items[0] if items else None

    def export_stmt(self, items: list[Any]) -> Any:
        return items[0] if items else None

    # ── DATA step UPDATE ──────────────────────────────

    def update_data_stmt(self, items: list[Any]) -> UpdateDataNode:
        datasets = [item for item in _non_tokens(items) if isinstance(item, DatasetRefNode)]
        return UpdateDataNode(datasets=datasets)

    # ── SUBSTR assignment ─────────────────────────────

    def substr_assign_stmt(self, items: list[Any]) -> SubstrAssignNode:
        # items: [Token(SUBSTR), Token('('), Token(NAME), Token(','), start_expr,
        #         [Token(','), length_expr], Token(')'), Token('='), value_expr]
        target = ""
        start_expr = None
        length_expr = None
        value_expr = None
        non_tok = _non_tokens(items)
        tok_names = [str(it) for it in items if isinstance(it, Token)]

        # Find target name
        for it in items:
            if isinstance(it, Token) and it not in ("SUBSTR", "(", ")", ",", "=", ";"):
                target = str(it)
                break

        exprs = _non_tokens(items)
        if len(exprs) >= 1:
            start_expr = exprs[0]
        if len(exprs) >= 2:
            length_expr = exprs[1]
        if len(exprs) >= 3:
            value_expr = exprs[2]
        elif len(exprs) >= 2:
            value_expr = exprs[1]

        return SubstrAssignNode(target=target, start=start_expr, length=length_expr, expr=value_expr)

    # ── LENGTH statement ──────────────────────────────

    def length_stmt(self, items: list[Any]) -> LengthNode:
        items_list = [item for item in items if not isinstance(item, Token) and item is not None]
        return LengthNode(items=[item if isinstance(item, tuple) else (item, None) for item in items_list])

    def length_item(self, items: list[Any]) -> tuple[str, int | None]:
        name = ""
        length = None
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t.upper() == "LENGTH" or t == "$":
                    continue
                # Check if it's a NUMBER token
                if hasattr(item, 'type') and item.type == "NUMBER":
                    try:
                        length = int(t)
                    except ValueError:
                        length = None
                else:
                    name = t
            elif isinstance(item, (int, float)):
                length = int(item)
            elif isinstance(item, str):
                name = item
        return (name, length)

    # ── ATTRIB statement ──────────────────────────────

    def attrib_stmt(self, items: list[Any]) -> AttribNode:
        items_list = []
        for item in items:
            if isinstance(item, list):
                # attrib_item now returns a list of tuples
                items_list.extend(t for t in item if isinstance(t, tuple) and len(t) == 3)
            elif isinstance(item, tuple) and len(item) == 3:
                items_list.append(item)
        return AttribNode(items=items_list)

    def attrib_item(self, items: list[Any]) -> list[tuple[str, str, str]]:
        name = ""
        results = []
        for item in items:
            if isinstance(item, Token) and str(item).upper() not in ("ATTRIB",):
                name = str(item)
            elif isinstance(item, str):
                name = item
            elif isinstance(item, tuple) and len(item) == 2 and item[0]:
                results.append((name, item[0], item[1]))
        return results if results else [(name, "", "")]

    def format_spec(self, items: list[Any]) -> str:
        """Transform format spec like $10. or 8.2 into a string."""
        parts = []
        for item in items:
            if isinstance(item, Token):
                # Use .value for the actual token text
                s = str(item.value) if hasattr(item, 'value') else str(item)
                if s == "$":
                    parts.append("$")
                elif s == ".":
                    parts.append(".")
                else:
                    parts.append(s)
            elif isinstance(item, (int, float)):
                parts.append(str(int(item)))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    def attr_tag(self, items: list[Any]) -> tuple[str, str]:
        if not items or not isinstance(items[0], Token):
            return ("", "")
        keyword = str(items[0]).upper()
        # Filter out = tokens and find the actual value
        non_keyword = []
        for item in items[1:]:
            if isinstance(item, Token) and str(item) == "=":
                continue
            non_keyword.append(item)

        if keyword == "FORMAT":
            # format_spec returns a string
            val = non_keyword[0] if non_keyword else ""
            return ("FORMAT", str(val))
        elif keyword == "LABEL":
            val = _get_text(non_keyword[0]) if non_keyword else ""
            val = val.strip("'\"")
            return ("LABEL", val)
        elif keyword == "LENGTH":
            val = _get_text(non_keyword[0]) if non_keyword else ""
            return ("LENGTH", val)
        elif keyword == "INFORMAT":
            val = _get_name(non_keyword[0]) if non_keyword else ""
            return ("INFORMAT", val)
        return ("", "")

    # ── PUT statement ─────────────────────────────────

    def put_stmt(self, items: list[Any]) -> PutNode:
        non_tok = _non_tokens(items)
        return PutNode(items=non_tok)

    def put_item(self, items: list[Any]) -> Any:
        if not items:
            return None
        return items[0] if not isinstance(items[0], Token) else _get_text(items[0])

    def put_format(self, items: list[Any]) -> str:
        parts = [_get_text(it) for it in items if not isinstance(it, Token) or str(it) not in (";",)]
        return "".join(parts)

    # ── CALL routines ─────────────────────────────────

    def call_stmt(self, items: list[Any]) -> Any:
        from saslite.ast.data_step import CallMissingNode
        routine = ""
        args: list[Any] = []
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t.upper() in ("CALL", "(", ")", ";"):
                    continue
                if not routine:
                    routine = t.upper()
            elif isinstance(item, list):
                args = item
        if routine in ("SYMPUT", "SYMPUTX") and len(args) >= 2:
            return CallSymputNode(macro_var=args[0], value=args[1],
                                  trim=(routine == "SYMPUTX"))
        if routine == "MISSING":
            return CallMissingNode(variables=args)
        return None

    # ── Array subscript ───────────────────────────────

    def arr_ref(self, items: list[Any]) -> ArrayRefNode:
        name = ""
        index = None
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in ("[", "]"):
                    continue
                if not name:
                    name = t.upper()
            elif item is not None and index is None:
                index = item
        return ArrayRefNode(name=name, index=index)

    # ── EXISTS expression ─────────────────────────────

    def exists_expr(self, items: list[Any]) -> ExistsNode:
        sel = _non_tokens(items)
        return ExistsNode(select_node=sel[0] if sel else None)

    def not_exists_expr(self, items: list[Any]) -> BinaryOpNode:
        sel = _non_tokens(items)
        return BinaryOpNode(op="NOT", left=ExistsNode(select_node=sel[0] if sel else None))

    def calculated_ref(self, items: list[Any]) -> CalculatedNode:
        """CALCULATED name — reference to a computed column in SELECT."""
        # With keep_all_tokens: [CALCULATED, NAME]
        name_token = None
        for item in items:
            n = _get_name(item)
            if n.upper() != "CALCULATED":
                name_token = item
                break
        name = _get_name(name_token) if name_token is not None else ""
        return CalculatedNode(name=name)

    def scalar_subquery(self, items: list[Any]) -> ScalarSubqueryNode:
        """(SELECT ...) as scalar subquery expression."""
        sel = _non_tokens(items)
        return ScalarSubqueryNode(select_node=sel[0] if sel else None)

    # ── Window functions ──────────────────────────────

    def window_func(self, items: list[Any]) -> Any:
        from saslite.ast.expressions import WindowFuncNode
        func_name = ""
        args: list[Any] = []
        spec: dict[str, Any] = {}
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in ("(", ")") or t.upper() == "OVER":
                    continue
                if not func_name:
                    func_name = t.upper()
            elif isinstance(item, dict):
                spec = item
            elif isinstance(item, list):
                args = item
        return WindowFuncNode(
            func_name=func_name,
            args=args,
            partition_by=spec.get("partition", []),
            order_by=spec.get("order", []),
        )

    def window_spec(self, items: list[Any]) -> dict[str, Any]:
        partition: list[str] = []
        order: list[tuple[str, bool]] = []
        for item in items:
            if isinstance(item, list):
                if item and isinstance(item[0], tuple):
                    order = item
                elif item and isinstance(item[0], str):
                    partition = item
            elif isinstance(item, tuple):
                order.append(item)
        return {"partition": partition, "order": order}

    def window_name_list(self, items: list[Any]) -> list[str]:
        return [str(t).upper() for t in items
                if isinstance(t, Token) and str(t) != ","]

    def window_order_item(self, items: list[Any]) -> tuple[str, bool]:
        name = ""
        ascending = True
        for t in items:
            if isinstance(t, Token):
                s = str(t).upper()
                if s == "DESC":
                    ascending = False
                elif s == "ASC":
                    pass
                elif s != ",":
                    if not name:
                        name = s
        return (name, ascending)

    # ── PROC APPEND ───────────────────────────────────

    def proc_append(self, items: list[Any]) -> ProcNode:
        options: dict[str, Any] = {}
        for item in _non_tokens(items):
            if isinstance(item, tuple) and len(item) == 2:
                key, val = item
                if hasattr(val, "name"):
                    options[key.upper()] = val.name
                else:
                    options[key.upper()] = _get_text(val)
        return ProcNode(proc_name="APPEND", options=options)

    def append_opt(self, items: list[Any]) -> tuple[str, Any]:
        return self._extract_option(items)

    # ── PROC DATASETS ─────────────────────────────────

    def proc_datasets(self, items: list[Any]) -> ProcNode:
        options: dict[str, Any] = {}
        statements: list[Any] = []
        for item in _non_tokens(items):
            if isinstance(item, dict):
                if "action" in item:
                    # This is a ds_stmt action dict — goes to statements
                    statements.append(item)
                else:
                    # This is an options dict (from ds_opt)
                    options.update(item)
            elif isinstance(item, list):
                statements.extend(item)
            elif item is not None:
                statements.append(item)
        return ProcNode(proc_name="DATASETS", options=options, statements=statements)

    def ds_opt(self, items: list[Any]) -> dict[str, Any]:
        key, val = self._extract_option(items)
        if hasattr(val, "name"):
            val = val.name
        return {key.upper(): val}

    def ds_stmt(self, items: list[Any]) -> dict[str, Any]:
        """Transform PROC DATASETS sub-statement into a structured dict."""
        tokens = [str(t).upper() for t in items if isinstance(t, Token)]
        non_tokens = [item for item in items if not isinstance(item, Token) and item is not None]

        if any(t == "DELETE" for t in tokens):
            names = []
            for item in non_tokens:
                if isinstance(item, list):
                    names.extend(item)
                elif isinstance(item, str):
                    names.append(item)
            return {"action": "delete", "names": names}

        if any(t == "CONTENTS" for t in tokens):
            name = ""
            for item in non_tokens:
                if isinstance(item, str):
                    name = item
            return {"action": "contents", "name": name}

        if any(t == "MODIFY" for t in tokens):
            name = non_tokens[0] if non_tokens else ""
            rename = non_tokens[1] if len(non_tokens) > 1 else {}
            return {"action": "modify", "name": name, "rename": rename}

        return {"action": "unknown", "items": non_tokens}

    # ── FILENAME ──────────────────────────────────────

    def filename_stmt(self, items: list[Any]) -> FilenameNode:
        fileref = ""
        filepath = ""
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t.upper() == "FILENAME" or t == ";":
                    continue
                if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                    filepath = t[1:-1]
                else:
                    fileref = t.upper()
        return FilenameNode(fileref=fileref, filepath=filepath)

    # ── Generic helpers for new PROCs ─────────────────

    def _generic_proc(self, proc_name: str, items: list[Any]) -> ProcNode:
        """Build a ProcNode from generic option dicts / tuples and statements."""
        options: dict[str, Any] = {}
        statements: list[Any] = []
        for item in _non_tokens(items):
            if isinstance(item, dict):
                if "action" in item:
                    statements.append(item)
                else:
                    options.update(item)
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                options[item[0].upper()] = item[1]
            elif isinstance(item, list):
                statements.extend(item)
            elif item is not None:
                statements.append(item)
        return ProcNode(proc_name=proc_name, options=options, statements=statements)

    def _generic_opt(self, items: list[Any]) -> dict[str, Any]:
        """Build an options dict from `KEY = value` or flag tokens."""
        key = ""
        value: Any = True
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in ("=", ";"):
                    continue
                if not key:
                    key = t.upper()
                else:
                    value = _clean_token_value(item)
            elif item is not None:
                if hasattr(item, "name"):
                    value = item.name
                else:
                    value = _clean_token_value(item)
        return {key: value} if key else {}

    @staticmethod
    def _stmt_names(items: list[Any]) -> list[str]:
        """Extract a name list from statement items (skipping the keyword token)."""
        for item in items:
            if isinstance(item, list):
                return [n for n in item if isinstance(n, str)]
        return []

    # ── PROC TRANSPOSE ────────────────────────────────

    def proc_transpose(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("TRANSPOSE", items)

    def transpose_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def transpose_var(self, items: list[Any]) -> dict[str, Any]:
        return {"_VAR": self._stmt_names(items)}

    def transpose_id(self, items: list[Any]) -> dict[str, Any]:
        return {"_ID": self._stmt_names(items)}

    def transpose_by(self, items: list[Any]) -> dict[str, Any]:
        return {"_BY": self._stmt_names(items)}

    # ── PROC UNIVARIATE ───────────────────────────────

    def proc_univariate(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("UNIVARIATE", items)

    def univariate_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def univariate_var(self, items: list[Any]) -> VarListNode:
        return VarListNode(variables=self._stmt_names(items))

    def univariate_by(self, items: list[Any]) -> ByNode:
        return ByNode(variables=self._stmt_names(items))

    # ── PROC COMPARE ──────────────────────────────────

    def proc_compare(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("COMPARE", items)

    def compare_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def compare_var(self, items: list[Any]) -> VarListNode:
        return VarListNode(variables=self._stmt_names(items))

    def compare_id(self, items: list[Any]) -> ByNode:
        return ByNode(variables=self._stmt_names(items))

    # ── PROC COPY ─────────────────────────────────────

    def proc_copy(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("COPY", items)

    def copy_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def copy_select(self, items: list[Any]) -> dict[str, Any]:
        return {"_SELECT": self._stmt_names(items)}

    def copy_exclude(self, items: list[Any]) -> dict[str, Any]:
        return {"_EXCLUDE": self._stmt_names(items)}

    # ── PROC FORMAT ───────────────────────────────────

    def proc_format(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("FORMAT", items)

    def format_proc_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def value_stmt(self, items: list[Any]) -> dict[str, Any]:
        """VALUE [$]name range=label ... — returns a custom-format definition."""
        fmt_name = ""
        is_char = False
        ranges: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "$":
                    is_char = True
                elif t.upper() not in ("VALUE", ";"):
                    if not fmt_name:
                        fmt_name = t.upper()
            elif isinstance(item, dict):
                ranges.append(item)
        return {"action": "value", "name": fmt_name, "char": is_char, "ranges": ranges}

    def value_range(self, items: list[Any]) -> dict[str, Any]:
        keys: list[Any] = []
        label = ""
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t in ("=", ","):
                    continue
                if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                    label = t[1:-1]
            elif isinstance(item, tuple):
                keys.append(item)
        return {"keys": keys, "label": label}

    def value_key_range(self, items: list[Any]) -> tuple[str, Any, Any]:
        nums = [float(str(t)) for t in items if isinstance(t, Token) and str(t) not in ("-",)]
        return ("range", nums[0] if nums else None, nums[1] if len(nums) > 1 else None)

    def value_key_low(self, items: list[Any]) -> tuple[str, Any, Any]:
        nums = [float(str(t)) for t in items
                if isinstance(t, Token) and str(t) not in ("-",) and str(t).upper() != "LOW"]
        return ("range", float("-inf"), nums[0] if nums else None)

    def value_key_high(self, items: list[Any]) -> tuple[str, Any, Any]:
        nums = [float(str(t)) for t in items
                if isinstance(t, Token) and str(t) not in ("-",) and str(t).upper() != "HIGH"]
        return ("range", nums[0] if nums else None, float("inf"))

    def value_key_other(self, items: list[Any]) -> tuple[str, Any, Any]:
        return ("other", None, None)

    def value_key_num(self, items: list[Any]) -> tuple[str, Any, Any]:
        for t in items:
            if isinstance(t, Token):
                return ("exact", float(str(t)), None)
        return ("exact", None, None)

    def value_key_str(self, items: list[Any]) -> tuple[str, Any, Any]:
        for t in items:
            if isinstance(t, Token):
                s = str(t)
                if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
                    s = s[1:-1]
                return ("exact", s, None)
        return ("exact", "", None)

    # ── PROC TABULATE ─────────────────────────────────

    def proc_tabulate(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("TABULATE", items)

    def tabulate_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def tabulate_class(self, items: list[Any]) -> ClassNode:
        return ClassNode(variables=self._stmt_names(items))

    def tabulate_var(self, items: list[Any]) -> VarListNode:
        return VarListNode(variables=self._stmt_names(items))

    def tabulate_table(self, items: list[Any]) -> dict[str, Any]:
        terms = []
        for item in items:
            if isinstance(item, list):
                terms = item
        return {"action": "table", "terms": terms}

    def tabulate_expr(self, items: list[Any]) -> list[Any]:
        return [item for item in items if isinstance(item, list)]

    def tabulate_term(self, items: list[Any]) -> list[str]:
        return [str(t).upper() for t in items if isinstance(t, Token) and str(t) != "*"]

    # ── PROC REPORT ───────────────────────────────────

    def proc_report(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("REPORT", items)

    def report_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def report_column(self, items: list[Any]) -> dict[str, Any]:
        return {"action": "column", "names": self._stmt_names(items)}

    def report_define(self, items: list[Any]) -> dict[str, Any]:
        name = ""
        attrs: list[str] = []
        label = ""
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t.upper() in ("DEFINE", "/", ";", "="):
                    continue
                if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                    label = t[1:-1]
                elif not name:
                    name = t.upper()
                else:
                    attrs.append(t.upper())
            elif isinstance(item, str):
                s = item.strip()
                if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                    label = s[1:-1]
                else:
                    attrs.append(s.upper())
        return {"action": "define", "name": name, "attrs": attrs, "label": label}

    def report_def_attr(self, items: list[Any]) -> str:
        parts = [str(t) for t in items if isinstance(t, Token) and str(t) not in ("=",)]
        return " ".join(parts)

    def report_by(self, items: list[Any]) -> ByNode:
        return ByNode(variables=self._stmt_names(items))

    # ── PROC REG / LOGISTIC ───────────────────────────

    def proc_reg(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("REG", items)

    def reg_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def reg_model(self, items: list[Any]) -> dict[str, Any]:
        return self._parse_model_stmt(items)

    def reg_by(self, items: list[Any]) -> ByNode:
        return ByNode(variables=self._stmt_names(items))

    def reg_output(self, items: list[Any]) -> dict[str, Any]:
        return self._parse_output_stmt(items)

    def proc_logistic(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("LOGISTIC", items)

    def logistic_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def logistic_model(self, items: list[Any]) -> dict[str, Any]:
        return self._parse_model_stmt(items)

    def logistic_class(self, items: list[Any]) -> dict[str, Any]:
        """Parse CLASS statement for categorical variables."""
        vars_list = []
        options = {}
        for item in items:
            if isinstance(item, list):
                vars_list.extend([str(v).upper() for v in item if isinstance(v, str)])
            elif isinstance(item, str):
                vars_list.append(item.upper())
            elif isinstance(item, dict):
                options.update(item)
        return {"action": "class", "variables": vars_list, "options": options}

    def logistic_oddsratio(self, items: list[Any]) -> dict[str, Any]:
        """Parse ODDSRATIO statement."""
        vars_list = []
        options = {}
        for item in items:
            if isinstance(item, list):
                vars_list.extend([str(v).upper() for v in item if isinstance(v, str)])
            elif isinstance(item, str):
                vars_list.append(item.upper())
            elif isinstance(item, dict):
                options.update(item)
        return {"action": "oddsratio", "variables": vars_list, "options": options}

    def logistic_by(self, items: list[Any]) -> ByNode:
        return ByNode(variables=self._stmt_names(items))

    def logistic_output(self, items: list[Any]) -> dict[str, Any]:
        return self._parse_output_stmt(items)

    # ── PROC CORR ─────────────────────────────────────

    def proc_corr(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("CORR", items)

    def corr_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def corr_var(self, items: list[Any]) -> dict[str, Any]:
        return {"action": "var", "variables": self._stmt_names(items)}

    # ── PROC TTEST ────────────────────────────────────

    def proc_ttest(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("TTEST", items)

    def proc_npar1way(self, items: list[Any]) -> ProcNode:
        return self._generic_proc("NPAR1WAY", items)

    def npar1way_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def npar1way_fp(self, items: list[Any]) -> dict[str, Any]:
        for item in items:
            if isinstance(item, Token) and item.type in ('NUMBER', 'STRING'):
                return {'FP': True, 'FP_REFCLASS': _clean_token_value(item)}
        return {'FP': True}

    def ttest_opt(self, items: list[Any]) -> dict[str, Any]:
        return self._generic_opt(items)

    def ttest_plots(self, items: list[Any]) -> dict[str, Any]:
        return {"PLOTS": [str(item).upper() for item in items
                          if isinstance(item, Token) and item.type == "NAME"]}

    def ttest_var(self, items: list[Any]) -> dict[str, Any]:
        return {"action": "var", "variables": self._stmt_names(items)}

    def ttest_class(self, items: list[Any]) -> dict[str, Any]:
        return {"action": "class", "variables": self._stmt_names(items)}

    def ttest_paired(self, items: list[Any]) -> dict[str, Any]:
        names = [str(t) for t in items if isinstance(t, Token) and t.type == "NAME"]
        return {"action": "paired", "var1": names[0].upper() if len(names) > 0 else "",
                "var2": names[1].upper() if len(names) > 1 else ""}

    def model_opt(self, items: list[Any]) -> str:
        for t in items:
            if isinstance(t, Token):
                return str(t).upper()
        return ""

    def output_kv(self, items: list[Any]) -> tuple[str, Any]:
        return self._extract_option(items)

    def _parse_model_stmt(self, items: list[Any]) -> dict[str, Any]:
        """Parse MODEL y [(EVENT='1')] = x1 x2 ... [/ options];"""
        dependent = ""
        predictors: list[str] = []
        opts: list[str] = []
        event = ""
        seen_eq = False
        for item in items:
            if isinstance(item, Token):
                t = str(item)
                if t == "=":
                    # First '=' separates y from predictors; the EVENT= '='
                    # appears inside parens before any predictors, but the
                    # grammar yields it as a token too. Track only the first.
                    if not seen_eq and dependent:
                        seen_eq = True
                    continue
                if t.upper() in ("MODEL", "/", ";", "(", ")"):
                    continue
                if t.upper() == "EVENT":
                    continue
                if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                    event = t[1:-1]
                    continue
                if not dependent:
                    dependent = t.upper()
                elif not seen_eq:
                    # token between dependent and '=' (shouldn't happen)
                    pass
                else:
                    opts.append(t.upper())
            elif isinstance(item, list):
                predictors = [n.upper() for n in item if isinstance(n, str)]
            elif isinstance(item, str) and item:
                opts.append(item.upper())
        return {"action": "model", "y": dependent, "x": predictors,
                "options": opts, "event": event}

    def _parse_output_stmt(self, items: list[Any]) -> dict[str, Any]:
        """Parse OUTPUT OUT=ds P=var R=var;"""
        kv: dict[str, Any] = {}
        for item in items:
            if isinstance(item, tuple) and len(item) == 2 and item[0]:
                kv[item[0].upper()] = item[1]
        return {"action": "output", **kv}

    def proc_means(self, items):
        name = next(str(t).upper() for t in items if isinstance(t, Token) and str(t).upper() in ("MEANS", "SUMMARY"))
        return self._generic_proc(name, items)

    def means_opt(self, items):
        if len(items) == 1 and not isinstance(items[0], Token):
            return {str(items[0]).upper(): True}
        return self._generic_opt(items)

    def means_stmt(self, items):
        first = items[0]
        if not isinstance(first, Token):
            return first
        keyword = str(first).upper()
        if keyword == "VAR":
            return VarListNode(variables=self._stmt_names(items))
        if keyword == "CLASS":
            return ClassNode(variables=self._stmt_names(items))
        if keyword == "OUTPUT":
            out = next((i.name for i in items if isinstance(i, VariableNode)), "")
            specs = [i for i in items if isinstance(i, dict)]
            return {"action": "means_output", "out": out, "specs": specs,
                    "autoname": any(isinstance(i, Token) and str(i).upper() == "AUTONAME" for i in items)}
        return None

    def means_output_kv(self, items):
        tokens = [str(i).upper() for i in items if isinstance(i, Token)]
        equals = tokens.index("=")
        return {"stat": tokens[0], "variables": next((i for i in items if isinstance(i, list)), None),
                "names": tokens[equals+1:]}

    def analysis_by_var(self, items):
        name = next(str(i) for i in items if isinstance(i, Token) and i.type == "NAME")
        return (name, any(str(i).upper() == "DESCENDING" for i in items))

    def analysis_by_stmt(self, items):
        variables = [i for i in items if isinstance(i, tuple)]
        return ByNode(variables=[v[0] for v in variables], descending=[v[1] for v in variables],
                      notsorted=any(isinstance(i, Token) and str(i).upper() == "NOTSORTED" for i in items))

    def npar1way_stmt(self, items):
        return next((i for i in items if not isinstance(i, Token)), None)

    def ttest_stmt(self, items):
        return next((i for i in items if not isinstance(i, Token)), None)

    def signed_number(self, items):
        return float("".join(str(i) for i in items))

    def ods_graphics_opt(self, items):
        return self._generic_opt(items)

    def ods_graphics_stmt(self, items):
        options = {}
        for item in items:
            if isinstance(item, dict):
                options.update({"ODS_" + k: v for k, v in item.items()})
            elif isinstance(item, Token) and str(item).upper() in ("ON", "OFF"):
                options["ODS_GRAPHICS"] = str(item).upper() == "ON"
                options["PNG_EXPORT"] = str(item).upper() == "ON"
        return OptionsNode(options=options)

    def ods_listing_stmt(self, items):
        value = next(_clean_token_value(i) for i in items if isinstance(i, Token) and i.type == "STRING")
        return OptionsNode(options={"ODS_GPATH": value})

    def proc_sgplot(self, items):
        data = next(i.name for i in items if isinstance(i, VariableNode))
        return ProcNode(proc_name="SGPLOT", options={"DATA": data},
                        statements=[i for i in items if isinstance(i, dict)])

    def sgplot_histogram(self, items):
        return {"action": "histogram", "variables": [str(i) for i in items if isinstance(i, Token) and i.type == "NAME"]}

    def sgplot_hbox(self, items):
        return {"action": "hbox", "variables": [str(i) for i in items if isinstance(i, Token) and i.type == "NAME"]}

    def sgplot_scatter(self, items):
        return {"action": "scatter", "variables": [str(i) for i in items if isinstance(i, Token) and i.type == "NAME"]}
