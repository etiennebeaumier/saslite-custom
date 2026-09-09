"""Expression evaluator — evaluates AST expression nodes."""

from __future__ import annotations

import math
from typing import Any, Callable

from saslite.ast.expressions import (
    BinaryOpNode, FunctionCallNode, LiteralNode, UnaryOpNode, VariableNode,
    CaseNode, BetweenNode, LikeNode, ExistsNode, ArrayRefNode,
    CalculatedNode, ScalarSubqueryNode,
)
from saslite.runtime.types import is_missing, sas_bool
from saslite.diagnostics.errors import ExecutionError


class ExpressionEvaluator:
    """Evaluates expression AST nodes against a variable context."""

    def __init__(self, var_getter: Callable[[str], Any] | None = None,
                 session: Any = None) -> None:
        self._get_var = var_getter or (lambda name: None)
        self._functions: dict[str, Callable] = {}
        self._session = session
        self._arrays: dict[str, list] = {}
        self._array_vars: dict[str, list[str]] = {}  # array name -> PDV var names
        self._calculated_getter: Callable[[str], Any] | None = None

    def register_array_vars(self, name: str, var_names: list[str]) -> None:
        """Register a DATA step array as a list of PDV variable names."""
        self._array_vars[name.upper()] = [v.upper() for v in var_names]

    def set_calculated_getter(self, getter: Callable[[str], Any]) -> None:
        """Set a getter for CALCULATED column references in SQL context."""
        self._calculated_getter = getter

    def set_var_getter(self, getter: Callable[[str], Any]) -> None:
        """Replace the variable getter (for reusing evaluator across rows)."""
        self._get_var = getter

    def register_function(self, name: str, fn: Callable) -> None:
        self._functions[name.upper()] = fn

    def evaluate(self, node: Any) -> Any:
        """Evaluate an expression node, returning its value."""
        if node is None:
            return None

        if isinstance(node, LiteralNode):
            return node.value

        if isinstance(node, VariableNode):
            val = self._get_var(node.name)
            return val

        if isinstance(node, BinaryOpNode):
            return self._eval_binop(node)

        if isinstance(node, UnaryOpNode):
            return self._eval_unary(node)

        if isinstance(node, FunctionCallNode):
            return self._eval_func_call(node)

        if isinstance(node, CaseNode):
            return self._eval_case(node)

        if isinstance(node, BetweenNode):
            return self._eval_between(node)

        if isinstance(node, LikeNode):
            return self._eval_like(node)

        if isinstance(node, ExistsNode):
            return self._eval_exists(node)

        if isinstance(node, CalculatedNode):
            return self._eval_calculated(node)

        if isinstance(node, ScalarSubqueryNode):
            return self._eval_scalar_subquery(node)

        if isinstance(node, ArrayRefNode):
            return self._eval_array_ref(node)

        # Fallback for raw values
        if isinstance(node, (int, float, str)):
            return node

        raise ExecutionError(f"Cannot evaluate expression of type {type(node).__name__}")

    def _eval_binop(self, node: BinaryOpNode) -> Any:
        left = self.evaluate(node.left)
        op = node.op.upper()

        # IS NULL / IS NOT NULL (no right evaluation needed)
        if op == "IS NULL":
            return is_missing(left)
        if op == "IS NOT NULL":
            return not is_missing(left)

        right = self.evaluate(node.right)

        # Logical operators
        if op == "OR":
            return sas_bool(left) or sas_bool(right)
        if op == "AND":
            return sas_bool(left) and sas_bool(right)

        # Handle missing values in comparisons
        if op in ("=", "EQ", "NE", "<>", "^=", "~=", ">", "GT", ">=", "GE", "<", "LT", "<=", "LE"):
            return self._compare(left, op, right)

        # Arithmetic
        if is_missing(left) or is_missing(right):
            return float("nan")

        if op == "+":
            return self._safe_add(left, right)
        if op == "-":
            return self._to_num(left) - self._to_num(right)
        if op == "*":
            return self._to_num(left) * self._to_num(right)
        if op == "/":
            r = self._to_num(right)
            if r == 0:
                return float("nan")
            return self._to_num(left) / r
        if op == "**":
            return self._to_num(left) ** self._to_num(right)
        if op == "||":
            return str(left if left is not None else "") + str(right if right is not None else "")

        raise ExecutionError(f"Unknown operator: {op}")

    def _eval_case(self, node: CaseNode) -> Any:
        for cond, result in zip(node.conditions, node.results):
            if sas_bool(self.evaluate(cond)):
                return self.evaluate(result)
        if node.else_result is not None:
            return self.evaluate(node.else_result)
        return None

    def _eval_between(self, node: BetweenNode) -> Any:
        val = self.evaluate(node.expr)
        low = self.evaluate(node.low)
        high = self.evaluate(node.high)
        if is_missing(val) or is_missing(low) or is_missing(high):
            return False
        return val >= low and val <= high

    def _eval_like(self, node: LikeNode) -> Any:
        val = self.evaluate(node.expr)
        pattern = self.evaluate(node.pattern)
        from saslite.functions.char_funcs import like_match
        result = like_match(val, pattern)
        return not result if node.negated else result

    def _eval_unary(self, node: UnaryOpNode) -> Any:
        operand = self.evaluate(node.operand)
        if node.op.upper() == "NOT":
            return not sas_bool(operand)
        if node.op == "-":
            return -self._to_num(operand)
        if node.op == "+":
            return self._to_num(operand)
        return operand

    def _eval_func_call(self, node: FunctionCallNode) -> Any:
        name = node.name.upper()
        if name == "IN":
            # Special handling for IN operator
            if len(node.args) < 2:
                return False
            val = self.evaluate(node.args[0])
            for arg in node.args[1:]:
                if self.evaluate(arg) == val:
                    return True
            return False

        if name == "_IN_SUBQUERY_":
            # expr IN (SELECT col FROM ...)
            if len(node.args) < 2 or self._session is None:
                return False
            val = self.evaluate(node.args[0])
            subq = node.args[1]
            values = self._subquery_column_values(subq)
            for v in values:
                if v == val:
                    return True
                try:
                    if float(v) == float(val):
                        return True
                except (TypeError, ValueError):
                    pass
            return False

        if name == "DIM":
            # Special handling for DIM(array) - returns array size
            if len(node.args) < 1:
                raise ExecutionError("DIM requires an array name")
            # The argument should be a variable node representing the array name
            from saslite.ast.expressions import VariableNode
            if isinstance(node.args[0], VariableNode):
                array_name = node.args[0].name.upper()
                if array_name in self._array_vars:
                    return len(self._array_vars[array_name])
                if array_name in self._arrays:
                    return len(self._arrays[array_name])
            raise ExecutionError(f"DIM: array not found")

        fn = self._functions.get(name)
        if fn is None:
            raise ExecutionError(f"Unknown function: {name}")
        args = []
        for a in node.args:
            # Expand OF arr[*] into all array element values
            if (isinstance(a, FunctionCallNode) and a.name == "_OF_ARRAY_"
                    and a.args and isinstance(a.args[0], LiteralNode)):
                arr_name = str(a.args[0].value).upper()
                var_names = self._array_vars.get(arr_name)
                if var_names:
                    args.extend(self._get_var(v) for v in var_names)
                    continue
                arr_vals = self._arrays.get(arr_name)
                if arr_vals is not None:
                    args.extend(arr_vals)
                    continue
                raise ExecutionError(f"OF {arr_name}[*]: array not found")
            args.append(self.evaluate(a))
        try:
            return fn(*args)
        except Exception as e:
            raise ExecutionError(f"Error calling {name}: {e}") from e

    def _compare(self, left: Any, op: str, right: Any) -> bool:
        # Handle missing comparisons
        l_miss = is_missing(left)
        r_miss = is_missing(right)

        # Normalize operator mnemonics
        op_map = {
            "EQ": "=",
            "NE": "NE",
            "GT": ">",
            "GE": ">=",
            "LT": "<",
            "LE": "<=",
            "~=": "NE",
        }
        op = op_map.get(op, op)

        if op == "=":
            if l_miss and r_miss:
                return True  # SAS: missing = missing is True
            if l_miss or r_miss:
                return False
            return self._cmp_eq(left, right)
        if op in ("NE", "<>", "^="):
            if l_miss and r_miss:
                return False  # SAS: missing NE missing is False
            if l_miss or r_miss:
                return True
            return not self._cmp_eq(left, right)

        # Relational comparisons: missing is smallest
        if l_miss and r_miss:
            return False
        if l_miss:
            return op in ("<", "<=")
        if r_miss:
            return op in (">", ">=")

        try:
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
        except TypeError:
            # Fall back to string comparison
            return self._cmp_str(left, op, right)
        return False

    def _cmp_eq(self, left: Any, right: Any) -> bool:
        try:
            return left == right
        except Exception:
            return str(left) == str(right)

    def _cmp_str(self, left: Any, op: str, right: Any) -> bool:
        l, r = str(left), str(right)
        if op == ">":
            return l > r
        if op == ">=":
            return l >= r
        if op == "<":
            return l < r
        if op == "<=":
            return l <= r
        return False

    def _safe_add(self, left: Any, right: Any) -> Any:
        try:
            return self._to_num(left) + self._to_num(right)
        except (TypeError, ValueError):
            return str(left) + str(right)

    @staticmethod
    def _to_num(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val = val.strip()
            if val == "" or val == ".":
                return float("nan")
            return float(val)
        return float("nan")

    def register_array(self, name: str, values: list) -> None:
        """Register an array for subscript access."""
        self._arrays[name.upper()] = values

    def _eval_array_ref(self, node: ArrayRefNode) -> Any:
        """Evaluate array subscript: arr[i]."""
        idx = self.evaluate(node.index)
        if idx is None or (isinstance(idx, float) and math.isnan(idx)):
            return float("nan")
        idx = int(idx) - 1  # SAS arrays are 1-based
        arr = self._arrays.get(node.name.upper())
        if arr is None:
            # DATA step arrays are registered as accessor functions
            fn = self._functions.get(node.name.upper())
            if fn is not None:
                try:
                    return fn(idx + 1)
                except Exception:
                    return float("nan")
            # Try getting from variable getter (single variable treated as 1-element array)
            val = self._get_var(node.name)
            if idx == 0:
                return val
            return float("nan")
        if 0 <= idx < len(arr):
            return arr[idx]
        return float("nan")

    def _eval_exists(self, node: ExistsNode) -> Any:
        """Evaluate EXISTS(subquery) — returns True if subquery returns any rows.
        Uses cached dataset to avoid repeated loading."""
        if self._session is None:
            return False
        from saslite.ast.sql import SelectNode, FromTableNode
        sel = node.select_node
        if not isinstance(sel, SelectNode):
            return False
        from_table = sel.from_clause[0] if sel.from_clause else None
        if not isinstance(from_table, FromTableNode):
            return False
        try:
            # Use cached dataset from session if available
            cache_key = (from_table.libref.upper(), from_table.name.upper())
            if not hasattr(self, '_exists_cache'):
                self._exists_cache = {}
            if cache_key not in self._exists_cache:
                ds = self._session.get_dataset(from_table.libref, from_table.name)
                self._exists_cache[cache_key] = ds.data.copy()
            df = self._exists_cache[cache_key]
            if sel.where_clause:
                inner_alias = (from_table.alias or from_table.name).upper()
                mask = self._eval_exists_where(sel.where_clause, df, inner_alias)
                df = df[mask]
            return len(df) > 0
        except (KeyError, Exception):
            return False

    def _eval_exists_where(self, where_node: Any, df: Any,
                           inner_alias: str = "") -> list:
        """Evaluate WHERE for EXISTS — return boolean mask.

        Inner-table columns resolve first (with or without the inner table's
        alias prefix); anything else falls back to the OUTER row's variables,
        enabling correlated subqueries like `d.id = e.id`.
        """
        condition = where_node if not hasattr(where_node, "condition") else where_node.condition
        col_map = {c.upper(): c for c in df.columns}
        outer_get = self._get_var
        # Pre-register functions once
        fn_list = [(name, self._functions[name]) for name in self._functions
                    if name in self._functions and self._functions[name] is not None]
        mask = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()

            def _gv(n, _row=row, _cm=col_map):
                upper = n.upper()
                # Bare inner column
                if upper in _cm:
                    return _row[_cm[upper]]
                # Alias-qualified reference
                if "." in upper:
                    prefix, bare = upper.split(".", 1)
                    if prefix == inner_alias and bare in _cm:
                        return _row[_cm[bare]]
                    # Outer-qualified (e.g. E.DEPT_ID) — try outer row
                    outer_val = outer_get(n)
                    if outer_val is None:
                        outer_val = outer_get(bare)
                    return outer_val
                # Fallback: outer row variable (correlated reference)
                return outer_get(n)

            ev = ExpressionEvaluator(var_getter=_gv, session=self._session)
            for name, fn in fn_list:
                ev.register_function(name, fn)
            ev._calculated_getter = self._calculated_getter
            ev._exists_cache = getattr(self, '_exists_cache', {})
            try:
                mask.append(sas_bool(ev.evaluate(condition)))
            except Exception:
                mask.append(False)
        return mask

    def _eval_calculated(self, node: CalculatedNode) -> Any:
        """Evaluate CALCULATED name — reference to a computed column in current SELECT."""
        if self._calculated_getter is not None:
            return self._calculated_getter(node.name)
        return self._get_var(node.name)

    def _subquery_column_values(self, subq: Any) -> list:
        """Return all values of a subquery's first column (for IN (SELECT ...))."""
        from saslite.ast.sql import SelectNode, FromTableNode, SelectColumnNode
        sel = getattr(subq, "select_node", None)
        if not isinstance(sel, SelectNode) or self._session is None:
            return []
        try:
            from_table = sel.from_clause[0] if sel.from_clause else None
            if not isinstance(from_table, FromTableNode):
                return []
            cache_key = (from_table.libref.upper(), from_table.name.upper())
            if not hasattr(self, "_exists_cache"):
                self._exists_cache = {}
            if cache_key not in self._exists_cache:
                ds = self._session.get_dataset(from_table.libref, from_table.name)
                self._exists_cache[cache_key] = ds.data.copy()
            df = self._exists_cache[cache_key]
            if sel.where_clause:
                inner_alias = (from_table.alias or from_table.name).upper()
                mask = self._eval_exists_where(sel.where_clause, df, inner_alias)
                df = df[mask]
            col_map = {c.upper(): c for c in df.columns}
            first = sel.columns[0] if sel.columns else None
            expr = first.expr if isinstance(first, SelectColumnNode) else first
            if isinstance(expr, VariableNode) and expr.name != "*":
                bare = expr.name.upper().split(".")[-1]
                actual = col_map.get(bare, df.columns[0] if len(df.columns) else None)
            else:
                actual = df.columns[0] if len(df.columns) else None
            if actual is None:
                return []
            return df[actual].dropna().tolist()
        except (KeyError, Exception):
            return []

    def _eval_scalar_subquery(self, node: ScalarSubqueryNode) -> Any:
        """Evaluate scalar subquery: (SELECT col FROM ... WHERE ...) — returns single value."""
        if self._session is None:
            return None
        from saslite.ast.sql import SelectNode, FromTableNode, SelectColumnNode
        sel = node.select_node
        if not isinstance(sel, SelectNode):
            return None
        try:
            from_table = sel.from_clause[0] if sel.from_clause else None
            if not isinstance(from_table, FromTableNode):
                return None
            ds = self._session.get_dataset(from_table.libref, from_table.name)
            df = ds.data.copy()
            if sel.where_clause:
                inner_alias = (from_table.alias or from_table.name).upper()
                mask = self._eval_exists_where(sel.where_clause, df, inner_alias)
                df = df[mask]
            if len(df) == 0:
                return None

            # Evaluate the first SELECT column (aggregate or plain column)
            first = sel.columns[0] if sel.columns else None
            expr = first.expr if isinstance(first, SelectColumnNode) else first
            col_map = {c.upper(): c for c in df.columns}

            if isinstance(expr, FunctionCallNode):
                fn_name = expr.name.upper()
                agg_map = {"MAX": "max", "MIN": "min", "SUM": "sum",
                           "AVG": "mean", "MEAN": "mean", "COUNT": "count",
                           "N": "count", "STD": "std", "MEDIAN": "median"}
                if fn_name in agg_map:
                    if expr.args and isinstance(expr.args[0], VariableNode) \
                            and expr.args[0].name != "*":
                        arg_name = expr.args[0].name.upper().split(".")[-1]
                        actual = col_map.get(arg_name)
                        if actual is not None:
                            return getattr(df[actual], agg_map[fn_name])()
                    if fn_name in ("COUNT", "N"):
                        return len(df)
                    return None

            if isinstance(expr, VariableNode) and expr.name != "*":
                bare = expr.name.upper().split(".")[-1]
                actual = col_map.get(bare)
                if actual is not None:
                    return df.iloc[0][actual]

            col = df.columns[0]
            return df.iloc[0][col]
        except (KeyError, Exception):
            return None
