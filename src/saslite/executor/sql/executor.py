"""PROC SQL executor — executes SQL statements against session datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from saslite.ast.sql import (
    ProcSqlNode, SelectNode, SelectColumnNode, FromTableNode, JoinNode,
    CreateTableNode, InsertNode, UpdateSqlNode, DeleteSqlNode,
    OrderItemNode, SetOperationNode,
)
from saslite.ast.expressions import (
    VariableNode, FunctionCallNode, LiteralNode, BinaryOpNode,
    CalculatedNode, ScalarSubqueryNode, ExistsNode, CaseNode,
    UnaryOpNode, BetweenNode, LikeNode, ArrayRefNode, WindowFuncNode,
)
from saslite.runtime.dataset import Dataset
from saslite.runtime.execution_result import StepResult
from saslite.executor.expression_eval import ExpressionEvaluator
from saslite.session.session import Session
from saslite.diagnostics.reporter import Reporter
from saslite.functions import build_default_registry
from saslite.runtime.types import is_missing


class SqlExecutor:
    """Executes PROC SQL statements."""

    def __init__(self, session: Session, reporter: Reporter) -> None:
        self.session = session
        self.reporter = reporter
        self._fn_registry = build_default_registry()

    def _compute_window_functions(self, columns: list, df: pd.DataFrame,
                                  col_map: dict[str, str]):
        """Materialize window-function SELECT columns as DataFrame columns.

        Supports ROW_NUMBER/RANK/DENSE_RANK/SUM/AVG/COUNT/LAG/LEAD OVER
        (PARTITION BY ... ORDER BY ...). Returns (df, rewritten_columns)
        where each WindowFuncNode is replaced by a VariableNode pointing
        at the computed column.
        """
        has_window = any(
            isinstance(c, SelectColumnNode) and isinstance(c.expr, WindowFuncNode)
            for c in columns
        )
        if not has_window:
            return df, columns

        df = df.copy()
        new_columns: list = []
        wf_idx = 0
        global_sort_applied = False
        for col in columns:
            if not (isinstance(col, SelectColumnNode) and isinstance(col.expr, WindowFuncNode)):
                new_columns.append(col)
                continue

            wf = col.expr
            fn = wf.func_name.upper()

            temp_col = f"__wf_{wf_idx}__"
            wf_idx += 1

            part_cols = [col_map.get(p.upper()) or self._find_column(df, p)
                         for p in wf.partition_by]
            part_cols = [p for p in part_cols if p]
            order_cols = []
            order_asc = []
            for name, asc in wf.order_by:
                actual = col_map.get(name.upper()) or self._find_column(df, name)
                if actual:
                    order_cols.append(actual)
                    order_asc.append(asc)

            # Establish ordering for ordered functions
            work = df.reset_index(drop=True)
            if order_cols and not global_sort_applied:
                # Apply the sort to the dataframe itself
                df = df.sort_values(
                    by=order_cols, ascending=order_asc, kind="mergesort",
                ).reset_index(drop=True)
                work = df.copy()
                global_sort_applied = True
                sorted_idx = work.index
            elif order_cols:
                sorted_idx = work.sort_values(
                    by=order_cols, ascending=order_asc, kind="mergesort",
                ).index
            else:
                sorted_idx = work.index

            ordered = work.loc[sorted_idx]

            if fn in ("ROW_NUMBER", "RANK", "DENSE_RANK"):
                if part_cols:
                    grouped = ordered.groupby(part_cols, sort=False, dropna=False)
                    if fn == "ROW_NUMBER":
                        vals = grouped.cumcount() + 1
                    else:
                        if order_cols:
                            method = "min" if fn == "RANK" else "dense"
                            vals = grouped[order_cols[0]].rank(
                                method=method, ascending=order_asc[0]).astype("Int64")
                        else:
                            vals = grouped.cumcount() + 1
                else:
                    if fn == "ROW_NUMBER" or not order_cols:
                        vals = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
                    else:
                        method = "min" if fn == "RANK" else "dense"
                        vals = ordered[order_cols[0]].rank(
                            method=method, ascending=order_asc[0]).astype("Int64")
                result = pd.Series(vals, index=ordered.index)
                if not global_sort_applied:
                    result = result.reindex(work.index)

            elif fn in ("SUM", "AVG", "COUNT", "MIN", "MAX"):
                # Aggregate window functions
                if not wf.args:
                    new_columns.append(col)
                    continue
                # Get column name from first argument
                arg_col_name = self._expr_to_column_name(wf.args[0])
                if not arg_col_name:
                    new_columns.append(col)
                    continue
                actual_col = col_map.get(arg_col_name.upper())
                if not actual_col:
                    actual_col = self._find_column(df, arg_col_name)
                if not actual_col or actual_col not in df.columns:
                    new_columns.append(col)
                    continue

                if part_cols:
                    if fn == "SUM":
                        result = df.groupby(part_cols, sort=False)[actual_col].transform("sum")
                    elif fn == "AVG":
                        result = df.groupby(part_cols, sort=False)[actual_col].transform("mean")
                    elif fn == "COUNT":
                        result = df.groupby(part_cols, sort=False)[actual_col].transform("count")
                    elif fn == "MIN":
                        result = df.groupby(part_cols, sort=False)[actual_col].transform("min")
                    elif fn == "MAX":
                        result = df.groupby(part_cols, sort=False)[actual_col].transform("max")
                else:
                    col_series = df[actual_col]
                    if fn == "SUM":
                        result = pd.Series([col_series.sum()] * len(df), index=df.index)
                    elif fn == "AVG":
                        result = pd.Series([col_series.mean()] * len(df), index=df.index)
                    elif fn == "COUNT":
                        result = pd.Series([col_series.count()] * len(df), index=df.index)
                    elif fn == "MIN":
                        result = pd.Series([col_series.min()] * len(df), index=df.index)
                    elif fn == "MAX":
                        result = pd.Series([col_series.max()] * len(df), index=df.index)

            elif fn in ("LAG", "LEAD"):
                # LAG/LEAD functions
                if not wf.args:
                    new_columns.append(col)
                    continue
                # Get column name from first argument
                arg_col_name = self._expr_to_column_name(wf.args[0])
                if not arg_col_name:
                    new_columns.append(col)
                    continue
                actual_col = col_map.get(arg_col_name.upper())
                if not actual_col:
                    actual_col = self._find_column(df, arg_col_name)
                if not actual_col or actual_col not in df.columns:
                    new_columns.append(col)
                    continue

                offset = 1
                if len(wf.args) > 1:
                    # Try to get offset value
                    if isinstance(wf.args[1], LiteralNode):
                        offset = int(wf.args[1].value)

                if part_cols:
                    grouped = ordered.groupby(part_cols, sort=False, dropna=False)
                    if fn == "LAG":
                        vals = grouped[actual_col].shift(offset)
                    else:
                        vals = grouped[actual_col].shift(-offset)
                    result = pd.Series(vals, index=ordered.index)
                    if not global_sort_applied:
                        result = result.reindex(work.index)
                else:
                    if fn == "LAG":
                        vals = ordered[actual_col].shift(offset)
                    else:
                        vals = ordered[actual_col].shift(-offset)
                    result = pd.Series(vals, index=ordered.index)
                    if not global_sort_applied:
                        result = result.reindex(work.index)
            else:
                new_columns.append(col)
                continue

            df[temp_col] = result.to_numpy()
            try:
                df[temp_col] = df[temp_col].astype(int)
            except (ValueError, TypeError):
                pass

            alias = col.alias or fn
            new_columns.append(SelectColumnNode(
                expr=VariableNode(name=temp_col), alias=alias,
                col_length=col.col_length, col_format=col.col_format,
                col_label=col.col_label,
            ))
            col_map[temp_col.upper()] = temp_col

        return df, new_columns

    def run(self, step: ProcSqlNode) -> StepResult:
        """Execute all SQL statements in a PROC SQL block."""
        combined = StepResult(success=True)
        try:
            for stmt in step.statements:
                if isinstance(stmt, SelectNode):
                    result = self._execute_select(stmt)
                elif isinstance(stmt, SetOperationNode):
                    result = self._execute_set_op(stmt)
                elif isinstance(stmt, CreateTableNode):
                    result = self._execute_create_table(stmt)
                elif isinstance(stmt, InsertNode):
                    result = self._execute_insert(stmt)
                elif isinstance(stmt, UpdateSqlNode):
                    result = self._execute_update(stmt)
                elif isinstance(stmt, DeleteSqlNode):
                    result = self._execute_delete(stmt)
                else:
                    continue

                combined.output_messages.extend(result.output_messages)
                combined.notes.extend(result.notes)
                combined.warnings.extend(result.warnings)
                combined.rows_affected += result.rows_affected
                combined.dataset_name = result.dataset_name or combined.dataset_name

                if not result.success:
                    combined.success = False
                    combined.error = result.error
                    return combined

            return combined
        except Exception as e:
            return StepResult(success=False, error=str(e))

    def _execute_select(self, sel: SelectNode, return_df: bool = False):
        """Execute a SELECT statement. If return_df=True, returns (StepResult, DataFrame)."""
        import io
        from saslite.functions.date_funcs import _reset_datetime_cache
        _reset_datetime_cache()

        # Get source table(s)
        if not sel.from_clause:
            r = StepResult(success=True)
            return (r, pd.DataFrame()) if return_df else r

        # Build base dataframe from first table
        from_table = sel.from_clause[0] if sel.from_clause else None
        if not isinstance(from_table, FromTableNode):
            r = StepResult(success=True)
            return (r, pd.DataFrame()) if return_df else r

        df = self._load_table(from_table)

        # Build alias-to-columns map and column name resolution
        join_nodes = [item for item in sel.from_clause if isinstance(item, JoinNode)]
        has_joins = bool(join_nodes)

        # Prefix columns with table alias/name if there are JOINs
        if has_joins:
            table_alias = from_table.alias or from_table.name
            df = df.add_prefix(f"{table_alias}.")
            for jn in join_nodes:
                df = self._execute_join(df, jn, left_alias=table_alias)

        # Build column resolution map: upper_name -> actual_col_name
        col_map = self._build_col_map(df, has_joins)

        # ── Pre-compute EXISTS / scalar subqueries as DataFrame columns ──
        # Collect all subqueries from SELECT list and WHERE clause
        subq_map: dict[int, str] = {}  # id(node) -> temp_col_name
        subq_nodes: list[tuple[Any, str]] = []  # (node, temp_col_name)
        subq_idx = 0

        def _collect_subqueries(expr: Any) -> None:
            nonlocal subq_idx
            if expr is None or isinstance(expr, (LiteralNode, VariableNode)):
                return
            if isinstance(expr, (ExistsNode, ScalarSubqueryNode)):
                nid = id(expr)
                if nid not in subq_map:
                    name = f"__subq_{subq_idx}__"
                    subq_idx += 1
                    subq_map[nid] = name
                    subq_nodes.append((expr, name))
                return
            if isinstance(expr, BinaryOpNode):
                _collect_subqueries(expr.left)
                _collect_subqueries(expr.right)
            elif isinstance(expr, UnaryOpNode):
                _collect_subqueries(expr.operand)
            elif isinstance(expr, FunctionCallNode):
                if expr.name == "_IN_SUBQUERY_":
                    return  # evaluated per-row; keep its subquery intact
                for a in expr.args:
                    _collect_subqueries(a)
            elif isinstance(expr, CaseNode):
                for c, r in zip(expr.conditions, expr.results):
                    _collect_subqueries(c)
                    _collect_subqueries(r)
                if expr.else_result:
                    _collect_subqueries(expr.else_result)
            elif isinstance(expr, CalculatedNode):
                pass
            elif isinstance(expr, (BetweenNode, LikeNode)):
                _collect_subqueries(expr.expr)
                if isinstance(expr, BetweenNode):
                    _collect_subqueries(expr.low)
                    _collect_subqueries(expr.high)
                elif isinstance(expr, LikeNode):
                    _collect_subqueries(expr.pattern)
            elif isinstance(expr, ArrayRefNode):
                _collect_subqueries(expr.index)

        for col_node in sel.columns:
            if isinstance(col_node, SelectColumnNode):
                _collect_subqueries(col_node.expr)
        if sel.where_clause:
            cond = sel.where_clause if not hasattr(sel.where_clause, "condition") else sel.where_clause.condition
            _collect_subqueries(cond)

        # Batch-compute each subquery and add as temp column
        for node, temp_name in subq_nodes:
            vals = self._precompute_subquery(node, df, col_map)
            df[temp_name] = vals
            col_map[temp_name.upper()] = temp_name

        # Rewrite expressions: replace subquery nodes with VariableNode
        def _rewrite_subqueries(expr: Any) -> Any:
            if expr is None:
                return expr
            if isinstance(expr, (ExistsNode, ScalarSubqueryNode)):
                nid = id(expr)
                if nid in subq_map:
                    return VariableNode(name=subq_map[nid])
                return expr
            if isinstance(expr, BinaryOpNode):
                return BinaryOpNode(op=expr.op,
                    left=_rewrite_subqueries(expr.left),
                    right=_rewrite_subqueries(expr.right))
            if isinstance(expr, UnaryOpNode):
                return UnaryOpNode(op=expr.op, operand=_rewrite_subqueries(expr.operand))
            if isinstance(expr, FunctionCallNode):
                return FunctionCallNode(name=expr.name,
                    args=[_rewrite_subqueries(a) for a in expr.args])
            if isinstance(expr, CaseNode):
                return CaseNode(
                    conditions=[_rewrite_subqueries(c) for c in expr.conditions],
                    results=[_rewrite_subqueries(r) for r in expr.results],
                    else_result=_rewrite_subqueries(expr.else_result))
            if isinstance(expr, BetweenNode):
                return BetweenNode(
                    expr=_rewrite_subqueries(expr.expr),
                    low=_rewrite_subqueries(expr.low),
                    high=_rewrite_subqueries(expr.high),
                )
            if isinstance(expr, LikeNode):
                return LikeNode(
                    expr=_rewrite_subqueries(expr.expr),
                    pattern=_rewrite_subqueries(expr.pattern),
                    negated=expr.negated,
                )
            return expr

        # Rewrite SELECT columns and WHERE
        rewritten_cols = []
        for col_node in sel.columns:
            if isinstance(col_node, SelectColumnNode):
                rewritten_cols.append(SelectColumnNode(
                    expr=_rewrite_subqueries(col_node.expr),
                    alias=col_node.alias,
                    col_length=col_node.col_length,
                    col_format=col_node.col_format,
                    col_label=col_node.col_label,
                ))
            else:
                rewritten_cols.append(col_node)

        rewritten_where = None
        if sel.where_clause:
            cond = sel.where_clause if not hasattr(sel.where_clause, "condition") else sel.where_clause.condition
            rewritten_cond = _rewrite_subqueries(cond)
            from saslite.ast.expressions import UnaryOpNode as UON
            if isinstance(rewritten_cond, UON) and rewritten_cond.op == "NOT":
                rewritten_where = type(sel.where_clause)(condition=rewritten_cond)
            else:
                rewritten_where = type(sel.where_clause)(condition=rewritten_cond) if hasattr(sel.where_clause, 'condition') else rewritten_cond

        # ── Pre-compute CALCULATED columns referenced in WHERE ──
        # SAS allows WHERE to reference CALCULATED columns from the same SELECT.
        # We must evaluate those SELECT expressions first, add them as temp
        # columns, then rewrite CalculatedNode → VariableNode before WHERE eval.
        calc_map: dict[str, str] = {}  # UPPER(alias) -> temp_col_name
        calc_cols_to_eval: list[tuple[SelectColumnNode, str]] = []  # (col_node, temp_name)

        # Collect all CALCULATED references from the WHERE clause
        def _collect_calculated(expr: Any) -> set[str]:
            """Collect CALCULATED names referenced in an expression."""
            names: set[str] = set()
            if expr is None or isinstance(expr, (LiteralNode, VariableNode)):
                return names
            if isinstance(expr, CalculatedNode):
                names.add(expr.name.upper())
                return names
            if isinstance(expr, BinaryOpNode):
                names |= _collect_calculated(expr.left)
                names |= _collect_calculated(expr.right)
            elif isinstance(expr, UnaryOpNode):
                names |= _collect_calculated(expr.operand)
            elif isinstance(expr, FunctionCallNode):
                for a in expr.args:
                    names |= _collect_calculated(a)
            elif isinstance(expr, CaseNode):
                for c, r in zip(expr.conditions, expr.results):
                    names |= _collect_calculated(c)
                    names |= _collect_calculated(r)
                if expr.else_result:
                    names |= _collect_calculated(expr.else_result)
            elif isinstance(expr, (BetweenNode, LikeNode)):
                names |= _collect_calculated(expr.expr)
                if isinstance(expr, BetweenNode):
                    names |= _collect_calculated(expr.low)
                    names |= _collect_calculated(expr.high)
                elif isinstance(expr, LikeNode):
                    names |= _collect_calculated(expr.pattern)
            return names

        calc_names_in_where: set[str] = set()
        if rewritten_where:
            cond = rewritten_where if not hasattr(rewritten_where, "condition") else rewritten_where.condition
            calc_names_in_where = _collect_calculated(cond)

        # For each CALCULATED name, find the matching SELECT column and pre-compute it
        calc_temp_idx = 0
        for col_node in rewritten_cols:
            if not isinstance(col_node, SelectColumnNode):
                continue
            alias = col_node.alias or self._expr_to_column_name(col_node.expr) or ""
            if alias.upper() in calc_names_in_where:
                temp_name = f"__calc_{calc_temp_idx}__"
                calc_temp_idx += 1
                calc_map[alias.upper()] = temp_name
                calc_cols_to_eval.append((col_node, temp_name))

        # Evaluate the CALCULATED columns and add them to df
        for col_node, temp_name in calc_cols_to_eval:
            vals = self._eval_per_row(col_node.expr, df, col_map)
            df[temp_name] = vals
            # Update col_map so the temp column can be found
            col_map[temp_name.upper()] = temp_name

        # Rewrite CalculatedNode in WHERE → VariableNode pointing to temp column
        def _rewrite_calculated(expr: Any) -> Any:
            if expr is None:
                return expr
            if isinstance(expr, CalculatedNode):
                key = expr.name.upper()
                if key in calc_map:
                    return VariableNode(name=calc_map[key])
                return expr
            if isinstance(expr, BinaryOpNode):
                return BinaryOpNode(op=expr.op,
                    left=_rewrite_calculated(expr.left),
                    right=_rewrite_calculated(expr.right))
            if isinstance(expr, UnaryOpNode):
                return UnaryOpNode(op=expr.op, operand=_rewrite_calculated(expr.operand))
            if isinstance(expr, FunctionCallNode):
                return FunctionCallNode(name=expr.name,
                    args=[_rewrite_calculated(a) for a in expr.args])
            if isinstance(expr, CaseNode):
                return CaseNode(
                    conditions=[_rewrite_calculated(c) for c in expr.conditions],
                    results=[_rewrite_calculated(r) for r in expr.results],
                    else_result=_rewrite_calculated(expr.else_result))
            if isinstance(expr, BetweenNode):
                return BetweenNode(
                    expr=_rewrite_calculated(expr.expr),
                    low=_rewrite_calculated(expr.low),
                    high=_rewrite_calculated(expr.high),
                )
            if isinstance(expr, LikeNode):
                return LikeNode(
                    expr=_rewrite_calculated(expr.expr),
                    pattern=_rewrite_calculated(expr.pattern),
                    negated=expr.negated,
                )
            return expr

        if rewritten_where:
            cond = rewritten_where if not hasattr(rewritten_where, "condition") else rewritten_where.condition
            rewritten_cond = _rewrite_calculated(cond)
            from saslite.ast.expressions import UnaryOpNode as UON2
            if isinstance(rewritten_cond, UON2) and rewritten_cond.op == "NOT":
                rewritten_where = type(rewritten_where)(condition=rewritten_cond) if hasattr(rewritten_where, 'condition') else rewritten_cond
            else:
                rewritten_where = type(rewritten_where)(condition=rewritten_cond) if hasattr(rewritten_where, 'condition') else rewritten_cond

        # Apply WHERE (rewritten, no more subqueries or calculated refs)
        if rewritten_where:
            pre_len = len(df)
            mask = self._eval_where(rewritten_where, df, col_map=col_map)
            df = df[mask].reset_index(drop=True)

        # Check for GROUP BY
        if sel.group_by:
            # Use rewritten columns for grouped select
            sel_grouped = SelectNode(
                distinct=sel.distinct,
                columns=rewritten_cols,
                from_clause=sel.from_clause,
                where_clause=rewritten_where,
                group_by=sel.group_by,
                having_clause=sel.having_clause,
                order_by=sel.order_by,
            )
            return self._execute_select_grouped(sel_grouped, df, return_df, has_joins=has_joins)

        # Select columns — evaluate expressions (rewritten, no subqueries)
        df, rewritten_cols = self._compute_window_functions(rewritten_cols, df, col_map)
        df = self._apply_select_columns(rewritten_cols, df, col_map=col_map)

        # Drop temp subquery columns if they survived
        temp_cols = [c for c in df.columns if c.startswith("__subq_")]
        if temp_cols:
            df = df.drop(columns=temp_cols)

        # DISTINCT
        if sel.distinct:
            df = df.drop_duplicates()
            if not sel.order_by and len(df) > 1:
                df = df.sort_values(
                    by=list(df.columns),
                    ascending=[True] * len(df.columns),
                    kind="mergesort",
                    na_position="first",
                ).reset_index(drop=True)

        # ORDER BY
        if sel.order_by:
            sort_cols = []
            ascending = []
            for item in sel.order_by:
                if isinstance(item, OrderItemNode):
                    col_name = self._expr_to_column_name(item.expr)
                    if col_name:
                        actual = self._find_column(df, col_name)
                        if actual:
                            sort_cols.append(actual)
                            ascending.append(item.ascending)
            if sort_cols:
                df = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

        # Display results
        buf = io.StringIO()
        buf.write(f"\n{'=' * 60}\n")
        buf.write(f"  PROC SQL: {len(df)} rows selected\n")
        buf.write(f"{'=' * 60}\n\n")
        display = df.copy()
        display.index = range(1, len(display) + 1)
        display.index.name = "Obs"
        buf.write(display.to_string())
        buf.write("\n")
        output = buf.getvalue()
        self.reporter.log(output)

        # SELECT ... INTO :macrovar — store first-row values as macro vars
        if getattr(sel, "into_vars", None):
            for j, mv in enumerate(sel.into_vars):
                if j < len(df.columns) and len(df) > 0:
                    val = df.iloc[0, j]
                    if isinstance(val, float) and val == int(val):
                        sval = str(int(val))
                    else:
                        sval = str(val) if val is not None else ""
                    self.session.set_macro_var(mv, sval.strip())

        result = StepResult(
            success=True,
            dataset_name="PROC SQL",
            rows_affected=len(df),
            notes=[f"PROC SQL: {len(df)} rows selected."],
            output_messages=[output],
        )
        return (result, df) if return_df else result

    def _execute_set_op(self, node: SetOperationNode) -> StepResult:
        """Execute UNION/INTERSECT/EXCEPT set operations."""
        import io

        left_df = self._select_to_df(node.left)
        right_df = self._select_to_df(node.right)

        # Align columns by position (SAS convention)
        if list(left_df.columns) != list(right_df.columns):
            right_df.columns = left_df.columns[:len(right_df.columns)]

        op = node.op.upper()
        if op == "UNION":
            if node.all:
                result_df = pd.concat([left_df, right_df], ignore_index=True)
            else:
                result_df = pd.concat([left_df, right_df], ignore_index=True).drop_duplicates()
        elif op == "INTERSECT":
            merged = left_df.merge(right_df, how="inner")
            if not node.all:
                merged = merged.drop_duplicates()
            result_df = merged
        elif op == "EXCEPT":
            merged = left_df.merge(right_df, how="left", indicator=True)
            result_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
            if not node.all:
                result_df = result_df.drop_duplicates()
        else:
            return StepResult(success=False, error=f"Unknown set operation: {op}")

        result_df = result_df.reset_index(drop=True)

        # Display
        buf = io.StringIO()
        buf.write(f"\n{'=' * 60}\n")
        buf.write(f"  PROC SQL: {len(result_df)} rows selected\n")
        buf.write(f"{'=' * 60}\n\n")
        display = result_df.copy()
        display.index = range(1, len(display) + 1)
        display.index.name = "Obs"
        buf.write(display.to_string())
        buf.write("\n")
        output = buf.getvalue()
        self.reporter.log(output)

        return StepResult(
            success=True,
            rows_affected=len(result_df),
            notes=[f"PROC SQL: {len(result_df)} rows selected."],
            output_messages=[output],
        )

    def _select_to_df(self, sel: Any) -> pd.DataFrame:
        """Execute a SELECT and return the resulting DataFrame."""
        if isinstance(sel, SetOperationNode):
            left_df = self._select_to_df(sel.left)
            right_df = self._select_to_df(sel.right)
            if list(left_df.columns) != list(right_df.columns):
                right_df.columns = left_df.columns[:len(right_df.columns)]
            op = sel.op.upper()
            if op == "UNION":
                if sel.all:
                    return pd.concat([left_df, right_df], ignore_index=True)
                return pd.concat([left_df, right_df], ignore_index=True).drop_duplicates()
            elif op == "INTERSECT":
                return left_df.merge(right_df, how="inner").drop_duplicates()
            elif op == "EXCEPT":
                merged = left_df.merge(right_df, how="left", indicator=True)
                return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).drop_duplicates()
            return left_df

        if not isinstance(sel, SelectNode):
            return pd.DataFrame()

        result = self._execute_select(sel, return_df=True)
        if isinstance(result, tuple) and len(result) == 2:
            return result[1]
        return pd.DataFrame()

    def _execute_join(self, left_df: pd.DataFrame, join_node: JoinNode,
                      left_alias: str = "") -> pd.DataFrame:
        """Execute a JOIN clause and return merged dataframe."""
        if not isinstance(join_node.table, FromTableNode):
            return left_df

        right_ref = join_node.table
        # Use _load_table to apply dataset options (keep=, drop=, where=)
        right_df = self._load_table(right_ref)

        # Prefix right columns with table alias/name
        right_alias = right_ref.alias or right_ref.name
        right_df = right_df.add_prefix(f"{right_alias}.")

        how = "inner"
        if join_node.join_type == "LEFT":
            how = "left"
        elif join_node.join_type == "RIGHT":
            how = "right"
        elif join_node.join_type in ("FULL", "OUTER"):
            how = "outer"
        elif join_node.join_type == "CROSS":
            left_df = left_df.copy()
            right_df = right_df.copy()
            left_df["_key"] = 1
            right_df["_key"] = 1
            merged = left_df.merge(right_df, on="_key", how="outer", suffixes=("", "_r"))
            merged = merged.drop(columns=["_key"])
            return merged

        on_condition = join_node.on_condition
        if on_condition is None:
            return self._join_cross_fallback(left_df, right_df, how)

        # Try simple column-name join first
        left_col, right_col = self._extract_join_columns(on_condition)
        if left_col and right_col:
            # Left columns may have alias prefix (e.g., "t1.id"), try both
            left_actual = self._find_column(left_df, left_col)
            if left_actual is None and left_alias:
                left_actual = self._find_column(left_df, f"{left_alias}.{left_col}")
            # Right columns have alias prefix (e.g., "E0.subject"), try both
            right_actual = self._find_column(right_df, right_col)
            if right_actual is None:
                right_actual = self._find_column(right_df, f"{right_alias}.{right_col}")
            if left_actual and right_actual:
                return left_df.merge(
                    right_df, left_on=left_actual, right_on=right_actual,
                    how=how, suffixes=("", "_r"),
                )

        # Complex ON condition: extract AND-separated conditions,
        # compute merge keys via expression evaluation, then merge
        eq_pairs, non_eq_conditions = self._extract_eq_conditions(on_condition)
        if eq_pairs:
            try:
                return self._smart_join(
                    left_df, right_df, how, eq_pairs,
                    on_condition, right_alias, non_eq_conditions)
            except Exception:
                pass

        return self._join_cross_fallback(left_df, right_df, how)

    def _join_cross_fallback(self, left_df, right_df, how):
        """Cross join fallback with size guard."""

        n = len(left_df) * len(right_df)
        if n > 500_000:
            # Too large for cross join — just do left join with no matches
            if how == "left":
                null_right = pd.DataFrame(
                    {c: [None] * len(left_df) for c in right_df.columns})
                return pd.concat([left_df.reset_index(drop=True), null_right], axis=1)
            # For inner, return empty
            cols = list(left_df.columns) + list(right_df.columns)
            return pd.DataFrame(columns=cols)
        left_df = left_df.copy()
        right_df = right_df.copy()
        left_df["_key"] = 1
        right_df["_key"] = 1
        merged = left_df.merge(right_df, on="_key", how="outer" if how != "inner" else how, suffixes=("", "_r"))
        merged = merged.drop(columns=["_key"])
        return merged

    @staticmethod
    def _coerce_key_dtypes(left_vals, right_vals):
        """Ensure merge key columns have compatible dtypes."""
        import pandas as pd
        l_dtype = left_vals.dtype
        r_dtype = right_vals.dtype
        if l_dtype == r_dtype:
            return left_vals, right_vals
        # If either is object/string, convert both to string for safe merging
        if l_dtype == object or r_dtype == object:
            left_vals = left_vals.astype(str).replace("nan", "")
            right_vals = right_vals.astype(str).replace("nan", "")
            return left_vals, right_vals
        # Both numeric but different dtypes, convert to float
        from pandas.api.types import is_numeric_dtype
        if is_numeric_dtype(l_dtype) and is_numeric_dtype(r_dtype):
            return left_vals.astype(float), right_vals.astype(float)
        # Fallback: convert both to string
        left_vals = left_vals.astype(str).replace("nan", "")
        right_vals = right_vals.astype(str).replace("nan", "")
        return left_vals, right_vals

    def _smart_join(self, left_df, right_df, how, eq_pairs, on_condition, right_alias, non_eq_conditions=None):
        """Merge-based join for complex ON conditions.

        For each equality condition, compute keys on both tables using
        expression evaluation, then merge on those computed keys.
        Non-equality conditions are applied as post-merge filters.
        """
        if non_eq_conditions is None:
            non_eq_conditions = []

        # Build col_maps for both sides
        left_col_map = self._build_col_map(left_df, False)
        right_col_map = self._build_col_map(right_df, False)
        # Also map right alias-prefixed names: "E0.SUBJECT" → "E0.subject"
        for col in right_df.columns:
            right_col_map[col.upper()] = col

        # For each equality condition, classify sides and compute keys
        # Determine which side is left (references left_df columns)
        # and which is right (references right_df columns or right alias)
        left_col_names = {c.upper() for c in left_df.columns}
        right_col_names = {c.upper() for c in right_df.columns}

        merge_keys = []  # (left_expr, right_expr)
        filter_exprs = []  # conditions that are NOT equality or have literals on both sides

        # Add non-equality conditions from ON clause to filter_exprs
        filter_exprs.extend(non_eq_conditions)

        for left_expr, right_expr in eq_pairs:
            left_is_left = self._expr_uses_cols(left_expr, left_col_names, "")
            right_is_left = self._expr_uses_cols(right_expr, left_col_names, "")
            left_is_right = self._expr_uses_cols(right_expr, right_col_names, right_alias)
            right_is_right = self._expr_uses_cols(left_expr, right_col_names, right_alias)

            # A literal on one side means it's a filter, not a join key
            from saslite.ast.expressions import LiteralNode
            left_literal = isinstance(left_expr, LiteralNode)
            right_literal = isinstance(right_expr, LiteralNode)

            if left_literal and right_literal:
                continue  # constant comparison, skip

            if left_literal or right_literal:
                filter_exprs.append(BinaryOpNode(op="=", left=left_expr, right=right_expr))
                continue

            if left_is_left and not right_is_right:
                merge_keys.append((left_expr, right_expr))
            elif right_is_left and not left_is_right:
                merge_keys.append((right_expr, left_expr))
            elif left_is_left and left_is_right:
                # Ambiguous: both sides reference both tables (e.g., self-join on same column)
                merge_keys.append((left_expr, right_expr))
            else:
                filter_exprs.append(BinaryOpNode(op="=", left=left_expr, right=right_expr))

        if not merge_keys:
            raise ValueError("No usable merge keys")

        # Compute key columns on both sides (vectorized)
        left_key_cols = []  # list of Series
        right_key_cols = []
        for i, (l_expr, r_expr) in enumerate(merge_keys):
            l_vals = self._eval_vectorized(l_expr, left_df, left_col_map)
            r_vals = self._eval_vectorized(r_expr, right_df, right_col_map)
            l_name = f"__lk{i}__"
            r_name = f"__rk{i}__"
            left_key_cols.append((l_name, l_vals))
            right_key_cols.append((r_name, r_vals))

        # Build merge DataFrames with key columns
        left_merge = left_df.copy()
        right_merge = right_df.copy()
        merge_left_on = []
        merge_right_on = []
        for i, ((l_name, l_vals), (r_name, r_vals)) in enumerate(
                zip(left_key_cols, right_key_cols)):
            # Coerce to compatible dtypes to avoid merge errors
            l_vals, r_vals = self._coerce_key_dtypes(l_vals, r_vals)
            left_merge[l_name] = l_vals
            right_merge[r_name] = r_vals
            merge_left_on.append(l_name)
            merge_right_on.append(r_name)


        merged = left_merge.merge(
            right_merge,
            left_on=merge_left_on,
            right_on=merge_right_on,
            how=how,
            suffixes=("", "_r"),
        )
        if len(merged) > 0:
            k = merge_left_on[0]

        # Drop temp key columns
        temp_keys = [n for n, _ in left_key_cols] + [n for n, _ in right_key_cols]
        cols_to_drop = [c for c in temp_keys if c in merged.columns]
        if cols_to_drop:
            merged = merged.drop(columns=cols_to_drop)

        # Apply filter conditions (non-equality from ON clause)
        if filter_exprs:
            from saslite.ast.expressions import BinaryOpNode as BON
            combined = filter_exprs[0]
            for fe in filter_exprs[1:]:
                combined = BON(op="AND", left=combined, right=fe)

            merged_cm = self._build_col_map(merged, True)
            mask = self._eval_where(combined, merged, col_map=merged_cm)
            # For LEFT JOIN, also keep rows where right side is all-NULL
            if how == "left":
                right_cols = [c for c in merged.columns if c.startswith(f"{right_alias}.")]
                right_null = merged[right_cols].isna().all(axis=1)
                mask = mask | right_null
            merged = merged[mask].reset_index(drop=True)

        return merged

    def _extract_join_columns(self, condition: Any) -> tuple[str | None, str | None]:
        """Extract left and right column names from ON condition (e.g., a.id = b.id)."""
        if isinstance(condition, BinaryOpNode) and condition.op == "=":
            left = self._expr_to_column_name(condition.left)
            right = self._expr_to_column_name(condition.right)
            return left, right
        return None, None

    def _execute_select_grouped(self, sel: SelectNode, df: pd.DataFrame, return_df: bool = False, **kw):
        """Execute a SELECT with GROUP BY. If return_df=True, returns (StepResult, DataFrame)."""
        import io

        # Build per-row evaluator
        col_map = {c.upper(): c for c in df.columns}
        temp_col_idx = 0
        group_cols = []

        # Build a map of SELECT column aliases to their expressions
        select_alias_map = {}
        for col_node in sel.columns:
            if isinstance(col_node, SelectColumnNode) and col_node.alias:
                select_alias_map[col_node.alias.upper()] = col_node.expr

        for g in sel.group_by:
            col = self._expr_to_column_name(g)

            # Check if this is a CALCULATED reference
            if isinstance(g, CalculatedNode) and col:
                # Look up the expression from SELECT list
                select_expr = select_alias_map.get(col.upper())
                if select_expr:
                    # Evaluate the SELECT expression for each row
                    values = []
                    for i in range(len(df)):
                        row = df.iloc[i].to_dict()
                        def _get_var(n, _row=row, _cm=col_map):
                            actual_col = _cm.get(n.upper())
                            if actual_col:
                                return _row.get(actual_col)
                            if "." in n:
                                short = n.split(".", 1)[-1]
                                actual_col = _cm.get(short.upper())
                                if actual_col:
                                    return _row.get(actual_col)
                            return _row.get(n)
                        evaluator = ExpressionEvaluator(var_getter=_get_var)
                        for fn_name in self._fn_registry.names:
                            fn = self._fn_registry.get(fn_name)
                            if fn:
                                evaluator.register_function(fn_name, fn)
                        try:
                            values.append(evaluator.evaluate(select_expr))
                        except Exception:
                            values.append(None)
                    temp_name = f"__group_{temp_col_idx}__"
                    temp_col_idx += 1
                    df[temp_name] = values
                    group_cols.append(temp_name)
                    continue

            actual = self._find_column(df, col) if col else None
            if actual:
                group_cols.append(actual)
            else:
                # Complex expression (CASE WHEN, etc.) — evaluate per row
                values = []
                for i in range(len(df)):
                    row = df.iloc[i].to_dict()
                    def _get_var(n, _row=row, _cm=col_map):
                        actual_col = _cm.get(n.upper())
                        return _row.get(actual_col) if actual_col else _row.get(n)
                    evaluator = ExpressionEvaluator(var_getter=_get_var)
                    for fn_name in self._fn_registry.names:
                        fn = self._fn_registry.get(fn_name)
                        if fn:
                            evaluator.register_function(fn_name, fn)
                    try:
                        values.append(evaluator.evaluate(g))
                    except Exception:
                        values.append(None)
                temp_name = f"__group_{temp_col_idx}__"
                temp_col_idx += 1
                df[temp_name] = values
                group_cols.append(temp_name)

        if not group_cols:
            return StepResult(success=True)

        grouped = df.groupby(group_cols)

        # Resolve aggregate columns
        agg_funcs = {}
        star_count_aliases = []  # COUNT(*) aliases — computed via size()
        complex_agg_funcs = {}  # Aggregates with complex expressions (e.g., min(strip(col)))

        for col_node in sel.columns:
            if isinstance(col_node, SelectColumnNode):
                alias = col_node.alias or self._expr_to_column_name(col_node.expr) or ""
                # Strip table alias prefix from output column name
                if "." in alias:
                    alias = alias.split(".", 1)[-1]
                if isinstance(col_node.expr, FunctionCallNode):
                    fn_name = col_node.expr.name.upper()
                    inner_col = None
                    is_star = False
                    if col_node.expr.args:
                        arg0 = col_node.expr.args[0]
                        if isinstance(arg0, LiteralNode) and arg0.value == "*":
                            is_star = True
                        else:
                            inner_col = self._expr_to_column_name(arg0)

                    if is_star and fn_name == "COUNT":
                        star_count_aliases.append(alias or "COUNT")
                    elif inner_col:
                        # Simple case: min(col), max(col), etc.
                        actual_col = self._find_column(df, inner_col)
                        if actual_col:
                            pandas_agg = self._sas_agg_to_pandas(fn_name)
                            if pandas_agg:
                                agg_funcs[alias or actual_col] = (actual_col, pandas_agg)
                    elif col_node.expr.args:
                        # Complex case: min(strip(col)), max(datepart(dt)), etc.
                        # Need to evaluate the inner expression for each row, then aggregate
                        pandas_agg = self._sas_agg_to_pandas(fn_name)
                        if pandas_agg:
                            complex_agg_funcs[alias] = (col_node.expr, pandas_agg)

        if agg_funcs:
            agg_dict = {alias: pd.NamedAgg(column=col, aggfunc=fn) for alias, (col, fn) in agg_funcs.items()}
            result_df = grouped.agg(**agg_dict).reset_index()
        elif not star_count_aliases:
            result_df = grouped.size().reset_index(name="N")
        else:
            result_df = grouped.size().reset_index(name="__size__")

        # Rename groupby columns to SELECT aliases without table prefix
        n_gb = len(group_cols)
        gb_old_names = list(result_df.columns[:n_gb])
        gb_new_names = []
        for col_node in sel.columns:
            if not isinstance(col_node, SelectColumnNode):
                continue
            if isinstance(col_node.expr, FunctionCallNode):
                continue  # Skip aggregate columns
            alias = col_node.alias or self._expr_to_column_name(col_node.expr) or ""
            if "." in alias:
                alias = alias.split(".", 1)[-1]
            gb_new_names.append(alias)
        for old, new in zip(gb_old_names, gb_new_names):
            if old != new:
                result_df = result_df.rename(columns={old: new})

        # Rename temp group columns to SELECT aliases, drop unmatched
        temp_cols = [c for c in result_df.columns if c.startswith("__group_")]
        if temp_cols:
            # Build a map: expression_repr → alias
            expr_to_alias = {}
            for col_node in sel.columns:
                if isinstance(col_node, SelectColumnNode) and col_node.alias:
                    expr_to_alias[self._expr_repr(col_node.expr)] = col_node.alias
            for tc in temp_cols:
                # Find the original expression for this temp column
                idx = int(tc.replace("__group_", "").replace("__", ""))
                if idx < len(sel.group_by):
                    key = self._expr_repr(sel.group_by[idx])
                    if key in expr_to_alias:
                        result_df = result_df.rename(columns={tc: expr_to_alias[key]})
                        continue
                # Unmatched temp column — drop it
                result_df = result_df.drop(columns=[tc])

        # Add COUNT(*) columns via size()
        if star_count_aliases:
            if "__size__" in result_df.columns:
                for alias in star_count_aliases:
                    result_df[alias] = result_df["__size__"].values
                result_df = result_df.drop(columns=["__size__"])
            else:
                sizes = grouped.size().reset_index(name="__size__")
                for alias in star_count_aliases:
                    result_df[alias] = sizes["__size__"].values

        # Handle complex aggregate functions (e.g., min(strip(col)))
        if complex_agg_funcs:
            for alias, (func_node, pandas_agg_name) in complex_agg_funcs.items():
                # For each group, evaluate the inner expression and then aggregate
                agg_results = []
                for group_key, group_df in grouped:
                    # Evaluate the inner expression (e.g., strip(instancename)) for each row
                    inner_values = []
                    for i in range(len(group_df)):
                        row = group_df.iloc[i].to_dict()
                        def _get_var(n, _row=row, _cm=col_map):
                            actual_col = _cm.get(n.upper())
                            return _row.get(actual_col) if actual_col else _row.get(n)
                        evaluator = ExpressionEvaluator(var_getter=_get_var)
                        for fn_name in self._fn_registry.names:
                            fn = self._fn_registry.get(fn_name)
                            if fn:
                                evaluator.register_function(fn_name, fn)
                        try:
                            # Evaluate the inner expression (the first argument of the aggregate function)
                            inner_expr = func_node.args[0] if func_node.args else None
                            if inner_expr:
                                val = evaluator.evaluate(inner_expr)
                                inner_values.append(val)
                        except Exception:
                            inner_values.append(None)

                    # Apply the aggregate function to the evaluated values
                    if inner_values:
                        # Filter out None/NaN values for aggregation
                        valid_values = [v for v in inner_values if v is not None and (not isinstance(v, float) or not pd.isna(v))]
                        if valid_values:
                            # Use pandas Series aggregation method
                            series = pd.Series(valid_values)
                            agg_result = getattr(series, pandas_agg_name)()
                        else:
                            agg_result = None
                    else:
                        agg_result = None
                    agg_results.append(agg_result)

                # Add the aggregated column to result_df
                result_df[alias] = agg_results

        # Handle non-aggregate columns (literals, pass-through)
        for col_node in sel.columns:
            if not isinstance(col_node, SelectColumnNode):
                continue
            alias = col_node.alias or self._expr_to_column_name(col_node.expr) or ""
            if not alias:
                continue
            # Skip if already in result (from aggregate or GROUP BY)
            if self._find_column(result_df, alias):
                continue
            # Literal value — add constant column
            if isinstance(col_node.expr, LiteralNode):
                result_df[alias] = col_node.expr.value

        # Apply HAVING
        if sel.having_clause:
            mask = self._eval_having(sel.having_clause, result_df, agg_funcs, star_count_aliases)
            result_df = result_df[mask].reset_index(drop=True)

        # ORDER BY
        if sel.order_by:
            sort_cols = []
            ascending = []
            for item in sel.order_by:
                col = self._expr_to_column_name(item.expr)
                if col:
                    actual = self._find_column(result_df, col)
                    if actual:
                        sort_cols.append(actual)
                        ascending.append(item.ascending)
            if sort_cols:
                result_df = result_df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

        # Display results
        buf = io.StringIO()
        buf.write(f"\n{'=' * 60}\n")
        buf.write(f"  PROC SQL: {len(result_df)} rows selected (grouped)\n")
        buf.write(f"{'=' * 60}\n\n")
        display = result_df.copy()
        display.index = range(1, len(display) + 1)
        display.index.name = "Obs"
        buf.write(display.to_string())
        buf.write("\n")
        output = buf.getvalue()
        self.reporter.log(output)

        result = StepResult(
            success=True,
            rows_affected=len(result_df),
            notes=[f"PROC SQL: {len(result_df)} rows selected (grouped)."],
            output_messages=[output],
        )
        return (result, result_df) if return_df else result

    def _execute_create_table(self, node: CreateTableNode) -> StepResult:
        """CREATE TABLE AS SELECT."""
        if node.select and isinstance(node.select, SelectNode):
            # Execute the select and get the result DataFrame
            result, df = self._execute_select(node.select, return_df=True)
            out_ds = Dataset.from_dataframe(df, name=node.name, libref=node.libref)
            # Apply column attributes from SELECT
            self._apply_col_attrs(out_ds, node.select.columns)
            self.session.put_dataset(node.libref, node.name, out_ds)
            return StepResult(
                success=True,
                dataset_name=f"{node.libref.upper()}.{node.name.upper()}",
                rows_affected=len(df),
                notes=[f"Table {node.libref.upper()}.{node.name.upper()} created with {len(df)} rows."],
            )
        return StepResult(success=True)

    def _execute_insert(self, node: InsertNode) -> StepResult:
        """INSERT INTO ... VALUES(...) or INSERT INTO ... SELECT."""
        try:
            ds = self.session.get_dataset(node.libref, node.name)
        except KeyError:
            return StepResult(success=False, error=f"Table {node.libref}.{node.name} not found")

        # Build row from values
        if node.values:
            row_values = [self._eval_expr(v) for v in node.values]
            cols = node.columns if node.columns else list(ds.data.columns)
            row = dict(zip(cols, row_values))
            new_df = pd.concat([ds.data, pd.DataFrame([row])], ignore_index=True)
            out_ds = Dataset.from_dataframe(new_df, name=node.name, libref=node.libref)
            self.session.put_dataset(node.libref, node.name, out_ds)
            return StepResult(success=True, rows_affected=1)

        # INSERT INTO ... SELECT
        if node.select and isinstance(node.select, SelectNode):
            from_table = node.select.from_clause[0] if node.select.from_clause else None
            if isinstance(from_table, FromTableNode):
                src_ds = self.session.get_dataset(from_table.libref, from_table.name)
                src_df = src_ds.data.copy()
                if node.select.where_clause:
                    mask = self._eval_where(node.select.where_clause, src_df)
                    src_df = src_df[mask]
                result_cols = self._resolve_select_columns(node.select.columns, src_df)
                if result_cols is not None:
                    src_df = src_df[result_cols]
                # Match columns to target table
                target_cols = list(ds.data.columns)
                for col in target_cols:
                    if col not in src_df.columns:
                        src_df[col] = None
                src_df = src_df[target_cols] if target_cols else src_df
                new_df = pd.concat([ds.data, src_df], ignore_index=True)
                out_ds = Dataset.from_dataframe(new_df, name=node.name, libref=node.libref)
                self.session.put_dataset(node.libref, node.name, out_ds)
                return StepResult(success=True, rows_affected=len(src_df))

        return StepResult(success=True)

    def _execute_update(self, node: UpdateSqlNode) -> StepResult:
        """UPDATE ... SET ... WHERE."""
        try:
            ds = self.session.get_dataset(node.libref, node.name)
        except KeyError:
            return StepResult(success=False, error=f"Table {node.libref}.{node.name} not found")

        df = ds.data.copy()

        # Apply WHERE to get affected rows
        if node.where_clause:
            mask = self._eval_where(node.where_clause, df)
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        # Apply assignments
        col_map = {c.upper(): c for c in df.columns}
        target_indexes = list(df[mask].index)
        for assign in node.assignments:
            if hasattr(assign, "target") and hasattr(assign, "expr"):
                actual_target = col_map.get(assign.target.upper())
                if not actual_target:
                    continue
                values = []
                for idx in target_indexes:
                    row = df.loc[idx].to_dict()
                    values.append(self._eval_expr(assign.expr, row, col_map))
                if pd.api.types.is_numeric_dtype(df[actual_target]):
                    df[actual_target] = df[actual_target].astype("float64")
                df.loc[target_indexes, actual_target] = pd.Series(values, index=target_indexes)

        out_ds = Dataset.from_dataframe(df, name=node.name, libref=node.libref)
        self.session.put_dataset(node.libref, node.name, out_ds)
        return StepResult(success=True, rows_affected=int(mask.sum()))

    def _execute_delete(self, node: DeleteSqlNode) -> StepResult:
        """DELETE FROM ... WHERE."""
        try:
            ds = self.session.get_dataset(node.libref, node.name)
        except KeyError:
            return StepResult(success=False, error=f"Table {node.libref}.{node.name} not found")

        df = ds.data.copy()
        if node.where_clause:
            mask = self._eval_where(node.where_clause, df)
            df = df[~mask]
        else:
            df = df.iloc[0:0]  # Delete all

        out_ds = Dataset.from_dataframe(df, name=node.name, libref=node.libref)
        self.session.put_dataset(node.libref, node.name, out_ds)
        return StepResult(success=True, rows_affected=len(ds.data) - len(df))

    def _eval_where(self, where_node: Any, df: pd.DataFrame,
                    col_map: dict[str, str] | None = None,
                    outer_row: dict[str, Any] | None = None) -> pd.Series:
        """Evaluate WHERE clause as a boolean mask."""
        from saslite.runtime.types import sas_bool
        from saslite.ast.expressions import ExistsNode, ScalarSubqueryNode

        condition = where_node if not hasattr(where_node, "condition") else where_node.condition
        if col_map is None:
            col_map = {c.upper(): c for c in df.columns}

        # Check if condition contains EXISTS/ScalarSubquery (needs per-row eval)
        if self._has_subquery(condition):
            return self._eval_where_perrow(condition, df, col_map, outer_row)

        # Vectorized path
        if outer_row is None:
            try:
                result = self._eval_vectorized(condition, df, col_map)
                mask = result.apply(lambda v: sas_bool(v))
                return mask.fillna(False)
            except Exception:
                pass

        # Fallback: per-row
        return self._eval_where_perrow(condition, df, col_map, outer_row)

    def _has_subquery(self, expr) -> bool:
        """Check if expression contains EXISTS or scalar subquery nodes."""
        from saslite.ast.expressions import (
            ExistsNode, ScalarSubqueryNode, BinaryOpNode, UnaryOpNode,
            FunctionCallNode, CaseNode, BetweenNode, LikeNode, LiteralNode,
        )
        if expr is None:
            return False
        if isinstance(expr, (ExistsNode, ScalarSubqueryNode)):
            return True
        if isinstance(expr, BinaryOpNode):
            return self._has_subquery(expr.left) or self._has_subquery(expr.right)
        if isinstance(expr, UnaryOpNode):
            return self._has_subquery(expr.operand)
        if isinstance(expr, FunctionCallNode):
            if expr.name == "_IN_SUBQUERY_":
                return True
            return any(self._has_subquery(a) for a in expr.args)
        if isinstance(expr, CaseNode):
            return any(self._has_subquery(c) for c in expr.conditions) or \
                   any(self._has_subquery(r) for r in expr.results) or \
                   self._has_subquery(expr.else_result)
        if isinstance(expr, BetweenNode):
            return self._has_subquery(expr.expr)
        if isinstance(expr, LikeNode):
            return self._has_subquery(expr.expr)
        return False

    def _eval_where_perrow(self, condition, df, col_map, outer_row=None):
        """Per-row WHERE evaluation fallback for correlated subqueries."""
        from saslite.runtime.types import sas_bool

        fn_list = [(name, self._fn_registry.get(name)) for name in self._fn_registry.names
                    if self._fn_registry.get(name) is not None]
        records = df.to_dict(orient="records")
        current_row = {}
        def _gv(n, _row=current_row, _cm=col_map):
            actual = _cm.get(n.upper())
            return _row.get(actual) if actual else _row.get(n)
        ev = ExpressionEvaluator(var_getter=_gv, session=self.session)
        for name, fn in fn_list:
            ev.register_function(name, fn)

        mask = []
        for rec in records:
            current_row.clear()
            if outer_row:
                current_row.update(outer_row)
            current_row.update(rec)
            try:
                mask.append(sas_bool(ev.evaluate(condition)))
            except Exception:
                mask.append(False)
        return pd.Series(mask, index=df.index)

    def _eval_having(self, having_node: Any, df: pd.DataFrame,
                     agg_columns: dict[str, tuple[str, str]],
                     star_count_aliases: list[str] | None = None) -> pd.Series:
        """Evaluate HAVING clause using pre-aggregated result columns.

        agg_columns maps alias -> (source_col, pandas_agg_func).
        When HAVING references an aggregate like COUNT(id), we resolve it
        to the result column instead of re-evaluating per raw value.
        """
        from saslite.ast.expressions import FunctionCallNode
        condition = having_node if not hasattr(having_node, "condition") else having_node.condition
        # Build a map: (fn_name, source_col) -> result_col_name
        fn_col_map: dict[tuple[str, str], str] = {}
        for alias, (src_col, agg_fn) in agg_columns.items():
            # Map back from pandas name to SAS name
            sas_name = {"count": "COUNT", "sum": "SUM", "mean": "MEAN",
                        "min": "MIN", "max": "MAX", "std": "STD",
                        "median": "MEDIAN"}.get(agg_fn, agg_fn.upper())
            fn_col_map[(sas_name, src_col.upper())] = alias

        # Add COUNT(*) to the map
        star_map: dict[str, str] = {}
        if star_count_aliases:
            for alias in star_count_aliases:
                star_map["COUNT"] = alias

        # Rewrite condition: replace FunctionCallNode with VariableNode
        # if the function maps to a result column
        rewritten = self._rewrite_agg_expr(condition, fn_col_map, star_map)

        # Now evaluate per-row using the result DataFrame
        mask = []
        from saslite.runtime.types import sas_bool
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            eval_row = ExpressionEvaluator(var_getter=lambda n, r=row: r.get(n))
            for name in self._fn_registry.names:
                fn = self._fn_registry.get(name)
                if fn:
                    eval_row.register_function(name, fn)
            try:
                result = eval_row.evaluate(rewritten)
                mask.append(sas_bool(result))
            except Exception:
                mask.append(False)
        return pd.Series(mask, index=df.index)

    def _rewrite_agg_expr(self, expr: Any, fn_col_map: dict[tuple[str, str], str],
                          star_map: dict[str, str] | None = None) -> Any:
        """Rewrite aggregate function calls to column references for HAVING."""
        from saslite.ast.expressions import FunctionCallNode, VariableNode, BinaryOpNode, UnaryOpNode, LiteralNode
        if isinstance(expr, FunctionCallNode):
            fn_name = expr.name.upper()
            if expr.args:
                # Handle COUNT(*) — LiteralNode with value "*"
                arg0 = expr.args[0]
                if isinstance(arg0, LiteralNode) and arg0.value == "*" and star_map and fn_name in star_map:
                    return VariableNode(name=star_map[fn_name])
                col_name = self._expr_to_column_name(arg0)
                if col_name:
                    key = (fn_name, col_name.upper())
                    if key in fn_col_map:
                        return VariableNode(name=fn_col_map[key])
            return expr
        if isinstance(expr, BinaryOpNode):
            return BinaryOpNode(
                op=expr.op,
                left=self._rewrite_agg_expr(expr.left, fn_col_map, star_map),
                right=self._rewrite_agg_expr(expr.right, fn_col_map, star_map),
            )
        if isinstance(expr, UnaryOpNode):
            return UnaryOpNode(
                op=expr.op,
                operand=self._rewrite_agg_expr(expr.operand, fn_col_map, star_map),
            )
        return expr

    def _resolve_select_columns(self, columns: list, df: pd.DataFrame) -> list[str] | None:
        """Resolve SELECT column list to actual column names."""
        if not columns:
            return None
        result = []
        for col in columns:
            if isinstance(col, SelectColumnNode):
                if isinstance(col.expr, VariableNode) and col.expr.name == "*":
                    return None  # SELECT *
                name = col.alias or self._expr_to_column_name(col.expr)
                if name:
                    # Case-insensitive column matching
                    actual = self._find_column(df, name)
                    if actual:
                        result.append(actual)
                    elif col.alias:
                        # Computed column — evaluate per-row
                        try:
                            col_map = {c.upper(): c for c in df.columns}
                            vals = []
                            for i in range(len(df)):
                                row = df.iloc[i].to_dict()
                                def _gv(n, _row=row, _cm=col_map):
                                    actual = _cm.get(n.upper())
                                    return _row.get(actual) if actual else _row.get(n)
                                ev = ExpressionEvaluator(var_getter=_gv)
                                for fn_name in self._fn_registry.names:
                                    fn = self._fn_registry.get(fn_name)
                                    if fn:
                                        ev.register_function(fn_name, fn)
                                try:
                                    vals.append(ev.evaluate(col.expr))
                                except Exception:
                                    vals.append(None)
                            df[col.alias] = vals
                            result.append(col.alias)
                        except Exception:
                            pass
        return result if result else None

    def _find_column(self, df: pd.DataFrame, name: str) -> str | None:
        """Find column name case-insensitively, handling alias prefix."""
        if not name:
            return None
        if name in df.columns:
            return name
        for col in df.columns:
            if col.upper() == name.upper():
                return col
        # Try without alias prefix
        if "." in name:
            short = name.split(".", 1)[-1]
            return self._find_column(df, short)
        return None

    @staticmethod
    def _series_is_missing(series: pd.Series) -> pd.Series:
        """Vectorized SAS missing-value test for SQL expressions.

        In SQL context, empty strings are treated as NULL (SAS semantics).
        """
        missing = series.isna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # In SQL, empty string is NULL
            missing = missing | series.map(
                lambda value: isinstance(value, str) and (is_missing(value) or value == "")
            ).fillna(False)
        return missing.astype(bool)

    def _expr_to_column_name(self, expr: Any) -> str | None:
        """Extract column name from expression."""
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, CalculatedNode):
            return expr.name
        return None

    @staticmethod
    def _expr_repr(expr: Any) -> str:
        """Create a canonical string representation of an expression for comparison."""
        from saslite.ast.expressions import (
            CaseNode, BinaryOpNode, UnaryOpNode, FunctionCallNode,
            LiteralNode, VariableNode, BetweenNode, LikeNode,
        )
        if isinstance(expr, CaseNode):
            parts = ["CASE"]
            for cond, res in zip(expr.conditions, expr.results):
                parts.append(f"WHEN {SqlExecutor._expr_repr(cond)} THEN {SqlExecutor._expr_repr(res)}")
            if expr.else_result is not None:
                parts.append(f"ELSE {SqlExecutor._expr_repr(expr.else_result)}")
            parts.append("END")
            return " ".join(parts)
        if isinstance(expr, BinaryOpNode):
            return f"({SqlExecutor._expr_repr(expr.left)} {expr.op} {SqlExecutor._expr_repr(expr.right)})"
        if isinstance(expr, UnaryOpNode):
            return f"({expr.op} {SqlExecutor._expr_repr(expr.operand)})"
        if isinstance(expr, FunctionCallNode):
            args = ",".join(SqlExecutor._expr_repr(a) for a in expr.args)
            return f"{expr.name}({args})"
        if isinstance(expr, LiteralNode):
            return repr(expr.value)
        if isinstance(expr, VariableNode):
            return expr.name
        return str(id(expr))

    def _eval_expr(
        self,
        expr: Any,
        row: dict[str, Any] | None = None,
        col_map: dict[str, str] | None = None,
    ) -> Any:
        """Evaluate an expression."""
        if row is None:
            row = {}
        if col_map is None:
            col_map = {}

        def _get_var(name: str) -> Any:
            actual = col_map.get(name.upper())
            return row.get(actual) if actual else row.get(name)

        evaluator = ExpressionEvaluator(var_getter=_get_var)
        for name in self._fn_registry.names:
            fn = self._fn_registry.get(name)
            if fn:
                evaluator.register_function(name, fn)
        return evaluator.evaluate(expr)

    @staticmethod
    def _sas_agg_to_pandas(sas_name: str) -> str | None:
        """Map SAS aggregate function names to pandas."""
        mapping = {
            "N": "count",
            "SUM": "sum",
            "MEAN": "mean",
            "AVG": "mean",
            "MIN": "min",
            "MAX": "max",
            "STD": "std",
            "MEDIAN": "median",
            "COUNT": "count",
        }
        return mapping.get(sas_name)

    # ── New helper methods ──────────────────────────

    def _load_table(self, ft: FromTableNode) -> pd.DataFrame:
        """Load a table and apply dataset options (keep=, drop=, where=)."""

        if ft.select is not None:
            df = self._select_to_df(ft.select).copy()
        else:
            ds = self.session.get_dataset(ft.libref, ft.name)
            df = ds.data.copy()


        for opt in ft.ds_options:
            if isinstance(opt, dict):
                if "KEEP" in opt:
                    keep_cols = [c.upper() for c in opt["KEEP"]]
                    actual = [c for c in df.columns if c.upper() in keep_cols]
                    df = df[actual] if actual else df
                elif "DROP" in opt:
                    drop_cols = [c.upper() for c in opt["DROP"]]
                    actual = [c for c in df.columns if c.upper() not in drop_cols]
                    df = df[actual] if actual else df
                elif "WHERE" in opt:
                    cond = opt["WHERE"]
                    mask = self._eval_where(cond, df)
                    df = df[mask].reset_index(drop=True)

        return df

    def _build_col_map(self, df: pd.DataFrame, has_joins: bool) -> dict[str, str]:
        """Build column resolution map: UPPER(name) -> actual_col_name.
        Also maps ALIAS.COLUMN -> actual for qualified references."""
        col_map: dict[str, str] = {}
        for col in df.columns:
            col_map[col.upper()] = col
            if "." in col:
                # Map both the full qualified name and the short name
                short = col.split(".", 1)[-1]
                # Only map short name if it's unambiguous
                if short.upper() not in col_map:
                    col_map[short.upper()] = col
        return col_map

    def _resolve_col(self, name: str, col_map: dict[str, str]) -> str | None:
        """Resolve a column name using the column map, stripping alias prefix."""
        actual = col_map.get(name.upper())
        if not actual and "." in name:
            short = name.split(".", 1)[-1]
            actual = col_map.get(short.upper())
        return actual

    def _eval_vectorized(self, expr: Any, df: pd.DataFrame,
                         col_map: dict[str, str]) -> pd.Series:
        """Evaluate an expression on an entire DataFrame, returning a Series.

        Uses pandas vectorized operations instead of per-row Python loops.
        Falls back to .apply() for functions that can't be vectorized.
        """
        from saslite.ast.expressions import (
            VariableNode, FunctionCallNode, LiteralNode, BinaryOpNode,
            UnaryOpNode, CaseNode, BetweenNode, LikeNode, CalculatedNode,
        )

        if expr is None:
            return pd.Series([None] * len(df), index=df.index)

        if isinstance(expr, LiteralNode):
            return pd.Series([expr.value] * len(df), index=df.index)

        if isinstance(expr, VariableNode):
            col = col_map.get(expr.name.upper())
            if col and col in df.columns:
                return df[col]
            # Try without alias
            short = expr.name.upper().split(".", 1)[-1] if "." in expr.name else None
            if short:
                col = col_map.get(short)
                if col and col in df.columns:
                    return df[col]
            return pd.Series([None] * len(df), index=df.index)

        if isinstance(expr, CalculatedNode):
            col = col_map.get(expr.name.upper())
            if col and col in df.columns:
                return df[col]
            return pd.Series([None] * len(df), index=df.index)

        if isinstance(expr, FunctionCallNode):
            fn_name = expr.name.upper()
            args = [self._eval_vectorized(a, df, col_map) for a in expr.args]

            # Built-in vectorized operations for common functions
            if fn_name == "IN" and len(args) >= 2:
                result = pd.Series([False] * len(df), index=df.index)
                for candidate in args[1:]:
                    result = result | (args[0] == candidate)
                return result
            if fn_name == "STRIP" and len(args) == 1:
                return args[0].astype(str).str.strip()
            if fn_name == "COMPRESS" and len(args) >= 2:
                chars = args[1].iloc[0] if len(args) > 1 else ""
                if isinstance(chars, str) and not args[0].apply(lambda x: isinstance(x, str) and any(c in chars for c in str(x))).any():
                    # Fast path: no chars to remove
                    return args[0]
                return args[0].apply(lambda x: self._fn_compress(x, chars))
            if fn_name == "DATEPART" and len(args) == 1:
                return args[0].apply(self._fn_datepart)
            if fn_name == "INPUT" and len(args) >= 2:
                return args[0]
            if fn_name == "PUT" and len(args) >= 2:
                return args[0].astype(str)
            if fn_name == "UPCASE" and len(args) == 1:
                return args[0].astype(str).str.upper()
            if fn_name == "LOWCASE" and len(args) == 1:
                return args[0].astype(str).str.lower()
            if fn_name == "MISSING" and len(args) == 1:
                return self._series_is_missing(args[0])
            if fn_name == "LENGTH" and len(args) == 1:
                return args[0].astype(str).str.len()
            if fn_name == "SUBSTR" and 2 <= len(args) <= 3:
                start = int(args[1].iloc[0]) - 1 if len(args) > 1 else 0
                length = int(args[2].iloc[0]) if len(args) > 2 else None
                if length is not None:
                    return args[0].astype(str).str[start:start+length]
                return args[0].astype(str).str[start:]

            # Generic fallback: .apply()
            fn = self._fn_registry.get(fn_name)
            if fn:
                if len(args) == 1:
                    return args[0].apply(fn)
                elif len(args) == 2:
                    # For functions with 2 args where second is constant
                    arg1_val = args[1].iloc[0] if len(args) > 1 else None
                    if args[1].nunique() <= 1:
                        return args[0].apply(lambda x: fn(x, arg1_val))
                    return pd.Series([
                        fn(args[0].iloc[i], args[1].iloc[i])
                        for i in range(len(df))
                    ], index=df.index)
                else:
                    return pd.Series([
                        fn(*(a.iloc[i] for a in args))
                        for i in range(len(df))
                    ], index=df.index)

            # Unknown function — return None
            return pd.Series([None] * len(df), index=df.index)

        if isinstance(expr, BinaryOpNode):
            op = expr.op.upper()

            # Handle IS NULL / IS NOT NULL (don't evaluate right side)
            if op == "IS NULL":
                left = self._eval_vectorized(expr.left, df, col_map)
                return self._series_is_missing(left)
            if op == "IS NOT NULL":
                left = self._eval_vectorized(expr.left, df, col_map)
                return ~self._series_is_missing(left)

            # For other operators, evaluate both sides
            left = self._eval_vectorized(expr.left, df, col_map)
            right = self._eval_vectorized(expr.right, df, col_map)

            if op == "=":
                return left == right
            if op in ("NE", "<>", "^="):
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "||":
                return left.astype(str) + right.astype(str)
            if op == "AND":
                return left & right
            if op == "OR":
                return left | right

        if isinstance(expr, UnaryOpNode):
            operand = self._eval_vectorized(expr.operand, df, col_map)
            if expr.op.upper() == "NOT":
                return ~operand
            if expr.op == "-":
                return -operand

        if isinstance(expr, CaseNode):
            result = pd.Series([None] * len(df), index=df.index)
            for cond, res in zip(expr.conditions, expr.results):
                mask = self._eval_vectorized(cond, df, col_map).astype(bool)
                vals = self._eval_vectorized(res, df, col_map)
                result = result.where(~mask, vals)
            if expr.else_result:
                else_vals = self._eval_vectorized(expr.else_result, df, col_map)
                still_none = result.isna()
                result = result.where(~still_none, else_vals)
            return result

        if isinstance(expr, LikeNode):
            pattern = self._eval_vectorized(expr.pattern, df, col_map)
            subject = self._eval_vectorized(expr.expr, df, col_map)
            from saslite.functions.char_funcs import like_match
            result = pd.Series(
                [
                    like_match(subject.iloc[i], pattern.iloc[i])
                    for i in range(len(df))
                ],
                index=df.index,
            )
            return ~result if expr.negated else result

        if isinstance(expr, BetweenNode):
            val = self._eval_vectorized(expr.expr, df, col_map)
            low = self._eval_vectorized(expr.low, df, col_map)
            high = self._eval_vectorized(expr.high, df, col_map)
            return (val >= low) & (val <= high)

        # Fallback
        return pd.Series([None] * len(df), index=df.index)

    @staticmethod
    def _fn_compress(s, chars="", modifiers=""):
        """Vectorized compress function."""
        from saslite.functions.char_funcs import _to_str
        s = _to_str(s)
        if not chars:
            return s.replace(" ", "")
        mod = modifiers.upper() if modifiers else ""
        if "K" in mod:
            return "".join(c for c in s if c in chars)
        for c in chars:
            s = s.replace(c, "")
        return s

    @staticmethod
    def _fn_datepart(dt):
        """Vectorized datepart function."""
        import math
        if dt is None or (isinstance(dt, float) and math.isnan(dt)):
            return float("nan")
        try:
            return float(int(float(dt) / 86400))
        except (ValueError, TypeError):
            return float("nan")

    def _apply_select_columns(self, columns: list, df: pd.DataFrame,
                               col_map: dict[str, str] | None = None) -> pd.DataFrame:
        """Evaluate SELECT columns, compute expressions, apply aliases."""
        if col_map is None:
            col_map = self._build_col_map(df, False)

        # Check for SELECT *
        for col in columns:
            if isinstance(col, SelectColumnNode) and isinstance(col.expr, VariableNode) and col.expr.name == "*":
                return df

        # Check if there are any aggregate functions without GROUP BY
        has_aggregates = False
        for col_node in columns:
            if isinstance(col_node, SelectColumnNode) and isinstance(col_node.expr, FunctionCallNode):
                fn_name = col_node.expr.name.upper()
                if fn_name in ("COUNT", "SUM", "AVG", "MEAN", "MIN", "MAX", "STD", "MEDIAN"):
                    has_aggregates = True
                    break

        # If there are aggregates without GROUP BY, compute them once for the entire dataset
        if has_aggregates:
            result_row = {}
            for col_node in columns:
                if not isinstance(col_node, SelectColumnNode):
                    continue
                # Get the column name from expression (may include table alias prefix)
                expr_col_name = self._expr_to_column_name(col_node.expr)
                # If no explicit alias, strip table alias prefix from column name
                if col_node.alias:
                    alias = col_node.alias
                elif expr_col_name:
                    # Strip table alias prefix: "E.FOLDERNAME" -> "FOLDERNAME"
                    alias = expr_col_name.split(".")[-1] if "." in expr_col_name else expr_col_name
                else:
                    alias = ""

                if isinstance(col_node.expr, FunctionCallNode):
                    fn_name = col_node.expr.name.upper()
                    if fn_name in ("COUNT", "SUM", "AVG", "MEAN", "MIN", "MAX", "STD", "MEDIAN"):
                        # Handle aggregate function
                        if col_node.expr.args:
                            arg0 = col_node.expr.args[0]
                            distinct_agg = (isinstance(arg0, FunctionCallNode)
                                            and arg0.name == "_DISTINCT_")
                            if distinct_agg:
                                arg0 = arg0.args[0]
                            if isinstance(arg0, LiteralNode) and arg0.value == "*":
                                # COUNT(*)
                                result_row[alias] = len(df)
                            else:
                                # Check if it's a simple column reference
                                col_name = self._expr_to_column_name(arg0)
                                if col_name:
                                    actual_col = col_map.get(col_name.upper())
                                    if actual_col and actual_col in df.columns:
                                        series = df[actual_col]
                                        if distinct_agg:
                                            series = series.dropna().drop_duplicates()
                                        pandas_agg = self._sas_agg_to_pandas(fn_name)
                                        if pandas_agg:
                                            result_row[alias] = getattr(series, pandas_agg)()
                                        else:
                                            result_row[alias] = None
                                    else:
                                        result_row[alias] = None
                                else:
                                    # Complex expression (e.g., CASE WHEN, function call)
                                    # Evaluate the expression for each row, then aggregate
                                    vals = self._eval_per_row(arg0, df, col_map)
                                    # Filter out None/NaN values
                                    valid_vals = [v for v in vals if v is not None and (not isinstance(v, float) or not pd.isna(v))]
                                    if valid_vals:
                                        series = pd.Series(valid_vals)
                                        pandas_agg = self._sas_agg_to_pandas(fn_name)
                                        if pandas_agg:
                                            result_row[alias] = getattr(series, pandas_agg)()
                                        else:
                                            result_row[alias] = None
                                    else:
                                        result_row[alias] = None
                        else:
                            result_row[alias] = None
                        continue

                # Non-aggregate expression or literal
                if isinstance(col_node.expr, LiteralNode):
                    result_row[alias] = col_node.expr.value
                else:
                    # Evaluate expression on first row (or use constant)
                    vals = self._eval_per_row(col_node.expr, df.head(1), col_map)
                    result_row[alias] = vals[0] if vals else None

            # Return a single-row DataFrame
            return pd.DataFrame([result_row])

        # Process each column (no aggregates)
        result_cols = []
        computed_cols: dict[str, list] = {}  # alias -> [values]
        for col_node in columns:
            if not isinstance(col_node, SelectColumnNode):
                continue
            # Get the column name from expression (may include table alias prefix like "E.FOLDERNAME")
            expr_col_name = self._expr_to_column_name(col_node.expr)
            # If no explicit alias, strip table alias prefix from column name
            if col_node.alias:
                alias = col_node.alias
            elif expr_col_name:
                # Strip table alias prefix: "E.FOLDERNAME" -> "FOLDERNAME"
                alias = expr_col_name.split(".")[-1] if "." in expr_col_name else expr_col_name
            else:
                alias = ""

            # Try to resolve as a simple column reference first
            col_name = expr_col_name
            if col_name:
                actual = col_map.get(col_name.upper())
                if not actual and "." in col_name:
                    short = col_name.split(".", 1)[-1]
                    actual = col_map.get(short.upper())
                if actual:
                    if alias and alias.upper() != actual.upper():
                        df[alias] = df[actual]
                        result_cols.append(alias)
                    else:
                        result_cols.append(actual)
                    continue
            # Complex expression — evaluate per row
            if alias:
                vals = self._eval_per_row(col_node.expr, df, col_map)
                df[alias] = vals
                col_map[alias.upper()] = alias
                result_cols.append(alias)

        if result_cols:
            # Only keep selected columns
            existing = [c for c in result_cols if c in df.columns]
            df = df[existing]
        return df

    def _eval_per_row(self, expr: Any, df: pd.DataFrame,
                       col_map: dict[str, str] | None = None,
                       calculated_getter: Any = None) -> list:
        """Evaluate an expression per row of a DataFrame, returning a list of values."""
        if col_map is None:
            col_map = self._build_col_map(df, False)

        # Pre-convert DataFrame to list of dicts once (avoids per-row iloc.to_dict)
        records = df.to_dict(orient="records")

        # Pre-register functions once, reuse evaluator per row
        current_row: dict[str, Any] = {}
        def _gv(n, _row=current_row, _cm=col_map):
            actual = _cm.get(n.upper())
            if actual:
                return _row.get(actual)
            if "." in n:
                short = n.split(".", 1)[-1]
                actual = _cm.get(short.upper())
                if actual:
                    return _row.get(actual)
            return _row.get(n)

        ev = ExpressionEvaluator(var_getter=_gv, session=self.session)
        for fn_name in self._fn_registry.names:
            fn = self._fn_registry.get(fn_name)
            if fn:
                ev.register_function(fn_name, fn)
        if calculated_getter:
            ev.set_calculated_getter(calculated_getter)

        vals = []
        for rec in records:
            current_row.clear()
            current_row.update(rec)
            try:
                vals.append(ev.evaluate(expr))
            except Exception:
                vals.append(None)
        return vals

    def _precompute_subquery(
        self, node: Any, df: pd.DataFrame, col_map: dict[str, str]
    ) -> list:
        """Pre-compute a subquery (EXISTS or scalar) using merge-based semi-join."""
        from saslite.ast.sql import SelectNode, FromTableNode

        sel = node.select_node if hasattr(node, "select_node") else None
        if not isinstance(sel, SelectNode):
            return [False] * len(df) if isinstance(node, ExistsNode) else [None] * len(df)

        from_table = sel.from_clause[0] if sel.from_clause else None
        if not isinstance(from_table, FromTableNode):
            return [False] * len(df) if isinstance(node, ExistsNode) else [None] * len(df)

        try:
            inner_df = self._load_table(from_table)
        except (KeyError, Exception):
            return [False] * len(df) if isinstance(node, ExistsNode) else [None] * len(df)

        if not sel.where_clause:
            if isinstance(node, ExistsNode):
                return [len(inner_df) > 0] * len(df)
            # Scalar subquery without WHERE — evaluate once (handles aggregates)
            ev = ExpressionEvaluator(session=self.session)
            for name in self._fn_registry.names:
                fn = self._fn_registry.get(name)
                if fn:
                    ev.register_function(name, fn)
            val = ev._eval_scalar_subquery(node)
            return [val] * len(df)

        # Try merge-based approach for both EXISTS and scalar subqueries
        try:
            return self._semi_join_exists(
                sel, df, col_map, inner_df, from_table, node)
        except Exception:
            pass

        # Fallback: per-row with cached inner table
        return self._precompute_subquery_perrow(
            node, df, col_map, sel, inner_df, from_table)

    def _semi_join_exists(
        self, sel, df, col_map, inner_df, from_table, node=None,
    ) -> list:
        """Merge-based EXISTS / scalar COUNT(*) evaluation.

        Extracts equality join conditions from the WHERE clause,
        evaluates key columns on both sides, then does a pandas merge.
        For EXISTS returns boolean mask, for COUNT(*) returns counts.
        """
        from saslite.ast.expressions import (
            BinaryOpNode, VariableNode, FunctionCallNode,
            UnaryOpNode, LikeNode, BetweenNode, CaseNode,
            LiteralNode, CalculatedNode, ExistsNode, ScalarSubqueryNode,
        )

        inner_alias = from_table.alias or from_table.name

        # Extract top-level AND conditions from WHERE
        cond = sel.where_clause if not hasattr(sel.where_clause, "condition") else sel.where_clause.condition
        eq_conditions, non_eq_conditions = self._extract_eq_conditions(cond)

        if not eq_conditions:
            raise ValueError("No equality conditions found for semi-join")
        if non_eq_conditions:
            # Mixed conditions need per-row evaluation
            raise ValueError("Non-equality conditions present; falling back")

        # Classify each condition as inner_eq outer_eq
        # Build col_map for inner table that maps qualified names (V2.SUBJID → SUBJID)
        inner_col_map = self._build_col_map(inner_df, False)
        for col in inner_df.columns:
            qualified = f"{inner_alias}.{col}".upper()
            if qualified not in inner_col_map:
                inner_col_map[qualified] = col

        inner_key_cols = []  # (inner_expr, outer_expr)
        inner_col_names = {c.upper() for c in inner_df.columns}
        outer_col_names = {c.upper() for c in df.columns}
        for left_expr, right_expr in eq_conditions:
            # Determine which side references inner table by alias prefix
            # This handles self-joins where both tables have same column names
            left_is_inner = self._expr_uses_alias(left_expr, inner_alias)
            right_is_inner = self._expr_uses_alias(right_expr, inner_alias)

            if left_is_inner and not right_is_inner:
                inner_expr, outer_expr = left_expr, right_expr
            elif right_is_inner and not left_is_inner:
                inner_expr, outer_expr = right_expr, left_expr
            else:
                # Both or neither reference inner — skip
                continue

            inner_key_cols.append((inner_expr, outer_expr))

        if not inner_key_cols:
            raise ValueError("No usable inner/outer key pairs")

        # Compute keys for all inner rows (vectorized)
        inner_keys = []
        for inner_expr, _ in inner_key_cols:
            vals = self._eval_vectorized(inner_expr, inner_df, inner_col_map)
            inner_keys.append(vals)

        # Compute keys for all outer rows (vectorized)
        outer_keys = []
        for _, outer_expr in inner_key_cols:
            vals = self._eval_vectorized(outer_expr, df, col_map)
            outer_keys.append(vals)

        # Build merge DataFrames
        n_inner = len(inner_df)
        n_outer = len(df)
        n_keys = len(inner_key_cols)

        # Normalize values for consistent comparison
        def _normalize(vals):
            """Normalize values so hash/merge works (e.g., NaT → None)."""
            return [
                None if v is None or (isinstance(v, float) and pd.isna(v))
                else v for v in vals
            ]

        if n_keys == 1:
            inner_merge_df = pd.DataFrame({
                "__key__": _normalize(inner_keys[0]),
                "__inner_idx__": range(n_inner),
            })
            outer_merge_df = pd.DataFrame({
                "__key__": _normalize(outer_keys[0]),
                "__outer_idx__": range(n_outer),
            })
        else:
            # Multi-key merge: use separate columns
            inner_merge_df = pd.DataFrame({"__inner_idx__": range(n_inner)})
            outer_merge_df = pd.DataFrame({"__outer_idx__": range(n_outer)})
            for k in range(n_keys):
                inner_merge_df[f"__key{k}__"] = _normalize(inner_keys[k])
                outer_merge_df[f"__key{k}__"] = _normalize(outer_keys[k])

        # Merge to find matching outer rows
        merge_keys = (
            [f"__key{k}__" for k in range(n_keys)]
            if n_keys > 1 else ["__key__"]
        )
        merged = inner_merge_df.merge(
            outer_merge_df,
            on=merge_keys,
            how="inner",
            suffixes=("", "_o"),
        )

        # Build result based on node type
        from saslite.ast.sql import SelectColumnNode

        if node and isinstance(node, ScalarSubqueryNode):
            # COUNT(*) — count matching inner rows per outer row
            is_count = False
            if sel.columns:
                col0 = sel.columns[0]
                if isinstance(col0, SelectColumnNode) and isinstance(col0.expr, FunctionCallNode):
                    if col0.expr.name.upper() == "COUNT":
                        is_count = True
            if is_count:
                counts = merged.groupby("__outer_idx__").size()
                return [int(counts.get(i, 0)) for i in range(n_outer)]
            # Other scalar: first matching value
            return [None] * n_outer

        # EXISTS: True for outer rows that had at least one match
        matched_outer = set(merged["__outer_idx__"].values)
        return [i in matched_outer for i in range(n_outer)]

    def _extract_eq_conditions(self, cond):
        """Extract top-level equality conditions from a WHERE clause (flattening ANDs).

        Returns: (eq_pairs, non_eq_conditions)
        - eq_pairs: list of (left_expr, right_expr) for equality conditions
        - non_eq_conditions: list of non-equality conditions (OR, etc.)
        """
        from saslite.ast.expressions import BinaryOpNode
        eq_pairs = []
        non_eq = []

        if isinstance(cond, BinaryOpNode) and cond.op == "AND":
            left_eq, left_non_eq = self._extract_eq_conditions(cond.left)
            right_eq, right_non_eq = self._extract_eq_conditions(cond.right)
            eq_pairs.extend(left_eq)
            eq_pairs.extend(right_eq)
            non_eq.extend(left_non_eq)
            non_eq.extend(right_non_eq)
        elif isinstance(cond, BinaryOpNode) and cond.op == "=":
            eq_pairs.append((cond.left, cond.right))
        else:
            # Non-equality condition (OR, <, >, etc.)
            non_eq.append(cond)

        return eq_pairs, non_eq

    def _expr_uses_alias(self, expr, alias: str) -> bool:
        """Check if any VariableNode in the expression references the given alias prefix.

        For self-joins: E2.SUBJECT uses alias "E2", PDOMAIN.SUBJECT uses alias "PDOMAIN".
        """
        from saslite.ast.expressions import (
            VariableNode, BinaryOpNode, UnaryOpNode, FunctionCallNode,
            CaseNode, LiteralNode, CalculatedNode, BetweenNode, LikeNode,
        )
        if expr is None:
            return False
        if isinstance(expr, (LiteralNode, CalculatedNode)):
            return False
        if isinstance(expr, VariableNode):
            return expr.name.upper().startswith(alias.upper() + ".")
        if isinstance(expr, BinaryOpNode):
            return self._expr_uses_alias(expr.left, alias) or \
                   self._expr_uses_alias(expr.right, alias)
        if isinstance(expr, UnaryOpNode):
            return self._expr_uses_alias(expr.operand, alias)
        if isinstance(expr, FunctionCallNode):
            return any(self._expr_uses_alias(a, alias) for a in expr.args)
        if isinstance(expr, CaseNode):
            return any(self._expr_uses_alias(c, alias) for c in expr.conditions) or \
                   any(self._expr_uses_alias(r, alias) for r in expr.results)
        if isinstance(expr, BetweenNode):
            return self._expr_uses_alias(expr.expr, alias)
        if isinstance(expr, LikeNode):
            return self._expr_uses_alias(expr.expr, alias)
        return False

    def _expr_uses_cols(self, expr, col_set: set, table_alias: str = "") -> bool:
        """Check if any VariableNode in the expression references columns in col_set.

        col_set = uppercase column names belonging to the table.
        table_alias = uppercase alias prefix (e.g., "V2").
        Matches: "SUBJID" directly, or "V2.SUBJID" by alias prefix.
        """
        from saslite.ast.expressions import (
            VariableNode, BinaryOpNode, UnaryOpNode, FunctionCallNode,
            CaseNode, LiteralNode, CalculatedNode, BetweenNode, LikeNode,
        )
        if expr is None:
            return False
        if isinstance(expr, LiteralNode):
            return False
        if isinstance(expr, CalculatedNode):
            return False
        if isinstance(expr, VariableNode):
            name_upper = expr.name.upper()
            # Direct column match
            if name_upper in col_set:
                return True
            # Alias-prefixed match: V2.SUBJID → strip prefix, check col_set
            if "." in name_upper:
                short = name_upper.split(".", 1)[-1]
                if short in col_set:
                    return True
            # Check if starts with table alias
            if table_alias and name_upper.startswith(table_alias + "."):
                return True
            return False
        if isinstance(expr, BinaryOpNode):
            return self._expr_uses_cols(expr.left, col_set, table_alias) or \
                   self._expr_uses_cols(expr.right, col_set, table_alias)
        if isinstance(expr, UnaryOpNode):
            return self._expr_uses_cols(expr.operand, col_set, table_alias)
        if isinstance(expr, FunctionCallNode):
            return any(self._expr_uses_cols(a, col_set, table_alias) for a in expr.args)
        if isinstance(expr, CaseNode):
            return any(self._expr_uses_cols(c, col_set, table_alias) for c in expr.conditions) or \
                   any(self._expr_uses_cols(r, col_set, table_alias) for r in expr.results)
        if isinstance(expr, BetweenNode):
            return self._expr_uses_cols(expr.expr, col_set, table_alias)
        if isinstance(expr, LikeNode):
            return self._expr_uses_cols(expr.expr, col_set, table_alias)
        return False

    def _precompute_subquery_perrow(
        self, node: Any, df: pd.DataFrame, col_map: dict[str, str],
        sel: Any, inner_df: pd.DataFrame, from_table: Any,
    ) -> list:
        """Per-row subquery evaluation with cached inner table (fallback)."""
        from saslite.runtime.types import sas_bool

        fn_list = [(name, self._fn_registry.get(name))
                   for name in self._fn_registry.names
                   if self._fn_registry.get(name) is not None]

        cache_key = (from_table.libref.upper(), from_table.name.upper())

        values = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()

            def _get_var(n, _row=row, _cm=col_map):
                actual = _cm.get(n.upper())
                return _row.get(actual) if actual else _row.get(n)

            ev = ExpressionEvaluator(var_getter=_get_var, session=self.session)
            for name, fn in fn_list:
                ev.register_function(name, fn)
            ev._exists_cache = {cache_key: inner_df}

            try:
                if isinstance(node, ExistsNode):
                    values.append(sas_bool(ev.evaluate(node)))
                elif isinstance(node, ScalarSubqueryNode):
                    values.append(ev.evaluate(node))
                else:
                    values.append(None)
            except Exception:
                if isinstance(node, ExistsNode):
                    values.append(False)
                else:
                    values.append(None)

        return values

    def _apply_col_attrs(self, ds: Dataset, columns: list) -> None:
        """Apply column attributes (LENGTH, FORMAT, LABEL) from SELECT list to dataset metadata."""
        for col_node in columns:
            if not isinstance(col_node, SelectColumnNode):
                continue
            alias = col_node.alias or self._expr_to_column_name(col_node.expr) or ""
            if not alias:
                continue
            key = alias.upper()
            if key in ds.metadata.variables:
                var = ds.metadata.variables[key]
                if col_node.col_length is not None:
                    var.length = col_node.col_length
                if col_node.col_format:
                    var.format = col_node.col_format.rstrip(".")
                if col_node.col_label:
                    var.label = col_node.col_label
