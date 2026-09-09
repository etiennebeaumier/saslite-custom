"""Additional PROC implementations: TRANSPOSE, UNIVARIATE, COMPARE, COPY,
FORMAT, TABULATE, REPORT."""

from __future__ import annotations

import io
import math
from typing import Any

import pandas as pd

from saslite.ast.proc import ProcNode, VarListNode, ByNode, ClassNode
from saslite.runtime.dataset import Dataset
from saslite.runtime.execution_result import StepResult
from saslite.session.session import Session
from saslite.diagnostics.reporter import Reporter


def _resolve_dataset(session: Session, name: Any) -> Dataset:
    """Get a dataset from `lib.name` or bare `name` (WORK default)."""
    name = str(name)
    if "." in name:
        libref, member = name.split(".", 1)
    else:
        libref, member = "WORK", name
    return session.get_dataset(libref, member)


def _split_name(name: Any) -> tuple[str, str]:
    name = str(name)
    if "." in name:
        parts = name.split(".", 1)
        return parts[0].upper(), parts[1].upper()
    return "WORK", name.upper()


def _col_map(df: pd.DataFrame) -> dict[str, str]:
    return {c.upper(): c for c in df.columns}


# ─── PROC TRANSPOSE ────────────────────────────────────


def handle_proc_transpose(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC TRANSPOSE — rotate dataset (rows ↔ columns)."""
    data_name = proc.options.get("DATA", "")
    out_name = proc.options.get("OUT", "")
    prefix = str(proc.options.get("PREFIX", "COL"))
    name_col = str(proc.options.get("NAME", "_NAME_")).upper()

    if not data_name:
        return StepResult(success=False, error="PROC TRANSPOSE requires DATA=")

    try:
        ds = _resolve_dataset(session, data_name)
    except KeyError:
        return StepResult(success=False, error=f"Dataset {data_name} not found")

    df = ds.data.copy()
    cmap = _col_map(df)

    var_names = [v.upper() for v in proc.options.get("_VAR", [])]
    id_names = [v.upper() for v in proc.options.get("_ID", [])]
    by_names = [v.upper() for v in proc.options.get("_BY", [])]

    # Default VAR: all numeric columns not used in BY/ID
    if not var_names:
        var_names = [c.upper() for c in df.columns
                     if pd.api.types.is_numeric_dtype(df[c])
                     and c.upper() not in by_names and c.upper() not in id_names]
    var_cols = [cmap[v] for v in var_names if v in cmap]
    if not var_cols:
        return StepResult(success=False, error="PROC TRANSPOSE: no variables to transpose")

    id_col = cmap.get(id_names[0]) if id_names else None
    by_cols = [cmap[b] for b in by_names if b in cmap]

    def _transpose_block(block: pd.DataFrame) -> pd.DataFrame:
        """Transpose one BY group: one output row per VAR column."""
        out_rows = []
        for vc in var_cols:
            row: dict[str, Any] = {name_col: vc}
            if id_col is not None:
                for _, src in block.iterrows():
                    col_label = str(src[id_col])
                    row[col_label.upper()] = src[vc]
            else:
                for j, (_, src) in enumerate(block.iterrows(), 1):
                    row[f"{prefix}{j}".upper()] = src[vc]
            out_rows.append(row)
        return pd.DataFrame(out_rows)

    if by_cols:
        parts = []
        for key, block in df.groupby(by_cols, sort=False):
            t = _transpose_block(block)
            if not isinstance(key, tuple):
                key = (key,)
            for bc, kv in zip(by_cols, key):
                t.insert(0, bc, kv)
            parts.append(t)
        result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        # Keep BY columns first, then _NAME_, then data columns
        ordered = by_cols + [name_col] + [c for c in result.columns
                                          if c not in by_cols and c != name_col]
        result = result[[c for c in ordered if c in result.columns]]
    else:
        result = _transpose_block(df)

    out_libref, out_member = _split_name(out_name) if out_name else _split_name(data_name)
    out_ds = Dataset.from_dataframe(result, name=out_member, libref=out_libref)
    session.put_dataset(out_libref, out_member, out_ds)

    return StepResult(
        success=True,
        dataset_name=f"{out_libref}.{out_member}",
        rows_affected=len(result),
        notes=[f"Dataset {out_libref}.{out_member} created with {len(result)} observations "
               f"and {len(result.columns)} variables."],
    )


# ─── PROC UNIVARIATE ───────────────────────────────────


def handle_proc_univariate(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC UNIVARIATE — detailed descriptive statistics."""
    data_name = proc.options.get("DATA", "")
    noprint = proc.options.get("NOPRINT", False)
    if not data_name:
        return StepResult(success=False, error="PROC UNIVARIATE requires DATA=")

    try:
        ds = _resolve_dataset(session, data_name)
    except KeyError:
        return StepResult(success=False, error=f"Dataset {data_name} not found")

    df = ds.data
    cmap = _col_map(df)

    var_names: list[str] = []
    by_names: list[str] = []
    for stmt in proc.statements:
        if isinstance(stmt, VarListNode) and not var_names:
            var_names = [v.upper() for v in stmt.variables]
        elif isinstance(stmt, ByNode):
            by_names = [v.upper() for v in stmt.variables]

    if not var_names:
        var_names = [c.upper() for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    var_cols = [cmap[v] for v in var_names if v in cmap]
    if not var_cols:
        return StepResult(success=False, error="No valid numeric variables found")

    by_cols = [cmap[b] for b in by_names if b in cmap]

    buf = io.StringIO()
    buf.write(f"\n{'=' * 60}\n")
    buf.write(f"  PROC UNIVARIATE: {ds.metadata.libref}.{ds.metadata.member_name}\n")
    buf.write(f"{'=' * 60}\n")

    def _write_stats(block: pd.DataFrame, header: str = "") -> None:
        if header:
            buf.write(f"\n  --- {header} ---\n")
        for vc in var_cols:
            s = pd.to_numeric(block[vc], errors="coerce").dropna()
            buf.write(f"\n  Variable: {vc}\n")
            buf.write(f"  {'-' * 50}\n")
            n = len(s)
            buf.write(f"  {'N':<22} {n}\n")
            buf.write(f"  {'N Missing':<22} {block[vc].isna().sum()}\n")
            if n == 0:
                continue
            mean = s.mean()
            std = s.std() if n > 1 else float("nan")
            buf.write(f"  {'Mean':<22} {mean:.6g}\n")
            buf.write(f"  {'Std Deviation':<22} {std:.6g}\n")
            buf.write(f"  {'Variance':<22} {(std ** 2 if not math.isnan(std) else float('nan')):.6g}\n")
            buf.write(f"  {'Sum':<22} {s.sum():.6g}\n")
            buf.write(f"  {'Minimum':<22} {s.min():.6g}\n")
            buf.write(f"  {'Maximum':<22} {s.max():.6g}\n")
            buf.write(f"  {'Range':<22} {(s.max() - s.min()):.6g}\n")
            buf.write(f"  {'Skewness':<22} {s.skew() if n > 2 else float('nan'):.6g}\n")
            buf.write(f"  {'Kurtosis':<22} {s.kurt() if n > 3 else float('nan'):.6g}\n")
            buf.write("\n  Quantiles:\n")
            for label, q in (("100% Max", 1.0), ("99%", 0.99), ("95%", 0.95),
                             ("90%", 0.90), ("75% Q3", 0.75), ("50% Median", 0.50),
                             ("25% Q1", 0.25), ("10%", 0.10), ("5%", 0.05),
                             ("1%", 0.01), ("0% Min", 0.0)):
                buf.write(f"    {label:<14} {s.quantile(q):.6g}\n")
            # Extreme observations
            sorted_vals = s.sort_values()
            lowest = sorted_vals.head(5).tolist()
            highest = sorted_vals.tail(5).tolist()
            buf.write(f"\n  Lowest:  {', '.join(f'{v:g}' for v in lowest)}\n")
            buf.write(f"  Highest: {', '.join(f'{v:g}' for v in highest)}\n")

    if by_cols:
        for key, block in df.groupby(by_cols, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            header = "  ".join(f"{c}={k}" for c, k in zip(by_cols, key))
            _write_stats(block, header)
    else:
        _write_stats(df)

    output = buf.getvalue()
    if not noprint:
        reporter.log(output)

    return StepResult(success=True, rows_affected=ds.nrow,
                      output_messages=[] if noprint else [output])


# ─── PROC COMPARE ──────────────────────────────────────


def handle_proc_compare(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC COMPARE — compare two datasets."""
    base_name = proc.options.get("BASE", "")
    comp_name = proc.options.get("COMPARE", "")
    out_name = proc.options.get("OUT", "")
    brief = proc.options.get("BRIEF", False)

    if not base_name:
        return StepResult(success=False, error="PROC COMPARE requires BASE=")
    if not comp_name:
        return StepResult(success=False, error="PROC COMPARE requires COMPARE=")

    try:
        base_ds = _resolve_dataset(session, base_name)
    except KeyError:
        return StepResult(success=False, error=f"Dataset {base_name} not found")
    try:
        comp_ds = _resolve_dataset(session, comp_name)
    except KeyError:
        return StepResult(success=False, error=f"Dataset {comp_name} not found")

    base_df = base_ds.data
    comp_df = comp_ds.data

    id_names: list[str] = []
    var_names: list[str] = []
    for stmt in proc.statements:
        if isinstance(stmt, ByNode):
            id_names = [v.upper() for v in stmt.variables]
        elif isinstance(stmt, VarListNode):
            var_names = [v.upper() for v in stmt.variables]

    base_map = _col_map(base_df)
    comp_map = _col_map(comp_df)

    common_upper = [u for u in base_map if u in comp_map]
    base_only = [u for u in base_map if u not in comp_map]
    comp_only = [u for u in comp_map if u not in base_map]

    if var_names:
        common_upper = [u for u in common_upper if u in var_names]
    compare_cols = [u for u in common_upper if u not in id_names]

    buf = io.StringIO()
    buf.write(f"\n{'=' * 60}\n")
    buf.write(f"  PROC COMPARE: BASE={base_name}  COMPARE={comp_name}\n")
    buf.write(f"{'=' * 60}\n\n")
    buf.write(f"  Base observations:    {len(base_df)}\n")
    buf.write(f"  Compare observations: {len(comp_df)}\n")
    buf.write(f"  Common variables:     {len(common_upper)}\n")
    if base_only:
        buf.write(f"  Variables only in BASE:    {', '.join(base_only)}\n")
    if comp_only:
        buf.write(f"  Variables only in COMPARE: {', '.join(comp_only)}\n")

    # Align rows: by ID keys if given, else by position
    if id_names:
        b_ids = [base_map[i] for i in id_names if i in base_map]
        c_ids = [comp_map[i] for i in id_names if i in comp_map]
        if len(b_ids) != len(id_names) or len(c_ids) != len(id_names):
            return StepResult(success=False, error="PROC COMPARE: ID variable not found in both datasets")
        merged = base_df.merge(comp_df, left_on=b_ids, right_on=c_ids,
                               how="inner", suffixes=("_BASE", "_COMP"))
        pairs = []
        for u in compare_cols:
            b, c = base_map[u], comp_map[u]
            b_col = f"{b}_BASE" if f"{b}_BASE" in merged.columns else b
            c_col = f"{c}_COMP" if f"{c}_COMP" in merged.columns else c
            pairs.append((u, merged[b_col], merged[c_col]))
        id_frame = merged[b_ids]
    else:
        n = min(len(base_df), len(comp_df))
        pairs = [(u, base_df[base_map[u]].iloc[:n].reset_index(drop=True),
                  comp_df[comp_map[u]].iloc[:n].reset_index(drop=True))
                 for u in compare_cols]
        id_frame = pd.DataFrame({"_OBS_": range(1, n + 1)})

    total_diffs = 0
    diff_rows: list[dict[str, Any]] = []
    for u, b_vals, c_vals in pairs:
        b_vals = b_vals.reset_index(drop=True)
        c_vals = c_vals.reset_index(drop=True)
        both_na = b_vals.isna() & c_vals.isna()
        neq = ~(both_na | (b_vals == c_vals))
        # Numeric tolerance for float comparisons
        if pd.api.types.is_numeric_dtype(b_vals) and pd.api.types.is_numeric_dtype(c_vals):
            close = (b_vals - c_vals).abs() < 1e-12
            neq = neq & ~close.fillna(False)
        n_diff = int(neq.sum())
        if n_diff:
            total_diffs += n_diff
            if not brief:
                buf.write(f"\n  Variable {u}: {n_diff} differing value(s)\n")
                shown = 0
                for idx in neq[neq].index:
                    if shown >= 10:
                        buf.write("    ... (truncated)\n")
                        break
                    key_repr = ", ".join(f"{k}={v}" for k, v in
                                         id_frame.iloc[idx].to_dict().items())
                    buf.write(f"    [{key_repr}] base={b_vals.iloc[idx]} compare={c_vals.iloc[idx]}\n")
                    shown += 1
            for idx in neq[neq].index:
                row = {"_VAR_": u}
                row.update(id_frame.iloc[idx].to_dict())
                row["_BASE_"] = b_vals.iloc[idx]
                row["_COMPARE_"] = c_vals.iloc[idx]
                diff_rows.append(row)

    obs_diff = len(base_df) - len(comp_df)
    if obs_diff != 0:
        buf.write(f"\n  Observation count difference: {obs_diff:+d}\n")

    if total_diffs == 0 and obs_diff == 0 and not base_only and not comp_only:
        buf.write("\n  NOTE: No unequal values were found. All values compared are exactly equal.\n")
    else:
        buf.write(f"\n  Total value differences: {total_diffs}\n")

    if out_name:
        out_libref, out_member = _split_name(out_name)
        out_df = pd.DataFrame(diff_rows)
        out_ds = Dataset.from_dataframe(out_df, name=out_member, libref=out_libref)
        session.put_dataset(out_libref, out_member, out_ds)

    output = buf.getvalue()
    reporter.log(output)

    return StepResult(success=True, rows_affected=total_diffs, output_messages=[output])


# ─── PROC COPY ─────────────────────────────────────────


def handle_proc_copy(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC COPY — copy datasets between libraries."""
    in_lib = str(proc.options.get("IN", "")).upper()
    out_lib = str(proc.options.get("OUT", "")).upper()

    if not in_lib:
        return StepResult(success=False, error="PROC COPY requires IN=")
    if not out_lib:
        return StepResult(success=False, error="PROC COPY requires OUT=")

    select = [v.upper() for v in proc.options.get("_SELECT", [])]
    exclude = [v.upper() for v in proc.options.get("_EXCLUDE", [])]

    try:
        in_backend, _ = session.storage.resolve(in_lib, "_DUMMY_")
    except KeyError:
        return StepResult(success=False, error=f"Library {in_lib} is not assigned")
    try:
        session.storage.resolve(out_lib, "_DUMMY_")
    except KeyError:
        return StepResult(success=False, error=f"Library {out_lib} is not assigned")

    names = [n.upper() for n in in_backend.list_datasets()]
    if select:
        names = [n for n in names if n in select]
    if exclude:
        names = [n for n in names if n not in exclude]

    copied = []
    for name in names:
        try:
            ds = session.get_dataset(in_lib, name)
        except KeyError:
            continue
        new_ds = Dataset.from_dataframe(ds.data.copy(), name=name, libref=out_lib)
        session.put_dataset(out_lib, name, new_ds)
        copied.append(name)

    return StepResult(
        success=True,
        rows_affected=len(copied),
        notes=[f"Copied {len(copied)} dataset(s) from {in_lib} to {out_lib}: "
               f"{', '.join(copied) if copied else '(none)'}"],
    )


# ─── PROC FORMAT ───────────────────────────────────────


def handle_proc_format(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC FORMAT — define custom value formats (VALUE statement)."""
    defined = []
    for stmt in proc.statements:
        if not (isinstance(stmt, dict) and stmt.get("action") == "value"):
            continue
        fmt_name = stmt.get("name", "")
        if not fmt_name:
            continue
        is_char = stmt.get("char", False)
        ranges = stmt.get("ranges", [])
        key = ("$" + fmt_name) if is_char else fmt_name
        session._formats[key.upper()] = {
            "char": is_char,
            "ranges": ranges,
        }
        defined.append(key.upper())

    if not defined:
        return StepResult(success=False, error="PROC FORMAT requires at least one VALUE statement")

    return StepResult(
        success=True,
        notes=[f"Format(s) defined: {', '.join(defined)}"],
    )


def apply_custom_format(session: Session, fmt_name: str, value: Any) -> str | None:
    """Look up a value in a session-defined custom format. Returns None if
    the format is not defined or no range matches (other than OTHER)."""
    fmt = session._formats.get(fmt_name.upper().rstrip("."))
    if fmt is None:
        return None
    other_label = None
    for rng in fmt["ranges"]:
        label = rng.get("label", "")
        for kind, lo, hi in rng.get("keys", []):
            if kind == "other":
                other_label = label
            elif kind == "exact":
                if fmt["char"]:
                    if str(value) == str(lo):
                        return label
                else:
                    try:
                        if float(value) == float(lo):
                            return label
                    except (TypeError, ValueError):
                        pass
            elif kind == "range":
                try:
                    v = float(value)
                    if (lo is None or v >= lo) and (hi is None or v <= hi):
                        return label
                except (TypeError, ValueError):
                    pass
    return other_label


# ─── PROC TABULATE ─────────────────────────────────────


_TAB_STATS = {"N": "count", "MEAN": "mean", "SUM": "sum", "MIN": "min",
              "MAX": "max", "STD": "std", "MEDIAN": "median"}


def handle_proc_tabulate(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC TABULATE — simple class*var*stat summary tables."""
    data_name = proc.options.get("DATA", "")
    if not data_name:
        return StepResult(success=False, error="PROC TABULATE requires DATA=")

    try:
        ds = _resolve_dataset(session, data_name)
    except KeyError:
        return StepResult(success=False, error=f"Dataset {data_name} not found")

    df = ds.data
    cmap = _col_map(df)

    class_names: list[str] = []
    var_names: list[str] = []
    tables: list[list[list[str]]] = []
    for stmt in proc.statements:
        if isinstance(stmt, ClassNode):
            class_names = [v.upper() for v in stmt.variables]
        elif isinstance(stmt, VarListNode):
            var_names = [v.upper() for v in stmt.variables]
        elif isinstance(stmt, dict) and stmt.get("action") == "table":
            tables.append(stmt.get("terms", []))

    if not tables:
        return StepResult(success=False, error="PROC TABULATE requires TABLE statement")

    buf = io.StringIO()
    buf.write(f"\n{'=' * 60}\n")
    buf.write(f"  PROC TABULATE: {ds.metadata.libref}.{ds.metadata.member_name}\n")
    buf.write(f"{'=' * 60}\n")

    for terms in tables:
        # Each term is a list like [CLASSVAR, ANALYSISVAR, STAT]
        for term in terms:
            class_part = [t for t in term if t in class_names and t in cmap]
            analysis_part = [t for t in term if t in var_names and t in cmap]
            stat_part = [t for t in term if t in _TAB_STATS]

            stat = _TAB_STATS.get(stat_part[0], "mean") if stat_part else "mean"
            stat_label = stat_part[0] if stat_part else "MEAN"

            if not analysis_part:
                # Pure class frequency table
                if class_part:
                    actual = [cmap[c] for c in class_part]
                    counts = df.groupby(actual, dropna=False).size()
                    buf.write(f"\n  Table: {' * '.join(class_part)} (N)\n\n")
                    buf.write(counts.to_string())
                    buf.write("\n")
                continue

            a_cols = [cmap[a] for a in analysis_part]
            if class_part:
                actual = [cmap[c] for c in class_part]
                grouped = df.groupby(actual, dropna=False)[a_cols].agg(stat)
                buf.write(f"\n  Table: {' * '.join(class_part)} * "
                          f"{' * '.join(analysis_part)} ({stat_label})\n\n")
                buf.write(grouped.to_string())
                buf.write("\n")
            else:
                vals = df[a_cols].agg(stat)
                buf.write(f"\n  Table: {' * '.join(analysis_part)} ({stat_label})\n\n")
                buf.write(vals.to_string())
                buf.write("\n")

    output = buf.getvalue()
    reporter.log(output)
    return StepResult(success=True, rows_affected=ds.nrow, output_messages=[output])


# ─── PROC REPORT ───────────────────────────────────────


def handle_proc_report(proc: ProcNode, session: Session, reporter: Reporter) -> StepResult:
    """PROC REPORT — basic COLUMN/DEFINE report with GROUP + ANALYSIS support."""
    data_name = proc.options.get("DATA", "")
    out_name = proc.options.get("OUT", "")
    if not data_name:
        return StepResult(success=False, error="PROC REPORT requires DATA=")

    try:
        ds = _resolve_dataset(session, data_name)
    except KeyError:
        return StepResult(success=False, error=f"Dataset {data_name} not found")

    df = ds.data.copy()
    cmap = _col_map(df)

    columns: list[str] = []
    defines: dict[str, dict[str, Any]] = {}
    for stmt in proc.statements:
        if isinstance(stmt, dict):
            if stmt.get("action") == "column":
                columns = [c.upper() for c in stmt.get("names", [])]
            elif stmt.get("action") == "define":
                defines[stmt["name"]] = stmt

    if not columns:
        columns = [c.upper() for c in df.columns]

    actual_cols = [cmap[c] for c in columns if c in cmap]
    if not actual_cols:
        return StepResult(success=False, error="PROC REPORT: no valid columns")

    group_cols = []
    analysis: dict[str, str] = {}  # actual col -> stat
    for cu in columns:
        if cu not in cmap:
            continue
        d = defines.get(cu, {})
        attrs = [a.upper() for a in d.get("attrs", [])] if d else []
        if "GROUP" in attrs or "ORDER" in attrs:
            group_cols.append(cmap[cu])
        else:
            stat = next((s for s in ("SUM", "MEAN", "MIN", "MAX", "N") if s in attrs), None)
            if "ANALYSIS" in attrs or stat:
                analysis[cmap[cu]] = (stat or "SUM").lower()

    if group_cols and analysis:
        agg_spec = {col: ("count" if st == "n" else st) for col, st in analysis.items()}
        report_df = df.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    elif group_cols:
        report_df = df[actual_cols].drop_duplicates(subset=group_cols).reset_index(drop=True)
    else:
        report_df = df[actual_cols]

    # Apply column labels from DEFINE
    rename_labels = {}
    for cu, d in defines.items():
        label = d.get("label", "")
        if label and cu in cmap and cmap[cu] in report_df.columns:
            rename_labels[cmap[cu]] = label
    display_df = report_df.rename(columns=rename_labels)

    buf = io.StringIO()
    buf.write(f"\n{'=' * 60}\n")
    buf.write(f"  PROC REPORT: {ds.metadata.libref}.{ds.metadata.member_name}\n")
    buf.write(f"{'=' * 60}\n\n")
    display = display_df.copy()
    display.index = range(1, len(display) + 1)
    display.index.name = "Obs"
    buf.write(display.to_string())
    buf.write("\n")

    output = buf.getvalue()
    reporter.log(output)

    if out_name:
        out_libref, out_member = _split_name(out_name)
        out_ds = Dataset.from_dataframe(report_df, name=out_member, libref=out_libref)
        session.put_dataset(out_libref, out_member, out_ds)

    return StepResult(success=True, rows_affected=len(report_df), output_messages=[output])
