"""Validated, contiguous SAS BY groups shared by statistical procedures."""
from dataclasses import replace

import pandas as pd

from saslite.ast.proc import ByNode
from saslite.runtime.execution_result import StepResult


def missing(value):
    return bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())


def resolve_columns(frame, names):
    mapping = {str(c).upper(): c for c in frame.columns}
    columns = []
    for name in names:
        if name.upper() not in mapping:
            raise ValueError(f"variable {name.upper()} not found")
        actual = mapping[name.upper()]
        if actual not in columns:
            columns.append(actual)
    return columns


def by_statement(proc):
    return next((s for s in reversed(proc.statements) if isinstance(s, ByNode)), None)


def by_groups(frame, by):
    """Return (key, frame) groups without sorting or modifying the source."""
    if by is None:
        return [({}, frame)]
    columns = resolve_columns(frame, by.variables)
    if len(columns) != len(by.variables):
        raise ValueError("duplicate BY variables")
    descending = by.descending or [False] * len(columns)
    keys = [tuple((0, None) if missing(v) else (1, v) for v in row)
            for row in frame[columns].itertuples(index=False, name=None)]
    if not by.notsorted:
        for previous, current in zip(keys, keys[1:]):
            for a, b, desc in zip(previous, current, descending):
                if a == b:
                    continue
                if (a < b) == desc:
                    raise ValueError("BY variables are not properly sorted; use PROC SORT with the same BY statement, or BY ... NOTSORTED for contiguous groups")
                break
    boundaries = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i-1]] + [len(keys)]
    return [(dict(zip(columns, frame.iloc[start][columns])), frame.iloc[start:end])
            for start, end in zip(boundaries, boundaries[1:]) if end > start]


def group_label(key):
    return "BY " + "  ".join(f"{str(k).upper()}={'.' if missing(v) else v}" for k, v in key.items())


def run_grouped(proc, session, reporter, handler):
    from saslite.executor.proc.extras import _resolve_dataset
    by = by_statement(proc)
    if by is None:
        return handler(proc, session, reporter)
    try:
        dataset = _resolve_dataset(session, proc.options.get("DATA", ""))
        groups = by_groups(dataset.data, by)
    except (KeyError, ValueError, TypeError) as exc:
        return StepResult(success=False, error=f"PROC {proc.proc_name}: {exc}")
    if not groups:
        return StepResult(success=False, error=f"PROC {proc.proc_name}: no BY groups to analyze")
    combined = StepResult(rows_affected=dataset.nrow)
    errors = []
    for key, frame in groups:
        label = group_label(key)
        subset = replace(dataset, data=frame, metadata=dataset.metadata.copy())
        subset.metadata.row_count = len(frame)
        options = {**proc.options, "_DATASET": subset, "_BY_LABEL": label,
                   "_BY_VARIABLES": [str(c).upper() for c in key]}
        child = replace(proc, options=options, statements=[s for s in proc.statements if not isinstance(s, ByNode)])
        if not proc.options.get("NOPRINT"):
            reporter.log("\n  " + label)
            combined.output_messages.append("\n  " + label)
        try:
            result = handler(child, session, reporter)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result = StepResult(success=False, error=str(exc))
        combined.output_messages.extend(result.output_messages)
        combined.image_paths.extend(result.image_paths)
        combined.warnings.extend(f"{label}: {w}" for w in result.warnings)
        combined.notes.extend(f"{label}: {n}" for n in result.notes)
        if not result.success:
            errors.append(f"{label}: {result.error or 'analysis failed'}")
    combined.success = not errors
    combined.error = "; ".join(errors) or None
    return combined
