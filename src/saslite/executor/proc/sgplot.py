"""The initial SGPLOT subset: histogram, horizontal box, and scatter."""
import numpy as np
import pandas as pd

from saslite.executor.proc.extras import _resolve_dataset
from saslite.executor.proc.grouping import resolve_columns
from saslite.runtime.execution_result import StepResult
from saslite.runtime.png_graphics import PngWriter, draw_boxes, draw_histograms


def handle_proc_sgplot(proc, session, reporter):
    if len(proc.statements) != 1:
        return StepResult(success=False, error="PROC SGPLOT supports one HISTOGRAM, HBOX, or SCATTER statement per RUN")
    try:
        dataset = _resolve_dataset(session, proc.options['DATA'])
        statement = proc.statements[0]
        columns = [resolve_columns(dataset.data, [name])[0] for name in statement['variables']]
        for column in columns:
            if not pd.api.types.is_numeric_dtype(dataset.data[column]):
                raise ValueError(f"analysis variable {column} must be numeric")
        data = dataset.data[list(dict.fromkeys(columns))].replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            raise ValueError("no usable finite observations for this graph")
    except (KeyError, ValueError) as exc:
        return StepResult(success=False, error=f"PROC SGPLOT: {exc}")
    warnings = []
    if len(data) < dataset.nrow:
        warnings.append(f"PROC SGPLOT: omitted {dataset.nrow-len(data)} observations with missing or nonfinite plot values")
    writer = PngWriter(session, reporter, proc)
    if not writer.enabled:
        return StepResult(rows_affected=len(data), warnings=warnings,
                          notes=["PROC SGPLOT: PNG output is disabled; use ODS GRAPHICS ON (unless --no-plots is set)"])
    labels = []
    for col in columns:
        meta = dataset.metadata.variables.get(str(col).upper())
        labels.append((meta.label if meta else '') or str(col))
    kind = statement['action']
    variable = '_'.join(str(c) for c in columns)
    values = data[columns[0]].to_numpy(dtype=float)
    if kind == 'histogram':
        writer.draw(kind, variable, f'Histogram: {labels[0]}',
                    lambda ax: draw_histograms(ax, [values], [labels[0]], labels[0]))
    elif kind == 'hbox':
        writer.draw(kind, variable, f'Box plot: {labels[0]}',
                    lambda ax: draw_boxes(ax, [values], [labels[0]], labels[0]))
    else:
        def scatter(ax):
            ax.scatter(data[columns[0]], data[columns[1]], s=28, alpha=.7, color='#235e91')
            ax.set(xlabel=labels[0], ylabel=labels[1])
        writer.draw(kind, variable, f'{labels[1]} versus {labels[0]}', scatter)
    return StepResult(success=not writer.errors, error='; '.join(writer.errors) or None,
                      rows_affected=len(data), warnings=warnings, image_paths=writer.paths)
