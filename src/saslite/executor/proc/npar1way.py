"""Mann-Whitney-Wilcoxon, Fligner-Policello and Kruskal-Wallis tests."""
import io
import math
import textwrap

import numpy as np
import pandas as pd
from scipy import stats

from saslite.executor.proc.extras import _resolve_dataset, _col_map
from saslite.executor.proc.stats import _ttest_table, _ttest_number, _ttest_pvalue
from saslite.runtime.execution_result import StepResult
from saslite.runtime.terminal_plots import _boxes, _label, plot_width


def fligner_policello(samples, reference=1):
    """SAS placement-score formula; reference is a zero-based group index.

    Placements count opposite-group values below each observation, with half
    credit for ties. Sorting avoids allocating a pairwise n-by-m matrix.
    SAS's default difference is first-listed minus second-listed placements.
    """
    if len(samples) != 2 or any(len(s) < 2 for s in samples):
        raise ValueError('FP requires at least two observations in each of exactly two groups')
    if reference not in (0, 1):
        raise ValueError('FP reference must identify one of the two groups')
    values = [np.asarray(s, dtype=float) for s in samples]
    if any(not np.isfinite(s).all() for s in values):
        raise ValueError('FP requires finite observations')
    placements = []
    for i, sample in enumerate(values):
        other = np.sort(values[1-i])
        placements.append((np.searchsorted(other, sample, side='left').astype(float)
                           + np.searchsorted(other, sample, side='right')) / 2)
    means = [float(p.mean()) for p in placements]
    sums = [float(p.sum()) for p in placements]
    ss = [float(((p - mean) ** 2).sum()) for p, mean in zip(placements, means)]
    difference = sums[1-reference] - sums[reference]
    denominator = 2 * math.sqrt(ss[0] + ss[1] + means[0] * means[1])
    z = difference / denominator if denominator else float('nan')
    return {'placements': placements, 'means': means, 'sums': sums,
            'stds': [math.sqrt(v / (len(p)-1)) for v, p in zip(ss, placements)],
            'difference': difference, 'z': z, 'one_sided': float(stats.norm.sf(abs(z))),
            'two_sided': float(2 * stats.norm.sf(abs(z)))}


def _write_boxes(buf, samples, labels, variable, width):
    for line in _boxes(samples, labels, variable, width):
        buf.write('\n'.join(textwrap.wrap(_label(line), width=width, subsequent_indent='  ',
                                        replace_whitespace=False, drop_whitespace=False)) + '\n')
    buf.write('\n')


def handle_proc_npar1way(proc, session, reporter):
    from saslite.executor.proc.grouping import run_grouped
    return run_grouped(proc, session, reporter, _handle_proc_npar1way)


def _handle_proc_npar1way(proc, session, reporter):
    from saslite.runtime.png_graphics import PngWriter, draw_boxes
    data_name = proc.options.get('DATA', '')
    wilcoxon = bool(proc.options.get('WILCOXON') or proc.options.get('MWW'))
    fp = bool(proc.options.get('FP'))
    if not data_name or not (wilcoxon or fp):
        return StepResult(success=False, error='PROC NPAR1WAY requires DATA= and WILCOXON (MWW) and/or FP')
    correction = str(proc.options.get('CORRECT', 'YES')).upper()
    if correction not in ('YES', 'NO'):
        return StepResult(success=False, error='PROC NPAR1WAY: CORRECT must be YES or NO')
    raw_plots = proc.options.get('PLOTS', ['ALL'])
    plots = {str(p).upper() for p in (raw_plots if isinstance(raw_plots, list) else [raw_plots])}
    if plots - {'ALL', 'NONE', 'BOXPLOT', 'WILCOXONBOXPLOT', 'WILCOXON', 'FPBOXPLOT', 'FP'} or ('NONE' in plots and len(plots) > 1):
        return StepResult(success=False, error='PROC NPAR1WAY: use PLOTS=ALL, NONE, BOXPLOT, WILCOXONBOXPLOT or FPBOXPLOT')
    try:
        ds = proc.options.get("_DATASET") or _resolve_dataset(session, data_name)
    except KeyError as exc:
        return StepResult(success=False, error=f'PROC NPAR1WAY: {exc}')
    cmap = _col_map(ds.data)
    class_vars, var_cols = [], []
    for stmt in proc.statements:
        if isinstance(stmt, dict) and stmt.get('action') == 'class':
            class_vars = [v.upper() for v in stmt.get('variables', [])]
        elif isinstance(stmt, dict) and stmt.get('action') == 'var':
            var_cols = [v.upper() for v in stmt.get('variables', [])]
    if len(class_vars) != 1:
        return StepResult(success=False, error='PROC NPAR1WAY requires one CLASS variable')
    for name in class_vars + var_cols:
        if name not in cmap:
            return StepResult(success=False, error=f'PROC NPAR1WAY: variable {name} not found')
    class_name = class_vars[0]
    if not var_cols:
        var_cols = [c.upper() for c in ds.data.columns if c.upper() not in [class_name, *proc.options.get("_BY_VARIABLES", [])]
                    and pd.api.types.is_numeric_dtype(ds.data[c])]
    for name in var_cols:
        if not pd.api.types.is_numeric_dtype(ds.data[cmap[name]]):
            return StepResult(success=False, error=f'PROC NPAR1WAY: analysis variable {name} must be numeric')
    writer = PngWriter(session, reporter, proc)
    buf = io.StringIO()
    buf.write(f'\n  The NPAR1WAY Procedure\n  Data Set: {data_name.upper()}\n\n')
    used, warnings = 0, []
    for var in var_cols:
        values = pd.to_numeric(ds.data[cmap[var]], errors='coerce')
        classes = ds.data[cmap[class_name]]
        valid = values.notna() & classes.notna() & classes.map(lambda v: not isinstance(v, str) or bool(v.strip()))
        values, classes = values[valid], classes[valid]
        levels = list(classes.unique())
        if len(levels) < 2 or not np.isfinite(values).all():
            warnings.append(f'PROC NPAR1WAY: {var} needs finite observations in at least two classes')
            continue
        ranks = pd.Series(stats.rankdata(values, method='average'), index=values.index)
        total = len(values)
        center = (total + 1) / 2
        rank_ss = float(((ranks - center) ** 2).sum())
        samples = [values[classes == level].to_numpy() for level in levels]
        counts = [len(sample) for sample in samples]
        sums = [float(ranks[classes == level].sum()) for level in levels]
        sds = [math.sqrt(n * (total - n) / (total * (total - 1)) * rank_ss) for n in counts]
        labels = [f'{level:g}' if isinstance(level, (int, float, np.number)) else str(level) for level in levels]
        fp_result = None
        reference = 1
        if fp:
            if len(levels) != 2 or min(counts) < 2:
                return StepResult(success=False, image_paths=writer.paths, warnings=warnings, error=f'PROC NPAR1WAY FP: {var} requires at least two observations in each of exactly two groups')
            requested = proc.options.get('FP_REFCLASS', 2)
            if isinstance(requested, str):
                if requested not in labels:
                    return StepResult(success=False, image_paths=writer.paths, warnings=warnings, error=f'PROC NPAR1WAY FP: reference class {requested!r} not found')
                reference = labels.index(requested)
            elif requested in (1, 2):
                reference = int(requested) - 1
            else:
                return StepResult(success=False, image_paths=writer.paths, warnings=warnings, error='PROC NPAR1WAY FP: numeric REFCLASS must be 1 or 2')
            fp_result = fligner_policello(samples, reference)
        buf.write(f'  Variable: {var}\n  Classified by: {class_name}\n\n')
        if wilcoxon:
            rows = [[label, str(n), _ttest_number(s), _ttest_number(n * center),
                     _ttest_number(sd, 6), _ttest_number(s/n)]
                    for label, n, s, sd in zip(labels, counts, sums, sds)]
            _ttest_table(buf, 'Wilcoxon Scores (Rank Sums)',
                         [class_name, 'N', 'Sum of Scores', 'Expected Under H0', 'Std Dev Under H0', 'Mean Score'], rows)
            if len(np.unique(values)) < total:
                buf.write('  Average scores were used for ties.\n\n')
            if rank_ss == 0:
                warnings.append(f'PROC NPAR1WAY: {var} is constant; rank tests are undefined')
            if len(levels) == 2:
                # SAS uses the smaller group, first in data if counts are equal.
                selected = int(counts[1] < counts[0])
                delta = sums[selected] - counts[selected] * center
                adjusted = delta - .5 * np.sign(delta) if correction == 'YES' else delta
                z = adjusted / sds[selected] if sds[selected] else float('nan')
                side = '<' if delta <= 0 else '>'
                normal_p = float(stats.norm.sf(abs(z)))
                t_p = float(stats.t.sf(abs(z), total - 1))
                _ttest_table(buf, 'Wilcoxon Two-Sample Test', ['Statistic', 'Value'],
                             [['Statistic (S)', _ttest_number(sums[selected])],
                              ['Mann-Whitney U', _ttest_number(sums[selected] - counts[selected] * (counts[selected] + 1) / 2)],
                              ['Class used for S', labels[selected]]])
                _ttest_table(buf, 'Normal Approximation', ['Statistic', 'Value'],
                             [['Z', _ttest_number(z)],
                              [f'One-Sided Pr {side} Z', _ttest_pvalue(normal_p)],
                              ['Two-Sided Pr > |Z|', _ttest_pvalue(2 * normal_p)]])
                _ttest_table(buf, 't Approximation', ['Statistic', 'Value'],
                             [[f'One-Sided Pr {side} Z', _ttest_pvalue(t_p)],
                              ['Two-Sided Pr > |Z|', _ttest_pvalue(2 * t_p)]])
                if correction == 'YES':
                    buf.write('  Z includes a continuity correction of 0.5.\n\n')
            if rank_ss:
                kw, p_kw = stats.kruskal(*samples)
            else:
                kw = p_kw = float('nan')
            _ttest_table(buf, 'Kruskal-Wallis Test', ['Chi-Square', 'DF', 'Pr > Chi-Square'],
                         [[_ttest_number(kw), str(len(levels)-1), _ttest_pvalue(p_kw)]])
        if fp_result is not None:
            rows = [[label, str(n), _ttest_number(total_p), _ttest_number(mean), _ttest_number(sd)]
                    for label, n, total_p, mean, sd in zip(labels, counts, fp_result['sums'], fp_result['means'], fp_result['stds'])]
            _ttest_table(buf, 'Fligner-Policello Placements', [class_name, 'N', 'Sum', 'Mean', 'Std Dev'], rows)
            buf.write(f'  Reference class X: {labels[reference]}\n')
            buf.write(f'  Difference: {labels[1-reference]} - {labels[reference]} (placement sums)\n\n')
            side = '<' if fp_result['difference'] <= 0 else '>'
            _ttest_table(buf, 'Fligner-Policello Test', ['Statistic', 'Value'],
                         [['Difference', _ttest_number(fp_result['difference'])],
                          ['Z', _ttest_number(fp_result['z'])],
                          [f'One-Sided Pr {side} Z', _ttest_pvalue(fp_result['one_sided'])],
                          ['Two-Sided Pr > |Z|', _ttest_pvalue(fp_result['two_sided'])]])
            if not math.isfinite(fp_result['z']):
                warnings.append(f'PROC NPAR1WAY FP: {var} has a zero test denominator; Z and p-values are undefined')
        if (not proc.options.get('NOPRINT') and 'NONE' not in plots
                and session.get_option('ODS_GRAPHICS', True) and session.get_option('TERMINAL_PLOTS', True)):
            width = plot_width(session)
            buf.write(f'  NPAR1WAY Graphs: {var}\n')
            graph_labels = [f'{class_name} = {label}' for label in labels]
            if 'BOXPLOT' in plots:
                _write_boxes(buf, samples, graph_labels, var, width)
                writer.draw('boxplot', var, f'Raw data: {var}', lambda ax: draw_boxes(ax, samples, graph_labels, var))
            if wilcoxon and plots & {'ALL', 'WILCOXON', 'WILCOXONBOXPLOT'}:
                buf.write('  Wilcoxon Scores (MWW)\n')
                rank_samples = [ranks[classes == level].to_numpy() for level in levels]
                _write_boxes(buf, rank_samples, graph_labels, 'Wilcoxon Score', width)
                writer.draw('wilcoxonboxplot', var, f'Wilcoxon scores: {var}', lambda ax: draw_boxes(ax, rank_samples, graph_labels, 'Wilcoxon Score'))
            if fp_result is not None and plots & {'ALL', 'FP', 'FPBOXPLOT'}:
                buf.write('  Fligner-Policello Placements\n')
                _write_boxes(buf, fp_result['placements'], graph_labels, 'Placement', width)
                writer.draw('fpboxplot', var, f'FP placements: {var}', lambda ax: draw_boxes(ax, fp_result['placements'], graph_labels, 'Placement'))
        used += 1
    if not used:
        return StepResult(success=False, error='PROC NPAR1WAY: no eligible analysis variables', warnings=warnings)
    output = buf.getvalue()
    if not proc.options.get('NOPRINT'):
        reporter.log(output)
    return StepResult(success=not writer.errors, error="; ".join(writer.errors) or None, image_paths=writer.paths, rows_affected=ds.nrow, warnings=warnings,
                      output_messages=[] if proc.options.get('NOPRINT') else [output])
