"""MEANS/SUMMARY calculations and SAS-style output datasets."""
import io

import numpy as np
import pandas as pd

from saslite.ast.proc import VarListNode, ClassNode
from saslite.executor.proc.extras import _resolve_dataset, _split_name
from saslite.executor.proc.grouping import by_statement, by_groups, resolve_columns, group_label, missing
from saslite.runtime.dataset import Dataset
from saslite.runtime.execution_result import StepResult

STATISTICS = ('N', 'NMISS', 'MEAN', 'SUM', 'MIN', 'MAX', 'STD', 'MEDIAN')
DISPLAY_DEFAULT = ('N', 'MEAN', 'STD', 'MIN', 'MAX')
OUTPUT_DEFAULT = ('N', 'MIN', 'MAX', 'MEAN', 'STD')


def statistic(series, name):
    count = int(series.count())
    if name == 'N':
        return count
    if name == 'NMISS':
        return len(series) - count
    if name == 'SUM':
        return series.sum() if count else np.nan
    if not count or (name == 'STD' and count < 2):
        return np.nan
    return getattr(series, {'MEAN': 'mean', 'MIN': 'min', 'MAX': 'max', 'STD': 'std', 'MEDIAN': 'median'}[name])()


def _class_groups(frame, classes, type_value):
    included = [c for i,c in enumerate(classes) if type_value & (1 << (len(classes)-1-i))]
    if not included:
        return [({}, frame)]
    groups = []
    for key, block in frame.groupby(included, dropna=False, sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        groups.append((dict(zip(included, key)), block))
    return groups


def _output_mapping(output, variables, frame, reserved):
    mapping = []
    occupied = set(reserved)
    for spec in output['specs']:
        stat = spec['stat']
        if stat not in STATISTICS:
            raise ValueError(f"unsupported output statistic {stat}")
        selected = variables if spec['variables'] is None else resolve_columns(frame, spec['variables'])
        if any(c not in variables for c in selected):
            raise ValueError("OUTPUT variables must be numeric analysis variables from VAR")
        names = spec['names']
        if len(names) > len(selected):
            raise ValueError(f"{stat}= has more output names than analysis variables")
        # SAS partial name lists select the first corresponding variables.
        pairs = zip(selected, names) if names else ((c, None) for c in selected)
        for col, explicit in pairs:
            name = explicit or (f'{col}_{stat}' if output['autoname'] else str(col))
            name = name.upper()
            if name in occupied:
                if explicit or not output['autoname']:
                    raise ValueError(f"conflicting output name {name}; use distinct names or / AUTONAME")
                base, i = name, 2
                while name in occupied:
                    name = f'{base}{i}'
                    i += 1
            occupied.add(name)
            mapping.append((col, stat, name))
    return mapping


def handle_proc_means(proc, session, reporter):
    try:
        return _means(proc, session, reporter)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return StepResult(success=False, error=f'PROC {proc.proc_name}: {exc}')


def _means(proc, session, reporter):
    data_name = proc.options.get('DATA', '')
    if not data_name:
        raise ValueError('requires DATA=')
    dataset = _resolve_dataset(session, data_name)
    frame = dataset.data
    by = by_statement(proc)
    by_columns = resolve_columns(frame, by.variables) if by else []
    classes = resolve_columns(frame, [v for s in proc.statements if isinstance(s, ClassNode) for v in s.variables])
    if set(classes) & set(by_columns):
        raise ValueError('a variable cannot be both BY and CLASS')
    var_statements = [s for s in proc.statements if isinstance(s, VarListNode)]
    variables = resolve_columns(frame, [v for s in var_statements for v in s.variables])
    requested = [k for k in proc.options if k in STATISTICS]
    if not var_statements and proc.proc_name == 'MEANS':
        variables = [c for c in frame.columns if c not in classes + by_columns and pd.api.types.is_numeric_dtype(frame[c])]
    if proc.proc_name == 'SUMMARY' and not var_statements and requested:
        raise ValueError('SUMMARY requires VAR when statistics are requested')
    for col in variables:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'analysis variable {col} must be numeric')
        if np.isinf(frame[col].dropna().to_numpy(dtype=float)).any():
            raise ValueError(f'analysis variable {col} contains nonfinite values')
    outputs = [dict(s) for s in proc.statements if isinstance(s, dict) and s.get('action') == 'means_output']
    if proc.options.get('OUT'):
        outputs.append({'out': proc.options['OUT'], 'specs': [], 'autoname': False})
    reserved = {str(c).upper() for c in classes + by_columns} | {'_TYPE_', '_FREQ_', '_STAT_'}
    destinations = set()
    unnamed_index = 1
    for output in outputs:
        if not output['out']:
            while f'WORK.DATA{unnamed_index}' in destinations or session.storage.get_backend('WORK').exists(f'DATA{unnamed_index}'):
                unnamed_index += 1
            output['out'] = f'WORK.DATA{unnamed_index}'
        lib, name = _split_name(output['out'])
        dest = f'{lib}.{name}'
        if dest in destinations:
            raise ValueError(f'duplicate OUTPUT destination {dest}')
        destinations.add(dest)
        if output['specs'] and not variables:
            raise ValueError('OUTPUT statistics require numeric analysis variables; specify VAR')
        if not output['specs']:
            conflicts = [str(c).upper() for c in variables if str(c).upper() in reserved]
            if conflicts:
                raise ValueError('conflicting output name ' + ', '.join(conflicts) +
                                 '; specify explicit statistic output names')
        output['mapping'] = _output_mapping(output, variables, frame, reserved)
        output['rows'] = []
    groups = by_groups(frame, by)  # validate order before emitting output or writing datasets
    buf = io.StringIO()
    display = proc.options.get('PRINT', proc.proc_name == 'MEANS') and not proc.options.get('NOPRINT')
    maxdec = proc.options.get('MAXDEC', 4)
    if not isinstance(maxdec, (int, float)) or int(maxdec) != maxdec or not 0 <= maxdec <= 20:
        raise ValueError('MAXDEC must be an integer from 0 to 20')
    maximum_type = (1 << len(classes)) - 1
    output_types = [maximum_type] if proc.options.get('NWAY') else range(maximum_type+1)
    for by_key, block in groups:
        working = block.copy()
        for col in classes:
            is_missing = working[col].map(missing)
            if proc.options.get('MISSING'):
                # Treat blanks and nulls as the same missing class level.
                working[col] = working[col].astype(object).mask(is_missing, np.nan)
            else:
                working = working.loc[~is_missing]
        if display:
            buf.write(f'\n  The {proc.proc_name} Procedure\n  Data Set: {data_name.upper()}\n')
            if by_key:
                buf.write('  ' + group_label(by_key) + '\n')
        for type_value in output_types:
            for class_key, subgroup in _class_groups(working, classes, type_value):
                header = {str(k).upper(): v for k,v in by_key.items()}
                header.update({str(c).upper(): class_key.get(c, np.nan) for c in classes})
                header.update({'_TYPE_': type_value, '_FREQ_': len(subgroup)})
                needed = set(requested or DISPLAY_DEFAULT)
                for output in outputs:
                    needed.update(s for _,s,_ in output['mapping'])
                    if not output['specs']:
                        needed.update(OUTPUT_DEFAULT)
                calculated = {(c,s): statistic(subgroup[c],s) for c in variables for s in needed}
                for output in outputs:
                    if output['specs']:
                        output['rows'].append({**header, **{name:calculated[c,s] for c,s,name in output['mapping']}})
                    elif variables:
                        for s in OUTPUT_DEFAULT:
                            output['rows'].append({**header, '_STAT_':s, **{str(c).upper():calculated[c,s] for c in variables}})
                    else:
                        output['rows'].append(header.copy())
                if display and type_value == maximum_type:
                    if class_key:
                        buf.write('  ' + '  '.join(f'{c}={"." if missing(v) else v}' for c,v in class_key.items()) + '\n')
                    buf.write(f'  N Obs: {len(subgroup)}\n')
                    if variables:
                        table = pd.DataFrame([{s:calculated[c,s] for s in requested or DISPLAY_DEFAULT} for c in variables],
                                             index=[str(c).upper() for c in variables])
                        buf.write(table.to_string(na_rep='.', float_format=lambda v: f'{v:.{int(maxdec)}f}') + '\n')
    # All syntax, names, grouping, and computations have succeeded before persistence.
    for output in outputs:
        lib, name = _split_name(output['out'])
        columns = [str(c).upper() for c in by_columns + classes] + ['_TYPE_', '_FREQ_']
        columns += ([name for _,_,name in output['mapping']] if output['specs'] else
                    (['_STAT_']+[str(c).upper() for c in variables] if variables else []))
        data = pd.DataFrame(output['rows'], columns=columns)
        session.put_dataset(lib, name, Dataset.from_dataframe(data, name=name, libref=lib))
    text = buf.getvalue()
    if display:
        reporter.log(text)
    return StepResult(rows_affected=dataset.nrow, output_messages=[text] if display else [])
