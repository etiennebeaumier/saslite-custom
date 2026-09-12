"""Dependency-free text rendering of SAS-style TTEST diagnostic plots.

Numerics use numpy/scipy already required by SASLite. Histogram binning and
Scott KDE bandwidth are local choices; this is not an ODS graphics engine.
"""
from __future__ import annotations

import math
import shutil
import textwrap

import numpy as np
from scipy import stats


def _label(value):
    return ''.join(c if 32 <= ord(c) < 127 else '?' for c in str(value))


def _num(value):
    return f'{float(value):.4g}'


def plot_width(session):
    requested = session.get_option('PLOTWIDTH')
    if requested is None:
        requested = min(int(session.get_option('LINESIZE', 80)), shutil.get_terminal_size((80, 24)).columns)
    return max(40, min(160, int(requested)))


def _domain(values):
    low, high = float(np.min(values)), float(np.max(values))
    pad = (high - low) * .06 if high > low else max(abs(low) * .05, .5)
    return low - pad, high + pad


def _ticks(low, high, width):
    row = [' '] * width
    occupied = set()
    # Reserve endpoints before adding intermediate labels on narrow terminals.
    for fraction in (0, 1, .5, .25, .75):
        label = _num(low + fraction * (high - low))
        position = max(0, min(width - len(label), round(fraction * (width - 1)) - len(label) // 2))
        if not occupied.intersection(range(position - 1, position + len(label) + 1)):
            row[position:position + len(label)] = label
            occupied.update(range(position, position + len(label)))
    return ''.join(row).rstrip()


class Canvas:
    def __init__(self, xlim, ylim, width, height=11):
        self.xlim, self.ylim = xlim, ylim
        self.width, self.height = width - 12, height
        self.cells = [[' '] * self.width for _ in range(height)]

    def x(self, value):
        return round((value - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.width - 1))

    def y(self, value):
        return round((self.ylim[1] - value) / (self.ylim[1] - self.ylim[0]) * (self.height - 1))

    def point(self, x, y, mark='o', overlap=False):
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return
        col, row = self.x(x), self.y(y)
        if 0 <= col < self.width and 0 <= row < self.height:
            previous = self.cells[row][col]
            self.cells[row][col] = '*' if overlap and previous in ('o', '*') else mark

    def line(self, xs, ys, mark='.'):
        for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
            if not all(math.isfinite(float(v)) for v in (x0, y0, x1, y1)):
                continue
            steps = min(2000, max(abs(self.x(x1)-self.x(x0)), abs(self.y(y1)-self.y(y0)), 1) + 1)
            for x, y in zip(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps)):
                self.point(x, y, mark)

    def render(self, xlabel, ylabel):
        lines = [f'  {ylabel}']
        for i, row in enumerate(self.cells):
            value = self.ylim[1] - i / (self.height - 1) * (self.ylim[1] - self.ylim[0])
            tick = _num(value) if i in (0, self.height // 2, self.height - 1) else ''
            lines.append(f'{tick:>9} |' + ''.join(row))
        lines.extend(['          +' + '-' * self.width,
                      '           ' + _ticks(*self.xlim, self.width),
                      '           ' + _label(xlabel)])
        return lines


def box_summary(values):
    values = np.sort(np.asarray(values, dtype=float))
    q1, median, q3 = np.quantile(values, [.25, .5, .75], method='averaged_inverted_cdf')
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
    return float(q1), float(median), float(q3), float(inside[0]), float(inside[-1]), outliers


def histogram_data(samples, width):
    joined = np.concatenate(samples)
    bounds = _domain(joined)
    bins = max(3, min(12, (width - 12) // 4, math.ceil(math.sqrt(max(map(len, samples))))))
    edges = np.linspace(*bounds, bins + 1)
    percentages = [np.histogram(sample, edges)[0] * 100. / len(sample) for sample in samples]
    return edges, percentages


def density_data(sample, edges, points):
    xs = np.linspace(edges[0], edges[-1], points)
    sd = float(np.std(sample, ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return xs, np.zeros_like(xs), np.zeros_like(xs)
    scale = (edges[1] - edges[0]) * 100
    return xs, stats.norm.pdf(xs, np.mean(sample), sd) * scale, stats.gaussian_kde(sample, bw_method='scott')(xs) * scale


def qq_data(sample):
    xs = stats.norm.ppf((np.arange(1, len(sample)+1) - .375) / (len(sample) + .25))
    return xs, np.mean(sample) + np.std(sample, ddof=1) * xs


def _histograms(samples, labels, variable, width):
    edges, percentages = histogram_data(samples, width)
    curves = []
    for sample in samples:
        xs, normal, kernel = density_data(sample, edges, width - 12)
        curves.append((normal, kernel))
    ymax = max(float(np.max(values)) for values in [*percentages, *[c for pair in curves for c in pair]]) * 1.08
    lines = ['  Histograms and Density Curves', '  # histogram   . normal fit   ~ kernel density']
    lines.append('  Same bins and scales; percent within each sample.')
    for sample, label, counts, (normal, kernel) in zip(samples, labels, percentages, curves):
        lines.extend(['', f'  {label} (N={len(sample)})'])
        canvas = Canvas((edges[0], edges[-1]), (0, ymax), width)
        previous_bin = -1
        for col, x in enumerate(np.linspace(edges[0], edges[-1], canvas.width)):
            bin_index = min(len(counts)-1, max(0, np.searchsorted(edges, x, side='right') - 1))
            gap = previous_bin >= 0 and bin_index != previous_bin
            previous_bin = bin_index
            if gap:
                continue
            for row in range(canvas.height):
                level = ymax * (canvas.height - row - 1) / (canvas.height - 1)
                if counts[bin_index] > 0 and level <= counts[bin_index]:
                    canvas.cells[row][col] = '#'
        if np.std(sample, ddof=1) > 0:
            # Do not paint effectively-zero density over low-frequency bars.
            threshold = ymax / (canvas.height - 1) / 2
            canvas.line(xs, np.where(normal >= threshold, normal, np.nan), '.')
            canvas.line(xs, np.where(kernel >= threshold, kernel, np.nan), '~')
        else:
            lines.append('  Constant sample: density fits unavailable.')
        lines.extend(canvas.render(variable, 'Percent'))
    return lines


def _boxes(samples, labels, variable, width):
    bounds = _domain(np.concatenate(samples))
    canvas = Canvas(bounds, (0, 1), width)
    lines = ['  Box Plots', '  [==|==] middle 50%; | median; o outlier; M mean',
             '  Whiskers: extreme observations within 1.5 IQR.']
    for sample, label in zip(samples, labels):
        q1, median, q3, low, high, outliers = box_summary(sample)
        row = [' '] * canvas.width
        for left, right, mark in [(low, high, '-'), (q1, q3, '=')]:
            for col in range(canvas.x(left), canvas.x(right) + 1):
                row[col] = mark
        for value, mark in [(low, '|'), (high, '|'), (q1, '['), (q3, ']'), (median, '|')]:
            row[canvas.x(value)] = mark
        for value in outliers:
            row[canvas.x(value)] = 'o'
        mean_row = [' '] * canvas.width
        mean_row[canvas.x(float(np.mean(sample)))] = 'M'
        lines.extend([f'  {label} (N={len(sample)})', '           ' + ''.join(row),
                      '           ' + ''.join(mean_row),
                      f'  Q1={_num(q1)}  Median={_num(median)}  Q3={_num(q3)}; outliers={len(outliers)}'])
    lines.extend(['          +' + '-' * canvas.width,
                  '           ' + _ticks(*bounds, canvas.width), '           ' + variable])
    return lines


def _qq(samples, labels, variable, width):
    quantiles = [qq_data(s)[0] for s in samples]
    xlim = _domain(np.concatenate(quantiles))
    fits = [np.mean(s) + np.std(s, ddof=1) * np.array(xlim) for s in samples]
    ylim = _domain(np.concatenate([*samples, *fits]))
    lines = ['  Normal Q-Q Plots', '  o observation; * overlapping observations',
             '  . normal reference using sample mean and Std Dev']
    for sample, label, xs, fit in zip(samples, labels, quantiles, fits):
        lines.extend(['', f'  {label} (N={len(sample)})'])
        canvas = Canvas(xlim, ylim, width)
        canvas.line(np.array(xlim), fit, '.')
        for x, y in zip(xs, np.sort(sample)):
            canvas.point(x, y, overlap=True)
        lines.extend(canvas.render('Normal Quantiles', variable))
    return lines


def _intervals(intervals, h0, confidence, width):
    finite = [(label, mean, low, high) for label, mean, low, high in intervals
              if math.isfinite(float(mean)) and not any(math.isnan(float(v)) for v in (low, high))]
    if not finite:
        return ['  Mean Confidence Intervals: unavailable.']
    bounds = _domain([h0, *[v for _, mean, low, high in finite for v in (mean, low, high) if math.isfinite(v)]])
    canvas = Canvas(bounds, (0, 1), width)
    lines = [f'  {confidence:g}% Confidence Intervals for the Mean',
             f'  [---M---] confidence interval; : H0={_num(h0)}']
    reference = [' '] * canvas.width
    reference[canvas.x(h0)] = ':'
    lines.append('           ' + ''.join(reference))
    for label, mean, low, high in finite:
        lo, hi = low, high
        low = low if math.isfinite(low) else bounds[0]
        high = high if math.isfinite(high) else bounds[1]
        row = [' '] * canvas.width
        for col in range(canvas.x(low), canvas.x(high)+1):
            row[col] = '-'
        for value, mark in [(low, '[' if math.isfinite(lo) else '<'), (high, ']' if math.isfinite(hi) else '>'), (mean, 'M')]:
            row[canvas.x(value)] = mark
        lines.extend([f'  {label}: {_num(mean)} [{_num(lo) if math.isfinite(lo) else "-Infinity"}, {_num(hi) if math.isfinite(hi) else "Infinity"}]', '           ' + ''.join(row)])
    lines.extend(['          +' + '-' * canvas.width, '           ' + _ticks(*bounds, canvas.width),
                  '           Mean / Mean Difference'])
    return lines


def _paired(a, b, names, width, selected):
    bounds = _domain(np.concatenate([a, b]))
    lines = []
    if 'PROFILES' in selected:
        lines.extend([f'  Profiles Plot (N={len(a)} complete pairs)',
                      f'  Left: {names[0]}   Right: {names[1]}',
                      '  Each line joins the two values of one pair.'])
        canvas = Canvas((0, 1), bounds, width)
        for first, second in zip(a, b):
            canvas.line([0, 1], [first, second], '-')
        for first, second in zip(a, b):
            canvas.point(0, first, overlap=True)
            canvas.point(1, second, overlap=True)
        lines.extend(canvas.render('0 = first variable; 1 = second variable', 'Observed Value'))
    if 'AGREEMENT' in selected:
        lines.extend(['', f'  Agreement Plot (N={len(a)} complete pairs)',
                      '  o pair; * overlapping pairs; M sample means',
                      '  . equality line; - mean-difference line'])
        canvas = Canvas(bounds, bounds, width)
        canvas.line(bounds, bounds, '.')
        canvas.line(bounds, np.array(bounds) + np.mean(b-a), '-')
        for first, second in zip(a, b):
            canvas.point(first, second, overlap=True)
        canvas.point(np.mean(a), np.mean(b), 'M')
        lines.extend(canvas.render(names[0], names[1]))
    return lines


def render_ttest_plots(samples, labels, variable, intervals, h0, alpha, width=80,
                       selected=None, paired=None):
    """Return bounded-width ASCII panels; exact inference remains in tables."""
    selected = set(selected or ['SUMMARY', 'QQPLOT', 'INTERVAL', 'PROFILES', 'AGREEMENT'])
    samples = [np.asarray(s, dtype=float) for s in samples]
    if not samples or any(len(s) < 2 or not np.all(np.isfinite(s)) for s in samples):
        return '  Graphs unavailable: at least two finite observations per sample are required.\n'
    labels, variable = [_label(s) for s in labels], _label(variable)
    lines = [f'  TTEST Graphs: {variable}', '  Terminal rendering of SAS-style diagnostic plots.']
    if any(len(s) < 8 for s in samples):
        lines.append('  Small sample: distribution shape is difficult to assess.')
    if selected & {'SUMMARY', 'HISTOGRAM'}:
        lines.extend(['', *_histograms(samples, labels, variable, width)])
    if selected & {'SUMMARY', 'BOX'}:
        lines.extend(['', *_boxes(samples, labels, variable, width)])
    if selected & {'SUMMARY', 'INTERVAL'}:
        lines.extend(['', *_intervals(intervals, h0, 100 * (1-alpha), width)])
    if 'QQPLOT' in selected:
        lines.extend(['', *_qq(samples, labels, variable, width)])
    if paired is not None:
        a, b, names = paired
        lines.extend(['', *_paired(np.asarray(a), np.asarray(b), [_label(n) for n in names], width, selected)])
    # Long names and notes wrap outside the chart grid, never through it.
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(_label(line).rstrip(), width=width, subsequent_indent='  ',
                                     replace_whitespace=False, drop_whitespace=False) or [''])
    return '\n'.join(wrapped) + '\n\n'
