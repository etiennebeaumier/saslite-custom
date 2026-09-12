"""Headless PNG output; numerical inputs are shared with terminal plots."""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from scipy import stats

from saslite.runtime.terminal_plots import box_summary, histogram_data, density_data, qq_data


def graphics_enabled(session):
    return (session.get_option("PNG_EXPORT", False)
            and session.get_option("ODS_GRAPHICS", True)
            and session.get_option("TERMINAL_PLOTS", True))


def validate_ods(options, session):
    allowed = {"ODS_GRAPHICS", "ODS_IMAGENAME", "ODS_IMAGEFMT", "ODS_GPATH"}
    for name in options:
        if name.startswith("ODS_") and name not in allowed:
            raise ValueError(f"unsupported ODS graphics option {name[4:]}")
    if "ODS_IMAGEFMT" in options and str(options["ODS_IMAGEFMT"]).upper() != "PNG":
        raise ValueError("ODS GRAPHICS supports IMAGEFMT=PNG only")
    if "ODS_IMAGENAME" in options:
        name = str(options["ODS_IMAGENAME"])
        if not name.strip() or name in (".", "..") or any(c in name for c in ('/', '\\', '\x00')):
            raise ValueError("ODS IMAGENAME must be a filename base without directory separators")
    if "ODS_GPATH" in options:
        if not str(options["ODS_GPATH"]).strip():
            raise ValueError("ODS GPATH must be a nonempty folder path")
        path = Path(str(options["ODS_GPATH"])).expanduser()
        if not path.is_absolute():
            path = Path(session.get_option("GRAPHICS_BASE_DIR", Path.cwd())) / path
        options["ODS_GPATH"] = str(path.resolve())


def _safe_name(value):
    return re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._")[:160] or "graph"


class PngWriter:
    def __init__(self, session, reporter, proc):
        self.session, self.reporter = session, reporter
        self.enabled = graphics_enabled(session) and not proc.options.get("NOPRINT", False)
        self.prefix = proc.options.get("DATA", "dataset")
        self.group = proc.options.get("_BY_LABEL", "")
        self.paths = []
        self.errors = []

    def draw(self, kind, variable, title, draw):
        if not self.enabled:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(8, 16 / 3), dpi=150, facecolor="white", layout="constrained")
        FigureCanvasAgg(fig)
        try:
            ax = fig.subplots()
            ax.set_title(title + ("\n" + self.group if self.group else ""), fontsize=12, wrap=True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
            draw(ax)
            self.save(fig, f"{self.prefix}_{kind}_{variable}_{self.group}".strip("_"))
        except (OSError, ValueError, RuntimeError) as exc:
            self.errors.append(f"PNG {kind} ({variable}): {exc}")
        finally:
            fig.clear()

    def save(self, fig, base):
        folder = Path(self.session.get_option("ODS_GPATH",
                      Path(self.session.get_option("GRAPHICS_BASE_DIR", Path.cwd())) / "graphs"))
        base = str(self.session.get_option("ODS_IMAGENAME", _safe_name(base)))
        if base.lower().endswith(".png"):
            base = base[:-4]
        if not base:
            raise ValueError("IMAGENAME needs a nonempty filename base")
        # Render before creating a directory; publish a complete file exclusively.
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, facecolor="white")
        folder.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".saslite-", suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                f.write(buffer.getvalue())
            index = 0
            while True:
                target = folder / f"{base}{index if index else ''}.png"
                try:
                    os.link(temporary, target)
                    break
                except FileExistsError:
                    index += 1
            path = str(target.resolve())
            self.paths.append(path)
            self.reporter.note(f"Graph saved: {path}")
        except OSError as exc:
            raise OSError(f"cannot save graph in {folder}: {exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def draw_boxes(ax, samples, labels, variable):
    from matplotlib.patches import Rectangle
    for i, sample in enumerate(samples, 1):
        q1, median, q3, low, high, outliers = box_summary(sample)
        ax.plot([low, q1], [i, i], color="#235e91")
        ax.plot([q3, high], [i, i], color="#235e91")
        ax.vlines([low, high], i-.15, i+.15, color="#235e91")
        ax.add_patch(Rectangle((q1, i-.3), q3-q1, .6, facecolor="#d3e6f4", edgecolor="#235e91"))
        ax.vlines(median, i-.3, i+.3, color="#173d5e", linewidth=2)
        ax.scatter(outliers, np.full(len(outliers), i), facecolors="none", edgecolors="#235e91", s=25)
        ax.scatter([np.mean(sample)], [i], marker="D", s=22, color="#b05229", zorder=3,
                   label="Mean" if i == 1 else None)
    ax.set_yticks(range(1, len(labels)+1), labels)
    ax.set_ylim(.4, len(labels)+.6)
    ax.set_xlabel(variable)
    ax.margins(x=.08)
    ax.legend(fontsize=8)


def draw_histograms(ax, samples, labels, variable, density=False):
    edges, percentages = histogram_data(samples, 80)
    for sample, label, percent in zip(samples, labels, percentages):
        ax.bar(edges[:-1], percent, width=np.diff(edges), align='edge',
               alpha=.35, edgecolor='#235e91', linewidth=.7, label=label)
        if density and len(sample) > 1 and np.std(sample, ddof=1) > 0:
            xs, normal, kernel = density_data(sample, edges, 300)
            ax.plot(xs, normal, label=f"{label}: normal", linewidth=1.5)
            ax.plot(xs, kernel, '--', label=f"{label}: kernel", linewidth=1.5)
    ax.set(xlabel=variable, ylabel="Percent")
    ax.set_ylim(bottom=0)
    if density or len(labels) > 1:
        ax.legend(fontsize=8)


def draw_intervals(ax, intervals, h0):
    valid = [(label, mean, lo, hi) for label, mean, lo, hi in intervals
             if math.isfinite(mean) and not math.isnan(lo) and not math.isnan(hi)]
    values = [h0, *[v for _, mean, lo, hi in valid for v in (mean, lo, hi) if math.isfinite(v)]]
    low, high = min(values), max(values)
    margin = (high-low or max(abs(low), 1)) * .15
    left, right = low-margin, high+margin
    for i, (label, mean, lo, hi) in enumerate(valid, 1):
        a, b = (lo if math.isfinite(lo) else left), (hi if math.isfinite(hi) else right)
        ax.plot([a,b], [i,i], color="#235e91", linewidth=2)
        for end, finite in [(a, math.isfinite(lo)), (b, math.isfinite(hi))]:
            if finite:
                ax.vlines(end, i-.12, i+.12, color="#235e91")
            else:
                ax.annotate('', xy=(end,i), xytext=(mean,i), arrowprops=dict(arrowstyle='->', color='#235e91'))
        ax.scatter([mean], [i], color="#235e91", zorder=3)
    ax.axvline(h0, linestyle=':', color='#a34d2b', label=f'H0={h0:g}')
    ax.set_yticks(range(1,len(valid)+1), [x[0] for x in valid])
    ax.set(xlim=(left-margin/2, right+margin/2), ylim=(.4,len(valid)+.6), xlabel="Mean / mean difference")
    ax.legend(fontsize=8)


def ttest_pngs(writer, samples, labels, variable, intervals, h0, alpha, selected, paired=None):
    if not writer.enabled:
        return
    samples = [np.asarray(s, dtype=float) for s in samples]
    if any(len(s) < 2 or not np.isfinite(s).all() for s in samples):
        return
    if selected & {'SUMMARY', 'HISTOGRAM'}:
        writer.draw('histogram', variable, f'Distributions: {variable}',
                    lambda ax: draw_histograms(ax, samples, labels, variable, density=True))
    if selected & {'SUMMARY', 'BOX'}:
        writer.draw('box', variable, f'Box plots: {variable}', lambda ax: draw_boxes(ax, samples, labels, variable))
    if selected & {'SUMMARY', 'INTERVAL'}:
        writer.draw('interval', variable, f'{100*(1-alpha):g}% confidence intervals: {variable}',
                    lambda ax: draw_intervals(ax, intervals, h0))
    if 'QQPLOT' in selected:
        def qq(ax):
            for sample, label in zip(samples, labels):
                quantiles, reference = qq_data(sample)
                ax.scatter(quantiles, np.sort(sample), s=20, label=label)
                ax.plot(quantiles, reference, '--', linewidth=1)
            ax.set(xlabel='Normal quantiles', ylabel=variable)
            ax.legend(fontsize=8)
        writer.draw('qqplot', variable, f'Normal Q–Q plots: {variable}', qq)
    if paired is not None:
        a, b, names = paired
        if 'PROFILES' in selected:
            def profiles(ax):
                ax.plot([0,1], np.vstack([a,b]), '-o', alpha=.6, markersize=3)
                ax.set_xticks([0,1], names)
                ax.set_ylabel('Observed value')
            writer.draw('profiles', variable, f'Paired profiles (N={len(a)}): {variable}', profiles)
        if 'AGREEMENT' in selected:
            def agreement(ax):
                bounds = [min(np.min(a),np.min(b)), max(np.max(a),np.max(b))]
                ax.scatter(a,b,s=25)
                ax.plot(bounds,bounds,':',label='Equality')
                ax.plot(bounds,np.asarray(bounds)+np.mean(b-a),'--',label='Mean difference')
                ax.scatter([np.mean(a)],[np.mean(b)],marker='D',label='Means')
                ax.set(xlabel=names[0],ylabel=names[1])
                ax.legend(fontsize=8)
            writer.draw('agreement', variable, f'Paired agreement (N={len(a)}): {variable}', agreement)
