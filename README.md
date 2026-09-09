# SASLite Custom

A standalone, MIT-licensed customization of SASLite 0.4.1: run a practical subset
of SAS locally in Python, with SAS-style statistical tables, readable terminal
plots, native SAS dataset reading, and an extensive built-in usage guide.

**Release: `0.4.1+custom.6`.** Python 3.10 or newer. The distribution and Python
import remain `saslite`; the dedicated command is `saslite-custom`.
This is an independent project, unaffiliated with SAS Institute Inc.

## Install and run

```sh
git clone https://github.com/etiennebeaumier/saslite-custom.git
cd saslite-custom
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
saslite-custom --version
saslite-custom examples/ttest.sas
saslite-custom examples/rank_tests.sas
saslite-custom help
```

On Windows, create the environment with `py -m venv .venv` and activate it with
`.venv\Scripts\Activate.ps1` in PowerShell. Examples use synthetic, inline data
and need no course files or SAS installation. Core CLI testing runs on Linux
in CI and is also performed locally on macOS; Windows and the optional GUI are
not part of the automated platform checks.

For an exact release, download its wheel from
[GitHub Releases](https://github.com/etiennebeaumier/saslite-custom/releases) and run
`python -m pip install /path/to/saslite-0.4.1+custom.6-py3-none-any.whl`.
For an isolated command available across projects, use `pipx install` with that
wheel path. Private repository downloads require GitHub authentication.

`pip install saslite` installs the upstream PyPI distribution, not this fork.
Use a dedicated environment to avoid replacing one with the other. Both
`saslite` and `saslite-custom` launch this implementation when installed here.
Optional extras: `python -m pip install '.[excel]'` for openpyxl or
`python -m pip install '.[gui]'` for Flask/pywebview. GUI commands are
`saslite-gui` and `saslite-desktop` and require the GUI extra.

## What is customized?

| Area | Custom behavior |
| --- | --- |
| `PROC TTEST` | One-sample, independent and paired tests; descriptive statistics, mean/SD confidence limits, pooled and Satterthwaite results, Folded F variance test; `H0=`, `ALPHA=`, `NOPRINT` |
| `PROC NPAR1WAY` | Wilcoxon/Mann–Whitney U (`MWW` alias), tie adjustment, continuity correction, Kruskal–Wallis, Fligner–Policello (`FP`) with reference-class control |
| Terminal plots | Histograms, fitted density curves, box plots, Q–Q plots, confidence intervals, paired profiles/agreement, rank-score and placement boxes |
| Native data | Read real `.sas7bdat` through ReadStat, with a pandas fallback; script-adjacent library configuration and CLI overrides |
| Help | Packaged guide, procedure examples, live CLI/function inventory, and a mandatory source review guard for release builds |

The inherited interpreter also supports DATA steps, SQL, macros, common
functions, and several statistical and data-management procedures. Run
`saslite-custom help procedures` for the implemented inventory and scope.

## Everyday usage

```sh
saslite-custom your_program.sas
saslite-custom -e 'data demo; x=42; run; proc print data=demo; run;'
saslite-custom -i
saslite-custom --no-plots examples/ttest.sas
saslite-custom --plot-width 120 examples/paired.sas
saslite-custom examples/ttest.sas > results.txt 2>&1
saslite-custom help topics
saslite-custom help proc ttest
saslite-custom help fp
saslite-custom help function sqrt
```

Tables and run logs currently go to stderr, so capture both output streams.
In the interactive prompt, use `help;` or `help fp;`, and `exit;` to quit.
Each ordinary run starts a fresh session; WORK data live in memory.

For native data, assign the library without rewriting your SAS program:

```sh
saslite-custom --lib TMP1=/path/to/data program.sas
```

Or put `.saslite.json` beside `program.sas`:

```json
{"libraries": {"TMP1": "./data"}}
```

Then `proc print data=tmp1.food; run;` reads `data/food.sas7bdat` directly.
Relative configuration paths resolve against that configuration's directory.
CLI library assignments override configuration; both override matching LIBNAME
paths in SAS source. See [usage and limits](docs/USAGE.md) for storage behavior.

## Python API

```python
from saslite import SasInterpreter

sas = SasInterpreter()
result = sas.execute("data demo; x=42; doubled=x*2; run;")
if not result.success:
    raise RuntimeError(result.error)
print(sas.get_dataset("WORK", "DEMO"))
```

Run `python examples/python_api.py` or read `saslite-custom help python`.

## Limitations and validation

SASLite implements a subset of SAS, not a drop-in replacement. Some accepted
syntax is only partially implemented. TTEST currently provides normal,
two-sided analyses; TTEST/NPAR1WAY do not implement BY, WEIGHT, FREQ or exact
tests. FP uses an asymptotic normal approximation and assumes within-group
symmetry. Terminal plots approximate SAS ODS charts at text resolution.
Native `.sas7bdat` **writing is unsupported**; persistent writes use real XPT
files and cannot overwrite a native member of the same name.

Regression tests compare numerical results with published reference examples,
SciPy, and independent placement calculations, and check output and help
behavior. No SAS runtime was used for validation. The optional original-course
integration checks need local data excluded from this repository; skips are
reported explicitly. See [testing and releases](CONTRIBUTING.md).

## Develop

```sh
python -m pip install -e '.[dev]'
python -m unittest discover -s saslite_custom -p 'test_*.py' -v
python saslite_custom/check_help.py
python saslite_custom/build_wheel.py
python -m twine check dist/*
```

The reviewed builder produces both a wheel and source archive in `dist/`.
After source edits, review the rendered help and pass tests before recording
a new review. Do not refresh fingerprints just to silence a failed build.

| Path | Contents |
| --- | --- |
| `src/saslite/` | Complete interpreter, grammar, CLI, GUI assets and packaged help |
| `saslite_custom/` | Regression tests, review guard, release builder and chart contract |
| `examples/` | Standalone SAS and Python examples |
| `docs/` | Usage, statistical conventions and project provenance |
| `.github/workflows/ci.yml` | Python 3.10/3.13 tests, builds and wheel installation smoke checks |

Read [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md),
[provenance](docs/PROVENANCE.md), and the preserved upstream [MIT license](LICENSE).
