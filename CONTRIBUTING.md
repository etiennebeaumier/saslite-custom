# Contributing and releasing

## Development setup

Create a Python 3.10+ virtual environment, activate it, and run
`python -m pip install -e '.[dev]'`. The editable install imports `src/saslite`
and gives you the development build tools. Core tests use unittest, pandas, NumPy, SciPy, pyreadstat and Matplotlib Agg.
Install Flask (`python -m pip install flask`) to include the lightweight GUI
execution-JSON regression; CI does so. No browser, desktop GUI or SAS runtime is required.

```sh
python -m unittest discover -s saslite_custom -p 'test_*.py' -v
saslite-custom examples/ttest.sas
saslite-custom examples/rank_tests.sas
saslite-custom examples/paired.sas
python examples/python_api.py
saslite-custom examples/coursework.sas
saslite-custom examples/graphics.sas
```

Tests cover statistical reference values, missing/tied/degenerate samples,
plot controls and dimensions, config failures, XPT persistence, native-write
protection, help navigation and review invalidation. Native read dispatch also
has synthetic mocked-reader tests. Actual native decoding is exercised by the
optional course tests, not by those mocked tests.

Seven original-course tests are skipped unless `SASLITE_COURSE_FIXTURES` points
to a directory containing `comptant.sas7bdat`, `food.sas7bdat`, both original
`reponse exercice 2 chapitre 3(1).sas` and `(2).sas` scripts, and `.saslite.json`.
That configuration must map TMP1 to the fixture directory. These checks also
verify original files remain unchanged. Do not commit course datasets or scripts.

```sh
SASLITE_COURSE_FIXTURES=/path/to/local/course python -m unittest discover -s saslite_custom -p 'test_*.py' -v
```

## Help and release workflow

1. Modify source in `src/saslite`; update affected topics and runnable examples
   in `src/saslite/cli/help_data.json`, tests, README and changelog. Installation,
   CLI, syntax and limitation changes all require matching documentation.
2. Inspect `python -m saslite help` and every affected topic in a terminal.
   Run the full regression suite and relevant examples. Internal changes still
   require this review. The fingerprint guard verifies review currency, not
   semantic completeness.
3. Run `python saslite_custom/check_help.py --record` only after that review and
   successful tests. Commit `saslite_custom/help_review.json` with the change.
4. Keep release versions aligned in `pyproject.toml`, `src/saslite/__init__.py`,
   README, changelog and installation examples. Run
   `python saslite_custom/build_wheel.py`; it rejects a stale review before
   invoking the standard build backend. Use this wrapper for release artifacts.
5. Run `python -m twine check dist/*`. Install the wheel in a fresh environment,
   change to a directory outside this checkout, and verify `saslite-custom help`,
   a SAS example, and the Python API. Confirm grammar, help and GUI assets are
   present in the wheel. CI repeats the core checks on Python 3.10 and 3.13.
6. Install the tested wheel into the maintained project environment and any
   existing shared SASLite environment. For the legacy shared layout, run
   `~/.local/share/saslite-custom/venv/bin/python -m pip install /path/to/wheel.whl`
   and `python -m pip check` with that interpreter. Its legacy shell launchers
   continue to use that environment. A normal pip installation does not create
   the old `saslite-python` launcher.
7. Commit, tag `v<VERSION>`, push, and attach the wheel and source archive to the
   matching GitHub release. Include validation results and limitations in the
   release notes. This repository does not automatically publish to PyPI.

For bug reports, include the package version, Python/OS version, a minimal SAS
example with synthetic data, expected behavior and actual output. Add regression
coverage for fixes. Avoid changing statistical conventions without documenting
and checking them against an independent reference.


For custom.7, review representative PNGs from examples/graphics.sas and both
statistical procedures, including one-sided intervals. PNG tests use temporary
directories. The example's default graphs/ folder is ignored by Git. Do not add
private datasets to make a chart fixture pass. CLI PNG smoke checks run against
the built wheel outside the checkout in both CI Python versions.
