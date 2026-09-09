# SASLite Custom maintenance

This standalone repository owns its source in `src/saslite`, not site-packages.

- Keep `saslite-custom help` current. Any behavior, syntax, CLI, installation or
  limitation change must update relevant topics and runnable examples in
  `src/saslite/cli/help_data.json`, regression tests and README.md.
- Inspect rendered help and run the regression suite before recording a review.
  Internal changes without user-visible effects still require a help review.
- Run `python saslite_custom/check_help.py --record` only after that review.
  Never refresh a fingerprint simply to bypass a failed build check.
- Build release artifacts with `python saslite_custom/build_wheel.py`, which
  rejects stale help reviews. Keep help and grammar resources in the wheel.
- Install tested releases into maintained project and existing shared SASLite
  environments when completing an authorized custom-library update.
- Keep course files, private datasets, virtual environments and credentials out
  of this repository. Optional local integration fixtures are documented in
  CONTRIBUTING.md.
