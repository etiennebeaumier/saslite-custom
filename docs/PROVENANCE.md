# Provenance and scope

This repository was extracted from the locally maintained SASLite
`0.4.1+custom.5` installation. Its base distribution is SASLite 0.4.1, authored
by SASLite contributors under the MIT license preserved verbatim in LICENSE.
The source snapshot contains the whole installed Python package, including its
parser grammar and GUI assets; it does not depend on a pre-existing customized
virtual environment or the historical TTEST patch.

The custom.1–custom.5 history is summarized in CHANGELOG.md. Those releases were
local modifications; this repository starts a new Git history at custom.6 and
does not claim to reproduce upstream commit history or its original test suite.
The tests here are the custom regression suite plus repository packaging checks.

Original course assignments, datasets, rendered course outputs, backups and old
wheels are intentionally excluded. Standalone examples contain synthetic inline
data. Reference statistical examples are identified in the test comments and
usage documentation. Optional tests can use privately held original fixtures.

The package keeps the distribution name `saslite` for compatibility with its
existing Python API. Install this repository or its release wheel explicitly;
the upstream PyPI package is a separate release stream. Neither this project
nor SASLite is affiliated with or endorsed by SAS Institute Inc.
