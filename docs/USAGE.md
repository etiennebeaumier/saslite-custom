# Usage and statistical conventions

The packaged guide is the detailed interface reference: `saslite-custom help`.
It is installed with the wheel and works independently of the repository.

## T-tests and terminal plots

See `examples/ttest.sas` for independent groups and `examples/paired.sas` for
paired measurements. Omit CLASS/PAIRED for a one-sample test. `H0=` applies to
the tested mean/difference, and `ALPHA=` controls confidence levels. Missing
observations are excluded per analysis; paired tests use complete pairs.
Undefined statistics display a dot and a warning.

Plots include histograms with normal/kernel fits, boxes, Q–Q plots and mean
confidence intervals; paired tests add profiles and agreement plots. Controls:

```sas
ods graphics on;
proc ttest data=sample plots=(summary qqplot);
    class group;
    var value;
run;
ods graphics off;
```

`SUMMARY` includes histograms, boxes and confidence intervals. Other selections
include `ALL`, `NONE`, `HISTOGRAM`, `BOX`, `QQPLOT`, `PROFILES`, `AGREEMENT` and
local `INTERVAL`. `NOPRINT` or CLI `--no-plots` avoids plot calculations.
Width can be set from 40 to 160 columns with `--plot-width`.
See the [chart contract](../saslite_custom/chart_contract.md) for quantiles,
whiskers, density fits, scales and text-rendering limitations.

## Rank tests

```sas
proc npar1way data=sample wilcoxon fp;
    class group;
    var value;
run;
```

Run this with the data in `examples/rank_tests.sas`. WILCOXON is the SAS name for
Mann–Whitney–Wilcoxon; MWW is a local alias. U is reported for the identified
group, computed as rank sum minus n(n+1)/2; it need not be the smaller U.
`CORRECT=NO` disables the default Wilcoxon continuity correction. More than two
groups produce Kruskal–Wallis results.

FP uses pairwise placements with half credit for ties and an asymptotic normal
approximation, without continuity correction. It assumes within-group symmetry
and requires exactly two groups, each with at least two nonmissing observations.
The default direction is first-listed minus second-listed class, with order
following first occurrence. `FP(REFCLASS=1)` reverses the reference; quoted
values such as `FP(REFCLASS='2')` select labels rather than numeric positions.
A zero denominator produces missing statistics and a warning.

Default NPAR1WAY boxes show Wilcoxon ranks and FP placements.
`PLOTS=(WILCOXONBOXPLOT FPBOXPLOT)` selects those panels; `PLOTS=NONE` suppresses
them and local `BOXPLOT` requests raw-data boxes. Exact tests and BY/WEIGHT/FREQ
statements are not implemented for TTEST or NPAR1WAY.

## Storage and paths

WORK is in-memory and cannot be remapped. Native `.sas7bdat` takes precedence
over same-name XPT or CSV data, with case-insensitive member lookup. Corrupt
native files fail instead of silently selecting another format. ReadStat uses
file-declared encoding, preserves SAS date/time numbers and retains available
labels/formats; pandas is a fallback reader. Full user format catalogs are not
supported.

Native writing is unavailable. Configured libraries persist real `.xpt` files
and refuse to overwrite a native member. For explicit persistent output, use
`saslite-custom --workdir ./saved --format xpt program.sas` and write a table
such as `DATA DISK.RESULT; ... RUN;`. The default native format is read-only;
`--workdir` assigns DISK, not WORK. Legacy XPT content with a `.sas7bdat`
extension remains readable, but renaming files does not create native data.

Use `--encoding latin1` or `--encoding cp1252` for older SAS source text. This
option does not override dataset encoding. Use absolute CSV paths when running
from a different working directory. `saslite-custom help libraries` explains
configuration precedence and resolution of relative paths.

## Reference checks

The suite checks the published SAS golf TTEST example, SAS NPAR1WAY reaction
times, the NIST FP example (Z = 1.55802, two-sided p = 0.11923), independent
pairwise placement calculations, and SciPy comparisons. No SAS runtime was run.

- [SAS TTEST group comparison](https://support.sas.com/documentation/cdl/en/statug/63962/HTML/default/statug_ttest_a0000000116.htm)
- [SAS NPAR1WAY example](https://support.sas.com/documentation/cdl/en/statug/67523/HTML/default/statug_npar1way_examples03.htm)
- [SAS FP conventions](https://support.sas.com/documentation/cdl/en/statug/65328/HTML/default/statug_npar1way_details20.htm)
- [NIST FP example](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/fligner.htm)
