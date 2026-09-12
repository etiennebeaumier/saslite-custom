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
them and local `BOXPLOT` requests raw-data boxes. BY groups, DESCENDING and NOTSORTED are supported; WEIGHT/FREQ and exact tests
remain unsupported for TTEST and NPAR1WAY.

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


## Coursework summaries and grouped inference

MEANS/SUMMARY use numeric variables, excluding missing observations separately
for each analysis variable. NMISS complements N. OUTPUT names follow VAR order;
a partial list saves only the corresponding first variables. With no explicit
names, analysis-variable names are reused; AUTONAME generates VARIABLE_STAT
names. Explicit collisions fail. Multiple OUTPUT statements and named variable
subsets such as MEAN(age)=mean_age are supported. The local PROC-level OUT=
extension now follows default OUTPUT semantics.

CLASS is evaluated independently within every BY group. _TYPE_ is a bitmask in
CLASS order (first variable is the highest bit); _FREQ_ is the number of included
rows, not the variable-specific nonmissing count. Default output includes all
observed class combinations plus the overall summary; NWAY keeps the full class
combination only. MISSING includes blank/null class values, otherwise rows
missing any class variable are excluded. BY missing values form ordinary groups.
When no output statistics are named, _STAT_ identifies N, MIN, MAX, MEAN, STD rows.
SUMMARY without VAR counts observations; it suppresses display unless PRINT is
requested. These defaults and schemas correct custom.6 behavior and can require
changes to downstream scripts that relied on its incorrect output.

BY input must be sorted in the requested variable order and direction. Use
PROC SORT first, DESCENDING before a descending variable, or trailing NOTSORTED
for contiguous groups. Noncontiguous repeated keys remain separate with
NOTSORTED. Grouping uses unformatted values. Statistical group failures identify
the key, permit other groups to run, and make the aggregate result unsuccessful.

TTEST SIDES=L uses the Student-t CDF and an upper finite confidence bound;
SIDES=U uses its survival function and a lower finite confidence bound. The
other bound is infinite. Both use the 1-alpha quantile, whereas SIDES=2 uses
1-alpha/2. SD confidence limits and Folded F are unchanged. Signed H0 is supported.
Paired analyses continue to use complete pairs. See examples/coursework.sas.

## PNG files and the initial ODS subset

Explicit `ODS GRAPHICS ON` enables PNG export as well as existing text plots.
Without ON, existing statistical programs remain text-only; SGPLOT reports that
PNG output is disabled. `ODS GRAPHICS OFF` suppresses all graphics in SASLite,
including SGPLOT. `--no-plots` is an overriding CLI control. NOPRINT and PLOTS=NONE
apply to both renderers for TTEST/NPAR1WAY.

```sas
ods listing gpath="./figures";
ods graphics on / imagename="intro" imagefmt=png;
proc sgplot data=intro_ex1; histogram age; run;
proc sgplot data=intro_ex1; hbox age; run;
proc sgplot data=intro_ex1; scatter x=age y=n_visite; run;
ods graphics off;
```

The synthetic examples/graphics.sas creates the input data and three images.
Default output is graphs/ beside the top-level program. Relative GPATH resolves
there when configured; includes inherit that same base. Inline/REPL/GUI code
without a filename uses the working directory. Options-only ODS GRAPHICS / ...;
does not enable or disable export. Only GPATH, IMAGENAME and IMAGEFMT=PNG are
implemented, without full ODS destinations or reporting. SGPLOT accepts exactly
one basic plot statement per RUN, without options, BY/WHERE or overlays.

Images are 1200 by 800, created with Matplotlib Agg without a display server.
Axis labels use available dataset labels. Each selected TTEST panel type or
NPAR1WAY raw/rank/placement box panel gets one image per analysis and BY group.
The existing paired agreement plot remains a first-versus-second measurement
plot with equality and mean-difference lines. One-sided interval plots use arrows.
Numerical helpers are shared with text; PNG histograms use nominal width 80
binning independent of terminal width. Rendering and automatic bins can differ
from SAS ODS. See saslite_custom/chart_contract.md.

Generated paths are logged and returned through StepResult.image_paths and the
aggregate RunSummary.image_paths, also present in GUI execution JSON. Filenames
identify data, plot and grouping unless overridden by IMAGENAME. Existing images
are never overwritten: numeric suffixes choose the next unused name. Rendering
finishes before a complete PNG is published. File failures mark the step failed;
previously saved images remain available. Image writing never changes input data.
