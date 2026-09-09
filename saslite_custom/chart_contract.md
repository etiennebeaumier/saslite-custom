# Terminal TTEST chart contract

Surface: plain-text monospaced terminal, no browser, image protocol, color,
network dependency, or automatic file opening. Default 80 columns, adaptive
to terminal width, explicit width override (40–160). Text also survives log
capture. Each panel includes axes, sample counts, labels and a symbol legend.

Question: how do the observed distributions differ, are there outliers or
departures from normality, and how uncertain are the estimated means?
Titles describe the data; graphs do not assert significance or normality.

Charts: comparative histograms (percent within group, common bins and axes)
with fitted normal and Gaussian kernel densities; horizontal schematic boxes
(SAS definition 5 quartiles, 1.5 IQR whiskers and outliers); normal Q–Q scatter
(Blom plotting positions and sample mean/SD reference); mean confidence
intervals with the specified H0 reference. Paired designs use complete-pair
differences plus profiles and agreement scatter with equality reference.

Data sufficiency: use all nonmissing analysis observations, require n >= 2.
Constant samples keep empirical marks; omit undefined densities and say so.
Small samples are labeled with n and a short shape-assessment note. No invented
observations, jitter or resampling. Terminal cell overlaps are identified.
Plot binning, KDE bandwidth (Scott), and text rasterization are documented
approximations, not pixel-identical SAS ODS output.

Validation: preserve existing inference tests, validate quartiles/outliers and
histogram denominators, check missing and paired observations, rendering at
40/80/120 columns, graph-selection controls, suppressed plots and constant
data. Inspect real course output and paired examples in the terminal.

Reference: https://support.sas.com/documentation/onlinedoc/stat/current/ttest.pdf

NPAR1WAY extension: default MWW box plots use pooled Wilcoxon rank scores;
FP box plots use each observation's cross-group placement (half credit for
ties). Plot titles and axes explicitly identify these scores. Counts and
filtering are identical to the corresponding test tables. Raw-data boxes
remain an explicit local BOXPLOT choice. All use the common terminal renderer
and shared axes, with no color dependency.
