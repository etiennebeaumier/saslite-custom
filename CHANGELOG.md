# Changelog

## 0.4.1+custom.7

- Add SGPLOT histogram, horizontal box and scatter PNG output, plus PNGs for every existing TTEST/NPAR1WAY graph type.
- Add explicit ODS PNG activation, GPATH/IMAGENAME/IMAGEFMT controls, collision-free filenames and saved paths in Python/GUI results.
- Preserve DATA-step LABEL text, including French accents, for dataset metadata and graph axes.
- Add Matplotlib Agg as a core dependency; render 1200 × 800 images without a display server.
- Repair MEANS/SUMMARY multi-variable OUTPUT, partial names, variable subsets, AUTONAME, NMISS, missing handling and name validation.
- Correct BY/CLASS combinations, NWAY/MISSING, _TYPE_/_FREQ_/_STAT_ schemas and distinct SUMMARY defaults. This changes previously incorrect summary output shapes.
- Add validated BY/DESCENDING/NOTSORTED analyses for MEANS/SUMMARY/TTEST/NPAR1WAY and group-specific diagnostics.
- Add SIDES=L/U and signed H0 to TTEST; show one-sided confidence intervals in text and PNGs.
- Expose the first failed-step error in RunSummary and emit t-test warnings once.
- Add synthetic coursework examples, numerical and PNG regressions, and updated packaged help.

## 0.4.1+custom.6

- Extract the complete custom interpreter into a portable source repository.
- Add standard Python packaging and a `saslite-custom` console entry point.
- Build wheels and source archives through the reviewed-help build wrapper.
- Add installation, usage, contributor, provenance and release documentation.
- Add standalone examples and CI for Python 3.10 and 3.13.
- Keep original-course integrations optional and local; preserve core regressions.
- Update packaged installation and maintenance help for standalone development.
- Statistical behavior is unchanged from custom.5.

## 0.4.1+custom.5

Packaged usage guide with live CLI/function inventories, help navigation,
procedure coverage checks, help-review fingerprints and shared installation.

## 0.4.1+custom.4

Mann–Whitney U output, MWW alias, Fligner–Policello test, reference-class
selection, and rank-score/placement terminal plots.

## 0.4.1+custom.3

Native SAS7BDAT reading, script-adjacent library mappings, CLI path overrides,
Wilcoxon/Kruskal–Wallis support, and explicit native-write protection.

## 0.4.1+custom.2

Terminal TTEST distribution, Q–Q, interval and paired diagnostic plots.

## 0.4.1+custom.1

SAS-style TTEST descriptive statistics, confidence limits, pooled and
Satterthwaite tests, Folded F, and GUI table layout improvements.
