# Paper evidence map

This document maps the planned manuscript sections, claims, tables, figures, and reproducibility records to their corresponding generated evidence.

## Abstract

Evidence sources:

- `paper/tables/Table1_MainResults.csv`: principal performance values.
- `paper/tables/Table2_FusionAblation.csv`: comparison of fusion strategies.
- `tables/Improvement_Over_Hybrid.csv`: improvements relative to the Hybrid Transformer.
- `json/consolidated_results.json`: authoritative aggregated numerical source.

## Introduction and contributions

Evidence sources:

- `README.md`: package scope, integration instructions, and reproducibility overview.
- `reproducibility/REPRODUCIBILITY.md`: experimental traceability and execution context.
- `reproducibility/reproducibility_metadata.json`: machine-readable reproducibility metadata.

Planned contribution claims:

- Comparative evaluation of individual, hybrid, early-fusion, and late-fusion models.
- Evaluation across five random seeds for every reported model.
- Reproducible and independently verifiable generation of publication assets.

## Methodology

Evidence sources:

- `supplementary/tables/TableS1_AllModels.csv`: complete set of metrics and model metadata.
- `json/consolidated_results.json`: model names, means, standard deviations, seeds, and source summaries.
- `reproducibility/reproducibility_metadata.json`: configuration and input-file provenance.

## Experimental setup

Evidence sources:

- `reproducibility/REPRODUCIBILITY.md`: software environment and reproducibility procedure.
- `reproducibility/reproducibility_metadata.json`: structured environment and source information.
- `MANIFEST.md`: generated-file inventory.
- `delivery/PACKAGE_CONTENTS.sha256`: SHA-256 hashes of packaged assets.

## Main results

Evidence sources:

- `paper/tables/Table1_MainResults.csv`: main numerical comparison.
- `paper/markdown/Table1_MainResults.md`: readable main-results table.
- `paper/latex/Table1_MainResults.tex`: manuscript-ready main-results table.
- `paper/text/Results_Draft.md`: automatically generated results narrative.
- `paper/text/Results_Draft.tex`: LaTeX results narrative.
- `markdown/results_summary.md`: concise derived results summary.

Main-result figures:

- `figures/models_f1.pdf`: F1 comparison.
- `figures/models_mcc.pdf`: MCC comparison.
- `figures/models_roc_auc.pdf`: ROC-AUC comparison.
- `figures/models_pr_auc.pdf`: PR-AUC comparison.

## Fusion ablation

Evidence sources:

- `paper/tables/Table2_FusionAblation.csv`: fusion ablation values.
- `paper/markdown/Table2_FusionAblation.md`: readable fusion-ablation table.
- `paper/latex/Table2_FusionAblation.tex`: manuscript-ready fusion-ablation table.
- `tables/Improvement_Over_Hybrid.csv`: absolute and relative improvements over the Hybrid Transformer.

Alpha-ablation evidence:

- `figures/alpha_f1.pdf`: validation F1 by alpha.
- `figures/alpha_mcc.pdf`: validation MCC by alpha.
- `figures/alpha_roc_auc.pdf`: validation ROC-AUC by alpha.
- `figures/alpha_pr_auc.pdf`: validation PR-AUC by alpha.

## Statistical analysis

Evidence sources:

- `paper/tables/Table3_StatisticalComparison.csv`: publication statistical comparison.
- `paper/latex/Table3_StatisticalComparison.tex`: manuscript-ready statistical table.
- `supplementary/tables/TableS2_StatisticalComparison.csv`: complete statistical output.
- `supplementary/latex/TableS2_StatisticalComparison.tex`: complete supplementary statistical table.

## Discussion

Evidence sources:

- `paper/tables/Table1_MainResults.csv`: comparison across all primary models.
- `paper/tables/Table2_FusionAblation.csv`: evidence for discussing fusion behavior.
- `tables/Improvement_Over_Hybrid.csv`: evidence for claims relative to the Hybrid Transformer.
- `supplementary/tables/TableS1_AllModels.csv`: precision, recall, Macro-F1, and complete metrics.

## Reproducibility and integrity

Evidence sources:

- `PUBLICATION_CHECK.txt`: structural, numerical, editorial, and dependency checks.
- `delivery/PACKAGE_CHECK.txt`: ZIP integrity and path-safety checks.
- `delivery/PACKAGE_CONTENTS.sha256`: packaged-file hash inventory.
- `delivery/ARCHIVE_SHA256.txt`: final ZIP hash.
- `reproducibility/reproducibility_metadata.json`: build timestamp, inputs, and environment metadata.

## Manuscript integration

Evidence sources:

- `paper/integration/ManuscriptAssets.tex`: main tables and figures.
- `supplementary/integration/SupplementaryAssets.tex`: supplementary tables and figures.
- `captions/table_captions.md`: table captions.
- `captions/figure_captions.md`: figure captions.

## Authoring rule

Every numerical statement in the manuscript must be traceable to `json/consolidated_results.json` or to one of the CSV tables listed in this evidence map.

Rounded display values must not be used to recompute differences, percentages, statistical results, or rankings.

The authoritative source for numerical calculations is the full-precision machine-readable output.
