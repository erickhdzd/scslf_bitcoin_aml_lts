# Reproducibility report

This report was generated automatically by `generate_paper_assets.py`.

## Generation

- Generated at (UTC): `1970-01-01T00:00:00+00:00`
- SOURCE_DATE_EPOCH: `0`
- Project root: `.`
- Output root: `results/paper_assets_final`
- Models processed: 6
- Decimal precision: 4

## Software environment

- Python: `3.13.5`
- Python implementation: `CPython`
- NumPy: `2.3.5`
- Matplotlib: `3.10.8`
- Operating system: `Linux`
- Platform: `Linux-6.18.35-x86_64-with-glibc2.41`
- Machine architecture: `x86_64`

## Consolidated results

- File: `/mnt/data/usenix_candidate_extract/results/paper_assets_final/json/consolidated_results.json`
- Size: `16948` bytes
- SHA-256: `0e4da912bdd61ce6d7c690d5fffacb4d65c407304b34b55ae3ccaa94a7da261b`

## Evaluation configuration

- Metrics: `balanced_accuracy`, `precision`, `recall`, `f1`, `macro_f1`, `mcc`, `roc_auc`, `pr_auc`
- Main-paper metrics: `f1`, `mcc`, `roc_auc`, `pr_auc`, `balanced_accuracy`
- Supplementary metrics: `f1`, `mcc`, `roc_auc`, `pr_auc`, `balanced_accuracy`, `precision`, `recall`, `macro_f1`

## Experiment sources

| Model | n_seeds | Seeds | Source file | SHA-256 |
|---|---:|---|---|---|
| MLP | 5 | 42, 43, 44, 45, 46 | `results/mlp_target/aggregate_summary.json` | `d5516b47949341d03384a2d847ddb995e65e6e9c1bee9fb9267b401fb99416a9` |
| Transformer | 5 | 42, 43, 44, 45, 46 | `results/transformer/aggregate_summary.json` | `2a3e9660a716b6b3096ae2f1df5f1a08ece49721b58fb750605a4cbebe620ecd` |
| Hybrid Transformer | 5 | 42, 43, 44, 45, 46 | `results/hybrid_transformer/aggregate_summary.json` | `a19215c78e7f1d53e4ea62ea9cd640dec4c8fe8b44f0fb8752de4272ea5d7a5d` |
| GraphSAGE | 5 | not reported | `results/graphsage/aggregate_summary.json` | `02cd750ad25c1f7762f854e60e91b0ce81afd53dcd4022e6a4aaceb0f6a0b49f` |
| Transformer + GraphSAGE | 5 | 42, 43, 44, 45, 46 | `results/transformer_graphsage/aggregate_summary.json` | `267d889c3f13bb708e928ad3ac2b7daf6bf47c7db8e0364bafbc5db163377024` |
| Late Fusion (ours) | 5 | 42, 43, 44, 45, 46 | `results/late_fusion/aggregate_summary.json` | `9c4495807bc0de0c57094d0c58db2cf917cdf2acf334afbc3955dcb38bb8bbab` |

## Reproduction notes

- Only experiment directories explicitly declared in `MODEL_CONFIG` are processed.
- Per-seed files under directories named `seed_*` are not used as aggregate-result sources.
- The SHA-256 values can be used to verify that the input summaries and consolidated results have not changed.
