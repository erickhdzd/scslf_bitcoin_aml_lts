# Reproducibility Guide

This guide describes how to reproduce the Bitcoin AML artifact from the raw Elliptic transaction dataset.

## 1. Start from a clean environment in Linux

From the repository root:

```bash
sudo apt update
sudo apt install -y python3.15 python3.15-venv
python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/check_environment.py
```

## 2. Start from a clean environment in Windows

From the repository root:

```bash
# First, install Python 3.15 and add it to PATH.
py -3.15 -m venv .venv
.venv\Scripts\Activate.ps1 # for PowerShell
.venv\Scripts\activate.bat # for cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/check_environment.py
```

The pinned Python dependencies are defined by `requirements.txt`.

## 3. Obtain the Elliptic dataset

The artifact does not redistribute the raw Elliptic data.

Obtain the classic Elliptic Bitcoin transaction dataset from its authorized distribution. The original dataset is commonly distributed with these filenames:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

The artifact expects:

```text
data/raw/txs_features.csv
data/raw/txs_classes.csv
data/raw/txs_edgelist.csv
```

Create the directory and copy or rename the files:

```bash
mkdir -p data/raw
mv data/raw/elliptic_txs_features.csv data/raw/txs_features.csv
mv data/raw/elliptic_txs_classes.csv data/raw/txs_classes.csv
mv data/raw/elliptic_txs_edgelist.csv data/raw/txs_edgelist.csv
```

If the files were downloaded elsewhere, copy them instead of using `mv`.

## 4. Inspect the raw data

```bash
python src/inspect_data.py
```

This stage performs structural and data-quality checks on the raw dataset.

## 5. Recreate the chronological processed splits

```bash
python src/preprocess.py
```

The preprocessing implementation fits learned scaling operations on the training partition and applies the fitted transformation to validation and test data.

Expected processed split sizes are:

```text
train      29,894
validation  7,829
test        8,841
```

The class prevalence changes across chronological partitions; this is intentional and is part of the temporal evaluation protocol.

## 6. Build structural Transformer sequences

```bash
python src/build_sequences.py --overwrite
```

The default maximum sequence length is 16. The target transaction is included in the sequence. The Elliptic graph contains edges between nodes in the same temporal snapshot, so same-time directed neighbors remain enabled by default.

To inspect the complete interface:

```bash
python src/build_sequences.py --help
```

## 7. Build the causal GraphSAGE graph

```bash
python src/prepare_graphsage_data.py
```

This stage constructs the GraphSAGE representation and retains unknown/unlabeled transactions as structural message-passing context.

## 8. Train the target-only MLP

```bash
python src/train_mlp.py --device auto
```

## 9. Train the Transformer

```bash
python src/train_transformer.py --device auto
```

## 10. Train the Hybrid Transformer

```bash
python src/train_hybrid_transformer.py --device auto
```

## 11. Train GraphSAGE

```bash
python src/train_graphsage.py --device auto
```

The primary runs use seeds:

```text
42 43 44 45 46
```

These are also the defaults of the primary training scripts.

## 12. Export GraphSAGE embeddings

After GraphSAGE training completes:

```bash
python src/export_graphsage_embeddings.py --device auto
```

The exported representations are generated seed by seed from the corresponding GraphSAGE checkpoint. Embeddings from different seeds must not be mixed.

## 13. Train Transformer + GraphSAGE

```bash
python src/train_transformer_graphsage.py --device auto
```

This stage combines Transformer representations with the seed-matched GraphSAGE embeddings.

## 14. Evaluate late fusion

```bash
python src/evaluate_late_fusion.py --device auto --alpha-step 0.05
```

Late fusion is performed on logits. The global fusion weight is selected using validation data only.

For the stored reference experiment, the selected weights are:

```text
alpha_hybrid    = 0.60
alpha_graphsage = 0.40
```

The selected candidate has mean validation F1:

```text
0.8461237810650355
```

The neighboring candidate `alpha_hybrid = 0.65` was evaluated but has a slightly lower mean validation F1 and therefore is not selected.

## 15. Run paired statistical analysis

```bash
python src/analyze_late_fusion_statistics.py
```

The default statistical analysis uses 100,000 bootstrap resamples. The stored analysis includes paired Wilcoxon, bootstrap intervals, exact sign-flip analysis, effect sizes, and win/tie/loss summaries.

## 16. Regenerate paper assets

```bash
python src/generate_paper_assets.py
```

This stage regenerates publication tables, figures, consolidated result files, and consistency reports from the experiment outputs.

To regenerate textual/tabular outputs without figures:

```bash
python src/generate_paper_assets.py --no-figures
```

## 17. Verify the stored artifact

Run:

```bash
python src/verify_artifact.py
```

The verifier checks:

- required result summaries;
- seeds 42–46;
- per-model aggregate F1, MCC, ROC-AUC, and PR-AUC;
- late-fusion selected alpha;
- late-fusion validation-selection value;
- consistency of the alpha-search maximum;
- publication/package check status;
- residual identifying paths or email addresses in the audited text artifacts.

A clean reference artifact should end with:

```text
Artifact verification: PASS
```

## 18. Reference aggregate results

The verifier uses the full-precision stored summaries. Rounded values useful for human comparison are:

| Model | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| MLP | 0.5107 | 0.5069 | 0.8501 | 0.4497 |
| Transformer | 0.5452 | 0.5523 | 0.8401 | 0.5154 |
| Hybrid Transformer | 0.5723 | 0.5875 | 0.8374 | 0.5276 |
| GraphSAGE | 0.4452 | 0.4353 | 0.8339 | 0.4441 |
| Transformer + GraphSAGE | 0.4808 | 0.4769 | 0.8386 | 0.4483 |
| Late Fusion | 0.5686 | 0.5880 | 0.8509 | 0.5376 |

These are means across the five primary seeds.

## 19. Reproducibility expectations

GPU training is not guaranteed to be bitwise identical across hardware, CUDA, driver, and PyTorch versions. A valid independent reproduction should preserve:

- the chronological data protocol;
- the same five seeds;
- the same model configurations;
- validation-only checkpoint/threshold selection;
- validation-only global-alpha selection;
- frozen test evaluation;
- comparable aggregate metrics within expected stochastic numerical variation.

If exact stored results are being evaluated rather than retraining, `src/verify_artifact.py` provides the deterministic integrity check.
