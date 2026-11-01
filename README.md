# Anonymous Bitcoin AML Artifact

This repository contains the code, experiment outputs, and reproducibility material for the anonymous submission on temporally realistic Bitcoin anti-money-laundering classification with MLP, Transformer, Hybrid Transformer, GraphSAGE, Transformer + GraphSAGE, and validation-selected late fusion.

## Artifact scope

The artifact is designed to support three levels of evaluation:

1. **Result verification** — verify the stored experiment summaries, seeds, late-fusion selection, and publication checks without retraining.
2. **Pipeline inspection** — inspect preprocessing, temporal splitting, sequence construction, graph construction, model training, validation-only model selection, test evaluation, statistical analysis, and paper-asset generation.
3. **Full reproduction** — starting from the Elliptic transaction dataset, regenerate processed data, train the evaluated models, evaluate late fusion, run statistical analysis, and regenerate paper assets.

The original Elliptic dataset is **not redistributed** in this repository. Obtain the classic Elliptic Bitcoin transaction dataset from its authorized distribution and place the three CSV files described below in `data/raw/`.

## Expected repository layout

```text
.
├── data/
│   ├── raw/
│   │   ├── txs_features.csv
│   │   ├── txs_classes.csv
│   │   └── txs_edgelist.csv
│   └── processed/
├── results/
├── src/
├── requirements.txt
├── README.md
└── REPRODUCIBILITY.md
```

If the downloaded files use the original `elliptic_` prefix, rename them as follows:

```bash
mv data/raw/elliptic_txs_features.csv data/raw/txs_features.csv
mv data/raw/elliptic_txs_classes.csv data/raw/txs_classes.csv
mv data/raw/elliptic_txs_edgelist.csv data/raw/txs_edgelist.csv
```

## Software environment

The reference environment uses Python 3.14 and the pinned packages in `requirements.txt`.

Create an isolated environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the environment:

```bash
python src/check_environment.py
```

For GPU execution, PyTorch must detect a CUDA-capable NVIDIA GPU. The training scripts also support `--device cpu`, but full CPU reproduction can be substantially slower.

## Quick verification

If the repository includes the stored experiment results, run:

```bash
python src/verify_artifact.py
```

A successful verification ends with:

```text
Artifact verification: PASS
```

This check does **not** retrain the models. It validates the stored scientific evidence and the main values reported by the artifact.

## Full reproduction

Detailed commands and expected intermediate products are provided in `REPRODUCIBILITY.md`.

The high-level order is:

```text
raw Elliptic data
        ↓
dataset inspection
        ↓
chronological preprocessing
        ↓
structural sequence construction
        ↓
causal GraphSAGE graph construction
        ↓
MLP / Transformer / Hybrid Transformer / GraphSAGE training
        ↓
GraphSAGE embedding export
        ↓
Transformer + GraphSAGE training
        ↓
validation-selected late fusion
        ↓
paired statistical analysis
        ↓
paper tables, figures, and consistency checks
```

## Experimental protocol

The primary neural experiments use seeds:

```text
42, 43, 44, 45, 46
```

Model selection and decision-threshold selection are performed on validation data. The test set is evaluated only after the corresponding selection is frozen.

Late fusion evaluates a validation grid with step `0.05`. The stored experiment selects:

```text
Hybrid Transformer weight : 0.60
GraphSAGE weight           : 0.40
```

The selection criterion is mean validation F1, with the tie-breaking rules recorded in the experiment metadata.

## Dataset handling

The artifact expects the classic Elliptic transaction graph with the three original components:

- transaction features;
- transaction class labels;
- transaction-to-transaction directed edges.

The raw dataset is intentionally excluded from version control. Processed data are regenerated locally from the raw files.

Unknown/unlabeled transactions are retained as structural context in the GraphSAGE graph. The Transformer sequence-construction implementation uses the labeled train/validation/test transaction population for its sequence index.

## Anonymity and privacy

The public artifact must not contain author names, affiliations, email addresses, usernames, hostnames, personal filesystem paths, or repository history that identifies the authors.

Before publication, run:

```bash
python src/sanitize_artifact_metadata.py --verify-only
```

The verification command should report no identifying user-directory paths or email addresses.

## Integrity notes

Exact bitwise equality across different GPUs, CUDA releases, PyTorch builds, and hardware is not guaranteed. Reproduction should instead verify the experimental protocol, the validation-only selection procedure, and numerical agreement within ordinary floating-point/training variability.

Do not commit `.venv/`, `.idea/`, `__pycache__/`, raw datasets, or local audit archives.


## CSCI 2026 preliminary extension
See `CSCI_ACTION_PLAN.md`. `src/csci_extension.py` provides temporal diagnostics and local evidence retrieval. It does not replace validation-calibrated metrics or implement a full generative RAG system.
