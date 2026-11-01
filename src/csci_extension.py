"""CSCI extension: auditable temporal evaluation and retrieval-grounded explanations.

No test-label access is used for model selection. This module only evaluates
already frozen predictions and retrieves from a user-supplied document corpus.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

read_csv_typed = cast(Callable[..., pd.DataFrame], pd.read_csv)


def evaluate(path: Path, out: Path) -> None:
    df = read_csv_typed(str(path))
    print("Columns:", list(df.columns))
    aliases = {
        "time": ("time_step", "target_time", "time", "timestep"),
        "label": ("y_true", "label", "true_label", "y"),
        "score": (
            "y_score",
            "probability",
            "score",
            "pred_prob",
            "predicted_probability",
        ),
    }
    found = {}
    for key, opts in aliases.items():
        found[key] = next((c for c in opts if c in df.columns), None)
    if any(v is None for v in found.values()):
        raise ValueError(
            f"Cannot resolve required columns: {found}; use a matching predictions CSV"
        )
    threshold_col = "threshold" if "threshold" in df.columns else None
    if threshold_col is None:
        raise ValueError(
            "Frozen validation-selected threshold is required in the predictions CSV"
        )
    if not np.isfinite(df[threshold_col].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite threshold detected")
    if df[threshold_col].nunique() != 1:
        raise ValueError(
            "Expected one frozen threshold per seed/model predictions file"
        )
    threshold = float(df[threshold_col].iloc[0])
    rows = []
    for t, part in df.groupby(found["time"], sort=True):
        y = part[found["label"]].astype(int).to_numpy()
        p = part[found["score"]].astype(float).to_numpy()
        pred = (p >= threshold).astype(int)
        rows.append(
            {
                "step": int(str(t)),
                "n": len(y),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
                "roc_auc": float(roc_auc_score(y, p))
                if len(np.unique(y)) == 2
                else None,
                "threshold": threshold,
                "f1": float(f1_score(y, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y, pred)),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)


def retrieve(corpus: Path, question: str, k: int) -> None:
    documents = []
    for file in sorted(corpus.glob("*.txt")):
        content = file.read_text(encoding="utf-8").strip()
        if content:
            documents.append((file.name, content))
    if not documents:
        raise ValueError("Provide at least one nonempty .txt source in the corpus")
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform([x[1] for x in documents] + [question])
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    evidence = [
        {
            "source": documents[i][0],
            "similarity": round(float(sims[i]), 5),
            "excerpt": documents[i][1][:700],
        }
        for i in np.argsort(-sims)[:k]
        if sims[i] > 0
    ]
    print(
        json.dumps(
            {
                "query": question,
                "retrieved_evidence": evidence,
                "explanation": "Evidence retrieval only; no unsupported claim or automated AML judgment is generated.",
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("temporal")
    e.add_argument("--predictions", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--corpus", type=Path, required=True)
    r.add_argument("--query", required=True)
    r.add_argument("-k", type=int, default=3)
    a = p.parse_args()
    if a.action == "temporal":
        evaluate(a.predictions, a.output)
    else:
        retrieve(a.corpus, a.query, a.k)


if __name__ == "__main__":
    main()
