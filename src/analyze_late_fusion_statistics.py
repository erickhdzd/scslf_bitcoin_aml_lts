from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

METRICS = (
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "mcc",
)


def args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--late-fusion-dir", type=Path, default=Path("results/late_fusion"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--bootstrap-resamples", type=int, default=100000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--confidence-level", type=float, default=0.95)
    return p.parse_args()


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_std(x):
    return float(np.mean(x)), float(np.std(x, ddof=1) if len(x) > 1 else 0)


def cohens_dz(d):
    s = float(np.std(d, ddof=1))
    return float(np.mean(d) / s) if s else (math.inf if np.mean(d) > 0 else 0.0)


def rank_biserial(d):
    d = d[~np.isclose(d, 0)]
    if not len(d):
        return 0.0
    a = np.abs(d)
    order = np.argsort(a)
    r = np.empty_like(a, float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and np.isclose(a[order[j]], a[order[i]]):
            j += 1
        r[order[i:j]] = (i + 1 + j) / 2
        i = j
    pos = float(r[d > 0].sum())
    neg = float(r[d < 0].sum())
    return (pos - neg) / (pos + neg)


def signflip_p(d):
    obs = abs(float(np.mean(d)))
    vals = []
    for s in product((-1.0, 1.0), repeat=len(d)):
        vals.append(abs(float(np.mean(d * np.array(s)))))
    return float(np.mean(np.array(vals) >= obs - 1e-15))


def boot_ci(d, n, cl, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    a = 1 - cl
    return tuple(map(float, np.quantile(m, [a / 2, 1 - a / 2])))


def analyze(metric, fusion, hybrid, n, cl, seed):
    d = fusion - hybrid
    fm, fs = mean_std(fusion)
    hm, hs = mean_std(hybrid)
    dm, ds = mean_std(d)
    try:
        w = wilcoxon(
            fusion, hybrid, alternative="two-sided", zero_method="wilcox", method="auto"
        )
        ws, wp = float(w.statistic), float(w.pvalue)
    except ValueError:
        ws, wp = math.nan, math.nan
    lo, hi = boot_ci(d, n, cl, seed)
    return {
        "metric": metric,
        "n_pairs": len(d),
        "fusion_mean": fm,
        "fusion_std": fs,
        "hybrid_mean": hm,
        "hybrid_std": hs,
        "mean_difference": dm,
        "difference_std": ds,
        "relative_change_percent": 100 * dm / hm if hm else math.nan,
        "bootstrap_ci_level": cl,
        "bootstrap_ci_lower": lo,
        "bootstrap_ci_upper": hi,
        "wilcoxon_statistic": ws,
        "wilcoxon_p_value": wp,
        "sign_flip_permutation_p_value": signflip_p(d),
        "cohens_dz": cohens_dz(d),
        "rank_biserial_correlation": rank_biserial(d),
        "wins": int((d > 0).sum()),
        "ties": int(np.isclose(d, 0).sum()),
        "losses": int((d < 0).sum()),
        "differences_by_seed": json.dumps(d.tolist()),
    }


def plot_alpha(rows, metric, path):
    x = np.array([float(r["alpha_hybrid"]) for r in rows])
    y = np.array([float(r[f"mean_validation_{metric}"]) for r in rows])
    s = np.array([float(r[f"std_validation_{metric}"]) for r in rows])
    o = np.argsort(x)
    x, y, s = x[o], y[o], s[o]
    b = int(np.argmax(y))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y - s, y + s, alpha=0.18)
    ax.scatter([x[b]], [y[b]], s=80, zorder=3)
    ax.axvline(x[b], ls="--", lw=1)
    ax.set_xlabel("Hybrid Transformer weight (α)")
    ax.set_ylabel("Mean validation " + metric.replace("_", " ").upper())
    ax.set_title("Global α ablation: " + metric.replace("_", " ").upper())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    a = args()
    base = a.late_fusion_dir.resolve()
    out = a.output_dir.resolve() if a.output_dir else base / "statistical_analysis"
    out.mkdir(parents=True, exist_ok=True)
    tests = read_csv(base / "test_metrics_per_seed.csv")
    alphas = read_csv(base / "global_alpha_search.csv")
    stats = []
    for i, m in enumerate(METRICS):
        f = np.array([float(r[m]) for r in tests])
        h = np.array([float(r["hybrid_only_" + m]) for r in tests])
        stats.append(
            analyze(
                m, f, h, a.bootstrap_resamples, a.confidence_level, a.bootstrap_seed + i
            )
        )
    write_csv(out / "paired_statistics.csv", stats)
    (out / "paired_statistics.json").write_text(
        json.dumps(
            {
                "comparison": "Late Fusion minus Hybrid Transformer",
                "bootstrap_resamples": a.bootstrap_resamples,
                "confidence_level": a.confidence_level,
                "important_note": "With five paired seeds, the smallest possible exact two-sided sign-flip p-value is 0.0625 when all nonzero differences point in the same direction.",
                "results": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(out / "alpha_ablation_table.csv", alphas)
    for m in ("f1", "mcc", "roc_auc", "pr_auc"):
        plot_alpha(alphas, m, out / f"alpha_vs_{m}.png")
    models: list[dict[str, str | float]] = []

    for name, prefix in (
        ("Hybrid Transformer", "hybrid_only_"),
        ("Late Fusion", ""),
    ):
        row: dict[str, str | float] = {
            "model": name,
        }

        for metric in METRICS:
            values = np.array(
                [float(result[prefix + metric]) for result in tests],
                dtype=np.float64,
            )

            mean_value, std_value = mean_std(values)

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = f"{mean_value:.4f} ± {std_value:.4f}"

        models.append(row)
    write_csv(out / "model_comparison_table.csv", models)
    lines = [
        "=" * 88,
        "LATE FUSION: PAIRED STATISTICAL ANALYSIS",
        "=" * 88,
        "",
        f"Paired seeds: {len(tests)}",
        f"Bootstrap: {a.bootstrap_resamples:,} resamples, {a.confidence_level:.1%} CI",
        "",
        "IMPORTANT: n=5 has low inferential power; exact two-sided tests cannot achieve p<0.05 even when all five paired differences have the same sign.",
        "",
    ]
    for r in stats:
        lines += [
            f"{r['metric']}: fusion {r['fusion_mean']:.4f} ± {r['fusion_std']:.4f}; hybrid {r['hybrid_mean']:.4f} ± {r['hybrid_std']:.4f}; Δ {r['mean_difference']:+.6f} [{r['bootstrap_ci_lower']:+.6f}, {r['bootstrap_ci_upper']:+.6f}]; Wilcoxon p={r['wilcoxon_p_value']:.6f}; sign-flip p={r['sign_flip_permutation_p_value']:.6f}; dz={r['cohens_dz']:+.4f}; wins/ties/losses={r['wins']}/{r['ties']}/{r['losses']}"
        ]
    (out / "analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Results saved in:", out)


if __name__ == "__main__":
    main()
