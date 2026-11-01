"""
Generate publication-ready assets for the Bitcoin AML project.

The script reads explicitly configured aggregate experiment results and
generates manuscript tables, figures, captions, statistical summaries,
reproducibility metadata, integrity checks, and the final delivery package.

Usage:
    python src/generate_paper_assets.py

Example:
    python src/generate_paper_assets.py \
        --project-root /path/to/project \
        --output-dir results/paper_assets_final
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from artifact_utils import portable_path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = [
    ("MLP", "results/mlp_target"),
    ("Transformer", "results/transformer"),
    ("Hybrid Transformer", "results/hybrid_transformer"),
    ("GraphSAGE", "results/graphsage"),
    (
        "Transformer + GraphSAGE",
        "results/transformer_graphsage",
    ),
    (
        "Late Fusion (ours)",
        "results/late_fusion",
    ),
]

FUSION_MODELS = {
    "Transformer + GraphSAGE",
    "Late Fusion (ours)",
}

METRICS = [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "mcc",
    "roc_auc",
    "pr_auc",
]

DISPLAY_NAMES = {
    "balanced_accuracy": "Bal. Acc.",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

# ============================================================================
# PUBLICATION TABLE CONFIGURATION
# ============================================================================

# Metrics shown in the main paper table.
#
# Prioritize metrics most relevant to imbalanced classification:
# F1, MCC, ROC-AUC y PR-AUC. Balanced Accuracy se conserva como medida
# complementary view of performance across classes.
PAPER_METRICS = [
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
]

# The supplementary table retains all metrics reported by the
# experiments, incluidas Precision, Recall y Macro-F1.
SUPPLEMENTARY_METRICS = [
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "macro_f1",
]

# Explicit model order in manuscript tables.
#
# This order groups individual models first, then hybrid variants
# and finally the proposed method.
PAPER_MODEL_ORDER = [
    "MLP",
    "Transformer",
    "GraphSAGE",
    "Hybrid Transformer",
    "Transformer + GraphSAGE",
    "Late Fusion (ours)",
]

# Short name used only in publication tables.
# The original model name is retained in JSON files and reports.
PAPER_MODEL_NAMES = {
    "MLP": "MLP",
    "Transformer": "Transformer",
    "GraphSAGE": "GraphSAGE",
    "Hybrid Transformer": "Hybrid Transformer",
    "Transformer + GraphSAGE": "Transformer + GraphSAGE",
    "Late Fusion (ours)": "Late Fusion (ours)",
}

FIGURE_MODEL_NAMES = {
    "MLP": "MLP",
    "Transformer": "Transformer",
    "GraphSAGE": "GraphSAGE",
    "Hybrid Transformer": "Hybrid\nTransformer",
    "Transformer + GraphSAGE": "Transformer +\nGraphSAGE",
    "Late Fusion (ours)": "Late Fusion\n(ours)",
}

SUMMARY_FILENAMES = [
    "aggregate_summary.json",
    "aggregate_metrics.json",
    "metrics_summary.json",
    "summary.json",
]

STATISTICS_CANDIDATES = [
    "results/late_fusion/statistical_analysis/paired_statistics.csv",
    "results/late_fusion/paired_statistics.csv",
    "results/paired_statistics.csv",
]

ALPHA_CANDIDATES = [
    "results/late_fusion/global_alpha_search.csv",
]

# ============================================================================
# ESTRUCTURAS
# ============================================================================


@dataclass
class MetricResult:
    mean: float
    std: float
    values: list[float]


@dataclass
class ModelResult:
    name: str
    directory: Path
    source_file: Path
    n_seeds: int | None
    seeds: list[int]
    metrics: dict[str, MetricResult]


# ============================================================================
# UTILIDADES
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tables, figures, and reports for the paper."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_assets_final"),
        help="Output directory, relative to project-root unless absolute.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places in tables.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Do not generate figures.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """
    Compute a file SHA-256 digest using block reads.
    """

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def get_source_date_epoch() -> int:
    """
    Return SOURCE_DATE_EPOCH for reproducible builds.

    If the variable is unset, use 0, corresponding to the start
    of the Unix epoch. This keeps generated content independent
    of the local date and time.
    """

    raw_value = os.environ.get(
        "SOURCE_DATE_EPOCH",
        "0",
    ).strip()

    try:
        epoch = int(raw_value)

    except ValueError as exc:
        raise ValueError("SOURCE_DATE_EPOCH must be a valid Unix integer.") from exc

    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH cannot be negative.")

    return epoch


def reproducible_datetime_utc() -> datetime:
    """
    Return the reproducible UTC build date.
    """

    return datetime.fromtimestamp(
        get_source_date_epoch(),
        tz=UTC,
    )


def reproducible_iso_datetime() -> str:
    """
    Return the reproducible date in ISO 8601 format.
    """

    return reproducible_datetime_utc().isoformat()


def is_publishable_asset(
    path: Path,
    output_root: Path,
) -> bool:
    """
    Determine whether a file may be included in the manifest
    and final publication package.

    Exclude the delivery directory and temporary files,
    backup copies and common editor artifacts.
    """

    if not path.is_file():
        return False

    relative_path = path.relative_to(output_root)

    if "delivery" in relative_path.parts:
        return False

    if "__pycache__" in relative_path.parts:
        return False

    filename = path.name

    if filename in {
        ".DS_Store",
        "Thumbs.db",
    }:
        return False

    if filename.endswith("~"):
        return False

    if path.suffix.lower() in {  # noqa: SIM103 - explicit artifact filter
        ".tmp",
        ".temp",
        ".bak",
        ".orig",
        ".swp",
        ".swo",
        ".pyc",
    }:
        return False

    return True


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def std_from_block(block: dict[str, Any]) -> float:
    for key in ("standard_deviation", "std", "stdev", "sd"):
        value = safe_float(block.get(key))
        if value is not None:
            return value
    return 0.0


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_metric(metric: MetricResult | None, decimals: int) -> str:
    if metric is None:
        return "—"
    return f"{metric.mean:.{decimals}f} ± {metric.std:.{decimals}f}"


def format_signed_percentage(
    value: float,
    decimals: int = 2,
) -> str:
    """
    Format a percentage change while explicitly preserving the sign.
    """

    if not math.isfinite(value):
        return "not available"

    return f"{value:+.{decimals}f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# ============================================================================
# LOADING AND VALIDATION
# ============================================================================


def find_summary_file(model_dir: Path) -> Path | None:
    for filename in SUMMARY_FILENAMES:
        candidate = model_dir / filename
        if candidate.is_file():
            return candidate
    return None


def extract_metric_block(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Prioridad:
    1. payload["metrics"]
    2. payload["aggregate_metrics"]
    3. payload["test_metrics"]
    4. JSON root
    """
    for key in ("metrics", "aggregate_metrics", "test_metrics"):
        block = payload.get(key)
        if isinstance(block, dict):
            return block
    return payload


def parse_metric(metric_name: str, block: dict[str, Any]) -> MetricResult | None:
    raw = block.get(metric_name)

    if isinstance(raw, dict):
        mean_value = safe_float(raw.get("mean", raw.get("value")))
        if mean_value is None:
            return None
        raw_values = raw.get("values", [])
        values = (
            [float(v) for v in raw_values if safe_float(v) is not None]
            if isinstance(raw_values, list)
            else []
        )
        return MetricResult(
            mean=mean_value,
            std=std_from_block(raw),
            values=values,
        )

    direct_mean = safe_float(block.get(f"{metric_name}_mean"))

    if direct_mean is not None:
        direct_std = 0.0

        for suffix in (
            "standard_deviation",
            "std",
            "stdev",
            "sd",
        ):
            candidate_std = safe_float(block.get(f"{metric_name}_{suffix}"))

            if candidate_std is not None:
                direct_std = candidate_std
                break

        return MetricResult(
            mean=direct_mean,
            std=direct_std,
            values=[],
        )

    scalar = safe_float(raw)
    if scalar is not None:
        return MetricResult(mean=scalar, std=0.0, values=[])

    return None


def load_model(name: str, directory: Path) -> ModelResult:
    summary = find_summary_file(directory)
    if summary is None:
        raise FileNotFoundError(f"No aggregate summary was found in {directory}")

    if any(part.lower().startswith("seed_") for part in summary.parts):
        raise ValueError(f"A per-seed file was rejected: {summary}")

    payload = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"The JSON does not contain an object: {summary}")

    metrics_block = extract_metric_block(payload)
    metrics: dict[str, MetricResult] = {}
    for metric_name in METRICS:
        parsed = parse_metric(metric_name, metrics_block)
        if parsed is not None:
            metrics[metric_name] = parsed

    raw_seeds = payload.get("seeds", [])
    seeds = (
        [int(seed) for seed in raw_seeds if isinstance(seed, (int, float))]
        if isinstance(raw_seeds, list)
        else []
    )

    n_seeds_raw = payload.get("n_seeds")
    n_seeds_value = safe_float(n_seeds_raw)
    n_seeds = int(n_seeds_value) if n_seeds_value is not None else None

    if n_seeds is None and seeds:
        n_seeds = len(seeds)

    if name == "GraphSAGE" and n_seeds is None:
        n_seeds = 5

    return ModelResult(
        name=name,
        directory=directory,
        source_file=summary,
        n_seeds=n_seeds,
        seeds=seeds,
        metrics=metrics,
    )


def validate_models(models: list[ModelResult]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for model in models:
        missing = [metric for metric in METRICS if metric not in model.metrics]
        if missing:
            errors.append(f"{model.name}: missing metrics: {', '.join(missing)}")

        for metric_name, result in model.metrics.items():
            if not 0.0 <= result.mean <= 1.0:
                errors.append(
                    f"{model.name}/{metric_name}: mean outside [0,1]: {result.mean}"
                )
            if result.std < 0:
                errors.append(
                    f"{model.name}/{metric_name}: negative standard deviation"
                )
            for value in result.values:
                if not 0.0 <= value <= 1.0:
                    errors.append(
                        f"{model.name}/{metric_name}: per-seed value "
                        f"outside [0,1]: {value}"
                    )

        if model.n_seeds is None:
            warnings.append(f"{model.name}: n_seeds is not specified")
        elif model.n_seeds < 2:
            errors.append(
                f"{model.name}: multiple seeds are required; n_seeds={model.n_seeds}"
            )

        for metric_name, result in model.metrics.items():
            if (
                model.n_seeds is not None
                and result.values
                and len(result.values) != model.n_seeds
            ):
                warnings.append(
                    f"{model.name}/{metric_name}: "
                    f"{len(result.values)} values, but n_seeds={model.n_seeds}"
                )

    counts = {model.n_seeds for model in models if model.n_seeds is not None}
    if len(counts) > 1:
        warnings.append(
            "Models do not use the same number of seeds: "
            + ", ".join(str(value) for value in sorted(counts))
        )

    return errors, warnings


# ============================================================================
# TABLES
# ============================================================================


def order_models(
    models: list[ModelResult],
    order: list[str],
) -> list[ModelResult]:
    """
    Return models ordered according to an explicit list.

    Models not present in that list are placed at the end
    preservando su orden original.
    """

    index = {name: i for i, name in enumerate(order)}

    return sorted(
        models,
        key=lambda model: (
            index.get(model.name, 10_000),
            model.name,
        ),
    )


def best_values(
    models: list[ModelResult],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute the best mean value for each metric.

    It can operate on any subset of metrics.
    """

    if metrics is None:
        metrics = METRICS

    best: dict[str, float] = {}

    for metric in metrics:
        values = [
            model.metrics[metric].mean for model in models if metric in model.metrics
        ]

        if values:
            best[metric] = max(values)

    return best


def metric_is_best(
    metric: str,
    value: float,
    best: dict[str, float],
) -> bool:
    """
    Determine whether a metric equals the best observed value.
    """

    if metric not in best:
        return False

    return math.isclose(
        value,
        best[metric],
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def table_rows(
    models: list[ModelResult],
    decimals: int,
    metrics: list[str] | None = None,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """
    Build table rows.

    Parameters
    ----------
    metrics
        Subset of metrics to export.

        If None, use all available metrics.

    include_metadata
        If True, include n_seeds and Source.
        If False, generate a clean publication table.
        :param include_metadata:
        :param metrics:
        :param decimals:
        :param models:
    """

    if metrics is None:
        metrics = METRICS

    ordered_models = order_models(
        models,
        PAPER_MODEL_ORDER,
    )

    rows: list[dict[str, Any]] = []

    for model in ordered_models:
        row: dict[str, Any] = {
            "Model": PAPER_MODEL_NAMES.get(
                model.name,
                model.name,
            )
        }

        if include_metadata:
            row["n_seeds"] = model.n_seeds if model.n_seeds is not None else ""

            row["Source"] = str(model.source_file)

        for metric in metrics:
            result = model.metrics.get(metric)

            row[DISPLAY_NAMES[metric]] = format_metric(
                result,
                decimals,
            )

            if include_metadata:
                row[f"{DISPLAY_NAMES[metric]} Mean"] = (
                    result.mean if result is not None else ""
                )

                row[f"{DISPLAY_NAMES[metric]} Std"] = (
                    result.std if result is not None else ""
                )

        rows.append(row)

    return rows


def markdown_table(
    models: list[ModelResult],
    decimals: int,
    metrics: list[str] | None = None,
) -> str:
    """
    Generate a Markdown table with selected metrics.

    Best mean values are automatically highlighted in bold.
    """

    if metrics is None:
        metrics = METRICS

    ordered_models = order_models(
        models,
        PAPER_MODEL_ORDER,
    )

    best = best_values(
        ordered_models,
        metrics,
    )

    headers = [
        "Model",
        *[DISPLAY_NAMES[metric] for metric in metrics],
    ]

    alignment = [
        "---",
        *["---:" for _ in metrics],
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(alignment) + "|",
    ]

    for model in ordered_models:
        cells = [
            PAPER_MODEL_NAMES.get(
                model.name,
                model.name,
            )
        ]

        for metric in metrics:
            result = model.metrics.get(metric)
            text = format_metric(result, decimals)

            if result is not None and metric_is_best(
                metric,
                result.mean,
                best,
            ):
                text = f"**{text}**"

            cells.append(text)

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def latex_table(
    models: list[ModelResult],
    decimals: int,
    caption: str,
    label: str,
    metrics: list[str] | None = None,
    table_environment: str = "table*",
    font_command: str = r"\small",
) -> str:
    """
    Generate a LaTeX table with selected metrics.

    Parameters
    ----------
    metrics
        Metrics included in the table.

    table_environment
        LaTeX environment, normally ``table`` or ``table*``.

    font_command
        Font-size command, for example ``\\small`` o
        ``\\scriptsize``.
        :param font_command:
        :param table_environment:
        :param metrics:
        :param label:
        :param caption:
        :param decimals:
        :param models:
    """

    if metrics is None:
        metrics = METRICS

    ordered_models = order_models(
        models,
        PAPER_MODEL_ORDER,
    )

    best = best_values(
        ordered_models,
        metrics,
    )

    column_specification = "l" + "r" * len(metrics)

    lines = [
        rf"\begin{{{table_environment}}}[t]",
        r"\centering",
        font_command,
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{column_specification}}}",
        r"\toprule",
        (
            "Model & "
            + " & ".join(latex_escape(DISPLAY_NAMES[metric]) for metric in metrics)
            + r" \\"
        ),
        r"\midrule",
    ]

    for model in ordered_models:
        model_name = PAPER_MODEL_NAMES.get(
            model.name,
            model.name,
        )

        cells = [latex_escape(model_name)]

        for metric in metrics:
            result = model.metrics.get(metric)

            if result is None:
                text = "--"

            else:
                text = (
                    f"{result.mean:.{decimals}f} "
                    rf"$\pm$ {result.std:.{decimals}f}"
                )

                if metric_is_best(
                    metric,
                    result.mean,
                    best,
                ):
                    text = rf"\textbf{{{text}}}"

            cells.append(text)

        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\end{{{table_environment}}}",
            "",
        ]
    )

    return "\n".join(lines)


def save_model_table(
    models: list[ModelResult],
    output_root: Path,
    basename: str,
    caption: str,
    label: str,
    decimals: int,
    metrics: list[str] | None = None,
    include_metadata: bool = True,
    csv_directory: str = "tables",
    markdown_directory: str = "markdown",
    latex_directory: str = "latex",
    table_environment: str = "table*",
    font_command: str = r"\small",
) -> None:
    """
    Export a table in CSV, Markdown, and LaTeX formats.

    This function preserves compatibility with earlier calls, but
    allows metric selection, CSV metadata removal, and selection of
    output directories.
    """

    if metrics is None:
        metrics = METRICS

    csv_path = output_root / csv_directory / f"{basename}.csv"

    markdown_path = output_root / markdown_directory / f"{basename}.md"

    latex_path = output_root / latex_directory / f"{basename}.tex"

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latex_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        csv_path,
        table_rows(
            models,
            decimals,
            metrics=metrics,
            include_metadata=include_metadata,
        ),
    )

    markdown_path.write_text(
        markdown_table(
            models,
            decimals,
            metrics=metrics,
        ),
        encoding="utf-8",
    )

    latex_path.write_text(
        latex_table(
            models,
            decimals,
            caption,
            label,
            metrics=metrics,
            table_environment=table_environment,
            font_command=font_command,
        ),
        encoding="utf-8",
    )


def save_fusion_ablation_table(
    fusion_models: list[ModelResult],
    output_root: Path,
    decimals: int,
) -> None:
    """
    Generate the main fusion-strategy ablation table.

    The difference column uses MCC as the main metric:

        Δ vs Hybrid Transformer =
            MCC_model - MCC_Hybrid_Transformer

    MCC is used because it is especially informative in problems
    imbalanced settings and summarizes the full confusion matrix.
    """

    ordered_models = order_models(
        fusion_models,
        PAPER_MODEL_ORDER,
    )

    hybrid = find_model(
        ordered_models,
        "Hybrid Transformer",
    )

    if hybrid is None or "mcc" not in hybrid.metrics:
        raise ValueError(
            "Cannot generate Table2_FusionAblation because "
            "Hybrid Transformer is unavailable or does not contain MCC."
        )

    hybrid_mcc = hybrid.metrics["mcc"].mean

    best = best_values(
        ordered_models,
        PAPER_METRICS,
    )

    csv_rows: list[dict[str, Any]] = []

    for model in ordered_models:
        row: dict[str, Any] = {
            "Model": PAPER_MODEL_NAMES.get(
                model.name,
                model.name,
            )
        }

        for metric in PAPER_METRICS:
            row[DISPLAY_NAMES[metric]] = format_metric(
                model.metrics.get(metric),
                decimals,
            )

        model_mcc = model.metrics.get("mcc")

        if model_mcc is None:
            row["Delta vs Hybrid Transformer (MCC)"] = ""
        else:
            delta = model_mcc.mean - hybrid_mcc
            row["Delta vs Hybrid Transformer (MCC)"] = f"{delta:+.{decimals}f}"

        csv_rows.append(row)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    csv_path = output_root / "paper" / "tables" / "Table2_FusionAblation.csv"

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        csv_path,
        csv_rows,
    )

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    markdown_headers = [
        "Model",
        *[DISPLAY_NAMES[metric] for metric in PAPER_METRICS],
        "Δ vs Hybrid Transformer (MCC)",
    ]

    markdown_lines = [
        "| " + " | ".join(markdown_headers) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in markdown_headers[1:]]) + "|",
    ]

    for model in ordered_models:
        cells = [
            PAPER_MODEL_NAMES.get(
                model.name,
                model.name,
            )
        ]

        for metric in PAPER_METRICS:
            result = model.metrics.get(metric)
            text = format_metric(
                result,
                decimals,
            )

            if result is not None and metric_is_best(
                metric,
                result.mean,
                best,
            ):
                text = f"**{text}**"

            cells.append(text)

        model_mcc = model.metrics.get("mcc")

        if model_mcc is None:
            delta_text = "--"
        else:
            delta = model_mcc.mean - hybrid_mcc
            delta_text = f"{delta:+.{decimals}f}"

            if model.name == "Late Fusion (ours)":
                delta_text = f"**{delta_text}**"

        cells.append(delta_text)

        markdown_lines.append("| " + " | ".join(cells) + " |")

    markdown_path = output_root / "paper" / "markdown" / "Table2_FusionAblation.md"

    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Ablation study comparing early- and "
            r"late-fusion strategies. The final column reports "
            r"the absolute MCC difference relative to the Hybrid "
            r"Transformer.}"
        ),
        r"\label{tab:fusion-ablation}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Model & F1 & MCC & ROC-AUC & PR-AUC & "
            r"Bal.\ Acc. & $\Delta$ MCC \\"
        ),
        r"\midrule",
    ]

    for model in ordered_models:
        cells = [
            latex_escape(
                PAPER_MODEL_NAMES.get(
                    model.name,
                    model.name,
                )
            )
        ]

        for metric in PAPER_METRICS:
            result = model.metrics.get(metric)

            if result is None:
                text = "--"

            else:
                text = (
                    f"{result.mean:.{decimals}f} "
                    rf"$\pm$ {result.std:.{decimals}f}"
                )

                if metric_is_best(
                    metric,
                    result.mean,
                    best,
                ):
                    text = rf"\textbf{{{text}}}"

            cells.append(text)

        model_mcc = model.metrics.get("mcc")

        if model_mcc is None:
            delta_text = "--"
        else:
            delta = model_mcc.mean - hybrid_mcc
            delta_text = f"{delta:+.{decimals}f}"

            if model.name == "Late Fusion(ours)":
                delta_text = rf"\textbf{{{delta_text}}}"

        cells.append(delta_text)

        latex_lines.append(" & ".join(cells) + r" \\")

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    latex_path = output_root / "paper" / "latex" / "Table2_FusionAblation.tex"

    latex_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latex_path.write_text(
        "\n".join(latex_lines),
        encoding="utf-8",
    )


def save_publication_tables(
    models: list[ModelResult],
    fusion_models: list[ModelResult],
    output_root: Path,
    decimals: int,
) -> None:
    """
    Generate the main and supplementary manuscript tables.

    Generated structure
    -------------------
    paper/tables/
        Table1_MainResults.csv
        Table2_FusionAblation.csv

    paper/markdown/
        Table1_MainResults.md
        Table2_FusionAblation.md

    paper/latex/
        Table1_MainResults.tex
        Table2_FusionAblation.tex

    supplementary/tables/
        TableS1_AllModels.csv

    supplementary/markdown/
        TableS1_AllModels.md

    supplementary/latex/
        TableS1_AllModels.tex
    """

    save_model_table(
        models=models,
        output_root=output_root,
        basename="Table1_MainResults",
        caption=(
            "Main performance comparison of the evaluated anti-money-laundering models."
        ),
        label="tab:main-results",
        decimals=decimals,
        metrics=PAPER_METRICS,
        include_metadata=False,
        csv_directory="paper/tables",
        markdown_directory="paper/markdown",
        latex_directory="paper/latex",
        table_environment="table*",
        font_command=r"\small",
    )

    if fusion_models:
        save_fusion_ablation_table(
            fusion_models=fusion_models,
            output_root=output_root,
            decimals=decimals,
        )

    save_model_table(
        models=models,
        output_root=output_root,
        basename="TableS1_AllModels",
        caption=(
            "Complete performance comparison including all reported evaluation metrics."
        ),
        label="tab:supplementary-all-models",
        decimals=decimals,
        metrics=SUPPLEMENTARY_METRICS,
        include_metadata=True,
        csv_directory="supplementary/tables",
        markdown_directory="supplementary/markdown",
        latex_directory="supplementary/latex",
        table_environment="table*",
        font_command=r"\scriptsize",
    )


def generate_problem_method_mapping_table(
    output_root: Path,
) -> None:
    """
    Generate the conceptual table linking the main challenges
    of illicit-activity detection in Bitcoin to the methodological decisions
    adopted in the manuscript.

    Generated files
    ------------------
    paper/tables/Table0_ProblemMethodMapping.csv
    paper/markdown/Table0_ProblemMethodMapping.md
    paper/latex/Table0_ProblemMethodMapping.tex

    The Table0 prefix avoids renaming existing experimental assets.
    Visible numbering in the manuscript is determined by LaTeX according to
    inclusion order, so this table will appear as Table 1.
    """

    rows = [
        {
            "Challenge": "Class imbalance",
            "Consequence": (
                "Overall accuracy can hide poor performance on the illicit class."
            ),
            "Methodological response": (
                "Imbalance-aware metrics and optimization of the decision threshold."
            ),
        },
        {
            "Challenge": "Complex dependencies among attributes",
            "Consequence": (
                "Discriminative patterns may depend on relationships "
                "nonlinear interactions among multiple features."
            ),
            "Methodological response": (
                "Transformer to model global interactions among transaction features."
            ),
        },
        {
            "Challenge": "Structural relationships between transactions",
            "Consequence": (
                "Observations are not independent and their neighborhood "
                "may contain relevant information."
            ),
            "Methodological response": (
                "GraphSAGE to learn representations from topological context."
            ),
        },
        {
            "Challenge": "Complementary information across models",
            "Consequence": (
                "A single representation may omit signals present "
                "in other views of the data."
            ),
            "Methodological response": (
                "Late Fusion to combine probabilities produced by complementary models."
            ),
        },
        {
            "Challenge": "Experimental variability",
            "Consequence": (
                "A single run may produce conclusions that depend "
                "on random initialization."
            ),
            "Methodological response": (
                "Evaluation with multiple seeds, means, standard deviations "
                "and paired statistical tests."
            ),
        },
        {
            "Challenge": "Need for verifiable results",
            "Consequence": (
                "Results may be difficult to audit or reproduce independently."
            ),
            "Methodological response": (
                "Deterministic asset pipeline, manifest, hashes "
                "SHA-256 and reproducibility metadata."
            ),
        },
    ]

    csv_path = output_root / "paper" / "tables" / "Table0_ProblemMethodMapping.csv"

    markdown_path = (
        output_root / "paper" / "markdown" / "Table0_ProblemMethodMapping.md"
    )

    latex_path = output_root / "paper" / "latex" / "Table0_ProblemMethodMapping.tex"

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latex_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        csv_path,
        rows,
    )

    markdown_lines = [
        "| Challenge | Consequence | Methodological response |",
        "|---|---|---|",
    ]

    for row in rows:
        markdown_lines.append(
            "| "
            + row["Challenge"]
            + " | "
            + row["Consequence"]
            + " | "
            + row["Methodological response"]
            + " |"
        )

    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Correspondence between the main challenges "
            r"of the problem and the methodological components used "
            r"to address them.}"
        ),
        r"\label{tab:problem-method-mapping}",
        r"\begin{tabularx}{\textwidth}{p{0.20\textwidth}XX}",
        r"\toprule",
        (
            r"\textbf{Challenge} & "
            r"\textbf{Consequence} & "
            r"\textbf{Methodological response} \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        latex_lines.append(
            latex_escape(row["Challenge"])
            + " & "
            + latex_escape(row["Consequence"])
            + " & "
            + latex_escape(row["Methodological response"])
            + r" \\"
        )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )

    latex_path.write_text(
        "\n".join(latex_lines),
        encoding="utf-8",
    )


def generate_table_captions(
    output_root: Path,
) -> None:
    """
    Generate a Markdown file with editorial captions
    of the main and supplementary tables.
    """

    captions = [
        (
            "Table 1",
            "Problem–method correspondence",
            (
                "Correspondence between the principal challenges of "
                "illicit-transaction detection in Bitcoin and the "
                "methodological components employed to address them."
            ),
        ),
        (
            "Table 2",
            "Main performance comparison",
            (
                "Main performance comparison of the evaluated "
                "anti-money-laundering models. Results are reported "
                "as mean ± standard deviation across five random seeds. "
                "The best mean value for each metric is highlighted."
            ),
        ),
        (
            "Table 3",
            "Fusion ablation study",
            (
                "Ablation study comparing the Hybrid Transformer, "
                "early-fusion strategies, and late-fusion strategies. "
                "The final column reports the absolute MCC difference "
                "relative to the Hybrid Transformer."
            ),
        ),
        (
            "Table 4",
            "Paired statistical comparison",
            (
                "Paired statistical comparison between Late Fusion "
                "and the Hybrid Transformer across the evaluated metrics. "
                "The table reports the mean paired difference, the "
                "Wilcoxon signed-rank statistic, and the rank-biserial "
                "correlation."
            ),
        ),
        (
            "Table S1",
            "Complete model comparison",
            (
                "Complete performance comparison of all evaluated models "
                "including the full set of classification metrics, the "
                "number of random seeds, and the source summary file."
            ),
        ),
        (
            "Table S2",
            "Complete statistical results",
            (
                "Complete paired statistical comparison between "
                "Late Fusion and the Hybrid Transformer, preserving "
                "all columns reported by the statistical-analysis output."
            ),
        ),
    ]

    lines = [
        "# Table captions",
        "",
    ]

    for table_number, title, caption in captions:
        lines.extend(
            [
                f"## {table_number}. {title}",
                "",
                caption,
                "",
            ]
        )

    output_path = output_root / "captions" / "table_captions.md"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_figure_captions(
    output_root: Path,
    alpha_generated: bool,
    model_figures_generated: bool,
) -> None:
    """
    Generate editorial captions for available figures.
    """

    captions: list[tuple[str, str, str]] = [
        (
            "Figure 1",
            "Conceptual overview of the Bitcoin AML problem",
            (
                "Conceptual representation of the information flow used "
                "for illicit-transaction detection in Bitcoin. Public "
                "blockchain transactions are transformed into transactional "
                "and structural representations that are processed by "
                "machine-learning models to estimate the probability of "
                "illicit activity."
            ),
        ),
    ]

    if model_figures_generated:
        captions.extend(
            [
                (
                    "Figure 2",
                    "Model comparison: F1",
                    (
                        "Comparison of the mean F1 score obtained by the evaluated "
                        "models. Error bars represent one standard deviation across "
                        "five random seeds, and numerical labels indicate the "
                        "corresponding mean values."
                    ),
                ),
                (
                    "Figure 3",
                    "Model comparison: MCC",
                    (
                        "Comparison of the mean Matthews correlation coefficient "
                        "obtained by the evaluated models. Error bars represent one "
                        "standard deviation across five random seeds, and numerical "
                        "labels indicate the corresponding mean values."
                    ),
                ),
                (
                    "Figure 4",
                    "Model comparison: ROC-AUC",
                    (
                        "Comparison of the mean ROC-AUC obtained by the evaluated "
                        "models. Error bars represent one standard deviation across "
                        "five random seeds, and numerical labels indicate the "
                        "corresponding mean values."
                    ),
                ),
                (
                    "Figure 5",
                    "Model comparison: PR-AUC",
                    (
                        "Comparison of the mean PR-AUC obtained by the evaluated "
                        "models. Error bars represent one standard deviation across "
                        "five random seeds, and numerical labels indicate the "
                        "corresponding mean values."
                    ),
                ),
            ]
        )

    if alpha_generated:
        captions.extend(
            [
                (
                    "Figure S1",
                    "Global alpha ablation: F1",
                    (
                        "Validation F1 score as a function of the Hybrid Transformer "
                        "weight α. The shaded region represents one standard "
                        "deviation across random seeds, and the selected operating "
                        "point is highlighted."
                    ),
                ),
                (
                    "Figure S2",
                    "Global alpha ablation: MCC",
                    (
                        "Validation MCC as a function of the Hybrid Transformer "
                        "weight α. The shaded region represents one standard "
                        "deviation across random seeds, and the selected operating "
                        "point is highlighted."
                    ),
                ),
                (
                    "Figure S3",
                    "Global alpha ablation: ROC-AUC",
                    (
                        "Validation ROC-AUC as a function of the Hybrid Transformer "
                        "weight α. The shaded region represents one standard "
                        "deviation across random seeds, and the selected operating "
                        "point is highlighted."
                    ),
                ),
                (
                    "Figure S4",
                    "Global alpha ablation: PR-AUC",
                    (
                        "Validation PR-AUC as a function of the Hybrid Transformer "
                        "weight α. The shaded region represents one standard "
                        "deviation across random seeds, and the selected operating "
                        "point is highlighted."
                    ),
                ),
            ]
        )

    lines = [
        "# Figure captions",
        "",
    ]

    for figure_number, title, caption in captions:
        lines.extend(
            [
                f"## {figure_number}. {title}",
                "",
                caption,
                "",
            ]
        )

    output_path = output_root / "captions" / "figure_captions.md"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_latex_integration_files(
    output_root: Path,
    statistics_generated: bool,
    alpha_generated: bool,
    model_figures_generated: bool,
) -> None:
    """
    Generate LaTeX snippets to integrate tables automatically
    and main and supplementary figures in the manuscript.

    Paths are expressed relative to the package root
    paper_assets_final.
    """

    paper_integration_directory = output_root / "paper" / "integration"

    supplementary_integration_directory = output_root / "supplementary" / "integration"

    paper_integration_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    supplementary_integration_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Recursos principales
    # ------------------------------------------------------------------

    paper_lines = [
        "% ============================================================",
        "% Main manuscript publication assets",
        "% Generated automatically by generate_paper_assets.py",
        "% ============================================================",
        "",
        "% Required packages:",
        r"% \usepackage{booktabs}",
        r"% \usepackage{graphicx}",
        r"% \usepackage{subcaption}",
        r"% \usepackage{tabularx}",
        "",
        "% Conceptual problem overview",
        r"\begin{figure*}[t]",
        r"\centering",
        (
            r"\includegraphics[width=0.98\textwidth]"
            r"{figures/fig01_problem_overview.pdf}"
        ),
        (
            r"\caption{Conceptual representation of the Bitcoin "
            r"anti-money-laundering classification problem.}"
        ),
        r"\label{fig:problem-overview}",
        r"\end{figure*}",
        "",
        "% Problem-to-method correspondence",
        r"\input{paper/latex/Table0_ProblemMethodMapping.tex}",
        "",
        "% Main result tables",
        r"\input{paper/latex/Table1_MainResults.tex}",
        "",
        r"\input{paper/latex/Table2_FusionAblation.tex}",
        "",
    ]

    if statistics_generated:
        paper_lines.extend(
            [
                r"\input{paper/latex/Table3_StatisticalComparison.tex}",
                "",
            ]
        )

    if model_figures_generated:
        paper_lines.extend(
            [
                "% Main model-comparison figures",
                r"\begin{figure*}[t]",
                r"\centering",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/models_f1.pdf}"
                ),
                r"\hfill",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/models_mcc.pdf}"
                ),
                r"\caption{Comparison of F1 and MCC across the evaluated models.}",
                r"\label{fig:model-comparison-f1-mcc}",
                r"\end{figure*}",
                "",
                r"\begin{figure*}[t]",
                r"\centering",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/models_roc_auc.pdf}"
                ),
                r"\hfill",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/models_pr_auc.pdf}"
                ),
                (
                    r"\caption{Comparison of ROC-AUC and PR-AUC across "
                    r"the evaluated models.}"
                ),
                r"\label{fig:model-comparison-auc}",
                r"\end{figure*}",
                "",
            ]
        )

    paper_path = paper_integration_directory / "ManuscriptAssets.tex"

    paper_path.write_text(
        "\n".join(paper_lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Recursos suplementarios
    # ------------------------------------------------------------------

    supplementary_lines = [
        "% ============================================================",
        "% Supplementary publication assets",
        "% Generated automatically by generate_paper_assets.py",
        "% ============================================================",
        "",
        "% Complete model-comparison table",
        r"\input{supplementary/latex/TableS1_AllModels.tex}",
        "",
    ]

    if statistics_generated:
        supplementary_lines.extend(
            [
                "% Complete paired statistical analysis",
                (
                    r"\input{supplementary/latex/"
                    r"TableS2_StatisticalComparison.tex}"
                ),
                "",
            ]
        )

    if alpha_generated:
        supplementary_lines.extend(
            [
                "% Alpha-ablation figures",
                r"\begin{figure*}[t]",
                r"\centering",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/alpha_f1.pdf}"
                ),
                r"\hfill",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/alpha_mcc.pdf}"
                ),
                r"\caption{Global alpha-ablation results for F1 and MCC.}",
                r"\label{fig:alpha-ablation-f1-mcc}",
                r"\end{figure*}",
                "",
                r"\begin{figure*}[t]",
                r"\centering",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/alpha_roc_auc.pdf}"
                ),
                r"\hfill",
                (
                    r"\includegraphics[width=0.48\textwidth]"
                    r"{figures/alpha_pr_auc.pdf}"
                ),
                (
                    r"\caption{Global alpha-ablation results for "
                    r"ROC-AUC and PR-AUC.}"
                ),
                r"\label{fig:alpha-ablation-auc}",
                r"\end{figure*}",
                "",
            ]
        )

    supplementary_path = supplementary_integration_directory / "SupplementaryAssets.tex"

    supplementary_path.write_text(
        "\n".join(supplementary_lines),
        encoding="utf-8",
    )


def generate_package_readme(
    output_root: Path,
) -> None:
    """
    Generate the main README for the publication package.
    """

    readme_lines = [
        "# Bitcoin AML publication assets",
        "",
        (
            "This directory contains the publication-ready tables, "
            "figures, captions, manuscript fragments, reproducibility "
            "metadata, and verification reports generated by "
            "`generate_paper_assets.py`."
        ),
        "",
        "## Directory structure",
        "",
        "- `paper/`: main-manuscript tables and LaTeX assets.",
        "- `supplementary/`: supplementary tables and LaTeX assets.",
        "- `figures/`: publication figures in PNG and PDF formats.",
        "- `captions/`: editorial captions for tables and figures.",
        "- `manuscript/`: automatically generated Results-section drafts.",
        (
            "- `reproducibility/`: environment, configuration, "
            "input-file hashes, and reproducibility metadata."
        ),
        "- `delivery/`: deterministic ZIP package and verification files.",
        "",
        "## Main manuscript integration",
        "",
        "The main manuscript assets can be included from the package root with:",
        "",
        "```latex",
        r"\input{paper/integration/ManuscriptAssets.tex}",
        "```",
        "",
        "## Supplementary integration",
        "",
        "The supplementary assets can be included from the package root with:",
        "",
        "```latex",
        r"\input{supplementary/integration/SupplementaryAssets.tex}",
        "```",
        "",
        "The generated LaTeX assets require:",
        "",
        "```latex",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{subcaption}",
        r"\usepackage{tabularx}",
        "```",
        "",
        "## Validation",
        "",
        (
            "`PUBLICATION_CHECK.txt` reports the structural, numerical, "
            "editorial, reproducibility, and LaTeX-dependency checks."
        ),
        "",
        (
            "`delivery/PACKAGE_CHECK.txt` reports the integrity "
            "verification of the deterministic ZIP archive."
        ),
        "",
        "## Hash verification",
        "",
        (
            "The SHA-256 hash of every packaged asset is listed in "
            "`delivery/PACKAGE_CONTENTS.sha256`."
        ),
        "",
        (
            "The SHA-256 hash of the ZIP archive is stored in "
            "`delivery/ARCHIVE_SHA256.txt`."
        ),
        "",
        "Example verification command:",
        "",
        "```bash",
        "sha256sum -c results/paper_assets_final/delivery/PACKAGE_CONTENTS.sha256",
        "```",
        "",
        "## Reproducible generation",
        "",
        (
            "Build timestamps are controlled with the "
            "`SOURCE_DATE_EPOCH` environment variable."
        ),
        "",
        "Example:",
        "",
        "```bash",
        "export SOURCE_DATE_EPOCH=1785366000",
        "python src/generate_paper_assets.py",
        "```",
        "",
        (
            "Two executions using the same inputs, software environment, "
            "configuration, and SOURCE_DATE_EPOCH should produce "
            "bit-for-bit identical ZIP archives."
        ),
        "",
        "## Generated-file inventory",
        "",
        "`MANIFEST.md` lists the generated publication assets and their file sizes.",
        "",
    ]

    readme_path = output_root / "README.md"

    readme_path.write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )


def generate_paper_evidence_map(
    output_root: Path,
    statistics_generated: bool,
    alpha_generated: bool,
    model_figures_generated: bool,
) -> None:
    """
    Generate a traceability map between planned sections
    of the manuscript and the publication resources supporting them.
    """

    lines = [
        "# Paper evidence map",
        "",
        (
            "This document maps the planned manuscript sections, "
            "claims, tables, figures, and reproducibility records "
            "to their corresponding generated evidence."
        ),
        "",
        "## Abstract",
        "",
        "Evidence sources:",
        "",
        "- `paper/tables/Table1_MainResults.csv`: principal performance values.",
        (
            "- `paper/tables/Table2_FusionAblation.csv`: comparison "
            "of fusion strategies."
        ),
        (
            "- `tables/Improvement_Over_Hybrid.csv`: improvements "
            "relative to the Hybrid Transformer."
        ),
        (
            "- `json/consolidated_results.json`: authoritative "
            "aggregated numerical source."
        ),
        "",
        "## Introduction and contributions",
        "",
        "Evidence sources:",
        "",
        (
            "- `README.md`: package scope, integration instructions, "
            "and reproducibility overview."
        ),
        (
            "- `reproducibility/REPRODUCIBILITY.md`: experimental "
            "traceability and execution context."
        ),
        (
            "- `reproducibility/reproducibility_metadata.json`: "
            "machine-readable reproducibility metadata."
        ),
        "",
        "Planned contribution claims:",
        "",
        (
            "- Comparative evaluation of individual, hybrid, "
            "early-fusion, and late-fusion models."
        ),
        "- Evaluation across five random seeds for every reported model.",
        (
            "- Reproducible and independently verifiable generation "
            "of publication assets."
        ),
        "",
        "## Methodology",
        "",
        "Evidence sources:",
        "",
        (
            "- `supplementary/tables/TableS1_AllModels.csv`: complete "
            "set of metrics and model metadata."
        ),
        (
            "- `json/consolidated_results.json`: model names, means, "
            "standard deviations, seeds, and source summaries."
        ),
        (
            "- `reproducibility/reproducibility_metadata.json`: "
            "configuration and input-file provenance."
        ),
        "",
        "## Experimental setup",
        "",
        "Evidence sources:",
        "",
        (
            "- `reproducibility/REPRODUCIBILITY.md`: software "
            "environment and reproducibility procedure."
        ),
        (
            "- `reproducibility/reproducibility_metadata.json`: "
            "structured environment and source information."
        ),
        "- `MANIFEST.md`: generated-file inventory.",
        "- `delivery/PACKAGE_CONTENTS.sha256`: SHA-256 hashes of packaged assets.",
        "",
        "## Main results",
        "",
        "Evidence sources:",
        "",
        "- `paper/tables/Table1_MainResults.csv`: main numerical comparison.",
        "- `paper/markdown/Table1_MainResults.md`: readable main-results table.",
        (
            "- `paper/latex/Table1_MainResults.tex`: manuscript-ready "
            "main-results table."
        ),
        "- `paper/text/Results_Draft.md`: automatically generated results narrative.",
        "- `paper/text/Results_Draft.tex`: LaTeX results narrative.",
        "- `markdown/results_summary.md`: concise derived results summary.",
        "",
    ]

    if model_figures_generated:
        lines.extend(
            [
                "Main-result figures:",
                "",
                "- `figures/models_f1.pdf`: F1 comparison.",
                "- `figures/models_mcc.pdf`: MCC comparison.",
                "- `figures/models_roc_auc.pdf`: ROC-AUC comparison.",
                "- `figures/models_pr_auc.pdf`: PR-AUC comparison.",
                "",
            ]
        )

    lines.extend(
        [
            "## Fusion ablation",
            "",
            "Evidence sources:",
            "",
            "- `paper/tables/Table2_FusionAblation.csv`: fusion ablation values.",
            (
                "- `paper/markdown/Table2_FusionAblation.md`: readable "
                "fusion-ablation table."
            ),
            (
                "- `paper/latex/Table2_FusionAblation.tex`: "
                "manuscript-ready fusion-ablation table."
            ),
            (
                "- `tables/Improvement_Over_Hybrid.csv`: absolute and "
                "relative improvements over the Hybrid Transformer."
            ),
            "",
        ]
    )

    if alpha_generated:
        lines.extend(
            [
                "Alpha-ablation evidence:",
                "",
                "- `figures/alpha_f1.pdf`: validation F1 by alpha.",
                "- `figures/alpha_mcc.pdf`: validation MCC by alpha.",
                "- `figures/alpha_roc_auc.pdf`: validation ROC-AUC by alpha.",
                "- `figures/alpha_pr_auc.pdf`: validation PR-AUC by alpha.",
                "",
            ]
        )

    lines.extend(
        [
            "## Statistical analysis",
            "",
        ]
    )

    if statistics_generated:
        lines.extend(
            [
                "Evidence sources:",
                "",
                (
                    "- `paper/tables/Table3_StatisticalComparison.csv`: "
                    "publication statistical comparison."
                ),
                (
                    "- `paper/latex/Table3_StatisticalComparison.tex`: "
                    "manuscript-ready statistical table."
                ),
                (
                    "- `supplementary/tables/"
                    "TableS2_StatisticalComparison.csv`: complete "
                    "statistical output."
                ),
                (
                    "- `supplementary/latex/"
                    "TableS2_StatisticalComparison.tex`: complete "
                    "supplementary statistical table."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No statistical-comparison assets were generated for this build.",
                "",
            ]
        )

    lines.extend(
        [
            "## Discussion",
            "",
            "Evidence sources:",
            "",
            (
                "- `paper/tables/Table1_MainResults.csv`: comparison "
                "across all primary models."
            ),
            (
                "- `paper/tables/Table2_FusionAblation.csv`: evidence "
                "for discussing fusion behavior."
            ),
            (
                "- `tables/Improvement_Over_Hybrid.csv`: evidence for "
                "claims relative to the Hybrid Transformer."
            ),
            (
                "- `supplementary/tables/TableS1_AllModels.csv`: "
                "precision, recall, Macro-F1, and complete metrics."
            ),
            "",
            "## Reproducibility and integrity",
            "",
            "Evidence sources:",
            "",
            (
                "- `PUBLICATION_CHECK.txt`: structural, numerical, "
                "editorial, and dependency checks."
            ),
            "- `delivery/PACKAGE_CHECK.txt`: ZIP integrity and path-safety checks.",
            "- `delivery/PACKAGE_CONTENTS.sha256`: packaged-file hash inventory.",
            "- `delivery/ARCHIVE_SHA256.txt`: final ZIP hash.",
            (
                "- `reproducibility/reproducibility_metadata.json`: "
                "build timestamp, inputs, and environment metadata."
            ),
            "",
            "## Manuscript integration",
            "",
            "Evidence sources:",
            "",
            "- `paper/integration/ManuscriptAssets.tex`: main tables and figures.",
            (
                "- `supplementary/integration/SupplementaryAssets.tex`: "
                "supplementary tables and figures."
            ),
            "- `captions/table_captions.md`: table captions.",
            "- `captions/figure_captions.md`: figure captions.",
            "",
            "## Authoring rule",
            "",
            (
                "Every numerical statement in the manuscript must be "
                "traceable to `json/consolidated_results.json` or to one "
                "of the CSV tables listed in this evidence map."
            ),
            "",
            (
                "Rounded display values must not be used to recompute "
                "differences, percentages, statistical results, or "
                "rankings."
            ),
            "",
            (
                "The authoritative source for numerical calculations is "
                "the full-precision machine-readable output."
            ),
            "",
        ]
    )

    evidence_map_path = output_root / "paper" / "PAPER_EVIDENCE_MAP.md"

    evidence_map_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_map_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def validate_latex_integration_dependencies(
    output_root: Path,
    integration_path: Path,
) -> list[str]:
    """
    Verify that dependencies declared through \\input y
    \\includegraphics exist inside the output directory.
    """

    errors: list[str] = []

    if not integration_path.is_file():
        errors.append(
            "The LaTeX integration file does not exist: "
            f"{integration_path.relative_to(output_root)}"
        )
        return errors

    try:
        integration_text = integration_path.read_text(encoding="utf-8")

    except OSError as exc:
        errors.append(
            "Could not read the LaTeX integration file "
            f"{integration_path.relative_to(output_root)}: {exc}"
        )
        return errors

    input_references = re.findall(
        r"\\input[{]([^{}]+)[}]",
        integration_text,
    )

    graphic_references = re.findall(
        (
            r"\\includegraphics"
            r"(?:\[[^]]*])?"
            r"[{]([^{}]+)[}]"
        ),
        integration_text,
    )

    for reference in input_references:
        normalized_reference = reference.strip()

        referenced_path = output_root / normalized_reference

        if not referenced_path.is_file():
            errors.append(
                "Missing LaTeX dependency in "
                f"{integration_path.relative_to(output_root)}: "
                f"\\input{{{normalized_reference}}}"
            )

    for reference in graphic_references:
        normalized_reference = reference.strip()

        referenced_path = output_root / normalized_reference

        if referenced_path.suffix:
            candidate_paths = [
                referenced_path,
            ]

        else:
            candidate_paths = [
                referenced_path.with_suffix(".pdf"),
                referenced_path.with_suffix(".png"),
                referenced_path.with_suffix(".jpg"),
                referenced_path.with_suffix(".jpeg"),
            ]

        if not any(candidate_path.is_file() for candidate_path in candidate_paths):
            errors.append(
                "Missing LaTeX figure in "
                f"{integration_path.relative_to(output_root)}: "
                f"\\includegraphics{{{normalized_reference}}}"
            )

    return errors


def generate_reproducibility_report(
    output_root: Path,
    models: list[ModelResult],
    decimals: int,
) -> None:
    """
    Generate reproducibility metadata in Markdown and JSON.

    The report records the execution environment, configuration
    editorial settings, available seeds, source files, and the
    SHA-256 hash of the consolidated JSON.
    """

    reproducibility_directory = output_root / "reproducibility"

    reproducibility_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    consolidated_path = output_root / "json" / "consolidated_results.json"

    consolidated_sha256: str | None = None
    consolidated_size_bytes: int | None = None

    if consolidated_path.is_file():
        consolidated_bytes = consolidated_path.read_bytes()

        consolidated_sha256 = hashlib.sha256(consolidated_bytes).hexdigest()

        consolidated_size_bytes = len(consolidated_bytes)

    source_date_epoch = get_source_date_epoch()

    generated_at_utc = reproducible_iso_datetime()

    model_records: list[dict[str, Any]] = []

    for model in models:
        source_exists = model.source_file.is_file()

        source_sha256: str | None = None
        source_size_bytes: int | None = None

        if source_exists:
            source_bytes = model.source_file.read_bytes()

            source_sha256 = hashlib.sha256(source_bytes).hexdigest()

            source_size_bytes = len(source_bytes)

        model_records.append(
            {
                "name": model.name,
                "directory": portable_path(model.directory),
                "source_file": portable_path(model.source_file),
                "source_exists": source_exists,
                "source_size_bytes": source_size_bytes,
                "source_sha256": source_sha256,
                "n_seeds": model.n_seeds,
                "seeds": model.seeds,
                "available_metrics": sorted(model.metrics.keys()),
            }
        )

    metadata: dict[str, Any] = {
        "generated_at_utc": generated_at_utc,
        "source_date_epoch": source_date_epoch,
        "project_root": ".",
        "output_root": portable_path(output_root),
        "script": "src/generate_paper_assets.py",
        "environment": {
            "python_version": sys.version,
            "python_implementation": (platform.python_implementation()),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy_version": np.__version__,
            "matplotlib_version": plt.matplotlib.__version__,
        },
        "configuration": {
            "reproducible_build": True,
            "decimals": decimals,
            "model_count": len(models),
            "metrics": METRICS,
            "paper_metrics": PAPER_METRICS,
            "supplementary_metrics": SUPPLEMENTARY_METRICS,
            "paper_model_order": PAPER_MODEL_ORDER,
            "fusion_models": sorted(FUSION_MODELS),
        },
        "consolidated_results": {
            "path": portable_path(consolidated_path),
            "exists": consolidated_path.is_file(),
            "size_bytes": consolidated_size_bytes,
            "sha256": consolidated_sha256,
        },
        "models": model_records,
    }

    json_path = reproducibility_directory / "reproducibility_metadata.json"

    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# Reproducibility report",
        "",
        "This report was generated automatically by `generate_paper_assets.py`.",
        "",
        "## Generation",
        "",
        f"- Generated at (UTC): `{generated_at_utc}`",
        f"- SOURCE_DATE_EPOCH: `{source_date_epoch}`",
        "- Project root: `.`",
        f"- Output root: `{portable_path(output_root)}`",
        f"- Models processed: {len(models)}",
        f"- Decimal precision: {decimals}",
        "",
        "## Software environment",
        "",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Python implementation: `{platform.python_implementation()}`",
        f"- NumPy: `{np.__version__}`",
        f"- Matplotlib: `{plt.matplotlib.__version__}`",
        f"- Operating system: `{platform.system()}`",
        f"- Platform: `{platform.platform()}`",
        f"- Machine architecture: `{platform.machine()}`",
        "",
        "## Consolidated results",
        "",
        f"- File: `{consolidated_path}`",
        (
            f"- Size: `{consolidated_size_bytes}` bytes"
            if consolidated_size_bytes is not None
            else "- Size: unavailable"
        ),
        (
            f"- SHA-256: `{consolidated_sha256}`"
            if consolidated_sha256 is not None
            else "- SHA-256: unavailable"
        ),
        "",
        "## Evaluation configuration",
        "",
        ("- Metrics: " + ", ".join(f"`{metric}`" for metric in METRICS)),
        (
            "- Main-paper metrics: "
            + ", ".join(f"`{metric}`" for metric in PAPER_METRICS)
        ),
        (
            "- Supplementary metrics: "
            + ", ".join(f"`{metric}`" for metric in SUPPLEMENTARY_METRICS)
        ),
        "",
        "## Experiment sources",
        "",
        "| Model | n_seeds | Seeds | Source file | SHA-256 |",
        "|---|---:|---|---|---|",
    ]

    for record in model_records:
        seeds_text = (
            ", ".join(str(seed) for seed in record["seeds"])
            if record["seeds"]
            else "not reported"
        )

        n_seeds_text = (
            str(record["n_seeds"]) if record["n_seeds"] is not None else "not reported"
        )

        source_hash_text = (
            str(record["source_sha256"])
            if record["source_sha256"] is not None
            else "unavailable"
        )

        markdown_lines.append(
            f"| {record['name']} "
            f"| {n_seeds_text} "
            f"| {seeds_text} "
            f"| `{record['source_file']}` "
            f"| `{source_hash_text}` |"
        )

    markdown_lines.extend(
        [
            "",
            "## Reproduction notes",
            "",
            (
                "- Only experiment directories explicitly declared in "
                "`MODEL_CONFIG` are processed."
            ),
            (
                "- Per-seed files under directories named `seed_*` are "
                "not used as aggregate-result sources."
            ),
            (
                "- The SHA-256 values can be used to verify that the "
                "input summaries and consolidated results have not changed."
            ),
            "",
        ]
    )

    markdown_path = reproducibility_directory / "REPRODUCIBILITY.md"

    markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def generate_manifest(
    output_root: Path,
) -> None:
    """
    Generate a deterministic inventory of produced resources.

    The manifest includes only files that actually exist
    inside the output directory.
    """

    manifest_path = output_root / "MANIFEST.md"

    generated_files = sorted(
        (
            path
            for path in output_root.rglob("*")
            if (
                path != manifest_path
                and is_publishable_asset(
                    path,
                    output_root,
                )
            )
        ),
        key=lambda path: path.relative_to(output_root).as_posix(),
    )

    lines = [
        "# Generated assets manifest",
        "",
        (
            "This file lists the publication assets generated by "
            "`generate_paper_assets.py`."
        ),
        "",
        f"Total files: {len(generated_files)}",
        "",
        "| File | Format | Size (bytes) |",
        "|---|---:|---:|",
    ]

    for path in generated_files:
        relative_path = path.relative_to(output_root).as_posix()

        extension = path.suffix.lower().lstrip(".") if path.suffix else "none"

        size_bytes = path.stat().st_size

        lines.append(f"| `{relative_path}` | {extension} | {size_bytes} |")

    lines.append("")

    manifest_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_delivery_package(
    output_root: Path,
) -> bool:
    """
    Generate and verify a deterministic publication ZIP package.

    The ``delivery`` directory is excluded to prevent the ZIP,
    its hashes, and its reports from being included recursively.
    """

    delivery_directory = output_root / "delivery"

    delivery_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = delivery_directory / "bitcoin_aml_publication_assets.zip"

    contents_hash_path = delivery_directory / "PACKAGE_CONTENTS.sha256"

    archive_hash_path = delivery_directory / "ARCHIVE_SHA256.txt"

    package_check_path = delivery_directory / "PACKAGE_CHECK.txt"

    package_root_name = "bitcoin_aml_publication_assets"

    # ------------------------------------------------------------------
    # Deterministic file selection
    # ------------------------------------------------------------------

    source_files = sorted(
        (
            path
            for path in output_root.rglob("*")
            if is_publishable_asset(
                path,
                output_root,
            )
        ),
        key=lambda path: path.relative_to(output_root).as_posix(),
    )

    content_hashes: dict[str, str] = {}

    for source_path in source_files:
        relative_path = source_path.relative_to(output_root).as_posix()

        content_hashes[relative_path] = sha256_file(source_path)

    hash_lines = [
        f"{digest}  {relative_path}"
        for relative_path, digest in sorted(content_hashes.items())
    ]

    contents_hash_text = "\n".join(hash_lines) + "\n"

    contents_hash_path.write_text(
        contents_hash_text,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Deterministic ZIP creation
    # ------------------------------------------------------------------

    source_datetime = reproducible_datetime_utc()

    # ZIP no admite fechas anteriores a 1980.
    if source_datetime.year < 1980:
        fixed_datetime = (
            1980,
            1,
            1,
            0,
            0,
            0,
        )
    else:
        fixed_datetime = (
            source_datetime.year,
            source_datetime.month,
            source_datetime.day,
            source_datetime.hour,
            source_datetime.minute,
            source_datetime.second - source_datetime.second % 2,
        )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source_path in source_files:
            relative_path = source_path.relative_to(output_root).as_posix()

            archive_name = f"{package_root_name}/{relative_path}"

            archive_info = zipfile.ZipInfo(
                filename=archive_name,
                date_time=fixed_datetime,
            )

            archive_info.compress_type = zipfile.ZIP_DEFLATED

            archive_info.create_system = 3

            # Regular file with 0644 permissions.
            archive_info.external_attr = 0o100644 << 16

            archive_info.flag_bits |= 0x800

            archive.writestr(
                archive_info,
                source_path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

        hash_archive_name = f"{package_root_name}/PACKAGE_CONTENTS.sha256"

        hash_archive_info = zipfile.ZipInfo(
            filename=hash_archive_name,
            date_time=fixed_datetime,
        )

        hash_archive_info.compress_type = zipfile.ZIP_DEFLATED

        hash_archive_info.create_system = 3
        hash_archive_info.external_attr = 0o100644 << 16
        hash_archive_info.flag_bits |= 0x800

        archive.writestr(
            hash_archive_info,
            contents_hash_text.encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )

    archive_sha256 = sha256_file(archive_path)

    archive_hash_path.write_text(
        f"{archive_sha256}  {archive_path.name}\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # ZIP content verification
    # ------------------------------------------------------------------

    verification_errors: list[str] = []
    verified_files = 0
    duplicate_member_count = 0
    unsafe_member_count = 0

    expected_archive_names = {
        f"{package_root_name}/{relative_path}" for relative_path in content_hashes
    }

    expected_archive_names.add(f"{package_root_name}/PACKAGE_CONTENTS.sha256")

    try:
        with zipfile.ZipFile(
            archive_path,
            mode="r",
        ) as archive:
            corrupt_member = archive.testzip()

            if corrupt_member is not None:
                verification_errors.append(f"Miembro ZIP corrupto: {corrupt_member}")

            archive_names = archive.namelist()

            # --------------------------------------------------------------
            # Security audit of ZIP names and paths
            # --------------------------------------------------------------

            member_counts: dict[str, int] = {}

            for member_name in archive_names:
                member_counts[member_name] = member_counts.get(member_name, 0) + 1

            duplicate_members = sorted(
                member_name for member_name, count in member_counts.items() if count > 1
            )

            duplicate_member_count = len(duplicate_members)

            for member_name in duplicate_members:
                verification_errors.append(f"Duplicate name inside ZIP: {member_name}")

            expected_root_prefix = f"{package_root_name}/"

            for member_name in archive_names:
                if "\\" in member_name:
                    unsafe_member_count += 1

                    verification_errors.append(
                        f"Disallowed path separator inside ZIP: {member_name}"
                    )

                member_path = PurePosixPath(member_name)

                if member_path.is_absolute():
                    unsafe_member_count += 1

                    verification_errors.append(
                        f"Absolute path is not allowed inside ZIP: {member_name}"
                    )

                if ".." in member_path.parts:
                    unsafe_member_count += 1

                    verification_errors.append(
                        "Path traversal component is not allowed inside ZIP: "
                        f"{member_name}"
                    )

                if not member_name.startswith(expected_root_prefix):
                    unsafe_member_count += 1

                    verification_errors.append(
                        f"Member outside the expected root directory: {member_name}"
                    )

            actual_archive_names = set(archive_names)

            missing_archive_members = sorted(
                expected_archive_names - actual_archive_names
            )

            unexpected_archive_members = sorted(
                actual_archive_names - expected_archive_names
            )

            for member_name in missing_archive_members:
                verification_errors.append(f"File missing from ZIP: {member_name}")

            for member_name in unexpected_archive_members:
                verification_errors.append(f"Unexpected file in ZIP: {member_name}")

            for (
                relative_path,
                expected_hash,
            ) in sorted(content_hashes.items()):
                archive_name = f"{package_root_name}/{relative_path}"

                if archive_name not in actual_archive_names:
                    continue

                archived_bytes = archive.read(archive_name)

                archived_hash = hashlib.sha256(archived_bytes).hexdigest()

                verified_files += 1

                if archived_hash != expected_hash:
                    verification_errors.append(
                        f"Different hash inside ZIP: {relative_path}"
                    )

            packaged_hash_manifest_name = f"{package_root_name}/PACKAGE_CONTENTS.sha256"

            if packaged_hash_manifest_name in actual_archive_names:
                packaged_hash_manifest = archive.read(
                    packaged_hash_manifest_name
                ).decode("utf-8")

                if packaged_hash_manifest != contents_hash_text:
                    verification_errors.append(
                        "PACKAGE_CONTENTS.sha256 dentro "
                        "of the ZIP does not match the "
                        "inventario externo."
                    )

    except (
        OSError,
        zipfile.BadZipFile,
        UnicodeDecodeError,
    ) as exc:
        verification_errors.append(f"Could not verify ZIP: {exc}")

    archive_hash_after_verification = (
        sha256_file(archive_path) if archive_path.is_file() else ""
    )

    if archive_hash_after_verification != archive_sha256:
        verification_errors.append("The ZIP hash changed during verification.")

    package_passed = not verification_errors

    check_lines = [
        "DELIVERY PACKAGE CHECK",
        "=" * 44,
        "",
        f"SOURCE_DATE_EPOCH     : {get_source_date_epoch()}",
        f"Build timestamp UTC   : {reproducible_iso_datetime()}",
        f"Source files          : {len(source_files)}",
        f"Expected ZIP members  : {len(expected_archive_names)}",
        f"Verified file hashes  : {verified_files}",
        f"Duplicate ZIP members : {duplicate_member_count}",
        f"Unsafe path violations: {unsafe_member_count}",
        f"Verification errors   : {len(verification_errors)}",
        f"Archive size (bytes)  : {archive_path.stat().st_size}",
        f"Archive SHA-256       : {archive_sha256}",
        "",
        f"STATUS: {'PASS' if package_passed else 'FAIL'}",
        "",
    ]

    if verification_errors:
        check_lines.extend(
            [
                "ERRORS",
                "-" * 44,
                *[f"- {error}" for error in verification_errors],
                "",
            ]
        )
    else:
        check_lines.extend(
            [
                "The delivery ZIP is complete and readable.",
                (
                    "All ZIP member names passed the duplicate-name, "
                    "root-directory, and path-safety checks."
                ),
                "All packaged files match their recorded SHA-256 hashes.",
                "The archive contains no unexpected members.",
                "",
            ]
        )

    package_check_path.write_text(
        "\n".join(check_lines),
        encoding="utf-8",
    )

    return package_passed


# ============================================================================
# STATISTICS
# ============================================================================


def first_existing(root: Path, candidates: list[str]) -> Path | None:
    for relative in candidates:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def generate_statistics_table(
    project_root: Path,
    output_root: Path,
    decimals: int,
) -> bool:
    """
    Generate two versions of the statistical comparison:

    1. Table3_StatisticalComparison
       Compact table intended for the main manuscript.

    2. TableS2_StatisticalComparison
       Complete table intended for supplementary material.

    Main-column selection uses aliases to
    tolerar variaciones en los encabezados del CSV fuente.
    """

    source = first_existing(
        project_root,
        STATISTICS_CANDIDATES,
    )

    if source is None:
        return False

    rows = read_csv(source)

    if not rows:
        return False

    full_headers = list(rows[0].keys())

    # ------------------------------------------------------------------
    # Utilidades locales
    # ------------------------------------------------------------------

    def normalized_header(header: str) -> str:
        return (
            header.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "percent")
        )

    normalized_to_original = {
        normalized_header(header): header for header in full_headers
    }

    def detect_statistics_header(
        candidates: list[str],
    ) -> str | None:
        for candidate in candidates:
            normalized_candidate = normalized_header(candidate)

            if normalized_candidate in normalized_to_original:
                return normalized_to_original[normalized_candidate]

        return None

    def markdown_from_rows(
        rows_data: list[dict[str, Any]],
        headers: list[str],
    ) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|",
        ]

        for table_row in rows_data:
            lines.append(
                "| "
                + " | ".join(str(table_row.get(header, "")) for header in headers)
                + " |"
            )

        return "\n".join(lines) + "\n"

    def latex_from_rows(
        rows_data: list[dict[str, Any]],
        headers: list[str],
        caption: str,
        label: str,
        font_command: str,
    ) -> str:
        column_specification = "l" + "r" * max(0, len(headers) - 1)

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            font_command,
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            rf"\begin{{tabular}}{{{column_specification}}}",
            r"\toprule",
            (" & ".join(latex_escape(header) for header in headers) + r" \\"),
            r"\midrule",
        ]

        for table_row in rows_data:
            lines.append(
                " & ".join(
                    latex_escape(
                        str(
                            table_row.get(
                                header,
                                "",
                            )
                        )
                    )
                    for header in headers
                )
                + r" \\"
            )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Complete supplementary table
    # ------------------------------------------------------------------

    supplementary_csv_path = (
        output_root / "supplementary" / "tables" / "TableS2_StatisticalComparison.csv"
    )

    supplementary_markdown_path = (
        output_root / "supplementary" / "markdown" / "TableS2_StatisticalComparison.md"
    )

    supplementary_latex_path = (
        output_root / "supplementary" / "latex" / "TableS2_StatisticalComparison.tex"
    )

    supplementary_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    supplementary_markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    supplementary_latex_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        supplementary_csv_path,
        rows,
    )

    supplementary_markdown_path.write_text(
        markdown_from_rows(
            rows,
            full_headers,
        ),
        encoding="utf-8",
    )

    supplementary_latex_path.write_text(
        latex_from_rows(
            rows_data=rows,
            headers=full_headers,
            caption=(
                "Complete paired statistical comparison between "
                "Late Fusion and the Hybrid Transformer."
            ),
            label="tab:supplementary-paired-statistics",
            font_command=r"\scriptsize",
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Selection of columns for the main table
    # ------------------------------------------------------------------

    metric_header = detect_statistics_header(
        [
            "metric",
            "metric_name",
            "evaluation_metric",
            "measure",
        ]
    )

    difference_header = detect_statistics_header(
        [
            "mean_difference",
            "mean_diff",
            "difference",
            "delta",
            "absolute_delta",
            "mean_delta",
            "observed_difference",
        ]
    )

    statistic_header = detect_statistics_header(
        [
            "statistic",
            "test_statistic",
            "wilcoxon_statistic",
            "wilcoxon_w",
            "w_statistic",
            "t_statistic",
            "z_statistic",
        ]
    )

    p_value_header = detect_statistics_header(
        [
            "p_value",
            "pvalue",
            "p_val",
            "raw_p_value",
            "adjusted_p_value",
            "corrected_p_value",
            "p_value_adjusted",
        ]
    )

    effect_size_header = detect_statistics_header(
        [
            "effect_size",
            "cohens_d",
            "cohen_d",
            "rank_biserial",
            "rank_biserial_correlation",
            "cliffs_delta",
        ]
    )

    significance_header = detect_statistics_header(
        [
            "significant",
            "is_significant",
            "statistically_significant",
            "reject_null",
            "reject_h0",
            "significance",
        ]
    )

    preferred_columns = [
        (
            metric_header,
            "Metric",
            "metric",
        ),
        (
            difference_header,
            "Mean difference",
            "number",
        ),
        (
            statistic_header,
            "Wilcoxon statistic",
            "number",
        ),
        (
            p_value_header,
            "p-value",
            "p_value",
        ),
        (
            effect_size_header,
            (
                "Rank-biserial correlation"
                if effect_size_header is not None
                and normalized_header(effect_size_header)
                in {
                    "rank_biserial",
                    "rank_biserial_correlation",
                }
                else "Effect size"
            ),
            "number",
        ),
        (
            significance_header,
            "Significant",
            "text",
        ),
    ]

    selected_columns: list[tuple[str, str, str]] = []
    selected_sources: set[str] = set()

    for source_header, publication_header, value_type in preferred_columns:
        if source_header is None:
            continue

        if source_header in selected_sources:
            continue

        selected_columns.append(
            (
                source_header,
                publication_header,
                value_type,
            )
        )

        selected_sources.add(source_header)

    # The manuscript should keep at most six columns.
    selected_columns = selected_columns[:6]

    # Safe fallback if too few columns were recognized.
    if len(selected_columns) < 2:
        selected_columns = [
            (
                header,
                header,
                "text",
            )
            for header in full_headers[:6]
        ]

    def format_statistics_metric(
        raw_metric_value: Any,
    ) -> str:
        """
        Convert the internal metric identifier to its name
        editorial.
        """
        text = str(raw_metric_value).strip()

        normalized = normalized_header(text)

        if normalized in DISPLAY_NAMES:
            return DISPLAY_NAMES[normalized]

        metric_aliases = {
            "balanced_accuracy": "Bal. Acc.",
            "macro_f1": "Macro-F1",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "f1": "F1",
            "mcc": "MCC",
            "precision": "Precision",
            "recall": "Recall",
        }

        if normalized in metric_aliases:
            return metric_aliases[normalized]

        return text.replace("_", " ").strip().title()

    def format_statistics_number(
        raw_number_value: Any,
        number_kind: str,
    ) -> str:
        """
        Format numeric values for the main table.

        The supplementary table retains original values without
        modification.
        """

        text = str(raw_number_value).strip()

        number = safe_float(raw_number_value)

        if number is None:
            return text

        if number_kind == "p_value":
            if number < 0.0001:
                return f"{number:.2e}"

            return f"{number:.4f}"

        return f"{number:.{decimals}f}"

    compact_headers = [
        publication_header for _, publication_header, _ in selected_columns
    ]

    compact_rows: list[dict[str, Any]] = []

    for row in rows:
        compact_row: dict[str, Any] = {}

        for (
            source_header,
            publication_header,
            value_type,
        ) in selected_columns:
            value = row.get(
                source_header,
                "",
            )

            if value_type == "metric":
                formatted_value = format_statistics_metric(value)

            elif value_type in {
                "number",
                "p_value",
            }:
                formatted_value = format_statistics_number(
                    value,
                    value_type,
                )

            else:
                formatted_value = str(value).strip()

            compact_row[publication_header] = formatted_value

        compact_rows.append(compact_row)

    # ------------------------------------------------------------------
    # Compact main table
    # ------------------------------------------------------------------

    paper_csv_path = (
        output_root / "paper" / "tables" / "Table3_StatisticalComparison.csv"
    )

    paper_markdown_path = (
        output_root / "paper" / "markdown" / "Table3_StatisticalComparison.md"
    )

    paper_latex_path = (
        output_root / "paper" / "latex" / "Table3_StatisticalComparison.tex"
    )

    paper_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paper_markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paper_latex_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        paper_csv_path,
        compact_rows,
    )

    paper_markdown_path.write_text(
        markdown_from_rows(
            compact_rows,
            compact_headers,
        ),
        encoding="utf-8",
    )

    paper_latex_path.write_text(
        latex_from_rows(
            rows_data=compact_rows,
            headers=compact_headers,
            caption=(
                "Paired statistical comparison between Late Fusion "
                "and the Hybrid Transformer."
            ),
            label="tab:paired-statistics",
            font_command=r"\small",
        ),
        encoding="utf-8",
    )

    return True


# ============================================================================
# FIGURES
# ============================================================================


def generate_problem_overview_figure(
    output_root: Path,
) -> None:
    """
    Generates a deterministic conceptual overview of the Bitcoin
    illicit-transaction detection pipeline.

    Generated files
    ---------------
    figures/fig01_problem_overview.png
    figures/fig01_problem_overview.pdf
    """

    stages = [
        (
            "1. Participants",
            "Entities that send\nand receive funds.",
        ),
        (
            "2. Addresses",
            "Pseudonymous\ncryptographic identifiers.",
        ),
        (
            "3. Transactions",
            "Transfers of value with\nobservable attributes.",
        ),
        (
            "4. Blockchain",
            "Public ledger and\nstructural relationships.",
        ),
        (
            "5. Information\nExtraction",
            "Transactional features\nand graph context.",
        ),
        (
            "6. Machine Learning\nModels",
            "Transformer, GraphSAGE,\nand Late Fusion.",
        ),
    ]

    fig, axis = plt.subplots(
        figsize=(16.0, 7.6),
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    # ------------------------------------------------------------------
    # Main pipeline geometry
    # ------------------------------------------------------------------

    box_width = 0.135
    box_height = 0.31
    start_x = 0.025
    gap = 0.025
    center_y = 0.57

    box_facecolor = "#EAF2F8"
    box_edgecolor = "#5B7FA3"
    output_facecolor = "#EDF7ED"
    output_edgecolor = "#4F7F58"

    for index, (title, description) in enumerate(stages):
        x_position = start_x + index * (box_width + gap)

        rectangle = plt.Rectangle(
            (
                x_position,
                center_y - box_height / 2,
            ),
            box_width,
            box_height,
            facecolor=box_facecolor,
            edgecolor=box_edgecolor,
            linewidth=1.25,
        )

        axis.add_patch(rectangle)

        axis.text(
            x_position + box_width / 2,
            center_y + 0.080,
            title,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            linespacing=1.15,
        )

        axis.plot(
            [
                x_position + 0.018,
                x_position + box_width - 0.018,
            ],
            [
                center_y + 0.015,
                center_y + 0.015,
            ],
            linewidth=0.8,
            color=box_edgecolor,
        )

        axis.text(
            x_position + box_width / 2,
            center_y - 0.065,
            description,
            ha="center",
            va="center",
            fontsize=7.8,
            linespacing=1.35,
        )

        if index < len(stages) - 1:
            arrow_start = x_position + box_width
            arrow_end = x_position + box_width + gap

            axis.annotate(
                "",
                xy=(
                    arrow_end - 0.003,
                    center_y,
                ),
                xytext=(
                    arrow_start + 0.003,
                    center_y,
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.4,
                    "color": "black",
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    # ------------------------------------------------------------------
    # Classification output
    # ------------------------------------------------------------------

    output_x = 0.50
    output_y = 0.125
    output_width = 0.42
    output_height = 0.17

    output_box = plt.Rectangle(
        (
            output_x - output_width / 2,
            output_y - output_height / 2,
        ),
        output_width,
        output_height,
        facecolor=output_facecolor,
        edgecolor=output_edgecolor,
        linewidth=1.4,
    )

    axis.add_patch(output_box)

    axis.text(
        output_x,
        output_y + 0.030,
        "Transaction Classification",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#204D28",
    )

    axis.text(
        output_x,
        output_y - 0.030,
        "Estimated illicit-class probability and final legitimate / illicit decision",
        ha="center",
        va="center",
        fontsize=8.3,
    )

    last_box_center_x = start_x + (len(stages) - 1) * (box_width + gap) + box_width / 2

    axis.annotate(
        "",
        xy=(
            output_x,
            output_y + output_height / 2,
        ),
        xytext=(
            last_box_center_x,
            center_y - box_height / 2,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
            "color": "black",
            "connectionstyle": "arc3,rad=0.18",
        },
    )

    # ------------------------------------------------------------------
    # Title and subtitle
    # ------------------------------------------------------------------

    axis.text(
        0.5,
        0.925,
        (
            "Conceptual Workflow for Automatic Detection "
            "of Illicit Activities in Bitcoin"
        ),
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
    )

    axis.text(
        0.5,
        0.855,
        (
            "Public blockchain information is transformed into "
            "representations suitable for supervised classification."
        ),
        ha="center",
        va="center",
        fontsize=10.2,
    )

    # Use explicit margins instead of tight_layout so that Matplotlib does
    # not compress the diagram or alter the box spacing.
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.025,
        top=0.985,
    )

    png_path = output_root / "figures" / "fig01_problem_overview.png"

    pdf_path = output_root / "figures" / "fig01_problem_overview.pdf"

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def plot_model_metric(
    models: list[ModelResult],
    metric: str,
    output_root: Path,
) -> None:
    available = order_models(
        [model for model in models if metric in model.metrics],
        PAPER_MODEL_ORDER,
    )
    if len(available) < 2:
        return

    names = [
        FIGURE_MODEL_NAMES.get(
            model.name,
            model.name,
        )
        for model in available
    ]
    means = np.array(
        [model.metrics[metric].mean for model in available],
        dtype=float,
    )
    stds = np.array(
        [model.metrics[metric].std for model in available],
        dtype=float,
    )
    positions = np.arange(len(names))

    fig, axis = plt.subplots(figsize=(max(11.0, len(names) * 1.4), 5.8))
    bars = axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=4,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        names,
        rotation=0,
        ha="center",
        fontsize=9,
    )
    axis.margins(x=0.02)
    axis.set_ylabel(DISPLAY_NAMES[metric])
    axis.set_title(DISPLAY_NAMES[metric])
    axis.grid(True, axis="y", alpha=0.25)
    for bar, mean, std in zip(
        bars,
        means,
        stds,
    ):
        axis.annotate(
            f"{mean:.4f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                mean,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    lower = max(0.0, float(np.min(means - stds)) - 0.05)
    upper = min(1.0, float(np.max(means + stds)) + 0.07)
    if upper > lower:
        axis.set_ylim(lower, upper)

    fig.tight_layout()
    fig.savefig(
        output_root / "figures" / f"models_{metric}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_root / "figures" / f"models_{metric}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
    )


def detect_column(
    headers: list[str],
    candidates: list[str],
) -> str | None:
    normalized_headers = {normalize_column_name(header): header for header in headers}

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized_headers:
            return normalized_headers[normalized_candidate]

    return None


def plot_alpha_ablation(
    project_root: Path,
    output_root: Path,
) -> bool:
    source = first_existing(project_root, ALPHA_CANDIDATES)
    if source is None:
        print("[ALPHA] No candidate file was found.")
        return False

    rows = read_csv(source)
    if not rows:
        print(f"[ALPHA] The file is empty: {source}")
        return False

    headers = list(rows[0].keys())

    print(f"[ALPHA] File found: {source}")
    print(f"[ALPHA] Available headers: {headers}")

    alpha_col = detect_column(
        headers,
        [
            "alpha",
            "global_alpha",
            "fusion_alpha",
            "alpha_hybrid",
            "hybrid_alpha",
            "weight_hybrid",
            "hybrid_weight",
            "alpha_value",
        ],
    )
    print(f"[ALPHA] Detected alpha column: {alpha_col}")
    if alpha_col is None:
        return False

    plotted = False
    for metric in ("f1", "mcc", "roc_auc", "pr_auc"):
        mean_col = detect_column(
            headers,
            [
                f"validation_{metric}_mean",
                f"val_{metric}_mean",
                f"test_{metric}_mean",
                f"mean_validation_{metric}",
                f"mean_val_{metric}",
                f"mean_test_{metric}",
                f"mean_{metric}",
                f"{metric}_mean",
                f"validation_{metric}",
                f"val_{metric}",
                f"test_{metric}",
                metric,
            ],
        )
        if mean_col is None:
            continue

        std_col = detect_column(
            headers,
            [
                f"std_validation_{metric}",
                f"std_val_{metric}",
                f"std_test_{metric}",
                f"validation_{metric}_standard_deviation",
                f"validation_{metric}_std",
                f"val_{metric}_standard_deviation",
                f"val_{metric}_std",
                f"test_{metric}_standard_deviation",
                f"test_{metric}_std",
                f"{metric}_standard_deviation",
                f"{metric}_std",
                f"std_{metric}",
                f"sd_{metric}",
            ],
        )

        print(f"[ALPHA] metric={metric} | mean_col={mean_col} | std_col={std_col}")

        x: list[float] = []
        y: list[float] = []
        yerr: list[float] = []
        for row in rows:
            alpha = safe_float(row.get(alpha_col))
            score = safe_float(row.get(mean_col))
            if alpha is None or score is None:
                continue
            x.append(alpha)
            y.append(score)
            yerr.append(
                safe_float(row.get(std_col)) or 0.0 if std_col is not None else 0.0
            )

        if not x:
            continue

        order = np.argsort(np.asarray(x))
        x_array = np.asarray(x)[order]
        y_array = np.asarray(y)[order]
        err_array = np.asarray(yerr)[order]
        best_index = int(np.argmax(y_array))

        fig, axis = plt.subplots(figsize=(7.2, 4.8))
        axis.plot(x_array, y_array, marker="o")
        if np.any(err_array > 0):
            axis.fill_between(
                x_array,
                y_array - err_array,
                y_array + err_array,
                alpha=0.18,
            )
        axis.axvline(
            x_array[best_index],
            linestyle="--",
            linewidth=1,
        )
        axis.scatter(
            [x_array[best_index]],
            [y_array[best_index]],
            s=80,
            zorder=3,
        )
        axis.annotate(
            (
                f"Best α={x_array[best_index]:.2f}\n"
                f"{DISPLAY_NAMES[metric]}={y_array[best_index]:.4f}"
            ),
            xy=(x_array[best_index], y_array[best_index]),
            xytext=(8, 12),
            textcoords="offset points",
        )
        axis.set_xlabel("Hybrid Transformer weight (α)")
        axis.set_ylabel(f"Validation {DISPLAY_NAMES[metric]}")
        axis.set_title(f"Global alpha ablation: {DISPLAY_NAMES[metric]}")
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_root / "figures" / f"alpha_{metric}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            output_root / "figures" / f"alpha_{metric}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
        plotted = True

    return plotted


# ============================================================================
# PAPER SUMMARY
# ============================================================================


def find_model(
    models: list[ModelResult],
    name: str,
) -> ModelResult | None:
    for model in models:
        if model.name == name:
            return model
    return None


def generate_improvement_table(
    models: list[ModelResult],
    output_root: Path,
) -> list[dict[str, Any]]:
    hybrid = find_model(models, "Hybrid Transformer")
    late = find_model(models, "Late Fusion (ours)")
    if hybrid is None or late is None:
        return []

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        if metric not in hybrid.metrics or metric not in late.metrics:
            continue
        baseline = hybrid.metrics[metric].mean
        proposed = late.metrics[metric].mean
        delta = proposed - baseline
        relative = 100.0 * delta / baseline if baseline != 0 else math.nan
        rows.append(
            {
                "Metric": DISPLAY_NAMES[metric],
                "Hybrid Transformer": baseline,
                "Late Fusion": proposed,
                "Absolute Delta": delta,
                "Relative Change (%)": relative,
            }
        )

    write_csv(
        output_root / "tables" / "Improvement_Over_Hybrid.csv",
        rows,
    )
    return rows


def generate_results_text(
    models: list[ModelResult],
    improvement_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    best = best_values(models)
    best_models: dict[str, list[str]] = {}
    for metric in METRICS:
        if metric not in best:
            continue
        best_models[metric] = [
            model.name
            for model in models
            if metric in model.metrics
            and math.isclose(
                model.metrics[metric].mean,
                best[metric],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]

    paragraphs: list[str] = [
        "# Automatically generated results summary\n",
        (
            "The following text is a draft generated directly from the "
            "experiment summaries and should be reviewed before submission.\n"
        ),
    ]

    for metric in ("f1", "mcc", "roc_auc", "pr_auc"):
        if metric not in best:
            continue
        names = ", ".join(best_models[metric])
        paragraphs.append(
            f"The highest {DISPLAY_NAMES[metric]} was obtained by "
            f"{names}, with a mean value of {best[metric]:.4f}."
        )

    if improvement_rows:
        selected = {row["Metric"]: row for row in improvement_rows}
        if "ROC-AUC" in selected and "PR-AUC" in selected and "MCC" in selected:
            roc = selected["ROC-AUC"]["Absolute Delta"]
            pr = selected["PR-AUC"]["Absolute Delta"]
            mcc = selected["MCC"]["Absolute Delta"]
            paragraphs.append(
                "Compared with the Hybrid Transformer, Late Fusion "
                f"changed ROC-AUC by {roc:+.4f}, PR-AUC by {pr:+.4f}, "
                f"and MCC by {mcc:+.4f}."
            )
        if "F1" in selected:
            f1 = selected["F1"]["Absolute Delta"]
            paragraphs.append(
                "The corresponding change in F1 was "
                f"{f1:+.4f}, indicating that the principal benefit of "
                "late fusion is more evident in ranking quality and MCC "
                "than in threshold-dependent F1."
            )

    (output_root / "markdown" / "results_summary.md").write_text(
        "\n\n".join(paragraphs) + "\n",
        encoding="utf-8",
    )


def generate_manuscript_results_section(
    models: list[ModelResult],
    improvement_rows: list[dict[str, Any]],
    output_root: Path,
    decimals: int,
) -> None:
    """
    Generate an editorial Results section in Markdown and LaTeX.

    The content is derived exclusively from aggregate results,
    the comparison against the Hybrid Transformer and, when
    available, the supplementary statistical table.
    """

    ordered_models = order_models(
        models,
        PAPER_MODEL_ORDER,
    )

    hybrid = find_model(
        ordered_models,
        "Hybrid Transformer",
    )

    proposed = find_model(
        ordered_models,
        "Late Fusion (ours)",
    )

    if hybrid is None or proposed is None:
        return

    output_directory = output_root / "paper" / "text"

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    improvement_by_metric = {
        str(row.get("Metric", "")).strip(): row for row in improvement_rows
    }

    def metric_result(
        model: ModelResult,
        metric_key: str,
    ) -> MetricResult | None:
        return model.metrics.get(metric_key)

    def metric_text(
        model: ModelResult,
        metric_key: str,
    ) -> str:
        result = metric_result(
            model,
            metric_key,
        )

        if result is None:
            return "not available"

        return f"{result.mean:.{decimals}f} ± {result.std:.{decimals}f}"

    def improvement_values(
        metric_key: str,
    ) -> tuple[float | None, float | None]:
        display_name = DISPLAY_NAMES[metric_key]

        improvement_row = improvement_by_metric.get(display_name)

        if improvement_row is None:
            return None, None

        parsed_absolute_delta = safe_float(improvement_row.get("Absolute Delta"))

        parsed_relative_change = safe_float(improvement_row.get("Relative Change (%)"))

        return (
            parsed_absolute_delta,
            parsed_relative_change,
        )

    # ------------------------------------------------------------------
    # Identification of the best models
    # ------------------------------------------------------------------

    best = best_values(
        ordered_models,
        PAPER_METRICS,
    )

    best_model_names: dict[str, list[str]] = {}

    for metric_name in PAPER_METRICS:
        best_value = best.get(metric_name)

        if best_value is None:
            continue

        best_model_names[metric_name] = [
            PAPER_MODEL_NAMES.get(
                model.name,
                model.name,
            )
            for model in ordered_models
            if (
                metric_name in model.metrics
                and math.isclose(
                    model.metrics[metric_name].mean,
                    best_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ]

    # ------------------------------------------------------------------
    # Paired statistics
    # ------------------------------------------------------------------

    statistics_path = (
        output_root / "supplementary" / "tables" / "TableS2_StatisticalComparison.csv"
    )

    statistics_by_metric: dict[str, dict[str, str]] = {}

    if statistics_path.is_file():
        try:
            statistics_rows = read_csv(statistics_path)

        except (OSError, csv.Error):
            statistics_rows = []

        for row in statistics_rows:
            metric_name = str(row.get("metric", "")).strip()

            if metric_name:
                statistics_by_metric[metric_name] = row

    # ------------------------------------------------------------------
    # Valores principales
    # ------------------------------------------------------------------

    f1_delta, f1_relative = improvement_values("f1")
    mcc_delta, mcc_relative = improvement_values("mcc")
    roc_delta, roc_relative = improvement_values("roc_auc")
    pr_delta, pr_relative = improvement_values("pr_auc")
    bal_delta, bal_relative = improvement_values("balanced_accuracy")

    markdown_paragraphs: list[str] = [
        "# Results",
        (
            f"Six anti-money-laundering models were evaluated using "
            f"{len(PAPER_METRICS)} main performance measures. The comparison "
            f"included standalone neural and graph-based models, the Hybrid "
            f"Transformer baseline, a Transformer + GraphSAGE fusion model, "
            f"and the proposed late-fusion strategy. All reported aggregate "
            f"results were computed across five experimental runs."
        ),
        (
            "Late Fusion achieved "
            f"{metric_text(proposed, 'f1')} F1, "
            f"{metric_text(proposed, 'mcc')} MCC, "
            f"{metric_text(proposed, 'roc_auc')} ROC-AUC, "
            f"{metric_text(proposed, 'pr_auc')} PR-AUC, and "
            f"{metric_text(proposed, 'balanced_accuracy')} balanced "
            f"accuracy."
        ),
    ]

    comparison_parts: list[str] = []

    comparison_data = [
        (
            "F1",
            f1_delta,
            f1_relative,
        ),
        (
            "MCC",
            mcc_delta,
            mcc_relative,
        ),
        (
            "ROC-AUC",
            roc_delta,
            roc_relative,
        ),
        (
            "PR-AUC",
            pr_delta,
            pr_relative,
        ),
        (
            "balanced accuracy",
            bal_delta,
            bal_relative,
        ),
    ]

    for (
        metric_label,
        absolute_delta,
        relative_change,
    ) in comparison_data:
        if absolute_delta is None:
            continue

        sentence = f"{metric_label} by {absolute_delta:+.{decimals}f}"

        if relative_change is not None:
            sentence += f" ({format_signed_percentage(relative_change)})"

        comparison_parts.append(sentence)

    if comparison_parts:
        markdown_paragraphs.append(
            "Relative to the Hybrid Transformer, the proposed method "
            "changed "
            + ", ".join(comparison_parts[:-1])
            + (
                f", and {comparison_parts[-1]}."
                if len(comparison_parts) > 1
                else f"{comparison_parts[-1]}."
            )
        )

    leading_metrics: list[str] = []

    for metric_name in PAPER_METRICS:
        names = best_model_names.get(
            metric_name,
            [],
        )

        if "Late Fusion (ours)" in names:
            leading_metrics.append(DISPLAY_NAMES[metric_name])

    if leading_metrics:
        markdown_paragraphs.append(
            "Late Fusion obtained the highest mean result for "
            + ", ".join(leading_metrics)
            + " among the evaluated configurations. These results indicate "
            "that the late-fusion mechanism is particularly beneficial for "
            "the metrics in which the proposed model leads the comparison."
        )
    else:
        markdown_paragraphs.append(
            "Although Late Fusion did not obtain the highest mean value "
            "for every metric, its combined results provide a balanced "
            "trade-off across threshold-dependent classification quality, "
            "ranking performance, and correlation-based evaluation."
        )

    if mcc_delta is not None and pr_delta is not None and roc_delta is not None:
        if mcc_delta > 0 and pr_delta > 0 and roc_delta > 0:
            markdown_paragraphs.append(
                "The simultaneous improvement in MCC, ROC-AUC, and PR-AUC "
                "suggests that the proposed fusion strategy improves both "
                "the global discrimination of transactions and the quality "
                "of predictions for the minority illicit class. This is "
                "especially relevant because accuracy alone can be "
                "misleading in highly imbalanced anti-money-laundering data."
            )
        else:
            markdown_paragraphs.append(
                "The metric changes were not uniformly positive. Therefore, "
                "the proposed method should be interpreted as a trade-off "
                "rather than as a universal improvement across all measures."
            )

    # ------------------------------------------------------------------
    # Statistical summary
    # ------------------------------------------------------------------

    statistical_sentences: list[str] = []

    for metric_name in (
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    ):
        row = statistics_by_metric.get(metric_name)

        if row is None:
            continue

        mean_difference = safe_float(row.get("mean_difference"))

        wilcoxon_p_value = safe_float(row.get("wilcoxon_p_value"))

        permutation_p_value = safe_float(row.get("sign_flip_permutation_p_value"))

        rank_biserial = safe_float(row.get("rank_biserial_correlation"))

        components = [
            DISPLAY_NAMES[metric_name],
        ]

        if mean_difference is not None:
            components.append(f"mean difference {mean_difference:+.{decimals}f}")

        if wilcoxon_p_value is not None:
            components.append(f"Wilcoxon p={wilcoxon_p_value:.4g}")

        if permutation_p_value is not None:
            components.append(f"permutation p={permutation_p_value:.4g}")

        if rank_biserial is not None:
            components.append(
                f"rank-biserial correlation {rank_biserial:+.{decimals}f}"
            )

        statistical_sentences.append(", ".join(components))

    if statistical_sentences:
        markdown_paragraphs.append(
            "The paired statistical analysis produced the following "
            "metric-level results: "
            + "; ".join(statistical_sentences)
            + ". Because only five paired experimental runs were available, "
            "the statistical evidence should be interpreted together with "
            "the effect sizes and the direction of the seed-level "
            "differences, rather than from p-values alone."
        )

    markdown_paragraphs.append(
        "Overall, the results support Late Fusion as a competitive "
        "fusion strategy for Bitcoin transaction classification. The model "
        "combines the complementary information captured by the Hybrid "
        "Transformer and GraphSAGE components while preserving reproducible "
        "evaluation across the same experimental seeds. The generated text "
        "should nevertheless be reviewed and adapted to the final narrative "
        "and terminology of the manuscript before submission."
    )

    markdown_path = output_directory / "Results_Draft.md"

    markdown_path.write_text(
        "\n\n".join(markdown_paragraphs) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # LaTeX version
    # ------------------------------------------------------------------

    latex_paragraphs = [
        r"\section{Results}",
    ]

    for paragraph in markdown_paragraphs[1:]:
        latex_paragraphs.extend(
            [
                "",
                latex_escape(paragraph),
            ]
        )

    latex_path = output_directory / "Results_Draft.tex"

    latex_path.write_text(
        "\n".join(latex_paragraphs) + "\n",
        encoding="utf-8",
    )


# ============================================================================
# V5 PUBLICATION UTILITIES
# ============================================================================


def setup_publication_style() -> None:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def generate_publication_check(
    output_root: Path,
    models: list[ModelResult],
    statistics_generated: bool,
    alpha_generated: bool,
    model_figures_generated: bool,
    decimals: int,
) -> bool:
    """
    Check:

    1. Existence of expected resources.
    2. Numeric consistency between consolidated_results.json
       and Table1_MainResults.csv.
    """

    required_files = [
        "README.md",
        "MANIFEST.md",
        "paper/PAPER_EVIDENCE_MAP.md",
        "paper/tables/Table1_MainResults.csv",
        "paper/markdown/Table1_MainResults.md",
        "paper/latex/Table1_MainResults.tex",
        "paper/tables/Table2_FusionAblation.csv",
        "paper/markdown/Table2_FusionAblation.md",
        "paper/latex/Table2_FusionAblation.tex",
        "supplementary/tables/TableS1_AllModels.csv",
        "supplementary/markdown/TableS1_AllModels.md",
        "supplementary/latex/TableS1_AllModels.tex",
        "tables/Improvement_Over_Hybrid.csv",
        "json/consolidated_results.json",
        "markdown/results_summary.md",
        "paper/text/Results_Draft.md",
        "paper/text/Results_Draft.tex",
        "generation_report.txt",
        "captions/table_captions.md",
        "captions/figure_captions.md",
        "reproducibility/REPRODUCIBILITY.md",
        "reproducibility/reproducibility_metadata.json",
        "paper/integration/ManuscriptAssets.tex",
        "supplementary/integration/SupplementaryAssets.tex",
    ]

    if statistics_generated:
        required_files.extend(
            [
                "paper/tables/Table3_StatisticalComparison.csv",
                "paper/markdown/Table3_StatisticalComparison.md",
                "paper/latex/Table3_StatisticalComparison.tex",
                "supplementary/tables/TableS2_StatisticalComparison.csv",
                "supplementary/markdown/TableS2_StatisticalComparison.md",
                "supplementary/latex/TableS2_StatisticalComparison.tex",
            ]
        )

    if model_figures_generated:
        for metric in ("f1", "mcc", "roc_auc", "pr_auc"):
            required_files.extend(
                [
                    f"figures/models_{metric}.png",
                    f"figures/models_{metric}.pdf",
                ]
            )

    if alpha_generated:
        for metric in ("f1", "mcc", "roc_auc", "pr_auc"):
            required_files.extend(
                [
                    f"figures/alpha_{metric}.png",
                    f"figures/alpha_{metric}.pdf",
                ]
            )

    missing_files = [
        relative_path
        for relative_path in required_files
        if not (output_root / relative_path).is_file()
    ]

    consistency_errors: list[str] = []
    checked_values = 0

    json_models: dict[
        str,
        dict[str, Any],
    ] = {}

    tolerance = 0.5 * (10 ** (-decimals)) + 1e-12

    def parse_table_metric(
        raw_value: Any,
        current_model_name: str,
        current_metric_label: str,
    ) -> tuple[float, float] | None:
        """
        Interpret a cell with format:

            mean ± standard_deviation
        """

        text = str(raw_value).strip()

        if not text or text == "—":
            consistency_errors.append(
                f"Empty value for {current_model_name}/{current_metric_label}."
            )
            return None

        if "±" not in text:
            consistency_errors.append(
                f"Invalid format for "
                f"{current_model_name}/"
                f"{current_metric_label}: "
                f"{text!r}"
            )
            return None

        parts = text.split("±")

        if len(parts) != 2:
            consistency_errors.append(
                f"Invalid format for "
                f"{current_model_name}/"
                f"{current_metric_label}: "
                f"{text!r}"
            )
            return None

        mean_value = safe_float(parts[0].strip())

        std_value = safe_float(parts[1].strip())

        if mean_value is None or std_value is None:
            consistency_errors.append(
                f"Non-numeric value for "
                f"{current_model_name}/"
                f"{current_metric_label}: "
                f"{text!r}"
            )
            return None

        return (
            mean_value,
            std_value,
        )

    json_path = output_root / "json" / "consolidated_results.json"

    table_path = output_root / "paper" / "tables" / "Table1_MainResults.csv"

    # ------------------------------------------------------------------
    # Consistencia entre JSON consolidado y Table 1
    # ------------------------------------------------------------------

    if json_path.is_file() and table_path.is_file():
        try:
            consolidated = json.loads(json_path.read_text(encoding="utf-8"))

            table_rows_data = read_csv(table_path)

        except (OSError, json.JSONDecodeError, csv.Error) as exc:
            consistency_errors.append(f"Could not read comparison files: {exc}")

        else:
            raw_models = consolidated.get("models", [])

            if not isinstance(raw_models, list):
                consistency_errors.append("The JSON 'models' field is not a list.")
                raw_models = []

            for model_data in raw_models:
                if not isinstance(model_data, dict):
                    consistency_errors.append(
                        "The JSON contains an invalid model entry."
                    )
                    continue

                model_name = model_data.get("name")

                if not isinstance(model_name, str):
                    consistency_errors.append(
                        "The JSON contains a model without a valid name."
                    )
                    continue

                publication_name = PAPER_MODEL_NAMES.get(
                    model_name,
                    model_name,
                )

                if publication_name in json_models:
                    consistency_errors.append(
                        f"Duplicate model in JSON: {publication_name}"
                    )
                    continue

                json_models[publication_name] = model_data

            csv_models: dict[str, dict[str, str]] = {}

            for row in table_rows_data:
                model_name = str(row.get("Model", "")).strip()

                if not model_name:
                    consistency_errors.append(
                        "Table1 contains a row without a model name."
                    )
                    continue

                if model_name in csv_models:
                    consistency_errors.append(
                        f"Duplicate model in Table1: {model_name}"
                    )
                    continue

                csv_models[model_name] = row

            for model_name in json_models:
                if model_name not in csv_models:
                    consistency_errors.append(
                        f"Model missing from Table1: {model_name}"
                    )

            for model_name in csv_models:
                if model_name not in json_models:
                    consistency_errors.append(f"Model missing from JSON: {model_name}")

            for model_name in sorted(set(json_models) & set(csv_models)):
                json_model = json_models[model_name]
                csv_row = csv_models[model_name]

                json_metrics = json_model.get("metrics", {})

                if not isinstance(json_metrics, dict):
                    consistency_errors.append(
                        f"Invalid metrics in JSON for {model_name}."
                    )
                    continue

                for metric_name in PAPER_METRICS:
                    metric_label = DISPLAY_NAMES[metric_name]

                    json_metric = json_metrics.get(metric_name)

                    if not isinstance(json_metric, dict):
                        consistency_errors.append(
                            f"Metric missing from JSON: {model_name}/{metric_name}"
                        )
                        continue

                    if metric_label not in csv_row:
                        consistency_errors.append(
                            f"Column missing from Table1: {metric_label}"
                        )
                        continue

                    json_mean = safe_float(json_metric.get("mean"))

                    json_std = safe_float(json_metric.get("standard_deviation"))

                    if json_mean is None:
                        consistency_errors.append(
                            f"Invalid mean in JSON: {model_name}/{metric_name}"
                        )
                        continue

                    if json_std is None:
                        consistency_errors.append(
                            f"Invalid standard deviation in JSON: "
                            f"{model_name}/{metric_name}"
                        )
                        continue

                    parsed = parse_table_metric(
                        csv_row.get(metric_label),
                        model_name,
                        metric_label,
                    )

                    if parsed is None:
                        continue

                    csv_mean, csv_std = parsed

                    checked_values += 2

                    if not math.isclose(
                        csv_std,
                        json_std,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different standard deviation in TableS1 for "
                            f"{model_name}/{metric_label}: "
                            f"JSON={json_std:.{decimals}f}, "
                            f"CSV={csv_std:.{decimals}f}"
                        )

                    # ------------------------------------------------------------------
                    # Consistencia entre Table 3 y Table S2
                    # ------------------------------------------------------------------

                    table3_path = (
                        output_root
                        / "paper"
                        / "tables"
                        / "Table3_StatisticalComparison.csv"
                    )

                    table_s2_path = (
                        output_root
                        / "supplementary"
                        / "tables"
                        / "TableS2_StatisticalComparison.csv"
                    )

                    if (
                        statistics_generated
                        and table3_path.is_file()
                        and table_s2_path.is_file()
                    ):
                        try:
                            table3_rows = read_csv(table3_path)
                            table_s2_rows = read_csv(table_s2_path)
                        except (OSError, csv.Error) as exc:
                            consistency_errors.append(
                                f"Could not read statistical tables: {exc}"
                            )

                    if not math.isclose(
                        csv_std,
                        json_std,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different standard deviation for "
                            f"{model_name}/{metric_label}: "
                            f"JSON={json_std:.{decimals}f}, "
                            f"CSV={csv_std:.{decimals}f}"
                        )

    fusion_table_path = output_root / "paper" / "tables" / "Table2_FusionAblation.csv"

    if json_path.is_file() and fusion_table_path.is_file():
        try:
            fusion_rows = read_csv(fusion_table_path)

        except (OSError, csv.Error) as exc:
            consistency_errors.append(
                f"Could not read Table2_FusionAblation.csv: {exc}"
            )

        else:
            fusion_csv_models: dict[str, dict[str, str]] = {}

            for row in fusion_rows:
                model_name = str(row.get("Model", "")).strip()

                if not model_name:
                    consistency_errors.append(
                        "Table2 contains a row without a model name."
                    )
                    continue

                if model_name in fusion_csv_models:
                    consistency_errors.append(
                        f"Duplicate model in Table2: {model_name}"
                    )
                    continue

                fusion_csv_models[model_name] = row

            expected_fusion_names = {
                PAPER_MODEL_NAMES.get(model.name, model.name)
                for model in models
                if (model.name == "Hybrid Transformer" or model.name in FUSION_MODELS)
            }

            for model_name in expected_fusion_names:
                if model_name not in fusion_csv_models:
                    consistency_errors.append(
                        f"Model missing from Table2: {model_name}"
                    )

            for model_name in fusion_csv_models:
                if model_name not in expected_fusion_names:
                    consistency_errors.append(
                        f"Unexpected model in Table2: {model_name}"
                    )

            hybrid_json_model = json_models.get(
                PAPER_MODEL_NAMES.get(
                    "Hybrid Transformer",
                    "Hybrid Transformer",
                )
            )

            hybrid_mcc = None

            if isinstance(hybrid_json_model, dict):
                hybrid_metrics = hybrid_json_model.get("metrics", {})

                if isinstance(hybrid_metrics, dict):
                    hybrid_mcc_block = hybrid_metrics.get("mcc")

                    if isinstance(hybrid_mcc_block, dict):
                        hybrid_mcc = safe_float(hybrid_mcc_block.get("mean"))

            if hybrid_mcc is None:
                consistency_errors.append(
                    "Could not obtain Hybrid Transformer MCC from JSON."
                )

            for model_name in sorted(expected_fusion_names & set(fusion_csv_models)):
                json_model = json_models.get(model_name)
                csv_row = fusion_csv_models[model_name]

                if not isinstance(json_model, dict):
                    consistency_errors.append(
                        f"Invalid model in JSON for Table2: {model_name}"
                    )
                    continue

                json_metrics = json_model.get("metrics", {})

                if not isinstance(json_metrics, dict):
                    consistency_errors.append(
                        f"Invalid metrics in JSON for {model_name}."
                    )
                    continue

                for metric_name in PAPER_METRICS:
                    metric_label = DISPLAY_NAMES[metric_name]
                    json_metric = json_metrics.get(metric_name)

                    if not isinstance(json_metric, dict):
                        consistency_errors.append(
                            f"Metric missing from JSON for Table2: "
                            f"{model_name}/{metric_name}"
                        )
                        continue

                    parsed = parse_table_metric(
                        csv_row.get(metric_label),
                        model_name,
                        metric_label,
                    )

                    if parsed is None:
                        continue

                    csv_mean, csv_std = parsed

                    json_mean = safe_float(json_metric.get("mean"))

                    json_std = safe_float(json_metric.get("standard_deviation"))

                    if json_mean is None or json_std is None:
                        consistency_errors.append(
                            f"Invalid values in JSON for {model_name}/{metric_name}"
                        )
                        continue

                    checked_values += 2

                    if not math.isclose(
                        csv_mean,
                        json_mean,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different mean in Table2 for "
                            f"{model_name}/{metric_label}: "
                            f"JSON={json_mean:.{decimals}f}, "
                            f"CSV={csv_mean:.{decimals}f}"
                        )

                    if not math.isclose(
                        csv_std,
                        json_std,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different standard deviation in Table2 for "
                            f"{model_name}/{metric_label}: "
                            f"JSON={json_std:.{decimals}f}, "
                            f"CSV={csv_std:.{decimals}f}"
                        )

                if hybrid_mcc is not None:
                    json_mcc_block = json_metrics.get("mcc")

                    if not isinstance(json_mcc_block, dict):
                        consistency_errors.append(
                            f"MCC missing from JSON for {model_name}."
                        )
                        continue

                    model_mcc = safe_float(json_mcc_block.get("mean"))

                    if model_mcc is None:
                        consistency_errors.append(
                            f"Invalid MCC in JSON for {model_name}."
                        )
                        continue

                    delta_column = "Delta vs Hybrid Transformer (MCC)"

                    csv_delta = safe_float(csv_row.get(delta_column))

                    if csv_delta is None:
                        consistency_errors.append(
                            f"Invalid delta in Table2 for {model_name}."
                        )
                        continue

                    expected_delta = model_mcc - hybrid_mcc
                    checked_values += 1

                    if not math.isclose(
                        csv_delta,
                        expected_delta,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different delta in Table2 for {model_name}: "
                            f"esperado={expected_delta:+.{decimals}f}, "
                            f"CSV={csv_delta:+.{decimals}f}"
                        )

    supplementary_table_path = (
        output_root / "supplementary" / "tables" / "TableS1_AllModels.csv"
    )

    if (
        supplementary_table_path.is_file()
        and "json_models" in locals()
        and "tolerance" in locals()
        and "parse_table_metric" in locals()
    ):
        try:
            supplementary_rows = read_csv(supplementary_table_path)

        except (OSError, csv.Error) as exc:
            consistency_errors.append(f"Could not read TableS1_AllModels.csv: {exc}")

        else:
            supplementary_csv_models: dict[
                str,
                dict[str, str],
            ] = {}

            for row in supplementary_rows:
                model_name = str(row.get("Model", "")).strip()

                if not model_name:
                    consistency_errors.append(
                        "TableS1 contains a row without a model name."
                    )
                    continue

                if model_name in supplementary_csv_models:
                    consistency_errors.append(
                        f"Duplicate model in TableS1: {model_name}"
                    )
                    continue

                supplementary_csv_models[model_name] = row

            expected_model_names = set(json_models)

            for model_name in expected_model_names:
                if model_name not in supplementary_csv_models:
                    consistency_errors.append(
                        f"Model missing from TableS1: {model_name}"
                    )

            for model_name in supplementary_csv_models:
                if model_name not in expected_model_names:
                    consistency_errors.append(
                        f"Unexpected model in TableS1: {model_name}"
                    )

            for model_name in sorted(
                expected_model_names & set(supplementary_csv_models)
            ):
                json_model = json_models.get(model_name)
                csv_row = supplementary_csv_models[model_name]

                if not isinstance(json_model, dict):
                    consistency_errors.append(
                        f"Invalid model in JSON for TableS1: {model_name}"
                    )
                    continue

                json_metrics = json_model.get("metrics", {})

                if not isinstance(json_metrics, dict):
                    consistency_errors.append(
                        f"Invalid metrics in JSON for TableS1/{model_name}."
                    )
                    continue

                for metric_name in PAPER_METRICS:
                    metric_label = DISPLAY_NAMES[metric_name]
                    json_metric = json_metrics.get(metric_name)

                    if not isinstance(json_metric, dict):
                        consistency_errors.append(
                            f"Metric missing from JSON for TableS1: "
                            f"{model_name}/{metric_name}"
                        )
                        continue

                    if metric_label not in csv_row:
                        consistency_errors.append(
                            f"Column missing from TableS1: {metric_label}"
                        )
                        continue

                    parsed = parse_table_metric(
                        csv_row.get(metric_label),
                        model_name,
                        metric_label,
                    )

                    if parsed is None:
                        continue

                    csv_mean, csv_std = parsed

                    json_mean = safe_float(json_metric.get("mean"))

                    json_std = safe_float(json_metric.get("standard_deviation"))

                    if json_mean is None:
                        consistency_errors.append(
                            f"Invalid mean in JSON for TableS1: "
                            f"{model_name}/{metric_name}"
                        )
                        continue

                    if json_std is None:
                        consistency_errors.append(
                            f"Invalid standard deviation in JSON for TableS1: "
                            f"{model_name}/{metric_name}"
                        )
                        continue

                    checked_values += 2

                    if not math.isclose(
                        csv_mean,
                        json_mean,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different mean in TableS1 for "
                            f"{model_name}/{metric_label}: "
                            f"JSON={json_mean:.{decimals}f}, "
                            f"CSV={csv_mean:.{decimals}f}"
                        )

                    if not math.isclose(
                        csv_std,
                        json_std,
                        rel_tol=0.0,
                        abs_tol=tolerance,
                    ):
                        consistency_errors.append(
                            f"Different standard deviation in TableS1 for "
                            f"{model_name}/{metric_label}: "
                            f"JSON={json_std:.{decimals}f}, "
                            f"CSV={csv_std:.{decimals}f}"
                        )

    # ------------------------------------------------------------------
    # Validation estructural de Table 3 y Table S2
    # ------------------------------------------------------------------

    table3_path = output_root / "paper" / "tables" / "Table3_StatisticalComparison.csv"

    table_s2_path = (
        output_root / "supplementary" / "tables" / "TableS2_StatisticalComparison.csv"
    )

    if statistics_generated and table3_path.is_file() and table_s2_path.is_file():
        try:
            table3_rows = read_csv(table3_path)
            table_s2_rows = read_csv(table_s2_path)

        except (OSError, csv.Error) as exc:
            consistency_errors.append(f"Could not read statistical tables: {exc}")

        else:
            if not table3_rows:
                consistency_errors.append(
                    "Table3_StatisticalComparison.csv contains no rows."
                )

            if not table_s2_rows:
                consistency_errors.append(
                    "TableS2_StatisticalComparison.csv contains no rows."
                )

            table3_metric_aliases = {
                display_name: metric_name
                for metric_name, display_name in DISPLAY_NAMES.items()
            }

            table3_metrics: list[str] = []

            for row in table3_rows:
                display_metric = str(row.get("Metric", "")).strip()

                if not display_metric:
                    consistency_errors.append("Table3 contains a row without a metric.")
                    continue

                metric_name = table3_metric_aliases.get(display_metric)

                if metric_name is None:
                    consistency_errors.append(
                        f"Unexpected metric in Table3: {display_metric}"
                    )
                    continue

                table3_metrics.append(metric_name)

            table_s2_metrics: list[str] = []

            for row in table_s2_rows:
                metric_name = str(row.get("metric", "")).strip()

                if not metric_name:
                    consistency_errors.append(
                        "TableS2 contains a row without a metric."
                    )
                    continue

                if metric_name not in DISPLAY_NAMES:
                    consistency_errors.append(
                        f"Unexpected metric in TableS2: {metric_name}"
                    )
                    continue

                table_s2_metrics.append(metric_name)

            duplicated_table3_metrics = sorted(
                {
                    metric_name
                    for metric_name in table3_metrics
                    if table3_metrics.count(metric_name) > 1
                }
            )

            for metric_name in duplicated_table3_metrics:
                consistency_errors.append(
                    f"Duplicate metric in Table3: {DISPLAY_NAMES[metric_name]}"
                )

            duplicated_table_s2_metrics = sorted(
                {
                    metric_name
                    for metric_name in table_s2_metrics
                    if table_s2_metrics.count(metric_name) > 1
                }
            )

            for metric_name in duplicated_table_s2_metrics:
                consistency_errors.append(f"Duplicate metric in TableS2: {metric_name}")

            table3_metric_set = set(table3_metrics)
            table_s2_metric_set = set(table_s2_metrics)

            for metric_name in sorted(table_s2_metric_set - table3_metric_set):
                consistency_errors.append(
                    f"Metric missing from Table3: {DISPLAY_NAMES[metric_name]}"
                )

            for metric_name in sorted(table3_metric_set - table_s2_metric_set):
                consistency_errors.append(f"Metric missing from TableS2: {metric_name}")

            if len(table3_rows) != len(table_s2_rows):
                consistency_errors.append(
                    f"The statistical tables have different numbers "
                    f"of rows: Table3={len(table3_rows)}, "
                    f"TableS2={len(table_s2_rows)}"
                )

    # ------------------------------------------------------------------
    # Editorial checks for Markdown and LaTeX files
    # ------------------------------------------------------------------

    editorial_text_files = sorted(
        {
            relative_path
            for relative_path in required_files
            if Path(relative_path).suffix.lower()
            in {
                ".md",
                ".tex",
            }
        }
    )

    latex_labels: dict[str, list[str]] = {}

    for relative_path in editorial_text_files:
        asset_path = output_root / relative_path

        if not asset_path.is_file():
            # La ausencia ya se registra mediante missing_files.
            continue

        try:
            asset_text = asset_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(
                f"Could not read editorial resource {relative_path}: {exc}"
            )
            continue

        if not asset_text.strip():
            consistency_errors.append(f"Empty editorial resource: {relative_path}")
            continue

        if asset_path.suffix.lower() != ".tex":
            continue

        is_table_latex_file = "/latex/" in f"/{relative_path}"

        if not is_table_latex_file:
            continue

        if "\\caption{" not in asset_text:
            consistency_errors.append(f"LaTeX file without caption: {relative_path}")

        labels_in_file = re.findall(
            r"\\label[{]([^{}]+)[}]",
            asset_text,
        )

        if not labels_in_file:
            consistency_errors.append(f"LaTeX file without label: {relative_path}")

        for label in labels_in_file:
            normalized_label = label.strip()

            if not normalized_label:
                consistency_errors.append(f"Empty LaTeX label: {relative_path}")
                continue

            latex_labels.setdefault(
                normalized_label,
                [],
            ).append(relative_path)

    for label, source_files in sorted(latex_labels.items()):
        if len(source_files) <= 1:
            continue

        consistency_errors.append(
            f"Label LaTeX duplicado '{label}' in: {', '.join(source_files)}"
        )

    table_captions_path = output_root / "captions" / "table_captions.md"

    if table_captions_path.is_file():
        try:
            table_captions_text = table_captions_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read table_captions.md: {exc}")

        else:
            expected_table_caption_headings = [
                "## Table 1.",
                "## Table 2.",
                "## Table S1.",
            ]

            if statistics_generated:
                expected_table_caption_headings.extend(
                    [
                        "## Table 3.",
                        "## Table S2.",
                    ]
                )

            for heading in expected_table_caption_headings:
                if heading not in table_captions_text:
                    consistency_errors.append(
                        f"Caption missing from table_captions.md: {heading}"
                    )

    figure_captions_path = output_root / "captions" / "figure_captions.md"

    if figure_captions_path.is_file():
        try:
            figure_captions_text = figure_captions_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read figure_captions.md: {exc}")

        else:
            expected_figure_caption_headings: list[str] = []

            if model_figures_generated:
                expected_figure_caption_headings.extend(
                    [
                        "## Figure 1.",
                        "## Figure 2.",
                        "## Figure 3.",
                        "## Figure 4.",
                    ]
                )

            if alpha_generated:
                expected_figure_caption_headings.extend(
                    [
                        "## Figure S1.",
                        "## Figure S2.",
                        "## Figure S3.",
                        "## Figure S4.",
                    ]
                )

            for heading in expected_figure_caption_headings:
                if heading not in figure_captions_text:
                    consistency_errors.append(
                        f"Caption missing from figure_captions.md: {heading}"
                    )

    # ------------------------------------------------------------------
    # Validation of the generated Results section
    # ------------------------------------------------------------------

    results_markdown_path = output_root / "paper" / "text" / "Results_Draft.md"

    results_latex_path = output_root / "paper" / "text" / "Results_Draft.tex"

    if results_markdown_path.is_file():
        try:
            results_markdown_text = results_markdown_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read Results_Draft.md: {exc}")

        else:
            required_results_terms = [
                "# Results",
                "Late Fusion",
                "Hybrid Transformer",
                "F1",
                "MCC",
                "ROC-AUC",
                "PR-AUC",
            ]

            for term in required_results_terms:
                if term not in results_markdown_text:
                    consistency_errors.append(
                        f"Content missing from Results_Draft.md: {term}"
                    )

            results_word_count = len(results_markdown_text.split())

            if results_word_count < 250:
                consistency_errors.append(
                    f"Results_Draft.md es demasiado breve: "
                    f"{results_word_count} palabras."
                )

    if results_latex_path.is_file():
        try:
            results_latex_text = results_latex_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read Results_Draft.tex: {exc}")

        else:
            if r"\section{Results}" not in results_latex_text:
                consistency_errors.append(
                    "Results_Draft.tex does not contain \\section{Results}."
                )

    # ------------------------------------------------------------------
    # Reproducibility report validation
    # ------------------------------------------------------------------

    reproducibility_json_path = (
        output_root / "reproducibility" / "reproducibility_metadata.json"
    )

    if reproducibility_json_path.is_file():
        try:
            reproducibility_metadata = json.loads(
                reproducibility_json_path.read_text(encoding="utf-8")
            )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            consistency_errors.append(
                f"Could not read reproducibility_metadata.json: {exc}"
            )

        else:
            environment_block = reproducibility_metadata.get("environment")

            configuration_block = reproducibility_metadata.get("configuration")

            consolidated_block = reproducibility_metadata.get("consolidated_results")

            reproducibility_models = reproducibility_metadata.get("models")

            reported_source_date_epoch = reproducibility_metadata.get(
                "source_date_epoch"
            )

            if reported_source_date_epoch != get_source_date_epoch():
                consistency_errors.append(
                    "SOURCE_DATE_EPOCH is inconsistent in the reproducibility report."
                )

            reported_generation_time = str(
                reproducibility_metadata.get(
                    "generated_at_utc",
                    "",
                )
            ).strip()

            if reported_generation_time != reproducible_iso_datetime():
                consistency_errors.append(
                    "generated_at_utc does not match SOURCE_DATE_EPOCH."
                )

            if not isinstance(environment_block, dict):
                consistency_errors.append(
                    "Environment metadata is missing from the reproducibility report."
                )

            if not isinstance(configuration_block, dict):
                consistency_errors.append(
                    "Configuration missing from the report reproducibility."
                )

            if not isinstance(consolidated_block, dict):
                consistency_errors.append(
                    "Consolidated JSON metadata is missing "
                    "en el reproducibility report."
                )

            if not isinstance(reproducibility_models, list):
                consistency_errors.append(
                    "Model list is missing from the reproducibility."
                )

            elif len(reproducibility_models) != len(models):
                consistency_errors.append(
                    f"Inconsistent model count in the report "
                    f"reproducibility report: "
                    f"expected={len(models)}, "
                    f"found={len(reproducibility_models)}"
                )

            if isinstance(configuration_block, dict):
                reported_model_count = safe_float(
                    configuration_block.get("model_count")
                )

                if (
                    reported_model_count is None
                    or not reported_model_count.is_integer()
                    or int(reported_model_count) != len(models)
                ):
                    consistency_errors.append(
                        "model_count is inconsistent in the report reproducibility."
                    )

            if isinstance(consolidated_block, dict):
                reported_sha256 = str(
                    consolidated_block.get(
                        "sha256",
                        "",
                    )
                ).strip()

                if not re.fullmatch(
                    r"[0-9a-f]{64}",
                    reported_sha256,
                ):
                    consistency_errors.append(
                        "Invalid SHA-256 hash for consolidated_results.json."
                    )

                elif json_path.is_file():
                    current_sha256 = hashlib.sha256(json_path.read_bytes()).hexdigest()

                    if reported_sha256 != current_sha256:
                        consistency_errors.append(
                            "El hash SHA-256 registrado no coincide "
                            "con consolidated_results.json."
                        )

    # ------------------------------------------------------------------
    # Validation of LaTeX integration files
    # ------------------------------------------------------------------

    manuscript_assets_path = (
        output_root / "paper" / "integration" / "ManuscriptAssets.tex"
    )

    supplementary_assets_path = (
        output_root / "supplementary" / "integration" / "SupplementaryAssets.tex"
    )

    if manuscript_assets_path.is_file():
        try:
            manuscript_assets_text = manuscript_assets_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read ManuscriptAssets.tex: {exc}")

        else:
            required_main_inputs = [
                r"\input{paper/latex/Table1_MainResults.tex}",
                r"\input{paper/latex/Table2_FusionAblation.tex}",
            ]

            if statistics_generated:
                required_main_inputs.append(
                    r"\input{paper/latex/"
                    r"Table3_StatisticalComparison.tex}"
                )

            for required_input in required_main_inputs:
                if required_input not in manuscript_assets_text:
                    consistency_errors.append(
                        "Required entry missing from "
                        "ManuscriptAssets.tex: "
                        f"{required_input}"
                    )

            if model_figures_generated:
                required_main_figures = [
                    "figures/models_f1.pdf",
                    "figures/models_mcc.pdf",
                    "figures/models_roc_auc.pdf",
                    "figures/models_pr_auc.pdf",
                ]

                for figure_path in required_main_figures:
                    if figure_path not in manuscript_assets_text:
                        consistency_errors.append(
                            "Required figure missing from "
                            "ManuscriptAssets.tex: "
                            f"{figure_path}"
                        )

    if supplementary_assets_path.is_file():
        try:
            supplementary_assets_text = supplementary_assets_path.read_text(
                encoding="utf-8"
            )

        except OSError as exc:
            consistency_errors.append(f"Could not read SupplementaryAssets.tex: {exc}")

        else:
            required_supplementary_inputs = [
                (
                    r"\input{supplementary/latex/"
                    r"TableS1_AllModels.tex}"
                ),
            ]

            if statistics_generated:
                required_supplementary_inputs.append(
                    r"\input{supplementary/latex/"
                    r"TableS2_StatisticalComparison.tex}"
                )

            for required_input in required_supplementary_inputs:
                if required_input not in supplementary_assets_text:
                    consistency_errors.append(
                        "Required entry missing from "
                        "SupplementaryAssets.tex: "
                        f"{required_input}"
                    )

            if alpha_generated:
                required_alpha_figures = [
                    "figures/alpha_f1.pdf",
                    "figures/alpha_mcc.pdf",
                    "figures/alpha_roc_auc.pdf",
                    "figures/alpha_pr_auc.pdf",
                ]

                for figure_path in required_alpha_figures:
                    if figure_path not in supplementary_assets_text:
                        consistency_errors.append(
                            "Required figure missing from "
                            "SupplementaryAssets.tex: "
                            f"{figure_path}"
                        )

    # ------------------------------------------------------------------
    # Actual LaTeX dependency resolution
    # ------------------------------------------------------------------

    consistency_errors.extend(
        validate_latex_integration_dependencies(
            output_root=output_root,
            integration_path=manuscript_assets_path,
        )
    )

    consistency_errors.extend(
        validate_latex_integration_dependencies(
            output_root=output_root,
            integration_path=supplementary_assets_path,
        )
    )

    # ------------------------------------------------------------------
    # Validation estructural del README principal
    # ------------------------------------------------------------------

    readme_path = output_root / "README.md"

    if readme_path.is_file():
        try:
            readme_text = readme_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read README.md: {exc}")

        else:
            required_readme_fragments = [
                "# Bitcoin AML publication assets",
                "## Directory structure",
                "## Main manuscript integration",
                "## Supplementary integration",
                "## Validation",
                "## Hash verification",
                "## Reproducible generation",
                "## Generated-file inventory",
                (
                    r"\input{paper/integration/"
                    r"ManuscriptAssets.tex}"
                ),
                (
                    r"\input{supplementary/integration/"
                    r"SupplementaryAssets.tex}"
                ),
                "PUBLICATION_CHECK.txt",
                "PACKAGE_CHECK.txt",
                "PACKAGE_CONTENTS.sha256",
                "ARCHIVE_SHA256.txt",
                "SOURCE_DATE_EPOCH",
                "MANIFEST.md",
            ]

            for fragment in required_readme_fragments:
                if fragment not in readme_text:
                    consistency_errors.append(
                        "README.md does not contain the required fragment "
                        f"requerido: {fragment}"
                    )

    # ------------------------------------------------------------------
    # Validation estructural del mapa de evidencias
    # ------------------------------------------------------------------

    evidence_map_path = output_root / "paper" / "PAPER_EVIDENCE_MAP.md"

    if evidence_map_path.is_file():
        try:
            evidence_map_text = evidence_map_path.read_text(encoding="utf-8")

        except OSError as exc:
            consistency_errors.append(f"Could not read PAPER_EVIDENCE_MAP.md: {exc}")

        else:
            required_evidence_fragments = [
                "# Paper evidence map",
                "## Abstract",
                "## Introduction and contributions",
                "## Methodology",
                "## Experimental setup",
                "## Main results",
                "## Fusion ablation",
                "## Statistical analysis",
                "## Discussion",
                "## Reproducibility and integrity",
                "## Manuscript integration",
                "## Authoring rule",
                "Table1_MainResults.csv",
                "Table2_FusionAblation.csv",
                "TableS1_AllModels.csv",
                "consolidated_results.json",
                "PUBLICATION_CHECK.txt",
                "PACKAGE_CHECK.txt",
                "ManuscriptAssets.tex",
                "SupplementaryAssets.tex",
            ]

            for fragment in required_evidence_fragments:
                if fragment not in evidence_map_text:
                    consistency_errors.append(
                        "PAPER_EVIDENCE_MAP.md does not contain "
                        "el fragmento requerido: "
                        f"{fragment}"
                    )

    check_passed = not missing_files and not consistency_errors

    lines = [
        "PUBLICATION ASSET CONSISTENCY CHECK",
        "=" * 44,
        "",
        f"Models loaded        : {len(models)}",
        f"Expected files       : {len(required_files)}",
        f"Missing files        : {len(missing_files)}",
        f"Numeric values tested: {checked_values}",
        f"Editorial files tested: {len(editorial_text_files)}",
        f"Consistency errors   : {len(consistency_errors)}",
        "",
        f"STATUS: {'PASS' if check_passed else 'FAIL'}",
        "",
    ]

    if missing_files:
        lines.extend(
            [
                "MISSING FILES",
                "-" * 44,
            ]
        )

        for relative_path in missing_files:
            lines.append(f"[MISSING] {relative_path}")

        lines.append("")

    if consistency_errors:
        lines.extend(
            [
                "CONSISTENCY ERRORS",
                "-" * 44,
            ]
        )

        for error in consistency_errors:
            lines.append(f"[ERROR] {error}")

        lines.append("")

    if check_passed:
        lines.extend(
            [
                "All expected publication assets exist.",
                (
                    "Table1_MainResults.csv, Table2_FusionAblation.csv, "
                    "and TableS1_AllModels.csv are numerically consistent "
                    "with consolidated_results.json. Table3 and TableS2 "
                    "are structurally consistent. Markdown and LaTeX "
                    "publication assets passed the editorial checks. "
                    "The reproducibility metadata and SHA-256 hashes "
                    "are valid. The manuscript-ready Results section "
                    "passed its structural checks."
                ),
                (
                    "The LaTeX manuscript-integration files passed "
                    "their structural and dependency-resolution checks."
                ),
                "The package README passed its structural checks.",
                (
                    "The manuscript evidence map passed its structural "
                    "and traceability checks."
                ),
                "",
            ]
        )

    check_path = output_root / "PUBLICATION_CHECK.txt"

    check_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return check_passed


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    output_root = args.output_dir
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()

    for subdir in (
        "tables",
        "markdown",
        "latex",
        "figures",
        "json",
        "captions",
        "supplementary",
    ):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    setup_publication_style()

    models: list[ModelResult] = []
    load_errors: list[str] = []

    for name, relative_dir in MODEL_CONFIG:
        model_dir = project_root / relative_dir
        try:
            model = load_model(name, model_dir)
            models.append(model)
        except Exception as exc:  # noqa: BLE001 - isolate model-loading failures
            load_errors.append(f"{name}: {exc}")

    if not models:
        raise RuntimeError("No model could be loaded. Check MODEL_CONFIG.")

    validation_errors, validation_warnings = validate_models(models)

    fusion_models = [
        model
        for model in models
        if (model.name == "Hybrid Transformer" or model.name in FUSION_MODELS)
    ]

    save_publication_tables(
        models=models,
        fusion_models=fusion_models,
        output_root=output_root,
        decimals=args.decimals,
    )

    generate_problem_method_mapping_table(
        output_root=output_root,
    )

    generate_table_captions(
        output_root=output_root,
    )

    statistics_generated = generate_statistics_table(
        project_root,
        output_root,
        args.decimals,
    )

    improvement_rows = generate_improvement_table(
        models,
        output_root,
    )

    if not args.no_figures:
        generate_problem_overview_figure(
            output_root=output_root,
        )

        for metric in ("f1", "mcc", "roc_auc", "pr_auc"):
            plot_model_metric(
                models,
                metric,
                output_root,
            )

        model_figures_generated = True

        alpha_generated = plot_alpha_ablation(
            project_root,
            output_root,
        )

    else:
        model_figures_generated = False
        alpha_generated = False

    generate_figure_captions(
        output_root=output_root,
        alpha_generated=alpha_generated,
        model_figures_generated=model_figures_generated,
    )

    generate_latex_integration_files(
        output_root=output_root,
        statistics_generated=statistics_generated,
        alpha_generated=alpha_generated,
        model_figures_generated=model_figures_generated,
    )

    generate_package_readme(
        output_root=output_root,
    )

    generate_paper_evidence_map(
        output_root=output_root,
        statistics_generated=statistics_generated,
        alpha_generated=alpha_generated,
        model_figures_generated=model_figures_generated,
    )

    generate_results_text(
        models,
        improvement_rows,
        output_root,
    )

    generate_manuscript_results_section(
        models=models,
        improvement_rows=improvement_rows,
        output_root=output_root,
        decimals=args.decimals,
    )

    consolidated = {
        "models": [
            {
                "name": model.name,
                "directory": portable_path(model.directory),
                "source_file": portable_path(model.source_file),
                "n_seeds": model.n_seeds,
                "seeds": model.seeds,
                "metrics": {
                    metric_name: {
                        "mean": result.mean,
                        "standard_deviation": result.std,
                        "values": result.values,
                    }
                    for metric_name, result in model.metrics.items()
                },
            }
            for model in models
        ],
        "load_errors": load_errors,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }
    (output_root / "json" / "consolidated_results.json").write_text(
        json.dumps(
            consolidated,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    generate_reproducibility_report(
        output_root=output_root,
        models=models,
        decimals=args.decimals,
    )

    report = [
        "=" * 88,
        "PAPER ASSET GENERATION REPORT — VERSION 5",
        "=" * 88,
        "",
        "Project root     : .",
        f"Output directory : {output_root}",
        f"Models processed : {len(models)}",
        "",
        "LOADED MODELS",
        "-" * 88,
    ]

    for model in models:
        report.append(
            f"[OK] {model.name} | n_seeds={model.n_seeds} | {model.source_file}"
        )

    if load_errors:
        report.extend(
            [
                "",
                "LOAD ERRORS",
                "-" * 88,
                *[f"[ERROR] {item}" for item in load_errors],
            ]
        )

    if validation_errors:
        report.extend(
            [
                "",
                "VALIDATION ERRORS",
                "-" * 88,
                *[f"[ERROR] {item}" for item in validation_errors],
            ]
        )

    if validation_warnings:
        report.extend(
            [
                "",
                "VALIDATION WARNINGS",
                "-" * 88,
                *[f"[WARNING] {item}" for item in validation_warnings],
            ]
        )

    report.extend(
        [
            "",
            "GENERATED ASSETS",
            "-" * 88,
            "- Table0_ProblemMethodMapping: paper/ CSV, Markdown, LaTeX",
            (
                "- Conceptual problem-overview figure: generated"
                if not args.no_figures
                else "- Conceptual problem-overview figure: disabled"
            ),
            "- Table1_MainResults: paper/ CSV, Markdown, LaTeX",
            (
                "- Table2_FusionAblation: paper/ CSV, Markdown, LaTeX"
                if fusion_models
                else "- Table2_FusionAblation: no fusion models available"
            ),
            "- TableS1_AllModels: supplementary/ CSV, Markdown, LaTeX",
            (
                "- Table3_StatisticalComparison: paper/ CSV, Markdown, LaTeX"
                if statistics_generated
                else "- Table3_StatisticalComparison: source CSV not found"
            ),
            (
                "- TableS2_StatisticalComparison: supplementary/ CSV, Markdown, LaTeX"
                if statistics_generated
                else "- TableS2_StatisticalComparison: source CSV not found"
            ),
            "- Improvement_Over_Hybrid.csv",
            "- consolidated_results.json",
            "- Results_Draft.md and Results_Draft.tex",
            "- REPRODUCIBILITY.md",
            "- reproducibility_metadata.json",
            "- results_summary.md",
            "- README.md",
            "- PAPER_EVIDENCE_MAP.md",
            "- ManuscriptAssets.tex",
            "- SupplementaryAssets.tex",
            (
                "- Alpha-ablation figures: generated"
                if alpha_generated
                else "- Alpha-ablation figures: source CSV not found or disabled"
            ),
            (
                "- Model-comparison figures: generated"
                if not args.no_figures
                else "- Model-comparison figures: disabled"
            ),
            "",
            "IMPORTANT",
            "-" * 88,
            "This version reads only explicitly configured experiment folders.",
            "It never searches unknown folders and never reads seed_xx/metrics.json.",
            (
                "Build timestamps are controlled by SOURCE_DATE_EPOCH "
                "for reproducible package generation."
            ),
            "",
        ]
    )
    report.extend(
        [
            "- Table captions: captions/table_captions.md",
            "- Figure captions: captions/figure_captions.md",
            "- MANIFEST.md",
        ]
    )
    report.extend(
        [
            "- Delivery package: delivery/bitcoin_aml_publication_assets.zip",
            "- Package content hashes: delivery/PACKAGE_CONTENTS.sha256",
            "- Archive hash: delivery/ARCHIVE_SHA256.txt",
            "- Package verification: delivery/PACKAGE_CHECK.txt",
        ]
    )

    report_text = "\n".join(report)
    (output_root / "generation_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    generate_manifest(output_root)

    publication_check_passed = generate_publication_check(
        output_root=output_root,
        models=models,
        statistics_generated=statistics_generated,
        alpha_generated=alpha_generated,
        model_figures_generated=model_figures_generated,
        decimals=args.decimals,
    )

    delivery_package_passed = generate_delivery_package(
        output_root=output_root,
    )

    print(report_text)

    if publication_check_passed:
        print("[CHECK] Publication assets: PASS")
    else:
        print("[CHECK] Publication assets: FAIL")

    if delivery_package_passed:
        print("[CHECK] Delivery package: PASS")
    else:
        print("[CHECK] Delivery package: FAIL")


if __name__ == "__main__":
    main()
