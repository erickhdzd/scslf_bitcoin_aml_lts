from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from artifact_utils import portable_path

read_csv_typed = cast(Callable[..., pd.DataFrame], pd.read_csv)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

INPUT_TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"

ANALYSIS_DIR: Final[Path] = RESULTS_DIR / "analysis" / "baselines"

ANALYSIS_TABLES_DIR: Final[Path] = ANALYSIS_DIR / "tables"

ANALYSIS_FIGURES_DIR: Final[Path] = ANALYSIS_DIR / "figures"

DETAILED_RESULTS_PATH: Final[Path] = INPUT_TABLES_DIR / "baseline_results_detailed.csv"

SUMMARY_RESULTS_PATH: Final[Path] = INPUT_TABLES_DIR / "baseline_results_summary.csv"

TIME_TEST_PATH: Final[Path] = PROCESSED_DIR / "time_test.npy"

Y_TEST_PATH: Final[Path] = PROCESSED_DIR / "y_test.npy"


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPECTED_MODELS: Final[tuple[str, ...]] = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
)

EXPECTED_SEEDS: Final[tuple[int, ...]] = (
    42,
    43,
    44,
    45,
    46,
)

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "mlp": "MLP",
}

METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    "test_accuracy": "Accuracy",
    "test_balanced_accuracy": "Balanced Accuracy",
    "test_precision_illicit": "Illicit Precision",
    "test_recall_illicit": "Illicit Recall",
    "test_f1_illicit": "Illicit F1",
    "test_macro_f1": "Macro-F1",
    "test_roc_auc": "ROC-AUC",
    "test_pr_auc": "PR-AUC",
    "test_matthews_correlation": "MCC",
    "training_seconds": "Training time (s)",
}


# ============================================================================
# GENERAL UTILITIES
# ============================================================================


def print_separator(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def ensure_output_directories() -> None:
    ANALYSIS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def verify_input_files() -> None:
    required_files = [
        DETAILED_RESULTS_PATH,
        SUMMARY_RESULTS_PATH,
        TIME_TEST_PATH,
        Y_TEST_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    for model_name in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            expected_prediction_path = (
                INPUT_TABLES_DIR / f"predictions_{model_name}_seed_{seed}.csv"
            )

            if not expected_prediction_path.exists():
                missing_files.append(expected_prediction_path)

    if missing_files:
        formatted = "\n".join(str(path) for path in missing_files)

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{formatted}\n\n"
            "Run the complete training pipeline first."
        )


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    output_path = ANALYSIS_FIGURES_DIR / filename

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Figure saved: {output_path}")


def prediction_path(
    model_name: str,
    seed: int,
) -> Path:
    return INPUT_TABLES_DIR / f"predictions_{model_name}_seed_{seed}.csv"


def read_prediction_file(
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    path = prediction_path(
        model_name=model_name,
        seed=seed,
    )

    dataframe = read_csv_typed(str(path))

    required_columns = {
        "true_label",
        "probability_illicit",
        "predicted_label",
        "threshold",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")

    return dataframe


# ============================================================================
# RESULT VALIDATION
# ============================================================================


def load_and_validate_results() -> pd.DataFrame:
    print_separator("RESULT VALIDATION")

    results = read_csv_typed(str(DETAILED_RESULTS_PATH))

    expected_rows = len(EXPECTED_MODELS) * len(EXPECTED_SEEDS)

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} experiments, but found {len(results)}."
        )

    duplicated = results.duplicated(subset=["model", "seed"]).sum()

    if duplicated != 0:
        raise ValueError(f"Found {duplicated} combinations duplicate model-seed pairs.")

    found_models = set(results["model"].unique())

    expected_models = set(EXPECTED_MODELS)

    if found_models != expected_models:
        raise ValueError(
            "The models found do not match "
            "the expected models.\n"
            f"Found: {sorted(found_models)}\n"
            f"Expected: {sorted(expected_models)}"
        )

    for model_name in EXPECTED_MODELS:
        model_seeds = set(
            results.loc[
                results["model"] == model_name,
                "seed",
            ].astype(int)
        )

        if model_seeds != set(EXPECTED_SEEDS):
            raise ValueError(f"Incorrect seeds for {model_name}: {sorted(model_seeds)}")

    numeric_metrics = [
        "threshold",
        "validation_f1_illicit",
        "validation_pr_auc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision_illicit",
        "test_recall_illicit",
        "test_f1_illicit",
        "test_macro_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_matthews_correlation",
    ]

    for column in numeric_metrics:
        if results[column].isna().any():
            raise ValueError(f"Column {column} contains NaN.")

    bounded_metrics = [column for column in numeric_metrics if column != "threshold"]

    for column in bounded_metrics:
        column_values = results[column].to_numpy(
            dtype=np.float64,
        )

        invalid = (column_values < 0.0) | (column_values > 1.0)

        if np.any(invalid):
            raise ValueError(f"Column {column} contains values outside [0, 1].")

    print(f"Experimentos: {len(results)}")
    print(f"Models: {results['model'].nunique()}")
    print(f"Seeds: {results['seed'].nunique()}")
    print(f"Duplicates: {duplicated}")
    print("Validation completed successfully.")

    return results


def validate_prediction_files(
    y_test: np.ndarray,
) -> None:
    print_separator("PREDICTION VALIDATION")

    for model_name in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            dataframe = read_prediction_file(
                model_name=model_name,
                seed=seed,
            )

            if len(dataframe) != len(y_test):
                raise ValueError(
                    f"Incorrect length for "
                    f"{model_name}, seed {seed}: "
                    f"{len(dataframe)} versus {len(y_test)}."
                )

            stored_labels = dataframe["true_label"].to_numpy(dtype=np.int8)

            if not np.array_equal(
                stored_labels,
                y_test,
            ):
                raise ValueError(
                    f"Labels for {model_name}, seed {seed}, do not match y_test."
                )

            probabilities = dataframe["probability_illicit"].to_numpy(dtype=np.float64)

            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Non-finite probabilities found for {model_name}, seed {seed}."
                )

            if (probabilities < 0.0).any() or (probabilities > 1.0).any():
                raise ValueError(
                    f"Probabilities outside [0, 1] found for {model_name}, seed {seed}."
                )

    print("All 20 prediction files are valid.")


# ============================================================================
# SUMMARY TABLES
# ============================================================================


def build_statistical_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision_illicit",
        "test_recall_illicit",
        "test_f1_illicit",
        "test_macro_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_matthews_correlation",
        "training_seconds",
        "threshold",
        "validation_f1_illicit",
        "validation_pr_auc",
    ]

    rows: list[dict[str, float | str]] = []

    for model_name in EXPECTED_MODELS:
        model_results = results[results["model"] == model_name]

        row: dict[str, float | str] = {
            "model": model_name,
            "display_name": MODEL_DISPLAY_NAMES[model_name],
        }

        for metric in metrics:
            metric_values = model_results[metric].to_numpy(
                dtype=np.float64,
            )

            row[f"{metric}_mean"] = float(metric_values.mean())

            row[f"{metric}_std"] = float(metric_values.std(ddof=1))

            row[f"{metric}_min"] = float(metric_values.min())

            row[f"{metric}_max"] = float(metric_values.max())

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by="test_f1_illicit_mean",
        ascending=False,
    ).reset_index(drop=True)

    output_path = ANALYSIS_TABLES_DIR / "baseline_statistical_summary.csv"

    summary.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    return summary


def build_paper_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    paper_table = pd.DataFrame()

    paper_table["Model"] = summary["display_name"]

    metrics = {
        "Balanced Accuracy": "test_balanced_accuracy",
        "Illicit Precision": "test_precision_illicit",
        "Illicit Recall": "test_recall_illicit",
        "Illicit F1": "test_f1_illicit",
        "Macro-F1": "test_macro_f1",
        "ROC-AUC": "test_roc_auc",
        "PR-AUC": "test_pr_auc",
        "MCC": "test_matthews_correlation",
    }

    for display_name, metric_name in metrics.items():
        mean_column = f"{metric_name}_mean"
        std_column = f"{metric_name}_std"

        mean_values = summary[mean_column].to_numpy(
            dtype=np.float64,
        )

        std_values = summary[std_column].to_numpy(
            dtype=np.float64,
        )

        paper_table[display_name] = [
            f"{float(mean):.4f} ± {float(std):.4f}"
            for mean, std in zip(
                mean_values,
                std_values,
                strict=True,
            )
        ]

    output_path = ANALYSIS_TABLES_DIR / "baseline_paper_table.csv"

    paper_table.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    return paper_table


def build_model_ranking(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ranking = summary[
        [
            "model",
            "display_name",
            "test_f1_illicit_mean",
            "test_f1_illicit_std",
            "test_pr_auc_mean",
            "test_pr_auc_std",
            "test_matthews_correlation_mean",
            "test_matthews_correlation_std",
            "test_precision_illicit_mean",
            "test_recall_illicit_mean",
        ]
    ].copy()

    ranking["f1_rank"] = (
        ranking["test_f1_illicit_mean"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    ranking["pr_auc_rank"] = (
        ranking["test_pr_auc_mean"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    ranking["mcc_rank"] = (
        ranking["test_matthews_correlation_mean"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    ranking["average_rank"] = ranking[
        [
            "f1_rank",
            "pr_auc_rank",
            "mcc_rank",
        ]
    ].mean(axis=1)

    ranking = ranking.sort_values(
        by=[
            "average_rank",
            "test_f1_illicit_mean",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    output_path = ANALYSIS_TABLES_DIR / "baseline_model_ranking.csv"

    ranking.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    return ranking


# ============================================================================
# METRIC FIGURES
# ============================================================================


def plot_metric_with_error_bars(
    summary: pd.DataFrame,
    metric_name: str,
    filename: str,
    title: str,
    y_label: str,
) -> None:
    ordered = summary.sort_values(
        by=f"{metric_name}_mean",
        ascending=False,
    )

    labels = ordered["display_name"].tolist()

    means = ordered[f"{metric_name}_mean"].to_numpy(dtype=np.float64)

    standard_deviations = ordered[f"{metric_name}_std"].to_numpy(dtype=np.float64)

    x_positions = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(9.5, 6.0))

    bars = axis.bar(
        x_positions,
        means,
        yerr=standard_deviations,
        capsize=6,
    )

    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xticks(
        x_positions,
        labels,
        rotation=15,
        ha="right",
    )

    if metric_name != "training_seconds":
        axis.set_ylim(
            0.0,
            min(
                1.0,
                float(max(means + standard_deviations)) * 1.18,
            ),
        )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    for bar, mean in zip(
        bars,
        means,
        strict=True,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(
        figure=figure,
        filename=filename,
    )


def plot_main_metrics(
    summary: pd.DataFrame,
) -> None:
    plot_metric_with_error_bars(
        summary=summary,
        metric_name="test_f1_illicit",
        filename="baseline_illicit_f1.png",
        title="Illicit-class F1 score",
        y_label="F1 score",
    )

    plot_metric_with_error_bars(
        summary=summary,
        metric_name="test_pr_auc",
        filename="baseline_pr_auc.png",
        title="Precision–Recall area under the curve",
        y_label="PR-AUC",
    )

    plot_metric_with_error_bars(
        summary=summary,
        metric_name="test_matthews_correlation",
        filename="baseline_mcc.png",
        title="Matthews correlation coefficient",
        y_label="MCC",
    )

    plot_metric_with_error_bars(
        summary=summary,
        metric_name="training_seconds",
        filename="baseline_training_time.png",
        title="Training time",
        y_label="Seconds",
    )


# ============================================================================
# PRECISION–RECALL CURVES
# ============================================================================


def plot_precision_recall_curves(
    y_test: np.ndarray,
) -> pd.DataFrame:
    print_separator("PRECISION–RECALL CURVES")

    recall_grid = np.linspace(
        0.0,
        1.0,
        1001,
    )

    figure, axis = plt.subplots(figsize=(8.5, 7.0))

    curve_rows: list[dict[str, float | int | str]] = []

    prevalence = float(np.mean(y_test == 1))

    axis.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.5,
        label=f"Random baseline ({prevalence:.4f})",
    )

    for model_name in EXPECTED_MODELS:
        interpolated_precisions: list[np.ndarray] = []
        pr_auc_values: list[float] = []

        for seed in EXPECTED_SEEDS:
            predictions = read_prediction_file(
                model_name=model_name,
                seed=seed,
            )

            probabilities = predictions["probability_illicit"].to_numpy(
                dtype=np.float64
            )

            precision, recall, _ = precision_recall_curve(
                y_test,
                probabilities,
            )

            pr_auc_value = float(
                average_precision_score(
                    y_test,
                    probabilities,
                )
            )

            pr_auc_values.append(pr_auc_value)

            ascending_recall = recall[::-1]
            ascending_precision = precision[::-1]

            unique_recall, unique_indices = np.unique(
                ascending_recall,
                return_index=True,
            )

            unique_precision = ascending_precision[unique_indices]

            interpolated = np.asarray(
                np.interp(
                    recall_grid,
                    unique_recall,
                    unique_precision,
                ),
                dtype=np.float64,
            ).reshape(-1)

            interpolated_precisions.append(interpolated)

        precision_matrix = np.vstack(interpolated_precisions)

        mean_precision = precision_matrix.mean(axis=0)

        std_precision = precision_matrix.std(
            axis=0,
            ddof=1,
        )

        mean_pr_auc = float(np.mean(pr_auc_values))

        std_pr_auc = float(
            np.std(
                pr_auc_values,
                ddof=1,
            )
        )

        axis.plot(
            recall_grid,
            mean_precision,
            linewidth=2.2,
            label=(
                f"{MODEL_DISPLAY_NAMES[model_name]} "
                f"({mean_pr_auc:.4f} ± {std_pr_auc:.4f})"
            ),
        )

        axis.fill_between(
            recall_grid,
            np.clip(
                mean_precision - std_precision,
                0.0,
                1.0,
            ),
            np.clip(
                mean_precision + std_precision,
                0.0,
                1.0,
            ),
            alpha=0.12,
        )

        for recall_value, precision_value, std_value in zip(
            recall_grid,
            mean_precision,
            std_precision,
            strict=True,
        ):
            curve_rows.append(
                {
                    "model": model_name,
                    "recall": float(recall_value),
                    "precision_mean": float(precision_value),
                    "precision_std": float(std_value),
                    "pr_auc_mean": mean_pr_auc,
                    "pr_auc_std": std_pr_auc,
                }
            )

    axis.set_title("Mean Precision–Recall curves across five seeds")

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)

    axis.grid(
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        loc="best",
        fontsize=9,
    )

    save_figure(
        figure=figure,
        filename="baseline_precision_recall_curves.png",
    )

    curves = pd.DataFrame(curve_rows)

    output_path = ANALYSIS_TABLES_DIR / "precision_recall_mean_curves.csv"

    curves.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    return curves


# ============================================================================
# CURVAS ROC
# ============================================================================


def plot_roc_curves(
    y_test: np.ndarray,
) -> pd.DataFrame:
    print_separator("CURVAS ROC")

    false_positive_grid = np.linspace(
        0.0,
        1.0,
        1001,
    )

    figure, axis = plt.subplots(figsize=(8.5, 7.0))

    curve_rows: list[dict[str, float | str]] = []

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier",
    )

    for model_name in EXPECTED_MODELS:
        interpolated_true_positive_rates: list[np.ndarray] = []

        roc_auc_values: list[float] = []

        for seed in EXPECTED_SEEDS:
            predictions = read_prediction_file(
                model_name=model_name,
                seed=seed,
            )

            probabilities = predictions["probability_illicit"].to_numpy(
                dtype=np.float64
            )

            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test,
                probabilities,
            )

            roc_auc_value = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            )

            roc_auc_values.append(roc_auc_value)

            interpolated = np.asarray(
                np.interp(
                    false_positive_grid,
                    false_positive_rate,
                    true_positive_rate,
                ),
                dtype=np.float64,
            ).reshape(-1)

            interpolated[0] = 0.0
            interpolated[-1] = 1.0

            interpolated_true_positive_rates.append(interpolated)

        tpr_matrix = np.vstack(interpolated_true_positive_rates)

        mean_tpr = tpr_matrix.mean(axis=0)
        mean_tpr[-1] = 1.0

        std_tpr = tpr_matrix.std(
            axis=0,
            ddof=1,
        )

        mean_roc_auc = float(np.mean(roc_auc_values))

        std_roc_auc = float(
            np.std(
                roc_auc_values,
                ddof=1,
            )
        )

        axis.plot(
            false_positive_grid,
            mean_tpr,
            linewidth=2.2,
            label=(
                f"{MODEL_DISPLAY_NAMES[model_name]} "
                f"({mean_roc_auc:.4f} ± "
                f"{std_roc_auc:.4f})"
            ),
        )

        axis.fill_between(
            false_positive_grid,
            np.clip(
                mean_tpr - std_tpr,
                0.0,
                1.0,
            ),
            np.clip(
                mean_tpr + std_tpr,
                0.0,
                1.0,
            ),
            alpha=0.12,
        )

        mean_curve_auc = float(
            auc(
                false_positive_grid,
                mean_tpr,
            )
        )

        for fpr_value, tpr_value, std_value in zip(
            false_positive_grid,
            mean_tpr,
            std_tpr,
            strict=True,
        ):
            curve_rows.append(
                {
                    "model": model_name,
                    "false_positive_rate": float(fpr_value),
                    "true_positive_rate_mean": float(tpr_value),
                    "true_positive_rate_std": float(std_value),
                    "roc_auc_mean": mean_roc_auc,
                    "roc_auc_std": std_roc_auc,
                    "mean_curve_auc": mean_curve_auc,
                }
            )

    axis.set_title("Mean ROC curves across five seeds")

    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)

    axis.grid(
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        loc="lower right",
        fontsize=9,
    )

    save_figure(
        figure=figure,
        filename="baseline_roc_curves.png",
    )

    curves = pd.DataFrame(curve_rows)

    output_path = ANALYSIS_TABLES_DIR / "roc_mean_curves.csv"

    curves.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    return curves


# ============================================================================
# VALIDATION-TEST GAP
# ============================================================================


def analyze_generalization_gap(
    results: pd.DataFrame,
) -> pd.DataFrame:
    print_separator("VALIDATION-TEST GAP")

    gap_results = results[
        [
            "model",
            "seed",
            "validation_f1_illicit",
            "test_f1_illicit",
            "validation_pr_auc",
            "test_pr_auc",
        ]
    ].copy()

    gap_results["f1_gap"] = (
        gap_results["validation_f1_illicit"] - gap_results["test_f1_illicit"]
    )

    gap_results["pr_auc_gap"] = (
        gap_results["validation_pr_auc"] - gap_results["test_pr_auc"]
    )

    output_path = ANALYSIS_TABLES_DIR / "validation_test_generalization_gap.csv"

    gap_results.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    summary = (
        gap_results.groupby("model")
        .agg(
            validation_f1_mean=(
                "validation_f1_illicit",
                "mean",
            ),
            test_f1_mean=(
                "test_f1_illicit",
                "mean",
            ),
            f1_gap_mean=(
                "f1_gap",
                "mean",
            ),
            f1_gap_std=(
                "f1_gap",
                "std",
            ),
            validation_pr_auc_mean=(
                "validation_pr_auc",
                "mean",
            ),
            test_pr_auc_mean=(
                "test_pr_auc",
                "mean",
            ),
            pr_auc_gap_mean=(
                "pr_auc_gap",
                "mean",
            ),
            pr_auc_gap_std=(
                "pr_auc_gap",
                "std",
            ),
        )
        .reset_index()
    )

    summary["display_name"] = summary["model"].map(MODEL_DISPLAY_NAMES)

    summary = summary.sort_values(
        by="f1_gap_mean",
        ascending=False,
    ).reset_index(drop=True)

    summary_path = ANALYSIS_TABLES_DIR / "validation_test_gap_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"Table saved: {summary_path}")

    x_positions = np.arange(len(summary))
    width = 0.36

    figure, axis = plt.subplots(figsize=(10.0, 6.0))

    axis.bar(
        x_positions - width / 2.0,
        summary["validation_f1_mean"],
        width=width,
        label="Validation F1",
    )

    axis.bar(
        x_positions + width / 2.0,
        summary["test_f1_mean"],
        width=width,
        label="Test F1",
    )

    axis.set_title("Validation-to-test degradation")

    axis.set_ylabel("Illicit-class F1")

    axis.set_xticks(
        x_positions,
        summary["display_name"],
        rotation=15,
        ha="right",
    )

    axis.set_ylim(0.0, 1.0)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend()

    save_figure(
        figure=figure,
        filename="validation_test_f1_gap.png",
    )

    return summary


# ============================================================================
# ANALYSIS BY TIME STEP
# ============================================================================


def calculate_time_step_metrics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    ).ravel()

    result: dict[str, float | int] = {
        "samples": len(y_true),
        "licit_samples": int((y_true == 0).sum()),
        "illicit_samples": int((y_true == 1).sum()),
        "illicit_prevalence": float(np.mean(y_true == 1)),
        "precision_illicit": float(
            precision_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "recall_illicit": float(
            recall_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "f1_illicit": float(
            f1_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_predicted,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_predicted,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )
    else:
        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


def build_temporal_metrics(
    time_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_separator("ANALYSIS BY TIME STEP")

    detailed_rows: list[dict[str, float | int | str]] = []

    unique_time_steps = np.sort(np.unique(time_test))

    for model_name in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            predictions = read_prediction_file(
                model_name=model_name,
                seed=seed,
            )

            probabilities = predictions["probability_illicit"].to_numpy(
                dtype=np.float64
            )

            predicted_labels = predictions["predicted_label"].to_numpy(dtype=np.int8)

            for time_step in unique_time_steps:
                mask = time_test == time_step

                metrics = calculate_time_step_metrics(
                    y_true=y_test[mask],
                    y_predicted=predicted_labels[mask],
                    probabilities=probabilities[mask],
                )

                row: dict[
                    str,
                    float | int | str,
                ] = {
                    "model": model_name,
                    "display_name": (MODEL_DISPLAY_NAMES[model_name]),
                    "seed": int(seed),
                    "time_step": int(time_step),
                }

                row.update(metrics)
                detailed_rows.append(row)

    detailed = pd.DataFrame(detailed_rows)

    detailed_path = ANALYSIS_TABLES_DIR / "baseline_metrics_by_time_step_detailed.csv"

    detailed.to_csv(
        detailed_path,
        index=False,
    )

    print(f"Table saved: {detailed_path}")

    metric_columns = [
        "precision_illicit",
        "recall_illicit",
        "f1_illicit",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    aggregations: dict[
        str,
        list[str],
    ] = {metric: ["mean", "std"] for metric in metric_columns}

    temporal_summary = detailed.groupby(
        [
            "model",
            "display_name",
            "time_step",
        ],
        as_index=False,
    ).agg(aggregations)

    temporal_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in temporal_summary.columns
    ]

    temporal_summary = temporal_summary.rename(
        columns={
            "model_": "model",
            "display_name_": "display_name",
            "time_step_": "time_step",
        }
    )

    summary_path = ANALYSIS_TABLES_DIR / "baseline_metrics_by_time_step_summary.csv"

    temporal_summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"Table saved: {summary_path}")

    return detailed, temporal_summary


def plot_temporal_metric(
    temporal_summary: pd.DataFrame,
    metric_name: str,
    filename: str,
    title: str,
    y_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.5))

    for model_name in EXPECTED_MODELS:
        model_data = temporal_summary[
            temporal_summary["model"] == model_name
        ].sort_values("time_step")

        x_values = model_data["time_step"].to_numpy(dtype=int)

        mean_values = model_data[f"{metric_name}_mean"].to_numpy(dtype=np.float64)

        std_values = (
            model_data[f"{metric_name}_std"].fillna(0.0).to_numpy(dtype=np.float64)
        )

        axis.plot(
            x_values,
            mean_values,
            marker="o",
            linewidth=2.0,
            label=MODEL_DISPLAY_NAMES[model_name],
        )

        axis.fill_between(
            x_values,
            np.clip(
                mean_values - std_values,
                0.0,
                1.0,
            ),
            np.clip(
                mean_values + std_values,
                0.0,
                1.0,
            ),
            alpha=0.12,
        )

    axis.set_title(title)
    axis.set_xlabel("Time step")
    axis.set_ylabel(y_label)

    axis.set_xticks(sorted(temporal_summary["time_step"].unique()))

    axis.set_ylim(0.0, 1.02)

    axis.grid(
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        loc="best",
        fontsize=9,
    )

    save_figure(
        figure=figure,
        filename=filename,
    )


def plot_temporal_results(
    temporal_summary: pd.DataFrame,
) -> None:
    plot_temporal_metric(
        temporal_summary=temporal_summary,
        metric_name="f1_illicit",
        filename="baseline_f1_by_time_step.png",
        title="Illicit-class F1 by test time step",
        y_label="F1 score",
    )

    plot_temporal_metric(
        temporal_summary=temporal_summary,
        metric_name="precision_illicit",
        filename="baseline_precision_by_time_step.png",
        title="Illicit precision by test time step",
        y_label="Precision",
    )

    plot_temporal_metric(
        temporal_summary=temporal_summary,
        metric_name="recall_illicit",
        filename="baseline_recall_by_time_step.png",
        title="Illicit recall by test time step",
        y_label="Recall",
    )

    plot_temporal_metric(
        temporal_summary=temporal_summary,
        metric_name="pr_auc",
        filename="baseline_pr_auc_by_time_step.png",
        title="PR-AUC by test time step",
        y_label="PR-AUC",
    )


# ============================================================================
# AGGREGATED CONFUSION MATRICES
# ============================================================================


def build_aggregate_confusion_matrices(
    results: pd.DataFrame,
) -> pd.DataFrame:
    print_separator("AGGREGATED CONFUSION MATRICES")

    rows: list[dict[str, float | str]] = []

    for model_name in EXPECTED_MODELS:
        model_results = results[results["model"] == model_name]

        confusion_columns = {
            "true_negatives": "test_true_negatives",
            "false_positives": "test_false_positives",
            "false_negatives": "test_false_negatives",
            "true_positives": "test_true_positives",
        }

        row: dict[str, float | str] = {
            "model": model_name,
            "display_name": MODEL_DISPLAY_NAMES[model_name],
        }

        for display_column, source_column in confusion_columns.items():
            row[f"{display_column}_mean"] = float(model_results[source_column].mean())

            row[f"{display_column}_std"] = float(
                model_results[source_column].std(ddof=1)
            )

        rows.append(row)

        mean_matrix = np.array(
            [
                [
                    row["true_negatives_mean"],
                    row["false_positives_mean"],
                ],
                [
                    row["false_negatives_mean"],
                    row["true_positives_mean"],
                ],
            ],
            dtype=np.float64,
        )

        figure, axis = plt.subplots(figsize=(6.2, 5.4))

        image = axis.imshow(
            mean_matrix,
            interpolation="nearest",
        )

        figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

        axis.set_title(f"Mean confusion matrix\n{MODEL_DISPLAY_NAMES[model_name]}")

        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

        axis.set_xticks(
            [0, 1],
            ["Licit", "Illicit"],
        )

        axis.set_yticks(
            [0, 1],
            ["Licit", "Illicit"],
        )

        threshold = mean_matrix.max() / 2.0

        for row_index in range(2):
            for column_index in range(2):
                value = mean_matrix[
                    row_index,
                    column_index,
                ]

                axis.text(
                    column_index,
                    row_index,
                    f"{value:,.1f}",
                    ha="center",
                    va="center",
                    color=("white" if value > threshold else "black"),
                )

        save_figure(
            figure=figure,
            filename=f"mean_confusion_matrix_{model_name}.png",
        )

    summary = pd.DataFrame(rows)

    output_path = ANALYSIS_TABLES_DIR / "aggregate_confusion_matrices.csv"

    summary.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    return summary


# ============================================================================
# STABILITY ACROSS SEEDS
# ============================================================================


def analyze_seed_stability(
    results: pd.DataFrame,
) -> pd.DataFrame:
    print_separator("STABILITY ACROSS SEEDS")

    metrics = [
        "test_f1_illicit",
        "test_pr_auc",
        "test_matthews_correlation",
        "test_precision_illicit",
        "test_recall_illicit",
        "threshold",
    ]

    rows: list[dict[str, float | str]] = []

    for model_name in EXPECTED_MODELS:
        model_results = results[results["model"] == model_name]

        for metric_name in metrics:
            mean_value = float(model_results[metric_name].mean())

            standard_deviation = float(model_results[metric_name].std(ddof=1))

            coefficient_of_variation = (
                standard_deviation / abs(mean_value) if mean_value != 0.0 else np.nan
            )

            rows.append(
                {
                    "model": model_name,
                    "display_name": (MODEL_DISPLAY_NAMES[model_name]),
                    "metric": metric_name,
                    "mean": mean_value,
                    "standard_deviation": standard_deviation,
                    "coefficient_of_variation": float(coefficient_of_variation),
                    "minimum": float(model_results[metric_name].min()),
                    "maximum": float(model_results[metric_name].max()),
                    "range": float(
                        model_results[metric_name].max()
                        - model_results[metric_name].min()
                    ),
                }
            )

    stability = pd.DataFrame(rows)

    output_path = ANALYSIS_TABLES_DIR / "baseline_seed_stability.csv"

    stability.to_csv(
        output_path,
        index=False,
    )

    print(f"Table saved: {output_path}")

    f1_results = results.pivot(
        index="seed",
        columns="model",
        values="test_f1_illicit",
    )

    figure, axis = plt.subplots(figsize=(10.0, 6.0))

    for model_name in EXPECTED_MODELS:
        axis.plot(
            f1_results.index,
            f1_results[model_name],
            marker="o",
            linewidth=2.0,
            label=MODEL_DISPLAY_NAMES[model_name],
        )

    axis.set_title("Illicit-class F1 stability across seeds")

    axis.set_xlabel("Seed")
    axis.set_ylabel("Illicit-class F1")

    axis.set_xticks(list(EXPECTED_SEEDS))

    axis.set_ylim(0.0, 1.0)

    axis.grid(
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        loc="best",
        fontsize=9,
    )

    save_figure(
        figure=figure,
        filename="baseline_f1_by_seed.png",
    )

    return stability


# ============================================================================
# JSON SUMMARY
# ============================================================================


def save_analysis_metadata(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    time_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    best_f1_row = summary.iloc[summary["test_f1_illicit_mean"].argmax()]

    best_pr_auc_row = summary.iloc[summary["test_pr_auc_mean"].argmax()]

    best_mcc_row = summary.iloc[summary["test_matthews_correlation_mean"].argmax()]

    metadata = {
        "python_version": sys.version,
        "project_root": ".",
        "number_of_experiments": len(results),
        "number_of_models": int(results["model"].nunique()),
        "number_of_seeds": int(results["seed"].nunique()),
        "models": list(EXPECTED_MODELS),
        "seeds": list(EXPECTED_SEEDS),
        "test_samples": len(y_test),
        "test_time_steps": [int(value) for value in np.sort(np.unique(time_test))],
        "test_illicit_samples": int((y_test == 1).sum()),
        "test_licit_samples": int((y_test == 0).sum()),
        "test_illicit_prevalence": float(np.mean(y_test == 1)),
        "best_mean_illicit_f1": {
            "model": str(best_f1_row["model"]),
            "display_name": str(best_f1_row["display_name"]),
            "mean": float(best_f1_row["test_f1_illicit_mean"]),
            "standard_deviation": float(best_f1_row["test_f1_illicit_std"]),
        },
        "best_mean_pr_auc": {
            "model": str(best_pr_auc_row["model"]),
            "display_name": str(best_pr_auc_row["display_name"]),
            "mean": float(best_pr_auc_row["test_pr_auc_mean"]),
            "standard_deviation": float(best_pr_auc_row["test_pr_auc_std"]),
        },
        "best_mean_mcc": {
            "model": str(best_mcc_row["model"]),
            "display_name": str(best_mcc_row["display_name"]),
            "mean": float(best_mcc_row["test_matthews_correlation_mean"]),
            "standard_deviation": float(best_mcc_row["test_matthews_correlation_std"]),
        },
        "overall_ranking": ranking[
            [
                "model",
                "average_rank",
                "f1_rank",
                "pr_auc_rank",
                "mcc_rank",
            ]
        ].to_dict(orient="records"),
    }

    output_path = ANALYSIS_DIR / "baseline_analysis_metadata.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Metadata saved: {portable_path(output_path)}")


# ============================================================================
# MAIN PROGRAM
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze baseline experiment outputs and generate aggregate tables and figures."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    print_separator("BASELINE MODEL ANALYSIS")

    print(f"Project: {portable_path(PROJECT_ROOT)}")
    print(f"Results: {DETAILED_RESULTS_PATH}")
    print(f"Output: {ANALYSIS_DIR}")

    ensure_output_directories()
    verify_input_files()

    results = load_and_validate_results()

    y_test = np.load(Y_TEST_PATH).astype(np.int8)

    time_test = np.load(TIME_TEST_PATH)

    if len(time_test) != len(y_test):
        raise ValueError("time_test.npy and y_test.npy have different lengths.")

    validate_prediction_files(
        y_test=y_test,
    )

    summary = build_statistical_summary(
        results=results,
    )

    paper_table = build_paper_table(
        summary=summary,
    )

    ranking = build_model_ranking(
        summary=summary,
    )

    plot_main_metrics(
        summary=summary,
    )

    plot_precision_recall_curves(
        y_test=y_test,
    )

    plot_roc_curves(
        y_test=y_test,
    )

    gap_summary = analyze_generalization_gap(
        results=results,
    )

    temporal_detailed, temporal_summary = build_temporal_metrics(
        time_test=time_test,
        y_test=y_test,
    )

    plot_temporal_results(
        temporal_summary=temporal_summary,
    )

    aggregate_confusions = build_aggregate_confusion_matrices(
        results=results,
    )

    seed_stability = analyze_seed_stability(
        results=results,
    )

    save_analysis_metadata(
        results=results,
        summary=summary,
        ranking=ranking,
        time_test=time_test,
        y_test=y_test,
    )

    print_separator("PAPER TABLE")

    print(paper_table.to_string(index=False))

    print_separator("MODEL RANKING")

    print(
        ranking[
            [
                "display_name",
                "f1_rank",
                "pr_auc_rank",
                "mcc_rank",
                "average_rank",
            ]
        ].to_string(index=False)
    )

    print_separator("MEAN VALIDATION-TEST GAP")

    print(
        gap_summary[
            [
                "display_name",
                "validation_f1_mean",
                "test_f1_mean",
                "f1_gap_mean",
                "validation_pr_auc_mean",
                "test_pr_auc_mean",
                "pr_auc_gap_mean",
            ]
        ].to_string(index=False)
    )

    print_separator("ANALYSIS COMPLETED")

    print(f"Tables generated in: {ANALYSIS_TABLES_DIR}")

    print(f"Figures generated in: {ANALYSIS_FIGURES_DIR}")

    print(f"Detailed temporal rows: {len(temporal_detailed)}")

    print(f"Aggregate matrices: {len(aggregate_confusions)}")

    print(f"Stability records: {len(seed_stability)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
