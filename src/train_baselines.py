from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeAlias, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from artifact_utils import portable_path

BaselineModel: TypeAlias = (
    LogisticRegression
    | RandomForestClassifier
    | HistGradientBoostingClassifier
    | MLPClassifier
)

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures" / "baselines"
CHECKPOINTS_DIR: Final[Path] = RESULTS_DIR / "checkpoints" / "baselines"
LOGS_DIR: Final[Path] = RESULTS_DIR / "logs"

REQUIRED_FILES: Final[tuple[str, ...]] = (
    "x_train.npy",
    "x_validation.npy",
    "x_test.npy",
    "y_train.npy",
    "y_validation.npy",
    "y_test.npy",
    "feature_columns.json",
)

DEFAULT_SEEDS: Final[tuple[int, ...]] = (
    42,
    43,
    44,
    45,
    46,
)

QUICK_SEEDS: Final[list[int]] = [DEFAULT_SEEDS[0]]
FULL_SEEDS: Final[list[int]] = list(DEFAULT_SEEDS)


# ============================================================================
# RESULT STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision_illicit: float
    recall_illicit: float
    f1_illicit: float
    macro_f1: float
    roc_auc: float
    pr_auc: float
    matthews_correlation: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class ExperimentResult:
    model: str
    seed: int
    threshold: float
    validation_f1_illicit: float
    validation_pr_auc: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_precision_illicit: float
    test_recall_illicit: float
    test_f1_illicit: float
    test_macro_f1: float
    test_roc_auc: float
    test_pr_auc: float
    test_matthews_correlation: float
    test_true_negatives: int
    test_false_positives: int
    test_false_negatives: int
    test_true_positives: int
    test_false_positive_rate: float
    test_false_negative_rate: float
    training_seconds: float
    validation_prediction_seconds: float
    test_prediction_seconds: float


# ============================================================================
# GENERAL UTILITIES
# ============================================================================


def print_separator(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def ensure_directories() -> None:
    for directory in (
        TABLES_DIR,
        FIGURES_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def verify_required_files() -> None:
    missing = [
        PROCESSED_DATA_DIR / filename
        for filename in REQUIRED_FILES
        if not (PROCESSED_DATA_DIR / filename).exists()
    ]

    if missing:
        formatted = "\n".join(str(path) for path in missing)

        raise FileNotFoundError(
            "Preprocessing files are missing:\n"
            f"{formatted}\n\n"
            "Run this first: python src/preprocess.py"
        )


def load_arrays() -> dict[str, np.ndarray]:
    print_separator("DATA LOADING")

    arrays = {
        "x_train": np.load(PROCESSED_DATA_DIR / "x_train.npy"),
        "x_validation": np.load(PROCESSED_DATA_DIR / "x_validation.npy"),
        "x_test": np.load(PROCESSED_DATA_DIR / "x_test.npy"),
        "y_train": np.load(PROCESSED_DATA_DIR / "y_train.npy"),
        "y_validation": np.load(PROCESSED_DATA_DIR / "y_validation.npy"),
        "y_test": np.load(PROCESSED_DATA_DIR / "y_test.npy"),
    }

    for name, array in arrays.items():
        print(f"{name:15s}: shape={array.shape}, dtype={array.dtype}")

    return arrays


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    print_separator("DATA VALIDATION")

    for split in ("train", "validation", "test"):
        x = arrays[f"x_{split}"]
        y = arrays[f"y_{split}"]

        if len(x) != len(y):
            raise ValueError(
                f"X and y have different lengths in {split}: {len(x)} versus {len(y)}."
            )

        if np.isnan(x).any():
            raise ValueError(f"NaN values found in x_{split}.")

        if np.isinf(x).any():
            raise ValueError(f"Infinite values found in x_{split}.")

        labels = np.unique(y)

        if not np.array_equal(
            labels,
            np.array([0, 1], dtype=labels.dtype),
        ):
            raise ValueError(f"Unexpected labels in {split}: {labels.tolist()}")

        licit = int((y == 0).sum())
        illicit = int((y == 1).sum())

        print(
            f"{split:10s}: "
            f"total={len(y):,}, "
            f"licit={licit:,}, "
            f"illicit={illicit:,}, "
            f"illicit={illicit / len(y) * 100:.4f}%"
        )


# ============================================================================
# MODEL CREATION
# ============================================================================


def build_models(
    seed: int,
    quick: bool,
) -> dict[str, BaselineModel]:
    """
    Create comparable models that account for class imbalance.

    Quick-mode parameters reduce execution time.
    """
    random_forest_estimators = 150 if quick else 500
    mlp_max_iter = 120 if quick else 300
    hgb_iterations = 150 if quick else 350

    models: dict[str, BaselineModel] = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=random_forest_estimators,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=hgb_iterations,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
            verbose=False,
        ),
    }

    return models


def fit_supports_sample_weight(
    model: BaselineModel,
) -> bool:
    """
    Check whether the estimator fit method accepts sample_weight.
    """

    fit_method = getattr(model, "fit", None)

    if fit_method is None or not callable(fit_method):
        raise TypeError(f"{model.__class__.__name__} does not provide a fit invocable.")

    signature = inspect.signature(fit_method)

    return "sample_weight" in signature.parameters


def train_model(
    model: BaselineModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """
    Train the model and return elapsed seconds.

    LogisticRegression and RandomForest already incorporate balancing
    through class_weight in its configuration. To avoid
    double weighting, they do not receive additional sample_weight.

    Other models receive balanced sample weights when their method
    fit admite sample_weight.
    """
    start = time.perf_counter()

    already_uses_class_weight = isinstance(
        model,
        (
            LogisticRegression,
            RandomForestClassifier,
        ),
    )

    if not already_uses_class_weight and fit_supports_sample_weight(model):
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
        )
    else:
        model.fit(
            x_train,
            y_train,
        )

    return time.perf_counter() - start


# ============================================================================
# PROBABILITIES AND THRESHOLD
# ============================================================================


def predict_probabilities(
    model: BaselineModel,
    x: np.ndarray,
) -> np.ndarray:
    """
    Return the probability of the illicit class.
    """

    raw_predict_proba = getattr(
        model,
        "predict_proba",
        None,
    )

    if not callable(raw_predict_proba):
        raise TypeError(
            f"{model.__class__.__name__} does not implement "
            "a callable predict_proba method."
        )

    predict_proba = cast(
        Callable[[np.ndarray], np.ndarray],
        raw_predict_proba,
    )

    probabilities = np.asarray(
        predict_proba(x),
        dtype=np.float64,
    )

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "predict_proba did not produce a valid binary matrix: "
            f"{probabilities.shape}"
        )

    return probabilities[:, 1].copy()


def find_best_f1_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """
    Select on validation the threshold that maximizes illicit-class F1.

    precision_recall_curve returns one fewer threshold than the number
    of precision and recall values.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_index = int(np.nanargmax(f1_values))

    best_threshold = float(thresholds[best_index])
    best_f1 = float(f1_values[best_index])

    return best_threshold, best_f1


# ============================================================================
# METRICS
# ============================================================================


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> EvaluationMetrics:
    predictions = (probabilities >= threshold).astype(np.int8)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        precision_illicit=float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        recall_illicit=float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        f1_illicit=float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        macro_f1=float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        matthews_correlation=float(matthews_corrcoef(y_true, predictions)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        false_positive_rate=float(false_positive_rate),
        false_negative_rate=float(false_negative_rate),
    )


# ============================================================================
# SAVING RESULTS
# ============================================================================


def save_predictions(
    model_name: str,
    seed: int,
    threshold: float,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    predictions = (probabilities >= threshold).astype(np.int8)

    output = pd.DataFrame(
        {
            "true_label": y_true.astype(np.int8),
            "probability_illicit": probabilities,
            "predicted_label": predictions,
            "threshold": threshold,
        }
    )

    output_path = TABLES_DIR / f"predictions_{model_name}_seed_{seed}.csv"

    output.to_csv(output_path, index=False)


def save_confusion_matrix(
    model_name: str,
    seed: int,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    predictions = (probabilities >= threshold).astype(np.int8)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Licit", "Illicit"],
    )

    figure, axis = plt.subplots(figsize=(6.5, 5.5))

    display.plot(
        ax=axis,
        values_format=",d",
        colorbar=False,
    )

    axis.set_title(f"{model_name}\nseed={seed}, threshold={threshold:.4f}")

    figure.tight_layout()

    output_path = FIGURES_DIR / f"confusion_matrix_{model_name}_seed_{seed}.png"

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_model(
    model_name: str,
    seed: int,
    model: BaselineModel,
) -> None:
    output_path = CHECKPOINTS_DIR / f"{model_name}_seed_{seed}.joblib"

    joblib.dump(model, output_path)


def save_model_metadata(
    model_name: str,
    seed: int,
    model: BaselineModel,
    threshold: float,
) -> None:
    metadata = {
        "model": model_name,
        "seed": seed,
        "threshold": threshold,
        "class_name": model.__class__.__name__,
        "parameters": model.get_params(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    output_path = CHECKPOINTS_DIR / f"{model_name}_seed_{seed}_metadata.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ============================================================================
# RUNNING AN EXPERIMENT
# ============================================================================


def run_single_experiment(
    model_name: str,
    model: BaselineModel,
    seed: int,
    arrays: dict[str, np.ndarray],
) -> ExperimentResult:
    print_separator(f"MODEL: {model_name} | SEED: {seed}")

    configure_reproducibility(seed)

    x_train = arrays["x_train"]
    y_train = arrays["y_train"]

    x_validation = arrays["x_validation"]
    y_validation = arrays["y_validation"]

    x_test = arrays["x_test"]
    y_test = arrays["y_test"]

    print("Entrenando...")

    with warnings.catch_warnings(record=False):
        warnings.simplefilter(
            "ignore",
            category=ConvergenceWarning,
        )

        training_seconds = train_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
        )

    print(f"Training completed in {training_seconds:.2f} segundos.")

    validation_start = time.perf_counter()

    validation_probabilities = predict_probabilities(
        model,
        x_validation,
    )

    validation_prediction_seconds = time.perf_counter() - validation_start

    threshold, validation_f1 = find_best_f1_threshold(
        y_true=y_validation,
        probabilities=validation_probabilities,
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    print(f"Selected threshold: {threshold:.6f}")
    print(f"Validation illicit-class F1: {validation_f1:.6f}")
    print(f"Validation PR-AUC: {validation_pr_auc:.6f}")

    test_start = time.perf_counter()

    test_probabilities = predict_probabilities(
        model,
        x_test,
    )

    test_prediction_seconds = time.perf_counter() - test_start

    test_metrics = calculate_metrics(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=threshold,
    )

    print("\nTest results:")
    print(f"  Accuracy:             {test_metrics.accuracy:.6f}")
    print(f"  Balanced accuracy:    {test_metrics.balanced_accuracy:.6f}")
    print(f"  Illicit precision:    {test_metrics.precision_illicit:.6f}")
    print(f"  Illicit recall:       {test_metrics.recall_illicit:.6f}")
    print(f"  Illicit F1:           {test_metrics.f1_illicit:.6f}")
    print(f"  Macro-F1:             {test_metrics.macro_f1:.6f}")
    print(f"  ROC-AUC:              {test_metrics.roc_auc:.6f}")
    print(f"  PR-AUC:               {test_metrics.pr_auc:.6f}")
    print(f"  MCC:                  {test_metrics.matthews_correlation:.6f}")
    print(
        f"  TN={test_metrics.true_negatives:,}, "
        f"FP={test_metrics.false_positives:,}, "
        f"FN={test_metrics.false_negatives:,}, "
        f"TP={test_metrics.true_positives:,}"
    )

    save_model(
        model_name=model_name,
        seed=seed,
        model=model,
    )

    save_model_metadata(
        model_name=model_name,
        seed=seed,
        model=model,
        threshold=threshold,
    )

    save_predictions(
        model_name=model_name,
        seed=seed,
        threshold=threshold,
        y_true=y_test,
        probabilities=test_probabilities,
    )

    save_confusion_matrix(
        model_name=model_name,
        seed=seed,
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=threshold,
    )

    return ExperimentResult(
        model=model_name,
        seed=seed,
        threshold=threshold,
        validation_f1_illicit=float(validation_f1),
        validation_pr_auc=float(validation_pr_auc),
        test_accuracy=test_metrics.accuracy,
        test_balanced_accuracy=test_metrics.balanced_accuracy,
        test_precision_illicit=test_metrics.precision_illicit,
        test_recall_illicit=test_metrics.recall_illicit,
        test_f1_illicit=test_metrics.f1_illicit,
        test_macro_f1=test_metrics.macro_f1,
        test_roc_auc=test_metrics.roc_auc,
        test_pr_auc=test_metrics.pr_auc,
        test_matthews_correlation=test_metrics.matthews_correlation,
        test_true_negatives=test_metrics.true_negatives,
        test_false_positives=test_metrics.false_positives,
        test_false_negatives=test_metrics.false_negatives,
        test_true_positives=test_metrics.true_positives,
        test_false_positive_rate=test_metrics.false_positive_rate,
        test_false_negative_rate=test_metrics.false_negative_rate,
        training_seconds=float(training_seconds),
        validation_prediction_seconds=float(validation_prediction_seconds),
        test_prediction_seconds=float(test_prediction_seconds),
    )


# ============================================================================
# SUMMARY ACROSS SEEDS
# ============================================================================


def aggregate_results(
    detailed_results: pd.DataFrame,
) -> pd.DataFrame:
    metrics: list[str] = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision_illicit",
        "test_recall_illicit",
        "test_f1_illicit",
        "test_macro_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_matthews_correlation",
        "test_false_positive_rate",
        "test_false_negative_rate",
        "training_seconds",
    ]

    metric_frame = detailed_results.loc[
        :,
        metrics,
    ].astype(np.float64)

    model_series = detailed_results["model"].astype("string")

    grouped = metric_frame.groupby(
        model_series,
        sort=True,
    )

    means = grouped.mean().add_suffix("_mean")

    standard_deviations = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")

    summary = means.join(standard_deviations).reset_index()

    summary = summary.sort_values(
        by="test_f1_illicit_mean",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def create_formatted_paper_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a compact mean ± standard deviation table.
    """
    output = pd.DataFrame()

    output["Model"] = summary["model"]

    metric_mapping = {
        "Balanced Accuracy": "test_balanced_accuracy",
        "Illicit precision": "test_precision_illicit",
        "Illicit recall": "test_recall_illicit",
        "Illicit F1": "test_f1_illicit",
        "Macro-F1": "test_macro_f1",
        "ROC-AUC": "test_roc_auc",
        "PR-AUC": "test_pr_auc",
        "MCC": "test_matthews_correlation",
    }

    for display_name, internal_name in metric_mapping.items():
        mean_column = f"{internal_name}_mean"
        std_column = f"{internal_name}_std"

        output[display_name] = [
            f"{float(mean):.4f} ± {float(std):.4f}"
            for mean, std in zip(
                summary[mean_column].to_numpy(
                    dtype=np.float64,
                ),
                summary[std_column].to_numpy(
                    dtype=np.float64,
                ),
                strict=True,
            )
        ]

    return output


# ============================================================================
# ARGUMENTOS
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline models to detect illicit Bitcoin transactions."
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a single seed with reduced settings to validate the pipeline.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "mlp",
        ],
        default=None,
        help="Allow execution of selected models only. All models run by default.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN PROGRAM
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the tabular baseline models used by the artifact."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    arguments = parse_arguments()

    print_separator("BASELINE MODEL TRAINING")

    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Project: {portable_path(PROJECT_ROOT)}")
    print(f"Quick mode: {arguments.quick}")

    ensure_directories()
    verify_required_files()

    arrays = load_arrays()
    validate_arrays(arrays)

    seeds = QUICK_SEEDS if arguments.quick else FULL_SEEDS

    selected_models = set(arguments.models) if arguments.models is not None else None

    results: list[ExperimentResult] = []

    for seed in seeds:
        models = build_models(
            seed=seed,
            quick=arguments.quick,
        )

        for model_name, model in models.items():
            if selected_models is not None and model_name not in selected_models:
                continue

            result = run_single_experiment(
                model_name=model_name,
                model=model,
                seed=seed,
                arrays=arrays,
            )

            results.append(result)

            partial_results = pd.DataFrame([asdict(item) for item in results])

            partial_results.to_csv(
                TABLES_DIR / "baseline_results_detailed.csv",
                index=False,
            )

    detailed_results = pd.DataFrame([asdict(result) for result in results])

    detailed_path = TABLES_DIR / "baseline_results_detailed.csv"

    detailed_results.to_csv(
        detailed_path,
        index=False,
    )

    summary = aggregate_results(detailed_results)

    summary_path = TABLES_DIR / "baseline_results_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    paper_table = create_formatted_paper_table(summary)

    paper_table_path = TABLES_DIR / "baseline_results_paper_table.csv"

    paper_table.to_csv(
        paper_table_path,
        index=False,
    )

    print_separator("SUMMARY TABLE")

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(paper_table.to_string(index=False))

    print_separator("TRAINING COMPLETED")

    print(f"Detailed results: {detailed_path}")
    print(f"Statistical summary: {summary_path}")
    print(f"Paper table: {paper_table_path}")
    print(f"Models: {CHECKPOINTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
