"""
Late-fusion for Bitcoin AML classification.

This version constrains the fusion rule to a single global alpha selected from
validation data across all seeds. Only logit fusion is used:

    fused_logit = alpha * hybrid_logit + (1 - alpha) * graphsage_logit

Protocol:
1. Load Hybrid Transformer validation/test predictions for every seed.
2. Reconstruct each GraphSAGE checkpoint and obtain validation/test probabilities.
3. Align by transaction_id, time_step, and label.
4. For every alpha in the grid:
   a. Select one threshold per seed using only that seed's validation split.
   b. Compute validation metrics for every seed.
   c. Aggregate validation metrics across seeds.
5. Select one global alpha using mean validation F1.
   Tie-breaking:
      mean validation MCC
      mean validation PR-AUC
      lower validation F1 standard deviation
      alpha closer to 1.0
6. Apply that same alpha to every seed's test split, with each seed's
   validation-selected threshold.
7. Report paired deltas against the Hybrid Transformer reference.

Example:
    python src/evaluate_late_fusion.py \
        --seeds 42 43 44 45 46 \
        --hybrid-dir results/hybrid_transformer \
        --graphsage-dir results/graphsage \
        --graph-data-dir data/processed/graphsage \
        --output-dir results/late_fusion
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
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

from artifact_utils import portable_path

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


@dataclass(frozen=True)
class PredictionTable:
    transaction_ids: np.ndarray
    time_steps: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass
class SeedBundle:
    seed: int
    hybrid_validation: PredictionTable
    hybrid_test: PredictionTable
    graph_validation: PredictionTable
    graph_test: PredictionTable
    graph_checkpoint_epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate global-alpha logit late fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hybrid-dir",
        type=Path,
        default=Path("results/hybrid_transformer"),
    )
    parser.add_argument(
        "--graphsage-dir",
        type=Path,
        default=Path("results/graphsage"),
    )
    parser.add_argument(
        "--graph-data-dir",
        type=Path,
        default=Path("data/processed/graphsage"),
    )
    parser.add_argument(
        "--graphsage-script",
        type=Path,
        default=Path("src/train_graphsage.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/late_fusion"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def device_to_string(
    device: torch.device,
) -> str:
    """
    Return a stable representation of the device in use.
    """

    if device.type == "cuda":
        return "cuda:0"

    return "cpu"


def resolve_device(
    requested: str,
) -> torch.device:
    """
    Resolve the requested device.
    """

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        return torch.device("cuda:0")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {requested!r}")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_csv(path: Path) -> PredictionTable:
    require_file(path)

    ids: list[int] = []
    times: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"transaction_id", "time_step", "label", "probability"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            ids.append(int(row["transaction_id"]))
            times.append(int(row["time_step"]))
            labels.append(int(row["label"]))
            probabilities.append(float(row["probability"]))

    table = PredictionTable(
        transaction_ids=np.asarray(ids, dtype=np.int64),
        time_steps=np.asarray(times, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
    )
    validate_table(table, str(path))
    return table


def validate_table(table: PredictionTable, source: str) -> None:
    n = table.transaction_ids.size
    if n == 0:
        raise ValueError(f"Empty prediction table: {source}")
    if not (
        table.time_steps.size == n
        and table.labels.size == n
        and table.probabilities.size == n
    ):
        raise ValueError(f"Inconsistent array lengths in {source}.")
    if np.unique(table.transaction_ids).size != n:
        raise ValueError(f"Duplicate transaction IDs in {source}.")
    if not set(np.unique(table.labels).tolist()).issubset({0, 1}):
        raise ValueError(f"Non-binary labels in {source}.")
    if not np.isfinite(table.probabilities).all():
        raise ValueError(f"Non-finite probabilities in {source}.")
    if np.any((table.probabilities < 0.0) | (table.probabilities > 1.0)):
        raise ValueError(f"Probabilities outside [0,1] in {source}.")


def import_graphsage_module(script_path: Path) -> Any:
    require_file(script_path)
    spec = importlib.util.spec_from_file_location(
        "train_graphsage_for_global_late_fusion",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import GraphSAGE module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_graphsage_model(
    module: Any,
    checkpoint: dict[str, Any],
    input_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    config = checkpoint.get("configuration")
    if not isinstance(config, dict):
        raise TypeError("GraphSAGE checkpoint lacks configuration.")

    model = module.CausalGraphSAGE(
        input_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        classifier_hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def graphsage_prediction_table(
    model: torch.nn.Module,
    graph_data: Any,
    mask: torch.Tensor,
) -> PredictionTable:
    logits = model(graph_data.x, graph_data.adjacency)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    idx = indices.cpu().numpy()

    table = PredictionTable(
        transaction_ids=np.asarray(graph_data.ids[idx], dtype=np.int64),
        time_steps=np.asarray(graph_data.times[idx], dtype=np.int64),
        labels=graph_data.y[indices].to(torch.int64).cpu().numpy(),
        probabilities=torch.sigmoid(logits[indices])
        .float()
        .cpu()
        .numpy()
        .astype(np.float64),
    )
    validate_table(table, "generated GraphSAGE predictions")
    return table


def align_tables(
    first: PredictionTable,
    second: PredictionTable,
    context: str,
) -> tuple[PredictionTable, PredictionTable]:
    first_order = np.argsort(first.transaction_ids)
    second_order = np.argsort(second.transaction_ids)

    a = PredictionTable(
        first.transaction_ids[first_order],
        first.time_steps[first_order],
        first.labels[first_order],
        first.probabilities[first_order],
    )
    b = PredictionTable(
        second.transaction_ids[second_order],
        second.time_steps[second_order],
        second.labels[second_order],
        second.probabilities[second_order],
    )

    if not np.array_equal(a.transaction_ids, b.transaction_ids):
        raise ValueError(f"{context}: transaction IDs differ.")
    if not np.array_equal(a.time_steps, b.time_steps):
        raise ValueError(f"{context}: time steps differ.")
    if not np.array_equal(a.labels, b.labels):
        raise ValueError(f"{context}: labels differ.")
    return a, b


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def fuse_logits(
    hybrid_probabilities: np.ndarray,
    graph_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return alpha * safe_logit(hybrid_probabilities) + (1.0 - alpha) * safe_logit(
        graph_probabilities
    )


def select_f1_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.0, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )

    best_f1 = float(np.max(f1_values))
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=1e-12, atol=1e-12))
    center = float(np.median(scores))
    best_index = int(candidates[np.argmin(np.abs(thresholds[candidates] - center))])
    return float(thresholds[best_index]), best_f1


def calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(labels, scores))
            if np.unique(labels).size == 2
            else math.nan
        ),
        "pr_auc": (
            float(average_precision_score(labels, scores))
            if np.unique(labels).size == 2
            else math.nan
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_alpha_grid(step: float) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("--alpha-step must be in (0,1].")
    count = round(1.0 / step)
    grid = np.linspace(0.0, 1.0, count + 1)
    if grid.size < 2 or not np.isclose(grid[1] - grid[0], step):
        raise ValueError("--alpha-step must divide 1 exactly.")
    return grid.astype(np.float64)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) if array.size > 1 else 0.0),
    )


def load_seed_bundle(
    seed: int,
    args: argparse.Namespace,
    graph_module: Any,
    graph_data: Any,
    device: torch.device,
) -> SeedBundle:
    hybrid_seed = args.hybrid_dir / f"seed_{seed}"
    graph_seed = args.graphsage_dir / f"seed_{seed}"

    hybrid_validation = read_prediction_csv(hybrid_seed / "validation_predictions.csv")
    hybrid_test = read_prediction_csv(hybrid_seed / "test_predictions.csv")

    checkpoint_path = graph_seed / "best_model.pt"
    require_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_graphsage_model(
        graph_module,
        checkpoint,
        int(graph_data.x.shape[1]),
        device,
    )

    graph_validation = graphsage_prediction_table(
        model,
        graph_data,
        graph_data.validation_mask,
    )
    graph_test = graphsage_prediction_table(
        model,
        graph_data,
        graph_data.test_mask,
    )

    hybrid_validation, graph_validation = align_tables(
        hybrid_validation,
        graph_validation,
        f"seed {seed} validation",
    )
    hybrid_test, graph_test = align_tables(
        hybrid_test,
        graph_test,
        f"seed {seed} test",
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return SeedBundle(
        seed=seed,
        hybrid_validation=hybrid_validation,
        hybrid_test=hybrid_test,
        graph_validation=graph_validation,
        graph_test=graph_test,
        graph_checkpoint_epoch=int(checkpoint["epoch"]),
    )


def global_candidate_key(
    row: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    return (
        float(row["mean_validation_f1"]),
        float(row["mean_validation_mcc"]),
        float(row["mean_validation_pr_auc"]),
        -float(row["std_validation_f1"]),
        -abs(float(row["alpha_hybrid"]) - 1.0),
    )


def prediction_rows(
    hybrid: PredictionTable,
    graph: PredictionTable,
    fused_scores: np.ndarray,
    threshold: float,
    alpha: float,
) -> list[dict[str, Any]]:
    predictions = (fused_scores >= threshold).astype(np.int64)
    return [
        {
            "transaction_id": int(transaction_id),
            "time_step": int(time_step),
            "label": int(label),
            "hybrid_probability": float(hybrid_probability),
            "graphsage_probability": float(graph_probability),
            "fused_logit": float(fused_score),
            "prediction": int(prediction),
            "threshold": float(threshold),
            "alpha_hybrid": float(alpha),
            "alpha_graphsage": float(1.0 - alpha),
        }
        for (
            transaction_id,
            time_step,
            label,
            hybrid_probability,
            graph_probability,
            fused_score,
            prediction,
        ) in zip(
            hybrid.transaction_ids,
            hybrid.time_steps,
            hybrid.labels,
            hybrid.probabilities,
            graph.probabilities,
            fused_scores,
            predictions,
            strict=True,
        )
    ]


def main() -> None:
    args = parse_args()
    args.hybrid_dir = args.hybrid_dir.resolve()
    args.graphsage_dir = args.graphsage_dir.resolve()
    args.graph_data_dir = args.graph_data_dir.resolve()
    args.graphsage_script = args.graphsage_script.resolve()
    args.output_dir = args.output_dir.resolve()

    device = resolve_device(args.device)
    device_label = device_to_string(device)
    alpha_grid = make_alpha_grid(args.alpha_step)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("BITCOIN AML LATE FUSION: GLOBAL-ALPHA LOGIT FUSION")
    print("=" * 88)
    print(f"PyTorch             : {torch.__version__}")
    print(f"Device              : {device_label}")
    if device.type == "cuda":
        print(f"Hardware            : {torch.cuda.get_device_name(0)}")
    print(f"Hybrid directory    : {args.hybrid_dir}")
    print(f"GraphSAGE directory : {args.graphsage_dir}")
    print(f"Graph data directory: {args.graph_data_dir}")
    print(f"Output directory    : {args.output_dir}")
    print(f"Seeds               : {args.seeds}")
    print(f"Alpha grid          : {alpha_grid.tolist()}")

    graph_module = import_graphsage_module(args.graphsage_script)
    graph_data, graph_metadata = graph_module.load_graph(
        args.graph_data_dir,
        device,
    )

    bundles: list[SeedBundle] = []
    for seed in args.seeds:
        print(f"Loading seed {seed}...")
        bundles.append(
            load_seed_bundle(
                seed,
                args,
                graph_module,
                graph_data,
                device,
            )
        )

    global_search_rows: list[dict[str, Any]] = []
    per_alpha_seed_rows: list[dict[str, Any]] = []

    for alpha in alpha_grid:
        alpha_seed_metrics: list[dict[str, float | int]] = []

        for bundle in bundles:
            validation_scores = fuse_logits(
                bundle.hybrid_validation.probabilities,
                bundle.graph_validation.probabilities,
                float(alpha),
            )
            threshold, _ = select_f1_threshold(
                bundle.hybrid_validation.labels,
                validation_scores,
            )
            metrics = calculate_metrics(
                bundle.hybrid_validation.labels,
                validation_scores,
                threshold,
            )
            alpha_seed_metrics.append(metrics)

            row: dict[str, Any] = {
                "alpha_hybrid": float(alpha),
                "alpha_graphsage": float(1.0 - alpha),
                "seed": bundle.seed,
                "threshold": float(threshold),
            }
            row.update({f"validation_{k}": v for k, v in metrics.items()})
            per_alpha_seed_rows.append(row)

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for metric in METRICS:
            mean_value, std_value = mean_std(
                [float(item[metric]) for item in alpha_seed_metrics]
            )
            means[metric] = mean_value
            stds[metric] = std_value

        global_search_rows.append(
            {
                "alpha_hybrid": float(alpha),
                "alpha_graphsage": float(1.0 - alpha),
                **{f"mean_validation_{metric}": means[metric] for metric in METRICS},
                **{f"std_validation_{metric}": stds[metric] for metric in METRICS},
            }
        )

    best_global = max(global_search_rows, key=global_candidate_key)
    selected_alpha = float(best_global["alpha_hybrid"])

    write_csv(
        args.output_dir / "global_alpha_search.csv",
        global_search_rows,
    )
    write_csv(
        args.output_dir / "validation_metrics_by_alpha_seed.csv",
        per_alpha_seed_rows,
    )

    print("\n" + "=" * 88)
    print("GLOBAL ALPHA SELECTION")
    print("=" * 88)
    print(f"Selected alpha Hybrid   : {selected_alpha:.2f}")
    print(f"Selected alpha GraphSAGE: {1.0 - selected_alpha:.2f}")
    print(
        f"Mean validation F1      : "
        f"{best_global['mean_validation_f1']:.4f} ± "
        f"{best_global['std_validation_f1']:.4f}"
    )
    print(
        f"Mean validation MCC     : "
        f"{best_global['mean_validation_mcc']:.4f} ± "
        f"{best_global['std_validation_mcc']:.4f}"
    )
    print(
        f"Mean validation PR-AUC  : "
        f"{best_global['mean_validation_pr_auc']:.4f} ± "
        f"{best_global['std_validation_pr_auc']:.4f}"
    )

    seed_results: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for bundle in bundles:
        validation_scores = fuse_logits(
            bundle.hybrid_validation.probabilities,
            bundle.graph_validation.probabilities,
            selected_alpha,
        )
        threshold, _ = select_f1_threshold(
            bundle.hybrid_validation.labels,
            validation_scores,
        )
        validation_metrics = calculate_metrics(
            bundle.hybrid_validation.labels,
            validation_scores,
            threshold,
        )

        test_scores = fuse_logits(
            bundle.hybrid_test.probabilities,
            bundle.graph_test.probabilities,
            selected_alpha,
        )
        test_metrics = calculate_metrics(
            bundle.hybrid_test.labels,
            test_scores,
            threshold,
        )

        hybrid_threshold, _ = select_f1_threshold(
            bundle.hybrid_validation.labels,
            bundle.hybrid_validation.probabilities,
        )
        hybrid_test_metrics = calculate_metrics(
            bundle.hybrid_test.labels,
            bundle.hybrid_test.probabilities,
            hybrid_threshold,
        )

        delta = {
            metric: float(test_metrics[metric]) - float(hybrid_test_metrics[metric])
            for metric in METRICS
        }

        seed_result = {
            "seed": bundle.seed,
            "alpha_hybrid": selected_alpha,
            "alpha_graphsage": 1.0 - selected_alpha,
            "threshold": threshold,
            "graph_checkpoint_epoch": bundle.graph_checkpoint_epoch,
            "validation": validation_metrics,
            "test": test_metrics,
            "hybrid_only_reference": {
                "threshold": hybrid_threshold,
                "test": hybrid_test_metrics,
            },
            "delta_vs_hybrid": delta,
        }
        seed_results.append(seed_result)

        seed_dir = args.output_dir / f"seed_{bundle.seed}"
        write_json(seed_dir / "seed_summary.json", seed_result)
        write_csv(
            seed_dir / "validation_predictions.csv",
            prediction_rows(
                bundle.hybrid_validation,
                bundle.graph_validation,
                validation_scores,
                threshold,
                selected_alpha,
            ),
        )
        write_csv(
            seed_dir / "test_predictions.csv",
            prediction_rows(
                bundle.hybrid_test,
                bundle.graph_test,
                test_scores,
                threshold,
                selected_alpha,
            ),
        )

        row: dict[str, Any] = {
            "seed": bundle.seed,
            "alpha_hybrid": selected_alpha,
            "alpha_graphsage": 1.0 - selected_alpha,
            "threshold": threshold,
            "graph_checkpoint_epoch": bundle.graph_checkpoint_epoch,
        }
        for metric in METRICS:
            row[metric] = test_metrics[metric]
            row[f"hybrid_only_{metric}"] = hybrid_test_metrics[metric]
            row[f"delta_{metric}"] = delta[metric]
        row.update(
            {
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
            }
        )
        test_rows.append(row)

        print("\n" + "-" * 88)
        print(f"SEED {bundle.seed}")
        print("-" * 88)
        print(f"Validation threshold: {threshold:.6f}")
        print(f"Validation F1       : {validation_metrics['f1']:.4f}")
        print(f"Test F1             : {test_metrics['f1']:.4f}")
        print(f"Test MCC            : {test_metrics['mcc']:.4f}")
        print(f"Test ROC-AUC        : {test_metrics['roc_auc']:.4f}")
        print(f"Test PR-AUC         : {test_metrics['pr_auc']:.4f}")
        print(f"Delta F1 vs hybrid  : {delta['f1']:+.4f}")

    aggregate: dict[str, Any] = {
        "selected_alpha_hybrid": selected_alpha,
        "selected_alpha_graphsage": 1.0 - selected_alpha,
        "selection_criterion": (
            "mean validation F1; ties by mean validation MCC, "
            "mean validation PR-AUC, lower F1 std, alpha nearest 1"
        ),
        "selected_validation_summary": best_global,
        "metrics": {},
        "hybrid_only_reference_metrics": {},
        "delta_vs_hybrid": {},
        "seeds": args.seeds,
    }

    for metric in METRICS:
        ensemble_values = [float(result["test"][metric]) for result in seed_results]
        hybrid_values = [
            float(result["hybrid_only_reference"]["test"][metric])
            for result in seed_results
        ]
        delta_values = [
            float(result["delta_vs_hybrid"][metric]) for result in seed_results
        ]

        ensemble_mean, ensemble_std = mean_std(ensemble_values)
        hybrid_mean, hybrid_std = mean_std(hybrid_values)
        delta_mean, delta_std = mean_std(delta_values)

        aggregate["metrics"][metric] = {
            "mean": ensemble_mean,
            "standard_deviation": ensemble_std,
            "values": ensemble_values,
        }
        aggregate["hybrid_only_reference_metrics"][metric] = {
            "mean": hybrid_mean,
            "standard_deviation": hybrid_std,
            "values": hybrid_values,
        }
        aggregate["delta_vs_hybrid"][metric] = {
            "mean": delta_mean,
            "standard_deviation": delta_std,
            "values": delta_values,
        }

    config = {
        "hybrid_dir": portable_path(args.hybrid_dir),
        "graphsage_dir": portable_path(args.graphsage_dir),
        "graph_data_dir": portable_path(args.graph_data_dir),
        "graphsage_script": portable_path(args.graphsage_script),
        "output_dir": portable_path(args.output_dir),
        "seeds": args.seeds,
        "alpha_step": args.alpha_step,
        "alpha_grid": alpha_grid.tolist(),
        "device": device_label,
        "graph_metadata": graph_metadata,
    }

    write_json(args.output_dir / "experiment_config.json", config)
    write_csv(args.output_dir / "test_metrics_per_seed.csv", test_rows)
    write_json(args.output_dir / "aggregate_summary.json", aggregate)

    print("\n" + "=" * 88)
    print("FINAL TEST SUMMARY")
    print("=" * 88)
    for metric in METRICS:
        item = aggregate["metrics"][metric]
        delta = aggregate["delta_vs_hybrid"][metric]["mean"]
        print(
            f"{metric:20s}: {item['mean']:.4f} ± "
            f"{item['standard_deviation']:.4f} "
            f"(Δ hybrid {delta:+.4f})"
        )
    print(f"Global hybrid alpha : {selected_alpha:.2f}")
    print(f"Results saved in    : {portable_path(args.output_dir)}")


if __name__ == "__main__":
    main()
