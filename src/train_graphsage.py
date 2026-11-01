"""
Train a full-batch causal GraphSAGE model for Bitcoin AML classification.

This implementation uses only PyTorch; PyTorch Geometric is not required.
Message passing aggregates directed incoming neighbors from a graph prepared by
prepare_graphsage_data.py. Loss is computed only on training nodes, the
classification threshold and checkpoint are selected only on validation nodes,
and test is evaluated once with the selected checkpoint.

Smoke test:
    python src/train_graphsage.py \
        --seeds 42 \
        --num-epochs 3 \
        --output-dir results/graphsage_test

Formal experiment:
    python src/train_graphsage.py \
        --seeds 42 43 44 45 46 \
        --num-epochs 200 \
        --output-dir results/graphsage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
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
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from artifact_utils import portable_path

SUMMARY_METRICS = (
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
class ExperimentConfig:
    graph_dir: str
    output_dir: str
    seeds: tuple[int, ...]
    num_epochs: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    gradient_clip_norm: float
    hidden_dim: int
    num_layers: int
    classifier_hidden_dim: int
    dropout: float
    device: str


@dataclass
class GraphData:
    x: Tensor
    y: Tensor
    times: np.ndarray
    ids: np.ndarray
    train_mask: Tensor
    validation_mask: Tensor
    test_mask: Tensor
    adjacency: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal full-batch GraphSAGE AML classifier."
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/processed/graphsage"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/graphsage"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--classifier-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be greater than zero.")
    if args.patience <= 0:
        raise ValueError("--patience must be greater than zero.")
    if args.hidden_dim <= 0 or args.classifier_hidden_dim <= 0:
        raise ValueError("Hidden dimensions must be greater than zero.")
    if args.num_layers not in (1, 2, 3):
        raise ValueError("--num-layers must be 1, 2, or 3.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


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


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required graph file not found: {path}")


def standardize_with_train(
    x: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = x[train_mask]
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0

    standardized = (x.astype(np.float64) - mean) / std
    standardized = np.clip(standardized, -10.0, 10.0).astype(np.float32)
    return standardized, mean.astype(np.float32), std.astype(np.float32)


def build_mean_incoming_adjacency(
    edge_index: np.ndarray,
    node_count: int,
    device: torch.device,
) -> Tensor:
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index.npy must have shape [2, edge_count].")

    source = torch.from_numpy(
        edge_index[0].astype(
            np.int64,
            copy=False,
        )
    )

    target = torch.from_numpy(
        edge_index[1].astype(
            np.int64,
            copy=False,
        )
    )

    indegree = torch.bincount(
        target,
        minlength=node_count,
    ).clamp_min(1)

    incoming_degree = indegree[target].to(
        dtype=torch.float32,
    )

    values: Tensor = incoming_degree.reciprocal()

    indices = torch.stack(
        (target, source),
        dim=0,
    )

    adjacency = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(
            node_count,
            node_count,
        ),
        dtype=torch.float32,
        check_invariants=False,
    ).coalesce()

    return adjacency.to(device)


def load_graph(
    graph_dir: Path, device: torch.device
) -> tuple[GraphData, dict[str, Any]]:
    paths = {
        "x": graph_dir / "graph_x.npy",
        "y": graph_dir / "graph_y.npy",
        "ids": graph_dir / "graph_ids.npy",
        "times": graph_dir / "graph_times.npy",
        "split": graph_dir / "graph_split.npy",
        "edges": graph_dir / "edge_index.npy",
        "metadata": graph_dir / "graph_metadata.json",
    }
    for key, path in paths.items():
        if key != "metadata":
            require_file(path)

    x = np.asarray(np.load(paths["x"]), dtype=np.float32)
    y = np.asarray(np.load(paths["y"]), dtype=np.float32).reshape(-1)
    ids = np.asarray(np.load(paths["ids"])).reshape(-1)
    times = np.asarray(np.load(paths["times"]), dtype=np.int64).reshape(-1)
    split = np.asarray(np.load(paths["split"]), dtype=np.int8).reshape(-1)
    edge_index = np.asarray(np.load(paths["edges"]), dtype=np.int64)

    if x.ndim != 2:
        raise ValueError("graph_x.npy must have shape [nodes, features].")
    if not (x.shape[0] == y.size == ids.size == times.size == split.size):
        raise ValueError("Graph node arrays have inconsistent sizes.")
    if not np.isfinite(x).all():
        raise ValueError("graph_x.npy contains NaN or infinite values.")
    if not set(np.unique(split).tolist()).issubset({-1, 0, 1, 2}):
        raise ValueError("graph_split.npy must contain only -1, 0, 1, and 2.")

    train_np = split == 0
    validation_np = split == 1
    test_np = split == 2
    # The selected Local/Aggregate features are already normalized in the
    # Elliptic source file. Re-standardizing them would make this GNN
    # experiment less comparable with the sequence models.
    mean = np.zeros(x.shape[1], dtype=np.float32)
    std = np.ones(x.shape[1], dtype=np.float32)

    adjacency = build_mean_incoming_adjacency(
        edge_index=edge_index,
        node_count=x.shape[0],
        device=device,
    )

    metadata: dict[str, Any] = {}
    if paths["metadata"].is_file():
        with paths["metadata"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata["standardization"] = {
        "fit_split": "not_applied_already_normalized_features",
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "clip": [-10.0, 10.0],
    }

    return (
        GraphData(
            x=torch.from_numpy(x).to(device),
            y=torch.from_numpy(y).to(device),
            times=times,
            ids=ids,
            train_mask=torch.from_numpy(train_np).to(device),
            validation_mask=torch.from_numpy(validation_np).to(device),
            test_mask=torch.from_numpy(test_np).to(device),
            adjacency=adjacency,
        ),
        metadata,
    )


class GraphSAGELayer(nn.Module):
    """Mean-aggregating GraphSAGE layer with separate self and neighbor weights."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim, bias=True)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.zeros_(self.self_linear.bias)
        nn.init.xavier_uniform_(self.neighbor_linear.weight)
        self.norm.reset_parameters()

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        neighbor_mean = torch.sparse.mm(adjacency, x)
        output = self.self_linear(x) + self.neighbor_linear(neighbor_mean)
        return self.norm(output)


class CausalGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        classifier_hidden_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        dimensions = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            GraphSAGELayer(dimensions[i], dimensions[i + 1]) for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
            hidden = torch.nn.functional.gelu(hidden)
            hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(-1)


def calculate_pos_weight(y: Tensor, train_mask: Tensor) -> Tensor:
    train_y = y[train_mask]
    positives = torch.sum(train_y == 1).to(torch.float32)
    negatives = torch.sum(train_y == 0).to(torch.float32)
    if positives.item() == 0:
        raise ValueError("Training split contains no positive samples.")
    return negatives / positives


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5, float(f1_score(labels, probabilities >= 0.5, zero_division=0))

    f1_values = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best = int(np.nanargmax(f1_values))
    return float(thresholds[best]), float(f1_values[best])


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if np.unique(labels).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["roc_auc"] = math.nan
        metrics["pr_auc"] = math.nan

    return metrics


@torch.no_grad()
def probabilities_for_mask(
    model: nn.Module,
    data: GraphData,
    mask: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(data.x, data.adjacency)
    probabilities = torch.sigmoid(logits[mask]).float().cpu().numpy()
    labels = data.y[mask].to(torch.int8).cpu().numpy()
    return labels, probabilities


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(
    seed: int,
    config: ExperimentConfig,
    data: GraphData,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_global_seed(seed)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = CausalGraphSAGE(
        input_dim=data.x.shape[1],
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        classifier_hidden_dim=config.classifier_hidden_dim,
        dropout=config.dropout,
    ).to(device)

    pos_weight = calculate_pos_weight(data.y, data.train_mask)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_learning_rate,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = seed_dir / "best_model.pt"

    start = time.perf_counter()

    device_label = device_to_string(device)

    print("\n" + "=" * 88)
    print(f"SEED {seed}")
    print("=" * 88)
    print(f"Device              : {device_label}")
    print(f"Nodes               : {data.x.shape[0]:,}")
    print(f"Features            : {data.x.shape[1]:,}")
    print(f"Parameters          : {parameter_count:,}")
    print(f"Positive weight     : {pos_weight.item():.6f}")
    print(f"Maximum epochs      : {config.num_epochs}")

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.x, data.adjacency)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        validation_labels, validation_probabilities = probabilities_for_mask(
            model, data, data.validation_mask
        )
        threshold, validation_f1 = select_threshold(
            validation_labels, validation_probabilities
        )
        validation_metrics = compute_metrics(
            validation_labels,
            validation_probabilities,
            threshold,
        )

        improved = validation_f1 > best_f1 + config.min_delta
        if improved:
            best_f1 = validation_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "threshold": threshold,
                    "validation_metrics": validation_metrics,
                    "configuration": asdict(config),
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "validation_f1": validation_metrics["f1"],
            "validation_mcc": validation_metrics["mcc"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "threshold": threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
            "best": improved,
        }
        history.append(row)

        marker = " | BEST" if improved else ""
        print(
            f"Epoch {epoch:03d}/{config.num_epochs:03d} | "
            f"train_loss={loss.item():.6f} | "
            f"val_F1={validation_metrics['f1']:.4f} | "
            f"val_MCC={validation_metrics['mcc']:.4f} | "
            f"threshold={threshold:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{row['seconds']:.2f}s{marker}"
        )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_threshold = float(checkpoint["threshold"])

    test_labels, test_probabilities = probabilities_for_mask(
        model, data, data.test_mask
    )
    test_metrics = compute_metrics(
        test_labels,
        test_probabilities,
        selected_threshold,
    )

    validation_labels, validation_probabilities = probabilities_for_mask(
        model, data, data.validation_mask
    )
    validation_metrics = compute_metrics(
        validation_labels,
        validation_probabilities,
        selected_threshold,
    )

    test_indices = (
        torch.nonzero(data.test_mask, as_tuple=False).squeeze(1).cpu().numpy()
    )
    prediction_rows = [
        {
            "transaction_id": data.ids[index],
            "time_step": int(data.times[index]),
            "label": int(test_labels[position]),
            "probability": float(test_probabilities[position]),
            "prediction": int(test_probabilities[position] >= selected_threshold),
        }
        for position, index in enumerate(test_indices)
    ]

    result = {
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "threshold": selected_threshold,
        "parameter_count": parameter_count,
        "positive_weight": float(pos_weight.item()),
        "training_seconds": time.perf_counter() - start,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    write_csv(seed_dir / "training_history.csv", history)
    write_csv(seed_dir / "test_predictions.csv", prediction_rows)
    write_json(seed_dir / "seed_summary.json", result)

    print("-" * 88)
    print(f"Best epoch          : {result['best_epoch']}")
    print(f"Validation threshold: {selected_threshold:.6f}")
    print(f"Test F1             : {test_metrics['f1']:.4f}")
    print(f"Test MCC            : {test_metrics['mcc']:.4f}")
    print(f"Test PR-AUC         : {test_metrics['pr_auc']:.4f}")
    print(f"Training time       : {result['training_seconds'] / 60.0:.2f} min")

    return result


def aggregate_results(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "seed": result["seed"],
            "best_epoch": result["best_epoch"],
            "threshold": result["threshold"],
        }
        row.update(result["test_metrics"])
        rows.append(row)

    summary: dict[str, Any] = {"seed_count": len(results), "metrics": {}}
    for metric in SUMMARY_METRICS:
        values = np.asarray(
            [result["test_metrics"][metric] for result in results],
            dtype=np.float64,
        )
        summary["metrics"][metric] = {
            "mean": float(np.nanmean(values)),
            "standard_deviation": float(
                np.nanstd(values, ddof=1) if values.size > 1 else 0.0
            ),
            "values": values.tolist(),
        }

    return rows, summary


def main() -> int:
    args = parse_args()
    validate_args(args)

    graph_dir = args.graph_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device = resolve_device(args.device)

    device_label = device_to_string(device)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = ExperimentConfig(
        graph_dir=portable_path(graph_dir),
        output_dir=portable_path(output_dir),
        seeds=tuple(args.seeds),
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        gradient_clip_norm=args.gradient_clip_norm,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        device=device_label,
    )

    data, graph_metadata = load_graph(graph_dir, device)
    write_json(output_dir / "experiment_config.json", asdict(config))
    write_json(output_dir / "graph_metadata_used.json", graph_metadata)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    print("=" * 88)
    print("BITCOIN AML CAUSAL GRAPHSAGE")
    print("=" * 88)
    print(f"PyTorch             : {torch.__version__}")
    print(f"Device              : {device_label}")
    print(f"Hardware            : {gpu_name}")
    print(f"Graph directory     : {portable_path(graph_dir)}")
    print(f"Output directory    : {portable_path(output_dir)}")
    print(f"Seeds               : {list(config.seeds)}")
    print(f"Directed edges      : {data.adjacency._nnz():,}")

    all_start = time.perf_counter()
    results = [
        run_seed(
            seed=seed,
            config=config,
            data=data,
            output_dir=output_dir,
            device=device,
        )
        for seed in config.seeds
    ]

    rows, summary = aggregate_results(results)

    total_seconds = float(time.perf_counter() - all_start)

    summary["total_seconds"] = total_seconds
    summary["configuration"] = asdict(config)

    write_csv(output_dir / "test_metrics_per_seed.csv", rows)
    write_json(output_dir / "aggregate_summary.json", summary)

    print("\n" + "=" * 88)
    print("FINAL TEST SUMMARY")
    print("=" * 88)
    for metric in SUMMARY_METRICS:
        values = summary["metrics"][metric]
        print(
            f"{metric:20s}: {values['mean']:.4f} ± {values['standard_deviation']:.4f}"
        )
    print(f"Total time          : {total_seconds / 60.0:.2f} min")
    print(f"Results saved in    : {portable_path(output_dir)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by the user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
