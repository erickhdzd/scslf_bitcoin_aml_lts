"""
Train and evaluate a projected gated Transformer + GraphSAGE fusion model for Bitcoin AML classification.

The model combines:
    1. A contextual Transformer representation of the target transaction.
    2. A frozen GraphSAGE node embedding exported from the matching seed.
    3. The original 165 target features as a direct residual branch.

Each branch is normalized and projected to a common latent space. A learned gate controls the GraphSAGE contribution before classification.

Smoke test:
    python src/train_transformer_graphsage.py \
        --seeds 42 \
        --num-epochs 3 \
        --output-dir results/transformer_graphsage_test

Formal experiment:
    python src/train_transformer_graphsage.py \
        --seeds 42 43 44 45 46 \
        --num-epochs 80 \
        --output-dir results/transformer_graphsage
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
from typing import Any, cast

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
from torch.utils.data import DataLoader, Dataset

from artifact_utils import portable_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    sequence_dir: str
    graphsage_embedding_dir: str
    output_dir: str
    seeds: tuple[int, ...]
    num_epochs: int
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    gradient_clip_norm: float
    embedding_dim: int
    num_heads: int
    num_layers: int
    feedforward_dim: int
    projection_dim: int
    fusion_hidden_dim: int
    classifier_hidden_dim: int
    dropout: float
    num_workers: int
    amp: bool
    device: str


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN behavior improves repeatability. Exact bitwise
    # identity is not guaranteed across different PyTorch/CUDA/GPU versions.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Initialize each DataLoader worker deterministically."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]):
    """Memory-mapped sequence dataset augmented with frozen GraphSAGE embeddings."""

    def __init__(
        self,
        sequence_dir: Path,
        split: str,
        graph_embeddings: Tensor,
        graph_id_to_row: dict[int, int],
    ) -> None:
        self.split = split
        self.graph_embeddings = graph_embeddings

        self.x = np.load(
            sequence_dir / f"sequence_x_{split}.npy", mmap_mode="r", allow_pickle=False
        )
        self.mask = np.load(
            sequence_dir / f"sequence_mask_{split}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.y = np.load(
            sequence_dir / f"sequence_y_{split}.npy", mmap_mode="r", allow_pickle=False
        )
        self.target_ids = np.load(
            sequence_dir / f"sequence_target_ids_{split}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.target_times = np.load(
            sequence_dir / f"sequence_target_times_{split}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self._validate_shapes()

        graph_rows: list[int] = []
        missing_ids: list[int] = []
        for raw_id in np.asarray(self.target_ids).reshape(-1):
            transaction_id = int(raw_id)
            row = graph_id_to_row.get(transaction_id)
            if row is None:
                missing_ids.append(transaction_id)
            else:
                graph_rows.append(row)
        if missing_ids:
            preview = missing_ids[:10]
            raise ValueError(
                f"{split}: {len(missing_ids)} target IDs have no GraphSAGE embedding; examples={preview}"
            )
        self.graph_rows = np.asarray(graph_rows, dtype=np.int64)

    def _validate_shapes(self) -> None:
        n = self.x.shape[0]
        if self.x.ndim != 3:
            raise ValueError(
                f"{self.split}: sequence_x must be three-dimensional; received {self.x.shape}."
            )
        if self.mask.shape != self.x.shape[:2]:
            raise ValueError(
                f"{self.split}: mask shape {self.mask.shape} does not match x shape {self.x.shape[:2]}."
            )
        for name, array in (
            ("y", self.y),
            ("target_ids", self.target_ids),
            ("target_times", self.target_times),
        ):
            if array.shape != (n,):
                raise ValueError(
                    f"{self.split}: {name} must have shape ({n},), received {array.shape}."
                )
        if not np.all(self.mask[:, -1]):
            raise ValueError(
                f"{self.split}: the final token must be valid for every sample."
            )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = torch.from_numpy(np.array(self.x[index], dtype=np.float32, copy=True))
        valid_mask = torch.from_numpy(
            np.array(self.mask[index], dtype=np.bool_, copy=True)
        )
        graph_embedding = self.graph_embeddings[int(self.graph_rows[index])].to(
            dtype=torch.float32
        )
        label = torch.tensor(float(self.y[index]), dtype=torch.float32)
        target_id = torch.tensor(int(self.target_ids[index]), dtype=torch.int64)
        target_time = torch.tensor(int(self.target_times[index]), dtype=torch.int64)
        return x, valid_mask, graph_embedding, label, target_id, target_time


@dataclass
class DataBundle:
    train_dataset: SequenceDataset
    validation_dataset: SequenceDataset
    test_dataset: SequenceDataset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def create_data_bundle(
    sequence_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    graph_embeddings: Tensor,
    graph_id_to_row: dict[int, int],
) -> DataBundle:
    train_dataset = SequenceDataset(
        sequence_dir, "train", graph_embeddings, graph_id_to_row
    )
    validation_dataset = SequenceDataset(
        sequence_dir, "validation", graph_embeddings, graph_id_to_row
    )
    test_dataset = SequenceDataset(
        sequence_dir, "test", graph_embeddings, graph_id_to_row
    )

    expected_shape = train_dataset.x.shape[1:]
    if validation_dataset.x.shape[1:] != expected_shape:
        raise ValueError("Validation sequence dimensions differ from train.")
    if test_dataset.x.shape[1:] != expected_shape:
        raise ValueError("Test sequence dimensions differ from train.")

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    return DataBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransformerGraphSAGEFusion(nn.Module):
    """Projected and gated multimodal fusion model.

    Architecture:
        Transformer target representation [B, 192] -> LayerNorm -> Linear -> [B, 128]
        GraphSAGE embedding              [B, 192] -> LayerNorm -> Linear -> [B, 128]
        Raw target features              [B, 165] -> LayerNorm -> Linear -> [B, 128]

        graph gate = sigmoid(Linear([transformer_projection; graph_projection]))
        gated graph = graph gate * graph_projection

        concatenation [B, 384] -> 128 -> 64 -> 1
    """

    def __init__(
        self,
        input_dim: int,
        max_sequence_length: int,
        embedding_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        projection_dim: int = 128,
        fusion_hidden_dim: int = 128,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.20,
        graph_embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if min(input_dim, graph_embedding_dim, projection_dim) <= 0:
            raise ValueError("Input and projection dimensions must be positive.")

        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.projection_dim = projection_dim

        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
            enable_nested_tensor=False,
        )

        self.transformer_branch = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_branch = nn.Sequential(
            nn.LayerNorm(graph_embedding_dim),
            nn.Linear(graph_embedding_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.raw_branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_gate = nn.Sequential(
            nn.Linear(2 * projection_dim, projection_dim),
            nn.Sigmoid(),
        )

        fusion_dim = 3 * projection_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)

        nn.init.zeros_(cast(Tensor, self.input_projection.bias))

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.input_projection:
                weight = cast(
                    Tensor,
                    module.weight,
                )

                nn.init.xavier_uniform_(weight)

                if module.bias is not None:
                    bias = cast(
                        Tensor,
                        module.bias,
                    )

                    nn.init.zeros_(bias)

    def forward(self, x: Tensor, valid_mask: Tensor, graph_embedding: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be rank 3; received {tuple(x.shape)}.")
        if valid_mask.shape != x.shape[:2]:
            raise ValueError("valid_mask must match the first two dimensions of x.")

        batch_size, sequence_length, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, received {input_dim}."
            )
        if sequence_length > self.max_sequence_length:
            raise ValueError("Sequence length exceeds the configured maximum.")
        if not torch.all(valid_mask[:, -1]):
            raise ValueError("The final target token must be valid.")
        if graph_embedding.shape != (batch_size, self.graph_embedding_dim):
            raise ValueError(
                f"graph_embedding must have shape ({batch_size}, {self.graph_embedding_dim}); "
                f"received {tuple(graph_embedding.shape)}."
            )

        positions = (
            torch.arange(sequence_length, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        hidden = self.input_projection(x) + self.position_embedding(positions)
        hidden = self.input_dropout(hidden)
        encoded = self.encoder(hidden, src_key_padding_mask=~valid_mask)

        transformer_target = self.transformer_branch(encoded[:, -1, :])
        graph_target = self.graph_branch(graph_embedding)
        raw_target = self.raw_branch(x[:, -1, :])

        gate = self.graph_gate(torch.cat((transformer_target, graph_target), dim=-1))
        gated_graph = gate * graph_target
        fused = torch.cat((transformer_target, gated_graph, raw_target), dim=-1)
        return self.classifier(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics and threshold selection
# ---------------------------------------------------------------------------


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def safe_pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


def select_f1_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """
    Select the validation threshold that maximizes positive-class F1.

    Ties are resolved by choosing the threshold nearest to 0.5, producing
    deterministic behavior without favoring systematically lower or higher
    thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if thresholds.size == 0:
        return 0.5, 0.0

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
    candidate_indices = np.flatnonzero(
        np.isclose(f1_values, best_f1, rtol=1e-12, atol=1e-12)
    )
    best_index = int(
        candidate_indices[np.argmin(np.abs(thresholds[candidate_indices] - 0.5))]
    )

    return float(thresholds[best_index]), best_f1


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(y_true, probabilities),
        "pr_auc": safe_pr_auc(y_true, probabilities),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_by_time_step(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_times: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for time_step in np.unique(target_times):
        selector = target_times == time_step
        step_y = y_true[selector]
        step_probabilities = probabilities[selector]
        metrics = calculate_metrics(step_y, step_probabilities, threshold)

        rows.append(
            {
                "time_step": int(time_step),
                "n_samples": int(selector.sum()),
                "n_positive": int(step_y.sum()),
                **metrics,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------


@dataclass
class PredictionOutput:
    mean_loss: float
    labels: np.ndarray
    probabilities: np.ndarray
    target_ids: np.ndarray
    target_times: np.ndarray


def move_batch_to_device(
    batch: tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ],
    device: torch.device,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    non_blocking = device.type == "cuda"

    (
        x,
        valid_mask,
        graph_embedding,
        labels,
        target_ids,
        target_times,
    ) = batch

    return (
        x.to(
            device,
            non_blocking=non_blocking,
        ),
        valid_mask.to(
            device,
            non_blocking=non_blocking,
        ),
        graph_embedding.to(
            device,
            non_blocking=non_blocking,
        ),
        labels.to(
            device,
            non_blocking=non_blocking,
        ),
        target_ids.to(
            device,
            non_blocking=non_blocking,
        ),
        target_times.to(
            device,
            non_blocking=non_blocking,
        ),
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        x, valid_mask, graph_embedding, labels, _, _ = move_batch_to_device(
            batch, device
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(x, valid_mask, graph_embedding)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss encountered: {loss.item()}."
            )

        scaled_loss = scaler.scale(loss)

        scaled_loss.backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        # GradScaler skips optimizer.step() and reduces its scale whenever AMP
        # detects non-finite gradients.
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.inference_mode()
def predict(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> PredictionOutput:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    label_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []

    for batch in data_loader:
        x, valid_mask, graph_embedding, labels, target_ids, target_times = (
            move_batch_to_device(
                batch,
                device,
            )
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(x, valid_mask, graph_embedding)
            loss = criterion(logits, labels)

        probabilities = torch.sigmoid(logits.float())

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        label_chunks.append(labels.cpu().numpy().astype(np.int64))
        probability_chunks.append(probabilities.cpu().numpy())
        id_chunks.append(target_ids.cpu().numpy())
        time_chunks.append(target_times.cpu().numpy())

    return PredictionOutput(
        mean_loss=total_loss / max(total_examples, 1),
        labels=np.concatenate(label_chunks),
        probabilities=np.concatenate(probability_chunks),
        target_ids=np.concatenate(id_chunks),
        target_times=np.concatenate(time_chunks),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_csv(
    path: Path,
    output: PredictionOutput,
    threshold: float,
) -> None:
    predictions = (output.probabilities >= threshold).astype(np.int64)

    rows = [
        {
            "transaction_id": int(transaction_id),
            "time_step": int(time_step),
            "label": int(label),
            "probability": float(probability),
            "prediction": int(prediction),
            "threshold": float(threshold),
        }
        for transaction_id, time_step, label, probability, prediction in zip(
            output.target_ids,
            output.target_times,
            output.labels,
            output.probabilities,
            predictions,
            strict=True,
        )
    ]
    write_csv(path, rows)


# ---------------------------------------------------------------------------
# One-seed experiment
# ---------------------------------------------------------------------------


def build_model(
    input_dim: int,
    max_sequence_length: int,
    graph_embedding_dim: int,
    config: ExperimentConfig,
) -> TransformerGraphSAGEFusion:
    return TransformerGraphSAGEFusion(
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
        projection_dim=config.projection_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        classifier_hidden_dim=config.classifier_hidden_dim,
        dropout=config.dropout,
        graph_embedding_dim=graph_embedding_dim,
    )


def calculate_pos_weight(train_labels: np.ndarray) -> float:
    positives = int(np.sum(train_labels == 1))
    negatives = int(np.sum(train_labels == 0))

    if positives == 0 or negatives == 0:
        raise ValueError(
            "Training labels must contain both positive and negative classes."
        )

    return negatives / positives


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_seed(
    seed: int,
    config: ExperimentConfig,
    sequence_dir: Path,
    graphsage_embedding_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    print("\n" + "=" * 88)
    print(f"SEED {seed}")
    print("=" * 88)

    set_global_seed(seed)
    device_label = device_to_string(device)

    embedding_path = graphsage_embedding_dir / f"graphsage_embeddings_seed{seed}.pt"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing GraphSAGE embeddings: {embedding_path}")
    graph_embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if not isinstance(graph_embeddings, Tensor) or graph_embeddings.ndim != 2:
        raise ValueError(f"Invalid GraphSAGE embedding tensor in {embedding_path}")
    graph_embeddings = graph_embeddings.to(dtype=torch.float32).contiguous()
    graph_ids = np.asarray(np.load(graphsage_embedding_dir / "graph_ids.npy")).reshape(
        -1
    )
    if graph_ids.size != graph_embeddings.shape[0]:
        raise ValueError("graph_ids.npy and GraphSAGE embedding row counts differ.")
    graph_id_to_row = {
        int(transaction_id): row for row, transaction_id in enumerate(graph_ids)
    }
    if len(graph_id_to_row) != graph_ids.size:
        raise ValueError("graph_ids.npy contains duplicate transaction IDs.")

    data = create_data_bundle(
        sequence_dir=sequence_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        seed=seed,
        graph_embeddings=graph_embeddings,
        graph_id_to_row=graph_id_to_row,
    )

    input_dim = int(data.train_dataset.x.shape[2])
    max_sequence_length = int(data.train_dataset.x.shape[1])
    graph_embedding_dim = int(graph_embeddings.shape[1])

    model = build_model(
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        graph_embedding_dim=graph_embedding_dim,
        config=config,
    ).to(device)

    parameter_count = count_parameters(model)

    train_labels = np.asarray(data.train_dataset.y, dtype=np.int64)
    pos_weight_value = calculate_pos_weight(train_labels)
    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

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

    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best_model.pt"

    print(f"Device              : {device_label}")
    print(f"AMP                 : {amp_enabled}")
    print(f"Train samples       : {len(data.train_dataset):,}")
    print(f"Validation samples  : {len(data.validation_dataset):,}")
    print(f"Test samples        : {len(data.test_dataset):,}")
    print(f"Sequence shape      : {max_sequence_length} × {input_dim}")
    print(f"Graph embedding     : {graph_embedding_dim}")
    print(f"Projection dimension: {config.projection_dim}")
    print(f"Fusion dimension    : {3 * config.projection_dim}")
    print(f"Parameters          : {parameter_count:,}")
    print(f"Positive weight     : {pos_weight_value:.6f}")
    print(f"Batch size          : {config.batch_size}")
    print(f"Maximum epochs      : {config.num_epochs}")

    best_validation_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            data_loader=data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip_norm=config.gradient_clip_norm,
        )

        validation_output = predict(
            model=model,
            data_loader=data.validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        validation_threshold, validation_optimal_f1 = select_f1_threshold(
            validation_output.labels,
            validation_output.probabilities,
        )
        validation_metrics = calculate_metrics(
            validation_output.labels,
            validation_output.probabilities,
            validation_threshold,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_start

        history_row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "validation_loss": validation_output.mean_loss,
            "validation_threshold": validation_threshold,
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "validation_mcc": validation_metrics["mcc"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(history_row)

        improved = validation_optimal_f1 > best_validation_f1 + config.min_delta

        if improved:
            best_validation_f1 = validation_optimal_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_threshold": validation_threshold,
                    "validation_metrics": validation_metrics,
                    "input_dim": input_dim,
                    "max_sequence_length": max_sequence_length,
                    "graph_embedding_dim": graph_embedding_dim,
                    "graphsage_embedding_file": portable_path(embedding_path),
                    "model_config": {
                        "embedding_dim": config.embedding_dim,
                        "num_heads": config.num_heads,
                        "num_layers": config.num_layers,
                        "feedforward_dim": config.feedforward_dim,
                        "projection_dim": config.projection_dim,
                        "fusion_hidden_dim": config.fusion_hidden_dim,
                        "classifier_hidden_dim": config.classifier_hidden_dim,
                        "dropout": config.dropout,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{config.num_epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_output.mean_loss:.6f} | "
            f"val_F1={validation_metrics['f1']:.4f} | "
            f"val_MCC={validation_metrics['mcc']:.4f} | "
            f"threshold={validation_threshold:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{epoch_seconds:.1f}s" + (" | BEST" if improved else "")
        )

        scheduler.step()

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}.")
            break

    training_seconds = time.perf_counter() - training_start
    write_csv(seed_dir / "training_history.csv", history)

    if not checkpoint_path.is_file():
        raise RuntimeError("No checkpoint was saved.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = float(checkpoint["validation_threshold"])
    best_epoch = int(checkpoint["epoch"])

    validation_output = predict(
        model=model,
        data_loader=data.validation_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    validation_metrics = calculate_metrics(
        validation_output.labels,
        validation_output.probabilities,
        best_threshold,
    )

    # Test is evaluated only after the best validation checkpoint and threshold
    # have been fixed.
    test_output = predict(
        model=model,
        data_loader=data.test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    test_metrics = calculate_metrics(
        test_output.labels,
        test_output.probabilities,
        best_threshold,
    )
    test_temporal_metrics = metrics_by_time_step(
        test_output.labels,
        test_output.probabilities,
        test_output.target_times,
        best_threshold,
    )

    write_predictions_csv(
        seed_dir / "validation_predictions.csv",
        validation_output,
        best_threshold,
    )
    write_predictions_csv(
        seed_dir / "test_predictions.csv",
        test_output,
        best_threshold,
    )
    write_csv(
        seed_dir / "test_metrics_by_time_step.csv",
        test_temporal_metrics,
    )

    seed_result: dict[str, Any] = {
        "seed": seed,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "parameter_count": parameter_count,
        "pos_weight": pos_weight_value,
        "graph_embedding_dim": graph_embedding_dim,
        "graphsage_embedding_file": portable_path(embedding_path),
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "validation_loss": validation_output.mean_loss,
        "test_loss": test_output.mean_loss,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(seed_dir / "metrics.json", seed_result)

    print("-" * 88)
    print(f"Best epoch          : {best_epoch}")
    print(f"Validation threshold: {best_threshold:.6f}")
    print(f"Test F1             : {test_metrics['f1']:.4f}")
    print(f"Test MCC            : {test_metrics['mcc']:.4f}")
    print(f"Test PR-AUC         : {test_metrics['pr_auc']:.4f}")
    print(f"Training time       : {training_seconds / 60.0:.2f} min")

    del model, optimizer, scheduler, scaler, graph_embeddings, graph_id_to_row
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return seed_result


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------


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


def create_aggregate_summary(
    seed_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_seed_rows: list[dict[str, Any]] = []

    for result in seed_results:
        row: dict[str, Any] = {
            "seed": result["seed"],
            "best_epoch": result["best_epoch"],
            "threshold": result["threshold"],
            "training_seconds": result["training_seconds"],
        }
        for metric in SUMMARY_METRICS:
            row[metric] = result["test"][metric]
        row.update(
            {
                "tn": result["test"]["tn"],
                "fp": result["test"]["fp"],
                "fn": result["test"]["fn"],
                "tp": result["test"]["tp"],
            }
        )
        per_seed_rows.append(row)

    summary: dict[str, Any] = {
        "n_seeds": len(seed_results),
        "seeds": [result["seed"] for result in seed_results],
        "metrics": {},
    }

    for metric in SUMMARY_METRICS:
        values = np.array(
            [float(result["test"][metric]) for result in seed_results],
            dtype=np.float64,
        )
        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            mean = float("nan")
            standard_deviation = float("nan")
        else:
            mean = float(np.mean(finite_values))
            standard_deviation = (
                float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
            )

        summary["metrics"][metric] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "values": values.tolist(),
        }

    return per_seed_rows, summary


# ---------------------------------------------------------------------------
# CLI and validation
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Transformer + GraphSAGE fusion model for Bitcoin AML classification."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("data/processed/sequences"),
    )
    parser.add_argument(
        "--graphsage-embedding-dir",
        type=Path,
        default=Path("data/processed/graphsage_embeddings"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/transformer_graphsage"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)

    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--classifier-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        action="store_true",
        dest="amp",
        help="Enable CUDA automatic mixed precision.",
    )
    amp_group.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable automatic mixed precision.",
    )
    parser.set_defaults(amp=True)

    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda:0")

    return torch.device(requested_device)


def device_to_string(
    device: torch.device,
) -> str:
    """
    Return a stable representation of the device.
    """

    if device.type == "cuda":
        device_index = 0 if device.index is None else int(device.index)

        return f"cuda:{device_index}"

    return "cpu"


def validate_arguments(args: argparse.Namespace) -> None:
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    if args.min_learning_rate < 0:
        raise ValueError("--min-learning-rate cannot be negative.")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate cannot exceed --learning-rate.")
    if args.patience <= 0:
        raise ValueError("--patience must be greater than zero.")
    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be greater than zero.")
    if args.embedding_dim % args.num_heads != 0:
        raise ValueError("--embedding-dim must be divisible by --num-heads.")
    if args.projection_dim <= 0:
        raise ValueError("--projection-dim must be positive.")
    if args.fusion_hidden_dim <= 0:
        raise ValueError("--fusion-hidden-dim must be greater than zero.")
    if args.classifier_hidden_dim <= 0:
        raise ValueError("--classifier-hidden-dim must be greater than zero.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicate values.")


def verify_required_files(
    sequence_dir: Path, graphsage_embedding_dir: Path, seeds: list[int]
) -> None:
    required_files = [
        sequence_dir / "sequence_building_metadata.json",
    ]

    for split in ("train", "validation", "test"):
        required_files.extend(
            [
                sequence_dir / f"sequence_x_{split}.npy",
                sequence_dir / f"sequence_mask_{split}.npy",
                sequence_dir / f"sequence_y_{split}.npy",
                sequence_dir / f"sequence_target_ids_{split}.npy",
                sequence_dir / f"sequence_target_times_{split}.npy",
            ]
        )

    required_files.extend(
        [
            graphsage_embedding_dir / "graph_ids.npy",
            graphsage_embedding_dir / "graph_times.npy",
            graphsage_embedding_dir / "embedding_manifest.json",
        ]
    )
    required_files.extend(
        graphsage_embedding_dir / f"graphsage_embeddings_seed{seed}.pt"
        for seed in seeds
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The following required files are missing:\n" + formatted
        )


def main() -> int:
    args = parse_arguments()
    validate_arguments(args)

    sequence_dir = args.sequence_dir.resolve()
    graphsage_embedding_dir = args.graphsage_embedding_dir.resolve()
    output_dir = args.output_dir.resolve()
    verify_required_files(sequence_dir, graphsage_embedding_dir, args.seeds)

    device = resolve_device(args.device)
    device_label = device_to_string(device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device.index or 0)
    else:
        gpu_name = "CPU"

    config = ExperimentConfig(
        sequence_dir=portable_path(sequence_dir),
        graphsage_embedding_dir=portable_path(graphsage_embedding_dir),
        output_dir=portable_path(output_dir),
        seeds=tuple(args.seeds),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        gradient_clip_norm=args.gradient_clip_norm,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        projection_dim=args.projection_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        num_workers=args.num_workers,
        amp=args.amp,
        device=device_label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_config.json", asdict(config))

    print("=" * 88)
    print("BITCOIN AML PROJECTED GATED TRANSFORMER + GRAPHSAGE FUSION")
    print("=" * 88)
    print(f"PyTorch             : {torch.__version__}")
    print(f"Device              : {device_label}")
    print(f"Hardware            : {gpu_name}")
    print(f"Sequence directory  : {portable_path(sequence_dir)}")
    print(f"Graph embeddings    : {portable_path(graphsage_embedding_dir)}")
    print(f"Output directory    : {portable_path(output_dir)}")
    print(f"Seeds               : {list(config.seeds)}")

    all_start = time.perf_counter()
    seed_results: list[dict[str, Any]] = []

    for seed in config.seeds:
        seed_result = run_seed(
            seed=seed,
            config=config,
            sequence_dir=sequence_dir,
            graphsage_embedding_dir=graphsage_embedding_dir,
            output_dir=output_dir,
            device=device,
        )
        seed_results.append(seed_result)

    per_seed_rows, aggregate_summary = create_aggregate_summary(seed_results)
    aggregate_summary["total_seconds"] = time.perf_counter() - all_start
    aggregate_summary["configuration"] = asdict(config)

    write_csv(output_dir / "test_metrics_per_seed.csv", per_seed_rows)
    write_json(output_dir / "aggregate_summary.json", aggregate_summary)

    print("\n" + "=" * 88)
    print("FINAL TEST SUMMARY")
    print("=" * 88)

    for metric in SUMMARY_METRICS:
        values = aggregate_summary["metrics"][metric]

        mean_value = float(values["mean"])

        standard_deviation = float(values["standard_deviation"])

        print(f"{metric:20s}: {mean_value:.4f} ± {standard_deviation:.4f}")

    total_seconds = float(aggregate_summary["total_seconds"])

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
