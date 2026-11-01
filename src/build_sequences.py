from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

read_csv_typed = cast(Callable[..., pd.DataFrame], pd.read_csv)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Final[Path] = PROCESSED_DIR / "sequences"

SPLITS: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "test",
)


@dataclass(frozen=True)
class SplitData:
    """Processed data corresponding to a temporal split."""

    name: str
    transaction_ids: np.ndarray
    time_steps: np.ndarray
    features: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class GraphDirection:
    """Result of edge-direction detection."""

    source_column: str
    target_column: str
    forward_strict: int
    reverse_strict: int
    same_time: int
    comparable_edges: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build structural sequences based on "
            "direct predecessors in the transaction graph."
        )
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=16,
        help=(
            "Maximum length of each sequence, including the "
            "target transaction. Default: 16."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--force-direction",
        choices=("txId1_to_txId2", "txId2_to_txId1"),
        default=None,
        help="Force edge direction. If omitted, it is detected from time steps.",
    )

    parser.add_argument(
        "--include-same-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow directed neighbors from the same time_step. "
            "This must remain enabled for Elliptic because its "
            "transaction-to-transaction edges connect nodes from the same "
            "temporal snapshot."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sequence files.",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_split(name: str) -> SplitData:
    csv_path = PROCESSED_DIR / f"{name}.csv"
    x_path = PROCESSED_DIR / f"x_{name}.npy"
    y_path = PROCESSED_DIR / f"y_{name}.npy"
    time_path = PROCESSED_DIR / f"time_{name}.npy"

    for path in (csv_path, x_path, y_path, time_path):
        require_file(path)

    frame = read_csv_typed(
        str(csv_path),
        usecols=["transaction_id", "time_step", "label"],
    )

    features = np.load(x_path, allow_pickle=False)
    labels = np.load(y_path, allow_pickle=False).reshape(-1)
    time_steps_npy = np.load(time_path, allow_pickle=False).reshape(-1)

    transaction_ids = frame["transaction_id"].to_numpy(dtype=np.int64)
    time_steps_csv = frame["time_step"].to_numpy(dtype=np.int16)
    labels_csv = frame["label"].to_numpy(dtype=np.int8)

    if features.ndim != 2:
        raise ValueError(
            f"{x_path.name} must be two-dimensional; shape: {features.shape}"
        )

    row_counts = {
        "CSV": len(frame),
        "X": features.shape[0],
        "Y": labels.shape[0],
        "time": time_steps_npy.shape[0],
    }

    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Inconsistent row count in {name}: {row_counts}")

    if not np.array_equal(
        time_steps_csv.astype(np.int64),
        time_steps_npy.astype(np.int64),
    ):
        raise ValueError(f"time_{name}.npy does not match {name}.csv")

    if not np.array_equal(
        labels_csv.astype(np.int64),
        labels.astype(np.int64),
    ):
        raise ValueError(f"y_{name}.npy does not match the label column of {name}.csv")

    if not np.isfinite(features).all():
        raise ValueError(f"x_{name}.npy contains NaN or infinite values.")

    if len(np.unique(transaction_ids)) != len(transaction_ids):
        raise ValueError(f"{name}.csv contains duplicate transaction_id values.")

    return SplitData(
        name=name,
        transaction_ids=transaction_ids,
        time_steps=time_steps_csv,
        features=features.astype(np.float32, copy=False),
        labels=labels.astype(np.int8, copy=False),
    )


def validate_splits(splits: dict[str, SplitData]) -> int:
    feature_dimensions = {split.features.shape[1] for split in splits.values()}

    if len(feature_dimensions) != 1:
        raise ValueError(
            "The splits do not have the same number of features: "
            f"{sorted(feature_dimensions)}"
        )

    feature_dimension = next(iter(feature_dimensions))

    all_ids = np.concatenate([splits[name].transaction_ids for name in SPLITS])

    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Repeated transaction_id values exist across splits.")

    train_max = int(splits["train"].time_steps.max())
    validation_min = int(splits["validation"].time_steps.min())
    validation_max = int(splits["validation"].time_steps.max())
    test_min = int(splits["test"].time_steps.min())

    if train_max >= validation_min:
        raise ValueError("Train and validation do not respect temporal order.")

    if validation_max >= test_min:
        raise ValueError("Validation and test do not respect temporal order.")

    print("\nSplit validation")
    print("-" * 72)

    for name in SPLITS:
        split = splits[name]
        print(
            f"{name:10s}: "
            f"{len(split.transaction_ids):7d} transactions | "
            f"time {int(split.time_steps.min()):2d}–"
            f"{int(split.time_steps.max()):2d} | "
            f"illicit {int(split.labels.sum()):5d} | "
            f"dimension {split.features.shape[1]}"
        )

    return feature_dimension


def build_global_node_index(
    splits: dict[str, SplitData],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, int],
    dict[int, str],
]:
    all_features = np.concatenate(
        [splits[name].features for name in SPLITS],
        axis=0,
    )

    all_ids = np.concatenate(
        [splits[name].transaction_ids for name in SPLITS],
        axis=0,
    )

    all_times = np.concatenate(
        [splits[name].time_steps for name in SPLITS],
        axis=0,
    )

    id_to_index = {
        int(transaction_id): index for index, transaction_id in enumerate(all_ids)
    }

    id_to_split: dict[int, str] = {}

    for name in SPLITS:
        for transaction_id in splits[name].transaction_ids:
            id_to_split[int(transaction_id)] = name

    return (
        all_ids,
        all_times,
        all_features,
        id_to_index,
        id_to_split,
    )


def determine_graph_direction(
    forced_direction: str | None,
) -> GraphDirection:
    """
    Determine graph direction.

    In Elliptic, txId1 -> txId2 represents the monetary flow
    from a source transaction to a destination transaction.

    All edge endpoints belong to the same time_step,
    therefore direction cannot be inferred from time alone.
    """

    if forced_direction in (None, "txId1_to_txId2"):
        source_column = "txId1"
        target_column = "txId2"

    elif forced_direction == "txId2_to_txId1":
        source_column = "txId2"
        target_column = "txId1"

    else:
        raise ValueError(f"Unrecognized direction: {forced_direction}")

    return GraphDirection(
        source_column=source_column,
        target_column=target_column,
        forward_strict=0,
        reverse_strict=0,
        same_time=0,
        comparable_edges=0,
    )


def build_predecessor_map(
    edges: pd.DataFrame,
    graph_direction: GraphDirection,
    id_to_index: dict[int, int],
    all_times: np.ndarray,
    include_same_time: bool,
) -> tuple[dict[int, list[int]], dict[str, int]]:
    predecessor_sets: dict[int, set[int]] = defaultdict(set)

    statistics = {
        "total_edges": len(edges),
        "edges_with_both_known_nodes": 0,
        "edges_with_unknown_node": 0,
        "self_loops_removed": 0,
        "temporally_invalid_edges_removed": 0,
        "same_time_edges_removed": 0,
        "valid_edges": 0,
    }

    if not include_same_time:
        raise ValueError(
            "Elliptic connects transactions within the same time_step. "
            "Do not use --no-include-same-time because it would remove "
            "the graph edges."
        )

    edge_columns = edges[
        [
            graph_direction.source_column,
            graph_direction.target_column,
        ]
    ]

    for source_id, target_id in edge_columns.itertuples(
        index=False,
        name=None,
    ):
        source_id = int(source_id)
        target_id = int(target_id)

        source_index = id_to_index.get(source_id)
        target_index = id_to_index.get(target_id)

        if source_index is None or target_index is None:
            statistics["edges_with_unknown_node"] += 1
            continue

        statistics["edges_with_both_known_nodes"] += 1

        if source_id == target_id:
            statistics["self_loops_removed"] += 1
            continue

        source_time = int(all_times[source_index])
        target_time = int(all_times[target_index])

        if source_time > target_time:
            statistics["temporally_invalid_edges_removed"] += 1
            continue

        if source_time == target_time and not include_same_time:
            statistics["same_time_edges_removed"] += 1
            continue

        predecessor_sets[target_id].add(source_id)
        statistics["valid_edges"] += 1

    predecessors: dict[int, list[int]] = {}

    for target_id, source_ids in predecessor_sets.items():
        predecessors[target_id] = sorted(source_ids)

    return predecessors, statistics


def check_context_split(
    context_id: int,
    target_split: str,
    id_to_split: dict[int, str],
) -> bool:
    context_split = id_to_split[context_id]

    allowed_context = {
        "train": {"train"},
        "validation": {"train", "validation"},
        "test": {"train", "validation", "test"},
    }

    return context_split in allowed_context[target_split]


def build_split_sequences(
    split: SplitData,
    max_sequence_length: int,
    feature_dimension: int,
    predecessors: dict[int, list[int]],
    id_to_index: dict[int, int],
    id_to_split: dict[int, str],
    all_times: np.ndarray,
    all_features: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    number_of_targets = len(split.transaction_ids)

    sequences = np.zeros(
        (
            number_of_targets,
            max_sequence_length,
            feature_dimension,
        ),
        dtype=np.float32,
    )

    masks = np.zeros(
        (number_of_targets, max_sequence_length),
        dtype=np.bool_,
    )

    sequence_lengths = np.ones(
        number_of_targets,
        dtype=np.int16,
    )

    available_predecessor_counts = np.zeros(
        number_of_targets,
        dtype=np.int32,
    )

    selected_predecessor_counts = np.zeros(
        number_of_targets,
        dtype=np.int16,
    )

    selected_transaction_ids = np.full(
        (number_of_targets, max_sequence_length),
        fill_value=-1,
        dtype=np.int64,
    )

    max_predecessors = max_sequence_length - 1

    for row_index, target_id_raw in enumerate(split.transaction_ids):
        target_id = int(target_id_raw)
        target_index = id_to_index[target_id]
        target_time = int(all_times[target_index])

        valid_predecessors: list[int] = []

        for predecessor_id in predecessors.get(target_id, []):
            predecessor_index = id_to_index[predecessor_id]
            predecessor_time = int(all_times[predecessor_index])

            if predecessor_time > target_time:
                continue

            if not check_context_split(
                context_id=predecessor_id,
                target_split=split.name,
                id_to_split=id_to_split,
            ):
                continue

            valid_predecessors.append(predecessor_id)

        available_predecessor_counts[row_index] = len(valid_predecessors)

        if max_predecessors > 0:
            selected_predecessors = valid_predecessors[-max_predecessors:]
        else:
            selected_predecessors = []

        token_ids = selected_predecessors + [target_id]
        current_length = len(token_ids)
        start_position = max_sequence_length - current_length

        sequence_lengths[row_index] = current_length
        selected_predecessor_counts[row_index] = len(selected_predecessors)

        for offset, transaction_id in enumerate(token_ids):
            sequence_position = start_position + offset
            node_index = id_to_index[transaction_id]

            sequences[
                row_index,
                sequence_position,
                :,
            ] = all_features[node_index]

            masks[row_index, sequence_position] = True
            selected_transaction_ids[
                row_index,
                sequence_position,
            ] = transaction_id

        if selected_transaction_ids[row_index, -1] != target_id:
            raise RuntimeError(
                "The target transaction was not placed in the final sequence position."
            )

    statistics: dict[str, float | int] = {
        "number_of_sequences": int(number_of_targets),
        "sequence_length": int(max_sequence_length),
        "feature_dimension": int(feature_dimension),
        "sequences_without_predecessors": int(
            np.sum(available_predecessor_counts == 0)
        ),
        "sequences_with_predecessors": int(np.sum(available_predecessor_counts > 0)),
        "sequences_truncated": int(
            np.sum(available_predecessor_counts > max_predecessors)
        ),
        "mean_available_predecessors": float(available_predecessor_counts.mean()),
        "median_available_predecessors": float(np.median(available_predecessor_counts)),
        "max_available_predecessors": int(available_predecessor_counts.max(initial=0)),
        "mean_selected_predecessors": float(selected_predecessor_counts.mean()),
        "mean_sequence_length": float(sequence_lengths.mean()),
    }

    arrays = {
        "x": sequences,
        "mask": masks,
        "y": split.labels.astype(np.int8, copy=False),
        "target_ids": split.transaction_ids.astype(
            np.int64,
            copy=False,
        ),
        "target_times": split.time_steps.astype(
            np.int16,
            copy=False,
        ),
        "sequence_lengths": sequence_lengths,
        "available_predecessor_counts": available_predecessor_counts,
        "selected_predecessor_counts": selected_predecessor_counts,
        "sequence_transaction_ids": selected_transaction_ids,
    }

    return arrays, statistics


def ensure_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_outputs = list(output_dir.glob("*.npy"))
    existing_outputs.extend(output_dir.glob("*.json"))

    if existing_outputs and not overwrite:
        preview = "\n".join(
            f"  - {path.name}" for path in sorted(existing_outputs)[:10]
        )

        raise FileExistsError(
            "The output directory already contains files. "
            "Use --overwrite to replace them:\n"
            f"{preview}"
        )


def save_split_arrays(
    output_dir: Path,
    split_name: str,
    arrays: dict[str, np.ndarray],
) -> None:
    filename_mapping = {
        "x": f"sequence_x_{split_name}.npy",
        "mask": f"sequence_mask_{split_name}.npy",
        "y": f"sequence_y_{split_name}.npy",
        "target_ids": f"sequence_target_ids_{split_name}.npy",
        "target_times": f"sequence_target_times_{split_name}.npy",
        "sequence_lengths": f"sequence_lengths_{split_name}.npy",
        "available_predecessor_counts": (
            f"available_predecessor_counts_{split_name}.npy"
        ),
        "selected_predecessor_counts": (
            f"selected_predecessor_counts_{split_name}.npy"
        ),
        "sequence_transaction_ids": f"sequence_transaction_ids_{split_name}.npy",
    }

    for key, filename in filename_mapping.items():
        path = output_dir / filename
        np.save(path, arrays[key], allow_pickle=False)


def main() -> int:
    args = parse_args()

    if args.max_sequence_length < 1:
        raise ValueError("--max-sequence-length must be at least 1.")

    print("=" * 72)
    print("TRANSFORMER SEQUENCE CONSTRUCTION")
    print("=" * 72)
    print(f"Project: {PROJECT_ROOT.name}")
    print(f"Maximum length: {args.max_sequence_length}")
    print(f"Output: {args.output_dir}")

    ensure_output_directory(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    splits = {name: load_split(name) for name in SPLITS}

    feature_dimension = validate_splits(splits)

    (
        all_ids,
        all_times,
        all_features,
        id_to_index,
        id_to_split,
    ) = build_global_node_index(splits)

    edges_path = RAW_DIR / "txs_edgelist.csv"
    require_file(edges_path)

    edges = read_csv_typed(
        str(edges_path),
        usecols=["txId1", "txId2"],
        dtype={
            "txId1": np.int64,
            "txId2": np.int64,
        },
    )

    graph_direction = determine_graph_direction(
        forced_direction=args.force_direction,
    )

    print("\nGraph direction")
    print("-" * 72)
    print(
        f"Direction used: "
        f"{graph_direction.source_column} -> "
        f"{graph_direction.target_column}"
    )
    print(
        "Interpretation: Bitcoin flow from the source transaction "
        "to the destination transaction."
    )
    print(
        "Note: edges connect transactions from the same time_step; "
        "temporal order does not determine its direction."
    )

    predecessors, graph_statistics = build_predecessor_map(
        edges=edges,
        graph_direction=graph_direction,
        id_to_index=id_to_index,
        all_times=all_times,
        include_same_time=args.include_same_time,
    )

    print("\nEdge filtering")
    print("-" * 72)

    for key, value in graph_statistics.items():
        print(f"{key:36s}: {value:,}")

    split_statistics: dict[
        str,
        dict[str, float | int],
    ] = {}

    metadata: dict[str, object] = {
        "project_root": ".",
        "max_sequence_length": args.max_sequence_length,
        "feature_dimension": feature_dimension,
        "padding": "left",
        "padding_value": 0.0,
        "mask_encoding": {
            "padding": False,
            "real_token": True,
        },
        "target_position": "final_sequence_position",
        "predecessor_selection": "direct_incoming_neighbors_sorted_by_transaction_id",
        "sequence_semantics": (
            "structural_incoming-neighborhood sequence; "
            "neighbor order is deterministic but not chronological"
        ),
        "include_same_time_predecessors": args.include_same_time,
        "graph_direction": {
            "source_column": graph_direction.source_column,
            "target_column": graph_direction.target_column,
            "forward_strict": graph_direction.forward_strict,
            "reverse_strict": graph_direction.reverse_strict,
            "same_time": graph_direction.same_time,
            "comparable_edges": graph_direction.comparable_edges,
        },
        "graph_statistics": graph_statistics,
        "split_statistics": split_statistics,
    }

    for split_name in SPLITS:
        print(f"\nBuilding sequences: {split_name}")

        arrays, statistics = build_split_sequences(
            split=splits[split_name],
            max_sequence_length=args.max_sequence_length,
            feature_dimension=feature_dimension,
            predecessors=predecessors,
            id_to_index=id_to_index,
            id_to_split=id_to_split,
            all_times=all_times,
            all_features=all_features,
        )

        save_split_arrays(
            output_dir=args.output_dir,
            split_name=split_name,
            arrays=arrays,
        )

        split_statistics[split_name] = statistics

        print(f"  Sequences: {statistics['number_of_sequences']:,}")
        print(
            f"  Without predecessors: {statistics['sequences_without_predecessors']:,}"
        )
        print(f"  With predecessors: {statistics['sequences_with_predecessors']:,}")
        print(f"  Truncadas: {statistics['sequences_truncated']:,}")
        print(f"  Mean length: {statistics['mean_sequence_length']:.3f}")

    metadata_path = args.output_dir / "sequence_building_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("SEQUENCES GENERATED SUCCESSFULLY")
    print("=" * 72)
    print(f"Directory: {args.output_dir}")
    print(f"Metadata: {metadata_path.name}")
    print(f"Processed nodes: {len(all_ids):,}")
    print(f"Input dimension: {feature_dimension}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
