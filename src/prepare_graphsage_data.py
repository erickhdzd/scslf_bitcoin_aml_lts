"""
Build the full causal Elliptic transaction graph for GraphSAGE.

All transactions from txs_features.csv become graph nodes, including class 3
(unknown) transactions. The exact train/validation/test membership of the
previous experiments is recovered from sequence_target_ids_*.npy.

For a fair comparison with the 165-dimensional sequence experiments, only:
    Local_feature_1 ... Local_feature_93
    Aggregate_feature_1 ... Aggregate_feature_72
are used. The additional raw transaction statistics are intentionally excluded.

Class mapping:
    1 -> illicit -> 1
    2 -> licit   -> 0
    3 -> unknown -> -1

Unknown nodes participate in message passing but never in loss, threshold
selection, or test evaluation.

Example:
    python src/prepare_graphsage_data.py \
        --features-path data/raw/txs_features.csv \
        --classes-path data/raw/txs_classes.csv \
        --edges-path data/raw/txs_edgelist.csv \
        --output-dir data/processed/graphsage
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from artifact_utils import portable_path

SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full causal GraphSAGE graph with unknown nodes."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/raw/txs_features.csv"),
    )
    parser.add_argument(
        "--classes-path",
        type=Path,
        default=Path("data/raw/txs_classes.csv"),
    )
    parser.add_argument(
        "--edges-path",
        type=Path,
        default=Path("data/raw/txs_edgelist.csv"),
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("data/processed/sequences"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/graphsage"),
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def normalize_id(value: str | float) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_split_ids(sequence_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split in SPLITS:
        path = sequence_dir / f"sequence_target_ids_{split}.npy"
        require_file(path)
        values = np.asarray(np.load(path)).reshape(-1)
        result[split] = {normalize_id(value) for value in values}
    return result


def load_classes(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Class file has no header.")
        fieldnames = list(reader.fieldnames)

        required = {
            "txId",
            "class",
        }

        if not required.issubset(fieldnames):
            raise ValueError(
                f"Class file must contain {required}; found {fieldnames!r}."
            )
        for row in reader:
            tx_id = normalize_id(row["txId"])
            raw_class = str(row["class"]).strip().lower()
            mapping = {
                "1": 1,
                "illicit": 1,
                "2": 0,
                "licit": 0,
                "3": -1,
                "unknown": -1,
            }
            if raw_class not in mapping:
                raise ValueError(f"Unsupported class value: {row['class']!r}")
            labels[tx_id] = mapping[raw_class]
    return labels


def load_features(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Feature file has no header.")

        feature_columns = [
            name
            for name in reader.fieldnames
            if name.startswith(("Local_feature_", "Aggregate_feature_"))
        ]
        if len(feature_columns) != 165:
            raise ValueError(
                "Expected exactly 165 Local/Aggregate features, "
                f"but found {len(feature_columns)}."
            )
        if "txId" not in reader.fieldnames or "Time step" not in reader.fieldnames:
            raise ValueError("Feature file must contain txId and Time step.")

        ids: list[str] = []
        times: list[int] = []
        rows: list[list[float]] = []

        for row_number, row in enumerate(reader, start=2):
            tx_id = normalize_id(row["txId"])
            try:
                time_step = int(float(row["Time step"]))
                features = [float(row[column]) for column in feature_columns]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid feature data at CSV row {row_number}."
                ) from error

            ids.append(tx_id)
            times.append(time_step)
            rows.append(features)

    ids_array = np.asarray(ids, dtype=str)
    if np.unique(ids_array).size != ids_array.size:
        raise ValueError("Duplicate txId values found in the feature file.")

    x = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    return (
        ids_array,
        np.asarray(times, dtype=np.int64),
        x,
        feature_columns,
    )


def main() -> int:
    args = parse_args()
    features_path = args.features_path.expanduser().resolve()
    classes_path = args.classes_path.expanduser().resolve()
    edges_path = args.edges_path.expanduser().resolve()
    sequence_dir = args.sequence_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (features_path, classes_path, edges_path):
        require_file(path)

    split_ids = load_split_ids(sequence_dir)
    classes = load_classes(classes_path)
    ids, times, x, feature_columns = load_features(features_path)

    id_to_index = {tx_id: index for index, tx_id in enumerate(ids.tolist())}
    node_count = ids.size

    y = np.full(node_count, -1, dtype=np.int8)
    split = np.full(node_count, -1, dtype=np.int8)

    missing_class_rows = 0
    for index, tx_id in enumerate(ids.tolist()):
        label = classes.get(tx_id)
        if label is None:
            missing_class_rows += 1
            continue
        y[index] = label

    split_code = {"train": 0, "validation": 1, "test": 2}
    missing_split_ids: dict[str, list[str]] = {}
    for split_name, transaction_ids in split_ids.items():
        missing: list[str] = []
        for tx_id in transaction_ids:
            index = id_to_index.get(tx_id)
            if index is None:
                missing.append(tx_id)
                continue
            if y[index] not in (0, 1):
                raise ValueError(
                    f"Target ID {tx_id} in {split_name} has unknown/missing class."
                )
            split[index] = split_code[split_name]
        missing_split_ids[split_name] = missing

    if any(missing_split_ids.values()):
        counts = {name: len(values) for name, values in missing_split_ids.items()}
        raise ValueError(f"Some sequence target IDs are absent from features: {counts}")

    expected_split_counts = {name: len(values) for name, values in split_ids.items()}
    actual_split_counts = {
        "train": int(np.sum(split == 0)),
        "validation": int(np.sum(split == 1)),
        "test": int(np.sum(split == 2)),
    }
    if expected_split_counts != actual_split_counts:
        raise ValueError(
            f"Split count mismatch: expected {expected_split_counts}, "
            f"found {actual_split_counts}."
        )

    sources: list[int] = []
    targets: list[int] = []
    raw_edge_rows = 0
    missing_endpoint_rows = 0
    future_edges_removed = 0
    self_edges_removed = 0

    with edges_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Edge file has no header.")
        fieldnames = list(reader.fieldnames)

        required = {
            "txId1",
            "txId2",
        }

        if not required.issubset(fieldnames):
            raise ValueError(
                f"Edge file must contain {required}; found {fieldnames!r}."
            )

        for row in reader:
            raw_edge_rows += 1
            source_id = normalize_id(row["txId1"])
            target_id = normalize_id(row["txId2"])
            source_index = id_to_index.get(source_id)

            target_index = id_to_index.get(target_id)

            if source_index is None or target_index is None:
                missing_endpoint_rows += 1
                continue

            source = int(source_index)

            target = int(target_index)

            if source == target:
                self_edges_removed += 1
                continue

            if times[source] > times[target]:
                future_edges_removed += 1
                continue

            sources.append(source)

            targets.append(target)

    if not sources:
        raise RuntimeError("No causal graph edges were retained.")

    pairs = np.column_stack(
        (
            np.asarray(sources, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
        )
    )
    unique_pairs = np.unique(pairs, axis=0)
    duplicate_edges_removed = int(pairs.shape[0] - unique_pairs.shape[0])
    edge_index = unique_pairs.T.astype(np.int64, copy=False)

    indegree = np.bincount(edge_index[1], minlength=node_count)
    outdegree = np.bincount(edge_index[0], minlength=node_count)
    node_count_int = int(node_count)
    feature_dimension = int(x.shape[1])
    edge_count = int(edge_index.shape[1])
    licit_count = int(np.count_nonzero(y == 0))
    illicit_count = int(np.count_nonzero(y == 1))
    unknown_count = int(np.count_nonzero(y == -1))
    message_passing_only_count = int(np.count_nonzero(split == -1))
    nodes_with_incoming_context = int(np.count_nonzero(indegree > 0))
    nodes_without_incoming_edges = int(np.count_nonzero(indegree == 0))
    mean_indegree = float(np.mean(indegree))
    max_indegree = int(np.max(indegree))
    mean_outdegree = float(np.mean(outdegree))
    max_outdegree = int(np.max(outdegree))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "graph_x.npy", x)
    np.save(output_dir / "graph_y.npy", y)
    np.save(output_dir / "graph_ids.npy", ids)
    np.save(output_dir / "graph_times.npy", times)
    np.save(output_dir / "graph_split.npy", split)
    np.save(output_dir / "edge_index.npy", edge_index)

    metadata = {
        "features_path": portable_path(features_path),
        "classes_path": portable_path(classes_path),
        "edges_path": portable_path(edges_path),
        "sequence_dir": portable_path(sequence_dir),
        "node_count": node_count_int,
        "feature_dimension": feature_dimension,
        "feature_columns": feature_columns,
        "edge_count": edge_count,
        "raw_edge_rows": int(raw_edge_rows),
        "missing_endpoint_rows": int(missing_endpoint_rows),
        "future_edges_removed": int(future_edges_removed),
        "self_edges_removed": int(self_edges_removed),
        "duplicate_edges_removed": int(duplicate_edges_removed),
        "missing_class_rows": int(missing_class_rows),
        "class_counts": {
            "licit": licit_count,
            "illicit": illicit_count,
            "unknown": unknown_count,
        },
        "split_counts": actual_split_counts,
        "message_passing_only_nodes": message_passing_only_count,
        "nodes_with_incoming_neighbors": nodes_with_incoming_context,
        "nodes_without_incoming_neighbors": nodes_without_incoming_edges,
        "mean_indegree": mean_indegree,
        "max_indegree": max_indegree,
        "mean_outdegree": mean_outdegree,
        "max_outdegree": max_outdegree,
        "causal_rule": "source_time <= target_time",
        "directed": True,
    }

    with (output_dir / "graph_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("=" * 92)
    print("FULL CAUSAL GRAPHSAGE DATASET")
    print("=" * 92)
    print(f"All nodes                    : {node_count_int:,}")
    print(f"Feature dimension            : {feature_dimension:,}")
    print(f"Licit nodes                  : {licit_count:,}")
    print(f"Illicit nodes                : {illicit_count:,}")
    print(f"Unknown/context nodes        : {unknown_count:,}")
    print(f"Train/validation/test        : {actual_split_counts}")
    print(f"Retained causal edges        : {edge_count:,}")
    print(f"Nodes with incoming context  : {nodes_with_incoming_context:,}")
    print(f"Nodes without incoming edges : {nodes_without_incoming_edges:,}")
    print(f"Mean indegree                : {mean_indegree:.4f}")
    print(f"Future edges removed         : {future_edges_removed:,}")
    print(f"Missing endpoints            : {missing_endpoint_rows:,}")
    print(f"Output directory             : {portable_path(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
