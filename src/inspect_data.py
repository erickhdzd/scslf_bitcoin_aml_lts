from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

from artifact_utils import portable_path

read_csv_typed = cast(Callable[..., pd.DataFrame], pd.read_csv)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = {
    "features": RAW_DATA_DIR / "txs_features.csv",
    "classes": RAW_DATA_DIR / "txs_classes.csv",
    "edges": RAW_DATA_DIR / "txs_edgelist.csv",
}


def print_separator(title: str) -> None:
    """Print a visible heading to separate sections."""
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def human_readable_size(size_bytes: int) -> str:
    """Convert a byte count to a human-readable representation."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size_bytes} B"


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute the SHA-256 digest of a file.

    The hash records the exact dataset version
    used in the experiments.
    """
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def detect_delimiter(file_path: Path) -> str:
    """Try to detect the delimiter used in the CSV."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(8192)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def count_rows(file_path: Path) -> int:
    """
    Count lines without loading the entire file into memory.

    The result includes the header when the CSV has one.
    """
    with file_path.open("rb") as file:
        return sum(1 for _ in file)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with cleaned column names."""
    normalized = df.copy()
    normalized.columns = [
        str(column).strip().replace("\ufeff", "") for column in normalized.columns
    ]
    return normalized


def inspect_file_metadata(name: str, file_path: Path) -> None:
    """Display basic file information."""
    print_separator(f"FILE: {name}")

    print(f"Path: {portable_path(file_path)}")
    print(f"Existe: {file_path.exists()}")

    if not file_path.exists():
        return

    print(f"Size: {human_readable_size(file_path.stat().st_size)}")
    print(f"Detected delimiter: {detect_delimiter(file_path)!r}")

    line_count = count_rows(file_path)
    print(f"Physical line count: {line_count:,}")

    print("Calculando SHA-256...")
    print(f"SHA-256: {calculate_sha256(file_path)}")


def read_csv_safely(
    file_path: Path,
    *,
    nrows: int | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Read a CSV with basic delimiter detection and clear error messages."""
    delimiter = detect_delimiter(file_path)

    try:
        df = read_csv_typed(
            str(file_path),
            sep=delimiter,
            nrows=nrows,
            low_memory=low_memory,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        df = read_csv_typed(
            str(file_path),
            sep=delimiter,
            nrows=nrows,
            low_memory=low_memory,
            encoding="latin-1",
        )

    return normalize_column_names(df)


def inspect_sample(name: str, file_path: Path) -> None:
    """Load a small sample and display structure and data types."""
    if not file_path.exists():
        return

    print_separator(f"MUESTRA: {name}")

    sample = read_csv_safely(file_path, nrows=5)

    print(f"Sample dimensions: {sample.shape}")
    print(f"Number of columns: {sample.shape[1]}")

    print("\nColumn names:")
    for index, column in enumerate(sample.columns, start=1):
        print(f"{index:03d}: {column}")

    print("\nInferred types in the sample:")
    print(sample.dtypes.to_string())

    print("\nFirst rows:")
    with pd.option_context(
        "display.max_columns",
        20,
        "display.width",
        200,
        "display.max_colwidth",
        30,
    ):
        print(sample.head().to_string(index=False))


def find_candidate_column(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    """Find a column using common name variants."""
    lower_map = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def inspect_classes(file_path: Path) -> pd.DataFrame:
    """Load and fully inspect the class file."""
    print_separator("FULL CLASS ANALYSIS")

    classes = read_csv_safely(file_path)

    print(f"Dimensions: {classes.shape}")
    print(f"Complete duplicates: {classes.duplicated().sum():,}")
    print(f"Total missing values: {classes.isna().sum().sum():,}")

    class_column = find_candidate_column(
        classes.columns.tolist(),
        ("class", "label", "classes"),
    )

    transaction_id_column = find_candidate_column(
        classes.columns.tolist(),
        ("txId", "txid", "transaction_id", "transaction"),
    )

    print(f"Candidate identifier column: {transaction_id_column}")
    print(f"Candidate class column: {class_column}")

    if transaction_id_column is not None:
        print(
            "Unique identifiers:",
            f"{classes[transaction_id_column].nunique():,}",
        )
        print(
            "Duplicate identifiers:",
            f"{classes[transaction_id_column].duplicated().sum():,}",
        )

    if class_column is not None:
        counts = (
            classes[class_column].astype(str).value_counts(dropna=False).sort_index()
        )

        percentages = counts / len(classes) * 100

        summary = pd.DataFrame(
            {
                "cantidad": counts,
                "porcentaje": percentages.round(4),
            }
        )

        print("\nClass distribution:")
        print(summary.to_string())

    print("\nMissing values by column:")
    print(classes.isna().sum().to_string())

    return classes


def inspect_features(file_path: Path) -> pd.DataFrame:
    """Load and fully inspect the feature file."""
    print_separator("FULL FEATURE ANALYSIS")

    features = read_csv_safely(file_path)

    print(f"Dimensions: {features.shape}")
    print(f"Number of columns: {features.shape[1]}")
    print(f"Complete duplicates: {features.duplicated().sum():,}")
    print(f"Total missing values: {features.isna().sum().sum():,}")

    transaction_id_column = find_candidate_column(
        features.columns.tolist(),
        ("txId", "txid", "transaction_id", "transaction"),
    )

    time_column = find_candidate_column(
        features.columns.tolist(),
        ("time_step", "timestep", "time step", "time"),
    )

    print(f"Candidate identifier column: {transaction_id_column}")
    print(f"Columna candidata temporal: {time_column}")

    if transaction_id_column is not None:
        print(
            "Unique identifiers:",
            f"{features[transaction_id_column].nunique():,}",
        )
        print(
            "Duplicate identifiers:",
            f"{features[transaction_id_column].duplicated().sum():,}",
        )

    if time_column is not None:
        time_values = pd.Series(
            pd.to_numeric(
                features[time_column],
                errors="coerce",
            ),
            index=features.index,
            name=time_column,
        )

        minimum_time = time_values.min()
        maximum_time = time_values.max()
        unique_time_steps = int(time_values.nunique(dropna=True))

        minimum_time_value = int(minimum_time) if pd.notna(minimum_time) else None

        maximum_time_value = int(maximum_time) if pd.notna(maximum_time) else None

        print(f"Minimum time step: {minimum_time_value}")
        print(f"Maximum time step: {maximum_time_value}")
        print(f"Number of time steps: {unique_time_steps}")

        print("\nTransactions by time step:")

        time_step_counts = time_values.value_counts(dropna=False).sort_index()

        print(time_step_counts.to_string())

    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()

    print(f"\nNon-numeric columns: {non_numeric_columns}")

    missing = features.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values were found.")
    else:
        print("\nColumns with missing values:")
        print(missing.to_string())

    print("\nStatistical summary of the first 10 numeric columns:")
    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    selected_columns = numeric_columns[:10]

    if selected_columns:
        print(features[selected_columns].describe().transpose().to_string())

    return features


def inspect_edges(file_path: Path) -> pd.DataFrame:
    """Load and fully inspect the edge list."""
    print_separator("FULL EDGE ANALYSIS")

    edges = read_csv_safely(file_path)

    print(f"Dimensions: {edges.shape}")
    print(f"Number of columns: {edges.shape[1]}")
    print(f"Complete duplicates: {edges.duplicated().sum():,}")
    print(f"Total missing values: {edges.isna().sum().sum():,}")

    print("\nColumn names:")
    print(edges.columns.tolist())

    if edges.shape[1] >= 2:
        source_column = edges.columns[0]
        target_column = edges.columns[1]

        print(f"Assumed source column: {source_column}")
        print(f"Assumed target column: {target_column}")

        print(
            "Unique source nodes:",
            f"{edges[source_column].nunique():,}",
        )
        print(
            "Unique destination nodes:",
            f"{edges[target_column].nunique():,}",
        )

        self_loops = (
            edges[source_column].astype(str) == edges[target_column].astype(str)
        ).sum()

        print(f"Autoaristas: {self_loops:,}")

    return edges


def inspect_relationships(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    """Check basic consistency across the three files."""
    print_separator("CROSS-FILE CONSISTENCY")

    feature_id = find_candidate_column(
        features.columns.tolist(),
        ("txId", "txid", "transaction_id", "transaction"),
    )

    class_id = find_candidate_column(
        classes.columns.tolist(),
        ("txId", "txid", "transaction_id", "transaction"),
    )

    if feature_id is None or class_id is None:
        print("Automatic detection failed for identifier columns.")
        return

    feature_ids = set(features[feature_id].astype(str))
    class_ids = set(classes[class_id].astype(str))

    print(f"IDs in features: {len(feature_ids):,}")
    print(f"IDs in classes: {len(class_ids):,}")
    print(f"IDs compartidos: {len(feature_ids & class_ids):,}")
    print(
        "IDs with features but no class:",
        f"{len(feature_ids - class_ids):,}",
    )
    print(
        "IDs with class but no features:",
        f"{len(class_ids - feature_ids):,}",
    )

    if edges.shape[1] >= 2:
        source_ids = set(edges.iloc[:, 0].astype(str))
        target_ids = set(edges.iloc[:, 1].astype(str))
        edge_ids = source_ids | target_ids

        print(f"IDs present in edges: {len(edge_ids):,}")
        print(
            "Edge IDs without features:",
            f"{len(edge_ids - feature_ids):,}",
        )


def print_expected_reference() -> None:
    """Display the official counts used as reference."""
    print_separator("ELLIPTIC REFERENCE VALUES++")

    expected: dict[str, Any] = {
        "Transactions": 203_769,
        "Edges": 234_355,
        "Pasos temporales": 49,
        "Illicit": 4_545,
        "Licit": 42_019,
        "Desconocidas": 157_205,
        "Reported features": 183,
    }

    for key, value in expected.items():
        if isinstance(value, int):
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the raw Elliptic dataset and report structural and data-quality diagnostics."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    print_separator("ELLIPTIC DATASET INSPECTION++")

    print(f"Python: {sys.version}")
    print("Project root: .")
    print(f"Data directory: {portable_path(RAW_DATA_DIR)}")

    missing_files = [
        file_path for file_path in EXPECTED_FILES.values() if not file_path.exists()
    ]

    for name, file_path in EXPECTED_FILES.items():
        inspect_file_metadata(name, file_path)

    if missing_files:
        print_separator("ERROR: MISSING FILES")

        for file_path in missing_files:
            print(file_path)

        print("\nPlace the three CSV files inside data/raw and run the program again.")
        return 1

    for name, file_path in EXPECTED_FILES.items():
        inspect_sample(name, file_path)

    classes = inspect_classes(EXPECTED_FILES["classes"])
    features = inspect_features(EXPECTED_FILES["features"])
    edges = inspect_edges(EXPECTED_FILES["edges"])

    inspect_relationships(
        features=features,
        classes=classes,
        edges=edges,
    )

    print_expected_reference()

    print_separator("INSPECTION COMPLETED")
    print("All three files were processed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
