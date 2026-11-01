from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from artifact_utils import portable_path

read_csv_typed = cast(Callable[..., pd.DataFrame], pd.read_csv)

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
LOGS_DIR: Final[Path] = RESULTS_DIR / "logs"

FEATURES_FILE: Final[Path] = RAW_DATA_DIR / "txs_features.csv"
CLASSES_FILE: Final[Path] = RAW_DATA_DIR / "txs_classes.csv"

RANDOM_SEED: Final[int] = 42

# Initial temporal split.
TRAIN_MAX_TIME_STEP: Final[int] = 34
VALIDATION_MAX_TIME_STEP: Final[int] = 41

# A column is removed if all of its values are identical.
REMOVE_CONSTANT_COLUMNS: Final[bool] = True

# CSV files retain the original features.
# Standardized arrays are stored separately as NumPy files.
SAVE_NORMALIZED_ARRAYS: Final[bool] = True


# ============================================================================
# CONFIGURATION AND STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class DatasetSplitSummary:
    name: str
    rows: int
    licit: int
    illicit: int
    illicit_percentage: float
    minimum_time_step: int
    maximum_time_step: int


@dataclass(frozen=True)
class PreprocessingSummary:
    random_seed: int
    transaction_id_column: str
    time_column: str
    original_label_column: str
    total_feature_rows: int
    total_class_rows: int
    merged_rows: int
    labeled_rows: int
    unknown_rows_removed: int
    original_feature_count: int
    final_feature_count: int
    constant_columns_removed: list[str]
    columns_with_missing_values_removed: list[str]
    train_max_time_step: int
    validation_max_time_step: int
    train: DatasetSplitSummary
    validation: DatasetSplitSummary
    test: DatasetSplitSummary


# ============================================================================
# GENERAL UTILITIES
# ============================================================================


def print_separator(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def verify_input_files() -> None:
    missing_files = [
        path for path in (FEATURES_FILE, CLASSES_FILE) if not path.exists()
    ]

    if missing_files:
        formatted = "\n".join(str(path) for path in missing_files)

        raise FileNotFoundError(
            "The following files were not found:\n"
            f"{formatted}\n\n"
            "Place them inside data/raw."
        )


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_column_name(column: str | float) -> str:
    return str(column).strip().replace("\ufeff", "").replace(" ", "_").replace("-", "_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def find_column(
    columns: list[str],
    candidates: tuple[str, ...],
    description: str,
) -> str:
    normalized_map = {column.lower().replace("_", ""): column for column in columns}

    for candidate in candidates:
        normalized_candidate = candidate.lower().replace("_", "")

        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    raise ValueError(
        f"Could not identify the column for {description}.\n"
        f"Available columns: {columns}"
    )


def read_csv(file_path: Path) -> pd.DataFrame:
    df = read_csv_typed(
        str(file_path),
        encoding="utf-8-sig",
        low_memory=False,
    )

    return normalize_column_names(df)


def normalize_identifier(series: pd.Series) -> pd.Series:
    """
    Convert identifiers to text without changing their semantic value.

    This prevents one file from reading an ID as an integer and another as text.
    """
    return series.astype("string").str.strip()


# ============================================================================
# LOADING AND COLUMN IDENTIFICATION
# ============================================================================


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print_separator("FILE LOADING")

    print(f"Features: {portable_path(FEATURES_FILE)}")
    features = read_csv(FEATURES_FILE)

    print(f"Classes: {portable_path(CLASSES_FILE)}")
    classes = read_csv(CLASSES_FILE)

    print(f"Feature dimensions: {features.shape}")
    print(f"Class dimensions: {classes.shape}")

    return features, classes


def identify_columns(
    features: pd.DataFrame,
    classes: pd.DataFrame,
) -> tuple[str, str, str, str]:
    feature_id_column = find_column(
        features.columns.tolist(),
        (
            "txId",
            "tx_id",
            "transaction_id",
            "transaction",
            "id",
        ),
        "feature identifier",
    )

    class_id_column = find_column(
        classes.columns.tolist(),
        (
            "txId",
            "tx_id",
            "transaction_id",
            "transaction",
            "id",
        ),
        "class identifier",
    )

    time_column = find_column(
        features.columns.tolist(),
        (
            "time_step",
            "timestep",
            "time",
            "step",
        ),
        "time step",
    )

    label_column = find_column(
        classes.columns.tolist(),
        (
            "class",
            "label",
            "classes",
            "target",
        ),
        "label",
    )

    print_separator("IDENTIFIED COLUMNS")
    print(f"Feature ID: {feature_id_column}")
    print(f"Class ID: {class_id_column}")
    print(f"Time step: {time_column}")
    print(f"Original label: {label_column}")

    return (
        feature_id_column,
        class_id_column,
        time_column,
        label_column,
    )


# ============================================================================
# LABEL VALIDATION AND CONVERSION
# ============================================================================


def show_original_label_distribution(
    classes: pd.DataFrame,
    label_column: str,
) -> None:
    print_separator("ORIGINAL LABEL DISTRIBUTION")

    distribution = (
        classes[label_column].astype("string").str.strip().value_counts(dropna=False)
    )

    print(distribution.to_string())


def normalize_label_value(
    value: str | float | bool | None,
) -> int | None:
    """
    Convert original labels to:

        0 = licit
        1 = illicit
        None = desconocida
    """

    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip().lower()

    illicit_values = {
        "1",
        "illicit",
        "illegal",
        "fraud",
        "fraudulent",
        "malicious",
    }

    licit_values = {
        "2",
        "licit",
        "legal",
        "legitimate",
        "benign",
    }

    unknown_values = {
        "3",
        "unknown",
        "unclassified",
        "unlabeled",
        "unlabelled",
        "none",
        "nan",
        "",
    }

    if normalized in illicit_values:
        return 1

    if normalized in licit_values:
        return 0

    if normalized in unknown_values:
        return None

    raise ValueError(
        "An unrecognized label was found: "
        f"{value!r}. Review the output of inspect_data.py."
    )


def prepare_classes(
    classes: pd.DataFrame,
    class_id_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, int]:
    result = classes[[class_id_column, label_column]].copy()

    result[class_id_column] = normalize_identifier(result[class_id_column])

    result["label"] = result[label_column].map(normalize_label_value).astype("Int8")

    if result[class_id_column].duplicated().any():
        duplicate_count = int(result[class_id_column].duplicated().sum())

        raise ValueError(
            f"The class file contains {duplicate_count:,} duplicate identifiers."
        )

    unknown_rows = int(result["label"].isna().sum())

    result = result.dropna(subset=["label"]).copy()
    result["label"] = result["label"].astype("int8")

    result = result.rename(columns={class_id_column: "transaction_id"})

    return result[["transaction_id", "label"]], unknown_rows


# ============================================================================
# FEATURE PREPARATION
# ============================================================================


def prepare_features(
    features: pd.DataFrame,
    feature_id_column: str,
    time_column: str,
) -> pd.DataFrame:
    result = features.copy()

    result[feature_id_column] = normalize_identifier(result[feature_id_column])

    if result[feature_id_column].duplicated().any():
        duplicate_count = int(result[feature_id_column].duplicated().sum())

        raise ValueError(
            f"The feature file contains {duplicate_count:,} duplicate identifiers."
        )

    result[time_column] = pd.to_numeric(
        result[time_column],
        errors="raise",
    ).astype("int16")

    result = result.rename(
        columns={
            feature_id_column: "transaction_id",
            time_column: "time_step",
        }
    )

    feature_columns = [
        column
        for column in result.columns
        if column not in {"transaction_id", "time_step"}
    ]

    for column in feature_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def merge_labeled_data(
    features: pd.DataFrame,
    classes: pd.DataFrame,
) -> pd.DataFrame:
    print_separator("FEATURE AND LABEL MERGE")

    merged = features.merge(
        classes,
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.sort_values(
        by=["time_step", "transaction_id"],
        kind="stable",
    ).reset_index(drop=True)

    print(f"Merged labeled transactions: {len(merged):,}")

    if merged.empty:
        raise ValueError(
            "The merge produced no records. Check identifier types and contents."
        )

    return merged


def remove_unusable_columns(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_columns = [
        column
        for column in data.columns
        if column
        not in {
            "transaction_id",
            "time_step",
            "label",
        }
    ]

    missing_columns = [
        column for column in feature_columns if data[column].isna().any()
    ]

    result = data.drop(
        columns=missing_columns,
        errors="ignore",
    ).copy()

    constant_columns: list[str] = []

    if REMOVE_CONSTANT_COLUMNS:
        remaining_feature_columns = [
            column
            for column in result.columns
            if column
            not in {
                "transaction_id",
                "time_step",
                "label",
            }
        ]

        constant_columns = [
            column
            for column in remaining_feature_columns
            if result[column].nunique(dropna=False) <= 1
        ]

        result = result.drop(
            columns=constant_columns,
            errors="ignore",
        )

    return result, missing_columns, constant_columns


# ============================================================================
# TEMPORAL SPLIT
# ============================================================================


def validate_time_steps(data: pd.DataFrame) -> None:
    minimum = int(data["time_step"].min())
    maximum = int(data["time_step"].max())
    unique_count = int(data["time_step"].nunique())

    print_separator("PASOS TEMPORALES")
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")
    print(f"Number of steps: {unique_count}")

    print("\nLabeled transactions by time step:")
    print(
        data.groupby("time_step")["label"]
        .agg(
            total="size",
            illicit="sum",
        )
        .assign(
            licit=lambda frame: frame["total"] - frame["illicit"],
            illicit_percentage=lambda frame: frame["illicit"] / frame["total"] * 100,
        )
        .to_string()
    )

    if maximum <= VALIDATION_MAX_TIME_STEP:
        raise ValueError(
            "There are not enough time steps to create "
            "the test set with the current configuration."
        )


def temporal_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data[data["time_step"] <= TRAIN_MAX_TIME_STEP].copy()

    validation = data[
        data["time_step"].between(
            TRAIN_MAX_TIME_STEP + 1,
            VALIDATION_MAX_TIME_STEP,
        )
    ].copy()

    test = data[data["time_step"] > VALIDATION_MAX_TIME_STEP].copy()

    for name, split in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if split.empty:
            raise ValueError(
                f"The {name} split is empty. Check the temporal boundaries."
            )

        if split["label"].nunique() < 2:
            raise ValueError(f"The {name} does not contain both classes.")

    return train, validation, test


# ============================================================================
# SUMMARIES AND STATISTICS
# ============================================================================


def summarize_split(
    name: str,
    data: pd.DataFrame,
) -> DatasetSplitSummary:
    illicit = int(data["label"].sum())
    licit = int(len(data) - illicit)

    return DatasetSplitSummary(
        name=name,
        rows=len(data),
        licit=licit,
        illicit=illicit,
        illicit_percentage=round(
            illicit / len(data) * 100,
            6,
        ),
        minimum_time_step=int(data["time_step"].min()),
        maximum_time_step=int(data["time_step"].max()),
    )


def print_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print_separator("TEMPORAL SPLIT")

    summaries = [
        summarize_split("Training", train),
        summarize_split("Validation", validation),
        summarize_split("Test", test),
    ]

    table = pd.DataFrame([asdict(summary) for summary in summaries])

    print(table.to_string(index=False))


# ============================================================================
# STANDARDIZATION
# ============================================================================


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column
        not in {
            "transaction_id",
            "time_step",
            "label",
        }
    ]


def fit_and_save_scaler(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    print_separator("STANDARDIZATION")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(train[feature_columns]).astype(np.float32)

    x_validation = scaler.transform(validation[feature_columns]).astype(np.float32)

    x_test = scaler.transform(test[feature_columns]).astype(np.float32)

    scaler_path = PROCESSED_DATA_DIR / "standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print(f"Scaler saved: {portable_path(scaler_path)}")

    if SAVE_NORMALIZED_ARRAYS:
        np.save(
            PROCESSED_DATA_DIR / "x_train.npy",
            x_train,
        )
        np.save(
            PROCESSED_DATA_DIR / "x_validation.npy",
            x_validation,
        )
        np.save(
            PROCESSED_DATA_DIR / "x_test.npy",
            x_test,
        )

        np.save(
            PROCESSED_DATA_DIR / "y_train.npy",
            train["label"].to_numpy(dtype=np.int8),
        )
        np.save(
            PROCESSED_DATA_DIR / "y_validation.npy",
            validation["label"].to_numpy(dtype=np.int8),
        )
        np.save(
            PROCESSED_DATA_DIR / "y_test.npy",
            test["label"].to_numpy(dtype=np.int8),
        )

        np.save(
            PROCESSED_DATA_DIR / "time_train.npy",
            train["time_step"].to_numpy(dtype=np.int16),
        )
        np.save(
            PROCESSED_DATA_DIR / "time_validation.npy",
            validation["time_step"].to_numpy(dtype=np.int16),
        )
        np.save(
            PROCESSED_DATA_DIR / "time_test.npy",
            test["time_step"].to_numpy(dtype=np.int16),
        )

        print("Standardized NumPy arrays saved.")
        print(f"x_train: {x_train.shape}")
        print(f"x_validation: {x_validation.shape}")
        print(f"x_test: {x_test.shape}")


# ============================================================================
# SAVING RESULTS
# ============================================================================


def save_csv_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print_separator("SAVING DATA SPLITS")

    output_files = {
        "train.csv": train,
        "validation.csv": validation,
        "test.csv": test,
    }

    for filename, frame in output_files.items():
        output_path = PROCESSED_DATA_DIR / filename

        frame.to_csv(
            output_path,
            index=False,
        )

        print(f"{filename}: {len(frame):,} rows → {output_path}")


def save_metadata(
    feature_columns: list[str],
    summary: PreprocessingSummary,
) -> None:
    feature_path = PROCESSED_DATA_DIR / "feature_columns.json"

    with feature_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_columns,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary_path = PROCESSED_DATA_DIR / "preprocessing_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(summary),
            file,
            indent=2,
            ensure_ascii=False,
        )

    hashes = {
        FEATURES_FILE.name: calculate_sha256(FEATURES_FILE),
        CLASSES_FILE.name: calculate_sha256(CLASSES_FILE),
    }

    hashes_path = PROCESSED_DATA_DIR / "source_hashes.json"

    with hashes_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            hashes,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Columns saved: {portable_path(feature_path)}")
    print(f"Summary saved: {portable_path(summary_path)}")
    print(f"Hashes saved: {portable_path(hashes_path)}")


# ============================================================================
# FINAL VALIDATION
# ============================================================================


def validate_no_temporal_leakage(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    assert train["time_step"].max() < validation["time_step"].min()
    assert validation["time_step"].max() < test["time_step"].min()

    train_ids = set(train["transaction_id"])
    validation_ids = set(validation["transaction_id"])
    test_ids = set(test["transaction_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    print_separator("INFORMATION LEAKAGE VALIDATION")
    print("No time steps are shared across splits.")
    print("No identifiers are repeated across splits.")


# ============================================================================
# MAIN PROGRAM
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the Elliptic dataset using the chronological train/validation/test protocol."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    print_separator("ELLIPTIC PREPROCESSING++")

    print(f"Python: {sys.version}")
    print(f"Project: {portable_path(PROJECT_ROOT)}")
    print(f"Seed: {RANDOM_SEED}")

    configure_reproducibility(RANDOM_SEED)
    ensure_directories()
    verify_input_files()

    features_raw, classes_raw = load_source_data()

    (
        feature_id_column,
        class_id_column,
        time_column,
        label_column,
    ) = identify_columns(
        features=features_raw,
        classes=classes_raw,
    )

    show_original_label_distribution(
        classes=classes_raw,
        label_column=label_column,
    )

    classes_labeled, unknown_rows = prepare_classes(
        classes=classes_raw,
        class_id_column=class_id_column,
        label_column=label_column,
    )

    features = prepare_features(
        features=features_raw,
        feature_id_column=feature_id_column,
        time_column=time_column,
    )

    original_feature_count = len(get_feature_columns(features))

    data = merge_labeled_data(
        features=features,
        classes=classes_labeled,
    )

    merged_rows_before_cleaning = len(data)

    validate_time_steps(data)

    # Split before feature selection to prevent evaluation-period information
    # from influencing preprocessing decisions.
    train, validation, test = temporal_split(data)

    (
        train,
        missing_columns,
        constant_columns,
    ) = remove_unusable_columns(train)

    dropped_columns = missing_columns + constant_columns
    validation = validation.drop(columns=dropped_columns)
    test = test.drop(columns=dropped_columns)
    data = data.drop(columns=dropped_columns)

    feature_columns = get_feature_columns(train)

    if not feature_columns:
        raise ValueError("No features remained after training-only cleaning.")

    if (
        get_feature_columns(validation) != feature_columns
        or get_feature_columns(test) != feature_columns
    ):
        raise ValueError("Feature columns differ between temporal splits.")

    if validation[feature_columns].isna().any().any():
        raise ValueError("Validation contains missing selected features.")

    if test[feature_columns].isna().any().any():
        raise ValueError("Test contains missing selected features.")

    validate_no_temporal_leakage(
        train=train,
        validation=validation,
        test=test,
    )

    print_split_summary(
        train=train,
        validation=validation,
        test=test,
    )

    save_csv_splits(
        train=train,
        validation=validation,
        test=test,
    )

    fit_and_save_scaler(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
    )

    summary = PreprocessingSummary(
        random_seed=RANDOM_SEED,
        transaction_id_column=feature_id_column,
        time_column=time_column,
        original_label_column=label_column,
        total_feature_rows=len(features_raw),
        total_class_rows=len(classes_raw),
        merged_rows=merged_rows_before_cleaning,
        labeled_rows=len(data),
        unknown_rows_removed=unknown_rows,
        original_feature_count=original_feature_count,
        final_feature_count=len(feature_columns),
        constant_columns_removed=constant_columns,
        columns_with_missing_values_removed=missing_columns,
        train_max_time_step=TRAIN_MAX_TIME_STEP,
        validation_max_time_step=VALIDATION_MAX_TIME_STEP,
        train=summarize_split("train", train),
        validation=summarize_split("validation", validation),
        test=summarize_split("test", test),
    )

    save_metadata(
        feature_columns=feature_columns,
        summary=summary,
    )

    print_separator("PREPROCESSING COMPLETED")
    print(f"Final features: {len(feature_columns)}")
    print(f"Labeled transactions: {len(data):,}")
    print(f"Excluded unknown records: {unknown_rows:,}")
    print(f"Files generated in: {portable_path(PROCESSED_DATA_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
