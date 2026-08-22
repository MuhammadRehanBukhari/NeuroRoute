"""Prepare the BBBP molecular dataset for NeuroRoute."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project settings from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_sha256(file_path: Path) -> str:
    """Calculate a reproducibility hash for a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def smiles_to_fingerprint(
    smiles: str,
    generator,
) -> np.ndarray | None:
    """Convert one SMILES string into a Morgan fingerprint."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    fingerprint = generator.GetFingerprintAsNumPy(molecule)
    return fingerprint.astype(np.float32)


def save_split(
    output_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    smiles: np.ndarray,
) -> None:
    """Save one dataset split as a compressed NumPy file."""
    np.savez_compressed(
        output_path,
        X=features,
        y=labels,
        smiles=smiles,
    )


def prepare_data(config_path: str | Path) -> dict[str, Any]:
    """Validate, fingerprint, split, and save the BBBP dataset."""
    config = load_config(config_path)

    seed = int(config["project"]["seed"])

    raw_path = Path(config["data"]["raw_path"])
    processed_dir = Path(config["data"]["processed_dir"])
    smiles_column = config["data"]["smiles_column"]
    label_column = config["data"]["label_column"]

    radius = int(config["fingerprint"]["radius"])
    n_bits = int(config["fingerprint"]["n_bits"])

    train_fraction = float(config["split"]["train_fraction"])
    validation_fraction = float(
        config["split"]["validation_fraction"]
    )
    test_fraction = float(config["split"]["test_fraction"])

    split_total = (
        train_fraction
        + validation_fraction
        + test_fraction
    )

    if not np.isclose(split_total, 1.0):
        raise ValueError(
            "Train, validation, and test fractions must sum to 1."
        )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {raw_path}"
        )

    dataframe = pd.read_csv(raw_path)
    original_rows = len(dataframe)

    required_columns = {smiles_column, label_column}
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe[
        [smiles_column, label_column]
    ].copy()

    dataframe = dataframe.dropna(
        subset=[smiles_column, label_column]
    )

    dataframe[smiles_column] = (
        dataframe[smiles_column]
        .astype(str)
        .str.strip()
    )

    dataframe[label_column] = pd.to_numeric(
        dataframe[label_column],
        errors="coerce",
    )

    dataframe = dataframe.dropna(subset=[label_column])

    dataframe = dataframe[
        dataframe[label_column].isin([0, 1])
    ]

    dataframe[label_column] = dataframe[
        label_column
    ].astype(np.int64)

    # Remove molecules that appear with conflicting labels.
    label_counts = dataframe.groupby(smiles_column)[
        label_column
    ].nunique()

    conflicting_smiles = set(
        label_counts[label_counts > 1].index
    )

    conflicting_count = len(conflicting_smiles)

    if conflicting_smiles:
        dataframe = dataframe[
            ~dataframe[smiles_column].isin(
                conflicting_smiles
            )
        ]

    # Remove repeated molecular strings.
    rows_before_duplicates = len(dataframe)

    dataframe = dataframe.drop_duplicates(
        subset=[smiles_column]
    )

    duplicate_count = (
        rows_before_duplicates - len(dataframe)
    )

    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )
    )

    features: list[np.ndarray] = []
    labels: list[int] = []
    valid_smiles: list[str] = []

    invalid_count = 0

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    for smiles, label in dataframe[
        [smiles_column, label_column]
    ].itertuples(index=False, name=None):

        fingerprint = smiles_to_fingerprint(
            str(smiles),
            generator,
        )

        if fingerprint is None:
            invalid_count += 1
            continue

        features.append(fingerprint)
        labels.append(int(label))
        valid_smiles.append(str(smiles))

    if not features:
        raise ValueError(
            "No valid molecules remained after validation."
        )

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    smiles_array = np.asarray(valid_smiles)

    temporary_fraction = (
        validation_fraction + test_fraction
    )

    (
        X_train,
        X_temporary,
        y_train,
        y_temporary,
        smiles_train,
        smiles_temporary,
    ) = train_test_split(
        X,
        y,
        smiles_array,
        test_size=temporary_fraction,
        random_state=seed,
        stratify=y,
    )

    relative_test_fraction = (
        test_fraction / temporary_fraction
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
        smiles_validation,
        smiles_test,
    ) = train_test_split(
        X_temporary,
        y_temporary,
        smiles_temporary,
        test_size=relative_test_fraction,
        random_state=seed,
        stratify=y_temporary,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_split(
        processed_dir / "train.npz",
        X_train,
        y_train,
        smiles_train,
    )

    save_split(
        processed_dir / "validation.npz",
        X_validation,
        y_validation,
        smiles_validation,
    )

    save_split(
        processed_dir / "test.npz",
        X_test,
        y_test,
        smiles_test,
    )

    unique_labels, label_counts = np.unique(
        y,
        return_counts=True,
    )

    metadata = {
        "dataset": "BBBP",
        "raw_path": str(raw_path),
        "raw_sha256": calculate_sha256(raw_path),
        "original_rows": original_rows,
        "valid_rows": int(len(X)),
        "invalid_molecules": invalid_count,
        "removed_duplicates": duplicate_count,
        "removed_conflicting_smiles": conflicting_count,
        "fingerprint_radius": radius,
        "fingerprint_bits": n_bits,
        "seed": seed,
        "feature_shape": list(X.shape),
        "class_counts": {
            str(label): int(count)
            for label, count in zip(
                unique_labels,
                label_counts,
            )
        },
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
    }

    metadata_path = (
        processed_dir / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))

    return metadata


def main() -> None:
    """Run preprocessing from the command line."""
    parser = argparse.ArgumentParser(
        description="Prepare the BBBP dataset."
    )

    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file.",
    )

    arguments = parser.parse_args()
    prepare_data(arguments.config)


if __name__ == "__main__":
    main()
