"""Tests for the NeuroRoute molecular data pipeline."""

import json
from pathlib import Path

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator

from src.neuroroute.data_prep import (
    smiles_to_fingerprint,
)


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def test_valid_smiles_produces_fingerprint() -> None:
    """A valid molecule should produce 1,024 binary features."""
    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=1024,
        )
    )

    fingerprint = smiles_to_fingerprint(
        "CCO",
        generator,
    )

    assert fingerprint is not None
    assert fingerprint.shape == (1024,)
    assert fingerprint.dtype == np.float32
    assert set(np.unique(fingerprint)).issubset(
        {0.0, 1.0}
    )


def test_invalid_smiles_returns_none() -> None:
    """An invalid molecular string should be rejected."""
    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=1024,
        )
    )

    fingerprint = smiles_to_fingerprint(
        "not-a-valid-smiles",
        generator,
    )

    assert fingerprint is None


def test_processed_split_shapes() -> None:
    """Feature and label counts must match in every split."""
    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_path = Path(
            f"data/processed/{split_name}.npz"
        )

        assert split_path.exists()

        data = np.load(split_path)

        assert data["X"].ndim == 2
        assert data["X"].shape[1] == 1024
        assert len(data["X"]) == len(data["y"])
        assert len(data["X"]) == len(data["smiles"])

        assert data["X"].dtype == np.float32
        assert data["y"].dtype == np.int64

        assert set(np.unique(data["X"])).issubset(
            {0.0, 1.0}
        )

        assert set(np.unique(data["y"])).issubset(
            {0, 1}
        )


def test_splits_do_not_overlap() -> None:
    """No molecular string should occur in multiple splits."""
    train = np.load("data/processed/train.npz")
    validation = np.load(
        "data/processed/validation.npz"
    )
    test = np.load("data/processed/test.npz")

    train_smiles = set(train["smiles"].tolist())
    validation_smiles = set(
        validation["smiles"].tolist()
    )
    test_smiles = set(test["smiles"].tolist())

    assert train_smiles.isdisjoint(
        validation_smiles
    )

    assert train_smiles.isdisjoint(test_smiles)

    assert validation_smiles.isdisjoint(
        test_smiles
    )


def test_metadata_matches_saved_splits() -> None:
    """Metadata counts should match the generated arrays."""
    metadata_path = Path(
        "data/processed/metadata.json"
    )

    assert metadata_path.exists()

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    train = np.load("data/processed/train.npz")
    validation = np.load(
        "data/processed/validation.npz"
    )
    test = np.load("data/processed/test.npz")

    assert metadata["fingerprint_bits"] == 1024
    assert metadata["invalid_molecules"] == 11

    assert metadata["split_sizes"]["train"] == len(
        train["X"]
    )

    assert metadata["split_sizes"][
        "validation"
    ] == len(validation["X"])

    assert metadata["split_sizes"]["test"] == len(
        test["X"]
    )

    total_saved = (
        len(train["X"])
        + len(validation["X"])
        + len(test["X"])
    )

    assert total_saved == metadata["valid_rows"]