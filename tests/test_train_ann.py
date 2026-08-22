"""Tests for the NeuroRoute baseline ANN."""

import json
from pathlib import Path

import torch

from src.neuroroute.train_ann import MolecularANN


def create_model() -> MolecularANN:
    """Create the configured baseline ANN."""
    return MolecularANN(
        input_size=1024,
        hidden_sizes=[128, 32],
        dropout=0.20,
    )


def test_ann_output_shape() -> None:
    """The ANN should produce one logit per molecule."""
    model = create_model()
    features = torch.zeros(8, 1024)

    logits = model(features)

    assert logits.shape == (8,)


def test_ann_probabilities_are_valid() -> None:
    """Sigmoid probabilities must be between zero and one."""
    model = create_model()
    model.eval()

    features = torch.rand(5, 1024)

    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(features)
        )

    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_saved_checkpoint_loads() -> None:
    """The best saved weights should load successfully."""
    checkpoint_path = Path(
        "artifacts/ann/best_ann.pt"
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["input_size"] == 1024
    assert checkpoint["hidden_sizes"] == [128, 32]

    model = create_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    features = torch.zeros(2, 1024)

    with torch.no_grad():
        logits = model(features)

    assert logits.shape == (2,)


def test_saved_metrics_are_valid() -> None:
    """Saved evaluation metrics should have valid values."""
    metrics_path = Path(
        "artifacts/ann/metrics.json"
    )

    assert metrics_path.exists()

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    required_metrics = {
        "best_epoch",
        "best_validation_roc_auc",
        "test_accuracy",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
    }

    assert required_metrics.issubset(metrics)

    for metric_name in required_metrics - {"best_epoch"}:
        assert 0.0 <= metrics[metric_name] <= 1.0

    assert 1 <= metrics["best_epoch"] <= 20