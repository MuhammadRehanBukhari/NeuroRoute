"""Tests for ANN-to-SNN conversion."""

import json
from pathlib import Path

import torch

from src.neuroroute.convert_snn import (
    ConvertedSNN,
    transfer_ann_weights,
)


def create_snn() -> ConvertedSNN:
    """Create the configured converted SNN."""
    return ConvertedSNN(
        input_size=1024,
        hidden_sizes=[128, 32],
        beta=0.90,
        threshold=1.0,
    )


def test_snn_forward_shape() -> None:
    """The SNN should produce one score per molecule."""
    model = create_snn()
    model.eval()

    features = torch.zeros(4, 1024)

    generator = torch.Generator()
    generator.manual_seed(42)

    with torch.no_grad():
        logits, statistics = model(
            features,
            num_steps=5,
            input_rate_scale=0.20,
            generator=generator,
        )

    assert logits.shape == (4,)

    assert 0.0 <= (
        statistics["hidden1_spike_rate"]
    ) <= 1.0

    assert 0.0 <= (
        statistics["hidden2_spike_rate"]
    ) <= 1.0


def test_ann_weights_are_transferred() -> None:
    """Transferred SNN weights should equal ANN weights."""
    checkpoint = torch.load(
        "artifacts/ann/best_ann.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = create_snn()

    transfer_ann_weights(
        model,
        checkpoint,
    )

    ann_state = checkpoint["model_state_dict"]

    assert torch.equal(
        model.fc1.weight,
        ann_state["network.0.weight"],
    )

    assert torch.equal(
        model.fc2.weight,
        ann_state["network.3.weight"],
    )

    assert torch.equal(
        model.fc3.weight,
        ann_state["network.6.weight"],
    )


def test_converted_checkpoint_exists() -> None:
    """The converted SNN artifact should be loadable."""
    checkpoint_path = Path(
        "artifacts/snn/converted_snn.pt"
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["input_size"] == 1024
    assert checkpoint["hidden_sizes"] == [128, 32]
    assert checkpoint["num_steps"] == 25
    assert checkpoint["beta"] == 0.90


def test_snn_metrics_are_valid() -> None:
    """Saved SNN metrics should use valid ranges."""
    metrics_path = Path(
        "artifacts/snn/metrics.json"
    )

    assert metrics_path.exists()

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    bounded_metrics = [
        "validation_accuracy",
        "validation_roc_auc",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_hidden1_spike_rate",
        "validation_hidden2_spike_rate",
        "test_accuracy",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_hidden1_spike_rate",
        "test_hidden2_spike_rate",
    ]

    for metric_name in bounded_metrics:
        assert 0.0 <= metrics[metric_name] <= 1.0

    assert metrics["test_latency_ms_per_sample"] >= 0