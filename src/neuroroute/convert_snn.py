"""Convert the trained ANN into an evaluated spiking network."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ConvertedSNN(nn.Module):
    """Weight-transferred SNN with LIF hidden neurons."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        beta: float,
        threshold: float,
    ) -> None:
        super().__init__()

        first_hidden, second_hidden = hidden_sizes

        self.fc1 = nn.Linear(
            input_size,
            first_hidden,
        )
        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
        )

        self.fc2 = nn.Linear(
            first_hidden,
            second_hidden,
        )
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
        )

        self.fc3 = nn.Linear(
            second_hidden,
            1,
        )

    def forward(
        self,
        features: torch.Tensor,
        num_steps: int,
        input_rate_scale: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run rate-coded SNN inference."""
        membrane1 = self.lif1.init_leaky()
        membrane2 = self.lif2.init_leaky()

        output_current_sum = torch.zeros(
            features.shape[0],
            device=features.device,
        )

        hidden1_spikes = 0.0
        hidden2_spikes = 0.0
        hidden1_elements = 0
        hidden2_elements = 0

        spike_probability = torch.clamp(
            features * input_rate_scale,
            min=0.0,
            max=1.0,
        )

        for _ in range(num_steps):
            input_spikes = torch.bernoulli(
                spike_probability,
                generator=generator,
            )

            current1 = self.fc1(input_spikes)

            spikes1, membrane1 = self.lif1(
                current1,
                membrane1,
            )

            current2 = self.fc2(spikes1)

            spikes2, membrane2 = self.lif2(
                current2,
                membrane2,
            )

            output_current = self.fc3(
                spikes2
            ).squeeze(1)

            output_current_sum += output_current

            hidden1_spikes += float(
                spikes1.sum().item()
            )
            hidden2_spikes += float(
                spikes2.sum().item()
            )

            hidden1_elements += spikes1.numel()
            hidden2_elements += spikes2.numel()

        mean_output_current = (
            output_current_sum / num_steps
        )

        spike_statistics = {
            "hidden1_spike_rate": (
                hidden1_spikes / hidden1_elements
            ),
            "hidden2_spike_rate": (
                hidden2_spikes / hidden2_elements
            ),
        }

        return mean_output_current, spike_statistics


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one processed dataset split."""
    data = np.load(path)

    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
    )


def transfer_ann_weights(
    snn_model: ConvertedSNN,
    checkpoint: dict[str, Any],
) -> None:
    """Copy ANN linear-layer weights into the SNN."""
    state = checkpoint["model_state_dict"]

    with torch.no_grad():
        snn_model.fc1.weight.copy_(
            state["network.0.weight"]
        )
        snn_model.fc1.bias.copy_(
            state["network.0.bias"]
        )

        snn_model.fc2.weight.copy_(
            state["network.3.weight"]
        )
        snn_model.fc2.bias.copy_(
            state["network.3.bias"]
        )

        snn_model.fc3.weight.copy_(
            state["network.6.weight"]
        )
        snn_model.fc3.bias.copy_(
            state["network.6.bias"]
        )


def evaluate_snn(
    model: ConvertedSNN,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_steps: int,
    input_rate_scale: float,
    decision_threshold: float,
    seed: int,
) -> dict[str, float]:
    """Evaluate predictive quality, spikes, and latency."""
    model.eval()

    all_probabilities: list[float] = []

    hidden1_spike_rates: list[float] = []
    hidden2_spike_rates: list[float] = []

    generator = torch.Generator()
    generator.manual_seed(seed)

    start_time = time.perf_counter()

    with torch.no_grad():
        for start in range(
            0,
            len(features),
            batch_size,
        ):
            end = start + batch_size

            batch = torch.from_numpy(
                features[start:end]
            )

            logits, spike_statistics = model(
                batch,
                num_steps=num_steps,
                input_rate_scale=input_rate_scale,
                generator=generator,
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.extend(
                probabilities.numpy().tolist()
            )

            hidden1_spike_rates.append(
                spike_statistics[
                    "hidden1_spike_rate"
                ]
            )

            hidden2_spike_rates.append(
                spike_statistics[
                    "hidden2_spike_rate"
                ]
            )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    probability_array = np.asarray(
        all_probabilities
    )

    predictions = (
        probability_array >= decision_threshold
    ).astype(np.int64)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probability_array,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "hidden1_spike_rate": float(
            np.mean(hidden1_spike_rates)
        ),
        "hidden2_spike_rate": float(
            np.mean(hidden2_spike_rates)
        ),
        "latency_ms_per_sample": float(
            elapsed_seconds
            * 1000.0
            / len(features)
        ),
    }


def convert_and_evaluate(
    config_path: str | Path,
) -> dict[str, float]:
    """Load ANN weights, create SNN, evaluate, and track."""
    config = load_config(config_path)

    seed = int(config["project"]["seed"])
    num_steps = int(config["snn"]["num_steps"])
    beta = float(config["snn"]["beta"])
    threshold = float(
        config["snn"]["threshold"]
    )
    input_rate_scale = float(
        config["snn"]["input_rate_scale"]
    )
    decision_threshold = float(
        config["snn"]["decision_threshold"]
    )
    batch_size = int(
        config["evaluation"]["batch_size"]
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoint_path = Path(
        config["source_ann"]["checkpoint_path"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    input_size = int(checkpoint["input_size"])
    hidden_sizes = [
        int(value)
        for value in checkpoint["hidden_sizes"]
    ]

    model = ConvertedSNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        beta=beta,
        threshold=threshold,
    )

    transfer_ann_weights(model, checkpoint)

    X_validation, y_validation = load_data(
        config["data"]["validation_path"]
    )
    X_test, y_test = load_data(
        config["data"]["test_path"]
    )

    validation_metrics = evaluate_snn(
        model,
        X_validation,
        y_validation,
        batch_size,
        num_steps,
        input_rate_scale,
        decision_threshold,
        seed,
    )

    test_metrics = evaluate_snn(
        model,
        X_test,
        y_test,
        batch_size,
        num_steps,
        input_rate_scale,
        decision_threshold,
        seed + 1,
    )

    output_dir = Path("artifacts/snn")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_output = (
        output_dir / "converted_snn.pt"
    )
    metrics_output = (
        output_dir / "metrics.json"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "beta": beta,
            "threshold": threshold,
            "num_steps": num_steps,
            "input_rate_scale": input_rate_scale,
        },
        checkpoint_output,
    )

    combined_metrics = {
        **{
            f"validation_{name}": value
            for name, value
            in validation_metrics.items()
        },
        **{
            f"test_{name}": value
            for name, value in test_metrics.items()
        },
    }

    with metrics_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_metrics,
            file,
            indent=2,
        )

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="bbbp_converted_snn"
    ) as run:
        mlflow.set_tags(
            {
                "stage": "snn_evaluation",
                "model_type": "snn",
                "dataset": "BBBP",
                "conversion": "ann_weight_transfer",
                "parent_ann_run_id": config[
                    "source_ann"
                ]["mlflow_run_id"],
            }
        )

        mlflow.log_params(
            {
                "seed": seed,
                "input_size": input_size,
                "hidden_size_1": hidden_sizes[0],
                "hidden_size_2": hidden_sizes[1],
                "num_steps": num_steps,
                "beta": beta,
                "threshold": threshold,
                "input_rate_scale": (
                    input_rate_scale
                ),
                "decision_threshold": (
                    decision_threshold
                ),
                "batch_size": batch_size,
                "source_ann_run_id": config[
                    "source_ann"
                ]["mlflow_run_id"],
            }
        )

        mlflow.log_metrics(combined_metrics)

        mlflow.log_artifact(
            str(checkpoint_output),
            artifact_path="checkpoint",
        )

        mlflow.log_artifact(
            str(metrics_output),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(config_path),
            artifact_path="configuration",
        )

        print(f"MLflow run ID: {run.info.run_id}")

    print(json.dumps(combined_metrics, indent=2))

    return combined_metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Convert ANN weights to an SNN."
    )

    parser.add_argument(
        "--config",
        default="configs/snn.yaml",
    )

    arguments = parser.parse_args()

    convert_and_evaluate(arguments.config)


if __name__ == "__main__":
    main()