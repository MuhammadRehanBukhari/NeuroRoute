"""Train and track the NeuroRoute baseline ANN."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
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
from torch.utils.data import DataLoader, TensorDataset


class MolecularANN(nn.Module):
    """Small feed-forward network for BBBP classification."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        first_hidden, second_hidden = hidden_sizes

        self.network = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one logit for each molecule."""
        return self.network(features).squeeze(1)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    """Make training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def load_split(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from one processed split."""
    data = np.load(path)

    features = data["X"].astype(np.float32)
    labels = data["y"].astype(np.float32)

    return features, labels


def create_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model loss and classification metrics."""
    model.eval()

    losses: list[float] = []
    all_labels: list[float] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = loss_function(logits, labels)

            probabilities = torch.sigmoid(logits)

            losses.append(loss.item())
            all_labels.extend(
                labels.cpu().numpy().tolist()
            )
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    labels_array = np.asarray(all_labels)
    probabilities_array = np.asarray(
        all_probabilities
    )

    predictions = (
        probabilities_array >= threshold
    ).astype(np.int64)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(
            accuracy_score(
                labels_array,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels_array,
                probabilities_array,
            )
        ),
        "precision": float(
            precision_score(
                labels_array,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels_array,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels_array,
                predictions,
                zero_division=0,
            )
        ),
    }


def create_training_plot(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save loss and ROC-AUC training curves."""
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )
    axes[0].plot(
        epochs,
        history["validation_loss"],
        label="Validation loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("ANN loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["validation_roc_auc"],
        label="Validation ROC-AUC",
        color="darkgreen",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation quality")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train(config_path: str | Path) -> dict[str, float]:
    """Train, evaluate, save, and track the ANN."""
    config = load_config(config_path)

    seed = int(config["project"]["seed"])
    input_size = int(config["model"]["input_size"])
    hidden_sizes = [
        int(value)
        for value in config["model"]["hidden_sizes"]
    ]
    dropout = float(config["model"]["dropout"])

    epochs = int(config["training"]["epochs"])
    batch_size = int(
        config["training"]["batch_size"]
    )
    learning_rate = float(
        config["training"]["learning_rate"]
    )
    weight_decay = float(
        config["training"]["weight_decay"]
    )
    threshold = float(
        config["training"]["decision_threshold"]
    )

    set_seed(seed)

    device = torch.device("cpu")
    print(f"Device: {device}")

    X_train, y_train = load_split(
        "data/processed/train.npz"
    )
    X_validation, y_validation = load_split(
        "data/processed/validation.npz"
    )
    X_test, y_test = load_split(
        "data/processed/test.npz"
    )

    if X_train.shape[1] != input_size:
        raise ValueError(
            "Configured input size does not match fingerprints."
        )

    train_loader = create_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = create_loader(
        X_validation,
        y_validation,
        batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = create_loader(
        X_test,
        y_test,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    model = MolecularANN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    negative_count = float(
        np.sum(y_train == 0)
    )
    positive_count = float(
        np.sum(y_train == 1)
    )

    positive_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_dir = Path("artifacts/ann")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "best_ann.pt"
    metrics_path = output_dir / "metrics.json"
    plot_path = output_dir / "training_curves.png"

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_roc_auc": [],
    }

    best_validation_auc = float("-inf")
    best_epoch = 0

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="bbbp_baseline_ann"
    ) as run:
        mlflow.set_tags(
            {
                "stage": "training",
                "model_type": "ann",
                "dataset": "BBBP",
                "execution_target": "cpu",
            }
        )

        mlflow.log_params(
            {
                "seed": seed,
                "input_size": input_size,
                "hidden_size_1": hidden_sizes[0],
                "hidden_size_2": hidden_sizes[1],
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "decision_threshold": threshold,
                "fingerprint_radius": config[
                    "fingerprint"
                ]["radius"],
                "fingerprint_bits": config[
                    "fingerprint"
                ]["n_bits"],
                "train_samples": len(X_train),
                "validation_samples": len(
                    X_validation
                ),
                "test_samples": len(X_test),
            }
        )

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses: list[float] = []

            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(features)
                loss = loss_function(logits, labels)

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            train_loss = float(
                np.mean(batch_losses)
            )

            validation_metrics = evaluate(
                model,
                validation_loader,
                loss_function,
                threshold,
                device,
            )

            history["train_loss"].append(
                train_loss
            )
            history["validation_loss"].append(
                validation_metrics["loss"]
            )
            history[
                "validation_roc_auc"
            ].append(
                validation_metrics["roc_auc"]
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "validation_loss": (
                        validation_metrics["loss"]
                    ),
                    "validation_accuracy": (
                        validation_metrics["accuracy"]
                    ),
                    "validation_roc_auc": (
                        validation_metrics["roc_auc"]
                    ),
                    "validation_f1": (
                        validation_metrics["f1"]
                    ),
                },
                step=epoch,
            )

            if (
                validation_metrics["roc_auc"]
                > best_validation_auc
            ):
                best_validation_auc = (
                    validation_metrics["roc_auc"]
                )
                best_epoch = epoch

                torch.save(
                    {
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "input_size": input_size,
                        "hidden_sizes": hidden_sizes,
                        "dropout": dropout,
                        "seed": seed,
                    },
                    weights_path,
                )

            print(
                f"Epoch {epoch:02d}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={validation_metrics['loss']:.4f} "
                f"val_auc={validation_metrics['roc_auc']:.4f}"
            )

        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_metrics = evaluate(
            model,
            test_loader,
            loss_function,
            threshold,
            device,
        )

        final_metrics = {
            "best_epoch": best_epoch,
            "best_validation_roc_auc": (
                best_validation_auc
            ),
            "test_loss": test_metrics["loss"],
            "test_accuracy": (
                test_metrics["accuracy"]
            ),
            "test_roc_auc": (
                test_metrics["roc_auc"]
            ),
            "test_precision": (
                test_metrics["precision"]
            ),
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        }

        mlflow.log_metrics(final_metrics)

        create_training_plot(
            history,
            plot_path,
        )

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                final_metrics,
                file,
                indent=2,
            )

        mlflow.log_artifact(
            str(weights_path),
            artifact_path="checkpoints",
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(plot_path),
            artifact_path="plots",
        )

        mlflow.log_artifact(
            str(config_path),
            artifact_path="configuration",
        )

        mlflow.pytorch.log_model(
            model,
            name="ann_model",
            input_example=X_train[:5],
        )

        print()
        print(f"MLflow run ID: {run.info.run_id}")
        print(
            json.dumps(
                final_metrics,
                indent=2,
            )
        )

    return final_metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train the NeuroRoute ANN."
    )

    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
    )

    arguments = parser.parse_args()
    train(arguments.config)


if __name__ == "__main__":
    main()