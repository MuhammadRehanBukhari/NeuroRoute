"""Optuna search for SNN partitioning and placement parameters."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml

from src.neuroroute.partition_compare import evaluate_placement


BASE_HARDWARE_CONFIG = Path("configs/hardware.yaml")
BASE_PARTITION_CONFIG = Path("configs/partition_optimized.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def deployment_score(metrics: dict[str, Any]) -> float:
    """Combine deployment metrics into one minimization objective."""

    return (
        metrics["multicast_tree_links"]
        + 2.0 * metrics["maximum_router_entries"]
        + 0.25 * metrics["maximum_link_route_load"]
        + 0.5 * metrics["used_chips"]
        + 10_000.0 * metrics["router_overflow_events"]
    )


def make_trial_configs(
    directory: Path,
    neurons_per_core: int,
    max_cores_per_chip: int,
) -> Path:
    hardware = deepcopy(load_yaml(BASE_HARDWARE_CONFIG))
    partition = deepcopy(load_yaml(BASE_PARTITION_CONFIG))

    hardware["hardware"]["chip"][
        "neurons_per_core"
    ] = neurons_per_core

    hardware_path = directory / "hardware.yaml"
    partition_path = directory / "partition.yaml"

    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )

    partition["partition"]["name"] = "optuna_trial"
    partition["partition"]["hardware_config"] = str(
        hardware_path.resolve()
    )
    partition["partition"]["constraints"][
        "max_application_cores_per_chip"
    ] = max_cores_per_chip

    partition_path.write_text(
        yaml.safe_dump(partition, sort_keys=False),
        encoding="utf-8",
    )

    return partition_path


def run_optimization(
    tracking_uri: str,
    number_of_trials: int,
) -> dict[str, Any]:
    base_hardware = load_yaml(BASE_HARDWARE_CONFIG)
    populations = base_hardware["application"]["populations"]
    grid = base_hardware["hardware"]["grid"]
    chip_count = grid["width"] * grid["height"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("NeuroRoute-HPO")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="neuroroute_partition_hpo",
    )

    with mlflow.start_run(
        run_name="partition_hpo_parent"
    ) as parent_run:
        mlflow.set_tags(
            {
                "stage": "hyperparameter_optimization",
                "model_type": "snn",
                "optimizer": "optuna_tpe",
            }
        )
        mlflow.log_param("number_of_trials", number_of_trials)
        mlflow.log_param("objective", "weighted_deployment_cost")

        def objective(trial: optuna.Trial) -> float:
            neurons_per_core = trial.suggest_categorical(
                "neurons_per_core",
                [16, 24, 32, 48, 64],
            )
            max_cores_per_chip = trial.suggest_int(
                "max_application_cores_per_chip",
                2,
                8,
            )

            machine_vertices = sum(
                math.ceil(count / neurons_per_core)
                for count in populations.values()
            )
            required_cores_per_chip = math.ceil(
                machine_vertices / chip_count
            )

            if max_cores_per_chip < required_cores_per_chip:
                raise optuna.TrialPruned(
                    "Insufficient allocated cores"
                )

            with tempfile.TemporaryDirectory() as temporary:
                partition_path = make_trial_configs(
                    directory=Path(temporary),
                    neurons_per_core=neurons_per_core,
                    max_cores_per_chip=max_cores_per_chip,
                )

                metrics = evaluate_placement(partition_path)

            score = deployment_score(metrics)

            trial.set_user_attr("metrics", metrics)

            with mlflow.start_run(
                run_name=f"trial_{trial.number:03d}",
                nested=True,
            ):
                mlflow.set_tags(
                    {
                        "stage": "hpo_trial",
                        "strategy": "connectivity_aware",
                        "trial_number": str(trial.number),
                    }
                )

                mlflow.log_params(
                    {
                        "neurons_per_core": neurons_per_core,
                        "max_application_cores_per_chip": (
                            max_cores_per_chip
                        ),
                    }
                )

                mlflow.log_metrics(
                    {
                        "objective_score": score,
                        "used_chips": metrics["used_chips"],
                        "machine_vertices": metrics[
                            "machine_vertices"
                        ],
                        "independent_unicast_hops": metrics[
                            "independent_unicast_hops"
                        ],
                        "multicast_tree_links": metrics[
                            "multicast_tree_links"
                        ],
                        "maximum_router_entries": metrics[
                            "maximum_router_entries"
                        ],
                        "maximum_link_route_load": metrics[
                            "maximum_link_route_load"
                        ],
                        "router_overflow_events": metrics[
                            "router_overflow_events"
                        ],
                    }
                )

                mlflow.log_dict(
                    metrics,
                    "results/trial_metrics.json",
                )

            return score

        study.optimize(
            objective,
            n_trials=number_of_trials,
        )

        best_metrics = study.best_trial.user_attrs["metrics"]

        summary = {
            "parent_run_id": parent_run.info.run_id,
            "completed_trials": len(
                [
                    trial
                    for trial in study.trials
                    if trial.state
                    == optuna.trial.TrialState.COMPLETE
                ]
            ),
            "pruned_trials": len(
                [
                    trial
                    for trial in study.trials
                    if trial.state
                    == optuna.trial.TrialState.PRUNED
                ]
            ),
            "best_trial_number": study.best_trial.number,
            "best_objective_score": study.best_value,
            "best_parameters": study.best_params,
            "best_metrics": best_metrics,
        }

        mlflow.log_metrics(
            {
                "best_objective_score": study.best_value,
                "best_multicast_tree_links": best_metrics[
                    "multicast_tree_links"
                ],
                "best_maximum_router_entries": best_metrics[
                    "maximum_router_entries"
                ],
                "best_used_chips": best_metrics["used_chips"],
            }
        )

        mlflow.log_params(
            {
                f"best_{name}": value
                for name, value in study.best_params.items()
            }
        )

        mlflow.log_dict(
            summary,
            "results/hpo_summary.json",
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize NeuroRoute placement parameters"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
    )
    args = parser.parse_args()

    result = run_optimization(
        tracking_uri=args.tracking_uri,
        number_of_trials=args.trials,
    )

    output_path = Path(
        "artifacts/partition/hpo_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Partition hyperparameter optimization complete")
    print(f"Completed trials: {result['completed_trials']}")
    print(f"Pruned trials: {result['pruned_trials']}")
    print(f"Best trial: {result['best_trial_number']}")
    print(
        f"Best objective score: "
        f"{result['best_objective_score']:.4f}"
    )
    print(f"Best parameters: {result['best_parameters']}")
    print(f"Best metrics: {result['best_metrics']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()