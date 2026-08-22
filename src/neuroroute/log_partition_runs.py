"""Log naive and optimized placement results as MLflow runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.neuroroute.partition_compare import (
    compare_strategies,
    evaluate_placement,
)


PARTITION_CONFIGS = [
    Path("configs/partition_naive.yaml"),
    Path("configs/partition_optimized.yaml"),
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def log_partition_experiment(
    partition_config_path: Path,
) -> str:
    config = load_yaml(partition_config_path)["partition"]
    hardware_path = Path(config["hardware_config"])
    hardware = load_yaml(hardware_path)

    result = evaluate_placement(partition_config_path)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(
        run_name=config["mlflow"]["run_name"]
    ) as run:
        mlflow.set_tags(
            {
                "stage": "placement",
                "strategy": result["strategy"],
                "hardware_model": hardware["hardware"]["name"],
                "model_type": "snn",
                "dataset": "BBBP",
            }
        )

        mlflow.log_params(
            {
                "strategy": result["strategy"],
                "max_application_cores_per_chip": config[
                    "constraints"
                ]["max_application_cores_per_chip"],
                "grid_width": hardware["hardware"]["grid"]["width"],
                "grid_height": hardware["hardware"]["grid"]["height"],
                "neurons_per_core": hardware["hardware"]["chip"][
                    "neurons_per_core"
                ],
                "routing_table_capacity": result[
                    "routing_table_capacity"
                ],
            }
        )

        mlflow.log_metrics(
            {
                "used_chips": result["used_chips"],
                "machine_vertices": result["machine_vertices"],
                "machine_projections": result["machine_projections"],
                "independent_unicast_hops": result[
                    "independent_unicast_hops"
                ],
                "multicast_tree_links": result[
                    "multicast_tree_links"
                ],
                "maximum_router_entries": result[
                    "maximum_router_entries"
                ],
                "router_overflow_events": result[
                    "router_overflow_events"
                ],
                "maximum_link_route_load": result[
                    "maximum_link_route_load"
                ],
            }
        )

        mlflow.log_dict(
            result,
            "results/placement_metrics.json",
        )
        mlflow.log_artifact(
            str(partition_config_path),
            artifact_path="configs",
        )
        mlflow.log_artifact(
            str(hardware_path),
            artifact_path="configs",
        )

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log partitioning experiments to MLflow"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    run_ids = {}

    for config_path in PARTITION_CONFIGS:
        config = load_yaml(config_path)["partition"]
        run_ids[config["name"]] = log_partition_experiment(
            config_path
        )

    comparison = compare_strategies()
    output_path = Path(
        "artifacts/partition/mlflow_partition_runs.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_ids": run_ids,
                "comparison": comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("MLflow partitioning runs complete")

    for name, run_id in run_ids.items():
        print(f"{name}: {run_id}")

    print(
        "Experiment: "
        f"{args.tracking_uri}/#/experiments"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()