"""Validate and simulate deployment of a packaged NeuroRoute model."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np

from src.neuroroute.validate_deployment import validate_bundle


def run_deployment(
    bundle_directory: Path,
    tracking_uri: str,
    environment: str,
) -> dict[str, Any]:
    deployment_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    validation_start = time.perf_counter()
    validation = validate_bundle(bundle_directory)
    validation_ms = (
        time.perf_counter() - validation_start
    ) * 1000.0

    model_descriptors = list(
        (bundle_directory / "model").rglob("MLmodel")
    )
    model_directory = model_descriptors[0].parent
    model = mlflow.pyfunc.load_model(str(model_directory))

    # Deterministic deployment smoke-test batch.
    batch = np.zeros((8, 1024), dtype=np.float32)

    # Warm-up inference is excluded from latency measurement.
    model.predict(batch[:1])

    inference_times_ms: list[float] = []
    predictions = None

    for _ in range(20):
        start = time.perf_counter()
        predictions = np.asarray(model.predict(batch))
        inference_times_ms.append(
            (time.perf_counter() - start) * 1000.0
        )

    if predictions is None or not np.isfinite(predictions).all():
        raise ValueError("Deployment inference validation failed")

    average_batch_latency_ms = float(
        np.mean(inference_times_ms)
    )
    p95_batch_latency_ms = float(
        np.percentile(inference_times_ms, 95)
    )
    latency_per_sample_ms = (
        average_batch_latency_ms / len(batch)
    )

    model_information = validation["registered_model"]

    result = {
        "deployment_id": deployment_id,
        "status": "succeeded",
        "environment": environment,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": model_information["name"],
        "model_alias": model_information["alias"],
        "model_version": model_information["version"],
        "source_run_id": model_information[
            "source_run_id"
        ],
        "bundle_validation_ms": validation_ms,
        "verified_files": validation["verified_files"],
        "batch_size": len(batch),
        "inference_iterations": len(inference_times_ms),
        "average_batch_latency_ms": (
            average_batch_latency_ms
        ),
        "p95_batch_latency_ms": p95_batch_latency_ms,
        "average_latency_per_sample_ms": (
            latency_per_sample_ms
        ),
        "prediction_shape": list(predictions.shape),
        "machine_vertices": validation["machine_vertices"],
        "used_chips": validation["used_chips"],
        "multicast_tree_links": validation[
            "multicast_tree_links"
        ],
        "router_overflow_events": validation[
            "router_overflow_events"
        ],
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("NeuroRoute-Deployments")

    with mlflow.start_run(
        run_name=f"deploy_{deployment_id[:8]}"
    ) as run:
        mlflow.set_tags(
            {
                "stage": "deployment",
                "deployment_id": deployment_id,
                "deployment_status": "succeeded",
                "environment": environment,
                "model_alias": model_information["alias"],
                "model_version": model_information["version"],
                "target": (
                    "SpiNNaker-style educational simulator"
                ),
            }
        )

        mlflow.log_params(
            {
                "model_name": model_information["name"],
                "model_version": model_information["version"],
                "source_run_id": model_information[
                    "source_run_id"
                ],
                "batch_size": len(batch),
                "inference_iterations": len(
                    inference_times_ms
                ),
                "machine_vertices": validation[
                    "machine_vertices"
                ],
                "used_chips": validation["used_chips"],
            }
        )

        mlflow.log_metrics(
            {
                "bundle_validation_ms": validation_ms,
                "verified_files": validation[
                    "verified_files"
                ],
                "average_batch_latency_ms": (
                    average_batch_latency_ms
                ),
                "p95_batch_latency_ms": (
                    p95_batch_latency_ms
                ),
                "average_latency_per_sample_ms": (
                    latency_per_sample_ms
                ),
                "multicast_tree_links": validation[
                    "multicast_tree_links"
                ],
                "router_overflow_events": validation[
                    "router_overflow_events"
                ],
            }
        )

        result["mlflow_run_id"] = run.info.run_id

        mlflow.log_dict(
            result,
            "deployment/deployment_event.json",
        )
        mlflow.log_artifact(
            str(bundle_directory / "manifest.json"),
            artifact_path="deployment",
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a NeuroRoute deployment simulation"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "artifacts/deployment/neuroroute_bundle"
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--environment",
        default="local-simulation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_deployment(
        bundle_directory=args.bundle,
        tracking_uri=args.tracking_uri,
        environment=args.environment,
    )

    output_path = Path(
        "artifacts/deployment/deployment_event.json"
    )
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Deployment simulation complete")
    print(f"Status: {result['status']}")
    print(f"Deployment ID: {result['deployment_id']}")
    print(f"Model version: {result['model_version']}")
    print(f"MLflow run ID: {result['mlflow_run_id']}")
    print(
        f"Average latency/sample: "
        f"{result['average_latency_per_sample_ms']:.4f} ms"
    )
    print(
        f"P95 batch latency: "
        f"{result['p95_batch_latency_ms']:.4f} ms"
    )
    print(
        f"Routing overflow events: "
        f"{result['router_overflow_events']}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()