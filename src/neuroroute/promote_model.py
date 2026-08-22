"""Register an MLflow model and promote it after quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


DEFAULT_ANN_RUN_ID = "8037001c9f164e2c91f8148447c1abb8"


def promote_model(
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
    model_name: str,
    minimum_roc_auc: float,
    minimum_accuracy: float,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    run = client.get_run(run_id)
    metrics = run.data.metrics

    test_roc_auc = metrics.get("test_roc_auc")
    test_accuracy = metrics.get("test_accuracy")

    if test_roc_auc is None or test_accuracy is None:
        raise ValueError(
            "Source run does not contain required test metrics"
        )

    checks = {
        "test_roc_auc": {
            "value": test_roc_auc,
            "minimum": minimum_roc_auc,
            "passed": test_roc_auc >= minimum_roc_auc,
        },
        "test_accuracy": {
            "value": test_accuracy,
            "minimum": minimum_accuracy,
            "passed": test_accuracy >= minimum_accuracy,
        },
    }

    all_checks_passed = all(
        check["passed"] for check in checks.values()
    )

    if not all_checks_passed:
        raise ValueError(
            f"Promotion checks failed: {checks}"
        )

    model_uri = f"runs:/{run_id}/{artifact_path}"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        await_registration_for=60,
    )

    version = str(model_version.version)

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="validation_status",
        value="approved",
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="dataset",
        value="BBBP",
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="model_type",
        value="ANN",
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="source_run_id",
        value=run_id,
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="test_roc_auc",
        value=str(test_roc_auc),
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="test_accuracy",
        value=str(test_accuracy),
    )

    # Candidate means this version passed automated checks.
    client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=version,
    )

    # For this project, passing the declared gates promotes it to the
    # deployment-facing champion alias.
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            "BBBP ANN baseline approved by automated quality gates. "
            f"Source run: {run_id}. "
            f"Test ROC-AUC: {test_roc_auc:.6f}. "
            f"Test accuracy: {test_accuracy:.6f}."
        ),
    )

    champion = client.get_model_version_by_alias(
        name=model_name,
        alias="champion",
    )

    return {
        "model_name": model_name,
        "version": version,
        "source_run_id": run_id,
        "source_model_uri": model_uri,
        "champion_model_uri": f"models:/{model_name}@champion",
        "candidate_alias_version": version,
        "champion_alias_version": str(champion.version),
        "checks": checks,
        "promotion_status": "approved",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the NeuroRoute ANN model"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_ANN_RUN_ID,
    )
    parser.add_argument(
        "--artifact-path",
        default="ann_model",
    )
    parser.add_argument(
        "--model-name",
        default="NeuroRoute-BBBP-ANN",
    )
    parser.add_argument(
        "--minimum-roc-auc",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=0.85,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = promote_model(
        tracking_uri=args.tracking_uri,
        run_id=args.run_id,
        artifact_path=args.artifact_path,
        model_name=args.model_name,
        minimum_roc_auc=args.minimum_roc_auc,
        minimum_accuracy=args.minimum_accuracy,
    )

    output_path = Path(
        "artifacts/registry/model_promotion.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Model promotion complete")
    print(f"Registered model: {result['model_name']}")
    print(f"Version: {result['version']}")
    print(
        f"Champion URI: {result['champion_model_uri']}"
    )
    print(f"Quality checks: {result['checks']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()