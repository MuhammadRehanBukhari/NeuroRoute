"""Package the champion model and deployment configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from src.neuroroute.partition_compare import evaluate_placement
from src.neuroroute.placement import run_placement


MODEL_NAME = "NeuroRoute-BBBP-ANN"
MODEL_ALIAS = "champion"

CONFIG_FILES = [
    Path("configs/baseline.yaml"),
    Path("configs/snn.yaml"),
    Path("configs/hardware_hpo_best.yaml"),
    Path("configs/partition_hpo_best.yaml"),
]

DVC_FILES = [
    Path("data/raw/BBBP.csv.dvc"),
    Path("data/processed/train.npz.dvc"),
    Path("data/processed/validation.npz.dvc"),
    Path("data/processed/test.npz.dvc"),
    Path("data/processed/metadata.json.dvc"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def copy_required_file(
    source: Path,
    destination: Path,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required file missing: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_deployment_bundle(
    tracking_uri: str,
    output_root: Path,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    bundle_directory = output_root / "neuroroute_bundle"
    zip_path = output_root / "neuroroute_bundle.zip"

    if bundle_directory.exists():
        shutil.rmtree(bundle_directory)

    if zip_path.exists():
        zip_path.unlink()

    bundle_directory.mkdir(parents=True)

    # Download the exact model version referenced by the champion alias.
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(bundle_directory / "model"),
    )

    config_directory = bundle_directory / "configs"

    for config_path in CONFIG_FILES:
        copy_required_file(
            config_path,
            config_directory / config_path.name,
        )

    data_version_directory = bundle_directory / "data_versions"

    for dvc_path in DVC_FILES:
        copy_required_file(
            dvc_path,
            data_version_directory / dvc_path.name,
        )

    copy_required_file(
        Path("requirements-lock.txt"),
        bundle_directory / "requirements-lock.txt",
    )

    placement = run_placement(
        "configs/partition_hpo_best.yaml"
    )
    routing_metrics = evaluate_placement(
        "configs/partition_hpo_best.yaml"
    )

    results_directory = bundle_directory / "deployment"

    results_directory.mkdir(parents=True, exist_ok=True)

    (results_directory / "placement.json").write_text(
        json.dumps(placement, indent=2),
        encoding="utf-8",
    )
    (results_directory / "routing_metrics.json").write_text(
        json.dumps(routing_metrics, indent=2),
        encoding="utf-8",
    )

    tracked_files = []

    for path in sorted(bundle_directory.rglob("*")):
        if path.is_file():
            tracked_files.append(
                {
                    "path": str(
                        path.relative_to(bundle_directory)
                    ).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    manifest = {
        "bundle_format_version": "1.0",
        "project": "NeuroRoute",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "registered_model": {
            "name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": str(champion.version),
            "source_run_id": champion.run_id,
            "model_uri": model_uri,
        },
        "dataset": {
            "name": "BBBP",
            "versioning": "DVC",
            "metadata_files": [
                path.name for path in DVC_FILES
            ],
        },
        "deployment": {
            "target": (
                "SpiNNaker-style educational simulator"
            ),
            "placement_strategy": placement["strategy"],
            "machine_vertices": placement[
                "machine_vertices"
            ],
            "used_chips": placement["used_chips"],
            "maximum_router_entries": routing_metrics[
                "maximum_router_entries"
            ],
            "multicast_tree_links": routing_metrics[
                "multicast_tree_links"
            ],
            "router_overflow_events": routing_metrics[
                "router_overflow_events"
            ],
        },
        "limitations": [
            "No physical SpiNNaker2 execution was performed.",
            "Timing and bandwidth values are configurable estimates.",
            "The HPO neuron capacity requires hardware validation.",
        ],
        "files": tracked_files,
    }

    manifest_path = bundle_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=output_root,
        base_dir=bundle_directory.name,
    )

    return {
        "bundle_directory": str(bundle_directory),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "model_version": str(champion.version),
        "source_run_id": champion.run_id,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the NeuroRoute champion deployment"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/deployment"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_deployment_bundle(
        tracking_uri=args.tracking_uri,
        output_root=args.output_root,
    )

    print("Deployment bundle complete")
    print(f"Model version: {result['model_version']}")
    print(f"Source run: {result['source_run_id']}")
    print(f"Directory: {result['bundle_directory']}")
    print(f"Archive: {result['zip_path']}")
    print(f"Archive size: {result['zip_size_bytes']} bytes")
    print(f"Archive SHA256: {result['zip_sha256']}")


if __name__ == "__main__":
    main()