"""Validate deployment integrity and load the bundled MLflow model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_bundle(bundle_directory: Path) -> dict[str, Any]:
    manifest_path = bundle_directory / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError("Deployment manifest is missing")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    verified_files = 0

    for expected in manifest["files"]:
        path = bundle_directory / expected["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Manifest file is missing: {expected['path']}"
            )

        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)

        if actual_size != expected["size_bytes"]:
            raise ValueError(
                f"Size mismatch for {expected['path']}"
            )

        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {expected['path']}"
            )

        verified_files += 1

    model_descriptors = list(
        (bundle_directory / "model").rglob("MLmodel")
    )

    if len(model_descriptors) != 1:
        raise ValueError(
            "Expected exactly one bundled MLflow model, "
            f"found {len(model_descriptors)}"
        )

    model_directory = model_descriptors[0].parent
    model = mlflow.pyfunc.load_model(str(model_directory))

    # Validate the expected 1024-feature inference interface.
    sample = np.zeros((1, 1024), dtype=np.float32)
    prediction = np.asarray(model.predict(sample))

    if prediction.size == 0:
        raise ValueError("Bundled model returned no prediction")

    if not np.isfinite(prediction).all():
        raise ValueError(
            "Bundled model returned non-finite values"
        )

    routing_path = (
        bundle_directory
        / "deployment"
        / "routing_metrics.json"
    )
    routing = json.loads(
        routing_path.read_text(encoding="utf-8")
    )

    if routing["router_overflow_events"] != 0:
        raise ValueError(
            "Deployment has routing-table overflow events"
        )

    return {
        "status": "valid",
        "verified_files": verified_files,
        "model_directory": str(model_directory),
        "model_input_shape": list(sample.shape),
        "model_output_shape": list(prediction.shape),
        "model_output": prediction.tolist(),
        "registered_model": manifest["registered_model"],
        "machine_vertices": manifest["deployment"][
            "machine_vertices"
        ],
        "used_chips": manifest["deployment"]["used_chips"],
        "multicast_tree_links": manifest["deployment"][
            "multicast_tree_links"
        ],
        "router_overflow_events": routing[
            "router_overflow_events"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a NeuroRoute deployment bundle"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "artifacts/deployment/neuroroute_bundle"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_bundle(args.bundle)

    print("Deployment validation complete")
    print(f"Status: {result['status']}")
    print(f"Verified files: {result['verified_files']}")
    print(
        f"Registered model: "
        f"{result['registered_model']}"
    )
    print(
        f"Model input shape: "
        f"{result['model_input_shape']}"
    )
    print(
        f"Model output shape: "
        f"{result['model_output_shape']}"
    )
    print(f"Model output: {result['model_output']}")
    print(f"Machine vertices: {result['machine_vertices']}")
    print(f"Used chips: {result['used_chips']}")
    print(
        f"Multicast tree links: "
        f"{result['multicast_tree_links']}"
    )
    print(
        f"Router overflow events: "
        f"{result['router_overflow_events']}"
    )


if __name__ == "__main__":
    main()