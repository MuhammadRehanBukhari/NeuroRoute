"""Optional adapter for sPyNNaker, PACMAN, and spalloc."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PACKAGES = {
    "sPyNNaker": {
        "distribution": "sPyNNaker",
        "module": "pyNN.spiNNaker",
        "purpose": "PyNN execution on SpiNNaker hardware",
    },
    "PACMAN": {
        "distribution": "SpiNNaker-PACMAN",
"module": "pacman",
        "purpose": "partitioning, placement, and routing",
    },
    "spalloc": {
"distribution": "spalloc",
"module": "spinnman.spalloc",
        "purpose": "remote SpiNNaker machine allocation",
    },
}


def distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def inspect_toolchain() -> dict[str, Any]:
    packages = {}

    for name, information in PACKAGES.items():
        installed_version = distribution_version(
            information["distribution"]
        )
        import_available = module_available(
            information["module"]
        )

        packages[name] = {
            "distribution": information["distribution"],
            "module": information["module"],
            "purpose": information["purpose"],
            "version": installed_version,
            "import_available": import_available,
            "ready": (
                installed_version is not None
                and import_available
            ),
        }

    toolchain_ready = all(
        information["ready"]
        for information in packages.values()
    )

    return {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "packages": packages,
        "toolchain_ready": toolchain_ready,
        "hardware_access_verified": False,
        "mode": (
            "hardware-capable"
            if toolchain_ready
            else "software-simulation"
        ),
        "notes": [
            (
                "sPyNNaker provides the PyNN-facing SpiNNaker "
                "backend."
            ),
            (
                "PACMAN performs graph partitioning, placement, "
                "key allocation, and route generation."
            ),
            (
                "spalloc allocates authorized remote machine "
                "resources; credentials are not stored here."
            ),
            (
                "Package availability does not prove physical "
                "hardware access."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect optional SpiNNaker dependencies"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/spinnaker/toolchain_status.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_toolchain()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("SpiNNaker toolchain inspection complete")
    print(f"Operating system: {result['operating_system']}")
    print(f"Python: {result['python_version']}")

    for name, information in result["packages"].items():
        print(
            f"{name}: version={information['version']}, "
            f"import_available="
            f"{information['import_available']}"
        )

    print(f"Toolchain ready: {result['toolchain_ready']}")
    print(f"Mode: {result['mode']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()