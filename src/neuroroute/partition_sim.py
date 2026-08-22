"""Partition SNN populations and place them on simulated chips."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MachineVertex:
    """A population fragment that fits on one processing core."""

    vertex_id: str
    population: str
    start_neuron: int
    end_neuron: int
    neuron_count: int


@dataclass(frozen=True)
class Placement:
    """Location assigned to one machine vertex."""

    vertex_id: str
    population: str
    chip_x: int
    chip_y: int
    core_id: int
    neuron_count: int


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the hardware configuration."""

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    hardware = config["hardware"]
    grid = hardware["grid"]
    chip = hardware["chip"]

    if grid["width"] <= 0 or grid["height"] <= 0:
        raise ValueError("Grid dimensions must be positive")

    if chip["processing_cores"] <= chip["reserved_cores"]:
        raise ValueError("No application cores are available")

    if chip["neurons_per_core"] <= 0:
        raise ValueError("neurons_per_core must be positive")

    return config


def partition_population(
    population: str,
    neuron_count: int,
    neurons_per_core: int,
) -> list[MachineVertex]:
    """Split a population into core-sized machine vertices."""

    if neuron_count <= 0:
        raise ValueError(f"Population {population!r} must not be empty")

    vertices: list[MachineVertex] = []
    number_of_parts = math.ceil(neuron_count / neurons_per_core)

    for part_index in range(number_of_parts):
        start = part_index * neurons_per_core
        end = min(start + neurons_per_core, neuron_count)

        vertices.append(
            MachineVertex(
                vertex_id=f"{population}_{part_index:03d}",
                population=population,
                start_neuron=start,
                end_neuron=end,
                neuron_count=end - start,
            )
        )

    return vertices


def build_machine_vertices(
    populations: dict[str, int],
    neurons_per_core: int,
) -> list[MachineVertex]:
    """Partition every application population."""

    vertices: list[MachineVertex] = []

    for population, neuron_count in populations.items():
        vertices.extend(
            partition_population(
                population=population,
                neuron_count=int(neuron_count),
                neurons_per_core=neurons_per_core,
            )
        )

    return vertices


def create_chip_coordinates(
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Create deterministic chip coordinates."""

    return [
        (x, y)
        for y in range(height)
        for x in range(width)
    ]


def place_round_robin(
    vertices: list[MachineVertex],
    width: int,
    height: int,
    processing_cores: int,
    reserved_cores: int,
) -> list[Placement]:
    """Spread vertices across chips in round-robin order."""

    coordinates = create_chip_coordinates(width, height)
    usable_cores_per_chip = processing_cores - reserved_cores
    total_capacity = len(coordinates) * usable_cores_per_chip

    if len(vertices) > total_capacity:
        raise ValueError(
            f"{len(vertices)} vertices cannot fit into "
            f"{total_capacity} available cores"
        )

    placements: list[Placement] = []

    for index, vertex in enumerate(vertices):
        chip_index = index % len(coordinates)
        local_core_offset = index // len(coordinates)
        chip_x, chip_y = coordinates[chip_index]
        core_id = reserved_cores + local_core_offset

        placements.append(
            Placement(
                vertex_id=vertex.vertex_id,
                population=vertex.population,
                chip_x=chip_x,
                chip_y=chip_y,
                core_id=core_id,
                neuron_count=vertex.neuron_count,
            )
        )

    return placements


def validate_placements(
    vertices: list[MachineVertex],
    placements: list[Placement],
    width: int,
    height: int,
    processing_cores: int,
    reserved_cores: int,
    neurons_per_core: int,
) -> None:
    """Check placement completeness and hardware constraints."""

    expected_ids = {vertex.vertex_id for vertex in vertices}
    placed_ids = {placement.vertex_id for placement in placements}

    if expected_ids != placed_ids:
        raise ValueError("Every machine vertex must be placed exactly once")

    occupied_cores: set[tuple[int, int, int]] = set()

    for placement in placements:
        if not 0 <= placement.chip_x < width:
            raise ValueError(f"Invalid chip x-coordinate: {placement}")

        if not 0 <= placement.chip_y < height:
            raise ValueError(f"Invalid chip y-coordinate: {placement}")

        if not reserved_cores <= placement.core_id < processing_cores:
            raise ValueError(f"Invalid application core: {placement}")

        if placement.neuron_count > neurons_per_core:
            raise ValueError(f"Core neuron capacity exceeded: {placement}")

        location = (
            placement.chip_x,
            placement.chip_y,
            placement.core_id,
        )

        if location in occupied_cores:
            raise ValueError(f"Two vertices share one core: {location}")

        occupied_cores.add(location)


def run_partition(config_path: str | Path) -> dict[str, Any]:
    """Execute naive partitioning and placement."""

    config = load_config(config_path)

    hardware = config["hardware"]
    grid = hardware["grid"]
    chip = hardware["chip"]
    populations = config["application"]["populations"]

    vertices = build_machine_vertices(
        populations=populations,
        neurons_per_core=chip["neurons_per_core"],
    )

    placements = place_round_robin(
        vertices=vertices,
        width=grid["width"],
        height=grid["height"],
        processing_cores=chip["processing_cores"],
        reserved_cores=chip["reserved_cores"],
    )

    validate_placements(
        vertices=vertices,
        placements=placements,
        width=grid["width"],
        height=grid["height"],
        processing_cores=chip["processing_cores"],
        reserved_cores=chip["reserved_cores"],
        neurons_per_core=chip["neurons_per_core"],
    )

    used_chips = {
        (placement.chip_x, placement.chip_y)
        for placement in placements
    }

    population_parts: dict[str, int] = {}
    for vertex in vertices:
        population_parts[vertex.population] = (
            population_parts.get(vertex.population, 0) + 1
        )

    return {
        "strategy": "round_robin",
        "hardware": hardware["name"],
        "grid": {
            "width": grid["width"],
            "height": grid["height"],
        },
        "total_neurons": sum(populations.values()),
        "machine_vertices": len(vertices),
        "used_chips": len(used_chips),
        "population_parts": population_parts,
        "placements": [asdict(item) for item in placements],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition and place the NeuroRoute SNN"
    )
    parser.add_argument(
        "--config",
        default="configs/hardware.yaml",
        help="Hardware YAML configuration",
    )
    parser.add_argument(
        "--output",
        default="artifacts/partition/naive_partition.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_partition(args.config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Partitioning complete")
    print(f"Strategy: {result['strategy']}")
    print(f"Total neurons: {result['total_neurons']}")
    print(f"Machine vertices: {result['machine_vertices']}")
    print(f"Used chips: {result['used_chips']}")
    print(f"Population parts: {result['population_parts']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()