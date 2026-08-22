"""Naive and connectivity-aware machine-vertex placement."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.neuroroute.partition_sim import (
    Placement,
    build_machine_vertices,
    load_config,
    place_round_robin,
    validate_placements,
)
from src.neuroroute.routing import ChipTopology


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        result = yaml.safe_load(file)

    if not isinstance(result, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    return result


def place_connectivity_aware(
    vertices,
    population_order: list[str],
    topology: ChipTopology,
    processing_cores: int,
    reserved_cores: int,
    max_cores_per_chip: int,
) -> list[Placement]:
    """Greedily place connected layers close to one another."""

    physical_usable_cores = processing_cores - reserved_cores

    if max_cores_per_chip > physical_usable_cores:
        raise ValueError(
            "Configured per-chip allocation exceeds physical cores"
        )

    coordinates = sorted(topology.graph.nodes)
    total_capacity = len(coordinates) * max_cores_per_chip

    if len(vertices) > total_capacity:
        raise ValueError(
            f"{len(vertices)} vertices cannot fit into "
            f"{total_capacity} allocated cores"
        )

    centre = (
        (topology.width - 1) // 2,
        (topology.height - 1) // 2,
    )

    occupancy: Counter[tuple[int, int]] = Counter()
    placements: list[Placement] = []
    placements_by_population: dict[str, list[Placement]] = defaultdict(
        list
    )

    predecessor = {
        population_order[index]: population_order[index - 1]
        for index in range(1, len(population_order))
    }

    for vertex in vertices:
        candidates = [
            coordinate
            for coordinate in coordinates
            if occupancy[coordinate] < max_cores_per_chip
        ]

        previous_population = predecessor.get(vertex.population)
        previous_placements = (
            placements_by_population[previous_population]
            if previous_population is not None
            else []
        )

        def placement_score(
            coordinate: tuple[int, int],
        ) -> tuple[float, int, int, int, int]:
            if previous_placements:
                connection_cost = sum(
                    topology.distance(
                        coordinate,
                        (
                            placement.chip_x,
                            placement.chip_y,
                        ),
                    )
                    * placement.neuron_count
                    for placement in previous_placements
                )
            else:
                connection_cost = 0.0

            central_distance = topology.distance(
                coordinate,
                centre,
            )

            # Prefer:
            # 1. lower connectivity cost,
            # 2. chips nearer the centre,
            # 3. partially occupied chips,
            # 4. deterministic coordinates.
            return (
                connection_cost,
                central_distance,
                -occupancy[coordinate],
                coordinate[1],
                coordinate[0],
            )

        selected_chip = min(candidates, key=placement_score)
        core_id = reserved_cores + occupancy[selected_chip]

        placement = Placement(
            vertex_id=vertex.vertex_id,
            population=vertex.population,
            chip_x=selected_chip[0],
            chip_y=selected_chip[1],
            core_id=core_id,
            neuron_count=vertex.neuron_count,
        )

        placements.append(placement)
        placements_by_population[vertex.population].append(placement)
        occupancy[selected_chip] += 1

    return placements


def run_placement(
    partition_config_path: str | Path,
) -> dict[str, Any]:
    """Run the placement strategy selected by a YAML configuration."""

    partition_config = load_yaml(partition_config_path)["partition"]
    hardware_path = partition_config["hardware_config"]
    hardware_config = load_config(hardware_path)

    hardware = hardware_config["hardware"]
    grid = hardware["grid"]
    chip = hardware["chip"]
    populations = hardware_config["application"]["populations"]

    max_cores_per_chip = int(
        partition_config["constraints"][
            "max_application_cores_per_chip"
        ]
    )

    vertices = build_machine_vertices(
        populations=populations,
        neurons_per_core=chip["neurons_per_core"],
    )

    strategy = partition_config["strategy"]

    if strategy == "round_robin":
        # Restrict the existing round-robin placer to the same number
        # of allocated cores available to the optimized strategy.
        placements = place_round_robin(
            vertices=vertices,
            width=grid["width"],
            height=grid["height"],
            processing_cores=(
                chip["reserved_cores"] + max_cores_per_chip
            ),
            reserved_cores=chip["reserved_cores"],
        )

    elif strategy == "connectivity_aware":
        topology = ChipTopology.from_config(hardware_path)

        placements = place_connectivity_aware(
            vertices=vertices,
            population_order=list(populations),
            topology=topology,
            processing_cores=chip["processing_cores"],
            reserved_cores=chip["reserved_cores"],
            max_cores_per_chip=max_cores_per_chip,
        )

    else:
        raise ValueError(f"Unknown placement strategy: {strategy}")

    validate_placements(
        vertices=vertices,
        placements=placements,
        width=grid["width"],
        height=grid["height"],
        processing_cores=chip["processing_cores"],
        reserved_cores=chip["reserved_cores"],
        neurons_per_core=chip["neurons_per_core"],
    )

    occupancy: Counter[tuple[int, int]] = Counter(
        (placement.chip_x, placement.chip_y)
        for placement in placements
    )

    if occupancy and max(occupancy.values()) > max_cores_per_chip:
        raise ValueError("Allocated per-chip core limit was exceeded")

    population_parts: Counter[str] = Counter(
        vertex.population for vertex in vertices
    )

    return {
        "name": partition_config["name"],
        "strategy": strategy,
        "hardware_config": hardware_path,
        "max_application_cores_per_chip": max_cores_per_chip,
        "total_neurons": sum(populations.values()),
        "machine_vertices": len(vertices),
        "used_chips": len(occupancy),
        "maximum_cores_used_on_chip": max(
            occupancy.values(),
            default=0,
        ),
        "population_parts": dict(population_parts),
        "chip_occupancy": {
            f"{chip_coordinate[0]},{chip_coordinate[1]}": count
            for chip_coordinate, count in sorted(occupancy.items())
        },
        "placements": [
            asdict(placement)
            for placement in placements
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place partitioned SNN vertices"
    )
    parser.add_argument(
        "--partition-config",
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.partition_config)["partition"]

    output_value = (
        args.output
        if args.output is not None
        else config["output"]["path"]
    )

    result = run_placement(args.partition_config)

    output_path = Path(output_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Placement complete")
    print(f"Name: {result['name']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Machine vertices: {result['machine_vertices']}")
    print(f"Used chips: {result['used_chips']}")
    print(
        "Maximum allocated cores on one chip: "
        f"{result['maximum_cores_used_on_chip']}"
        f"/{result['max_application_cores_per_chip']}"
    )
    print(f"Chip occupancy: {result['chip_occupancy']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()