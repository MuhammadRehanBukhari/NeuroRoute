"""Deployment-wide multicast routing and router-pressure simulation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.neuroroute.multicast import build_multicast_tree
from src.neuroroute.partition_sim import run_partition
from src.neuroroute.routing import ChipTopology


ChipCoordinate = tuple[int, int]
DirectedLink = tuple[ChipCoordinate, ChipCoordinate]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    return config


def link_name(link: DirectedLink) -> str:
    """Convert a directed link into a JSON-safe name."""

    source, destination = link

    return (
        f"{source[0]},{source[1]}"
        f"->{destination[0]},{destination[1]}"
    )


def chip_name(chip: ChipCoordinate) -> str:
    return f"{chip[0]},{chip[1]}"


def simulate_deployment_routing(
    config_path: str | Path,
) -> dict[str, Any]:
    """Partition the SNN and route all adjacent-layer projections."""

    config = load_yaml(config_path)
    partition_result = run_partition(config_path)
    topology = ChipTopology.from_config(config_path)

    populations = list(config["application"]["populations"])
    routing_capacity = int(
        config["hardware"]["chip"]["routing_table_capacity"]
    )

    placements_by_population: dict[str, list[dict]] = defaultdict(list)

    for placement in partition_result["placements"]:
        placements_by_population[placement["population"]].append(
            placement
        )

    routing_entries: Counter[ChipCoordinate] = Counter()
    link_load: Counter[DirectedLink] = Counter()

    total_independent_unicast_hops = 0
    total_multicast_tree_links = 0
    routed_source_vertices = 0
    machine_projection_count = 0
    route_records: list[dict[str, Any]] = []

    # Create routes input -> hidden1 -> hidden2 -> output.
    for source_population, destination_population in zip(
        populations,
        populations[1:],
    ):
        source_placements = placements_by_population[source_population]
        destination_placements = placements_by_population[
            destination_population
        ]

        machine_projection_count += (
            len(source_placements) * len(destination_placements)
        )

        for source_placement in source_placements:
            source_chip = (
                source_placement["chip_x"],
                source_placement["chip_y"],
            )

            # Multiple destination cores on the same chip need only one
            # inter-chip multicast destination.
            destination_chips = sorted(
                {
                    (
                        placement["chip_x"],
                        placement["chip_y"],
                    )
                    for placement in destination_placements
                }
            )

            total_independent_unicast_hops += sum(
                topology.distance(source_chip, destination_chip)
                for destination_chip in destination_chips
            )

            remote_destinations = [
                destination
                for destination in destination_chips
                if destination != source_chip
            ]

            involved_chips: set[ChipCoordinate] = {source_chip}
            tree_links: tuple[DirectedLink, ...] = ()

            if remote_destinations:
                tree = build_multicast_tree(
                    topology=topology,
                    source=source_chip,
                    destinations=remote_destinations,
                )

                tree_links = tree.directed_links

                for link_source, link_destination in tree_links:
                    involved_chips.add(link_source)
                    involved_chips.add(link_destination)
                    link_load[(link_source, link_destination)] += 1

            # One routing key represents one source machine vertex.
            # Every router traversed by that key needs one simulated entry.
            for chip in involved_chips:
                routing_entries[chip] += 1

            total_multicast_tree_links += len(tree_links)
            routed_source_vertices += 1

            route_records.append(
                {
                    "source_vertex": source_placement["vertex_id"],
                    "source_population": source_population,
                    "destination_population": destination_population,
                    "source_chip": list(source_chip),
                    "destination_chips": [
                        list(chip) for chip in destination_chips
                    ],
                    "tree_link_count": len(tree_links),
                    "tree_links": [
                        {
                            "source": list(link_source),
                            "destination": list(link_destination),
                            "direction": topology.direction_between(
                                link_source,
                                link_destination,
                            ),
                        }
                        for link_source, link_destination in tree_links
                    ],
                }
            )

    all_chips = list(topology.graph.nodes)

    entries_by_chip = {
        chip_name(chip): routing_entries.get(chip, 0)
        for chip in all_chips
    }

    overflow_by_chip = {
        name: entries - routing_capacity
        for name, entries in entries_by_chip.items()
        if entries > routing_capacity
    }

    link_load_by_direction = {
        link_name(link): count
        for link, count in sorted(link_load.items())
    }

    maximum_router_entries = max(
        entries_by_chip.values(),
        default=0,
    )

    maximum_link_load = max(
        link_load_by_direction.values(),
        default=0,
    )

    saved_link_traversals = (
        total_independent_unicast_hops
        - total_multicast_tree_links
    )

    reduction_percent = (
        100.0
        * saved_link_traversals
        / total_independent_unicast_hops
        if total_independent_unicast_hops
        else 0.0
    )

    return {
        "strategy": partition_result["strategy"],
        "machine_vertices": partition_result["machine_vertices"],
        "machine_projections": machine_projection_count,
        "routed_source_vertices": routed_source_vertices,
        "independent_unicast_hops": (
            total_independent_unicast_hops
        ),
        "multicast_tree_links": total_multicast_tree_links,
        "saved_link_traversals": saved_link_traversals,
        "communication_reduction_percent": reduction_percent,
        "routing_table_capacity": routing_capacity,
        "maximum_router_entries": maximum_router_entries,
        "router_overflow_events": len(overflow_by_chip),
        "overflow_by_chip": overflow_by_chip,
        "entries_by_chip": entries_by_chip,
        "maximum_link_route_load": maximum_link_load,
        "link_route_load": link_load_by_direction,
        "routes": route_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate deployment-wide multicast routing"
    )
    parser.add_argument(
        "--config",
        default="configs/hardware.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/routing/deployment_routes.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = simulate_deployment_routing(args.config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Deployment routing simulation complete")
    print(f"Strategy: {result['strategy']}")
    print(f"Machine vertices: {result['machine_vertices']}")
    print(f"Machine projections: {result['machine_projections']}")
    print(
        "Independent unicast hops: "
        f"{result['independent_unicast_hops']}"
    )
    print(
        "Multicast tree links: "
        f"{result['multicast_tree_links']}"
    )
    print(
        "Communication reduction: "
        f"{result['communication_reduction_percent']:.2f}%"
    )
    print(
        "Maximum router entries: "
        f"{result['maximum_router_entries']}"
        f"/{result['routing_table_capacity']}"
    )
    print(
        "Router overflow events: "
        f"{result['router_overflow_events']}"
    )
    print(
        "Maximum directed-link route load: "
        f"{result['maximum_link_route_load']}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()