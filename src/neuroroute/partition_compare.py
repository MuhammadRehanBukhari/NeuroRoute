"""Compare routing cost for naive and optimized placements."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.neuroroute.multicast import build_multicast_tree
from src.neuroroute.placement import load_yaml, run_placement
from src.neuroroute.routing import ChipTopology


def evaluate_placement(
    partition_config_path: str | Path,
) -> dict[str, Any]:
    """Route one placement and calculate communication metrics."""

    partition_config = load_yaml(partition_config_path)["partition"]
    hardware_path = partition_config["hardware_config"]
    placement = run_placement(partition_config_path)

    with Path(hardware_path).open("r", encoding="utf-8") as file:
        hardware_config = yaml.safe_load(file)

    populations = list(
        hardware_config["application"]["populations"]
    )
    routing_capacity = int(
        hardware_config["hardware"]["chip"][
            "routing_table_capacity"
        ]
    )

    topology = ChipTopology.from_config(hardware_path)

    placements_by_population: dict[str, list[dict]] = defaultdict(
        list
    )

    for item in placement["placements"]:
        placements_by_population[item["population"]].append(item)

    total_unicast_hops = 0
    total_tree_links = 0
    machine_projections = 0
    routed_sources = 0

    routing_entries: Counter[tuple[int, int]] = Counter()
    link_route_load: Counter[
        tuple[tuple[int, int], tuple[int, int]]
    ] = Counter()

    for source_population, destination_population in zip(
        populations,
        populations[1:],
    ):
        sources = placements_by_population[source_population]
        destinations = placements_by_population[
            destination_population
        ]

        machine_projections += len(sources) * len(destinations)

        for source in sources:
            source_chip = (
                source["chip_x"],
                source["chip_y"],
            )

            destination_chips = sorted(
                {
                    (
                        destination["chip_x"],
                        destination["chip_y"],
                    )
                    for destination in destinations
                }
            )

            total_unicast_hops += sum(
                topology.distance(source_chip, destination_chip)
                for destination_chip in destination_chips
            )

            remote_destinations = [
                destination
                for destination in destination_chips
                if destination != source_chip
            ]

            involved_chips = {source_chip}

            if remote_destinations:
                tree = build_multicast_tree(
                    topology=topology,
                    source=source_chip,
                    destinations=remote_destinations,
                )

                total_tree_links += tree.link_count

                for link_source, link_destination in tree.directed_links:
                    involved_chips.add(link_source)
                    involved_chips.add(link_destination)
                    link_route_load[
                        (link_source, link_destination)
                    ] += 1

            for chip in involved_chips:
                routing_entries[chip] += 1

            routed_sources += 1

    maximum_router_entries = max(
        routing_entries.values(),
        default=0,
    )
    overflow_events = sum(
        entries > routing_capacity
        for entries in routing_entries.values()
    )
    maximum_link_route_load = max(
        link_route_load.values(),
        default=0,
    )

    return {
        "name": placement["name"],
        "strategy": placement["strategy"],
        "used_chips": placement["used_chips"],
        "machine_vertices": placement["machine_vertices"],
        "machine_projections": machine_projections,
        "routed_source_vertices": routed_sources,
        "independent_unicast_hops": total_unicast_hops,
        "multicast_tree_links": total_tree_links,
        "maximum_router_entries": maximum_router_entries,
        "routing_table_capacity": routing_capacity,
        "router_overflow_events": overflow_events,
        "maximum_link_route_load": maximum_link_route_load,
        "chip_occupancy": placement["chip_occupancy"],
    }


def reduction_percent(
    baseline: float,
    optimized: float,
) -> float:
    if baseline == 0:
        return 0.0

    return 100.0 * (baseline - optimized) / baseline


def compare_strategies() -> dict[str, Any]:
    naive = evaluate_placement(
        "configs/partition_naive.yaml"
    )
    optimized = evaluate_placement(
        "configs/partition_optimized.yaml"
    )

    return {
        "naive": naive,
        "optimized": optimized,
        "improvements": {
            "used_chips_reduction_percent": reduction_percent(
                naive["used_chips"],
                optimized["used_chips"],
            ),
            "unicast_hop_reduction_percent": reduction_percent(
                naive["independent_unicast_hops"],
                optimized["independent_unicast_hops"],
            ),
            "multicast_link_reduction_percent": reduction_percent(
                naive["multicast_tree_links"],
                optimized["multicast_tree_links"],
            ),
            "router_pressure_reduction_percent": reduction_percent(
                naive["maximum_router_entries"],
                optimized["maximum_router_entries"],
            ),
            "maximum_link_load_reduction_percent": reduction_percent(
                naive["maximum_link_route_load"],
                optimized["maximum_link_route_load"],
            ),
        },
    }


def main() -> None:
    result = compare_strategies()

    output_path = Path(
        "artifacts/partition/placement_comparison.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    naive = result["naive"]
    optimized = result["optimized"]
    improvements = result["improvements"]

    print("Placement comparison complete")
    print()
    print("Naive:")
    print(f"  Used chips: {naive['used_chips']}")
    print(
        f"  Independent unicast hops: "
        f"{naive['independent_unicast_hops']}"
    )
    print(
        f"  Multicast tree links: "
        f"{naive['multicast_tree_links']}"
    )
    print(
        f"  Maximum router entries: "
        f"{naive['maximum_router_entries']}"
    )
    print(
        f"  Maximum link route load: "
        f"{naive['maximum_link_route_load']}"
    )

    print()
    print("Connectivity-aware:")
    print(f"  Used chips: {optimized['used_chips']}")
    print(
        f"  Independent unicast hops: "
        f"{optimized['independent_unicast_hops']}"
    )
    print(
        f"  Multicast tree links: "
        f"{optimized['multicast_tree_links']}"
    )
    print(
        f"  Maximum router entries: "
        f"{optimized['maximum_router_entries']}"
    )
    print(
        f"  Maximum link route load: "
        f"{optimized['maximum_link_route_load']}"
    )

    print()
    print("Improvement:")
    print(
        "  Unicast-hop reduction: "
        f"{improvements['unicast_hop_reduction_percent']:.2f}%"
    )
    print(
        "  Multicast-link reduction: "
        f"{improvements['multicast_link_reduction_percent']:.2f}%"
    )
    print(
        "  Router-pressure reduction: "
        f"{improvements['router_pressure_reduction_percent']:.2f}%"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()