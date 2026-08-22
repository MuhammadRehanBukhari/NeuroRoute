"""Multicast route-tree construction and communication-cost comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from src.neuroroute.routing import ChipTopology, parse_coordinate


ChipCoordinate = tuple[int, int]
DirectedLink = tuple[ChipCoordinate, ChipCoordinate]


@dataclass(frozen=True)
class MulticastTree:
    """A source-rooted tree connecting multiple destination chips."""

    source: ChipCoordinate
    destinations: tuple[ChipCoordinate, ...]
    directed_links: tuple[DirectedLink, ...]
    paths: dict[ChipCoordinate, list[ChipCoordinate]]

    @property
    def link_count(self) -> int:
        return len(self.directed_links)

    @property
    def router_count(self) -> int:
        chips = {self.source}

        for source, destination in self.directed_links:
            chips.add(source)
            chips.add(destination)

        return len(chips)


def validate_terminals(
    topology: ChipTopology,
    source: ChipCoordinate,
    destinations: list[ChipCoordinate],
) -> tuple[ChipCoordinate, ...]:
    """Validate and deduplicate multicast destinations."""

    if not topology.contains(source):
        raise ValueError(f"Unknown multicast source: {source}")

    unique_destinations = tuple(dict.fromkeys(destinations))

    if not unique_destinations:
        raise ValueError("At least one multicast destination is required")

    for destination in unique_destinations:
        if not topology.contains(destination):
            raise ValueError(
                f"Unknown multicast destination: {destination}"
            )

    return unique_destinations


def independent_unicast_cost(
    topology: ChipTopology,
    source: ChipCoordinate,
    destinations: list[ChipCoordinate],
) -> int:
    """Count link traversals when sending a separate packet per target."""

    unique_destinations = validate_terminals(
        topology,
        source,
        destinations,
    )

    return sum(
        topology.distance(source, destination)
        for destination in unique_destinations
    )


def build_multicast_tree(
    topology: ChipTopology,
    source: ChipCoordinate,
    destinations: list[ChipCoordinate],
) -> MulticastTree:
    """Approximate a minimum-link multicast tree.

    NetworkX uses a Steiner-tree approximation. Non-terminal chips may be
    included when they reduce the total number of links.
    """

    unique_destinations = validate_terminals(
        topology,
        source,
        destinations,
    )

    terminals = {source, *unique_destinations}

    undirected_tree = nx.algorithms.approximation.steiner_tree(
        topology.graph,
        terminals,
    )

    # Orient the undirected tree away from the multicast source.
    directed_links = tuple(
        nx.bfs_edges(undirected_tree, source=source)
    )

    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(undirected_tree.nodes)
    directed_graph.add_edges_from(directed_links)

    paths = {
        destination: nx.shortest_path(
            directed_graph,
            source=source,
            target=destination,
        )
        for destination in unique_destinations
    }

    return MulticastTree(
        source=source,
        destinations=unique_destinations,
        directed_links=directed_links,
        paths=paths,
    )


def route_table_outputs(
    topology: ChipTopology,
    tree: MulticastTree,
) -> dict[ChipCoordinate, list[str]]:
    """Return outgoing link directions needed at each router."""

    outputs: dict[ChipCoordinate, list[str]] = {}

    for source, destination in tree.directed_links:
        direction = topology.direction_between(source, destination)
        outputs.setdefault(source, []).append(direction)

    for directions in outputs.values():
        directions.sort()

    return outputs


def compare_multicast_cost(
    topology: ChipTopology,
    source: ChipCoordinate,
    destinations: list[ChipCoordinate],
) -> dict:
    """Compare separate unicast packets with one multicast tree."""

    unicast_hops = independent_unicast_cost(
        topology,
        source,
        destinations,
    )

    tree = build_multicast_tree(
        topology,
        source,
        destinations,
    )

    saved_link_traversals = unicast_hops - tree.link_count

    reduction_percent = (
        100.0 * saved_link_traversals / unicast_hops
        if unicast_hops
        else 0.0
    )

    outputs = route_table_outputs(topology, tree)

    return {
        "source": list(tree.source),
        "destinations": [
            list(destination)
            for destination in tree.destinations
        ],
        "independent_unicast_hops": unicast_hops,
        "multicast_tree_links": tree.link_count,
        "multicast_tree_routers": tree.router_count,
        "saved_link_traversals": saved_link_traversals,
        "reduction_percent": reduction_percent,
        "tree_links": [
            {
                "source": list(link_source),
                "destination": list(link_destination),
                "direction": topology.direction_between(
                    link_source,
                    link_destination,
                ),
            }
            for link_source, link_destination in tree.directed_links
        ],
        "router_outputs": {
            f"{chip[0]},{chip[1]}": directions
            for chip, directions in outputs.items()
        },
        "paths": {
            f"{destination[0]},{destination[1]}": [
                list(chip) for chip in path
            ]
            for destination, path in tree.paths.items()
        },
    }


def parse_destinations(value: str) -> list[ChipCoordinate]:
    """Parse coordinates formatted as 'x,y;x,y;x,y'."""

    try:
        destinations = [
            parse_coordinate(item.strip())
            for item in value.split(";")
            if item.strip()
        ]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(
            "Destinations must have the form x,y;x,y"
        ) from error

    if not destinations:
        raise argparse.ArgumentTypeError(
            "At least one destination is required"
        )

    return destinations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an optimized multicast route tree"
    )
    parser.add_argument(
        "--config",
        default="configs/hardware.yaml",
    )
    parser.add_argument(
        "--source",
        type=parse_coordinate,
        default=(0, 0),
    )
    parser.add_argument(
        "--destinations",
        type=parse_destinations,
        default=[(3, 3), (3, 2), (2, 3)],
    )
    parser.add_argument(
        "--output",
        default="artifacts/routing/multicast_example.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    topology = ChipTopology.from_config(args.config)

    result = compare_multicast_cost(
        topology=topology,
        source=args.source,
        destinations=args.destinations,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Multicast route tree complete")
    print(f"Source: {result['source']}")
    print(f"Destinations: {result['destinations']}")
    print(
        "Independent unicast hops: "
        f"{result['independent_unicast_hops']}"
    )
    print(
        "Multicast tree links: "
        f"{result['multicast_tree_links']}"
    )
    print(
        "Saved link traversals: "
        f"{result['saved_link_traversals']}"
    )
    print(
        "Communication reduction: "
        f"{result['reduction_percent']:.2f}%"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()