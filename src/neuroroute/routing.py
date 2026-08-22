"""SpiNNaker-style chip topology and shortest-path routing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml


DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "east": (1, 0),
    "northeast": (1, -1),
    "north": (0, -1),
    "west": (-1, 0),
    "southwest": (-1, 1),
    "south": (0, 1),
}


@dataclass(frozen=True)
class RoutedPath:
    """A route between two chips."""

    source: tuple[int, int]
    destination: tuple[int, int]
    chips: list[tuple[int, int]]
    directions: list[str]

    @property
    def hop_count(self) -> int:
        return len(self.chips) - 1


class ChipTopology:
    """Bounded or wrap-around six-direction chip mesh."""

    def __init__(
        self,
        width: int,
        height: int,
        wrap_around: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Topology dimensions must be positive")

        self.width = width
        self.height = height
        self.wrap_around = wrap_around
        self.graph = nx.Graph()
        self._build_graph()

    @classmethod
    def from_config(cls, path: str | Path) -> "ChipTopology":
        with Path(path).open("r", encoding="utf-8") as file:
            config: dict[str, Any] = yaml.safe_load(file)

        grid = config["hardware"]["grid"]

        return cls(
            width=int(grid["width"]),
            height=int(grid["height"]),
            wrap_around=bool(grid["wrap_around"]),
        )

    def contains(self, chip: tuple[int, int]) -> bool:
        x, y = chip
        return 0 <= x < self.width and 0 <= y < self.height

    def _move(
        self,
        chip: tuple[int, int],
        direction: str,
    ) -> tuple[int, int] | None:
        dx, dy = DIRECTION_DELTAS[direction]
        x = chip[0] + dx
        y = chip[1] + dy

        if self.wrap_around:
            return x % self.width, y % self.height

        neighbour = (x, y)
        return neighbour if self.contains(neighbour) else None

    def _build_graph(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.graph.add_node((x, y))

        for chip in list(self.graph.nodes):
            for direction in DIRECTION_DELTAS:
                neighbour = self._move(chip, direction)

                if neighbour is not None and neighbour != chip:
                    self.graph.add_edge(chip, neighbour)

    def neighbours(
        self,
        chip: tuple[int, int],
    ) -> dict[str, tuple[int, int]]:
        if not self.contains(chip):
            raise ValueError(f"Unknown chip coordinate: {chip}")

        result: dict[str, tuple[int, int]] = {}

        for direction in DIRECTION_DELTAS:
            neighbour = self._move(chip, direction)

            if neighbour is not None and neighbour != chip:
                result[direction] = neighbour

        return result

    def direction_between(
        self,
        source: tuple[int, int],
        destination: tuple[int, int],
    ) -> str:
        for direction, neighbour in self.neighbours(source).items():
            if neighbour == destination:
                return direction

        raise ValueError(
            f"Chips {source} and {destination} are not direct neighbours"
        )

    def shortest_path(
        self,
        source: tuple[int, int],
        destination: tuple[int, int],
    ) -> RoutedPath:
        if not self.contains(source):
            raise ValueError(f"Unknown source chip: {source}")

        if not self.contains(destination):
            raise ValueError(f"Unknown destination chip: {destination}")

        chips = nx.shortest_path(
            self.graph,
            source=source,
            target=destination,
        )

        directions = [
            self.direction_between(first, second)
            for first, second in zip(chips, chips[1:])
        ]

        return RoutedPath(
            source=source,
            destination=destination,
            chips=chips,
            directions=directions,
        )

    def distance(
        self,
        source: tuple[int, int],
        destination: tuple[int, int],
    ) -> int:
        return self.shortest_path(source, destination).hop_count


def parse_coordinate(value: str) -> tuple[int, int]:
    """Convert a command-line value such as '2,3' into a coordinate."""

    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return int(x_text), int(y_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Coordinates must have the form x,y"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the NeuroRoute chip topology"
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
        "--destination",
        type=parse_coordinate,
        default=(3, 3),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = ChipTopology.from_config(args.config)
    route = topology.shortest_path(args.source, args.destination)

    print("Chip topology ready")
    print(f"Dimensions: {topology.width} x {topology.height}")
    print(f"Chips: {topology.graph.number_of_nodes()}")
    print(f"Physical links: {topology.graph.number_of_edges()}")
    print(f"Wrap around: {topology.wrap_around}")
    print(f"Source: {route.source}")
    print(f"Destination: {route.destination}")
    print(f"Route chips: {route.chips}")
    print(f"Directions: {route.directions}")
    print(f"Hop count: {route.hop_count}")


if __name__ == "__main__":
    main()