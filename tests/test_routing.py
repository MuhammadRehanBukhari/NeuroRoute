"""Tests for the six-direction chip topology."""

import pytest

from src.neuroroute.routing import ChipTopology


def test_four_by_four_topology_size() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    assert topology.graph.number_of_nodes() == 16
    assert topology.graph.number_of_edges() == 33


def test_interior_chip_has_six_neighbours() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    neighbours = topology.neighbours((1, 1))

    assert neighbours == {
        "east": (2, 1),
        "northeast": (2, 0),
        "north": (1, 0),
        "west": (0, 1),
        "southwest": (0, 2),
        "south": (1, 2),
    }


def test_corner_chip_respects_mesh_boundary() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    assert topology.neighbours((0, 0)) == {
        "east": (1, 0),
        "south": (0, 1),
    }


def test_route_across_grid_has_expected_hop_count() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    route = topology.shortest_path(
        source=(0, 0),
        destination=(3, 3),
    )

    assert route.chips[0] == (0, 0)
    assert route.chips[-1] == (3, 3)
    assert route.hop_count == 6
    assert len(route.directions) == route.hop_count


def test_diagonal_link_reduces_hop_count() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    route = topology.shortest_path(
        source=(0, 3),
        destination=(3, 0),
    )

    assert route.hop_count == 3
    assert route.directions == [
        "northeast",
        "northeast",
        "northeast",
    ]


def test_source_equal_to_destination_has_zero_hops() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    route = topology.shortest_path(
        source=(2, 2),
        destination=(2, 2),
    )

    assert route.chips == [(2, 2)]
    assert route.directions == []
    assert route.hop_count == 0


def test_wrap_around_creates_one_hop_route() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=True,
    )

    route = topology.shortest_path(
        source=(0, 0),
        destination=(3, 0),
    )

    assert route.hop_count == 1


def test_invalid_chip_is_rejected() -> None:
    topology = ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )

    with pytest.raises(ValueError, match="Unknown destination"):
        topology.shortest_path(
            source=(0, 0),
            destination=(10, 10),
        )