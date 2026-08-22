"""Tests for multicast route-tree construction."""

import pytest

from src.neuroroute.multicast import (
    build_multicast_tree,
    compare_multicast_cost,
    independent_unicast_cost,
    route_table_outputs,
)
from src.neuroroute.routing import ChipTopology


@pytest.fixture
def topology() -> ChipTopology:
    return ChipTopology(
        width=4,
        height=4,
        wrap_around=False,
    )


def test_single_destination_tree_matches_shortest_path(
    topology: ChipTopology,
) -> None:
    source = (0, 0)
    destination = (3, 3)

    tree = build_multicast_tree(
        topology,
        source,
        [destination],
    )

    assert tree.link_count == topology.distance(source, destination)
    assert tree.paths[destination][0] == source
    assert tree.paths[destination][-1] == destination


def test_multicast_tree_reaches_every_destination(
    topology: ChipTopology,
) -> None:
    source = (0, 0)
    destinations = [(3, 3), (3, 2), (2, 3)]

    tree = build_multicast_tree(
        topology,
        source,
        destinations,
    )

    assert set(tree.paths) == set(destinations)

    for destination in destinations:
        path = tree.paths[destination]
        assert path[0] == source
        assert path[-1] == destination


def test_multicast_result_is_a_tree(
    topology: ChipTopology,
) -> None:
    tree = build_multicast_tree(
        topology,
        source=(0, 0),
        destinations=[(3, 3), (3, 2), (2, 3)],
    )

    # Every finite tree has exactly one fewer edge than nodes.
    assert tree.link_count == tree.router_count - 1


def test_multicast_does_not_exceed_independent_unicast_cost(
    topology: ChipTopology,
) -> None:
    source = (0, 0)
    destinations = [(3, 3), (3, 2), (2, 3)]

    unicast_cost = independent_unicast_cost(
        topology,
        source,
        destinations,
    )

    tree = build_multicast_tree(
        topology,
        source,
        destinations,
    )

    assert tree.link_count <= unicast_cost


def test_duplicate_destinations_are_removed(
    topology: ChipTopology,
) -> None:
    tree = build_multicast_tree(
        topology,
        source=(0, 0),
        destinations=[(3, 3), (3, 3), (2, 2)],
    )

    assert tree.destinations == ((3, 3), (2, 2))


def test_router_outputs_match_tree_links(
    topology: ChipTopology,
) -> None:
    tree = build_multicast_tree(
        topology,
        source=(0, 0),
        destinations=[(3, 3), (3, 2), (2, 3)],
    )

    outputs = route_table_outputs(topology, tree)

    number_of_outputs = sum(
        len(directions)
        for directions in outputs.values()
    )

    assert number_of_outputs == tree.link_count

    for source, destination in tree.directed_links:
        direction = topology.direction_between(source, destination)
        assert direction in outputs[source]


def test_cost_comparison_is_internally_consistent(
    topology: ChipTopology,
) -> None:
    result = compare_multicast_cost(
        topology,
        source=(0, 0),
        destinations=[(3, 3), (3, 2), (2, 3)],
    )

    assert result["saved_link_traversals"] == (
        result["independent_unicast_hops"]
        - result["multicast_tree_links"]
    )

    assert 0.0 <= result["reduction_percent"] <= 100.0


def test_empty_destination_collection_is_rejected(
    topology: ChipTopology,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one multicast destination",
    ):
        build_multicast_tree(
            topology,
            source=(0, 0),
            destinations=[],
        )


def test_invalid_destination_is_rejected(
    topology: ChipTopology,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unknown multicast destination",
    ):
        build_multicast_tree(
            topology,
            source=(0, 0),
            destinations=[(10, 10)],
        )
