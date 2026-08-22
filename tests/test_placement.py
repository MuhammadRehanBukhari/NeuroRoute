"""Tests for placement strategies and their comparison."""

from src.neuroroute.partition_compare import compare_strategies
from src.neuroroute.placement import run_placement


def placement_locations(result: dict) -> set[tuple[int, int, int]]:
    return {
        (
            placement["chip_x"],
            placement["chip_y"],
            placement["core_id"],
        )
        for placement in result["placements"]
    }


def test_naive_placement_respects_constraints() -> None:
    result = run_placement("configs/partition_naive.yaml")

    assert result["machine_vertices"] == 38
    assert result["total_neurons"] == 1185
    assert result["maximum_cores_used_on_chip"] <= 4

    assert len(placement_locations(result)) == 38


def test_optimized_placement_respects_constraints() -> None:
    result = run_placement("configs/partition_optimized.yaml")

    assert result["machine_vertices"] == 38
    assert result["total_neurons"] == 1185
    assert result["maximum_cores_used_on_chip"] <= 4

    assert len(placement_locations(result)) == 38


def test_optimized_placement_is_deterministic() -> None:
    first = run_placement("configs/partition_optimized.yaml")
    second = run_placement("configs/partition_optimized.yaml")

    assert first["placements"] == second["placements"]
    assert first["chip_occupancy"] == second["chip_occupancy"]


def test_connectivity_aware_placement_uses_fewer_chips() -> None:
    naive = run_placement("configs/partition_naive.yaml")
    optimized = run_placement("configs/partition_optimized.yaml")

    assert optimized["used_chips"] < naive["used_chips"]


def test_optimized_placement_reduces_routing_cost() -> None:
    comparison = compare_strategies()

    naive = comparison["naive"]
    optimized = comparison["optimized"]

    assert (
        optimized["independent_unicast_hops"]
        < naive["independent_unicast_hops"]
    )
    assert (
        optimized["multicast_tree_links"]
        < naive["multicast_tree_links"]
    )


def test_both_strategies_avoid_router_overflow() -> None:
    comparison = compare_strategies()

    assert comparison["naive"]["router_overflow_events"] == 0
    assert comparison["optimized"]["router_overflow_events"] == 0