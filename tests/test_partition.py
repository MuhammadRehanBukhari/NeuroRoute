"""Tests for SNN partitioning and placement."""

import pytest

from src.neuroroute.partition_sim import (
    Placement,
    build_machine_vertices,
    partition_population,
    place_round_robin,
    run_partition,
    validate_placements,
)


def test_partition_population_preserves_all_neurons() -> None:
    vertices = partition_population(
        population="hidden",
        neuron_count=70,
        neurons_per_core=32,
    )

    assert len(vertices) == 3
    assert [vertex.neuron_count for vertex in vertices] == [32, 32, 6]
    assert sum(vertex.neuron_count for vertex in vertices) == 70

    assert vertices[0].start_neuron == 0
    assert vertices[0].end_neuron == 32
    assert vertices[-1].start_neuron == 64
    assert vertices[-1].end_neuron == 70


def test_build_machine_vertices_matches_network_size() -> None:
    populations = {
        "input": 64,
        "hidden": 33,
        "output": 1,
    }

    vertices = build_machine_vertices(
        populations=populations,
        neurons_per_core=32,
    )

    assert len(vertices) == 5
    assert sum(vertex.neuron_count for vertex in vertices) == 98

    population_counts = {
        name: sum(
            vertex.neuron_count
            for vertex in vertices
            if vertex.population == name
        )
        for name in populations
    }

    assert population_counts == populations


def test_round_robin_placement_has_unique_cores() -> None:
    vertices = partition_population(
        population="input",
        neuron_count=320,
        neurons_per_core=16,
    )

    placements = place_round_robin(
        vertices=vertices,
        width=4,
        height=2,
        processing_cores=6,
        reserved_cores=2,
    )

    locations = {
        (
            placement.chip_x,
            placement.chip_y,
            placement.core_id,
        )
        for placement in placements
    }

    assert len(placements) == 20
    assert len(locations) == 20
    assert min(item.core_id for item in placements) == 2
    assert max(item.core_id for item in placements) == 4


def test_placement_fails_when_machine_is_too_small() -> None:
    vertices = partition_population(
        population="input",
        neuron_count=160,
        neurons_per_core=16,
    )

    with pytest.raises(ValueError, match="cannot fit"):
        place_round_robin(
            vertices=vertices,
            width=1,
            height=1,
            processing_cores=6,
            reserved_cores=2,
        )


def test_validation_detects_two_vertices_on_one_core() -> None:
    vertices = partition_population(
        population="hidden",
        neuron_count=40,
        neurons_per_core=32,
    )

    invalid_placements = [
        Placement(
            vertex_id=vertices[0].vertex_id,
            population="hidden",
            chip_x=0,
            chip_y=0,
            core_id=2,
            neuron_count=32,
        ),
        Placement(
            vertex_id=vertices[1].vertex_id,
            population="hidden",
            chip_x=0,
            chip_y=0,
            core_id=2,
            neuron_count=8,
        ),
    ]

    with pytest.raises(ValueError, match="share one core"):
        validate_placements(
            vertices=vertices,
            placements=invalid_placements,
            width=2,
            height=2,
            processing_cores=8,
            reserved_cores=2,
            neurons_per_core=32,
        )


def test_project_configuration_produces_expected_partition() -> None:
    result = run_partition("configs/hardware.yaml")

    assert result["total_neurons"] == 1185
    assert result["machine_vertices"] == 38
    assert result["used_chips"] == 16

    assert result["population_parts"] == {
        "input": 32,
        "hidden1": 4,
        "hidden2": 1,
        "output": 1,
    }
