"""Map a PyNN network using sPyNNaker and PACMAN virtual mode."""

from __future__ import annotations

import time

import pyNN.spiNNaker as sim


def run_virtual_mapping() -> dict:
    """Build and map a network without physical hardware."""

    simulation_time_ms = 100.0
    mapping_start = time.perf_counter()

    sim.setup(timestep=1.0)

    try:
        # Explicitly control how populations are partitioned.
        sim.set_number_of_neurons_per_core(
            sim.SpikeSourcePoisson,
            64,
        )
        sim.set_number_of_neurons_per_core(
            sim.IF_curr_exp,
            64,
        )

        input_population = sim.Population(
            128,
            sim.SpikeSourcePoisson(
                rate=20.0,
            ),
            label="neuroroute_input",
            additional_parameters={"seed": 42},
        )

        hidden_population = sim.Population(
            128,
            sim.IF_curr_exp(
                tau_m=20.0,
                tau_syn_E=5.0,
                tau_syn_I=5.0,
                v_rest=-65.0,
                v_reset=-65.0,
                v_thresh=-50.0,
                tau_refrac=2.0,
            ),
            label="neuroroute_hidden",
        )

        output_population = sim.Population(
            32,
            sim.IF_curr_exp(
                tau_m=20.0,
                tau_syn_E=5.0,
                tau_syn_I=5.0,
                v_rest=-65.0,
                v_reset=-65.0,
                v_thresh=-50.0,
                tau_refrac=2.0,
            ),
            label="neuroroute_output",
        )

        sim.Projection(
            input_population,
            hidden_population,
            sim.FixedProbabilityConnector(
                p_connect=0.10,
            ),
            sim.StaticSynapse(
                weight=0.05,
                delay=1.0,
            ),
            receptor_type="excitatory",
            label="input_to_hidden",
        )

        sim.Projection(
            hidden_population,
            output_population,
            sim.AllToAllConnector(),
            sim.StaticSynapse(
                weight=0.05,
                delay=1.0,
            ),
            receptor_type="excitatory",
            label="hidden_to_output",
        )

        # In virtual mode this invokes graph partitioning, placement,
        # key allocation, route generation, and data generation.
        sim.run(simulation_time_ms)

    finally:
        sim.end()

    elapsed_seconds = time.perf_counter() - mapping_start

    return {
        "backend": "sPyNNaker",
        "execution_mode": "virtual_board",
        "hardware_execution": False,
        "simulation_time_ms": simulation_time_ms,
        "input_neurons": 128,
        "hidden_neurons": 128,
        "output_neurons": 32,
        "neurons_per_core": 64,
        "expected_machine_vertices": 5,
        "mapping_elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    result = run_virtual_mapping()

    print("sPyNNaker virtual mapping complete")

    for name, value in result.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()