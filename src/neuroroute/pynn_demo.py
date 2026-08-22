"""Small PyNN spiking-network demonstration using the Brian2 backend."""

from pathlib import Path

import brian2
import matplotlib.pyplot as plt

brian2.prefs.codegen.target = "numpy"

import pyNN.brian2 as sim


OUTPUT_PATH = Path("artifacts/pynn/pynn_demo.png")


def run_demo() -> dict:
    """Build, simulate, record, and plot a small PyNN network."""

    # Configure the simulator.
    sim.setup(timestep=0.1, min_delay=0.1)

    try:
        # Input neuron producing spikes at fixed times in milliseconds.
        input_population = sim.Population(
            1,
            sim.SpikeSourceArray(
                spike_times=[10.0, 20.0, 30.0, 40.0, 50.0]
            ),
            label="input",
        )

        # Four leaky integrate-and-fire neurons.
        output_population = sim.Population(
            4,
            sim.IF_curr_exp(
                tau_m=20.0,
                tau_syn_E=5.0,
                tau_syn_I=5.0,
                v_rest=-65.0,
                v_reset=-65.0,
                v_thresh=-50.0,
                tau_refrac=2.0,
            ),
            label="output",
        )

        output_population.initialize(v=-65.0)

        # Connect the input neuron to every output neuron.
        sim.Projection(
            input_population,
            output_population,
            sim.AllToAllConnector(),
            sim.StaticSynapse(weight=5.0, delay=1.0),
            receptor_type="excitatory",
            label="input_to_output",
        )

        # Record input spikes, output spikes, and membrane voltage.
        input_population.record("spikes")
        output_population.record(["spikes", "v"])

        # Run the network for 100 milliseconds.
        sim.run(100.0)

        input_data = input_population.get_data("spikes")
        output_data = output_population.get_data(["spikes", "v"])

        input_spikes = input_data.segments[0].spiketrains
        output_spikes = output_data.segments[0].spiketrains
        voltage_signal = output_data.segments[0].filter(name="v")[0]

        input_spike_count = sum(len(train) for train in input_spikes)
        output_spike_count = sum(len(train) for train in output_spikes)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Plot the membrane voltage of the first output neuron.
        times_ms = voltage_signal.times.rescale("ms").magnitude
        voltage_mv = voltage_signal[:, 0].rescale("mV").magnitude

        plt.figure(figsize=(9, 4))
        plt.plot(times_ms, voltage_mv)
        plt.axhline(
            -50.0,
            color="red",
            linestyle="--",
            label="Spike threshold",
        )
        plt.xlabel("Simulation time (ms)")
        plt.ylabel("Membrane voltage (mV)")
        plt.title("NeuroRoute PyNN/Brian2 demonstration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH, dpi=150)
        plt.close()

        return {
            "backend": "Brian2",
            "duration_ms": 100.0,
            "timestep_ms": 0.1,
            "input_neurons": len(input_population),
            "output_neurons": len(output_population),
            "input_spikes": input_spike_count,
            "output_spikes": output_spike_count,
            "plot": str(OUTPUT_PATH),
        }

    finally:
        # Always release simulator resources.
        sim.end()


def main() -> None:
    results = run_demo()

    print("PyNN simulation complete")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
