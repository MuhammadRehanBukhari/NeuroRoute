"""Link bandwidth, queue, and timing simulation for routed spike traffic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from src.neuroroute.partition_sim import run_partition
from src.neuroroute.traffic_sim import simulate_deployment_routing


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def directed_link_name(
    source: tuple[int, int],
    destination: tuple[int, int],
) -> str:
    return (
        f"{source[0]},{source[1]}"
        f"->{destination[0]},{destination[1]}"
    )


def simulate_link_traffic(
    config_path: str | Path,
) -> dict[str, Any]:
    """Convert routed spike rates into link utilization and latency."""

    config = load_config(config_path)
    routing = simulate_deployment_routing(config_path)
    partition = run_partition(config_path)

    hardware = config["hardware"]
    traffic = config["traffic"]

    duration_ms = float(
        hardware["simulation"]["duration_ms"]
    )
    capacity_packets_per_ms = float(
        hardware["links"]["bandwidth_packets_per_ms"]
    )
    propagation_ms = float(
        hardware["links"]["latency_ms_per_hop"]
    )
    router_processing_ms = float(
        traffic["timing"]["router_processing_ms"]
    )
    serialization_ms = float(
        traffic["timing"]["serialization_overhead_ms"]
    )
    maximum_queue_packets = int(
        traffic["queue"]["maximum_packets"]
    )
    packet_bytes = int(traffic["packet_bytes"])
    spike_rates_hz = traffic["spike_rates_hz"]

    placement_by_vertex = {
        placement["vertex_id"]: placement
        for placement in partition["placements"]
    }

    link_packet_rates: Counter[str] = Counter()
    route_packet_rates: dict[str, float] = {}

    # Convert every source machine vertex's neuron spike rate into
    # multicast packet traffic on every link in its route tree.
    for route in routing["routes"]:
        vertex_id = route["source_vertex"]
        population = route["source_population"]
        neuron_count = placement_by_vertex[vertex_id]["neuron_count"]

        packets_per_ms = (
            neuron_count
            * float(spike_rates_hz[population])
            / 1000.0
        )

        route_packet_rates[vertex_id] = packets_per_ms

        for link in route["tree_links"]:
            source = tuple(link["source"])
            destination = tuple(link["destination"])
            name = directed_link_name(source, destination)
            link_packet_rates[name] += packets_per_ms

    link_metrics: dict[str, dict[str, Any]] = {}

    total_dropped_packets = 0.0
    overloaded_links = 0
    peak_utilization = 0.0
    maximum_queue = 0.0

    for name, arrival_rate in sorted(link_packet_rates.items()):
        utilization = arrival_rate / capacity_packets_per_ms

        excess_rate = max(
            0.0,
            arrival_rate - capacity_packets_per_ms,
        )
        generated_queue = excess_rate * duration_ms
        queued_packets = min(
            generated_queue,
            maximum_queue_packets,
        )
        dropped_packets = max(
            0.0,
            generated_queue - maximum_queue_packets,
        )

        queue_delay_ms = (
            queued_packets / capacity_packets_per_ms
            if capacity_packets_per_ms > 0
            else float("inf")
        )

        link_latency_ms = (
            propagation_ms
            + router_processing_ms
            + serialization_ms
            + queue_delay_ms
        )

        if utilization > 1.0:
            overloaded_links += 1

        total_dropped_packets += dropped_packets
        peak_utilization = max(peak_utilization, utilization)
        maximum_queue = max(maximum_queue, queued_packets)

        link_metrics[name] = {
            "packet_rate_per_ms": arrival_rate,
            "packet_rate_per_second": arrival_rate * 1000.0,
            "bandwidth_bytes_per_second": (
                arrival_rate * 1000.0 * packet_bytes
            ),
            "capacity_packets_per_ms": capacity_packets_per_ms,
            "utilization": utilization,
            "queued_packets": queued_packets,
            "dropped_packets": dropped_packets,
            "estimated_link_latency_ms": link_latency_ms,
        }

    # Calculate maximum source-to-destination latency for every tree.
    route_latencies: dict[str, float] = {}

    for route in routing["routes"]:
        source = tuple(route["source_chip"])
        graph = nx.DiGraph()
        graph.add_node(source)

        for link in route["tree_links"]:
            link_source = tuple(link["source"])
            link_destination = tuple(link["destination"])
            name = directed_link_name(
                link_source,
                link_destination,
            )

            graph.add_edge(
                link_source,
                link_destination,
                latency=link_metrics[name][
                    "estimated_link_latency_ms"
                ],
            )

        destination_latencies: list[float] = []

        for destination_data in route["destination_chips"]:
            destination = tuple(destination_data)

            if destination == source:
                destination_latencies.append(0.0)
                continue

            destination_latencies.append(
                nx.shortest_path_length(
                    graph,
                    source=source,
                    target=destination,
                    weight="latency",
                )
            )

        route_latencies[route["source_vertex"]] = max(
            destination_latencies,
            default=0.0,
        )

    maximum_route_latency_ms = max(
        route_latencies.values(),
        default=0.0,
    )

    average_route_latency_ms = (
        sum(route_latencies.values()) / len(route_latencies)
        if route_latencies
        else 0.0
    )

    total_packets_generated = sum(
        rate * duration_ms
        for rate in route_packet_rates.values()
    )

    total_link_transmissions = sum(
        rate * duration_ms
        for rate in link_packet_rates.values()
    )

    return {
        "duration_ms": duration_ms,
        "packet_bytes": packet_bytes,
        "link_capacity_packets_per_ms": (
            capacity_packets_per_ms
        ),
        "active_directed_links": len(link_metrics),
        "total_source_packets": total_packets_generated,
        "total_link_transmissions": total_link_transmissions,
        "peak_link_utilization": peak_utilization,
        "overloaded_links": overloaded_links,
        "maximum_queue_packets": maximum_queue,
        "total_dropped_packets": total_dropped_packets,
        "average_route_latency_ms": average_route_latency_ms,
        "maximum_route_latency_ms": maximum_route_latency_ms,
        "route_packet_rates_per_ms": route_packet_rates,
        "route_latency_ms": route_latencies,
        "links": link_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate routed spike-packet traffic"
    )
    parser.add_argument(
        "--config",
        default="configs/hardware.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/routing/link_simulation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = simulate_link_traffic(args.config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Link traffic simulation complete")
    print(
        f"Active directed links: "
        f"{result['active_directed_links']}"
    )
    print(
        f"Total source packets: "
        f"{result['total_source_packets']:.2f}"
    )
    print(
        f"Total link transmissions: "
        f"{result['total_link_transmissions']:.2f}"
    )
    print(
        f"Peak link utilization: "
        f"{100.0 * result['peak_link_utilization']:.2f}%"
    )
    print(f"Overloaded links: {result['overloaded_links']}")
    print(
        f"Maximum queue: "
        f"{result['maximum_queue_packets']:.2f} packets"
    )
    print(
        f"Dropped packets: "
        f"{result['total_dropped_packets']:.2f}"
    )
    print(
        f"Average route latency: "
        f"{result['average_route_latency_ms']:.4f} ms"
    )
    print(
        f"Maximum route latency: "
        f"{result['maximum_route_latency_ms']:.4f} ms"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()