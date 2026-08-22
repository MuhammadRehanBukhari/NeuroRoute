"""Tests for link bandwidth, queue, and timing simulation."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.neuroroute.link_sim import simulate_link_traffic


def load_hardware_config() -> dict:
    with Path("configs/hardware.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def save_config(path: Path, config: dict) -> None:
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def test_nominal_traffic_has_no_packet_loss() -> None:
    result = simulate_link_traffic("configs/hardware.yaml")

    assert result["active_directed_links"] > 0
    assert result["total_source_packets"] > 0
    assert result["total_link_transmissions"] > 0
    assert result["peak_link_utilization"] < 1.0
    assert result["overloaded_links"] == 0
    assert result["maximum_queue_packets"] == 0
    assert result["total_dropped_packets"] == 0


def test_link_bandwidth_calculation_is_consistent() -> None:
    result = simulate_link_traffic("configs/hardware.yaml")
    packet_bytes = result["packet_bytes"]

    for metrics in result["links"].values():
        expected_bandwidth = (
            metrics["packet_rate_per_second"] * packet_bytes
        )

        assert metrics["bandwidth_bytes_per_second"] == pytest.approx(
            expected_bandwidth
        )


def test_low_capacity_produces_queueing_and_drops(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_hardware_config())

    config["hardware"]["links"][
        "bandwidth_packets_per_ms"
    ] = 1.0
    config["traffic"]["queue"]["maximum_packets"] = 10

    config_path = tmp_path / "overloaded.yaml"
    save_config(config_path, config)

    result = simulate_link_traffic(config_path)

    assert result["peak_link_utilization"] > 1.0
    assert result["overloaded_links"] > 0
    assert result["maximum_queue_packets"] == 10
    assert result["total_dropped_packets"] > 0


def test_more_per_hop_latency_increases_route_latency(
    tmp_path: Path,
) -> None:
    fast_config = deepcopy(load_hardware_config())
    slow_config = deepcopy(load_hardware_config())

    fast_config["hardware"]["links"]["latency_ms_per_hop"] = 0.1
    slow_config["hardware"]["links"]["latency_ms_per_hop"] = 0.2

    fast_path = tmp_path / "fast.yaml"
    slow_path = tmp_path / "slow.yaml"

    save_config(fast_path, fast_config)
    save_config(slow_path, slow_config)

    fast_result = simulate_link_traffic(fast_path)
    slow_result = simulate_link_traffic(slow_path)

    assert (
        slow_result["maximum_route_latency_ms"]
        > fast_result["maximum_route_latency_ms"]
    )


def test_route_latency_is_nonnegative() -> None:
    result = simulate_link_traffic("configs/hardware.yaml")

    assert result["average_route_latency_ms"] >= 0
    assert result["maximum_route_latency_ms"] >= 0
    assert (
        result["maximum_route_latency_ms"]
        >= result["average_route_latency_ms"]
    )