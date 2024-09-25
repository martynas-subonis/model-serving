import json
import re
from collections import defaultdict
from datetime import datetime
from os.path import exists
from statistics import mean, median, stdev
from typing import Any, DefaultDict, List

SPAN_FILES = ["torch_serving_spans.jsonl", "onnx_serving_spans.jsonl", "rust_onnx_serving_spans.jsonl"]
SERVICES_NAMES = ["Torch Serving", "Onnx Serving", "Rust Onnx Serving"]
CONTAINER_NAME_MAPPING = {
    "torch-serving-container": SERVICES_NAMES[0],
    "onnx-serving-container": SERVICES_NAMES[1],
    "rust-onnx-serving-container": SERVICES_NAMES[2],
}
METRICS_ORDER = [
    "async-downloading-image",
    "preprocessing-image",
    "model-inference",
    "preprocessing-and-model-inference",
    "CPU Usage",
    "Memory Usage",
    "Container Start Time",
]

from prettytable import PrettyTable


def compute_percentile(data: List[float], percentile: float) -> float | None:
    if not data:
        return None
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_data) - 1)
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def process_span_data(span_file: str, stats_data: dict[str, Any]) -> None:
    if not exists(span_file):
        print(f"{span_file} does not exist. Skipping.")
        return
    durations = defaultdict(list)
    with open(span_file, "r") as f:
        for line in f:
            span = json.loads(line)
            duration_ms = (
                datetime.fromisoformat(span["end_time"].rstrip("Z")) - datetime.fromisoformat(span["start_time"].rstrip("Z"))
            ).total_seconds() * 1000
            durations[span["name"]].append(duration_ms)

    service_name = " ".join(part.capitalize() for part in span_file.replace("_spans.jsonl", "").split("_"))

    for metric_label, values in durations.items():
        n = len(values)
        stats_data[metric_label][service_name] = {
            "mean": mean(values),
            "std_dev": stdev(values) if n > 1 else 0.0,
            "median": median(values),
            "p95": compute_percentile(values, 95),
            "p99": compute_percentile(values, 99),
            "n": n,
            "units": "ms",
        }


def process_container_stats(stats_data: dict[str, Any]) -> None:
    if not exists("benchmark_stats.log"):
        print("container_stats.log does not exist.")
        return

    cpu_usage, mem_usage = defaultdict(list), defaultdict(list)
    with open("benchmark_stats.log", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Timestamp:") or line.startswith("NAME"):
                continue

            parts = line.split()
            if len(parts) < 3:
                print(f"Unexpected format in line: {line}")
                continue

            container_name, cpu_percent_str, mem_usage_str = parts[0], parts[1], parts[2]
            cpu_percent = float(cpu_percent_str.strip("%"))

            mem_match = re.match(r"([0-9.]+)([A-Za-z]+)", mem_usage_str)
            if not mem_match:
                print(f"Could not parse memory usage in line: {line}")
                continue

            mem_value, mem_unit = mem_match.groups()
            mem_value = float(mem_value)
            unit_factors = {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024, "B": 1 / (1024 * 1024)}
            mem_usage_value = mem_value * unit_factors.get(mem_unit, 0)

            cpu_usage[container_name].append(cpu_percent)
            mem_usage[container_name].append(mem_usage_value)

    for container_name, service_name in CONTAINER_NAME_MAPPING.items():
        if container_name in cpu_usage:
            for metric_name, values, units in [
                ("CPU Usage", cpu_usage[container_name], "%"),
                ("Memory Usage", mem_usage[container_name], "MiB"),
            ]:
                n = len(values)
                stats_data[metric_name][service_name] = {
                    "mean": mean(values),
                    "std_dev": stdev(values) if n > 1 else 0.0,
                    "median": median(values),
                    "p95": compute_percentile(values, 95),
                    "p99": compute_percentile(values, 99),
                    "n": n,
                    "units": units,
                }


def process_start_times(stats_data: dict[str, Any]) -> None:
    if not exists("start_times.json"):
        print("start_times.json does not exist.")
        return

    with open("start_times.json", "r") as f:
        start_times = json.load(f)

    for container_name, times in start_times.items():
        service_name = CONTAINER_NAME_MAPPING[container_name]
        n = len(times)
        if n == 0:
            print(f"No data to compute statistics for {service_name}.")
            continue

        stats_data["Container Start Time"][service_name] = {
            "mean": mean(times),
            "std_dev": stdev(times) if n > 1 else 0.0,
            "median": median(times),
            "p95": compute_percentile(times, 95),
            "p99": compute_percentile(times, 99),
            "n": n,
            "units": "s",
        }


def main() -> None:
    stats_data: DefaultDict[str, Any] = defaultdict(dict)
    for span_file in SPAN_FILES:
        process_span_data(span_file, stats_data)

    process_container_stats(stats_data)
    process_start_times(stats_data)

    table = PrettyTable()
    table.field_names = ["Performance Indicator", "Stat"] + SERVICES_NAMES
    table.align = "l"
    for metric in METRICS_ORDER:
        metric_data = stats_data.get(metric, {})
        n_values = {service: metric_data.get(service, {}).get("n") for service in SERVICES_NAMES if metric_data.get(service)}
        n_str = f"n={list(n_values.values())[0]}" if n_values else ""
        metric_label = f"{metric} ({n_str})" if n_str else metric
        stats_list = ["Average", "Median", "P95", "P99"]
        for idx, stat in enumerate(stats_list):
            row = [metric_label if idx == 0 else "", stat]
            for service in SERVICES_NAMES:
                service_stats = metric_data.get(service)
                if service_stats:
                    units = service_stats["units"]
                    if stat == "Average":
                        mean_val = service_stats["mean"]
                        std_dev = service_stats["std_dev"]
                        value = f"{mean_val:.2f} ± {std_dev:.2f} {units}"
                    else:
                        value = f"{service_stats[stat.lower()]:.2f} {units}"
                else:
                    value = ""
                row.append(value)
            table.add_row(row)
    print(table)


if __name__ == "__main__":
    main()
