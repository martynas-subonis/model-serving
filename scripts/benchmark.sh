#!/bin/bash

# Configuration
stats_log="benchmark_stats.log"
monitoring_interval=0.01

# Define containers and their ports as parallel arrays
containers=(
    "onnx-serving-container"
    "torch-serving-container"
    "rust-onnx-serving-container"
)
ports=(
    "8080"
    "8081"
    "8082"
)

# Function to collect stats for a specific container
collect_stats() {
    local container=$1
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    {
        echo "Timestamp: $timestamp"
        docker stats --no-stream \
            --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
            "$container"
    } >> "$stats_log"
}

# Cleanup function
cleanup() {
    # Kill the monitoring process if it exists
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
    fi
}

trap cleanup EXIT

: > "$stats_log"

# Run benchmarks for each container
for i in "${!containers[@]}"; do
    container="${containers[$i]}"
    port="${ports[$i]}"

    echo "Testing $container on port $port..."
    {
        while true; do
            collect_stats "$container"
            sleep "$monitoring_interval"
        done
    } >/dev/null 2>&1 &
    MONITOR_PID=$!

    # Run ab and capture its exit code
    ab -n 5000 -c 50 -p images/rime_5868.json -T 'application/json' -s 3600 "http://localhost:$port/predict/"
    AB_EXIT_CODE=$?

    # Kill and wait for the monitoring process
    cleanup

    # If ab failed, exit with its code
    if [ $AB_EXIT_CODE -ne 0 ]; then
        exit $AB_EXIT_CODE
    fi
done

exit 0