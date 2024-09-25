#!/bin/bash

set -euo pipefail

num_runs=15

# Service configurations using parallel arrays
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

# Initialize arrays for storing times
declare -a times_onnx times_torch times_rust

echo "Measuring container start times for $num_runs runs..."
for i in $(seq 1 "$num_runs"); do
    echo "Run $i..."
    docker compose down >/dev/null 2>&1

    start_time=$(python -c 'import time; print("{0:.6f}".format(time.time()))')
    docker compose up -d >/dev/null 2>&1

    # Initialize ready status
    ready_onnx=0
    ready_torch=0
    ready_rust=0

    # Wait for all containers to be ready
    while true; do
        current_time=$(python -c 'import time; print("{0:.6f}".format(time.time()))')

        all_ready=true

        # Check onnx-serving
        if [ "$ready_onnx" -eq 0 ]; then
            if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${ports[0]}/" | grep -q 200; then
                duration=$(echo "scale=6; $current_time - $start_time" | bc)
                times_onnx+=("$duration")
                echo "${containers[0]} ready in $duration seconds"
                ready_onnx=1
            else
                all_ready=false
            fi
        fi

        # Check torch-serving
        if [ "$ready_torch" -eq 0 ]; then
            if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${ports[1]}/" | grep -q 200; then
                duration=$(echo "scale=6; $current_time - $start_time" | bc)
                times_torch+=("$duration")
                echo "${containers[1]} ready in $duration seconds"
                ready_torch=1
            else
                all_ready=false
            fi
        fi

        # Check rust-onnx-serving
        if [ "$ready_rust" -eq 0 ]; then
            if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${ports[2]}/" | grep -q 200; then
                duration=$(echo "scale=6; $current_time - $start_time" | bc)
                times_rust+=("$duration")
                echo "${containers[2]} ready in $duration seconds"
                ready_rust=1
            else
                all_ready=false
            fi
        fi

        if $all_ready; then
            break
        fi
        sleep 0.01
    done

    docker compose down >/dev/null 2>&1
    sleep 1
done

# Convert times to JSON using jq
json_object=$(jq -n \
    --arg svc1 "${containers[0]}" --arg svc2 "${containers[1]}" --arg svc3 "${containers[2]}" \
    --arg times1 "${times_onnx[*]}" --arg times2 "${times_torch[*]}" --arg times3 "${times_rust[*]}" \
    '{($svc1): ($times1 | split(" ") | map(tonumber)),
      ($svc2): ($times2 | split(" ") | map(tonumber)),
      ($svc3): ($times3 | split(" ") | map(tonumber))}')

echo "$json_object" > start_times.json
echo "Start times saved to start_times.json"