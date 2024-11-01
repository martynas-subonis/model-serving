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
declare -a times
declare -a ready

# Function to check if a container is ready
check_container_ready() {
    local container_idx=$1
    local container=${containers[$container_idx]}
    local port=${ports[$container_idx]}

    if [ "${ready[$container_idx]}" -eq 0 ]; then
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/" | grep -q 200; then
            local current_time=$(python -c 'import time; print("{0:.6f}".format(time.time()))')
            local duration=$(echo "scale=6; $current_time - $start_time" | bc)
            times[$container_idx]+="$duration "
            echo "$container ready in $duration seconds"
            ready[$container_idx]=1
            return 0
        fi
        return 1
    fi
    return 0
}

echo "Measuring container start times for $num_runs runs..."
for run in $(seq 1 "$num_runs"); do
    echo "Run $run..."
    docker compose down >/dev/null 2>&1

    # Reset ready status for each container
    for i in "${!containers[@]}"; do
        ready[$i]=0
    done

    start_time=$(python -c 'import time; print("{0:.6f}".format(time.time()))')
    docker compose up -d >/dev/null 2>&1

    # Wait for all containers to be ready
    while true; do
        all_ready=true

        # Check each container
        for i in "${!containers[@]}"; do
            if ! check_container_ready "$i"; then
                all_ready=false
            fi
        done

        if $all_ready; then
            break
        fi
        sleep 0.01
    done

    docker compose down >/dev/null 2>&1
done

echo -e "\nResults:"
echo "----------------------------------------"

for i in "${!containers[@]}"; do
    values=${times[$i]}
    avg=$(python3 -c "nums = [float(x) for x in '$values'.strip().split()]; print('{:.3f}'.format(sum(nums)/len(nums)))")
    echo "${containers[$i]}: $avg seconds average (over $num_runs runs)"
done
