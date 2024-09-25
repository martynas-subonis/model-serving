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

# Create payload file
echo '{"bucketName": "weather_imgs", "imagePath": "rime/5868.jpg"}' > payload.json

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

    ab -n 2000 -c 50 -p payload.json -T 'application/json' -s 3600 "http://localhost:$port/predict/"

    { kill $MONITOR_PID && wait $MONITOR_PID; } 2>/dev/null
done

# Clean up
rm payload.json