#!/bin/bash

set -euo pipefail

containers=("torch-serving-container" "onnx-serving-container" "rust-onnx-serving-container")
output_files=("torch_serving_spans.jsonl" "onnx_serving_spans.jsonl" "rust_onnx_serving_spans.jsonl")
num_containers=${#containers[@]}

for (( i=0; i < num_containers; i++ )); do
    container="${containers[$i]}"
    output_file="${output_files[$i]}"
    echo "Processing logs from $container and saving to $output_file..."

    # Truncate the output file to start fresh
    : > "$output_file"

    # Process everything in a single pipeline with error handling
    (docker logs "$container" 2>/dev/null || true) | \
        (sed -n '/^{/,/^}/p' || true) | \
        (tr -d '\r' || true) | \
        (jq -c --arg container "$container" \
            'select(
                .name != null and (
                    .name == "async-downloading-image" or
                    .name == "preprocessing-and-model-inference" or
                    .name == "preprocessing-image" or
                    .name == "model-inference"
                )
            ) | .container = $container' 2>/dev/null || true) >> "$output_file"

    # Check if the output file was created and has content
    if [ ! -s "$output_file" ]; then
        echo "Warning: No valid data was processed for $container"
    fi
done
echo "Span export completed."