# model-serving

This repository demonstrates various model serving strategies, from basic PyTorch implementations to ONNX Runtime in both Python and Rust.
Each approach is benchmarked to compare performance under standardized conditions. The goal of this project is to illustrate how different
serving strategies influence model service load capacity and to evaluate secondary metrics, including image size and container startup time.

An extensive write-up with detailed explanations about the technologies used and approaches taken can be found 
[here](https://martynassubonis.substack.com/p/optimize-for-speed-and-savings-high).

For eager readers - please refer to [Benchmark Setup](#benchmark-setup), [Benchmark Results](#benchmark-results) and [Conclusions](#conclusions) directly.

## Table of Contents

- [Pre-requisites](#pre-requisites)
-
    - [Software Requirements](#software-requirements)
-
    - [Domain-Specific Requirements](#domain-specific-requirements)
- [Served Model Context](#served-model-context)
- [Applications Context](#applications-context)
- [Running Applications](#running-applications)
- [Benchmark Context](#benchmark-context)
-
    - [Host System](#host-system)
-
    - [Containers](#containers)
-
    - [Benchmark Setup](#benchmark-setup)
- [Benchmark Results](#benchmark-results)
-
    - [Performance Metrics](#performance-metrics)
-
    - [Deployment Metrics](#deployment-metrics)
- [Conclusions](#conclusions)

## Pre-requisites

This project builds on top of [ml-workflows](https://github.com/martynas-subonis/ml-workflows), and assumes that the following criteria are
met:

### Software Requirements

- [python 3.12](https://www.python.org/downloads/)
- [docker](https://docs.docker.com/engine/install/)
- [gcloud](https://cloud.google.com/sdk/docs/install)
- [poetry](https://python-poetry.org/docs/#installation)
- [apache benchmark (ab)](https://httpd.apache.org/docs/2.4/programs/ab.html)

### Domain-Specific Requirements

- The [ml-workflows](https://github.com/martynas-subonis/ml-workflows) pipeline was executed successfully.
- The models `torch_model` and `optimized_onnx_with_transform_model` remain accessible
  in [Google Cloud Storage](https://cloud.google.com/storage) (GCS) buckets.
- Local `.env` file is created, with the following values set:

```.text
CLOUDSDK_AUTH_ACCESS_TOKEN=... # Run "gcloud auth application-default print-access-token" to get this value. Keep in mind this token is valid only for a specific period.
GOOGLE_CLOUD_PROJECT=... # Your GCP project.
ONNX_MODEL_URI=gs://... # URI of optimized ONNX model with tranformation layer.
TORCH_MODEL_URI=gs://... # URI of torch model state dict.
```

## Served Model Context

This project serves the small variant of [MobileNetV3](https://arxiv.org/abs/1905.02244), a lightweight model with only 1.53M parameters.

The model was fine-tuned for weather image classification using
the ["Weather Image Recognition"](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) dataset as part of
the [ml-workflows](https://github.com/martynas-subonis/ml-workflows) project. For fine-tuning, only the model head was modified and trained
while all other layers remained frozen. The model was trained for 10 epochs using an 80/10/10 train/validation/test split.

For more detailed information, please refer to [ml-workflows](https://github.com/martynas-subonis/ml-workflows) repository.

## Applications Context

Three different approaches are benchmarked in this project:

- Naive Model Serving with [PyTorch](https://pytorch.org/docs/stable/index.html) and [FastAPI](https://fastapi.tiangolo.com/) (Python): This setup uses PyTorch
  with `model.eval()` and `torch.inference_mode()` enabled. No ONNX or ONNX Runtime optimizations are applied; instead, we serve the model directly from its
  saved `state_dict` after training. Although this approach is less optimized, it remains common in practice, with Flask or Django being possible alternatives
  to FastAPI, making it a valuable baseline for our benchmarks. Please see [torch_serving](torch_serving).

- Optimized Model Serving with [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python.html) and FastAPI (Python): In this approach, we leverage ONNX
  Runtime for serving. Input transformation logic is embedded directly into the model’s computation
  graph, [and graph optimizations are applied offline](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode),
  providing a more efficient alternative to the naive approach. Please see [onnx_serving](onnx_serving).

- Optimized Model Serving with ONNX Runtime and [Actix-Web](https://actix.rs/docs/whatis) (Rust): Here, we use a Rust-based setup with ONNX
  Runtime ([built from source](https://onnxruntime.ai/docs/build/inferencing.html) and utilizing the [pykeio/ort wrapper](https://github.com/pykeio/ort)) and
  Actix-Web for serving. Like the previous setup, input transformation logic is embedded in the model graph, and offline graph optimizations are applied, aiming
  for maximum performance. Please see [rust_onnx_serving](rust_onnx_serving).

## Running Applications

To start the applications locally, simply run:

```bash
# Unlike --env-file argument, the first command exports env variables in a manner that works with docker compose secrets.
export $(grep -v '^#' .env | xargs) && docker compose up --build
```

This process may take some time on the first run, especially for the Docker image of the Rust application.

## Benchmark Context

**When interpreting benchmark results, avoid treating them as universally applicable values, as absolute performance can vary significantly with different
hardware, operating systems (OS), and C standard library implementations (e.g., glibc or musl), which affect the Application Binary Interface (ABI).**

Furthermore, performance metrics can differ based on the sizes of the input images; therefore, in a production environment, it would be important to
understand the distribution of image sizes. For the purposes of this exercise, the focus should be on the **relative performance differences** between different
serving strategies.

The most reliable way to assess model service performance on a specific host machine is to conduct direct testing in that environment.

### Host System

- Hardware: Apple M2 Max
- OS: macOS 15.0.1
- Docker:
-
    - Engine v27.2.0
-
    - Desktop 4.34.3

### Containers

- **CPU Allocation**: Each container was allocated 4 CPU cores.

- **Memory Allocation**: Memory was allocated dynamically, providing each container with as much memory as it required.

- **Worker and Thread Configuration**: To maximize CPU utilization and ensure that each container fully utilized its allocated CPU
  resources (achieving 400% CPU usage corresponding to 4 CPU cores), the following configurations were implemented:
    - **`onnx_serving`**:
        - **Uvicorn Workers**: 4
        - **ONNX Runtime Session Threads**:
            - **[Intra-Op Threads](https://onnxruntime.ai/docs/performance/tune-performance/threading.html#set-intra-op-thread-affinity)**: 1
            - **[Inter-Op Threads](https://onnxruntime.ai/docs/performance/tune-performance/threading.html#set-number-of-inter-op-threads)**: 1

    - **`torch_serving`**:
        - **Uvicorn Workers**: 4

    - **`rust_onnx_serving`**:
        - **Actix Web Workers**: 4
        - **ONNX Runtime Session Threads**:
            - **Intra-Op Threads**: 3
            - **Inter-Op Threads**: 1

### Benchmark Setup

- Benchmarking tool: [apache benchmark](https://httpd.apache.org/docs/2.4/programs/ab.html).

```bash
ab -n 40000 -c 50 -p images/rime_5868.json -T 'application/json' -s 3600 "http://localhost:$port/predict/"
```

-
    - `-n 40000`: total of 40000 requests.
-
    - `-c 50`: concurrency of 50.
- Payload image: `images/rime_5868.json`:

![Rime Image](images/rime_5868.jpg)

-
    - Original size: **393 KB**.
-
    - Payload size after [PIL](https://pillow.readthedocs.io/en/stable/) compression and base64 encoding (~33% increase): **304 KB**.

## Benchmark Results

### Performance Metrics

| **Metric**                                                | **torch-serving** | **onnx-serving** | **rust-onnx-serving** |
|-----------------------------------------------------------|-------------------|------------------|-----------------------|
| **Time taken for tests (seconds)**                        | 1122.988          | 156.538          | 121.604               |
| **Requests per second (mean)**                            | 35.62             | 255.53           | 328.94                |
| **Time per request (ms)**                                 | 1403.734          | 195.672          | 152.005               |
| **Time per request (ms, across all concurrent requests)** | 28.075            | 3.913            | 3.040                 |
| **Transfer rate (MB/s)**                                  | 10.54             | 75.58            | 97.28                 |
| **Memory Usage (MB, mean)**                               | 921.46            | 359.12           | 687.6                 |

### Deployment Metrics

| **Metric**                                | **torch-serving** | **onnx-serving** | **rust-onnx-serving** |
|-------------------------------------------|-------------------|------------------|-----------------------|
| **Docker image size (MB)**                | 650               | 296              | 48.3                  |
| **Application start time (s, n=15 mean)** | 4.342             | 1.396            | 0.348                 |

## Conclusions

- **ONNX Runtime Significantly Improves Performance:** Converting models to ONNX and serving them with ONNX Runtime greatly enhances throughput and reduces
  latency compared to serving with PyTorch. Specifically:
    - **`onnx-serving`** (Python) handles approximately **7.18 times** more requests per second than **`torch-serving`** (255.53 vs. 35.62 requests/sec).
    - **`rust-onnx-serving`** (Rust) achieves about **9.23 times** higher throughput than **`torch-serving`** (328.94 vs. 35.62 requests/sec).

- **Rust Implementation Delivers Highest Performance:** Despite higher memory usage than Python ONNX serving, the Rust implementation offers higher performance
  and advantages in deployment size and startup time:
    - **Throughput:** **`rust-onnx-serving`** is about **1.29 times** faster than **`onnx-serving`** (328.94 vs. 255.53 requests/sec).
    - **Startup Time:** Rust application starts in **0.348 seconds**, which is over **12 times faster** than **`torch-serving`** (4.342 seconds) and about **4
      times faster** than **`onnx-serving`** (1.396 seconds).
    - **Docker Image Size:** Rust image size is **48.3 MB**, which is approximately **13 times smaller** than **`torch-serving`** (650 MB) and about **6 times
      smaller** than **`onnx-serving`** (296 MB).

- **Memory Usage Difference:** The higher memory usage in Rust compared to Python ONNX serving stems from differences in
  implementations and libraries used:
    - **Image Processing Differences:** The Rust implementation uses less optimized image processing compared to Python's PIL and NumPy libraries,
      potentially leading to higher memory consumption.
    - **Library Efficiency:** The Rust `ort` crate is an unofficial wrapper and might manage memory differently compared to the official ONNX Runtime SDK
      for Python, which is mature and highly optimized.
    - **Threading Configuration:** The Rust implementation uses more intra-threads, which contributes to some additional memory consumption. However, this
      accounts for only a minor portion of the overall difference observed.

The last memory point is just a consequence of a more important factor: Python’s mature and extensive ecosystem for machine learning. Rewriting these
serving strategies in Rust can introduce challenges, such as increased development effort, potential performance trade-offs where optimized crates are
unavailable (or one has to write them), and added complexity. However, Rust's benefits may sometimes justify the effort, depending on specific business needs.

Using inference-optimized solutions like ONNX Runtime can significantly enhance model serving performance, especially for larger models. While this article
uses a small model (MobileNet V3-small, ~1.53 million parameters), the benefits of ONNX Runtime become more pronounced with more complex architectures. Its
ability to optimize computation graphs and streamline resource usage leads to higher throughput and reduced latency, making it invaluable for scaling
model-serving solutions.
