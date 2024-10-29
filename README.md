# model-serving

This repository demonstrates various model serving strategies, from basic PyTorch implementations to ONNX Runtime in both Python and Rust.
Each approach is benchmarked to compare performance under standardized conditions. The goal of this project is to illustrate how different
serving strategies influence model service load capacity and to evaluate secondary metrics, including image size and container startup time.

## Table of Contents

- [Pre-requisites](#pre-requisites)
-
    - [Software Requirements](#software-requirements)
-
    - [Domain-Specific Requirements](#domain-specific-requirements)
- [Served Model Context](#served-model-context)
- [Applications Context](#applications-context)
- [Benchmark Context](#benchmark-context)
- [Running Applications](#running-applications)

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

- Naive model serving using [PyTorch](https://pytorch.org/docs/stable/index.html) and [FastAPI](https://fastapi.tiangolo.com/) (Python),
  with `model.eval()` and `torch.inference_mode()`. Please see [torch_serving](torch_serving).
- Model serving using [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python.html)
  and [FastAPI](https://fastapi.tiangolo.com/) (Python), where input transformation logic was integrated into the graph
  and [graph optimizations were applied offline](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode).
  Please see [onnx_serving](onnx_serving).
- Model serving using [ONNX Runtime](https://onnxruntime.ai/docs/build/inferencing.html) (built from source and
  using [pykeio/ort](https://github.com/pykeio/ort) wrapper) and [actix-web](https://actix.rs/docs/whatis) (Rust), where input
  transformation logic was integrated into the graph
  and [graph optimizations were applied offline](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode).
  Please see [rust_onnx_serving](rust_onnx_serving).

## Benchmark Context

**IMPORTANT**: When interpreting benchmark results, it is important to avoid treating them as absolute values. The absolute performance can
vary significantly depending on the hardware, operating system (OS),
and [Application Binary Interface (ABI)](https://en.wikipedia.org/wiki/Application_binary_interface) (
e.g., [GNU](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html) or [MUSL](https://wiki.musl-libc.org/abi-manuals)) on which the model
is deployed.

Furthermore, performance metrics can differ based on the sizes of the input images; therefore, in a production environment, it is crucial to
understand the distribution of image sizes. For the purposes of this exercise, the focus should be on the **relative performance differences
** between different serving strategies.

To accurately assess how a model service will perform on a specific host machine, the only reliable method is to conduct direct testing.

### Host System

- Hardware: Apple M2 Max
- OS: macOS 15.0.1
- Docker:
-
    - Engine v27.2.0
-
    - Desktop 4.34.3

### Benchmark Setup

- Benchmarking tool: [apache benchmark](https://httpd.apache.org/docs/2.4/programs/ab.html).

```bash
ab -n 5000 -c 50 -p images/rime_5868.json -T 'application/json' -s 3600 "http://localhost:$port/predict/"
```

-
    - `-n 5000`: total of 5000 requests.
-
    - `-c 50`: concurrency of 50.
- Payload image: `images/rime_5868.json`:
  ![Rime Image](images/rime_5868.jpg)
-
    - Original size: **393 KB**.
-
    - Payload size after PIL compression and base64 encoding (~33% increase): **304 KB**.

## Running Applications

To start the applications locally, simply run:

```bash
# Unlike --env-file argument, the first command exports env variables in a manner that works with docker compose secrets.
export $(grep -v '^#' .env | xargs) && docker compose up --build
```

This process may take some time on the first run, especially for the Docker image of the Rust application.
