# syntax=docker/dockerfile:1.10.0

FROM rust:1.82 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    git \
    libc6-dev \
    libprotobuf-dev \
    libssl-dev \
    linux-headers-generic \
    pkg-config \
    protobuf-compiler \
    wget && \
    rustup target add aarch64-unknown-linux-gnu && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-aarch64.sh && \
    chmod +x cmake-3.30.5-linux-aarch64.sh && \
    ./cmake-3.30.5-linux-aarch64.sh --prefix=/usr/local --skip-license &&  \
    git clone --recursive https://github.com/Microsoft/onnxruntime.git && \
    cd ./onnxruntime &&  \
    ./build.sh \
        --allow_running_as_root \
        --arm64 \
        --build \
        --clean \
        --compile_no_warning_as_error \
        --config Release \
        --parallel \
        --skip_submodule_sync \
        --skip_tests \
        --update && \
    cd ..

ENV ORT_STRATEGY=system \
    ORT_LIB_LOCATION=/app/onnxruntime/build/Linux/Release

COPY Cargo.toml Cargo.lock main.rs ./
RUN cargo build --release --target aarch64-unknown-linux-gnu

FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:493.0.0-alpine AS model-downloader

WORKDIR /app

RUN --mount=type=secret,id=onnx_model_uri,env=MODEL_URI \
    --mount=type=secret,id=access_token,env=CLOUDSDK_AUTH_ACCESS_TOKEN \
    gcloud storage cp "$MODEL_URI" "model.onnx"

FROM gcr.io/distroless/static-debian12

WORKDIR /app
COPY --from=builder /app/target/aarch64-unknown-linux-gnu/release/rust_onnx_serving .
COPY --from=model-downloader /app/model.onnx .

COPY --from=builder /lib/aarch64-linux-gnu/libgcc_s.so.1 /lib/
COPY --from=builder "/lib/aarch64-linux-gnu/libstdc++.so.6" /lib/
COPY --from=builder /lib/aarch64-linux-gnu/libssl.so.3 /lib/
COPY --from=builder /lib/aarch64-linux-gnu/libcrypto.so.3 /lib/
COPY --from=builder /lib/aarch64-linux-gnu/libm.so.6 /lib/
COPY --from=builder /lib/aarch64-linux-gnu/libc.so.6 /lib/
COPY --from=builder /lib/ld-linux-aarch64.so.1 /lib/

EXPOSE 8082
ENV RUST_LOG=info
CMD ["./rust_onnx_serving"]
