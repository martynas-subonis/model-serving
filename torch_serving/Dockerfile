# syntax=docker/dockerfile:1.10.0

FROM python:3.12.5-slim AS venv-builder

RUN pip install --upgrade pip==24.2.0 && \
    pip install poetry==1.8.3

WORKDIR /app

COPY pyproject.toml poetry.toml poetry.lock ./

RUN poetry install --only main

FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:493.0.0-alpine AS model-downloader

WORKDIR /app

RUN --mount=type=secret,id=torch_model_uri,env=MODEL_URI \
    --mount=type=secret,id=access_token,env=CLOUDSDK_AUTH_ACCESS_TOKEN \
    gcloud storage cp "$MODEL_URI" "model.pth"

FROM python:3.12.5-slim AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY src src
COPY main.py .

EXPOSE 8081

COPY --from=venv-builder /app/.venv .venv
COPY --from=model-downloader /app/"model.pth" .

CMD uvicorn main:app \
    --host 0.0.0.0 \
    --port 8081 \
    --workers 4 \
    --limit-concurrency 2000 \
    --limit-max-requests 40000 \
    --timeout-keep-alive 480 \
    --timeout-graceful-shutdown 480 \
    --log-level warning
