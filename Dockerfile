# Multi-stage build: Rust binary + Python sidecar for librosa

# Stage 1: Build Rust binary
FROM rust:1.82-bookworm AS builder

# Install libtorch dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Create a Python venv with torch for building (tch crate needs it)
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

ENV LIBTORCH_USE_PYTORCH=1
ENV PATH="/opt/venv/bin:$PATH"

RUN cargo build --release

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# System dependencies for librosa and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps for sidecar (librosa only, minimal)
COPY requirements-sidecar.txt .
RUN pip install --no-cache-dir -r requirements-sidecar.txt

# Copy libtorch from builder's venv
COPY --from=builder /opt/venv/lib/python3.12/site-packages/torch /opt/torch
ENV LD_LIBRARY_PATH="/opt/torch/lib:$LD_LIBRARY_PATH"

# Copy Rust binary
COPY --from=builder /app/target/release/mertalizer /usr/local/bin/mertalizer

# Copy application files
COPY templates/ templates/
COPY static/ static/
COPY models/mert_encoder.pt models/mert_encoder.pt
COPY models/mert_encoder_config.json models/mert_encoder_config.json
COPY models/mertalizer_traced.pt models/mertalizer_traced.pt

# Create data directory
RUN mkdir -p data/history/audio

ENV PORT=3000
ENV PYTHON_BIN=python3
EXPOSE 3000

CMD ["mertalizer"]
