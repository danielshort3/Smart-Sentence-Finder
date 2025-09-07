# syntax=docker/dockerfile:1.7-labs

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS runtime

# Avoid prompts from any apt operations
ENV DEBIAN_FRONTEND=noninteractive

# System deps commonly needed by tokenizers, sentence-transformers, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure model caches to a shared mount point
ENV HF_HOME=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

# Create cache directories (will be bind-mounted from host for persistence)
RUN mkdir -p "$HF_HOME/hub" "$SENTENCE_TRANSFORMERS_HOME"

# Install Python deps (torch provided by base image)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts
COPY README.md ./README.md

ENV PYTHONPATH=/app/src

# Default command prints CLI help
ENTRYPOINT ["python", "-m", "smart_sentence_finder.cli"]
