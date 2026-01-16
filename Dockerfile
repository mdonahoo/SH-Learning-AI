# ==============================================================================
# Starship Horizons Learning AI - Production Dockerfile
#
# Multi-stage build with:
# - NVIDIA CUDA for GPU acceleration
# - Whisper for audio transcription
# - Ollama for LLM-powered reports
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Python dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Production image with CUDA + Ollama
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Application configuration
ENV WEB_SERVER_HOST=0.0.0.0 \
    WEB_SERVER_PORT=8000 \
    PRELOAD_WHISPER=true \
    WHISPER_MODEL=medium \
    WEB_MAX_UPLOAD_MB=100

# Ollama configuration
ENV OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODEL=llama3.2 \
    OLLAMA_TIMEOUT=120

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    # Networking
    curl \
    wget \
    # Process management
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy pre-built wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY configs/ ./configs/ 2>/dev/null || true
COPY .env.example ./.env.example
COPY deploy/docker-entrypoint.sh /entrypoint.sh

# Create directories and set permissions
RUN mkdir -p /app/data /app/logs /app/models /root/.ollama \
    && chmod +x /entrypoint.sh

# Pre-download Whisper model (cached in image)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')" \
    || echo "Whisper model will be downloaded on first use"

# Expose ports
# 8000 = Web server
# 11434 = Ollama API (internal)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Use entrypoint script to start Ollama + Web server
ENTRYPOINT ["/entrypoint.sh"]
