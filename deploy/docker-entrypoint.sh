#!/bin/bash
# ==============================================================================
# Docker Entrypoint Script
#
# Starts Ollama server and pulls models before starting the web application.
# ==============================================================================

set -e

# Colors for logging
log_info() { echo "[INFO] $1"; }
log_warn() { echo "[WARN] $1"; }
log_error() { echo "[ERROR] $1"; }

# Configuration from environment
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"
WHISPER_MODEL_SIZE="${WHISPER_MODEL_SIZE:-${WHISPER_MODEL:-large-v3}}"
WEB_SERVER_PORT="${WEB_SERVER_PORT:-8000}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-false}"

log_info "=== Starship Horizons Learning AI ==="
log_info "Ollama Model: $OLLAMA_MODEL"
log_info "Whisper Model: $WHISPER_MODEL_SIZE"
log_info "Web Server Port: $WEB_SERVER_PORT"

# ------------------------------------------------------------------------------
# Start Ollama Server
# ------------------------------------------------------------------------------
log_info "Starting Ollama server..."

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
log_info "Waiting for Ollama server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_info "Ollama server is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Ollama server failed to start"
        exit 1
    fi
    sleep 1
done

# ------------------------------------------------------------------------------
# Download Ollama Model (if not already present)
# ------------------------------------------------------------------------------
if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then
    # Check if model is already downloaded
    if ollama list | grep -q "$OLLAMA_MODEL"; then
        log_info "Ollama model '$OLLAMA_MODEL' already available"
    else
        log_info "Downloading Ollama model: $OLLAMA_MODEL"
        log_info "This may take several minutes on first run..."
        if ollama pull "$OLLAMA_MODEL"; then
            log_info "Model '$OLLAMA_MODEL' downloaded successfully"
        else
            log_warn "Failed to download model - LLM features may not work"
        fi
    fi
else
    log_info "Skipping model download (SKIP_MODEL_DOWNLOAD=true)"
fi

# ------------------------------------------------------------------------------
# Start Web Server
# ------------------------------------------------------------------------------
log_info "Starting web server on port $WEB_SERVER_PORT..."

# Execute the main command (uvicorn)
exec python -m uvicorn src.web.server:app \
    --host 0.0.0.0 \
    --port "$WEB_SERVER_PORT" \
    --workers 1
