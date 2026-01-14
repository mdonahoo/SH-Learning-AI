# Multi-Bridge Deployment Guide

This guide explains how to deploy the Starship Horizons Learning AI across multiple gaming computers for simultaneous recording of multiple bridges.

## Overview

Each bridge runs its own independent recording instance. Recordings are identified by a unique `BRIDGE_ID` and stored in bridge-specific directories.

## Architecture

### Option A: Local LLM (Per Bridge)
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Starship Horizons Game Server                    │
│                        (192.168.x.x:1864/1865)                      │
└───────────────┬─────────────────────┬─────────────────────┬─────────┘
                │                     │                     │
        ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
        │  Bridge-Alpha  │     │  Bridge-Beta  │     │ Bridge-Charlie│
        │   (Gaming PC)  │     │   (Gaming PC)  │     │   (Gaming PC) │
        │                │     │                │     │                │
        │ - Recording    │     │ - Recording    │     │ - Recording    │
        │ - Audio Capture│     │ - Audio Capture│     │ - Audio Capture│
        │ - Whisper      │     │ - Whisper      │     │ - Whisper      │
        │ - LLM (Ollama) │     │ - LLM (Ollama) │     │ - LLM (Ollama) │
        └───────┬────────┘     └───────┬────────┘     └───────┬────────┘
                │                      │                      │
                ▼                      ▼                      ▼
        game_recordings/       game_recordings/       game_recordings/
        └─Bridge-Alpha/        └─Bridge-Beta/         └─Bridge-Charlie/
```

### Option B: Centralized Remote LLM (Recommended)
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Starship Horizons Game Server                    │
│                        (192.168.x.x:1864/1865)                      │
└───────────────┬─────────────────────┬─────────────────────┬─────────┘
                │                     │                     │
        ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
        │  Bridge-Alpha  │     │  Bridge-Beta  │     │ Bridge-Charlie│
        │   (Gaming PC)  │     │   (Gaming PC)  │     │   (Gaming PC) │
        │                │     │                │     │                │
        │ - Recording    │     │ - Recording    │     │ - Recording    │
        │ - Audio Capture│     │ - Audio Capture│     │ - Audio Capture│
        │ - Whisper      │     │ - Whisper      │     │ - Whisper      │
        └───────┬────────┘     └───────┬────────┘     └───────┬────────┘
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │   Central LLM Server     │
                        │   (192.168.x.x:11434)    │
                        │                          │
                        │   - Ollama               │
                        │   - RTX 4090 (24GB)      │
                        │   - qwen2.5:14b          │
                        └──────────────────────────┘
```

## Hardware Requirements

### Component Analysis

The system has several resource-intensive components:

| Component | CPU Impact | RAM Impact | GPU/VRAM Impact | Notes |
|-----------|------------|------------|-----------------|-------|
| Game Telemetry | Low | Low (~100MB) | None | WebSocket + HTTP polling |
| Audio Capture | Low | Low (~50MB) | None | PyAudio streaming |
| **Whisper Transcription** | **High** | Moderate | **High (optional)** | Real-time speech-to-text |
| **Speaker Diarization** | Moderate | Moderate | **High (optional)** | Neural embeddings |
| Event Recording | Low | Moderate (~200MB) | None | In-memory storage |
| **LLM Analysis (Ollama)** | **Very High** | **High** | **Very High** | Report generation |

### Whisper Model Requirements

Whisper transcription is the primary real-time compute bottleneck:

| Model Size | Parameters | CPU Time (30s audio) | VRAM Required | Accuracy |
|------------|------------|---------------------|---------------|----------|
| tiny | 39M | ~1-2s | ~1GB | Basic |
| **base** (recommended) | 74M | ~2-4s | ~1GB | Good |
| small | 244M | ~5-10s | ~2GB | Better |
| medium | 769M | ~15-30s | ~4GB | Very Good |
| large-v3 | 1.5B | ~30-60s | ~10GB | Best |

**Recommendation**: Use `base` model for real-time transcription. It provides good accuracy while maintaining <4s latency on modern CPUs.

### LLM Requirements (Ollama)

LLM-powered report generation is the most demanding component:

| Model | Parameters | VRAM Required | CPU-Only Time | Quality |
|-------|------------|---------------|---------------|---------|
| llama3.2:3b | 3B | 4-6GB | Slow (~60s/report) | Good |
| qwen2.5:7b | 7B | 6-8GB | Very Slow | Better |
| **qwen2.5:14b** (default) | 14B | 12-16GB | Impractical | Excellent |
| llama3:70b | 70B | 40GB+ | N/A | Best |

**Critical Insight**: LLM analysis is optional and runs only at the END of recording (not real-time). You have several deployment options:

### Deployment Scenarios

#### Scenario A: Full Local Processing (Per Bridge)
Each gaming PC runs Whisper + Ollama locally.

**Requirements per PC:**
- CPU: 8-core modern processor (Intel i7-12700 / AMD Ryzen 7 5800X)
- RAM: 32GB DDR4/DDR5
- GPU: NVIDIA RTX 3060 (12GB) or RTX 4060 Ti (16GB)
- Storage: 500GB NVMe SSD

**Pros:** Complete autonomy, no network dependency
**Cons:** Expensive (3x GPU cost), slower LLM on smaller models

#### Scenario B: Distributed Recording + Centralized LLM (Recommended)
Gaming PCs handle recording + Whisper only. A central server runs Ollama.

**Requirements per Gaming PC:**
- CPU: 6-core modern processor (Intel i5 / AMD Ryzen 5)
- RAM: 16GB DDR4
- GPU: NVIDIA RTX 3060 (12GB) or CPU-only with `int8` quantization
- Storage: 256GB SSD

**Requirements for Central LLM Server:**
- CPU: 16+ cores (Intel Xeon / AMD EPYC / Threadripper)
- RAM: 64GB+ DDR4/DDR5
- GPU: NVIDIA RTX 4090 (24GB) or dual RTX 3090 (48GB total)
- Storage: 1TB NVMe SSD
- Alternative: Apple Mac Studio M2 Ultra (64GB+ unified memory)

**Pros:** Cost-effective, better LLM quality, centralized management
**Cons:** Requires network transfer of recordings

#### Scenario C: Post-Processing Only
Gaming PCs record audio + telemetry without real-time transcription. All AI processing happens offline.

**Requirements per Gaming PC:**
- CPU: 4-core processor (Intel i5 / AMD Ryzen 5)
- RAM: 8GB DDR4
- GPU: Not required
- Storage: 128GB SSD

**Post-Processing Server:**
- Same as Scenario B central server
- Can process all 3 bridges sequentially after missions

**Pros:** Cheapest gaming PC requirements, best LLM quality
**Cons:** No real-time feedback, manual post-processing step

### Recommended Configuration

For most deployments, we recommend **Scenario B**:

**3x Gaming PCs (Bridge Recording Stations):**
```
CPU: Intel Core i5-12400 or AMD Ryzen 5 5600X
RAM: 16GB DDR4-3200
GPU: NVIDIA RTX 3060 12GB (for Whisper acceleration)
     OR CPU-only with WHISPER_COMPUTE_TYPE=int8
Storage: 256GB NVMe SSD
Audio: USB conference microphone (e.g., Jabra Speak 410)
OS: Ubuntu 22.04 LTS or Windows 11
```

**1x Central Server (LLM Processing):**
```
CPU: AMD Ryzen 9 5950X or Intel i9-12900K
RAM: 64GB DDR4-3600
GPU: NVIDIA RTX 4090 24GB
Storage: 1TB NVMe SSD
OS: Ubuntu 22.04 LTS Server
```

### Cost Estimate (2024 USD)

| Component | Scenario A (Full Local) | Scenario B (Recommended) | Scenario C (Post-Process) |
|-----------|------------------------|-------------------------|--------------------------|
| 3x Gaming PCs | $3,600 - $4,500 | $2,400 - $3,000 | $1,500 - $2,100 |
| Central Server | N/A | $2,500 - $3,500 | $2,500 - $3,500 |
| **Total** | **$3,600 - $4,500** | **$4,900 - $6,500** | **$4,000 - $5,600** |

## Configuration

### Step 1: Set Bridge ID on Each Computer

Edit `.env` on each gaming computer:

**Computer 1 (Bridge Alpha):**
```bash
BRIDGE_ID=Bridge-Alpha
GAME_HOST=192.168.1.100
ENABLE_AUDIO_CAPTURE=true
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cuda  # or 'cpu' if no GPU
```

**Computer 2 (Bridge Beta):**
```bash
BRIDGE_ID=Bridge-Beta
GAME_HOST=192.168.1.100
ENABLE_AUDIO_CAPTURE=true
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cuda
```

**Computer 3 (Bridge Charlie):**
```bash
BRIDGE_ID=Bridge-Charlie
GAME_HOST=192.168.1.100
ENABLE_AUDIO_CAPTURE=true
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cuda
```

### Step 2: Configure Audio Devices

Each bridge needs its own microphone. Use the device discovery script:

```bash
python scripts/list_audio_devices.py
```

Then set the correct device in `.env`:

```bash
AUDIO_INPUT_DEVICE=2  # Your microphone's device index
```

### Step 3: Configure LLM (Local - Optional)

For local LLM reports, install Ollama and pull a model:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended model
ollama pull qwen2.5:14b  # Requires 16GB+ VRAM
# OR for smaller GPUs:
ollama pull llama3.2:3b  # Requires 6GB VRAM
```

To disable LLM reports (for Scenario C):
```bash
ENABLE_LLM_REPORTS=false
```

### Step 3 (Alternative): Configure Remote Ollama Server

For Scenario B (recommended), configure a centralized LLM server that all bridges connect to.

#### On the Central LLM Server

**1. Install Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**2. Configure Ollama to accept remote connections:**

By default, Ollama only listens on localhost. To accept connections from gaming PCs:

```bash
# Option A: Set environment variable before starting
export OLLAMA_HOST=0.0.0.0:11434
ollama serve

# Option B: Create systemd override (persistent)
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**3. Open firewall port:**
```bash
# Ubuntu/Debian
sudo ufw allow 11434/tcp

# RHEL/CentOS/Fedora
sudo firewall-cmd --add-port=11434/tcp --permanent
sudo firewall-cmd --reload

# Windows (PowerShell as Admin)
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -Port 11434 -Protocol TCP -Action Allow
```

**4. Pull the model:**
```bash
# Recommended for 24GB VRAM
ollama pull qwen2.5:14b

# Or quantized version for 12-16GB VRAM
ollama pull qwen2.5:14b-q4_0

# Or smaller model for 8GB VRAM
ollama pull qwen2.5:7b
```

**5. Verify server is running:**
```bash
# Check Ollama is listening on all interfaces
curl http://localhost:11434/api/tags

# From another machine, test connectivity
curl http://<SERVER_IP>:11434/api/tags
```

#### On Each Gaming PC (Bridge)

**1. Configure `.env` to use remote Ollama:**

```bash
# Remote Ollama Configuration
OLLAMA_HOST=http://192.168.1.50:11434   # Your LLM server's IP
OLLAMA_MODEL=qwen2.5:14b                 # Model installed on server
OLLAMA_TIMEOUT=300                       # Increase for network latency
```

**2. Test the connection:**
```bash
# Quick connectivity test
curl http://192.168.1.50:11434/api/tags

# Full Python test
python -c "
from src.llm.ollama_client import OllamaClient
client = OllamaClient()
print('Host:', client.host)
print('Connected:', client.check_connection())
print('Models:', client.list_models())
"
```

#### Complete Multi-Bridge Configuration Example

**Bridge-Alpha `.env`:**
```bash
# Bridge Identity
BRIDGE_ID=Bridge-Alpha

# Game Server
GAME_HOST=192.168.1.100
GAME_PORT_API=1864
GAME_PORT_WS=1865

# Audio (local)
ENABLE_AUDIO_CAPTURE=true
AUDIO_INPUT_DEVICE=2
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cuda

# LLM (remote)
OLLAMA_HOST=http://192.168.1.50:11434
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_TIMEOUT=300
ENABLE_LLM_REPORTS=true
```

**Bridge-Beta `.env`:**
```bash
BRIDGE_ID=Bridge-Beta
GAME_HOST=192.168.1.100
ENABLE_AUDIO_CAPTURE=true
AUDIO_INPUT_DEVICE=1
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cuda
OLLAMA_HOST=http://192.168.1.50:11434
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_TIMEOUT=300
```

**Bridge-Charlie `.env`:**
```bash
BRIDGE_ID=Bridge-Charlie
GAME_HOST=192.168.1.100
ENABLE_AUDIO_CAPTURE=true
AUDIO_INPUT_DEVICE=0
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
OLLAMA_HOST=http://192.168.1.50:11434
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_TIMEOUT=300
```

#### Remote Ollama Notes

1. **Concurrent Requests**: When multiple bridges finish recording simultaneously, Ollama queues requests and processes them sequentially. This is fine since LLM reports are generated post-recording.

2. **Timeout Settings**: Increase `OLLAMA_TIMEOUT` to 300-600 seconds for remote connections. Large reports with 14B+ models can take 2-3 minutes.

3. **Network Bandwidth**: LLM requests are text-only (typically <100KB). Network speed is not a bottleneck.

4. **Security**: Ollama has no built-in authentication. Only expose port 11434 on trusted networks. For untrusted networks, use SSH tunneling or a reverse proxy with auth.

5. **Model Preloading**: The first request after server restart loads the model into GPU memory (~30s for 14B models). Subsequent requests are faster.

### Step 4: Run Recording Scripts

On each computer, run the recording script:

```bash
python scripts/record_mission_with_audio.py
# Or with explicit parameters:
python scripts/record_mission_with_audio.py --bridge-id Bridge-Alpha --host 192.168.1.100
```

## Output Structure

Recordings are organized by bridge:

```
game_recordings/
├── Bridge-Alpha/
│   ├── Bridge-Alpha_GAME_20260114_143000/
│   │   ├── game_events.json
│   │   ├── transcripts.json
│   │   ├── combined_timeline.json
│   │   └── mission_report_llm.md
│   └── Bridge-Alpha_GAME_20260114_160000/
├── Bridge-Beta/
│   └── Bridge-Beta_GAME_20260114_143015/
└── Bridge-Charlie/
    └── Bridge-Charlie_GAME_20260114_143008/
```

## Collecting Recordings

After missions, you can sync recordings to a central location:

```bash
# From each gaming PC, sync to central storage
rsync -avz game_recordings/ central-server:/data/recordings/

# Or use a shared network drive
# Configure RECORDING_PATH in .env to point to network share
```

## GPU Optimization Tips

### For Whisper (Real-Time Transcription)

1. **Use CUDA if available:**
   ```bash
   WHISPER_DEVICE=cuda
   WHISPER_COMPUTE_TYPE=float16  # Fastest on GPU
   ```

2. **For CPU-only (no GPU):**
   ```bash
   WHISPER_DEVICE=cpu
   WHISPER_COMPUTE_TYPE=int8  # 2-3x faster than float32
   ```

3. **Reduce model size for faster processing:**
   ```bash
   WHISPER_MODEL_SIZE=base  # Good balance of speed/accuracy
   ```

### For Ollama (LLM Reports)

1. **Quantized models reduce VRAM usage:**
   ```bash
   ollama pull qwen2.5:14b-q4_0  # 4-bit quantization, ~8GB VRAM
   ```

2. **Configure timeouts for large models:**
   ```bash
   OLLAMA_TIMEOUT=300  # 5 minutes for large reports
   ```

## Troubleshooting

### Audio Not Recording
- Check `AUDIO_INPUT_DEVICE` matches your microphone
- Run `scripts/list_audio_devices.py` to verify device index
- Ensure microphone permissions are granted

### Whisper Too Slow
- Switch to `base` or `tiny` model
- Enable GPU: `WHISPER_DEVICE=cuda`
- Use INT8 quantization: `WHISPER_COMPUTE_TYPE=int8`

### Ollama Out of Memory
- Use smaller model: `ollama pull llama3.2:3b`
- Use quantized model: `ollama pull qwen2.5:7b-q4_0`
- Disable LLM reports: `ENABLE_LLM_REPORTS=false`

### Connection Issues
- Verify `GAME_HOST` is correct IP address
- Check firewall allows ports 1864/1865
- Test with: `curl http://{GAME_HOST}:1864/api/status`

### Remote Ollama Not Connecting
- Verify `OLLAMA_HOST` includes `http://` prefix and port: `http://192.168.1.50:11434`
- Check Ollama is configured to listen on all interfaces (`OLLAMA_HOST=0.0.0.0:11434`)
- Test from gaming PC: `curl http://<SERVER_IP>:11434/api/tags`
- Check firewall on LLM server allows port 11434
- Verify the model is installed: `ollama list` on the server

### Remote Ollama Timeout
- Increase `OLLAMA_TIMEOUT` to 300-600 seconds in `.env`
- First request after server restart is slow (model loading)
- Check server GPU memory: `nvidia-smi` - if full, model may be swapping
- Try a smaller/quantized model: `qwen2.5:14b-q4_0` or `qwen2.5:7b`

## Backwards Compatibility

If `BRIDGE_ID` is not set, the system operates exactly as before:
- Mission ID: `GAME_YYYYMMDD_HHMMSS`
- Directory: `game_recordings/GAME_YYYYMMDD_HHMMSS/`
- No bridge metadata in exports
