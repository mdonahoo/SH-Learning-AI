# Audio Transcription Setup Guide

Complete guide for setting up real-time audio transcription in the Starship Horizons Learning AI system.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Device Setup](#device-setup)
- [Model Setup](#model-setup)
- [Testing](#testing)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

The audio transcription system provides:

- **Real-time audio capture** from microphone using PyAudio
- **Local AI transcription** using Faster-Whisper (no cloud dependency)
- **Speaker diarization** to identify multiple crew members
- **Voice Activity Detection** for automatic speech segmentation
- **Engagement analytics** to measure crew participation and communication effectiveness
- **Synchronized timeline** combining game telemetry with audio transcripts

### Architecture

```
Microphone → PyAudio → VAD → Speaker ID → Whisper → Transcripts
                                                           ↓
Game Events → EventRecorder ────────────────────→ Combined Timeline
```

## System Requirements

### Hardware

- **Microphone**: Any USB or built-in microphone
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 1-3GB for Whisper models

### Software

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Audio System**:
  - Linux: ALSA or PulseAudio
  - macOS: CoreAudio
  - Windows/WSL2: PulseAudio

### Optional (Performance)

- **GPU**: CUDA-compatible GPU for faster transcription
- **CUDA Toolkit**: 11.x or 12.x

## Installation

### 1. Install System Dependencies

#### Ubuntu/Debian (WSL2)
```bash
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    pulseaudio
```

#### macOS
```bash
brew install portaudio ffmpeg
```

### 2. Install Python Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install audio components individually
pip install pyaudio faster-whisper torch torchaudio webrtcvad psutil
```

### 3. Verify Installation

```bash
python -c "import pyaudio; print('PyAudio:', pyaudio.__version__)"
python -c "from faster_whisper import WhisperModel; print('Faster-Whisper: OK')"
```

## Configuration

### Environment Variables

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` and configure audio settings:

```bash
# ==========================================
# AUDIO TRANSCRIPTION CONFIGURATION
# ==========================================

# Enable/Disable Audio Capture
ENABLE_AUDIO_CAPTURE=true

# Audio Device Configuration
AUDIO_INPUT_DEVICE=0              # Device index (use scripts/list_audio_devices.py)
AUDIO_SAMPLE_RATE=16000           # Sample rate in Hz (16000 recommended)
AUDIO_CHANNELS=1                  # Number of channels (1=mono, 2=stereo)
AUDIO_CHUNK_MS=100                # Chunk size in milliseconds

# Whisper Model Configuration
WHISPER_MODEL_SIZE=base           # tiny, base, small, medium, large-v3
WHISPER_DEVICE=cpu                # cpu or cuda
WHISPER_COMPUTE_TYPE=int8         # int8, float16, float32
WHISPER_MODEL_PATH=./data/models/whisper/

# Transcription Settings
TRANSCRIBE_REALTIME=true
TRANSCRIBE_LANGUAGE=en            # Language code or 'auto'
MIN_TRANSCRIPTION_CONFIDENCE=0.5  # Minimum confidence threshold (0.0-1.0)
TRANSCRIPTION_WORKERS=2           # Number of worker threads
MAX_SEGMENT_QUEUE_SIZE=100        # Max queued segments

# Voice Activity Detection (VAD)
VAD_ENERGY_THRESHOLD=500          # Energy threshold for speech detection
VAD_MIN_SPEECH_DURATION=0.3       # Minimum speech duration in seconds
VAD_MAX_SPEECH_DURATION=30        # Maximum speech duration in seconds
VAD_SILENCE_DURATION=0.8          # Silence duration to end utterance

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION=true
SPEAKER_SIMILARITY_THRESHOLD=0.7  # Cosine similarity threshold (0.0-1.0)
SPEAKER_PROFILE_UPDATE_RATE=0.3   # Exponential moving average alpha

# Engagement Analytics
ENABLE_ENGAGEMENT_METRICS=true
BRIDGE_ROLES=Captain,Helm,Tactical,Science,Engineering,Communications
```

### Model Size Selection

Choose based on your hardware and accuracy needs:

| Model | Size | RAM | CPU Speed | GPU Speed | Accuracy |
|-------|------|-----|-----------|-----------|----------|
| tiny | 75MB | 1GB | ~32x realtime | ~320x | Low |
| base | 145MB | 1GB | ~7x realtime | ~70x | Good |
| small | 466MB | 2GB | ~4x realtime | ~40x | Better |
| medium | 1.5GB | 5GB | ~2x realtime | ~20x | High |
| large-v3 | 3GB | 10GB | ~1x realtime | ~10x | Best |

**Recommended**: Start with `base` for good balance of speed and accuracy.

## Device Setup

### 1. List Available Audio Devices

```bash
python scripts/list_audio_devices.py
```

Output example:
```
======================================================================
Available Audio Input Devices for Starship Horizons
======================================================================

======================================================================
Device Index: 0
Name: USB Microphone
Max Input Channels: 1
Default Sample Rate: 48000 Hz
Host API: ALSA
```

### 2. Configure Device

Set `AUDIO_INPUT_DEVICE` in `.env` to the device index:

```bash
AUDIO_INPUT_DEVICE=0
```

### 3. Test Device

```bash
# Test audio capture only (no transcription)
python scripts/test_realtime_audio.py --mode capture --duration 10
```

## Model Setup

### Download Whisper Models

```bash
# Download base model (recommended)
python scripts/download_whisper_models.py --model base

# Download multiple models
python scripts/download_whisper_models.py --model tiny,base,small

# Download for GPU
python scripts/download_whisper_models.py --model base --device cuda --compute-type float16
```

Models are cached in `./data/models/whisper/` by default.

### Verify Model

```bash
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('base', device='cpu', compute_type='int8')
print('✓ Model loaded successfully')
"
```

## Testing

### 1. Test Audio Capture + VAD

Tests microphone input and voice activity detection:

```bash
python scripts/test_realtime_audio.py --mode capture --duration 15
```

Expected output:
- Segments detected when you speak
- Silence is ignored
- Each segment shows start/end times

### 2. Test Speaker Diarization

Tests speaker identification:

```bash
python scripts/test_realtime_audio.py --mode diarization --duration 20
```

Expected output:
- Different speakers assigned unique IDs
- Confidence scores for each identification
- Summary of unique speakers detected

### 3. Test Full Transcription Pipeline

Tests complete audio → text pipeline:

```bash
python scripts/test_realtime_audio.py --mode full --duration 30
```

Expected output:
- Real-time transcripts as you speak
- Speaker identification
- Engagement metrics (turn-taking, response times)
- Speaker participation statistics

### 4. Troubleshooting Tests

If tests fail, check:

```bash
# Verify PyAudio installation
python -c "import pyaudio; p = pyaudio.PyAudio(); print(f'Devices: {p.get_device_count()}')"

# Check audio permissions (Linux)
groups | grep audio  # Should show 'audio' group

# Test PulseAudio (WSL2)
pulseaudio --check && echo "PulseAudio running" || echo "PulseAudio not running"
```

## Usage

### Record Mission with Audio

```bash
# Basic usage (auto-detect mission name)
python scripts/record_mission_with_audio.py --host 192.168.68.55

# Specify mission name and duration
python scripts/record_mission_with_audio.py \
    --host 192.168.68.55 \
    --mission-name "Training Exercise Alpha" \
    --duration 600

# Without audio (events only)
python scripts/record_mission_with_audio.py \
    --host 192.168.68.55 \
    --no-audio
```

### Python API Usage

```python
from src.integration.game_recorder import GameRecorder

# Create recorder
recorder = GameRecorder(game_host="http://192.168.68.55:1864")

# Start recording with audio
mission_id = recorder.start_recording(mission_name="Alpha Mission")

# ... mission happens ...

# Get live stats including audio/engagement
stats = recorder.get_live_stats()
print(f"Transcripts: {stats.get('transcripts_count')}")
print(f"Speakers: {stats.get('conversation_summary', {}).get('unique_speakers')}")

# Get combined timeline (events + audio)
timeline = recorder.get_combined_timeline()

# Stop and export
summary = recorder.stop_recording()
print(f"Saved to: {summary['export_path']}")
```

### Direct Audio Service Usage

```python
from src.metrics.audio_transcript import AudioTranscriptService

# Create service
audio = AudioTranscriptService(
    mission_id="MISSION_001",
    auto_transcribe=True
)

# Start capture and transcription
audio.start_audio_capture()
audio.start_realtime_transcription()

# Monitor results
while recording:
    results = audio.get_transcription_results()
    for result in results.get('results', []):
        print(f"[{result['speaker_id']}]: {result['text']}")

# Get engagement metrics
engagement = audio.get_engagement_summary()
print(f"Turn-taking rate: {engagement['turn_taking_rate']}")

# Stop
audio.stop_audio_capture()
audio.stop_realtime_transcription()
```

## Troubleshooting

### PyAudio Installation Issues

**Error**: `_portaudio.so: cannot open shared object file`

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# Reinstall PyAudio
pip uninstall pyaudio
pip install pyaudio
```

**Error**: `No default input device`

```bash
# Check devices
python scripts/list_audio_devices.py

# Set device explicitly in .env
AUDIO_INPUT_DEVICE=0
```

### Whisper Model Issues

**Error**: `Failed to load Whisper model`

```bash
# Re-download model
rm -rf ./data/models/whisper/
python scripts/download_whisper_models.py --model base

# Check disk space
df -h ./data/models/
```

**Error**: `CUDA not available`

```bash
# Use CPU instead
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

### Audio Quality Issues

**Problem**: No speech detected (VAD not triggering)

```bash
# Lower VAD threshold in .env
VAD_ENERGY_THRESHOLD=200  # Default: 500
```

**Problem**: Too many false positives

```bash
# Raise VAD threshold
VAD_ENERGY_THRESHOLD=800
VAD_MIN_SPEECH_DURATION=0.5
```

**Problem**: Poor transcription accuracy

```bash
# Use larger model
WHISPER_MODEL_SIZE=small  # or medium

# Or adjust confidence threshold
MIN_TRANSCRIPTION_CONFIDENCE=0.6
```

### Speaker Diarization Issues

**Problem**: Same speaker gets multiple IDs

```bash
# Lower similarity threshold (more forgiving)
SPEAKER_SIMILARITY_THRESHOLD=0.6  # Default: 0.7
```

**Problem**: Different speakers get same ID

```bash
# Raise similarity threshold (more strict)
SPEAKER_SIMILARITY_THRESHOLD=0.8
```

### Performance Issues

**Problem**: Transcription lagging behind audio

```bash
# Increase worker threads
TRANSCRIPTION_WORKERS=4  # Default: 2

# Use faster model
WHISPER_MODEL_SIZE=tiny  # or base

# Reduce queue size
MAX_SEGMENT_QUEUE_SIZE=50
```

**Problem**: High CPU usage

```bash
# Use GPU if available
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Or use lighter model
WHISPER_MODEL_SIZE=tiny
WHISPER_COMPUTE_TYPE=int8
```

### WSL2 Audio Issues

**Problem**: No audio devices found

```bash
# Install PulseAudio
sudo apt-get install pulseaudio

# Configure PulseAudio for WSL2
# In Windows, ensure PulseAudio server is running
# Set PULSE_SERVER environment variable if needed
export PULSE_SERVER=tcp:$(grep nameserver /etc/resolv.conf | awk '{print $2}'):4713
```

## Performance Optimization

### CPU Optimization

1. **Use int8 quantization**:
   ```bash
   WHISPER_COMPUTE_TYPE=int8
   ```

2. **Limit worker threads**:
   ```bash
   TRANSCRIPTION_WORKERS=2
   ```

3. **Use smaller model**:
   ```bash
   WHISPER_MODEL_SIZE=base
   ```

### GPU Optimization

1. **Enable CUDA**:
   ```bash
   WHISPER_DEVICE=cuda
   WHISPER_COMPUTE_TYPE=float16
   ```

2. **Increase worker threads**:
   ```bash
   TRANSCRIPTION_WORKERS=4
   ```

3. **Use larger model**:
   ```bash
   WHISPER_MODEL_SIZE=medium
   ```

### Memory Optimization

1. **Reduce queue size**:
   ```bash
   MAX_SEGMENT_QUEUE_SIZE=50
   ```

2. **Limit speech duration**:
   ```bash
   VAD_MAX_SPEECH_DURATION=20
   ```

3. **Use smaller model**:
   ```bash
   WHISPER_MODEL_SIZE=tiny
   ```

## Advanced Configuration

### Custom VAD Parameters

Fine-tune voice activity detection:

```bash
# Very sensitive (captures soft speech)
VAD_ENERGY_THRESHOLD=200
VAD_MIN_SPEECH_DURATION=0.2
VAD_SILENCE_DURATION=0.5

# Very conservative (only clear speech)
VAD_ENERGY_THRESHOLD=1000
VAD_MIN_SPEECH_DURATION=0.5
VAD_SILENCE_DURATION=1.0
```

### Multi-Language Support

```bash
# Auto-detect language
TRANSCRIBE_LANGUAGE=auto

# Specific language
TRANSCRIBE_LANGUAGE=es  # Spanish
TRANSCRIBE_LANGUAGE=fr  # French
TRANSCRIBE_LANGUAGE=de  # German
```

### Custom Bridge Roles

Map speaker IDs to bridge positions:

```bash
BRIDGE_ROLES=Captain,XO,Helm,Tactical,Science,Engineering,Communications,Medical
```

## Appendix

### Supported Audio Formats

- **Input**: 16-bit PCM via microphone
- **Sample Rates**: 8000, 16000, 22050, 44100, 48000 Hz (16000 recommended)
- **Channels**: Mono (1) or Stereo (2) - Mono recommended

### Whisper Model Details

All models support:
- 99 languages
- Word-level timestamps
- Built-in VAD
- Multi-task (transcribe, translate)

### File Locations

```
./data/models/whisper/     # Whisper models cache
./game_recordings/         # Mission recordings
  └── GAME_YYYYMMDD_HHMMSS/
      ├── game_events.json       # Game telemetry
      ├── transcripts.json       # Audio transcripts
      └── combined_timeline.json # Merged timeline
```

### Environment Variable Reference

See `.env.example` for complete list with descriptions.

### Further Reading

- [Faster-Whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [PyAudio Documentation](https://people.csail.mit.edu/hubert/pyaudio/)
- [OpenAI Whisper](https://github.com/openai/whisper)
