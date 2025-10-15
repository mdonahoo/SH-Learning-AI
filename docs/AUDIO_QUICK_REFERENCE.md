# Audio Transcription Quick Reference

Quick reference card for audio transcription features.

## 🚀 Quick Start (3 Steps)

```bash
# 1. Download Whisper model
python scripts/download_whisper_models.py --model base

# 2. Enable in .env
ENABLE_AUDIO_CAPTURE=true

# 3. Record with audio
python scripts/record_mission_with_audio.py --host 192.168.68.55:1864
```

## 📋 Common Commands

### Setup
```bash
# List audio devices
python scripts/list_audio_devices.py

# Download models
python scripts/download_whisper_models.py --model base
python scripts/download_whisper_models.py --model small,medium
```

### Testing
```bash
# Test audio capture
python scripts/test_realtime_audio.py --mode capture --duration 10

# Test speaker ID
python scripts/test_realtime_audio.py --mode diarization --duration 15

# Test full pipeline
python scripts/test_realtime_audio.py --mode full --duration 30
```

### Recording
```bash
# Record mission with audio
python scripts/record_mission_with_audio.py --host 192.168.68.55:1864

# 5-minute recording
python scripts/record_mission_with_audio.py --host 192.168.68.55:1864 --duration 300

# Without audio
python scripts/record_mission_with_audio.py --host 192.168.68.55:1864 --no-audio
```

## ⚙️ Key Configuration

### .env Settings

```bash
# Enable/Disable
ENABLE_AUDIO_CAPTURE=true

# Audio Device
AUDIO_INPUT_DEVICE=0              # Find with: python scripts/list_audio_devices.py
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1

# Model (choose one)
WHISPER_MODEL_SIZE=tiny           # Fastest, least accurate
WHISPER_MODEL_SIZE=base           # ✅ Recommended balance
WHISPER_MODEL_SIZE=small          # Better accuracy, slower
WHISPER_MODEL_SIZE=medium         # High accuracy, much slower

# Device
WHISPER_DEVICE=cpu                # ✅ CPU (works everywhere)
WHISPER_DEVICE=cuda               # GPU (requires CUDA)

# Transcription
TRANSCRIBE_REALTIME=true
TRANSCRIBE_LANGUAGE=en            # or 'auto' for detection
MIN_TRANSCRIPTION_CONFIDENCE=0.5
```

### VAD Tuning

```bash
# Sensitive (captures soft speech)
VAD_ENERGY_THRESHOLD=200
VAD_MIN_SPEECH_DURATION=0.2

# Balanced (recommended)
VAD_ENERGY_THRESHOLD=500          # ✅ Default
VAD_MIN_SPEECH_DURATION=0.3

# Conservative (only clear speech)
VAD_ENERGY_THRESHOLD=1000
VAD_MIN_SPEECH_DURATION=0.5
```

## 🔧 Troubleshooting

### No Audio Devices
```bash
# Linux/WSL2
sudo apt-get install pulseaudio portaudio19-dev python3-pyaudio
pip uninstall pyaudio && pip install pyaudio
```

### No Speech Detected
```bash
# Lower threshold
VAD_ENERGY_THRESHOLD=200
```

### Too Many False Positives
```bash
# Raise threshold
VAD_ENERGY_THRESHOLD=800
VAD_MIN_SPEECH_DURATION=0.5
```

### Slow Transcription
```bash
# Use smaller/faster model
WHISPER_MODEL_SIZE=tiny

# Or more workers (CPU)
TRANSCRIPTION_WORKERS=4

# Or use GPU
WHISPER_DEVICE=cuda
```

### Poor Accuracy
```bash
# Use larger model
WHISPER_MODEL_SIZE=small

# Ensure good audio quality
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
```

### Speaker ID Issues

Same speaker → Multiple IDs:
```bash
SPEAKER_SIMILARITY_THRESHOLD=0.6  # More forgiving
```

Different speakers → Same ID:
```bash
SPEAKER_SIMILARITY_THRESHOLD=0.8  # More strict
```

## 📊 Model Comparison

| Model | Speed (CPU) | Accuracy | Use When |
|-------|-------------|----------|----------|
| tiny | 32x realtime | ⭐⭐ | Testing, very low-end systems |
| base | 7x realtime | ⭐⭐⭐ | **Most users** - good balance |
| small | 4x realtime | ⭐⭐⭐⭐ | Better accuracy needed |
| medium | 2x realtime | ⭐⭐⭐⭐⭐ | High accuracy, powerful CPU/GPU |

## 🎯 Performance Tips

### For Speed (Real-time on weak CPU)
```bash
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
TRANSCRIPTION_WORKERS=2
VAD_ENERGY_THRESHOLD=800
```

### For Accuracy (Powerful system)
```bash
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
TRANSCRIPTION_WORKERS=4
MIN_TRANSCRIPTION_CONFIDENCE=0.7
```

### For Balance (Recommended)
```bash
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
TRANSCRIPTION_WORKERS=2
MIN_TRANSCRIPTION_CONFIDENCE=0.5
```

## 📁 Output Files

After recording, find files in `game_recordings/MISSION_ID/`:

```
game_recordings/GAME_20250103_143022/
├── game_events.json         # Game telemetry
├── transcripts.json         # Audio transcripts
└── combined_timeline.json   # Merged timeline
```

## 🔍 Log Levels

```bash
# See everything
LOG_LEVEL=DEBUG

# Normal operation
LOG_LEVEL=INFO              # ✅ Default

# Errors only
LOG_LEVEL=ERROR
```

## 📞 Get Help

```bash
# List devices
python scripts/list_audio_devices.py

# Test with verbose output
python scripts/test_realtime_audio.py --mode full --duration 10

# Check logs
tail -f logs/app.log
```

## 🌐 Language Support

```bash
# Auto-detect
TRANSCRIBE_LANGUAGE=auto

# Specific language
TRANSCRIBE_LANGUAGE=en    # English
TRANSCRIBE_LANGUAGE=es    # Spanish
TRANSCRIBE_LANGUAGE=fr    # French
TRANSCRIBE_LANGUAGE=de    # German
TRANSCRIBE_LANGUAGE=ja    # Japanese
TRANSCRIBE_LANGUAGE=zh    # Chinese
# ... 99 languages supported
```

## 🎮 Integration Examples

### Python API
```python
from src.integration.game_recorder import GameRecorder

recorder = GameRecorder(game_host="http://192.168.68.55:1864")
mission_id = recorder.start_recording("Training Alpha")

# Get live stats
stats = recorder.get_live_stats()
print(f"Transcripts: {stats['transcripts_count']}")
print(f"Speakers: {stats['conversation_summary']['unique_speakers']}")

# Combined timeline
timeline = recorder.get_combined_timeline()

recorder.stop_recording()
```

### Audio Service Only
```python
from src.metrics.audio_transcript import AudioTranscriptService

audio = AudioTranscriptService(mission_id="TEST_001", auto_transcribe=True)
audio.start_audio_capture()

# Monitor results
while recording:
    results = audio.get_transcription_results()
    for r in results.get('results', []):
        print(f"[{r['speaker_id']}]: {r['text']}")

audio.stop_audio_capture()
```

---

**Full Documentation**: [Audio Setup Guide](AUDIO_SETUP_GUIDE.md)
