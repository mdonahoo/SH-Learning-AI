# Quick Start Guide

Get up and running with Starship Horizons Learning AI in under 10 minutes.

## Prerequisites

- Python 3.11 or higher
- Access to a Starship Horizons game server
- WSL2 (if on Windows)
- Audio input device (for transcription features)

## Installation

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SH-Learning-AI.git
cd SH-Learning-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required settings**:
```bash
# Update with your game server IP
GAME_HOST=192.168.68.55

# Audio device (optional, for transcription)
AUDIO_INPUT_DEVICE=0
```

**Find your audio device**:
```bash
python scripts/list_audio_devices.py
```

### 3. Download AI Models (First Time Only)

```bash
# Download Whisper model for transcription
python scripts/download_whisper_models.py --model base
```

## Basic Usage

### Test Connection

```bash
# Test WebSocket connection to game server
python scripts/test_websocket_live_manual.py
```

### Record a Mission (No Audio)

```bash
# Record game events only
python scripts/record_game.py --host http://192.168.68.55:1864

# Press Ctrl+C to stop recording
```

Output: `data/game_recordings/GAME_YYYYMMDD_HHMMSS/`

### Record with Audio Transcription

```bash
# Full recording with voice transcription
python scripts/record_mission_with_audio.py --host http://192.168.68.55:1864

# 30-second test
python scripts/record_mission_with_audio.py --host http://192.168.68.55:1864 --duration 30
```

Output includes:
- Game events (`game_events.json`)
- Audio transcripts (`transcripts.json`)
- Audio segments (`.wav` files, if `SAVE_RAW_AUDIO=true`)
- Combined timeline (`combined_timeline.json`)

### Generate Mission Report

```bash
# Generate report from recording
python scripts/generate_mission_report.py --mission data/game_recordings/GAME_20251015_120000
```

## Common Workflows

### Workflow 1: Basic Mission Recording

```bash
# 1. Start recording
python scripts/record_game.py

# 2. Play your mission in Starship Horizons

# 3. Stop recording (Ctrl+C)

# 4. View events
cat data/game_recordings/GAME_*/game_events.json | jq
```

### Workflow 2: Full Audio Analysis

```bash
# 1. Test your microphone
python scripts/test_realtime_audio.py --mode capture --duration 10

# 2. Record mission with audio
python scripts/record_mission_with_audio.py

# 3. Generate AI-powered report (requires Ollama)
python scripts/generate_mission_report.py --mission data/game_recordings/GAME_* --style entertaining
```

### Workflow 3: Station-Specific Monitoring

```bash
# Record only Tactical station events
python scripts/test_station_connection_manual.py --station Tactical

# Or Engineering station
python scripts/test_station_connection_manual.py --station Engineering
```

## Python API Usage

### Simple WebSocket Client

```python
from src.integration.starship_horizons_client import StarshipHorizonsClient
import asyncio

async def main():
    client = StarshipHorizonsClient(host="http://192.168.68.55:1864")

    if await client.connect():
        print("Connected!")

        # Receive 10 messages
        for _ in range(10):
            msg = await client.receive_message()
            if msg:
                print(f"Event: {msg.get('type')}")

        await client.disconnect()

asyncio.run(main())
```

### Recording with Context Manager

```python
from src.integration.game_recorder import GameRecorder
from pathlib import Path
import asyncio

async def record_session():
    recorder = GameRecorder(
        host="http://192.168.68.55:1864",
        enable_audio=True,
        output_dir=Path("./my_recordings")
    )

    session_path = await recorder.start()
    print(f"Recording to: {session_path}")

    # Let it record for 60 seconds
    await asyncio.sleep(60)

    stats = await recorder.stop()
    print(f"Recorded {stats['event_count']} events")
    print(f"Audio segments: {stats['audio_segments']}")

asyncio.run(record_session())
```

### Audio Transcription Only

```python
from src.audio.capture import AudioCapture
from src.audio.whisper_transcriber import WhisperTranscriber
import asyncio

async def transcribe_audio():
    capture = AudioCapture(sample_rate=16000)
    transcriber = WhisperTranscriber(model_size="base")

    capture.start()
    print("Listening for speech...")

    # Get 5 speech segments
    for i in range(5):
        audio = await capture.get_speech_segment(timeout=30.0)
        if audio is not None:
            result = transcriber.transcribe(audio)
            print(f"{i+1}. {result['text']} (confidence: {result['confidence']:.2f})")

    capture.stop()

asyncio.run(transcribe_audio())
```

### Mission Analysis

```python
from src.metrics.mission_summarizer import MissionSummarizer
from pathlib import Path
import asyncio

async def analyze_mission():
    summarizer = MissionSummarizer(
        mission_dir=Path("data/game_recordings/GAME_20251015_120000"),
        use_llm=True
    )

    # Get timeline
    timeline = summarizer.generate_timeline()
    print(f"Mission had {len(timeline)} events")

    # Performance metrics
    metrics = summarizer.analyze_performance()
    print(f"Crew engagement: {metrics['crew_metrics']}")

    # Generate report
    report = await summarizer.generate_report(style="professional")

    with open("mission_report.md", "w") as f:
        f.write(report)

asyncio.run(analyze_mission())
```

## Testing Audio Setup

### 1. Test Microphone

```bash
python scripts/test_realtime_audio.py --mode capture --duration 10
```

Expected: Should record and save a 10-second test file.

### 2. Test Voice Activity Detection

```bash
python scripts/test_realtime_audio.py --mode vad --duration 30
```

Expected: Should detect when you speak vs silence.

### 3. Test Transcription

```bash
python scripts/test_realtime_audio.py --mode full --duration 30
```

Expected: Should transcribe your speech in real-time.

### 4. Test Speaker Diarization

```bash
python scripts/test_realtime_audio.py --mode diarization --duration 60
```

Expected: Should identify different speakers.

## LLM Setup (Optional)

For AI-generated mission reports, install and run Ollama:

### Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows WSL
# Follow Linux instructions
```

### Download Model

```bash
ollama pull llama3.2
```

### Configure

```bash
# In .env
ENABLE_LLM_REPORTS=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
LLM_REPORT_STYLE=entertaining  # or professional, technical, casual
```

### Test LLM

```python
from src.llm.ollama_client import OllamaClient
import asyncio

async def test_llm():
    client = OllamaClient()

    if await client.check_connection():
        response = await client.generate("Tell me about Starship Horizons")
        print(response)
    else:
        print("Ollama not running!")

asyncio.run(test_llm())
```

## Troubleshooting

### Can't Connect to Game Server

```bash
# Test network connectivity
ping 192.168.68.55

# Check if game server is running
curl http://192.168.68.55:1864

# Verify ports in .env match your server
```

### Audio Not Working (WSL2)

```bash
# Install PulseAudio
sudo apt-get update
sudo apt-get install -y pulseaudio portaudio19-dev

# Check devices
python scripts/list_audio_devices.py

# If no devices, configure PulseAudio
export PULSE_SERVER=tcp:$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
```

### Whisper Model Download Fails

```bash
# Manual download
python scripts/download_whisper_models.py --model base --verbose

# Check storage space
df -h

# Verify model path
echo $WHISPER_MODEL_PATH
ls -lh ./data/models/whisper/
```

### "No module named 'src'"

```bash
# Make sure you're in project root
pwd  # Should show /path/to/SH-Learning-AI

# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use the scripts which handle this automatically
python scripts/record_game.py
```

### Tests Failing

```bash
# Update dependencies
pip install -r requirements-dev.txt

# Run tests with verbose output
pytest -v

# Run specific test
pytest tests/test_event_recorder.py -v
```

## Next Steps

- **Read the [API Documentation](API.md)** for detailed API reference
- **Check [Best Practices](BEST_PRACTICES.md)** for coding guidelines
- **See [Architecture](ARCHITECTURE.md)** to understand system design
- **Review [CLAUDE.md](../CLAUDE.md)** for development standards

## Getting Help

- Check existing [documentation](../docs/)
- Review [example scripts](../scripts/)
- Look at [test files](../tests/) for usage examples
- Open an issue on GitHub

## Summary

**To record a mission**:
```bash
python scripts/record_mission_with_audio.py --host http://YOUR_SERVER_IP:1864
```

**To generate a report**:
```bash
python scripts/generate_mission_report.py --mission data/game_recordings/GAME_*
```

That's it! You're ready to capture and analyze Starship Horizons missions.
