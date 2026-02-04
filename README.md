# Starship Horizons Learning AI

An AI-powered telemetry and game integration system for Starship Horizons bridge simulator, designed to capture, analyze, and learn from game sessions.

## 🎯 Project Overview

This project provides real-time telemetry capture and analysis for Starship Horizons, enabling:
- WebSocket-based game state monitoring
- Audio recording and transcription of crew communications
- Mission event tracking and analysis
- Automated session recording and playback

## 🚀 Current Features

### Core Functionality
- **WebSocket Integration**: Real-time connection to Starship Horizons game server
- **Telemetry Recording**: Capture and store game state, events, and metrics
- **Real-Time Audio Transcription**: Record and transcribe crew voice communications using local AI (Faster-Whisper)
- **Speaker Diarization**: Identify and track multiple bridge crew members
- **Engagement Analytics**: Measure crew participation, turn-taking, and communication effectiveness
- **Station Monitoring**: Track individual bridge station activities
- **Event Analysis**: Process and categorize game events
- **Synchronized Timeline**: Combine game telemetry with audio transcripts

### Training & Performance Analysis
- **NASA Teamwork Framework**: Evaluate crew performance across 5 dimensions (Communication, Coordination, Leadership, Monitoring, Adaptability)
- **Kirkpatrick's Training Model**: Assess training effectiveness at 4 levels (Reaction, Learning, Behavior, Results)
- **Bloom's Taxonomy**: Analyze cognitive skill development from knowledge through creation
- **Automated Assessment Reports**: LLM-powered narrative reports combining quantitative metrics with qualitative insights
- **Mission Debriefing**: Comprehensive post-mission analysis with recommendations

### Implemented Modules

#### `src/integration/`
- `starship_horizons_client.py` - Base WebSocket client for game connection
- `enhanced_game_client.py` - Enhanced client with advanced filtering
- `browser_mimic_websocket.py` - Browser-compatible WebSocket implementation
- `station_handlers.py` - Console-specific event handlers
- `smart_filters.py` - Intelligent event filtering system

#### `src/audio/`
- `config.py` - Audio system configuration
- `capture.py` - Real-time audio capture with PyAudio and VAD
- `whisper_transcriber.py` - Local AI transcription using Faster-Whisper
- `speaker_diarization.py` - Speaker identification and engagement analytics
- `microphone_test.py` - Microphone testing utilities
- `speaker_test.py` - Audio output testing
- `test_audio.py` - Comprehensive audio system tests

#### `src/metrics/`
- `event_recorder.py` - Event logging and persistence
- `mission_summarizer.py` - Mission analysis and reporting
- `audio_transcript.py` - Voice transcription processing
- `learning_evaluator.py` - NASA/Kirkpatrick/Bloom's assessment frameworks

#### `src/llm/`
- `ollama_client.py` - Local LLM integration for report generation
- `prompt_templates.py` - Professional analysis prompts
- `hybrid_prompts.py` - Combined framework assessment prompts
- `story_prompts.py` - Narrative mission story generation

## 📋 Requirements

### System Requirements
- Python 3.11+
- WSL2 (for Windows users)
- PulseAudio or ALSA for audio support
- NVIDIA GPU (optional, for enhanced AI features)

### Network Requirements
- Access to Starship Horizons server (default port: 1864-1867)
- WebSocket support

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/SH-Learning-AI.git
cd SH-Learning-AI
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **For development**
```bash
pip install -r requirements-dev.txt
```

## 🏃 Quick Start

### Basic WebSocket Connection
```python
from src.integration.starship_horizons_client import StarshipHorizonsClient

# Uses environment variables from .env file
client = StarshipHorizonsClient()
await client.connect()

# Or override with explicit host
client = StarshipHorizonsClient(host="http://192.168.68.55:1864")
await client.connect()
```

### Recording a Game Session

**With Audio Transcription** (Recommended):
```bash
# Record mission with real-time audio transcription
python scripts/record_mission_with_audio.py --host http://192.168.68.55:1864

# 30-second test recording
python scripts/record_mission_with_audio.py --host http://192.168.68.55:1864 --duration 30
```

**Game Events Only** (No Audio):
```bash
# Uses IP from .env file
python scripts/record_game.py

# Or override with --host parameter
python scripts/record_game.py --host http://192.168.68.55:1864
```

### Testing Audio Transcription
```bash
# Test audio capture and VAD
python scripts/test_realtime_audio.py --mode capture --duration 10

# Test speaker diarization
python scripts/test_realtime_audio.py --mode diarization --duration 15

# Test full transcription pipeline
python scripts/test_realtime_audio.py --mode full --duration 30
```

### Running Tests
```bash
pytest tests/
```

## 📁 Project Structure

```
SH-Learning-AI/
├── src/                    # Source code
│   ├── integration/       # Game integration modules
│   ├── audio/            # Audio processing
│   └── metrics/          # Analytics and metrics
├── scripts/               # Utility scripts
│   ├── record_game.py
│   └── record_unified_telemetry.py
├── tests/                 # Test suite
├── docs/                  # Documentation
├── data/                  # Data storage
├── logs/                  # Application logs
└── .devcontainer/        # VS Code dev container config
```

## 📚 Documentation

Complete documentation is available in the `docs/` directory:

- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running in 10 minutes
- **[API Documentation](docs/API.md)** - Complete API reference for all modules
- **[Architecture](docs/ARCHITECTURE.md)** - System design and architecture overview
- **[Best Practices](docs/BEST_PRACTICES.md)** - Coding standards and guidelines
- **[Audio Setup Guide](docs/AUDIO_SETUP_GUIDE.md)** - Detailed audio transcription setup
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project
- **[Development Standards](CLAUDE.md)** - Mandatory coding standards

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root (or copy from `.env.example`):

```bash
# Game Server Configuration
# IMPORTANT: Update GAME_HOST with your Starship Horizons server IP
GAME_HOST=192.168.68.55
GAME_PORT_API=1864
GAME_PORT_WS=1865

# Audio Transcription Configuration
ENABLE_AUDIO_CAPTURE=true            # Enable/disable audio transcription
AUDIO_INPUT_DEVICE=0                 # Microphone device index
AUDIO_SAMPLE_RATE=16000              # Sample rate (16000 recommended)
AUDIO_CHANNELS=1                     # Mono (1) or Stereo (2)

# Whisper AI Model
WHISPER_MODEL_SIZE=large-v3          # tiny, base, small, medium, large-v3
WHISPER_DEVICE=cpu                   # cpu or cuda
WHISPER_COMPUTE_TYPE=int8            # int8, float16, float32

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION=true
SPEAKER_SIMILARITY_THRESHOLD=0.7

# Engagement Analytics
ENABLE_ENGAGEMENT_METRICS=true

# Recording Settings
RECORDING_PATH=./data/recordings
LOG_LEVEL=INFO
```

**Quick Setup:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and update GAME_HOST with your server's IP address
nano .env

# List available audio devices
python scripts/list_audio_devices.py

# Download Whisper model (first time only)
python scripts/download_whisper_models.py --model base
```

### Audio Setup Guide

For detailed audio transcription setup including:
- System dependencies installation
- Audio device configuration
- Whisper model selection
- Performance optimization
- Troubleshooting

See **[Audio Setup Guide](docs/AUDIO_SETUP_GUIDE.md)**

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

## 🗺️ Roadmap

### Phase 1: Foundation ✅ (Complete)
- ✅ WebSocket telemetry capture
- ✅ Real-time audio transcription (Faster-Whisper)
- ✅ Speaker diarization and identification
- ✅ Voice Activity Detection (VAD)
- ✅ Engagement analytics
- ✅ Basic event processing
- ✅ Session recording with synchronized timeline
- ✅ NASA Teamwork Framework analysis
- ✅ Kirkpatrick's Training Model evaluation
- ✅ Bloom's Taxonomy cognitive assessment
- ✅ LLM-powered report generation (Ollama)
- ✅ Automated mission debriefing with recommendations

### Phase 2: Intelligence (In Progress)
- [ ] AI crew member implementation
- [ ] Natural language command processing
- [ ] Real-time performance coaching
- [ ] Predictive action suggestions based on historical data

### Phase 3: Advanced Features (Planned)
- [ ] Multi-agent crew coordination
- [ ] Voice synthesis for AI crew responses
- [ ] Predictive action suggestions
- [ ] Training scenario generation
- [ ] Real-time crew coaching

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Development Notes

### VS Code Dev Container
This project includes a VS Code dev container configuration for consistent development environments. The container includes:
- Python 3.11
- Audio device support
- GPU acceleration (if available)
- Pre-configured extensions

### Audio Setup in WSL2
For audio support in WSL2, ensure PulseAudio is configured:
```bash
# Install PulseAudio
sudo apt-get update
sudo apt-get install -y pulseaudio portaudio19-dev python3-pyaudio

# Configure PulseAudio server (if needed)
export PULSE_SERVER=tcp:$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
```

### First-Time Audio Setup

```bash
# 1. List available audio devices
python scripts/list_audio_devices.py

# 2. Update AUDIO_INPUT_DEVICE in .env with correct device index

# 3. Download Whisper model (only needed once)
python scripts/download_whisper_models.py --model base

# 4. Test audio capture
python scripts/test_realtime_audio.py --mode capture --duration 10

# 5. Test full transcription
python scripts/test_realtime_audio.py --mode full --duration 30
```

## ⚠️ Known Issues

- **WSL2 Audio**: May require PulseAudio configuration (see Audio Setup in WSL2)
- **WebSocket**: Reconnection can be flaky with certain network configurations
- **Spectator Mode**: Some game events may not be captured
- **GPU Transcription**: CUDA support requires matching PyTorch/CUDA versions
- **Speaker ID**: Works best with clear, consistent audio quality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Starship Horizons development team
- Contributors and testers
- Open source community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This project is in active development. Features and APIs may change.