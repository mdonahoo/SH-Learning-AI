# Starship Horizons Learning AI - AI/ML Technology Quick Reference

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      STARSHIP HORIZONS GAME                      │
│                    (WebSocket + HTTP API)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
[INTEGRATION LAYER]                    [INTEGRATION LAYER]
Game WebSocket Client              API Event Polling
└─────────────────────────────────────────────────┘
        │
        ├──────────────────┬──────────────────────────┐
        │                  │                          │
        ▼                  ▼                          ▼
[AUDIO LAYER]      [TELEMETRY LAYER]        [VIDEO LAYER]
├─ PyAudio         ├─ Event Recorder        └─ (Future)
├─ SimpleVAD       ├─ Mission Summarizer
├─ Whisper         └─ Timeline Gen
├─ Diarization
└─ Engagement
        │                  │
        └──────────────────┴──────────────────┐
                                              ▼
                                    [METRICS LAYER]
                                    ├─ Kirkpatrick
                                    ├─ Bloom's
                                    ├─ NASA
                                    └─ LearningEvaluator
                                              │
                                              ▼
                                    [LLM GENERATION]
                                    ├─ Ollama Client
                                    ├─ Hybrid Reports
                                    ├─ Stories
                                    └─ Analysis
                                              │
                                              ▼
                                    [OUTPUT]
                                    ├─ JSON
                                    ├─ Markdown
                                    └─ Narratives
```

---

## AI/ML Components Matrix

### Audio Processing (✅ Fully Implemented)

| Component | Technology | Type | Status |
|-----------|------------|------|--------|
| **Speech-to-Text** | Faster-Whisper v1.0+ | Neural (Transformer) | ✅ Production |
| **Speaker ID** | Spectral Features | Traditional ML | ✅ Production |
| **Advanced Speaker ID** | Pyannote 3.1 | Deep Learning | ✅ Available |
| **Voice Detection** | Energy + Whisper VAD | Hybrid | ✅ Production |
| **Engagement Analysis** | Custom metrics | Statistical | ✅ Production |

### Evaluation Frameworks (✅ Fully Implemented)

| Framework | Levels/Dimensions | Status |
|-----------|------------------|--------|
| **Kirkpatrick** | 4 levels (Reaction, Learning, Behavior, Results) | ✅ Production |
| **Bloom's Taxonomy** | 6 levels (Remember → Create) | ✅ Production |
| **NASA Teamwork** | 5 dimensions (Communication, Coordination, Leadership, Monitoring, Adaptability) | ✅ Production |

### LLM Integration (✅ Fully Implemented)

| Component | Technology | Status |
|-----------|------------|--------|
| **LLM Engine** | Ollama (Local) | ✅ Production |
| **Models** | Llama 3.2, Qwen 2.5 | ✅ Available |
| **Hybrid Reports** | Pre-computed + LLM | ✅ Production |
| **Story Generation** | Dialogue + LLM | ✅ Production |

### Computer Vision (⚠️ Not Implemented - Growth Opportunity)

| Component | Recommended Tech | Priority |
|-----------|-----------------|----------|
| **Emotion Detection** | DeepFace / FER | 🔴 High |
| **Pose Estimation** | MediaPipe | 🔴 High |
| **Gaze Tracking** | gaze-tracking | 🟡 Medium |
| **Screen Capture** | PyAutoGUI + OCR | 🟡 Medium |

---

## Technology Stack

### Core ML Libraries

```
Audio Transcription:
├─ faster-whisper >= 1.0.0
├─ torch >= 2.0.0
└─ torchaudio >= 2.0.0

Speaker Diarization:
├─ pyannote.audio >= 3.1.0
├─ scipy >= 1.10.0
└─ omegaconf >= 2.3.0

Voice Detection:
├─ webrtcvad >= 2.0.10
└─ numpy >= 1.26.4

LLM Integration:
├─ requests >= 2.31.0
└─ python-dotenv >= 1.0.0

Audio Capture:
├─ pyaudio == 0.2.14
└─ sounddevice == 0.4.6
```

### Key Models Deployed

1. **OpenAI Whisper (base model)**
   - Encoder-Decoder Transformer
   - 140M parameters
   - 99+ language support
   - 7x real-time (CPU), 70x (GPU)

2. **Pyannote Speaker Diarization 3.1**
   - End-to-end neural network
   - Trained on VoxCeleb
   - 192-dim embeddings
   - HuggingFace Hub

3. **Ollama-compatible LLMs**
   - Llama 3.2 (default)
   - Qwen 2.5:14b (alternative)
   - Local HTTP API

---

## Data Flow: Capture to Output

### Stage 1: Capture
```
Game Events (2 Hz)  ──┐
                       ├──→ [Event Recorder]
Microphone Audio (16kHz) ──┤
                       ├──→ [Audio Capture Manager]
                       
Result: Raw binary data + PCM audio
```

### Stage 2: Processing
```
Raw PCM Audio ──→ [VAD] ──→ [Whisper] ──→ [Diarization] ──→ Transcripts
                    ↓
              Speech segments
              
Raw Events ──→ [Filtering] ──→ [State Agg.] ──→ Normalized Events
                 
Result: Synchronized transcripts + events with metadata
```

### Stage 3: Analysis
```
Transcripts + Events ──→ [LearningEvaluator]
                          ├─ Kirkpatrick Analysis
                          ├─ Bloom's Analysis
                          ├─ NASA Analysis
                          └─ Mission Metrics
                          
Result: Pre-computed structured data (JSON)
```

### Stage 4: Narrative Generation
```
Pre-computed Data ──→ [Ollama] ──→ Markdown Report
     +
Actual Dialogue ──→ [Ollama] ──→ Mission Story
     +
Speaker Stats ──→ [Ollama] ──→ Crew Analysis

Result: Formatted narratives + reports
```

---

## Performance Characteristics

### Latency Budget

| Component | Latency | Constraint |
|-----------|---------|-----------|
| Audio Capture | <100ms | Real-time |
| VAD Processing | <5ms | Per chunk |
| Whisper Inference | 100-500ms | Per segment |
| Diarization | 10-50ms | Per segment |
| Analysis | <10ms | Post-hoc |
| **Total E2E** | **<700ms avg** | Acceptable |

### Memory Requirements

| Model | Size | Quantized |
|-------|------|-----------|
| Whisper (base) | 1.4 GB | 700 MB (int8) |
| Pyannote Diariztation | 500 MB | Not quantizable |
| Pyannote Embedding | 200 MB | 100 MB |
| **Total** | **2.1 GB** | **800 MB** |

### Scalability

```
Single Mission:
├─ ~50-200 utterances (typical training mission)
├─ ~5-10 speakers (bridge crew)
├─ ~30-60 minutes duration
└─ Processing time: 3-5 minutes (CPU)

Batch Processing:
├─ 10 missions: ~30-50 minutes
├─ 100 missions: 5-8 hours
└─ 1000 missions: 50-80 hours (CPU)

With GPU (NVIDIA CUDA):
└─ 10-20x faster
```

---

## Key Metrics Calculated

### Speaker Statistics
- Participation percentage (%) 
- Speaking time (seconds)
- Utterance count
- Average utterance duration
- Longest/shortest utterance

### Engagement Metrics
- Turn-taking rate (turns/minute)
- Response time (seconds)
- Interruption count
- Participation equity (0-100)

### Communication Patterns
- Dominant speaker (%)
- Speaker diversity
- Protocol adherence (%)
- Decision communications

### Cognitive Assessment
- Highest Bloom's level
- Cognitive distribution
- Knowledge level classification

### Team Performance (NASA)
- Communication score (0-100)
- Coordination score (0-100)
- Leadership score (0-100)
- Monitoring score (0-100)
- Adaptability score (0-100)

### Training Effectiveness (Kirkpatrick)
- Level 1: Engagement metrics
- Level 2: Learning outcomes
- Level 3: Behavior quality
- Level 4: Mission results

---

## Integration Points for Computer Vision

### Recommended Implementation Path

**Phase 1: Pose & Attention** (6-8 weeks)
```python
import mediapipe as mp

# Detect:
├─ Head direction
├─ Hand position  
├─ Posture confidence
└─ Overall attention level

# Correlate with:
├─ Speaking turns
├─ Response times
└─ Engagement patterns
```

**Phase 2: Emotion Recognition** (4-6 weeks)
```python
from deepface import DeepFace

# Detect:
├─ Facial expressions
├─ Emotion classification
├─ Emotion confidence
└─ Emotional arc

# Correlate with:
├─ Message urgency
├─ Decision confidence
└─ Communication tone
```

**Phase 3: Multi-Modal Fusion** (6-10 weeks)
```python
# Combine:
├─ Audio (transcripts + diarization)
├─ Video (emotions + pose)
├─ Game state (telemetry)
└─ LLM (unified narrative)

# Output:
└─ Holistic crew assessment
```

---

## Configuration Reference

### Key Environment Variables

```bash
# Audio Configuration
AUDIO_SAMPLE_RATE=16000              # Hz
AUDIO_CHANNELS=1                      # Mono
AUDIO_CHUNK_MS=100                    # Per chunk
AUDIO_INPUT_DEVICE=0                  # Device index

# Whisper Configuration
WHISPER_MODEL_SIZE=base               # tiny|base|small|medium|large-v3
WHISPER_DEVICE=cpu                    # cpu|cuda
WHISPER_COMPUTE_TYPE=int8             # int8|float16|float32
TRANSCRIBE_LANGUAGE=en                # Language code or 'auto'
MIN_TRANSCRIPTION_CONFIDENCE=0.5       # Threshold (0-1)

# Speaker Diarization
SPEAKER_SIMILARITY_THRESHOLD=0.7       # Threshold (0-1)
MIN_EXPECTED_SPEAKERS=1                # Min speakers
MAX_EXPECTED_SPEAKERS=6                # Max speakers

# Voice Activity Detection
VAD_ENERGY_THRESHOLD=500               # RMS threshold
MIN_SPEECH_DURATION=0.3                # Seconds
MIN_SILENCE_DURATION=0.5               # Seconds

# LLM Configuration
OLLAMA_HOST=http://localhost:11434     # Ollama server
OLLAMA_MODEL=llama3.2                  # Model name
OLLAMA_TIMEOUT=120                     # Seconds
```

---

## Production Checklist

- [x] Audio capture (PyAudio)
- [x] Voice detection (VAD)
- [x] Transcription (Whisper)
- [x] Speaker identification (Diarization)
- [x] Engagement analytics
- [x] Event recording
- [x] Mission analysis
- [x] Framework evaluation (Kirkpatrick, Bloom's, NASA)
- [x] LLM integration (Ollama)
- [x] Narrative generation
- [ ] Computer vision (emotion/pose) - HIGH PRIORITY
- [ ] Multi-modal fusion
- [ ] Real-time dashboard
- [ ] Distributed processing

---

## Documentation Links

| Document | Purpose |
|----------|---------|
| `/docs/ARCHITECTURE.md` | System architecture overview |
| `/docs/AUDIO_IMPLEMENTATION_SUMMARY.md` | Audio system details |
| `/docs/AUDIO_SETUP_GUIDE.md` | Audio setup instructions |
| `/docs/AUDIO_QUICK_REFERENCE.md` | Audio quick reference |
| `/docs/API.md` | API documentation |
| `/docs/BEST_PRACTICES.md` | Coding standards |
| **NEW:** `/docs/AI_ML_TECHNOLOGY_ANALYSIS.md` | This comprehensive report |

---

## Quick Start: Using AI Components

### Transcribe Audio
```python
from src.audio.whisper_transcriber import WhisperTranscriber

transcriber = WhisperTranscriber(model_size='base', device='cpu')
transcriber.load_model()
transcriber.start_workers()

# Queue audio segments
transcriber.queue_audio(audio_data, timestamp=0.0)

# Get results
results = transcriber.get_results()
```

### Identify Speakers
```python
from src.audio.speaker_diarization import SpeakerDiarizer

diarizer = SpeakerDiarizer()
speaker_id, confidence = diarizer.identify_speaker(audio_segment)
print(f"Speaker: {speaker_id} (confidence: {confidence:.2f})")
```

### Evaluate Mission
```python
from src.metrics.learning_evaluator import LearningEvaluator

evaluator = LearningEvaluator(events, transcripts)
results = evaluator.evaluate_all()

print(f"Kirkpatrick: {results['kirkpatrick']}")
print(f"Bloom's: {results['blooms_taxonomy']}")
print(f"NASA: {results['nasa_teamwork']}")
```

### Generate Report
```python
from src.llm.ollama_client import OllamaClient

client = OllamaClient()
report = client.generate_hybrid_report(structured_data, style='professional')
print(report)
```

---

## Support & Troubleshooting

### Whisper Issues
- GPU OOM: Use smaller model (tiny/base) or int8 quantization
- Slow inference: Reduce model size or enable GPU
- Poor transcription: Check audio quality (16kHz mono)

### Diarization Issues
- Too many false speakers: Increase similarity_threshold
- Missing speakers: Decrease similarity_threshold
- Neural model won't load: Ensure HuggingFace token set

### LLM Issues
- Connection refused: Check Ollama server running
- Out of memory: Use smaller model or reduce prompt length
- Poor quality: Adjust temperature (lower = more factual)

---

## References

- Whisper: https://github.com/openai/whisper
- Faster-Whisper: https://github.com/guillaumekln/faster-whisper
- Pyannote: https://github.com/pyannote/pyannote-audio
- Ollama: https://ollama.ai
- Kirkpatrick: https://en.wikipedia.org/wiki/Kirkpatrick%27s_four_levels_of_evaluation
- Bloom's Taxonomy: https://en.wikipedia.org/wiki/Bloom%27s_taxonomy
- NASA Teamwork: https://ntrs.nasa.gov/api/citations/19860011231/downloads/19860011231.pdf

---

**Last Updated**: 2025-11-02  
**Status**: Production Ready  
**Maintainer**: SH Learning AI Team
