# API Documentation

> **Last Updated**: 2025-10-15
>
> Complete reference for all public APIs in the Starship Horizons Learning AI system.

## Table of Contents

- [Integration Layer](#integration-layer)
  - [StarshipHorizonsClient](#starshiphorizonsclient)
  - [EnhancedGameClient](#enhancedgameclient)
  - [BrowserMimicWebSocket](#browsermimicwebsocket)
  - [GameRecorder](#gamerecorder)
- [Audio Layer](#audio-layer)
  - [AudioCapture](#audiocapture)
  - [WhisperTranscriber](#whispertranscriber)
  - [SpeakerDiarization](#speakerdiarization)
- [Metrics Layer](#metrics-layer)
  - [EventRecorder](#eventrecorder)
  - [MissionSummarizer](#missionsummarizer)
  - [AudioTranscriptService](#audiotranscriptservice)
  - [LearningEvaluator](#learningevaluator)
- [LLM Layer](#llm-layer)
  - [OllamaClient](#ollamaclient)

---

## Integration Layer

### StarshipHorizonsClient

**Location**: `src/integration/starship_horizons_client.py`

Base WebSocket client for connecting to Starship Horizons game server.

#### Constructor

```python
StarshipHorizonsClient(
    host: Optional[str] = None,
    port: Optional[int] = None,
    use_ssl: bool = False
)
```

**Parameters**:
- `host` (str, optional): Server hostname or IP. Defaults to `GAME_HOST` env var
- `port` (int, optional): Server port. Defaults to `GAME_PORT_WS` env var
- `use_ssl` (bool): Whether to use WSS instead of WS. Default: False

**Environment Variables**:
- `GAME_HOST`: Server hostname (default: `localhost`)
- `GAME_PORT_WS`: WebSocket port (default: `1865`)

#### Methods

##### `connect()`

```python
async def connect() -> bool
```

Establish WebSocket connection to the game server.

**Returns**: `True` if connection successful, `False` otherwise

**Raises**: `ConnectionError` if unable to connect after retries

**Example**:
```python
client = StarshipHorizonsClient()
if await client.connect():
    print("Connected successfully")
```

##### `disconnect()`

```python
async def disconnect() -> None
```

Gracefully close the WebSocket connection.

##### `send_message()`

```python
async def send_message(message: dict) -> bool
```

Send a message to the game server.

**Parameters**:
- `message` (dict): JSON-serializable message

**Returns**: `True` if sent successfully

##### `receive_message()`

```python
async def receive_message() -> Optional[dict]
```

Receive and parse next message from server.

**Returns**: Parsed message dict, or `None` if connection closed

---

### EnhancedGameClient

**Location**: `src/integration/enhanced_game_client.py`

Extended client with intelligent event filtering and station-specific handling.

#### Constructor

```python
EnhancedGameClient(
    host: Optional[str] = None,
    station: Optional[str] = None,
    enable_filters: bool = True
)
```

**Parameters**:
- `host` (str, optional): Server URL
- `station` (str, optional): Target station (e.g., "Tactical", "Engineering")
- `enable_filters` (bool): Enable smart filtering. Default: True

#### Methods

##### `start_recording()`

```python
async def start_recording(
    output_dir: Optional[Path] = None,
    filter_mode: str = "balanced"
) -> Path
```

Start recording filtered game events.

**Parameters**:
- `output_dir` (Path, optional): Directory for recordings
- `filter_mode` (str): Filter aggressiveness - "minimal", "balanced", "aggressive"

**Returns**: Path to recording directory

**Example**:
```python
client = EnhancedGameClient(station="Tactical")
await client.connect()
recording_path = await client.start_recording(filter_mode="balanced")
```

##### `stop_recording()`

```python
async def stop_recording() -> dict
```

Stop recording and get statistics.

**Returns**: Dict with event counts and filter statistics

---

### BrowserMimicWebSocket

**Location**: `src/integration/browser_mimic_websocket.py`

WebSocket client that mimics browser behavior for maximum compatibility.

#### Constructor

```python
BrowserMimicWebSocket(
    host: str,
    port: int = 1865,
    console_role: Optional[str] = None
)
```

**Parameters**:
- `host` (str): Server hostname or IP
- `port` (int): WebSocket port. Default: 1865
- `console_role` (str, optional): Bridge console role

#### Methods

##### `connect_with_handshake()`

```python
async def connect_with_handshake() -> bool
```

Connect with full browser-like HTTP → WebSocket upgrade.

**Returns**: `True` if connected and handshake completed

---

### GameRecorder

**Location**: `src/integration/game_recorder.py`

High-level recording orchestrator combining telemetry and audio.

#### Constructor

```python
GameRecorder(
    host: str,
    enable_audio: bool = True,
    output_dir: Optional[Path] = None
)
```

**Parameters**:
- `host` (str): Game server URL
- `enable_audio` (bool): Enable audio transcription. Default: True
- `output_dir` (Path, optional): Output directory for recordings

**Environment Variables**:
- `ENABLE_AUDIO_CAPTURE`: Global audio enable/disable
- `RECORDING_PATH`: Default recording output path

#### Methods

##### `start()`

```python
async def start() -> Path
```

Start recording session (telemetry + audio).

**Returns**: Path to session directory

##### `stop()`

```python
async def stop() -> dict
```

Stop recording and finalize files.

**Returns**: Recording metadata and statistics

**Example**:
```python
recorder = GameRecorder(host="http://192.168.1.100:1864")
session_path = await recorder.start()

# ... mission runs ...

stats = await recorder.stop()
print(f"Recorded {stats['event_count']} events")
print(f"Audio segments: {stats['audio_segments']}")
```

---

## Audio Layer

### AudioCapture

**Location**: `src/audio/capture.py`

Real-time audio capture with Voice Activity Detection (VAD).

#### Constructor

```python
AudioCapture(
    device_index: Optional[int] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    vad_threshold: float = 0.08
)
```

**Parameters**:
- `device_index` (int, optional): Input device index. Auto-detect if None
- `sample_rate` (int): Sample rate in Hz. Default: 16000
- `channels` (int): Number of channels (1=mono, 2=stereo). Default: 1
- `vad_threshold` (float): Voice activity threshold (0.0-1.0). Default: 0.08

**Environment Variables**:
- `AUDIO_INPUT_DEVICE`: Device index
- `AUDIO_SAMPLE_RATE`: Sample rate
- `AUDIO_CHANNELS`: Channel count
- `VAD_ENERGY_THRESHOLD`: VAD threshold

#### Methods

##### `start()`

```python
def start() -> None
```

Start audio capture stream.

##### `stop()`

```python
def stop() -> None
```

Stop audio capture.

##### `get_speech_segment()`

```python
async def get_speech_segment(timeout: float = 30.0) -> Optional[np.ndarray]
```

Wait for and return next speech segment detected by VAD.

**Parameters**:
- `timeout` (float): Maximum wait time in seconds

**Returns**: Audio data as numpy array (float32), or `None` on timeout

**Example**:
```python
capture = AudioCapture(sample_rate=16000)
capture.start()

segment = await capture.get_speech_segment(timeout=10.0)
if segment is not None:
    print(f"Captured {len(segment) / 16000:.2f}s of speech")

capture.stop()
```

---

### WhisperTranscriber

**Location**: `src/audio/whisper_transcriber.py`

AI-powered speech-to-text using Faster-Whisper.

#### Constructor

```python
WhisperTranscriber(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"
)
```

**Parameters**:
- `model_size` (str): Model size - "tiny", "base", "small", "medium", "large-v3"
- `device` (str): Compute device - "cpu" or "cuda"
- `compute_type` (str): Precision - "int8", "float16", "float32"

**Environment Variables**:
- `WHISPER_MODEL_SIZE`: Model size
- `WHISPER_DEVICE`: Device
- `WHISPER_COMPUTE_TYPE`: Compute type
- `WHISPER_MODEL_PATH`: Model cache path

#### Methods

##### `transcribe()`

```python
def transcribe(
    audio: np.ndarray,
    language: str = "en"
) -> dict
```

Transcribe audio segment.

**Parameters**:
- `audio` (np.ndarray): Audio data (float32, mono)
- `language` (str): Language code. Default: "en"

**Returns**: Dict with:
  - `text` (str): Transcribed text
  - `confidence` (float): Average confidence (0.0-1.0)
  - `segments` (list): Detailed segment info

**Example**:
```python
transcriber = WhisperTranscriber(model_size="base")
result = transcriber.transcribe(audio_data)
print(f"Transcript: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

### SpeakerDiarization

**Location**: `src/audio/speaker_diarization.py`

Speaker identification and engagement analytics.

#### Constructor

```python
SpeakerDiarization(
    use_neural: bool = True,
    similarity_threshold: float = 0.7,
    expected_speakers: int = 6
)
```

**Parameters**:
- `use_neural` (bool): Use neural embeddings. Default: True
- `similarity_threshold` (float): Speaker matching threshold (0.0-1.0)
- `expected_speakers` (int): Expected number of bridge crew

**Environment Variables**:
- `USE_NEURAL_DIARIZATION`: Enable neural mode
- `SPEAKER_SIMILARITY_THRESHOLD`: Threshold
- `EXPECTED_BRIDGE_CREW`: Expected speaker count

#### Methods

##### `identify_speaker()`

```python
def identify_speaker(audio: np.ndarray) -> str
```

Identify or assign speaker ID for audio segment.

**Parameters**:
- `audio` (np.ndarray): Audio segment

**Returns**: Speaker ID (e.g., "Speaker_1", "Captain")

##### `get_engagement_metrics()`

```python
def get_engagement_metrics() -> dict
```

Calculate crew engagement statistics.

**Returns**: Dict with:
  - `speaker_talk_times` (dict): Speaking time per speaker (seconds)
  - `turn_counts` (dict): Number of speaking turns
  - `interruptions` (int): Detected interruptions
  - `avg_response_time` (float): Average response delay

---

## Metrics Layer

### EventRecorder

**Location**: `src/metrics/event_recorder.py`

Records and manages game events with timestamps.

#### Constructor

```python
EventRecorder(output_dir: Optional[Path] = None)
```

**Parameters**:
- `output_dir` (Path, optional): Output directory for event logs

#### Methods

##### `record_event()`

```python
def record_event(
    event_type: str,
    data: dict,
    timestamp: Optional[float] = None
) -> None
```

Record a game event.

**Parameters**:
- `event_type` (str): Event category (e.g., "combat", "navigation")
- `data` (dict): Event payload
- `timestamp` (float, optional): Custom timestamp (Unix epoch)

##### `export_events()`

```python
def export_events(
    format: str = "json",
    filter_by_type: Optional[str] = None
) -> Path
```

Export recorded events.

**Parameters**:
- `format` (str): Export format - "json" or "csv"
- `filter_by_type` (str, optional): Filter by event type

**Returns**: Path to exported file

---

### MissionSummarizer

**Location**: `src/metrics/mission_summarizer.py`

Analyzes mission data and generates comprehensive reports.

#### Constructor

```python
MissionSummarizer(
    mission_dir: Path,
    use_llm: bool = True
)
```

**Parameters**:
- `mission_dir` (Path): Path to mission recording directory
- `use_llm` (bool): Use LLM for narrative generation. Default: True

**Environment Variables**:
- `ENABLE_LLM_REPORTS`: Global LLM enable/disable

#### Methods

##### `generate_timeline()`

```python
def generate_timeline() -> List[dict]
```

Create chronological mission timeline.

**Returns**: List of timeline events with timestamps

##### `analyze_performance()`

```python
def analyze_performance() -> dict
```

Analyze crew and ship performance metrics.

**Returns**: Dict with:
  - `crew_metrics` (dict): Per-crew statistics
  - `system_usage` (dict): Ship system utilization
  - `tactical_summary` (dict): Combat statistics
  - `navigation_metrics` (dict): Navigation performance

##### `generate_report()`

```python
async def generate_report(style: str = "professional") -> str
```

Generate comprehensive mission report.

**Parameters**:
- `style` (str): Report style - "professional", "entertaining", "technical"

**Returns**: Markdown-formatted report

**Example**:
```python
summarizer = MissionSummarizer(
    mission_dir=Path("data/game_recordings/GAME_20251015_120000"),
    use_llm=True
)

report = await summarizer.generate_report(style="entertaining")
print(report)
```

---

### AudioTranscriptService

**Location**: `src/metrics/audio_transcript.py`

Manages audio recording and transcription for missions.

#### Constructor

```python
AudioTranscriptService(
    output_dir: Path,
    enable_diarization: bool = True
)
```

**Parameters**:
- `output_dir` (Path): Output directory for audio files and transcripts
- `enable_diarization` (bool): Enable speaker identification

#### Methods

##### `start_recording()`

```python
async def start_recording() -> None
```

Start audio capture and transcription pipeline.

##### `stop_recording()`

```python
async def stop_recording() -> dict
```

Stop recording and finalize transcripts.

**Returns**: Recording statistics

##### `export_transcript()`

```python
def export_transcript(format: str = "json") -> Path
```

Export transcript data.

**Parameters**:
- `format` (str): "json", "srt", or "txt"

**Returns**: Path to transcript file

---

### LearningEvaluator

**Location**: `src/metrics/learning_evaluator.py`

Comprehensive training and performance analysis using multiple assessment frameworks.

#### Constructor

```python
LearningEvaluator(
    mission_data: dict,
    events: List[dict],
    transcripts: List[dict]
)
```

**Parameters**:
- `mission_data` (dict): Mission metadata (duration, name, etc.)
- `events` (list): Game events from recording
- `transcripts` (list): Audio transcripts with speaker IDs

#### Methods

##### `evaluate_all()`

```python
def evaluate_all() -> dict
```

Run all evaluation frameworks and return comprehensive assessment.

**Returns**: Dict with:
  - `kirkpatrick` (dict): Kirkpatrick's 4-level training model
  - `blooms_taxonomy` (dict): Bloom's cognitive development analysis
  - `nasa_teamwork` (dict): NASA's 5-dimension teamwork framework
  - `mission_metrics` (dict): Starship Horizons operational metrics

##### `evaluate_nasa_teamwork()`

```python
def evaluate_nasa_teamwork() -> dict
```

Evaluate crew performance using NASA's Teamwork Framework.

**Returns**: Dict with NASA's 5 dimensions:
  - `communication` (dict):
    - `score` (float): 0-100 communication quality
    - `clarity_avg` (float): Average transcription confidence
    - `assessment` (str): excellent/good/needs_improvement
  - `coordination` (dict):
    - `score` (float): 0-100 coordination quality
    - `speaker_switches` (int): Turn-taking count
    - `assessment` (str): excellent/good/needs_improvement
  - `leadership` (dict):
    - `score` (float): 0-100 leadership clarity
    - `primary_speaker_percentage` (float): Dominant speaker %
    - `assessment` (str): clear/dominant/distributed
  - `monitoring` (dict):
    - `score` (float): 0-100 situational awareness
    - `status_communications` (int): Status update count
    - `assessment` (str): excellent/good/needs_improvement
  - `adaptability` (dict):
    - `score` (float): 0-100 problem-solving ability
    - `adaptation_communications` (int): Adaptation event count
    - `assessment` (str): excellent/good/limited
  - `overall_teamwork_score` (float): Average of all 5 dimensions
  - `interpretation` (str): Summary interpretation

**Example**:
```python
from src.metrics.learning_evaluator import LearningEvaluator

# Load mission data
evaluator = LearningEvaluator(
    mission_data={'duration': 1800, 'name': 'Training Mission Alpha'},
    events=mission_events,
    transcripts=mission_transcripts
)

# Get NASA teamwork analysis
nasa_results = evaluator.evaluate_nasa_teamwork()

print(f"Overall Teamwork: {nasa_results['overall_teamwork_score']}/100")
print(f"Communication: {nasa_results['communication']['score']}/100")
print(f"Coordination: {nasa_results['coordination']['score']}/100")
print(f"Leadership: {nasa_results['leadership']['score']}/100")
print(f"Monitoring: {nasa_results['monitoring']['score']}/100")
print(f"Adaptability: {nasa_results['adaptability']['score']}/100")

# Get all evaluations
all_results = evaluator.evaluate_all()
```

##### `evaluate_kirkpatrick()`

```python
def evaluate_kirkpatrick() -> dict
```

Assess training effectiveness using Kirkpatrick's 4-level model.

**Returns**: Dict with:
  - `level_1_reaction` (dict): Engagement and participation
  - `level_2_learning` (dict): Knowledge acquisition
  - `level_3_behavior` (dict): Skill application
  - `level_4_results` (dict): Mission success outcomes

##### `evaluate_blooms_taxonomy()`

```python
def evaluate_blooms_taxonomy() -> dict
```

Analyze cognitive skill levels demonstrated during mission.

**Returns**: Dict with cognitive levels:
  - `remember` (int): Recall/recognition count
  - `understand` (int): Comprehension count
  - `apply` (int): Application count
  - `analyze` (int): Analysis count
  - `evaluate` (int): Evaluation count
  - `create` (int): Synthesis/creation count
  - `distribution` (dict): Percentage per level
  - `cognitive_maturity` (str): Overall assessment

---

## LLM Layer

### OllamaClient

**Location**: `src/llm/ollama_client.py`

Client for Ollama LLM server integration.

#### Constructor

```python
OllamaClient(
    host: Optional[str] = None,
    model: str = "llama3.2",
    timeout: int = 120
)
```

**Parameters**:
- `host` (str, optional): Ollama server URL. Defaults to `OLLAMA_HOST` env var
- `model` (str): Model name. Default: "llama3.2"
- `timeout` (int): Request timeout in seconds

**Environment Variables**:
- `OLLAMA_HOST`: Server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Model name
- `OLLAMA_TIMEOUT`: Timeout

#### Methods

##### `check_connection()`

```python
async def check_connection() -> bool
```

Verify Ollama server is accessible.

**Returns**: `True` if server responds

##### `generate()`

```python
async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str
```

Generate text completion.

**Parameters**:
- `prompt` (str): User prompt
- `system_prompt` (str, optional): System instructions
- `temperature` (float): Sampling temperature (0.0-2.0). Default: 0.7
- `max_tokens` (int, optional): Maximum response length

**Returns**: Generated text

**Example**:
```python
client = OllamaClient()

report = await client.generate(
    prompt="Summarize this mission data: ...",
    system_prompt="You are a starship operations analyst.",
    temperature=0.5
)
```

##### `generate_mission_summary()`

```python
async def generate_mission_summary(
    events: List[dict],
    transcripts: List[dict],
    style: str = "professional"
) -> str
```

Generate narrative mission summary from data.

**Parameters**:
- `events` (list): Game events
- `transcripts` (list): Audio transcripts
- `style` (str): Report style

**Returns**: Markdown summary

---

## Error Handling

All async methods may raise:

- `ConnectionError`: Network/connection issues
- `TimeoutError`: Operation exceeded timeout
- `ValueError`: Invalid parameters
- `RuntimeError`: Unexpected runtime errors

All methods use structured logging. Check logs for detailed error information.

## Configuration Best Practices

1. **Use environment variables** - Never hardcode configuration
2. **Validate inputs** - All public methods validate parameters
3. **Handle errors** - Wrap API calls in try/except blocks
4. **Check connections** - Always verify connectivity before operations
5. **Clean up resources** - Use context managers or explicit cleanup

## See Also

- [Quick Start Guide](QUICK_START.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Best Practices](BEST_PRACTICES.md)
- [Testing Guide](TESTING.md)
