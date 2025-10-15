# Integrated Audio Transcription Implementation Plan - Part 2
## Starship Horizons Learning AI - Implementation Phases 3-8

**Continuation of:** INTEGRATED_AUDIO_IMPLEMENTATION_PLAN.md
**Date:** 2025-10-02

---

## Phase 3: Audio Capture Manager (NEW)

**File:** `src/audio/capture.py`

This is a **NEW** module that handles real-time audio capture with PyAudio.

```python
"""
Audio Capture Manager for Starship Horizons Bridge Audio.

Handles real-time audio capture from microphone using PyAudio,
with integrated Voice Activity Detection for speech segmentation.
"""

import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Callable
import numpy as np
import pyaudio
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AudioCaptureManager:
    """
    Manages real-time audio capture with VAD for bridge communications.

    Features:
    - Continuous audio streaming via PyAudio
    - Integrated Voice Activity Detection
    - Speaker segment extraction
    - Thread-safe operation
    - Automatic resource cleanup

    Designed for low-latency (<100ms) real-time operation.
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        chunk_ms: Optional[int] = None,
        device_index: Optional[int] = None,
        enable_vad: bool = True,
        vad: Optional[object] = None
    ):
        """
        Initialize audio capture manager.

        Args:
            sample_rate: Audio sample rate in Hz (default from env: 16000)
            channels: Number of audio channels (default from env: 1)
            chunk_ms: Audio chunk size in milliseconds (default from env: 100)
            device_index: PyAudio device index (default from env: 0)
            enable_vad: Enable voice activity detection
            vad: External VAD instance (if None, creates SimpleVAD)
        """
        # Load configuration from environment
        self.sample_rate = sample_rate or int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
        self.channels = channels or int(os.getenv('AUDIO_CHANNELS', '1'))
        chunk_ms = chunk_ms or int(os.getenv('AUDIO_CHUNK_MS', '100'))
        self.device_index = device_index or int(os.getenv('AUDIO_INPUT_DEVICE', '0'))

        # Calculate chunk size in frames
        self.chunk_size = int(self.sample_rate * chunk_ms / 1000)

        # PyAudio components
        self.audio = None
        self.stream = None

        # VAD integration
        self.enable_vad = enable_vad
        if enable_vad:
            if vad is None:
                # Import SimpleVAD from speaker_diarization
                from src.audio.speaker_diarization import SimpleVAD
                self.vad = SimpleVAD(sample_rate=self.sample_rate)
            else:
                self.vad = vad
        else:
            self.vad = None

        # State management
        self.is_capturing = False
        self.recording_start_time = None
        self._lock = threading.Lock()

        # Callback for audio segments
        self._segment_callback: Optional[Callable[[np.ndarray, float, float], None]] = None

        # Statistics
        self.total_chunks = 0
        self.total_segments = 0

        logger.info(
            f"AudioCaptureManager initialized: "
            f"{self.sample_rate}Hz, {self.channels}ch, "
            f"{chunk_ms}ms chunks, device={self.device_index}, VAD={enable_vad}"
        )

    def set_segment_callback(
        self,
        callback: Callable[[np.ndarray, float, float], None]
    ):
        """
        Set callback function for audio segments.

        Args:
            callback: Function that receives (audio_data, start_time, end_time)
        """
        self._segment_callback = callback
        logger.debug("Segment callback registered")

    def start_capture(self) -> bool:
        """
        Start audio capture.

        Returns:
            True if capture started successfully
        """
        with self._lock:
            if self.is_capturing:
                logger.warning("Audio capture already running")
                return True

            try:
                # Initialize PyAudio
                self.audio = pyaudio.PyAudio()

                logger.info("Opening audio stream...")

                # Open audio stream with callback
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )

                # Start stream
                self.stream.start_stream()
                self.is_capturing = True
                self.recording_start_time = time.time()

                logger.info("✓ Audio capture started")
                return True

            except Exception as e:
                logger.error(f"Failed to start audio capture: {e}")
                self._cleanup()
                return False

    def stop_capture(self):
        """Stop audio capture and cleanup resources."""
        with self._lock:
            if not self.is_capturing:
                return

            self.is_capturing = False

            # Wait for any final VAD segments
            if self.vad:
                time.sleep(0.5)

            self._cleanup()

            duration = time.time() - self.recording_start_time
            logger.info(
                f"✓ Audio capture stopped - "
                f"Duration: {duration:.1f}s, "
                f"Chunks: {self.total_chunks}, "
                f"Segments: {self.total_segments}"
            )

    def _cleanup(self):
        """Cleanup PyAudio resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            finally:
                self.stream = None

        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")
            finally:
                self.audio = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for audio stream.

        Called automatically by PyAudio when audio data is available.
        This runs in PyAudio's internal thread - must be fast (<10ms).
        """
        if status:
            logger.warning(f"PyAudio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Convert to float32 for processing
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Update counter
        self.total_chunks += 1

        # Calculate timestamp
        current_time = time.time() - self.recording_start_time

        # Process through VAD if enabled
        if self.vad:
            result = self.vad.process_chunk(audio_float, current_time)

            if result is not None:
                # Complete utterance detected
                audio_segment, start_time, end_time = result
                self.total_segments += 1

                logger.debug(
                    f"Segment detected: {start_time:.2f}s - {end_time:.2f}s "
                    f"({end_time - start_time:.2f}s)"
                )

                # Call callback if set
                if self._segment_callback:
                    try:
                        self._segment_callback(audio_segment, start_time, end_time)
                    except Exception as e:
                        logger.error(f"Segment callback error: {e}")
        else:
            # No VAD, pass chunks directly to callback
            if self._segment_callback:
                try:
                    self._segment_callback(audio_float, current_time, current_time + len(audio_float)/self.sample_rate)
                except Exception as e:
                    logger.error(f"Chunk callback error: {e}")

        return (None, pyaudio.paContinue)

    def get_device_info(self) -> dict:
        """
        Get information about the audio input device.

        Returns:
            Device information dictionary
        """
        if not self.audio:
            self.audio = pyaudio.PyAudio()
            cleanup = True
        else:
            cleanup = False

        try:
            device_info = self.audio.get_device_info_by_index(self.device_index)
            return device_info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
        finally:
            if cleanup:
                self.audio.terminate()
                self.audio = None

    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()


def list_audio_devices():
    """List all available audio input devices."""
    audio = pyaudio.PyAudio()

    print("\n" + "="*70)
    print("Available Audio Input Devices for Starship Horizons")
    print("="*70)

    input_devices = []

    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))

    if not input_devices:
        print("\n⚠ No input devices found!")
        print("Check your microphone connections and permissions.")
    else:
        for i, info in input_devices:
            print(f"\n{'='*70}")
            print(f"Device Index: {i}")
            print(f"Name: {info['name']}")
            print(f"Max Input Channels: {info['maxInputChannels']}")
            print(f"Default Sample Rate: {int(info['defaultSampleRate'])} Hz")
            host_api = audio.get_host_api_info_by_index(info['hostApi'])
            print(f"Host API: {host_api['name']}")

    print("\n" + "="*70)
    print("Configuration:")
    print("="*70)
    print("\nSet AUDIO_INPUT_DEVICE in .env to use a specific device.")
    print("Example: AUDIO_INPUT_DEVICE=0")
    print("")

    audio.terminate()
```

---

## Phase 4: Whisper Transcription Module (NEW)

**File:** `src/audio/whisper_transcriber.py`

This is a **NEW** module that provides local AI transcription using Faster-Whisper.

```python
"""
Whisper Transcription Module for Starship Horizons Bridge Audio.

Handles speech-to-text transcription using Faster-Whisper (local AI model).
No cloud dependency - all processing happens locally.
"""

import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from dotenv import load_dotenv

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("faster-whisper not installed. Transcription unavailable.")

load_dotenv()

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Local AI transcription using Faster-Whisper.

    Features:
    - Local model inference (no cloud dependency)
    - Real-time transcription via worker threads
    - Automatic language detection
    - Word-level timestamps
    - Memory-efficient processing
    - Configurable model size and precision

    Performance:
    - base model: ~7x realtime on CPU, ~70x on GPU
    - small model: ~4x realtime on CPU, ~40x on GPU
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cpu or cuda)
            compute_type: Compute precision (int8, float16, float32)
            language: Language code (en, es, fr, etc.) or 'auto'
            num_workers: Number of transcription worker threads
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper not installed. "
                "Run: pip install faster-whisper"
            )

        # Load configuration
        self.model_size = model_size or os.getenv('WHISPER_MODEL_SIZE', 'base')
        self.device = device or os.getenv('WHISPER_DEVICE', 'cpu')
        self.compute_type = compute_type or os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
        self.language = language or os.getenv('TRANSCRIBE_LANGUAGE', 'en')
        self.num_workers = num_workers or int(os.getenv('TRANSCRIPTION_WORKERS', '2'))

        # Model path
        model_path = Path(os.getenv('WHISPER_MODEL_PATH', './data/models/whisper/'))
        model_path.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path

        # Model instance (lazy loaded)
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._model_lock = threading.Lock()

        # Transcription queue and workers
        self._transcription_queue = queue.Queue(
            maxsize=int(os.getenv('MAX_SEGMENT_QUEUE_SIZE', '100'))
        )
        self._worker_threads: List[threading.Thread] = []
        self._is_running = False

        # Results storage
        self._results_lock = threading.Lock()
        self._pending_results = []

        logger.info(
            f"WhisperTranscriber initialized: "
            f"model={self.model_size}, device={self.device}, "
            f"compute={self.compute_type}, language={self.language}, "
            f"workers={self.num_workers}"
        )

    def load_model(self) -> bool:
        """
        Load the Whisper model into memory.

        Returns:
            True if model loaded successfully
        """
        with self._model_lock:
            if self._model_loaded:
                logger.debug("Model already loaded")
                return True

            try:
                logger.info(f"Loading Whisper model: {self.model_size}")
                start_time = time.time()

                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.model_path)
                )

                load_time = time.time() - start_time
                self._model_loaded = True

                logger.info(f"✓ Whisper model loaded in {load_time:.2f}s")

                # Warm up model with dummy audio
                self._warmup_model()

                return True

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                return False

    def _warmup_model(self):
        """Warm up model with dummy inference to avoid first-call latency."""
        try:
            logger.info("Warming up Whisper model...")
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
            segments, info = self._model.transcribe(dummy_audio)
            list(segments)  # Force transcription to complete
            logger.info("✓ Model warmed up")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def start_workers(self):
        """Start transcription worker threads."""
        if self._is_running:
            logger.warning("Transcription workers already running")
            return

        # Ensure model is loaded
        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load Whisper model")

        self._is_running = True

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._transcription_worker,
                args=(i,),
                daemon=True,
                name=f"WhisperWorker-{i}"
            )
            worker.start()
            self._worker_threads.append(worker)

        logger.info(f"Started {self.num_workers} transcription worker threads")

    def stop_workers(self):
        """Stop transcription worker threads."""
        if not self._is_running:
            return

        logger.info("Stopping transcription workers...")
        self._is_running = False

        # Send sentinel values to wake up all workers
        for _ in range(self.num_workers):
            try:
                self._transcription_queue.put(None, timeout=1)
            except queue.Full:
                pass

        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5)

        self._worker_threads.clear()
        logger.info("✓ Transcription workers stopped")

    def queue_audio(
        self,
        audio_data: np.ndarray,
        timestamp: float,
        speaker_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        """
        Queue audio for transcription.

        Args:
            audio_data: Audio samples (float32, normalized)
            timestamp: Timestamp of audio start
            speaker_id: Optional speaker identifier
            metadata: Optional metadata dictionary
        """
        try:
            self._transcription_queue.put_nowait({
                'audio': audio_data,
                'timestamp': timestamp,
                'speaker_id': speaker_id,
                'metadata': metadata or {}
            })
        except queue.Full:
            logger.warning("Transcription queue full, dropping audio segment")

    def _transcription_worker(self, worker_id: int):
        """
        Worker thread that processes transcription queue.

        Args:
            worker_id: Worker thread identifier
        """
        logger.info(f"Whisper worker {worker_id} started")

        while self._is_running:
            try:
                # Get item from queue (timeout to allow checking is_running)
                item = self._transcription_queue.get(timeout=1)

                # Check for sentinel
                if item is None:
                    break

                # Transcribe audio
                result = self._transcribe_segment(
                    item['audio'],
                    item['timestamp'],
                    item.get('speaker_id'),
                    item['metadata']
                )

                # Store result if valid
                if result:
                    with self._results_lock:
                        self._pending_results.append(result)

                # Mark task done
                self._transcription_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Whisper worker {worker_id} stopped")

    def _transcribe_segment(
        self,
        audio_data: np.ndarray,
        timestamp: float,
        speaker_id: Optional[str],
        metadata: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single audio segment.

        Args:
            audio_data: Audio samples (float32, normalized)
            timestamp: Timestamp of audio start
            speaker_id: Speaker identifier
            metadata: Metadata dictionary

        Returns:
            Transcription result dictionary or None
        """
        try:
            start_time = time.time()

            # Transcribe with Whisper
            segments, info = self._model.transcribe(
                audio_data,
                language=None if self.language == 'auto' else self.language,
                vad_filter=True,  # Use built-in VAD
                word_timestamps=True
            )

            # Extract text and words
            transcription_text = []
            word_segments = []

            for segment in segments:
                transcription_text.append(segment.text)

                # Extract word-level timestamps
                if hasattr(segment, 'words'):
                    for word in segment.words:
                        word_segments.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })

            full_text = ' '.join(transcription_text).strip()

            # Skip empty transcriptions
            if not full_text:
                return None

            transcription_time = time.time() - start_time

            # Calculate average confidence
            avg_confidence = np.mean([
                w['probability'] for w in word_segments
            ]) if word_segments else 0.0

            # Check confidence threshold
            min_confidence = float(os.getenv('MIN_TRANSCRIPTION_CONFIDENCE', '0.5'))
            if avg_confidence < min_confidence:
                logger.debug(
                    f"Transcription confidence too low: {avg_confidence:.2f} < {min_confidence}"
                )
                return None

            result = {
                'timestamp': timestamp,
                'text': full_text,
                'confidence': float(avg_confidence),
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'transcription_time': transcription_time,
                'words': word_segments,
                'speaker_id': speaker_id,
                'metadata': metadata
            }

            logger.debug(
                f"Transcribed: '{full_text[:50]}...' "
                f"(confidence: {avg_confidence:.2f}, time: {transcription_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def get_results(self, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get pending transcription results.

        Args:
            max_results: Maximum number of results to return (None = all)

        Returns:
            List of transcription result dictionaries
        """
        with self._results_lock:
            if max_results:
                results = self._pending_results[:max_results]
                self._pending_results = self._pending_results[max_results:]
            else:
                results = self._pending_results.copy()
                self._pending_results.clear()

            return results

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file (for batch processing).

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription result dictionary
        """
        if not self._model_loaded:
            self.load_model()

        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=None if self.language == 'auto' else self.language,
                vad_filter=True,
                word_timestamps=True
            )

            full_text = ' '.join([segment.text for segment in segments])

            return {
                'text': full_text,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            }

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return {
                'text': '',
                'error': str(e)
            }

    def __enter__(self):
        """Context manager entry."""
        self.start_workers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers()
```

---

## Phase 5: Update AudioTranscriptService

**File:** `src/metrics/audio_transcript.py` (MODIFICATIONS)

Update the existing file to integrate real audio capture and transcription.

### 5.1 Add Imports

Add these imports to the top of the file (after existing imports):

```python
# NEW IMPORTS (add these)
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()
```

### 5.2 Update `__init__` Method

Modify the `__init__` method to add new components (around line 21):

```python
def __init__(self, mission_id: str, sample_rate: int = 16000, channels: int = 1,
             buffer_duration: int = 30, auto_transcribe: bool = False):
    """
    Initialize audio transcript service.

    Args:
        mission_id: Unique mission identifier
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        buffer_duration: Duration of audio buffers in seconds
        auto_transcribe: Whether to automatically transcribe audio
    """
    self.mission_id = mission_id
    self.sample_rate = sample_rate
    self.channels = channels
    self.buffer_duration = buffer_duration
    self.auto_transcribe = auto_transcribe

    self.is_recording = False
    self.is_paused = False
    self.recording_start_time = None
    self.recording_end_time = None

    self.audio_segments = []
    self.transcripts = []
    self.current_buffer = []
    self.buffer_start_time = None

    self._lock = threading.Lock()
    self._transcription_queue = queue.Queue()
    self._transcription_thread = None
    self._transcription_active = False

    self.storage_path = None
    self._speaker_profiles = {}

    # NEW: Audio capture and transcription components
    self._capture_manager = None
    self._whisper_transcriber = None
    self._speaker_diarizer = None
    self._engagement_analyzer = None

    # NEW: Initialize components if enabled
    enable_audio = os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() == 'true'
    if enable_audio:
        self._initialize_audio_components()
```

### 5.3 Add New Method: `_initialize_audio_components`

Add this new method (insert after `__init__`):

```python
def _initialize_audio_components(self):
    """Initialize audio capture and transcription components."""
    try:
        # Import components
        from src.audio.capture import AudioCaptureManager
        from src.audio.whisper_transcriber import WhisperTranscriber
        from src.audio.speaker_diarization import SpeakerDiarizer, EngagementAnalyzer

        # Initialize speaker diarizer
        enable_diarization = os.getenv('ENABLE_SPEAKER_DIARIZATION', 'true').lower() == 'true'
        if enable_diarization:
            self._speaker_diarizer = SpeakerDiarizer()
            logger.info("Speaker diarizer initialized")

        # Initialize engagement analyzer
        enable_engagement = os.getenv('ENABLE_ENGAGEMENT_METRICS', 'true').lower() == 'true'
        if enable_engagement:
            self._engagement_analyzer = EngagementAnalyzer()
            logger.info("Engagement analyzer initialized")

        # Initialize Whisper transcriber
        enable_transcription = os.getenv('TRANSCRIBE_REALTIME', 'true').lower() == 'true'
        if enable_transcription:
            self._whisper_transcriber = WhisperTranscriber()
            logger.info("Whisper transcriber initialized")

        logger.info("✓ Audio components initialized")

    except ImportError as e:
        logger.warning(f"Audio components not available: {e}")
        self._speaker_diarizer = None
        self._whisper_transcriber = None
        self._engagement_analyzer = None
    except Exception as e:
        logger.error(f"Failed to initialize audio components: {e}")
```

### 5.4 Update `transcribe_audio` Method

Replace the mock implementation (around line 176):

```python
def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio file to text.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcription result with text and confidence
    """
    try:
        # Use Whisper transcriber if available
        if self._whisper_transcriber:
            result = self._whisper_transcriber.transcribe_file(audio_path)
            return {
                "text": result.get('text', ''),
                "confidence": result.get('language_probability', 0.0),
                "timestamp": datetime.now().isoformat(),
                "language": result.get('language', 'unknown'),
                "duration": result.get('duration', 0.0)
            }
        else:
            # Fallback to mock if Whisper not available
            logger.warning("Whisper transcriber not available, using mock")
            return {
                "text": "Mock transcription",
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {
            "text": "",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
```

### 5.5 Update `_transcription_worker` Method

Replace the empty worker (around line 214):

```python
def _transcription_worker(self) -> None:
    """Worker thread for processing transcription queue."""
    try:
        if not self._whisper_transcriber:
            logger.warning("Whisper transcriber not available")
            return

        logger.info("Transcription worker started")

        while self._transcription_active:
            try:
                # Get results from Whisper
                results = self._whisper_transcriber.get_results(max_results=10)

                for result in results:
                    # Extract data
                    timestamp_dt = datetime.fromtimestamp(
                        self.recording_start_time.timestamp() + result['timestamp']
                    )
                    speaker_id = result.get('speaker_id', 'Unknown')
                    text = result['text']
                    confidence = result['confidence']

                    # Get speaker display name
                    if self._speaker_diarizer:
                        speaker_name = self._speaker_diarizer.get_speaker_display_name(speaker_id)
                    else:
                        speaker_name = speaker_id

                    # Add to transcripts
                    self.add_transcript(
                        timestamp=timestamp_dt,
                        speaker=speaker_name,
                        text=text,
                        confidence=confidence
                    )

                    logger.info(f"[{speaker_name}] {text}")

                # Sleep briefly to avoid busy-waiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Transcription worker error: {e}")

    except Exception as e:
        logger.error(f"Transcription worker failed: {e}")
    finally:
        logger.info("Transcription worker stopped")
```

### 5.6 Add New Method: `start_audio_capture`

Add this new method:

```python
def start_audio_capture(self, device_index: Optional[int] = None) -> bool:
    """
    Start audio capture from microphone.

    Args:
        device_index: Audio device index (None = use default from env)

    Returns:
        True if capture started successfully
    """
    try:
        from src.audio.capture import AudioCaptureManager
        from src.audio.speaker_diarization import SpeakerSegment

        # Create capture manager
        self._capture_manager = AudioCaptureManager(
            sample_rate=self.sample_rate,
            channels=self.channels,
            device_index=device_index,
            enable_vad=True,
            vad=self._speaker_diarizer.vad if self._speaker_diarizer else None
        )

        # Set callback to process audio segments
        def audio_segment_callback(audio_data: np.ndarray, start_time: float, end_time: float):
            """Handle audio segments from capture manager."""
            try:
                # Identify speaker
                if self._speaker_diarizer:
                    speaker_id, confidence = self._speaker_diarizer.identify_speaker(audio_data)
                else:
                    speaker_id = "Unknown"
                    confidence = 1.0

                # Create segment
                segment = SpeakerSegment(
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    audio_data=audio_data,
                    confidence=confidence
                )

                # Update engagement stats
                if self._engagement_analyzer:
                    self._engagement_analyzer.update_speaker_stats(speaker_id, segment)

                # Queue for transcription
                if self._whisper_transcriber and self.auto_transcribe:
                    self._whisper_transcriber.queue_audio(
                        audio_data,
                        start_time,
                        speaker_id=speaker_id
                    )

                # Add to buffer
                self.add_audio_chunk(audio_data)

            except Exception as e:
                logger.error(f"Audio segment callback error: {e}")

        self._capture_manager.set_segment_callback(audio_segment_callback)

        # Start capture
        if self._capture_manager.start_capture():
            logger.info("✓ Audio capture started")
            return True
        else:
            logger.error("Failed to start audio capture")
            return False

    except ImportError:
        logger.error("Audio capture module not available")
        return False
    except Exception as e:
        logger.error(f"Audio capture initialization failed: {e}")
        return False
```

### 5.7 Add New Method: `stop_audio_capture`

Add this new method:

```python
def stop_audio_capture(self):
    """Stop audio capture."""
    if self._capture_manager:
        self._capture_manager.stop_capture()
        logger.info("Audio capture stopped")
```

### 5.8 Update `start_realtime_transcription` Method

Replace the existing method (around line 194):

```python
def start_realtime_transcription(self) -> bool:
    """
    Start realtime transcription thread.

    Returns:
        True if started successfully
    """
    with self._lock:
        if self._transcription_active:
            logger.warning("Realtime transcription already active")
            return True

        try:
            # Check if Whisper available
            if not self._whisper_transcriber:
                logger.error("Whisper transcriber not initialized")
                return False

            # Start Whisper workers
            self._whisper_transcriber.start_workers()

            # Start transcription worker
            self._transcription_active = True
            self._transcription_thread = threading.Thread(
                target=self._transcription_worker,
                daemon=True
            )
            self._transcription_thread.start()

            logger.info("✓ Realtime transcription started")
            return True

        except Exception as e:
            logger.error(f"Failed to start realtime transcription: {e}")
            return False
```

### 5.9 Update `stop_realtime_transcription` Method

Update to stop Whisper workers:

```python
def stop_realtime_transcription(self):
    """Stop realtime transcription thread."""
    with self._lock:
        self._transcription_active = False

    if self._transcription_thread:
        self._transcription_queue.put(None)  # Sentinel
        self._transcription_thread.join(timeout=1)

    # Stop Whisper workers
    if self._whisper_transcriber:
        self._whisper_transcriber.stop_workers()

    logger.info("Realtime transcription stopped")
```

### 5.10 Add New Method: `get_engagement_summary`

Add this new method for mission analytics:

```python
def get_engagement_summary(self) -> Dict[str, Any]:
    """
    Get engagement and communication summary for mission.

    Returns:
        Dictionary with engagement metrics
    """
    if not self._engagement_analyzer:
        return {}

    return self._engagement_analyzer.get_mission_communication_summary()
```

---

## Phase 6: Update GameRecorder Integration

**File:** `src/integration/game_recorder.py` (MODIFICATIONS)

Update the existing file to enable audio capture during missions.

### 6.1 Modify `start_recording` Method

Find the section that initializes `AudioTranscriptService` (around line 84-90) and update it:

```python
# Initialize audio service
self.audio_service = AudioTranscriptService(
    mission_id=self.mission_id,
    sample_rate=16000,
    channels=1,
    auto_transcribe=True  # Enable auto-transcription
)

# Start audio capture if enabled
try:
    enable_audio = os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() == 'true'

    if enable_audio:
        logger.info("Starting audio capture and transcription...")

        # Start realtime transcription worker
        if self.audio_service.start_realtime_transcription():
            # Start audio capture from microphone
            device_index_str = os.getenv('AUDIO_INPUT_DEVICE', '0')
            try:
                device_index = int(device_index_str)
            except ValueError:
                device_index = None

            if self.audio_service.start_audio_capture(device_index=device_index):
                logger.info("✓ Audio recording and transcription active")
                logger.info("Bridge crew communications being captured")
            else:
                logger.warning("Audio capture failed, continuing without audio")
        else:
            logger.warning("Transcription unavailable, continuing without audio")
    else:
        logger.info("Audio capture disabled (set ENABLE_AUDIO_CAPTURE=true to enable)")

except Exception as e:
    logger.warning(f"Audio initialization failed: {e}")
    logger.info("Continuing mission recording without audio")
```

### 6.2 Update `stop_recording` Method

Find the `stop_recording` method and add audio cleanup:

```python
def stop_recording(self) -> Dict[str, Any]:
    """
    Stop recording a game session.

    Returns:
        Summary dictionary of the recording session
    """
    if not self.is_recording:
        logger.warning("Not currently recording")
        return {}

    # Stop client polling
    self.client.stop_polling()

    # Close WebSocket
    self.client.disconnect_websocket()

    # NEW: Stop audio capture and transcription
    if self.audio_service:
        try:
            logger.info("Stopping audio recording...")

            # Stop audio capture
            self.audio_service.stop_audio_capture()

            # Stop transcription
            self.audio_service.stop_realtime_transcription()

            # Get final recording duration
            audio_duration = self.audio_service.stop_recording()
            logger.info(f"Audio recording duration: {audio_duration:.2f}s")

            # Get engagement summary
            engagement_summary = self.audio_service.get_engagement_summary()
            if engagement_summary:
                logger.info(
                    f"Captured {engagement_summary.get('total_utterances', 0)} utterances "
                    f"from {engagement_summary.get('total_speakers', 0)} speakers"
                )

            # Export transcripts
            recording_path = Path(os.getenv('RECORDING_PATH', './data/recordings'))
            recording_path.mkdir(parents=True, exist_ok=True)

            transcript_path = recording_path / f"{self.mission_id}_transcript.json"
            self.audio_service.export_transcript(transcript_path)
            logger.info(f"Transcripts exported to: {transcript_path}")

        except Exception as e:
            logger.error(f"Audio cleanup failed: {e}")

    # Get summary from event recorder
    summary = self.event_recorder.get_session_summary()

    self.is_recording = False
    logger.info(f"Stopped recording mission: {self.mission_id}")

    return summary
```

### 6.3 Add Method: `get_combined_timeline`

Add this new method to synchronize telemetry and audio:

```python
def get_combined_timeline(self) -> List[Dict[str, Any]]:
    """
    Get combined timeline of game events and audio transcripts.

    Returns:
        List of timeline entries sorted by timestamp
    """
    timeline = []

    # Add game events
    if self.event_recorder:
        events = self.event_recorder.events
        for event in events:
            timeline.append({
                'type': 'game_event',
                'timestamp': event['timestamp'],
                'data': event
            })

    # Add audio transcripts
    if self.audio_service:
        transcripts = self.audio_service.get_all_transcripts()
        for transcript in transcripts:
            timeline.append({
                'type': 'audio_transcript',
                'timestamp': transcript['timestamp'],
                'speaker': transcript.get('speaker', 'Unknown'),
                'text': transcript.get('text', ''),
                'confidence': transcript.get('confidence', 0.0)
            })

    # Sort by timestamp
    timeline.sort(key=lambda x: x['timestamp'])

    return timeline
```

---

Continuing in next message...
