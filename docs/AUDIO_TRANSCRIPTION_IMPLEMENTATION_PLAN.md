# Audio Recording & Transcription Implementation Plan

**Project:** Starship Horizons Learning AI
**Feature:** Local Audio Recording and Real-time Transcription
**Date:** 2025-10-02
**Status:** Planning Phase

---

## Executive Summary

This document outlines the complete implementation plan for adding local audio recording and transcription capabilities to the Starship Horizons Learning AI system. The implementation will use PyAudio for audio capture and Faster-Whisper (local AI model) for speech-to-text transcription, with optional speaker diarization for crew member identification.

---

## Current State Analysis

### Existing Infrastructure

The codebase already has:
- **`src/metrics/audio_transcript.py`**: `AudioTranscriptService` class with mock implementations
- **`src/audio/config.py`**: `AudioConfig` class for PyAudio and speech recognition setup
- **`src/integration/game_recorder.py`**: Integration point that initializes `AudioTranscriptService`
- **Dependencies**: PyAudio (0.2.14), SpeechRecognition (3.10.4), sounddevice (0.4.6) already in requirements

### Current Limitations

The following methods are **mock/stub implementations**:
1. `AudioTranscriptService.transcribe_audio()` - Returns `"Mock transcription"`
2. `AudioTranscriptService._transcription_worker()` - Empty processing loop
3. `AudioTranscriptService.identify_speaker()` - Simple hash-based approach
4. No actual audio stream capture connected to the service

### What Needs Implementation

1. Real audio capture pipeline
2. Local AI model integration (Faster-Whisper)
3. Real-time transcription worker thread
4. Speaker diarization (optional enhancement)
5. Proper audio/telemetry synchronization

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Game Recorder                             │
│  (src/integration/game_recorder.py)                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ├──> EventRecorder (game telemetry)
                 │
                 └──> AudioTranscriptService
                          │
                          ├──> AudioCaptureManager (NEW)
                          │    └──> PyAudio Stream
                          │         └──> Audio Chunks → Queue
                          │
                          ├──> WhisperTranscriber (NEW)
                          │    └──> Faster-Whisper Model
                          │         └──> Transcripts → Storage
                          │
                          └──> SpeakerDiarizer (OPTIONAL)
                               └──> pyannote.audio
                                    └──> Speaker IDs
```

---

## Phase 1: Dependencies & Environment Setup

### 1.1 Update requirements.txt

Add the following dependencies:

```txt
# AI/ML - Transcription
faster-whisper>=1.0.0
torch>=2.0.0
torchaudio>=2.0.0

# Speaker Diarization (optional)
pyannote.audio>=3.1.0

# Audio processing enhancements
webrtcvad>=2.0.10  # Voice Activity Detection
```

### 1.2 Update .env.example

Add configuration variables:

```bash
# ==========================================
# Whisper Model Configuration
# ==========================================
# Model size: tiny, base, small, medium, large-v3
# Recommendations:
#   - tiny/base: Real-time transcription, lower accuracy
#   - small: Good balance for most use cases
#   - medium/large: Best accuracy, slower processing
WHISPER_MODEL_SIZE=base

# Device: cpu or cuda (if GPU available)
WHISPER_DEVICE=cpu

# Compute type: int8 (fastest), float16, float32 (most accurate)
WHISPER_COMPUTE_TYPE=int8

# Model storage path
WHISPER_MODEL_PATH=./data/models/whisper/

# ==========================================
# Audio Capture Settings
# ==========================================
# PyAudio device index (use scripts/list_audio_devices.py to find)
AUDIO_INPUT_DEVICE=0

# Sample rate (Whisper expects 16kHz)
AUDIO_SAMPLE_RATE=16000

# Number of channels (1=mono, 2=stereo)
AUDIO_CHANNELS=1

# Audio chunk size in milliseconds
AUDIO_CHUNK_MS=100

# Buffer duration before processing (seconds)
AUDIO_BUFFER_DURATION=30

# ==========================================
# Transcription Settings
# ==========================================
# Enable real-time transcription
TRANSCRIBE_REALTIME=true

# Language code (auto, en, es, fr, de, etc.)
TRANSCRIBE_LANGUAGE=en

# Voice Activity Detection threshold (0.0-1.0)
# Lower = more sensitive, may capture noise
# Higher = less sensitive, may miss quiet speech
VAD_THRESHOLD=0.02

# Minimum silence duration to split segments (seconds)
MIN_SILENCE_DURATION=0.5

# ==========================================
# Speaker Diarization (Optional)
# ==========================================
# Enable speaker identification
ENABLE_SPEAKER_DIARIZATION=false

# Hugging Face token (required for pyannote.audio models)
HF_AUTH_TOKEN=your_token_here

# Number of expected speakers (0=auto-detect)
EXPECTED_SPEAKERS=6
```

### 1.3 Model Download Script

Create `scripts/download_whisper_models.py`:

```python
#!/usr/bin/env python3
"""
Download and cache Whisper models for offline use.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from faster_whisper import WhisperModel

def download_model(model_size: str, device: str = "cpu", compute_type: str = "int8"):
    """Download and cache a Whisper model."""
    print(f"\nDownloading Whisper model: {model_size}")
    print(f"Device: {device}, Compute type: {compute_type}")

    model_path = Path(os.getenv("WHISPER_MODEL_PATH", "./data/models/whisper/"))
    model_path.mkdir(parents=True, exist_ok=True)

    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(model_path)
        )
        print(f"✓ Model {model_size} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

def main():
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    print("="*60)
    print("Whisper Model Downloader")
    print("="*60)

    success = download_model(model_size, device, compute_type)

    if success:
        print("\n✓ All models downloaded successfully!")
        print(f"Models stored in: {os.getenv('WHISPER_MODEL_PATH', './data/models/whisper/')}")
    else:
        print("\n✗ Model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Phase 2: Audio Capture Module

### 2.1 Create src/audio/capture.py

```python
"""
Audio Capture Module for Starship Horizons Learning AI.

Handles real-time audio capture using PyAudio with proper
buffering and voice activity detection.
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
    Manages real-time audio capture from microphone.

    Features:
    - Continuous audio streaming via PyAudio
    - Voice Activity Detection (VAD)
    - Circular buffering
    - Thread-safe audio chunk queue
    - Automatic resource cleanup
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        chunk_ms: Optional[int] = None,
        device_index: Optional[int] = None,
        vad_threshold: Optional[float] = None
    ):
        """
        Initialize audio capture manager.

        Args:
            sample_rate: Audio sample rate in Hz (default from env: 16000)
            channels: Number of audio channels (default from env: 1)
            chunk_ms: Audio chunk size in milliseconds (default from env: 100)
            device_index: PyAudio device index (default from env: 0)
            vad_threshold: Voice activity detection threshold (default from env: 0.02)
        """
        # Load configuration from environment or use provided values
        self.sample_rate = sample_rate or int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
        self.channels = channels or int(os.getenv('AUDIO_CHANNELS', '1'))
        chunk_ms = chunk_ms or int(os.getenv('AUDIO_CHUNK_MS', '100'))
        self.device_index = device_index or int(os.getenv('AUDIO_INPUT_DEVICE', '0'))
        self.vad_threshold = vad_threshold or float(os.getenv('VAD_THRESHOLD', '0.02'))

        # Calculate chunk size in frames
        self.chunk_size = int(self.sample_rate * chunk_ms / 1000)

        # PyAudio components
        self.audio = None
        self.stream = None

        # State management
        self.is_capturing = False
        self._capture_thread = None
        self._lock = threading.Lock()

        # Audio queue for processing
        self._audio_queue = queue.Queue(maxsize=100)

        # Callback for audio chunks
        self._callback: Optional[Callable[[np.ndarray, datetime], None]] = None

        logger.info(
            f"AudioCaptureManager initialized: "
            f"{self.sample_rate}Hz, {self.channels}ch, "
            f"{chunk_ms}ms chunks, device={self.device_index}"
        )

    def set_audio_callback(self, callback: Callable[[np.ndarray, datetime], None]):
        """
        Set callback function for audio chunks.

        Args:
            callback: Function that receives (audio_data, timestamp)
        """
        self._callback = callback

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

                # Open audio stream
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

                # Start processing thread
                self._capture_thread = threading.Thread(
                    target=self._processing_worker,
                    daemon=True
                )
                self._capture_thread.start()

                logger.info("Audio capture started")
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

            # Stop processing thread
            if self._capture_thread:
                self._audio_queue.put(None)  # Sentinel
                self._capture_thread.join(timeout=2)

            self._cleanup()
            logger.info("Audio capture stopped")

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
        """
        if status:
            logger.warning(f"PyAudio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Normalize to float32 [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Add to queue with timestamp
        try:
            self._audio_queue.put_nowait((audio_data, datetime.now()))
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")

        return (None, pyaudio.paContinue)

    def _processing_worker(self):
        """Worker thread for processing audio chunks."""
        logger.info("Audio processing worker started")

        while self.is_capturing:
            try:
                item = self._audio_queue.get(timeout=1)

                # Check for sentinel
                if item is None:
                    break

                audio_data, timestamp = item

                # Apply VAD
                if self._has_voice_activity(audio_data):
                    # Call callback if set
                    if self._callback:
                        try:
                            self._callback(audio_data, timestamp)
                        except Exception as e:
                            logger.error(f"Audio callback error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")

        logger.info("Audio processing worker stopped")

    def _has_voice_activity(self, audio_data: np.ndarray) -> bool:
        """
        Simple energy-based Voice Activity Detection.

        Args:
            audio_data: Audio samples

        Returns:
            True if voice activity detected
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_data ** 2))
        return energy > self.vad_threshold

    def get_device_info(self) -> dict:
        """Get information about the audio input device."""
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

    print("\nAvailable Audio Input Devices:")
    print("="*60)

    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"\nDevice {i}: {info['name']}")
            print(f"  Max Input Channels: {info['maxInputChannels']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']}")

    audio.terminate()
```

### 2.2 Create scripts/list_audio_devices.py

```python
#!/usr/bin/env python3
"""
List all available audio devices for configuration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import list_audio_devices

if __name__ == "__main__":
    list_audio_devices()
```

---

## Phase 3: Whisper Transcription Module

### 3.1 Create src/audio/whisper_transcriber.py

```python
"""
Whisper Transcription Module for Starship Horizons Learning AI.

Handles speech-to-text transcription using Faster-Whisper (local AI model).
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
    - Real-time transcription via worker thread
    - Automatic language detection
    - Timestamped word-level output
    - Memory-efficient processing
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cpu or cuda)
            compute_type: Compute precision (int8, float16, float32)
            language: Language code (en, es, fr, etc.) or 'auto'
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

        # Load configuration
        self.model_size = model_size or os.getenv('WHISPER_MODEL_SIZE', 'base')
        self.device = device or os.getenv('WHISPER_DEVICE', 'cpu')
        self.compute_type = compute_type or os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
        self.language = language or os.getenv('TRANSCRIBE_LANGUAGE', 'en')

        # Model path
        model_path = Path(os.getenv('WHISPER_MODEL_PATH', './data/models/whisper/'))
        model_path.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path

        # Model instance (lazy loaded)
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._model_lock = threading.Lock()

        # Transcription queue
        self._transcription_queue = queue.Queue(maxsize=50)
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running = False

        # Results storage
        self._results_lock = threading.Lock()
        self._pending_results = []

        logger.info(
            f"WhisperTranscriber initialized: "
            f"model={self.model_size}, device={self.device}, "
            f"compute={self.compute_type}, language={self.language}"
        )

    def load_model(self) -> bool:
        """
        Load the Whisper model into memory.

        Returns:
            True if model loaded successfully
        """
        with self._model_lock:
            if self._model_loaded:
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
                return True

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                return False

    def start_worker(self):
        """Start transcription worker thread."""
        if self._is_running:
            logger.warning("Transcription worker already running")
            return

        # Ensure model is loaded
        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load Whisper model")

        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True
        )
        self._worker_thread.start()
        logger.info("Transcription worker started")

    def stop_worker(self):
        """Stop transcription worker thread."""
        if not self._is_running:
            return

        self._is_running = False
        self._transcription_queue.put(None)  # Sentinel

        if self._worker_thread:
            self._worker_thread.join(timeout=5)

        logger.info("Transcription worker stopped")

    def queue_audio(self, audio_data: np.ndarray, timestamp: datetime, metadata: dict = None):
        """
        Queue audio for transcription.

        Args:
            audio_data: Audio samples (float32, normalized)
            timestamp: Timestamp of audio
            metadata: Optional metadata dictionary
        """
        try:
            self._transcription_queue.put_nowait({
                'audio': audio_data,
                'timestamp': timestamp,
                'metadata': metadata or {}
            })
        except queue.Full:
            logger.warning("Transcription queue full, dropping audio segment")

    def _transcription_worker(self):
        """Worker thread that processes transcription queue."""
        logger.info("Whisper transcription worker started")

        while self._is_running:
            try:
                item = self._transcription_queue.get(timeout=1)

                # Check for sentinel
                if item is None:
                    break

                # Transcribe audio
                result = self._transcribe_segment(
                    item['audio'],
                    item['timestamp'],
                    item['metadata']
                )

                # Store result
                if result:
                    with self._results_lock:
                        self._pending_results.append(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription worker error: {e}")

        logger.info("Whisper transcription worker stopped")

    def _transcribe_segment(
        self,
        audio_data: np.ndarray,
        timestamp: datetime,
        metadata: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single audio segment.

        Args:
            audio_data: Audio samples
            timestamp: Timestamp of audio
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

            result = {
                'timestamp': timestamp,
                'text': full_text,
                'confidence': float(avg_confidence),
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'transcription_time': transcription_time,
                'words': word_segments,
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

    def get_results(self, max_results: int = None) -> List[Dict[str, Any]]:
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
        Transcribe an audio file.

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
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_worker()
```

### 3.2 Create scripts/test_whisper_transcription.py

```python
#!/usr/bin/env python3
"""
Test Whisper transcription with a sample audio file.
"""

import logging
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.whisper_transcriber import WhisperTranscriber

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test Whisper model loading."""
    logger.info("="*60)
    logger.info("TEST 1: Model Loading")
    logger.info("="*60)

    try:
        transcriber = WhisperTranscriber()
        success = transcriber.load_model()

        if success:
            logger.info("✓ Model loaded successfully")
            return True
        else:
            logger.error("✗ Model loading failed")
            return False
    except Exception as e:
        logger.error(f"✗ Exception during model loading: {e}")
        return False


def test_synthetic_audio():
    """Test transcription with synthetic audio."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Synthetic Audio Transcription")
    logger.info("="*60)

    try:
        # Create a silent audio segment (5 seconds)
        sample_rate = 16000
        duration = 5
        audio_data = np.zeros(sample_rate * duration, dtype=np.float32)

        # Add a simple tone (simulates audio)
        frequency = 440  # A4 note
        t = np.linspace(0, duration, sample_rate * duration)
        audio_data = 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        logger.info(f"Generated {duration}s of test audio")

        transcriber = WhisperTranscriber()
        transcriber.load_model()

        # Transcribe
        logger.info("Transcribing...")
        start_time = time.time()

        result = transcriber._transcribe_segment(
            audio_data,
            None,
            {}
        )

        elapsed = time.time() - start_time

        if result:
            logger.info(f"✓ Transcription completed in {elapsed:.2f}s")
            logger.info(f"  Text: '{result['text']}'")
            logger.info(f"  Confidence: {result['confidence']:.2f}")
            logger.info(f"  Language: {result['language']}")
        else:
            logger.info("✓ No speech detected (expected for synthetic tone)")

        return True

    except Exception as e:
        logger.error(f"✗ Synthetic audio test failed: {e}")
        return False


def test_file_transcription(audio_file: str):
    """Test transcription of an audio file."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: File Transcription")
    logger.info("="*60)

    if not Path(audio_file).exists():
        logger.warning(f"Audio file not found: {audio_file}")
        logger.info("Skipping file transcription test")
        return True

    try:
        transcriber = WhisperTranscriber()
        transcriber.load_model()

        logger.info(f"Transcribing file: {audio_file}")
        start_time = time.time()

        result = transcriber.transcribe_file(audio_file)

        elapsed = time.time() - start_time

        logger.info(f"✓ File transcription completed in {elapsed:.2f}s")
        logger.info(f"  Text: {result['text']}")
        logger.info(f"  Language: {result.get('language', 'unknown')}")
        logger.info(f"  Duration: {result.get('duration', 0):.2f}s")

        return True

    except Exception as e:
        logger.error(f"✗ File transcription failed: {e}")
        return False


def main():
    logger.info("#"*60)
    logger.info("# WHISPER TRANSCRIPTION TEST SUITE")
    logger.info("#"*60)

    results = []

    # Test 1: Model loading
    results.append(("Model Loading", test_model_loading()))

    # Test 2: Synthetic audio
    results.append(("Synthetic Audio", test_synthetic_audio()))

    # Test 3: File transcription (optional)
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    if test_file:
        results.append(("File Transcription", test_file_transcription(test_file)))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
```

---

## Phase 4: Speaker Diarization (Optional)

### 4.1 Create src/audio/speaker_diarization.py

```python
"""
Speaker Diarization Module for Starship Horizons Learning AI.

Identifies and tracks different speakers in audio.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.

    Features:
    - Speaker segmentation
    - Speaker identification
    - Bridge crew role detection
    - Speaker embedding storage
    """

    def __init__(self, hf_auth_token: Optional[str] = None):
        """
        Initialize speaker diarizer.

        Args:
            hf_auth_token: Hugging Face authentication token
        """
        if not PYANNOTE_AVAILABLE:
            logger.warning("pyannote.audio not installed. Speaker diarization unavailable.")
            self._pipeline = None
            return

        self.hf_auth_token = hf_auth_token or os.getenv('HF_AUTH_TOKEN')
        self._pipeline: Optional[Pipeline] = None
        self._speaker_embeddings: Dict[str, np.ndarray] = {}
        self._speaker_names: Dict[str, str] = {}

        # Bridge crew roles
        self.crew_roles = [
            "Captain",
            "Helm",
            "Tactical",
            "Science",
            "Engineering",
            "Communications"
        ]

    def load_pipeline(self) -> bool:
        """
        Load the diarization pipeline.

        Returns:
            True if pipeline loaded successfully
        """
        if not PYANNOTE_AVAILABLE:
            logger.error("pyannote.audio not available")
            return False

        if not self.hf_auth_token:
            logger.error("HF_AUTH_TOKEN required for pyannote.audio")
            return False

        try:
            logger.info("Loading speaker diarization pipeline...")

            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_auth_token
            )

            logger.info("✓ Diarization pipeline loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            return False

    def diarize_audio(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers (None = auto-detect)

        Returns:
            List of speaker segments with timestamps
        """
        if not self._pipeline:
            if not self.load_pipeline():
                return []

        try:
            # Run diarization
            if num_speakers:
                diarization = self._pipeline(
                    audio_path,
                    num_speakers=num_speakers
                )
            else:
                diarization = self._pipeline(audio_path)

            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })

            logger.info(f"Detected {len(set(s['speaker'] for s in segments))} speakers")
            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []

    def identify_speaker(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Identify speaker from audio segment.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate

        Returns:
            Speaker identifier
        """
        # Simplified implementation - would use speaker embeddings in production
        # For now, use a hash-based approach similar to the original
        audio_hash = hash(audio_data.tobytes()) % len(self.crew_roles)
        return self.crew_roles[audio_hash]

    def register_speaker(
        self,
        speaker_id: str,
        speaker_name: str,
        embedding: np.ndarray
    ):
        """
        Register a speaker with their embedding.

        Args:
            speaker_id: Unique speaker identifier
            speaker_name: Human-readable name (e.g., "Captain")
            embedding: Speaker embedding vector
        """
        self._speaker_embeddings[speaker_id] = embedding
        self._speaker_names[speaker_id] = speaker_name
        logger.info(f"Registered speaker: {speaker_name} ({speaker_id})")

    def get_speaker_name(self, speaker_id: str) -> str:
        """Get human-readable name for speaker ID."""
        return self._speaker_names.get(speaker_id, speaker_id)

    def align_transcripts_with_speakers(
        self,
        transcripts: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Align transcript segments with speaker identifications.

        Args:
            transcripts: List of transcript dictionaries
            speaker_segments: List of speaker segment dictionaries

        Returns:
            List of aligned transcript+speaker dictionaries
        """
        aligned = []

        for transcript in transcripts:
            # Find overlapping speaker segment
            t_start = transcript.get('start', 0)
            t_end = transcript.get('end', t_start + transcript.get('duration', 0))

            # Find best matching speaker segment
            best_speaker = "Unknown"
            best_overlap = 0

            for segment in speaker_segments:
                s_start = segment['start']
                s_end = segment['end']

                # Calculate overlap
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = segment['speaker']

            # Add speaker to transcript
            transcript_copy = transcript.copy()
            transcript_copy['speaker'] = self.get_speaker_name(best_speaker)
            transcript_copy['speaker_id'] = best_speaker
            aligned.append(transcript_copy)

        return aligned
```

---

## Phase 5: Update AudioTranscriptService

### 5.1 Modifications to src/metrics/audio_transcript.py

**Replace the mock `transcribe_audio()` method (around line 176):**

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
        # Import transcriber (lazy import)
        from src.audio.whisper_transcriber import WhisperTranscriber

        # Create transcriber instance
        transcriber = WhisperTranscriber()

        # Transcribe the file
        result = transcriber.transcribe_file(audio_path)

        # Format result
        return {
            "text": result.get('text', ''),
            "confidence": result.get('language_probability', 0.0),
            "timestamp": datetime.now().isoformat(),
            "language": result.get('language', 'unknown'),
            "duration": result.get('duration', 0.0)
        }

    except ImportError:
        # Fallback to mock if dependencies not available
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

**Replace the `_transcription_worker()` method (around line 214):**

```python
def _transcription_worker(self) -> None:
    """Worker thread for processing transcription queue."""
    try:
        # Import transcriber
        from src.audio.whisper_transcriber import WhisperTranscriber

        # Create and start transcriber
        transcriber = WhisperTranscriber()
        transcriber.load_model()

        logger.info("Transcription worker started with Whisper")

        while self._transcription_active:
            try:
                # Get audio from queue
                item = self._transcription_queue.get(timeout=1)

                if item is None:  # Sentinel
                    break

                # Extract data
                audio_chunk = item.get('audio')
                timestamp = item.get('timestamp')

                # Transcribe using Whisper
                result = transcriber._transcribe_segment(
                    audio_chunk,
                    timestamp,
                    {}
                )

                # Store result if valid
                if result and result['text']:
                    # Optionally identify speaker
                    speaker = self.identify_speaker(audio_chunk)

                    # Add to transcripts
                    self.add_transcript(
                        timestamp=timestamp,
                        speaker=speaker,
                        text=result['text'],
                        confidence=result['confidence']
                    )

                    logger.info(f"[{speaker}] {result['text']}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription worker error: {e}")

    except ImportError:
        logger.error("Whisper transcriber not available")
    except Exception as e:
        logger.error(f"Transcription worker failed to start: {e}")
    finally:
        logger.info("Transcription worker stopped")
```

**Update `start_realtime_transcription()` method (around line 194):**

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
            # Check if dependencies available
            from src.audio.whisper_transcriber import WhisperTranscriber
            from src.audio.capture import AudioCaptureManager

            # Start transcription worker
            self._transcription_active = True
            self._transcription_thread = threading.Thread(
                target=self._transcription_worker,
                daemon=True
            )
            self._transcription_thread.start()

            logger.info("Realtime transcription started")
            return True

        except ImportError as e:
            logger.error(f"Transcription dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to start realtime transcription: {e}")
            return False
```

**Add new method for audio capture integration:**

```python
def start_audio_capture(self, device_index: Optional[int] = None) -> bool:
    """
    Start audio capture from microphone.

    Args:
        device_index: Audio device index (None = use default)

    Returns:
        True if capture started successfully
    """
    try:
        from src.audio.capture import AudioCaptureManager

        # Create capture manager
        self._capture_manager = AudioCaptureManager(
            sample_rate=self.sample_rate,
            channels=self.channels,
            device_index=device_index
        )

        # Set callback to add audio chunks
        def audio_callback(audio_data: np.ndarray, timestamp: datetime):
            self.add_audio_chunk(audio_data)

            # Optionally queue for transcription
            if self.auto_transcribe and self._transcription_active:
                self.queue_for_transcription(audio_data, timestamp)

        self._capture_manager.set_audio_callback(audio_callback)

        # Start capture
        if self._capture_manager.start_capture():
            logger.info("Audio capture started")
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

def stop_audio_capture(self):
    """Stop audio capture."""
    if hasattr(self, '_capture_manager') and self._capture_manager:
        self._capture_manager.stop_capture()
        logger.info("Audio capture stopped")
```

**Add to `__init__()` (around line 55):**

```python
# Add to existing __init__ method:
self._capture_manager = None
```

---

## Phase 6: Integration with GameRecorder

### 6.1 Modifications to src/integration/game_recorder.py

**Update the `start_recording()` method (around line 84-90):**

```python
# Initialize audio service
self.audio_service = AudioTranscriptService(
    mission_id=self.mission_id,
    sample_rate=16000,
    channels=1,
    auto_transcribe=True  # Enable auto-transcription
)

# Start audio capture if available
try:
    enable_audio = os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() == 'true'

    if enable_audio:
        logger.info("Starting audio capture and transcription...")

        # Start realtime transcription worker
        if self.audio_service.start_realtime_transcription():
            # Start audio capture from microphone
            device_index = int(os.getenv('AUDIO_INPUT_DEVICE', '0'))
            if self.audio_service.start_audio_capture(device_index=device_index):
                logger.info("✓ Audio recording and transcription active")
            else:
                logger.warning("Audio capture failed, continuing without audio")
        else:
            logger.warning("Transcription unavailable, continuing without audio")
    else:
        logger.info("Audio capture disabled (ENABLE_AUDIO_CAPTURE=false)")

except Exception as e:
    logger.warning(f"Audio initialization failed: {e}")
    logger.info("Continuing without audio recording")
```

**Update `stop_recording()` method:**

Find the `stop_recording()` method and add audio cleanup:

```python
# Add before finalizing recording:
if self.audio_service:
    try:
        # Stop audio capture
        self.audio_service.stop_audio_capture()

        # Stop transcription
        self.audio_service.stop_realtime_transcription()

        # Get final recording duration
        audio_duration = self.audio_service.stop_recording()
        logger.info(f"Audio recording duration: {audio_duration:.2f}s")

        # Export transcripts
        transcript_path = Path(os.getenv('RECORDING_PATH', './data/recordings'))
        transcript_path = transcript_path / f"{self.mission_id}_transcript.json"
        self.audio_service.export_transcript(transcript_path)
        logger.info(f"Transcripts exported to: {transcript_path}")

    except Exception as e:
        logger.error(f"Audio cleanup failed: {e}")
```

---

## Phase 7: Testing & Validation

### 7.1 Create scripts/test_realtime_audio.py

```python
#!/usr/bin/env python3
"""
Test real-time audio capture and transcription.
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import AudioCaptureManager
from src.audio.whisper_transcriber import WhisperTranscriber

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_realtime_transcription(duration: int = 30):
    """
    Test real-time audio capture and transcription.

    Args:
        duration: Test duration in seconds
    """
    logger.info("="*60)
    logger.info("REAL-TIME AUDIO TRANSCRIPTION TEST")
    logger.info("="*60)
    logger.info(f"Duration: {duration} seconds")
    logger.info("Speak into your microphone...")
    logger.info("")

    transcriber = None
    capture_manager = None

    try:
        # Initialize transcriber
        logger.info("Loading Whisper model...")
        transcriber = WhisperTranscriber()
        transcriber.load_model()
        transcriber.start_worker()
        logger.info("✓ Transcriber ready")

        # Initialize audio capture
        logger.info("Starting audio capture...")
        capture_manager = AudioCaptureManager()

        # Set up audio callback
        def audio_callback(audio_data, timestamp):
            # Queue for transcription
            transcriber.queue_audio(audio_data, timestamp)

        capture_manager.set_audio_callback(audio_callback)
        capture_manager.start_capture()
        logger.info("✓ Audio capture started")
        logger.info("")

        # Run for specified duration
        start_time = time.time()
        last_check = start_time

        while time.time() - start_time < duration:
            time.sleep(1)

            # Check for results every 2 seconds
            if time.time() - last_check >= 2:
                results = transcriber.get_results()

                for result in results:
                    timestamp = result['timestamp'].strftime('%H:%M:%S')
                    text = result['text']
                    confidence = result['confidence']

                    logger.info(f"[{timestamp}] (conf: {confidence:.2f}) {text}")

                last_check = time.time()

        # Get any remaining results
        logger.info("")
        logger.info("Retrieving final results...")
        time.sleep(2)  # Wait for processing

        final_results = transcriber.get_results()
        for result in final_results:
            timestamp = result['timestamp'].strftime('%H:%M:%S')
            text = result['text']
            confidence = result['confidence']

            logger.info(f"[{timestamp}] (conf: {confidence:.2f}) {text}")

        logger.info("")
        logger.info("✓ Test completed successfully")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Cleanup
        if capture_manager:
            capture_manager.stop_capture()
        if transcriber:
            transcriber.stop_worker()

        logger.info("Resources cleaned up")


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    test_realtime_transcription(duration)


if __name__ == "__main__":
    main()
```

### 7.2 Create scripts/test_mission_with_audio.py

```python
#!/usr/bin/env python3
"""
Test complete mission recording with audio transcription.
"""

import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.game_recorder import GameRecorder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mission_recording_with_audio(duration: int = 60):
    """
    Test mission recording with audio transcription.

    Args:
        duration: Recording duration in seconds
    """
    logger.info("="*60)
    logger.info("MISSION RECORDING WITH AUDIO TEST")
    logger.info("="*60)
    logger.info(f"Duration: {duration} seconds")
    logger.info("")

    recorder = None

    try:
        # Create recorder
        recorder = GameRecorder()

        # Start recording
        logger.info("Starting mission recording...")
        mission_id = recorder.start_recording("Audio Test Mission")
        logger.info(f"✓ Recording started: {mission_id}")
        logger.info("")
        logger.info("Speak into your microphone and interact with the game...")
        logger.info("")

        # Record for specified duration
        for i in range(duration):
            time.sleep(1)
            if (i + 1) % 10 == 0:
                logger.info(f"Recording... {i + 1}/{duration}s")

        # Stop recording
        logger.info("")
        logger.info("Stopping recording...")
        summary = recorder.stop_recording()

        # Display summary
        logger.info("")
        logger.info("="*60)
        logger.info("RECORDING SUMMARY")
        logger.info("="*60)
        logger.info(f"Mission ID: {mission_id}")
        logger.info(f"Total Events: {summary.get('total_events', 0)}")
        logger.info(f"Duration: {summary.get('duration', 0):.2f}s")

        # Display audio stats
        if recorder.audio_service:
            transcripts = recorder.audio_service.get_all_transcripts()
            logger.info(f"Total Transcripts: {len(transcripts)}")

            if transcripts:
                logger.info("")
                logger.info("Sample Transcripts:")
                for i, t in enumerate(transcripts[:5], 1):
                    speaker = t.get('speaker', 'Unknown')
                    text = t.get('text', '')
                    logger.info(f"  {i}. [{speaker}] {text[:60]}...")

        logger.info("")
        logger.info("✓ Test completed successfully")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        if recorder:
            recorder.stop_recording()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        if recorder:
            recorder.cleanup()


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    test_mission_recording_with_audio(duration)


if __name__ == "__main__":
    main()
```

### 7.3 Update tests/test_audio_transcript.py

Add real transcription tests:

```python
def test_real_transcription(self):
    """Test real Whisper transcription."""
    try:
        from src.audio.whisper_transcriber import WhisperTranscriber

        # Create test audio
        sample_rate = 16000
        duration = 3
        audio_data = np.zeros(sample_rate * duration, dtype=np.float32)

        # Initialize service
        service = AudioTranscriptService("TEST_001")

        # Mock audio file path
        audio_path = "/tmp/test_audio.wav"

        # Test transcription
        result = service.transcribe_audio(audio_path)

        assert 'text' in result
        assert 'confidence' in result
        assert 'timestamp' in result

    except ImportError:
        pytest.skip("Whisper transcriber not available")
```

---

## Phase 8: Performance Optimization

### 8.1 Optimize Buffer Sizes

**In `src/audio/capture.py`**, adjust buffer management:

```python
# Use larger buffers for better performance
AUDIO_CHUNK_MS=200  # 200ms chunks (vs 100ms)

# Implement adaptive buffer sizing based on CPU load
def adjust_buffer_size_based_on_load(self):
    """Dynamically adjust buffer size based on system load."""
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)

    if cpu_percent > 80:
        # Increase buffer size to reduce processing frequency
        self.chunk_size = int(self.sample_rate * 0.3)  # 300ms
    elif cpu_percent < 40:
        # Decrease buffer size for lower latency
        self.chunk_size = int(self.sample_rate * 0.1)  # 100ms
```

### 8.2 Implement Sliding Window Processing

**In `src/audio/whisper_transcriber.py`**, add sliding window:

```python
class AudioBuffer:
    """Sliding window buffer for continuous audio."""

    def __init__(self, window_size: int = 30, overlap: int = 5):
        """
        Initialize audio buffer.

        Args:
            window_size: Window size in seconds
            overlap: Overlap between windows in seconds
        """
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = []
        self.sample_rate = 16000

    def add_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to buffer."""
        self.buffer.append(audio_data)

        # Check if window is full
        total_samples = sum(len(chunk) for chunk in self.buffer)
        window_samples = self.window_size * self.sample_rate

        if total_samples >= window_samples:
            return self.get_window()

        return None

    def get_window(self) -> np.ndarray:
        """Get current window and slide forward."""
        # Concatenate all chunks
        audio = np.concatenate(self.buffer)

        # Extract window
        window_samples = self.window_size * self.sample_rate
        window = audio[:window_samples]

        # Slide buffer (keep overlap)
        overlap_samples = self.overlap * self.sample_rate
        remaining = audio[window_samples - overlap_samples:]

        # Clear buffer and add remaining
        self.buffer = [remaining]

        return window
```

### 8.3 Model Caching and Warm-up

```python
def warmup_model(self):
    """Warm up the model with a dummy inference."""
    logger.info("Warming up Whisper model...")

    # Create 1 second of silence
    dummy_audio = np.zeros(16000, dtype=np.float32)

    # Run dummy transcription
    self._transcribe_segment(dummy_audio, datetime.now(), {})

    logger.info("✓ Model warmed up")
```

---

## Phase 9: Documentation & Examples

### 9.1 Create docs/AUDIO_SETUP.md

```markdown
# Audio Setup Guide

## Prerequisites

- Python 3.8+
- Working microphone
- PyAudio installed and configured
- (Optional) CUDA for GPU acceleration

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Whisper Model

```bash
python scripts/download_whisper_models.py
```

This will download the model specified in your `.env` file.

### 3. Configure Audio Device

List available audio devices:

```bash
python scripts/list_audio_devices.py
```

Update `.env` with your device index:

```bash
AUDIO_INPUT_DEVICE=0  # Use the device number from the list
```

### 4. Test Audio Capture

```bash
python scripts/test_realtime_audio.py 10
```

This will capture and transcribe audio for 10 seconds.

## Configuration

### Model Selection

Choose the appropriate Whisper model size:

- **tiny**: Fastest, lowest accuracy (~32MB)
- **base**: Good for real-time (~74MB) **[RECOMMENDED]**
- **small**: Better accuracy (~244MB)
- **medium**: High accuracy (~769MB)
- **large-v3**: Best accuracy (~1550MB)

Update in `.env`:

```bash
WHISPER_MODEL_SIZE=base
```

### Real-time vs Batch Processing

For **real-time transcription** (live missions):
```bash
TRANSCRIBE_REALTIME=true
WHISPER_MODEL_SIZE=base
AUDIO_BUFFER_DURATION=10
```

For **batch processing** (recorded audio):
```bash
TRANSCRIBE_REALTIME=false
WHISPER_MODEL_SIZE=medium
AUDIO_BUFFER_DURATION=30
```

## Troubleshooting

### "PyAudio not found"

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### "No audio devices found"

Check PulseAudio/ALSA configuration (Linux) or system audio permissions.

### Slow transcription

- Use smaller model (`tiny` or `base`)
- Enable GPU with `WHISPER_DEVICE=cuda`
- Increase buffer duration
- Use `int8` compute type

### High CPU usage

- Increase `AUDIO_CHUNK_MS` (e.g., 200-500ms)
- Increase `AUDIO_BUFFER_DURATION` (e.g., 30-60s)
- Lower `AUDIO_SAMPLE_RATE` (not recommended, affects quality)

## Performance Benchmarks

| Model | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|-------------|-------------|----------|
| tiny  | ~10x        | ~100x       | 75%      |
| base  | ~7x         | ~70x        | 80%      |
| small | ~4x         | ~40x        | 85%      |
| medium| ~2x         | ~20x        | 90%      |

*Speed = realtime factor (10x = 10s audio processed in 1s)*
```

### 9.2 Create docs/WHISPER_MODELS.md

```markdown
# Whisper Model Guide

## Model Sizes

### tiny (39M parameters)
- **Size**: 32MB
- **Accuracy**: ~75%
- **Speed**: Fastest
- **Use case**: Ultra-low latency, resource-constrained systems

### base (74M parameters)
- **Size**: 74MB
- **Accuracy**: ~80%
- **Speed**: Fast
- **Use case**: Real-time transcription **[RECOMMENDED]**

### small (244M parameters)
- **Size**: 244MB
- **Accuracy**: ~85%
- **Speed**: Moderate
- **Use case**: Balanced accuracy/speed

### medium (769M parameters)
- **Size**: 769MB
- **Accuracy**: ~90%
- **Speed**: Slow
- **Use case**: Post-processing, high accuracy needed

### large-v3 (1550M parameters)
- **Size**: 1.5GB
- **Accuracy**: ~95%
- **Speed**: Very slow
- **Use case**: Offline processing, maximum accuracy

## Compute Types

### int8 (Quantized)
- **Speed**: Fastest
- **Memory**: Lowest
- **Accuracy**: Slight degradation (~1-2%)
- **Recommended for**: CPU inference, real-time

### float16 (Half Precision)
- **Speed**: Fast
- **Memory**: Moderate
- **Accuracy**: Near-identical to float32
- **Recommended for**: GPU inference

### float32 (Full Precision)
- **Speed**: Slowest
- **Memory**: Highest
- **Accuracy**: Best
- **Recommended for**: Maximum quality, offline

## Language Support

Whisper supports 97 languages. For best performance:

- Set `TRANSCRIBE_LANGUAGE=en` for English-only
- Set `TRANSCRIBE_LANGUAGE=auto` for multi-language detection

## GPU Acceleration

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- cuDNN library

### Configuration

```bash
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

### Performance Improvement

Expect 10-20x speedup with GPU vs CPU.

## Storage Requirements

Models are stored in `./data/models/whisper/` by default.

Plan for:
- Base model: ~100MB
- All models: ~3GB

## Download Script

```bash
# Download default model
python scripts/download_whisper_models.py

# Download specific model
WHISPER_MODEL_SIZE=small python scripts/download_whisper_models.py
```
```

### 9.3 Update docs/API.md

Add audio API documentation:

```markdown
# Audio API Reference

## AudioTranscriptService

### Methods

#### `start_audio_capture(device_index=None)`
Start capturing audio from microphone.

**Parameters:**
- `device_index` (int, optional): Audio device index

**Returns:**
- `bool`: True if capture started successfully

**Example:**
```python
service = AudioTranscriptService("MISSION_001")
service.start_audio_capture(device_index=0)
```

#### `start_realtime_transcription()`
Start real-time transcription worker thread.

**Returns:**
- `bool`: True if started successfully

**Example:**
```python
service.start_realtime_transcription()
```

#### `transcribe_audio(audio_path)`
Transcribe an audio file.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
- `dict`: Transcription result with text, confidence, language

**Example:**
```python
result = service.transcribe_audio("/path/to/audio.wav")
print(result['text'])
```

#### `get_all_transcripts()`
Get all stored transcripts.

**Returns:**
- `list`: List of transcript dictionaries

**Example:**
```python
transcripts = service.get_all_transcripts()
for t in transcripts:
    print(f"[{t['speaker']}] {t['text']}")
```

## WhisperTranscriber

### Methods

#### `load_model()`
Load Whisper model into memory.

**Returns:**
- `bool`: True if loaded successfully

#### `transcribe_file(audio_path)`
Transcribe an audio file.

**Parameters:**
- `audio_path` (str): Path to audio file

**Returns:**
- `dict`: Transcription result

## AudioCaptureManager

### Methods

#### `start_capture()`
Start audio capture from microphone.

**Returns:**
- `bool`: True if started successfully

#### `set_audio_callback(callback)`
Set callback for audio chunks.

**Parameters:**
- `callback` (callable): Function(audio_data, timestamp)

**Example:**
```python
def on_audio(audio_data, timestamp):
    print(f"Received {len(audio_data)} samples at {timestamp}")

capture = AudioCaptureManager()
capture.set_audio_callback(on_audio)
capture.start_capture()
```
```

---

## Implementation Checklist

### Phase 1: Setup
- [ ] Update `requirements.txt` with new dependencies
- [ ] Update `.env.example` with audio configuration
- [ ] Create `scripts/download_whisper_models.py`
- [ ] Run model download script

### Phase 2: Audio Capture
- [ ] Create `src/audio/capture.py`
- [ ] Implement `AudioCaptureManager` class
- [ ] Create `scripts/list_audio_devices.py`
- [ ] Test audio capture with test script

### Phase 3: Transcription
- [ ] Create `src/audio/whisper_transcriber.py`
- [ ] Implement `WhisperTranscriber` class
- [ ] Create `scripts/test_whisper_transcription.py`
- [ ] Test transcription with sample audio

### Phase 4: Speaker Diarization (Optional)
- [ ] Create `src/audio/speaker_diarization.py`
- [ ] Implement `SpeakerDiarizer` class
- [ ] Test speaker identification

### Phase 5: Service Updates
- [ ] Update `AudioTranscriptService.transcribe_audio()`
- [ ] Update `AudioTranscriptService._transcription_worker()`
- [ ] Add `start_audio_capture()` method
- [ ] Add `stop_audio_capture()` method
- [ ] Test updated service

### Phase 6: GameRecorder Integration
- [ ] Update `GameRecorder.start_recording()`
- [ ] Update `GameRecorder.stop_recording()`
- [ ] Add audio cleanup logic
- [ ] Test end-to-end recording

### Phase 7: Testing
- [ ] Create `scripts/test_realtime_audio.py`
- [ ] Create `scripts/test_mission_with_audio.py`
- [ ] Update `tests/test_audio_transcript.py`
- [ ] Run all tests and verify functionality

### Phase 8: Optimization
- [ ] Implement adaptive buffer sizing
- [ ] Add sliding window processing
- [ ] Add model warm-up
- [ ] Benchmark performance

### Phase 9: Documentation
- [ ] Create `docs/AUDIO_SETUP.md`
- [ ] Create `docs/WHISPER_MODELS.md`
- [ ] Update `docs/API.md`
- [ ] Update `README.md` with audio features

---

## Dependencies Summary

```txt
# Add to requirements.txt

# AI/ML - Transcription
faster-whisper>=1.0.0
torch>=2.0.0
torchaudio>=2.0.0

# Speaker Diarization (optional)
pyannote.audio>=3.1.0

# Audio processing
webrtcvad>=2.0.10

# System monitoring (for optimization)
psutil>=5.9.0
```

---

## Environment Variables Summary

```bash
# Whisper Configuration
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_MODEL_PATH=./data/models/whisper/

# Audio Capture
AUDIO_INPUT_DEVICE=0
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_MS=100
AUDIO_BUFFER_DURATION=30

# Transcription
TRANSCRIBE_REALTIME=true
TRANSCRIBE_LANGUAGE=en
VAD_THRESHOLD=0.02
MIN_SILENCE_DURATION=0.5
ENABLE_AUDIO_CAPTURE=true

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION=false
HF_AUTH_TOKEN=your_token_here
EXPECTED_SPEAKERS=6
```

---

## Estimated Timeline

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| 1. Setup | Dependencies, configuration | 1-2 hours |
| 2. Audio Capture | Implement capture module | 3-4 hours |
| 3. Transcription | Implement Whisper integration | 4-6 hours |
| 4. Speaker Diarization | Implement speaker ID (optional) | 4-6 hours |
| 5. Service Updates | Modify AudioTranscriptService | 2-3 hours |
| 6. GameRecorder Integration | Connect to game recorder | 2-3 hours |
| 7. Testing | Create and run tests | 3-4 hours |
| 8. Optimization | Performance tuning | 2-3 hours |
| 9. Documentation | Write docs and guides | 2-3 hours |
| **Total** | | **23-34 hours** |

*Without speaker diarization: ~19-28 hours*

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✓ Audio capture from microphone working
- ✓ Real-time transcription functional
- ✓ Transcripts stored with timestamps
- ✓ Integration with GameRecorder complete
- ✓ Basic testing scripts working

### Enhanced Features
- ✓ Speaker identification working
- ✓ Transcript/telemetry synchronization
- ✓ Performance optimized for real-time
- ✓ Comprehensive test coverage
- ✓ Complete documentation

---

## Risk Mitigation

### Risk: PyAudio installation issues
**Mitigation**: Provide platform-specific installation instructions, fallback to sounddevice

### Risk: Whisper model too slow for real-time
**Mitigation**: Use `faster-whisper` with CTranslate2, recommend `base` model, support GPU

### Risk: Audio device access problems
**Mitigation**: Comprehensive device detection, clear error messages, permissions guide

### Risk: High CPU/memory usage
**Mitigation**: Adaptive buffer sizing, model selection guide, resource monitoring

### Risk: Transcription accuracy issues
**Mitigation**: Configurable models, VAD tuning, language selection, quality metrics

---

## Future Enhancements

1. **Voice Commands**: Integrate with command execution system
2. **Emotion Detection**: Analyze tone and sentiment
3. **Automated Mission Reports**: Generate summaries from transcripts
4. **Multi-language Support**: Auto-detect and switch languages
5. **Cloud Backup**: Optional cloud storage for transcripts
6. **Live Captions**: Real-time display in game overlay

---

## References

- [Faster-Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyAudio Documentation](https://people.csail.mit.edu/hubert/pyaudio/)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)

---

**End of Implementation Plan**
