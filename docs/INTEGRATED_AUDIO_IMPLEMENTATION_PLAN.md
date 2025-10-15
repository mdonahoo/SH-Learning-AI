# Integrated Audio Transcription Implementation Plan
## Starship Horizons Learning AI - Real-Time Multi-Speaker Bridge Transcription

**Project:** Starship Horizons Learning AI
**Feature:** Local Audio Recording, Real-time Transcription, and Speaker Diarization
**Date:** 2025-10-02
**Status:** Planning Phase - Integrated Design
**Architecture:** Extends existing Starship Horizons codebase with proven discussion transcriber patterns

---

## Executive Summary

This document provides a complete implementation plan for adding real-time audio transcription and speaker diarization to the **existing Starship Horizons Learning AI codebase**. The design integrates proven patterns from discussion transcriber systems while respecting the existing architecture and standards defined in `CLAUDE.md`.

### Key Goals

1. **Extend existing architecture** - Build on `AudioTranscriptService`, `GameRecorder`, and `EventRecorder`
2. **Real-time bridge crew transcription** - Capture and transcribe 6+ bridge crew members during missions
3. **Speaker identification** - Automatically detect and track individual crew members
4. **Telemetry integration** - Synchronize audio transcripts with game events for mission analysis
5. **Local processing** - Use Faster-Whisper for offline transcription (no cloud dependency)
6. **Production ready** - Proper error handling, logging, testing, and documentation per project standards

---

## Current State Analysis

### Existing Infrastructure (Already Implemented)

```
src/
├── metrics/
│   ├── event_recorder.py          ✅ Records game telemetry
│   ├── audio_transcript.py        ✅ Mock audio transcription service
│   └── mission_summarizer.py      ✅ Mission analysis
├── integration/
│   ├── game_recorder.py           ✅ Main recording orchestrator
│   └── starship_horizons_client.py ✅ Game client
└── audio/
    ├── __init__.py                 ✅ Audio module
    └── config.py                   ✅ Audio configuration

tests/
└── test_audio_transcript.py       ✅ Basic audio tests

scripts/
└── (various test scripts)          ✅ Testing utilities
```

### What Currently Exists (Mock/Stub)

**`src/metrics/audio_transcript.py`** has:
- ✅ `AudioTranscriptService` class structure
- ✅ `start_recording()` / `stop_recording()` methods
- ✅ Transcript storage and export
- ✅ Speaker identification placeholders
- ❌ **Mock transcription** (returns "Mock transcription")
- ❌ **No real audio capture**
- ❌ **No speaker diarization**
- ❌ **Empty transcription worker thread**

**`src/integration/game_recorder.py`** has:
- ✅ Initializes `AudioTranscriptService` at line 85-89
- ✅ Connects to game telemetry
- ❌ **Audio capture not started**
- ❌ **No audio/telemetry synchronization**

### What We Need to Implement

1. **Real audio capture** - PyAudio integration with VAD
2. **Speaker diarization** - Identify 6+ bridge crew members
3. **Whisper transcription** - Replace mock with Faster-Whisper
4. **Active transcription worker** - Process audio segments in real-time
5. **Engagement analytics** - Track crew participation
6. **Telemetry sync** - Align audio with game events

---

## Architecture Design

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GameRecorder                                    │
│              (src/integration/game_recorder.py)                      │
│                                                                       │
│  ┌─────────────────────┐         ┌──────────────────────┐          │
│  │   EventRecorder     │         │ AudioTranscriptService│          │
│  │  (game telemetry)   │◄───────►│   (ENHANCED)          │          │
│  └─────────────────────┘         └──────────┬───────────┘          │
│                                               │                       │
└───────────────────────────────────────────────┼───────────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────┐
                    │                            │                        │
           ┌────────▼────────┐        ┌─────────▼──────────┐  ┌─────────▼──────────┐
           │ AudioCapture    │        │  SpeakerDiarizer   │  │  WhisperTranscriber│
           │  Manager        │        │    (NEW)           │  │     (NEW)          │
           │  (NEW)          │        │                    │  │                    │
           └────────┬────────┘        └─────────┬──────────┘  └─────────┬──────────┘
                    │                           │                        │
                    │                           │                        │
           ┌────────▼────────┐        ┌─────────▼──────────┐  ┌─────────▼──────────┐
           │  PyAudio Stream │        │  Speaker Features  │  │ Faster-Whisper     │
           │  + SimpleVAD    │        │  + Bridge Roles    │  │ Model (local)      │
           └─────────────────┘        └────────────────────┘  └────────────────────┘
```

### Data Flow

```
1. Audio Capture:
   Microphone → PyAudio → AudioCaptureManager → VAD Filter → Audio Segments

2. Speaker Identification:
   Audio Segment → Feature Extraction → Speaker Matching → Bridge Role Assignment

3. Transcription:
   Audio Segment → Whisper Model → Text + Confidence → Transcript Entry

4. Storage & Sync:
   Transcript + Speaker + Timestamp → AudioTranscriptService → Database
                                    ↓
                          EventRecorder (telemetry sync)
                                    ↓
                          Mission Timeline (combined view)
```

---

## Implementation Phases

## Phase 1: Dependencies & Configuration

### 1.1 Update requirements.txt

Add new dependencies to existing file:

```txt
# ============================================
# EXISTING DEPENDENCIES (keep as-is)
# ============================================
# Core Dependencies
numpy==1.26.4
websockets==12.0
aiohttp==3.9.1
asyncio-mqtt==0.16.2

# Audio Processing
SpeechRecognition==3.10.4
pyttsx3==2.90
pyaudio==0.2.14
sounddevice==0.4.6
pygame==2.5.2

# Data Processing
pandas==2.2.0
python-dateutil==2.8.2

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
rich==13.7.0

# Game Integration
playwright==1.40.0

# ============================================
# NEW DEPENDENCIES FOR AUDIO TRANSCRIPTION
# ============================================

# Voice Activity Detection
webrtcvad>=2.0.10

# AI/ML - Local Transcription
faster-whisper>=1.0.0
torch>=2.0.0
torchaudio>=2.0.0

# Performance Monitoring
psutil>=5.9.0

# OPTIONAL: Advanced Speaker Diarization
# Uncomment if using neural speaker embeddings
# pyannote.audio>=3.1.0
# resemblyzer>=0.1.1
```

### 1.2 Update .env.example

Add audio configuration to existing file:

```bash
# ==========================================
# EXISTING CONFIGURATION (keep as-is)
# ==========================================
GAME_HOST=192.168.68.55
GAME_PORT_API=1864
GAME_PORT_WS=1865
GAME_PORT_HTTPS=1866
GAME_PORT_WSS=1867

RECORDING_PATH=./data/recordings
LOG_LEVEL=INFO
SESSION_STORAGE_PATH=./data/sessions/
METRICS_DB_PATH=./data/metrics.db

DEBUG=False
TEST_MODE=False

# ==========================================
# AUDIO TRANSCRIPTION CONFIGURATION (NEW)
# ==========================================

# Enable audio capture during missions
ENABLE_AUDIO_CAPTURE=true

# Audio Device Configuration
# Use scripts/list_audio_devices.py to find your device index
AUDIO_INPUT_DEVICE=0
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_MS=100

# Whisper Model Configuration
# Model sizes: tiny, base, small, medium, large-v3
# Recommended: base (fast, good accuracy for bridge audio)
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_MODEL_PATH=./data/models/whisper/

# Voice Activity Detection (VAD)
# Adjust based on room noise and microphone setup
VAD_ENERGY_THRESHOLD=500
MIN_SPEECH_DURATION=0.3
MIN_SILENCE_DURATION=0.5

# Transcription Settings
TRANSCRIBE_REALTIME=true
TRANSCRIBE_LANGUAGE=en
MIN_TRANSCRIPTION_CONFIDENCE=0.5

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION=true
SPEAKER_SIMILARITY_THRESHOLD=0.7
EXPECTED_BRIDGE_CREW=6
BRIDGE_ROLES=Captain,Helm,Tactical,Science,Engineering,Communications

# Engagement Analytics
ENABLE_ENGAGEMENT_METRICS=true
ENGAGEMENT_UPDATE_INTERVAL=30.0

# Data Retention
SAVE_RAW_AUDIO=false
AUDIO_RETENTION_DAYS=0
TRANSCRIPT_RETENTION_DAYS=30

# Performance Tuning
MAX_SEGMENT_QUEUE_SIZE=100
TRANSCRIPTION_WORKERS=2
```

### 1.3 Create Model Download Script

**File:** `scripts/download_whisper_models.py`

```python
#!/usr/bin/env python3
"""
Download and cache Whisper models for offline bridge audio transcription.

Usage:
    python scripts/download_whisper_models.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_size: str, device: str = "cpu", compute_type: str = "int8") -> bool:
    """
    Download and cache a Whisper model.

    Args:
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (cpu or cuda)
        compute_type: Compute precision (int8, float16, float32)

    Returns:
        True if successful
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        logger.error("Or: pip install -r requirements.txt")
        return False

    logger.info(f"\nDownloading Whisper model: {model_size}")
    logger.info(f"Device: {device}, Compute type: {compute_type}")

    model_path = Path(os.getenv("WHISPER_MODEL_PATH", "./data/models/whisper/"))
    model_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Initializing model (this will download if not cached)...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(model_path)
        )

        # Test model with dummy audio
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        segments, info = model.transcribe(dummy_audio)
        list(segments)  # Force transcription to complete

        logger.info(f"✓ Model {model_size} downloaded and verified")
        logger.info(f"✓ Model stored in: {model_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        return False


def main():
    """Main entry point."""
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    logger.info("="*60)
    logger.info("Starship Horizons - Whisper Model Downloader")
    logger.info("="*60)
    logger.info("\nThis will download the Whisper model for offline use.")
    logger.info("The model will be cached and reused for future sessions.")
    logger.info("")

    success = download_model(model_size, device, compute_type)

    if success:
        logger.info("\n" + "="*60)
        logger.info("✓ Ready for bridge audio transcription!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("  1. Test audio: python scripts/test_realtime_audio.py")
        logger.info("  2. Start recording: python scripts/record_mission_with_audio.py")
    else:
        logger.error("\n✗ Model download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 1.4 Create Audio Device Lister

**File:** `scripts/list_audio_devices.py`

```python
#!/usr/bin/env python3
"""
List all available audio input devices.

Usage:
    python scripts/list_audio_devices.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyaudio
except ImportError:
    print("ERROR: PyAudio not installed")
    print("Install with: pip install pyaudio")
    sys.exit(1)


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


if __name__ == "__main__":
    list_audio_devices()
```

---

## Phase 2: Speaker Diarization Module (NEW)

**File:** `src/audio/speaker_diarization.py`

This is a **NEW** module that provides speaker detection and tracking.

```python
"""
Speaker Diarization for Starship Horizons Bridge Crew.

This module handles:
- Voice Activity Detection (VAD) for speech segmentation
- Speaker feature extraction and identification
- Bridge crew role mapping
- Engagement and participation analytics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """
    Represents a speaker's utterance segment.

    Attributes:
        speaker_id: Unique identifier for speaker (e.g., "speaker_1")
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        audio_data: Raw audio samples (float32, normalized)
        confidence: Speaker identification confidence (0-1)
        text: Transcribed text (optional, added later)
        bridge_role: Bridge crew role (optional, e.g., "Captain")
    """
    speaker_id: str
    start_time: float
    end_time: float
    audio_data: np.ndarray
    confidence: float
    text: Optional[str] = None
    bridge_role: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


class SimpleVAD:
    """
    Voice Activity Detection using energy thresholds.

    Uses RMS energy to detect speech vs silence, with configurable
    thresholds for different acoustic environments.

    Designed for real-time operation with low latency (<5ms per chunk).
    """

    def __init__(
        self,
        energy_threshold: Optional[float] = None,
        min_speech_duration: Optional[float] = None,
        min_silence_duration: Optional[float] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize VAD.

        Args:
            energy_threshold: RMS energy threshold for speech detection
            min_speech_duration: Minimum speech duration to process (seconds)
            min_silence_duration: Minimum silence to end utterance (seconds)
            sample_rate: Audio sample rate in Hz
        """
        self.energy_threshold = energy_threshold or float(
            os.getenv('VAD_ENERGY_THRESHOLD', '500')
        )
        self.sample_rate = sample_rate

        min_speech = min_speech_duration or float(
            os.getenv('MIN_SPEECH_DURATION', '0.3')
        )
        min_silence = min_silence_duration or float(
            os.getenv('MIN_SILENCE_DURATION', '0.5')
        )

        self.min_speech_samples = int(min_speech * sample_rate)
        self.min_silence_samples = int(min_silence * sample_rate)

        # State tracking
        self.is_speaking = False
        self.speech_start = None
        self.silence_counter = 0
        self.speech_buffer = []

        logger.debug(
            f"VAD initialized: threshold={self.energy_threshold}, "
            f"min_speech={min_speech}s, min_silence={min_silence}s"
        )

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        timestamp: float
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Process audio chunk and detect speech segments.

        This is called for every audio chunk (typically 100ms).
        Returns complete speech segments when silence is detected.

        Args:
            audio_chunk: Audio samples (float32 normalized or int16)
            timestamp: Current timestamp in seconds

        Returns:
            Tuple of (audio_segment, start_time, end_time) when utterance
            is complete, else None
        """
        # Normalize to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))

        if energy > self.energy_threshold:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start = timestamp
                self.speech_buffer = []
                logger.debug(f"Speech started at {timestamp:.2f}s (energy: {energy:.4f})")

            self.speech_buffer.append(audio_chunk)
            self.silence_counter = 0

        elif self.is_speaking:
            # Potential silence during speech
            self.silence_counter += len(audio_chunk)
            self.speech_buffer.append(audio_chunk)

            # Check if silence is long enough to end utterance
            if self.silence_counter >= self.min_silence_samples:
                # Check if speech was long enough
                total_samples = sum(len(chunk) for chunk in self.speech_buffer)

                if total_samples >= self.min_speech_samples:
                    # Complete utterance
                    audio_segment = np.concatenate(self.speech_buffer)
                    start = self.speech_start
                    end = timestamp
                    duration = end - start

                    logger.debug(
                        f"Speech segment complete: {start:.2f}s - {end:.2f}s "
                        f"(duration: {duration:.2f}s)"
                    )

                    # Reset state
                    self.is_speaking = False
                    self.speech_start = None
                    self.speech_buffer = []
                    self.silence_counter = 0

                    return (audio_segment, start, end)
                else:
                    # Too short, discard
                    logger.debug("Speech segment too short, discarding")
                    self.is_speaking = False
                    self.speech_buffer = []
                    self.silence_counter = 0

        return None

    def reset(self):
        """Reset VAD state (e.g., between sessions)."""
        self.is_speaking = False
        self.speech_start = None
        self.speech_buffer = []
        self.silence_counter = 0


class SpeakerDiarizer:
    """
    Identify and track multiple speakers using audio features.

    Uses simple spectral features for speaker comparison. For production
    use with better accuracy, consider:
    - resemblyzer (speaker embeddings)
    - pyannote.audio (neural diarization)

    This implementation prioritizes:
    - Low latency (<50ms per segment)
    - No GPU requirement
    - Reasonable accuracy for 6-8 speakers
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        bridge_roles: Optional[List[str]] = None
    ):
        """
        Initialize speaker diarizer.

        Args:
            similarity_threshold: Minimum similarity (0-1) to match speakers
            bridge_roles: List of bridge crew roles
        """
        self.similarity_threshold = similarity_threshold or float(
            os.getenv('SPEAKER_SIMILARITY_THRESHOLD', '0.7')
        )

        # Bridge crew roles from environment or defaults
        roles_str = os.getenv(
            'BRIDGE_ROLES',
            'Captain,Helm,Tactical,Science,Engineering,Communications'
        )
        self.bridge_roles = bridge_roles or roles_str.split(',')

        # Speaker profiles: speaker_id -> feature_vector
        self.speaker_profiles: Dict[str, np.ndarray] = {}

        # Speaker metadata
        self.speaker_names: Dict[str, str] = {}
        self.speaker_roles: Dict[str, str] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Speaker diarizer initialized: "
            f"threshold={self.similarity_threshold}, "
            f"expected_crew={len(self.bridge_roles)}"
        )

    def extract_speaker_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract speaker-identifying features from audio.

        Features extracted:
        - Zero-crossing rate (pitch estimation)
        - Energy statistics (mean, std, max, percentiles)
        - Spectral features (centroid, rolloff, bandwidth)
        - Simplified MFCCs (13 mel-scale bins)

        Args:
            audio_segment: Audio samples (float32, normalized)

        Returns:
            Feature vector (numpy array, ~22 dimensions)
        """
        # Ensure audio is 1D
        if len(audio_segment.shape) > 1:
            audio_segment = audio_segment.flatten()

        # Ensure minimum length for FFT
        if len(audio_segment) < 512:
            audio_segment = np.pad(audio_segment, (0, 512 - len(audio_segment)))

        features = []

        # 1. Pitch estimation (zero-crossing rate)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_segment))))
        zcr = zero_crossings / (2 * len(audio_segment))
        features.append(zcr)

        # 2. Energy statistics
        energy = np.abs(audio_segment)
        features.extend([
            np.mean(energy),
            np.std(energy),
            np.max(energy),
            np.percentile(energy, 95),
            np.percentile(energy, 5)
        ])

        # 3. Spectral features
        try:
            # Compute FFT
            fft = np.abs(np.fft.rfft(audio_segment))
            fft_norm = fft / (np.sum(fft) + 1e-8)
            freqs = np.fft.rfftfreq(len(audio_segment), 1/16000)

            # Spectral centroid (brightness)
            spectral_centroid = np.sum(freqs * fft_norm)
            features.append(spectral_centroid)

            # Spectral rolloff (95% energy point)
            cumsum = np.cumsum(fft_norm)
            rolloff_idx = np.where(cumsum >= 0.95)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            features.append(spectral_rolloff)

            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(
                np.sum(((freqs - spectral_centroid) ** 2) * fft_norm)
            )
            features.append(spectral_bandwidth)

            # 4. MFCC-like features (simplified mel-scale bins)
            n_mels = 13
            mel_filters = np.linspace(0, len(fft), n_mels + 1, dtype=int)
            mfcc_simple = []
            for i in range(n_mels):
                start, end = mel_filters[i], mel_filters[i+1]
                if end > start:
                    mel_energy = np.mean(fft[start:end])
                else:
                    mel_energy = 0
                mfcc_simple.append(np.log(mel_energy + 1e-8))
            features.extend(mfcc_simple)

        except Exception as e:
            self.logger.warning(f"Error extracting spectral features: {e}")
            # Pad with zeros if FFT fails
            features.extend([0.0] * 16)

        return np.array(features, dtype=np.float32)

    def cosine_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between feature vectors.

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Similarity score (0-1, where 1 = identical)
        """
        dot_product = np.dot(features1, features2)
        norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def identify_speaker(
        self,
        audio_segment: np.ndarray
    ) -> Tuple[str, float]:
        """
        Identify speaker from audio segment.

        Compares audio features against known speaker profiles.
        If no match above threshold, registers as new speaker.

        Args:
            audio_segment: Audio samples (float32, normalized)

        Returns:
            Tuple of (speaker_id, confidence)
        """
        features = self.extract_speaker_features(audio_segment)

        if not self.speaker_profiles:
            # First speaker
            speaker_id = "speaker_1"
            self.speaker_profiles[speaker_id] = features
            self.logger.info(f"Registered new speaker: {speaker_id}")
            return (speaker_id, 1.0)

        # Compare with known speakers
        best_match = None
        best_similarity = 0.0

        for speaker_id, profile in self.speaker_profiles.items():
            similarity = self.cosine_similarity(features, profile)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        # Check if similarity is high enough
        if best_similarity >= self.similarity_threshold:
            # Update speaker profile (exponential moving average)
            alpha = 0.1  # Learning rate
            self.speaker_profiles[best_match] = (
                (1 - alpha) * self.speaker_profiles[best_match] +
                alpha * features
            )
            return (best_match, best_similarity)
        else:
            # New speaker
            new_id = len(self.speaker_profiles) + 1
            speaker_id = f"speaker_{new_id}"
            self.speaker_profiles[speaker_id] = features
            self.logger.info(f"Registered new speaker: {speaker_id}")
            return (speaker_id, 1.0)

    def assign_speaker_names(self, speaker_mapping: Dict[str, str]):
        """
        Assign human-readable names to speaker IDs.

        Args:
            speaker_mapping: Dict like {"speaker_1": "Alice", "speaker_2": "Bob"}
        """
        for speaker_id, name in speaker_mapping.items():
            if speaker_id in self.speaker_profiles:
                self.speaker_names[speaker_id] = name
                self.logger.info(f"Assigned name: {speaker_id} → {name}")

    def assign_bridge_roles(self, role_mapping: Dict[str, str]):
        """
        Assign bridge crew roles to speakers.

        Args:
            role_mapping: Dict like {"speaker_1": "Captain", "speaker_2": "Helm"}
        """
        for speaker_id, role in role_mapping.items():
            if role in self.bridge_roles:
                self.speaker_roles[speaker_id] = role
                self.logger.info(f"Assigned role: {speaker_id} → {role}")
            else:
                self.logger.warning(f"Unknown bridge role: {role}")

    def get_speaker_display_name(self, speaker_id: str) -> str:
        """
        Get display name for speaker (priority: role > name > ID).

        Args:
            speaker_id: Speaker identifier

        Returns:
            Display name string
        """
        if speaker_id in self.speaker_roles:
            return self.speaker_roles[speaker_id]
        elif speaker_id in self.speaker_names:
            return self.speaker_names[speaker_id]
        else:
            return speaker_id

    def export_speaker_profiles(self) -> Dict[str, Dict]:
        """
        Export speaker profiles for persistence.

        Returns:
            Dictionary of speaker profiles with metadata
        """
        return {
            speaker_id: {
                'features': profile.tolist(),
                'name': self.speaker_names.get(speaker_id),
                'role': self.speaker_roles.get(speaker_id)
            }
            for speaker_id, profile in self.speaker_profiles.items()
        }

    def import_speaker_profiles(self, profiles: Dict[str, Dict]):
        """
        Import speaker profiles from persistence.

        Args:
            profiles: Dictionary of speaker profiles
        """
        for speaker_id, data in profiles.items():
            self.speaker_profiles[speaker_id] = np.array(
                data['features'],
                dtype=np.float32
            )
            if data.get('name'):
                self.speaker_names[speaker_id] = data['name']
            if data.get('role'):
                self.speaker_roles[speaker_id] = data['role']

        self.logger.info(f"Imported {len(profiles)} speaker profiles")


class EngagementAnalyzer:
    """
    Analyze bridge crew engagement and communication patterns.

    Tracks:
    - Speaking time per crew member
    - Turn-taking dynamics
    - Interruption patterns
    - Communication effectiveness

    Used for mission debriefs and crew performance analysis.
    """

    def __init__(self):
        """Initialize engagement analyzer."""
        self.speaker_stats = defaultdict(lambda: {
            'total_time': 0.0,
            'utterance_count': 0,
            'last_spoke': None,
            'avg_utterance_duration': 0.0,
            'longest_utterance': 0.0,
            'shortest_utterance': float('inf')
        })

        self.turn_taking_history = []
        self.interruption_count = defaultdict(int)

        self.logger = logging.getLogger(__name__)

    def update_speaker_stats(self, speaker_id: str, segment: SpeakerSegment):
        """
        Update statistics for a speaker.

        Args:
            speaker_id: Speaker identifier
            segment: Speaker segment data
        """
        stats = self.speaker_stats[speaker_id]

        stats['total_time'] += segment.duration
        stats['utterance_count'] += 1
        stats['last_spoke'] = segment.end_time
        stats['avg_utterance_duration'] = (
            stats['total_time'] / stats['utterance_count']
        )
        stats['longest_utterance'] = max(
            stats['longest_utterance'], segment.duration
        )
        stats['shortest_utterance'] = min(
            stats['shortest_utterance'], segment.duration
        )

        # Track turn-taking
        self.turn_taking_history.append({
            'speaker': speaker_id,
            'timestamp': segment.start_time,
            'duration': segment.duration
        })

        # Detect interruptions (speaker change within 0.5s)
        if len(self.turn_taking_history) > 1:
            prev = self.turn_taking_history[-2]
            time_gap = segment.start_time - (prev['timestamp'] + prev['duration'])

            if time_gap < 0.5 and prev['speaker'] != speaker_id:
                self.interruption_count[speaker_id] += 1
                self.logger.debug(f"Interruption detected: {speaker_id}")

    def calculate_engagement_scores(self) -> Dict[str, Dict]:
        """
        Calculate engagement metrics for all speakers.

        Returns:
            Dictionary of engagement scores (0-100) per speaker
        """
        if not self.speaker_stats:
            return {}

        total_speaking_time = sum(
            s['total_time'] for s in self.speaker_stats.values()
        )
        num_speakers = len(self.speaker_stats)

        scores = {}

        for speaker_id, stats in self.speaker_stats.items():
            # Participation score (relative to equal share)
            ideal_share = 1.0 / num_speakers if num_speakers > 0 else 1.0
            actual_share = stats['total_time'] / (total_speaking_time + 1e-8)
            participation_score = min(100, (actual_share / ideal_share) * 100)

            # Turn-taking consistency
            speaker_turns = [
                t for t in self.turn_taking_history
                if t['speaker'] == speaker_id
            ]

            if len(speaker_turns) > 1:
                turn_intervals = [
                    speaker_turns[i+1]['timestamp'] - speaker_turns[i]['timestamp']
                    for i in range(len(speaker_turns) - 1)
                ]
                consistency = 100 - min(100, np.std(turn_intervals) * 10)
            else:
                consistency = 50

            # Interruption penalty
            interruption_penalty = min(
                20, self.interruption_count[speaker_id] * 5
            )

            # Overall engagement score
            engagement_score = (
                participation_score * 0.5 +
                consistency * 0.4 -
                interruption_penalty * 0.1
            )
            engagement_score = max(0, min(100, engagement_score))

            scores[speaker_id] = {
                'engagement_score': round(engagement_score, 1),
                'participation_score': round(participation_score, 1),
                'consistency_score': round(consistency, 1),
                'speaking_time_seconds': round(stats['total_time'], 1),
                'utterance_count': stats['utterance_count'],
                'avg_utterance_duration': round(stats['avg_utterance_duration'], 2),
                'longest_utterance': round(stats['longest_utterance'], 2),
                'interruptions': self.interruption_count[speaker_id]
            }

        return scores

    def get_mission_communication_summary(self) -> Dict:
        """
        Generate mission communication summary.

        Returns:
            Dictionary with comprehensive communication metrics
        """
        engagement_scores = self.calculate_engagement_scores()

        return {
            'total_speakers': len(self.speaker_stats),
            'total_utterances': len(self.turn_taking_history),
            'total_interruptions': sum(self.interruption_count.values()),
            'speaker_scores': engagement_scores,
            'turn_taking_balance': self._calculate_turn_balance(),
            'communication_effectiveness': self._calculate_effectiveness()
        }

    def _calculate_turn_balance(self) -> float:
        """Calculate how evenly balanced turn-taking is (0-100)."""
        if not self.speaker_stats:
            return 0.0

        speaking_times = [s['total_time'] for s in self.speaker_stats.values()]

        # Calculate coefficient of variation
        mean_time = np.mean(speaking_times)
        std_time = np.std(speaking_times)

        if mean_time == 0:
            return 0.0

        cv = std_time / mean_time

        # Convert to 0-100 scale (lower CV = better balance)
        balance_score = max(0, 100 - (cv * 50))

        return round(balance_score, 1)

    def _calculate_effectiveness(self) -> float:
        """Calculate overall communication effectiveness (0-100)."""
        if not self.turn_taking_history:
            return 0.0

        avg_durations = [
            s['avg_utterance_duration']
            for s in self.speaker_stats.values()
        ]

        # Ideal utterance: 2-8 seconds
        duration_scores = []
        for d in avg_durations:
            if 2 <= d <= 8:
                duration_scores.append(100)
            else:
                duration_scores.append(max(0, 100 - abs(5 - d) * 10))

        duration_score = np.mean(duration_scores)

        # Interruption score (fewer = better)
        total_interruptions = sum(self.interruption_count.values())
        interruption_rate = total_interruptions / len(self.turn_taking_history)
        interruption_score = max(0, 100 - (interruption_rate * 200))

        # Combine scores
        effectiveness = (duration_score * 0.6 + interruption_score * 0.4)

        return round(effectiveness, 1)
```

---

Due to length constraints, I need to continue in a second file. Shall I:

1. **Create Part 2** with the remaining phases (Audio Capture, Whisper Integration, AudioTranscriptService updates, GameRecorder integration)?
2. **Create implementation scripts** ready to run?
3. **Focus on a specific phase** you want detailed first?