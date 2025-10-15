# Integrated Audio Transcription Implementation Plan - Part 3
## Starship Horizons Learning AI - Testing & Documentation (Phases 7-8)

**Continuation of:** INTEGRATED_AUDIO_IMPLEMENTATION_PLAN_PART2.md
**Date:** 2025-10-02

---

## Phase 7: Testing & Validation

### 7.1 Create Test Script: `scripts/test_realtime_audio.py`

**File:** `scripts/test_realtime_audio.py`

```python
#!/usr/bin/env python3
"""
Test real-time audio capture and transcription.

Usage:
    python scripts/test_realtime_audio.py [duration]
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.audio.capture import AudioCaptureManager
from src.audio.whisper_transcriber import WhisperTranscriber
from src.audio.speaker_diarization import SpeakerDiarizer, EngagementAnalyzer

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
    print("\n" + "="*70)
    print("🎙️  REAL-TIME AUDIO TRANSCRIPTION TEST")
    print("="*70)
    print(f"Duration: {duration} seconds")
    print("Speak into your microphone...")
    print("Press Ctrl+C to stop early")
    print("="*70 + "\n")

    capture_manager = None
    transcriber = None
    diarizer = None
    analyzer = None

    try:
        # Initialize components
        logger.info("Initializing components...")

        # Speaker diarizer
        diarizer = SpeakerDiarizer()
        analyzer = EngagementAnalyzer()

        # Whisper transcriber
        transcriber = WhisperTranscriber()
        transcriber.load_model()
        transcriber.start_workers()
        logger.info("✓ Transcriber ready")

        # Audio capture
        capture_manager = AudioCaptureManager(enable_vad=True)

        # Set up callback
        from src.audio.speaker_diarization import SpeakerSegment

        def audio_callback(audio_data, start_time, end_time):
            """Handle audio segments."""
            # Identify speaker
            speaker_id, confidence = diarizer.identify_speaker(audio_data)

            # Create segment
            segment = SpeakerSegment(
                speaker_id=speaker_id,
                start_time=start_time,
                end_time=end_time,
                audio_data=audio_data,
                confidence=confidence
            )

            # Update stats
            analyzer.update_speaker_stats(speaker_id, segment)

            # Queue for transcription
            transcriber.queue_audio(
                audio_data,
                start_time,
                speaker_id=speaker_id
            )

        capture_manager.set_segment_callback(audio_callback)
        capture_manager.start_capture()
        logger.info("✓ Audio capture started")
        print("")

        # Run for specified duration
        start_time = time.time()
        last_check = start_time

        while time.time() - start_time < duration:
            time.sleep(1)

            # Check for results every 2 seconds
            if time.time() - last_check >= 2:
                results = transcriber.get_results()

                for result in results:
                    timestamp = datetime.fromtimestamp(
                        start_time + result['timestamp']
                    ).strftime('%H:%M:%S')

                    speaker_id = result.get('speaker_id', 'Unknown')
                    speaker_name = diarizer.get_speaker_display_name(speaker_id)
                    text = result['text']
                    confidence = result['confidence']

                    print(f"[{timestamp}] {speaker_name}: {text} (conf: {confidence:.2f})")

                last_check = time.time()

        # Get any remaining results
        print("\nProcessing remaining transcriptions...")
        time.sleep(2)

        final_results = transcriber.get_results()
        for result in final_results:
            timestamp = datetime.fromtimestamp(
                start_time + result['timestamp']
            ).strftime('%H:%M:%S')

            speaker_id = result.get('speaker_id', 'Unknown')
            speaker_name = diarizer.get_speaker_display_name(speaker_id)
            text = result['text']
            confidence = result['confidence']

            print(f"[{timestamp}] {speaker_name}: {text} (conf: {confidence:.2f})")

        # Display summary
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)

        engagement = analyzer.calculate_engagement_scores()
        for speaker_id, stats in engagement.items():
            speaker_name = diarizer.get_speaker_display_name(speaker_id)
            print(f"\n{speaker_name}:")
            print(f"  Speaking time: {stats['speaking_time_seconds']:.1f}s")
            print(f"  Utterances: {stats['utterance_count']}")
            print(f"  Engagement: {stats['engagement_score']}/100")

        print("\n" + "="*70)
        print("✓ Test completed successfully")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        # Cleanup
        if capture_manager:
            capture_manager.stop_capture()
        if transcriber:
            transcriber.stop_workers()

        logger.info("Resources cleaned up")


def main():
    """Main entry point."""
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    try:
        test_realtime_transcription(duration)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 7.2 Create Test Script: `scripts/record_mission_with_audio.py`

**File:** `scripts/record_mission_with_audio.py`

```python
#!/usr/bin/env python3
"""
Record a Starship Horizons mission with full audio transcription.

Usage:
    python scripts/record_mission_with_audio.py [duration] [mission_name]
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.integration.game_recorder import GameRecorder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def record_mission_with_audio(duration: int = 300, mission_name: str = "Test Mission"):
    """
    Record a mission with full audio transcription.

    Args:
        duration: Recording duration in seconds
        mission_name: Name of the mission
    """
    print("\n" + "="*70)
    print("🚀 STARSHIP HORIZONS MISSION RECORDER")
    print("="*70)
    print(f"Mission: {mission_name}")
    print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
    print("\nThis will record:")
    print("  • Game telemetry (events, state changes)")
    print("  • Bridge crew audio (with speaker identification)")
    print("  • Real-time transcription")
    print("  • Engagement metrics")
    print("\nPress Ctrl+C to stop recording early")
    print("="*70 + "\n")

    recorder = None

    try:
        # Create recorder
        logger.info("Initializing mission recorder...")
        recorder = GameRecorder()

        # Start recording
        logger.info(f"Starting mission recording: {mission_name}")
        mission_id = recorder.start_recording(mission_name)

        print(f"\n✓ Recording started: {mission_id}")
        print("\nBridge crew: Speak naturally into your microphones")
        print("The system will transcribe and identify speakers automatically\n")

        # Record for specified duration
        start_time = time.time()
        for i in range(duration):
            time.sleep(1)

            # Progress indicator every 30 seconds
            elapsed = i + 1
            if elapsed % 30 == 0:
                remaining = duration - elapsed
                print(f"Recording... {elapsed}s elapsed, {remaining}s remaining")

            # Show recent transcripts every 10 seconds
            if elapsed % 10 == 0 and recorder.audio_service:
                recent = recorder.audio_service.get_all_transcripts()[-3:]
                if recent:
                    print("\nRecent transcripts:")
                    for t in recent:
                        speaker = t.get('speaker', 'Unknown')
                        text = t.get('text', '')[:60]
                        print(f"  [{speaker}] {text}...")

        # Stop recording
        print("\n\nStopping recording...")
        summary = recorder.stop_recording()

        # Display comprehensive summary
        print("\n" + "="*70)
        print("📊 MISSION RECORDING SUMMARY")
        print("="*70)

        print(f"\nMission ID: {mission_id}")
        print(f"Mission Name: {mission_name}")
        print(f"Duration: {summary.get('duration', 0):.1f}s")

        print("\n🎮 Game Telemetry:")
        print(f"  Total Events: {summary.get('total_events', 0)}")
        print(f"  Event Types: {len(summary.get('event_types', {}))}")

        # Audio summary
        if recorder.audio_service:
            transcripts = recorder.audio_service.get_all_transcripts()
            print(f"\n🎙️  Audio Recording:")
            print(f"  Total Transcripts: {len(transcripts)}")

            engagement = recorder.audio_service.get_engagement_summary()
            if engagement:
                print(f"  Speakers Detected: {engagement.get('total_speakers', 0)}")
                print(f"  Total Utterances: {engagement.get('total_utterances', 0)}")
                print(f"  Turn Balance: {engagement.get('turn_taking_balance', 0):.1f}/100")
                print(f"  Communication Effectiveness: {engagement.get('communication_effectiveness', 0):.1f}/100")

                # Show speaker breakdown
                speaker_scores = engagement.get('speaker_scores', {})
                if speaker_scores:
                    print("\n  Speaker Breakdown:")
                    for speaker_id, stats in speaker_scores.items():
                        print(f"    {speaker_id}:")
                        print(f"      Speaking time: {stats['speaking_time_seconds']:.1f}s")
                        print(f"      Utterances: {stats['utterance_count']}")
                        print(f"      Engagement: {stats['engagement_score']:.1f}/100")

            # Show sample transcripts
            if transcripts:
                print("\n  Sample Transcripts:")
                for i, t in enumerate(transcripts[:5], 1):
                    speaker = t.get('speaker', 'Unknown')
                    text = t.get('text', '')[:60]
                    confidence = t.get('confidence', 0)
                    print(f"    {i}. [{speaker}] {text}... (conf: {confidence:.2f})")

        # Export info
        recording_path = Path("./data/recordings")
        print(f"\n💾 Files Saved:")
        print(f"  Transcripts: {recording_path}/{mission_id}_transcript.json")
        print(f"  Events: {recording_path}/{mission_id}_events.json")

        # Generate combined timeline
        if recorder.audio_service:
            timeline = recorder.get_combined_timeline()
            print(f"  Combined Timeline: {len(timeline)} entries")

        print("\n" + "="*70)
        print("✓ Mission recording completed successfully")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user")
        if recorder:
            recorder.stop_recording()
    except Exception as e:
        logger.error(f"Recording failed: {e}", exc_info=True)
    finally:
        if recorder:
            # Ensure cleanup
            try:
                if recorder.is_recording:
                    recorder.stop_recording()
            except:
                pass


def main():
    """Main entry point."""
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mission_name = sys.argv[2] if len(sys.argv) > 2 else "Test Mission"

    try:
        record_mission_with_audio(duration, mission_name)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 7.3 Update Existing Test: `tests/test_audio_transcript.py`

Add real transcription tests to the existing file:

```python
# Add these test methods to the existing TestAudioTranscriptService class

def test_real_transcription_integration(self):
    """Test real Whisper transcription if available."""
    try:
        from src.audio.whisper_transcriber import WhisperTranscriber

        # Create service with transcription enabled
        service = AudioTranscriptService(
            "TEST_MISSION_001",
            auto_transcribe=True
        )

        # Check if components initialized
        assert service._whisper_transcriber is not None
        assert service._speaker_diarizer is not None

    except ImportError:
        pytest.skip("Whisper transcriber not available")


def test_speaker_diarization_integration(self):
    """Test speaker diarization integration."""
    try:
        from src.audio.speaker_diarization import SpeakerDiarizer, SpeakerSegment
        import numpy as np

        # Create service
        service = AudioTranscriptService("TEST_MISSION_002")

        # Create mock audio segment
        audio_data = np.random.randn(16000).astype(np.float32)  # 1 second
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=1.0,
            audio_data=audio_data,
            confidence=0.95
        )

        # Test engagement analyzer
        if service._engagement_analyzer:
            service._engagement_analyzer.update_speaker_stats("speaker_1", segment)
            scores = service._engagement_analyzer.calculate_engagement_scores()
            assert "speaker_1" in scores

    except ImportError:
        pytest.skip("Speaker diarization not available")


def test_audio_capture_initialization(self):
    """Test audio capture manager initialization."""
    try:
        from src.audio.capture import AudioCaptureManager

        service = AudioTranscriptService("TEST_MISSION_003")

        # Test device info retrieval
        if service._capture_manager:
            device_info = service._capture_manager.get_device_info()
            assert isinstance(device_info, dict)

    except ImportError:
        pytest.skip("Audio capture not available")
```

### 7.4 Create Unit Tests: `tests/test_speaker_diarization.py`

**File:** `tests/test_speaker_diarization.py`

```python
#!/usr/bin/env python3
"""
Unit tests for speaker diarization module.
"""

import unittest
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.speaker_diarization import (
    SimpleVAD,
    SpeakerDiarizer,
    SpeakerSegment,
    EngagementAnalyzer
)


class TestSimpleVAD(unittest.TestCase):
    """Test SimpleVAD class."""

    def setUp(self):
        """Set up test fixtures."""
        self.vad = SimpleVAD(energy_threshold=0.01, sample_rate=16000)

    def test_silence_detection(self):
        """Test that silence is properly ignored."""
        silence = np.zeros(1024, dtype=np.float32)
        result = self.vad.process_chunk(silence, 0.0)
        self.assertIsNone(result)

    def test_speech_detection(self):
        """Test that speech is detected."""
        # Generate synthetic speech (white noise)
        np.random.seed(42)
        speech = np.random.randn(48000).astype(np.float32) * 0.1

        # Feed in chunks
        chunk_size = 1024
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i:i+chunk_size]
            timestamp = i / 16000
            result = self.vad.process_chunk(chunk, timestamp)

            if result is not None:
                audio, start, end = result
                self.assertGreater(end - start, 0.3)  # Min duration
                self.assertIsInstance(audio, np.ndarray)
                break

    def test_vad_reset(self):
        """Test VAD state reset."""
        speech = np.random.randn(1024).astype(np.float32) * 0.1
        self.vad.process_chunk(speech, 0.0)

        self.vad.reset()

        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.speech_buffer), 0)


class TestSpeakerDiarizer(unittest.TestCase):
    """Test SpeakerDiarizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.diarizer = SpeakerDiarizer()

    def test_feature_extraction(self):
        """Test audio feature extraction."""
        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32) * 0.1

        features = self.diarizer.extract_speaker_features(audio)

        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertEqual(features.dtype, np.float32)

    def test_speaker_identification(self):
        """Test speaker identification."""
        np.random.seed(42)
        audio1 = np.random.randn(16000).astype(np.float32) * 0.1
        audio2 = np.random.randn(16000).astype(np.float32) * 0.1

        speaker1, conf1 = self.diarizer.identify_speaker(audio1)
        speaker2, conf2 = self.diarizer.identify_speaker(audio2)

        self.assertEqual(speaker1, "speaker_1")
        self.assertIsInstance(conf1, float)
        self.assertGreater(conf1, 0.0)
        self.assertLessEqual(conf1, 1.0)

    def test_speaker_name_assignment(self):
        """Test speaker name assignment."""
        # Register a speaker first
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        speaker_id, _ = self.diarizer.identify_speaker(audio)

        # Assign name
        self.diarizer.assign_speaker_names({speaker_id: "Alice"})

        display_name = self.diarizer.get_speaker_display_name(speaker_id)
        self.assertEqual(display_name, "Alice")

    def test_bridge_role_assignment(self):
        """Test bridge crew role assignment."""
        # Register a speaker
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        speaker_id, _ = self.diarizer.identify_speaker(audio)

        # Assign role
        self.diarizer.assign_bridge_roles({speaker_id: "Captain"})

        display_name = self.diarizer.get_speaker_display_name(speaker_id)
        self.assertEqual(display_name, "Captain")


class TestEngagementAnalyzer(unittest.TestCase):
    """Test EngagementAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EngagementAnalyzer()

    def test_speaker_stats_update(self):
        """Test speaker statistics update."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=2.0,
            audio_data=audio,
            confidence=0.95
        )

        self.analyzer.update_speaker_stats("speaker_1", segment)

        self.assertIn("speaker_1", self.analyzer.speaker_stats)
        stats = self.analyzer.speaker_stats["speaker_1"]
        self.assertEqual(stats['utterance_count'], 1)
        self.assertAlmostEqual(stats['total_time'], 2.0, places=1)

    def test_engagement_score_calculation(self):
        """Test engagement score calculation."""
        # Create multiple speaker segments
        for i in range(3):
            speaker_id = f"speaker_{i+1}"
            audio = np.random.randn(8000).astype(np.float32) * 0.1

            segment = SpeakerSegment(
                speaker_id=speaker_id,
                start_time=i * 2.0,
                end_time=(i + 1) * 2.0,
                audio_data=audio,
                confidence=0.9
            )

            self.analyzer.update_speaker_stats(speaker_id, segment)

        scores = self.analyzer.calculate_engagement_scores()

        self.assertEqual(len(scores), 3)
        for speaker_id, stats in scores.items():
            self.assertIn('engagement_score', stats)
            self.assertGreaterEqual(stats['engagement_score'], 0)
            self.assertLessEqual(stats['engagement_score'], 100)

    def test_mission_summary(self):
        """Test mission communication summary."""
        # Add some test data
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=2.0,
            audio_data=audio,
            confidence=0.95
        )

        self.analyzer.update_speaker_stats("speaker_1", segment)

        summary = self.analyzer.get_mission_communication_summary()

        self.assertIn('total_speakers', summary)
        self.assertIn('total_utterances', summary)
        self.assertIn('speaker_scores', summary)
        self.assertIn('turn_taking_balance', summary)
        self.assertIn('communication_effectiveness', summary)


if __name__ == '__main__':
    unittest.main()
```

---

## Phase 8: Documentation & Deployment

### 8.1 Create User Guide: `docs/AUDIO_SETUP_GUIDE.md`

**File:** `docs/AUDIO_SETUP_GUIDE.md`

```markdown
# Audio Transcription Setup Guide
## Starship Horizons Learning AI - Bridge Crew Audio Recording

---

## Prerequisites

- Python 3.8 or higher
- Working microphone
- 4GB RAM minimum (8GB recommended for larger models)
- 2GB free disk space (for Whisper models)

---

## Installation

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Download Whisper Model

```bash
# Download the default base model (~74MB)
python scripts/download_whisper_models.py
```

This will download and cache the Whisper model locally. The model only needs to be downloaded once.

**Model Options:**
- `tiny` - Fastest, lowest accuracy (~32MB)
- `base` - **Recommended** - Good balance (~74MB)
- `small` - Better accuracy (~244MB)
- `medium` - High accuracy (~769MB)
- `large-v3` - Best accuracy (~1.5GB)

To use a different model, edit `.env`:
```bash
WHISPER_MODEL_SIZE=small
```

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

Test your microphone and transcription:

```bash
# Test for 30 seconds
python scripts/test_realtime_audio.py 30
```

Speak into your microphone. You should see real-time transcriptions appear.

---

## Configuration

### Basic Configuration (.env)

```bash
# Enable audio capture
ENABLE_AUDIO_CAPTURE=true

# Audio device
AUDIO_INPUT_DEVICE=0

# Whisper model
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Voice Activity Detection
VAD_ENERGY_THRESHOLD=500  # Adjust based on room noise
MIN_SPEECH_DURATION=0.3
MIN_SILENCE_DURATION=0.5

# Speaker Diarization
ENABLE_SPEAKER_DIARIZATION=true
SPEAKER_SIMILARITY_THRESHOLD=0.7
EXPECTED_BRIDGE_CREW=6
```

### Tuning for Your Environment

**Noisy Environment:**
```bash
VAD_ENERGY_THRESHOLD=1000  # Increase threshold
MIN_SILENCE_DURATION=0.8    # Longer pauses required
```

**Quiet Environment:**
```bash
VAD_ENERGY_THRESHOLD=300   # Lower threshold
MIN_SILENCE_DURATION=0.3    # Shorter pauses acceptable
```

**Many Speakers (>6):**
```bash
SPEAKER_SIMILARITY_THRESHOLD=0.8  # More strict matching
EXPECTED_BRIDGE_CREW=8
```

**Few Speakers (2-3):**
```bash
SPEAKER_SIMILARITY_THRESHOLD=0.6  # More lenient
EXPECTED_BRIDGE_CREW=3
```

---

## Usage

### Record a Mission with Audio

```bash
# Record 5-minute mission
python scripts/record_mission_with_audio.py 300 "Training Mission Alpha"
```

This will:
1. Connect to Starship Horizons game
2. Record all game telemetry
3. Capture bridge crew audio
4. Transcribe speech in real-time
5. Identify speakers automatically
6. Generate engagement metrics

### Output Files

After recording, you'll find:

```
data/recordings/
├── GAME_20250102_143045_transcript.json  # Audio transcripts
├── GAME_20250102_143045_events.json      # Game telemetry
└── GAME_20250102_143045_combined.json    # Synchronized timeline
```

### Assign Speaker Names/Roles

After a recording, you can assign names to detected speakers:

```python
from src.integration.game_recorder import GameRecorder

recorder = GameRecorder()
# ... after recording ...

# Assign crew roles
recorder.audio_service._speaker_diarizer.assign_bridge_roles({
    "speaker_1": "Captain",
    "speaker_2": "Helm",
    "speaker_3": "Tactical",
    "speaker_4": "Science",
    "speaker_5": "Engineering",
    "speaker_6": "Communications"
})
```

---

## Troubleshooting

### No Audio Detected

**Problem:** Microphone not working

**Solutions:**
1. Check `AUDIO_INPUT_DEVICE` setting
2. Run `python scripts/list_audio_devices.py`
3. Test with system audio recorder
4. Check OS permissions for microphone access

### Poor Transcription Accuracy

**Problem:** Many transcription errors

**Solutions:**
1. Use a better quality microphone
2. Reduce background noise
3. Use a larger model: `WHISPER_MODEL_SIZE=small`
4. Speak more clearly and directly into mic
5. Adjust `VAD_ENERGY_THRESHOLD` to filter noise

### Speakers Not Identified Correctly

**Problem:** All speech attributed to one speaker

**Solutions:**
1. Lower `SPEAKER_SIMILARITY_THRESHOLD` to 0.6
2. Ensure speakers use separate microphones
3. Check that audio is mixed properly (not mono-mixed)
4. Manually assign roles after recording

### High CPU Usage

**Problem:** Computer slow during recording

**Solutions:**
1. Use smaller model: `WHISPER_MODEL_SIZE=tiny`
2. Reduce workers: `TRANSCRIPTION_WORKERS=1`
3. Increase chunk size: `AUDIO_CHUNK_MS=200`
4. Enable GPU if available: `WHISPER_DEVICE=cuda`

### Empty Transcripts

**Problem:** No transcriptions generated

**Solutions:**
1. Check `MIN_TRANSCRIPTION_CONFIDENCE` setting
2. Verify audio levels with `test_realtime_audio.py`
3. Ensure `TRANSCRIBE_REALTIME=true`
4. Check logs for error messages

---

## Performance Benchmarks

### Model Performance (CPU)

| Model  | Speed    | Accuracy | Memory |
|--------|----------|----------|--------|
| tiny   | ~10x RT  | 75%      | 1GB    |
| base   | ~7x RT   | 80%      | 1GB    |
| small  | ~4x RT   | 85%      | 2GB    |
| medium | ~2x RT   | 90%      | 4GB    |

*RT = Realtime (10x = 10 seconds processed in 1 second)*

### Recommended Configurations

**Real-time Bridge Audio (6+ speakers):**
```bash
WHISPER_MODEL_SIZE=base
WHISPER_COMPUTE_TYPE=int8
TRANSCRIPTION_WORKERS=2
AUDIO_CHUNK_MS=100
```

**Post-Mission Analysis (accuracy priority):**
```bash
WHISPER_MODEL_SIZE=medium
WHISPER_COMPUTE_TYPE=float16
TRANSCRIPTION_WORKERS=4
```

**Low-Resource Systems:**
```bash
WHISPER_MODEL_SIZE=tiny
TRANSCRIPTION_WORKERS=1
AUDIO_CHUNK_MS=200
```

---

## Advanced Usage

### Export Transcript Formats

Transcripts can be exported in multiple formats:

```python
service.export_transcript("mission.txt", format='txt')   # Plain text
service.export_transcript("mission.srt", format='srt')   # Subtitles
service.export_transcript("mission.json", format='json') # Full data
```

### Engagement Analytics

Access detailed crew engagement metrics:

```python
summary = recorder.audio_service.get_engagement_summary()

print(f"Total Speakers: {summary['total_speakers']}")
print(f"Turn Balance: {summary['turn_taking_balance']}/100")
print(f"Effectiveness: {summary['communication_effectiveness']}/100")

for speaker_id, stats in summary['speaker_scores'].items():
    print(f"{speaker_id}: {stats['engagement_score']}/100")
```

### Synchronized Timeline

Get combined game events + audio transcripts:

```python
timeline = recorder.get_combined_timeline()

for entry in timeline:
    if entry['type'] == 'game_event':
        print(f"[GAME] {entry['data']}")
    elif entry['type'] == 'audio_transcript':
        print(f"[{entry['speaker']}] {entry['text']}")
```

---

## Support

For issues or questions:

1. Check this guide's troubleshooting section
2. Review logs in `./logs/`
3. Test with `scripts/test_realtime_audio.py`
4. Open an issue on GitHub

---

## Privacy & Data Retention

By default:
- Raw audio is **not saved** (`SAVE_RAW_AUDIO=false`)
- Transcripts are kept for 30 days
- Speaker IDs are anonymized after 7 days

To change retention:

```bash
TRANSCRIPT_RETENTION_DAYS=30
ANONYMIZE_AFTER_DAYS=7
SAVE_RAW_AUDIO=false  # NEVER change this without crew consent
```

**IMPORTANT:** Always obtain consent from all crew members before recording bridge audio.
```

---

Continuing in next response with deployment checklist and final summary...
