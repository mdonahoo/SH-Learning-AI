#!/usr/bin/env python3
"""
Live Integration Tests for Starship Horizons Learning AI.

These tests verify actual integration with real services:
- Ollama LLM server
- Audio capture hardware
- Game server WebSocket/API
- Multi-bridge recording

Tests skip gracefully when services are unavailable.
Run with: pytest tests/test_integration_live.py -v
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES AND HELPERS
# =============================================================================

def check_ollama_available() -> bool:
    """Check if Ollama server is accessible."""
    try:
        from src.llm.ollama_client import OllamaClient
        client = OllamaClient()
        return client.check_connection()
    except Exception:
        return False


def check_game_server_available() -> bool:
    """Check if game server is accessible."""
    try:
        from src.integration.starship_horizons_client import StarshipHorizonsClient
        host = os.getenv('GAME_HOST', 'localhost')
        port = os.getenv('GAME_PORT_API', '1864')
        client = StarshipHorizonsClient(f"http://{host}:{port}")
        return client.test_connection()
    except Exception:
        return False


def check_audio_available() -> bool:
    """Check if audio capture is available."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        return device_count > 0
    except Exception:
        return False


def check_whisper_available() -> bool:
    """Check if Whisper model is available."""
    try:
        from src.audio.whisper_transcriber import WhisperTranscriber
        transcriber = WhisperTranscriber()
        model = transcriber.load_model()
        return model is not None
    except Exception:
        return False


# Pytest markers for conditional skipping
requires_ollama = pytest.mark.skipif(
    not check_ollama_available(),
    reason="Ollama server not available"
)

requires_game_server = pytest.mark.skipif(
    not check_game_server_available(),
    reason="Game server not available"
)

requires_audio = pytest.mark.skipif(
    not check_audio_available(),
    reason="Audio hardware not available"
)

requires_whisper = pytest.mark.skipif(
    not check_whisper_available(),
    reason="Whisper model not available"
)


# =============================================================================
# OLLAMA INTEGRATION TESTS
# =============================================================================

class TestOllamaIntegration:
    """Test real Ollama server integration."""

    @requires_ollama
    def test_ollama_connection(self):
        """Test actual connection to Ollama server."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()
        assert client.check_connection() is True
        assert client.host is not None

    @requires_ollama
    def test_ollama_list_models(self):
        """Test listing models from real Ollama server."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()
        models = client.list_models()

        assert isinstance(models, list)
        # Should have at least one model if Ollama is set up
        print(f"Available models: {models}")

    @requires_ollama
    def test_ollama_generate_simple(self):
        """Test simple text generation with real Ollama."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()

        # Simple prompt that should work with any model
        result = client.generate(
            prompt="Say 'hello' and nothing else.",
            temperature=0.1,
            max_tokens=50
        )

        assert result is not None
        assert len(result) > 0
        print(f"Generated: {result[:100]}")

    @requires_ollama
    def test_ollama_generate_mission_summary(self):
        """Test mission summary generation with real Ollama."""
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient()

        # Minimal mission data
        mission_data = {
            "mission_id": "TEST_001",
            "mission_name": "Test Mission",
            "duration": "00:05:00",
            "events": [
                {"type": "mission_start", "timestamp": "00:00:00"},
                {"type": "alert_yellow", "timestamp": "00:01:00"},
                {"type": "mission_complete", "timestamp": "00:05:00"}
            ],
            "transcripts": [
                {"speaker": "Captain", "text": "Set condition yellow."}
            ]
        }

        result = client.generate_mission_summary(mission_data, style="professional")

        assert result is not None
        assert len(result) > 100  # Should be a substantial summary
        print(f"Summary length: {len(result)} chars")
        print(f"Summary preview: {result[:200]}...")

    @requires_ollama
    def test_ollama_timeout_handling(self):
        """Test that timeout is properly configured for remote servers."""
        from src.llm.ollama_client import OllamaClient

        # Test with custom timeout
        client = OllamaClient(timeout=5)
        assert client.timeout == 5

        # Test from environment
        original = os.environ.get('OLLAMA_TIMEOUT')
        os.environ['OLLAMA_TIMEOUT'] = '300'
        client2 = OllamaClient()
        assert client2.timeout == 300

        # Restore
        if original:
            os.environ['OLLAMA_TIMEOUT'] = original
        else:
            os.environ.pop('OLLAMA_TIMEOUT', None)

    @requires_ollama
    def test_ollama_remote_host_configuration(self):
        """Test that remote host can be configured."""
        from src.llm.ollama_client import OllamaClient

        # Get current host from env
        host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        client = OllamaClient()

        assert client.host == host.rstrip('/')
        assert client.check_connection() is True

        print(f"Connected to Ollama at: {client.host}")
        print(f"Using model: {client.model}")


# =============================================================================
# GAME SERVER INTEGRATION TESTS
# =============================================================================

class TestGameServerIntegration:
    """Test real game server integration."""

    @requires_game_server
    def test_game_server_connection(self):
        """Test actual connection to game server."""
        from src.integration.starship_horizons_client import StarshipHorizonsClient

        host = os.getenv('GAME_HOST', 'localhost')
        port = os.getenv('GAME_PORT_API', '1864')
        client = StarshipHorizonsClient(f"http://{host}:{port}")

        assert client.test_connection() is True

    @requires_game_server
    def test_game_server_status(self):
        """Test fetching game status from real server."""
        from src.integration.starship_horizons_client import StarshipHorizonsClient

        host = os.getenv('GAME_HOST', 'localhost')
        port = os.getenv('GAME_PORT_API', '1864')
        client = StarshipHorizonsClient(f"http://{host}:{port}")

        status = client.get_game_status()

        assert status is not None
        assert isinstance(status, dict)
        print(f"Game status: {status}")

    @requires_game_server
    def test_game_recorder_with_real_server(self):
        """Test GameRecorder with real game server (short recording)."""
        from src.integration.game_recorder import GameRecorder

        # Disable audio for this test
        os.environ['ENABLE_AUDIO_CAPTURE'] = 'false'

        recorder = GameRecorder(bridge_id="Test-Bridge")

        assert recorder.client.test_connection() is True

        # Start recording
        mission_id = recorder.start_recording(mission_name="Integration Test")
        assert mission_id is not None
        assert "Test-Bridge" in mission_id

        # Record for 2 seconds
        time.sleep(2)

        # Stop and get summary
        summary = recorder.stop_recording()

        assert summary is not None
        assert 'mission_id' in summary
        assert 'statistics' in summary

        print(f"Recorded {summary['statistics'].get('total_events', 0)} events")

    @requires_game_server
    def test_websocket_telemetry_client(self):
        """Test WebSocket telemetry connection to real server."""
        from src.integration.websocket_telemetry_client import ShipTelemetryClient

        host = os.getenv('GAME_HOST', 'localhost')
        port = int(os.getenv('GAME_PORT_WS', '1865'))

        client = ShipTelemetryClient(host=host, port=port)

        # Try to connect (may fail if WebSocket not available)
        try:
            connected = client.connect()
            if connected:
                # Get some telemetry
                time.sleep(1)
                status = client.get_ship_status_summary()
                print(f"Ship status: {status}")
                client.disconnect()
        except Exception as e:
            pytest.skip(f"WebSocket connection failed: {e}")

    @requires_game_server
    def test_enhanced_game_client(self):
        """Test EnhancedGameClient with real server."""
        from src.integration.enhanced_game_client import EnhancedGameClient

        client = EnhancedGameClient()

        assert client.test_connection() is True

        # Get initial status
        status = client.get_game_status()
        assert status is not None

        # Start monitoring briefly
        events_received = []

        def event_handler(event):
            events_received.append(event)

        client.add_event_callback(event_handler)
        client.start_monitoring(interval=0.5)

        time.sleep(2)

        client.stop_monitoring()

        print(f"Received {len(events_received)} events during monitoring")


# =============================================================================
# AUDIO INTEGRATION TESTS
# =============================================================================

class TestAudioIntegration:
    """Test real audio capture integration."""

    @requires_audio
    def test_audio_device_enumeration(self):
        """Test enumerating audio devices."""
        import pyaudio

        p = pyaudio.PyAudio()
        device_count = p.get_device_count()

        assert device_count > 0

        print(f"Found {device_count} audio devices:")
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (inputs: {info['maxInputChannels']})")

        p.terminate()

    @requires_audio
    def test_audio_capture_manager_initialization(self):
        """Test AudioCaptureManager initialization with real device."""
        from src.audio.capture import AudioCaptureManager

        manager = AudioCaptureManager(
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            enable_vad=False
        )

        assert manager is not None
        assert manager.sample_rate == 16000

    @requires_audio
    def test_audio_capture_short_recording(self):
        """Test short audio capture (1 second)."""
        from src.audio.capture import AudioCaptureManager
        import numpy as np

        captured_audio = []

        def on_segment(audio_data, start_time, end_time):
            captured_audio.append(audio_data)

        manager = AudioCaptureManager(
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            enable_vad=False  # Disable VAD to get raw chunks
        )

        # Set callback
        manager.set_segment_callback(on_segment)

        # Start capture
        success = manager.start_capture()
        if not success:
            pytest.skip("Failed to start audio capture - no input device")

        # Capture for 1 second
        time.sleep(1)

        # Stop capture
        manager.stop_capture()

        assert len(captured_audio) > 0
        print(f"Captured {len(captured_audio)} audio chunks")

        # Verify audio data format
        if captured_audio:
            chunk = captured_audio[0]
            assert isinstance(chunk, np.ndarray)
            print(f"Chunk shape: {chunk.shape}, dtype: {chunk.dtype}")

    @requires_audio
    def test_voice_activity_detection_real(self):
        """Test VAD with real audio input."""
        from src.audio.capture import AudioCaptureManager
        from src.audio.speaker_diarization import SimpleVAD
        import numpy as np

        speech_segments = []

        def on_segment(audio_data, start_time, end_time):
            speech_segments.append({
                'duration': end_time - start_time,
                'samples': len(audio_data)
            })

        # Create VAD with sensitive thresholds
        vad = SimpleVAD(
            energy_threshold=0.02,
            min_speech_duration=0.1,
            min_silence_duration=0.2,
            sample_rate=16000
        )

        manager = AudioCaptureManager(
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            enable_vad=True,
            vad=vad
        )

        # Set callback for speech segments
        manager.set_segment_callback(on_segment)

        success = manager.start_capture()
        if not success:
            pytest.skip("Failed to start audio capture")

        print("Recording for 3 seconds - speak to test VAD...")
        time.sleep(3)

        manager.stop_capture()

        print(f"Detected {len(speech_segments)} speech segments")
        for i, seg in enumerate(speech_segments):
            print(f"  Segment {i}: {seg['duration']:.2f}s, {seg['samples']} samples")


# =============================================================================
# WHISPER TRANSCRIPTION TESTS
# =============================================================================

class TestWhisperIntegration:
    """Test real Whisper transcription."""

    @requires_whisper
    def test_whisper_model_loading(self):
        """Test Whisper model loads correctly."""
        from src.audio.whisper_transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()

        # Load model
        success = transcriber.load_model()
        assert success is True
        assert transcriber._model_loaded is True
        print(f"Loaded Whisper model: {transcriber.model_size}")

    @requires_whisper
    def test_whisper_transcribe_silence(self):
        """Test Whisper handles silence correctly."""
        from src.audio.whisper_transcriber import WhisperTranscriber
        import numpy as np

        transcriber = WhisperTranscriber()
        transcriber.load_model()

        # Generate 2 seconds of silence (needs minimum length)
        silence = np.zeros(32000, dtype=np.float32)

        # Use private method for synchronous testing
        result = transcriber._transcribe_segment(
            audio_data=silence,
            timestamp=0.0,
            speaker_id=None,
            metadata={}
        )

        # Should return empty or None for silence
        if result:
            print(f"Silence transcription: {result.get('text', '')}")
        else:
            print("Silence correctly returned no transcription")

    @requires_whisper
    @requires_audio
    def test_whisper_live_transcription(self):
        """Test live audio transcription with Whisper."""
        from src.audio.capture import AudioCaptureManager
        from src.audio.speaker_diarization import SimpleVAD
        from src.audio.whisper_transcriber import WhisperTranscriber
        import numpy as np

        transcriber = WhisperTranscriber()
        transcriber.load_model()

        transcriptions = []

        def on_segment(audio_data, start_time, end_time):
            # Only process segments with enough audio
            if len(audio_data) < 16000:  # At least 1 second
                return

            # Transcribe the speech segment
            result = transcriber._transcribe_segment(
                audio_data=audio_data,
                timestamp=start_time,
                speaker_id=None,
                metadata={}
            )
            if result and result.get('text'):
                transcriptions.append({
                    'text': result['text'],
                    'confidence': result.get('confidence', 0),
                    'duration': end_time - start_time
                })
                print(f"Transcribed: {result['text'][:50]}...")

        # Create VAD with reasonable thresholds
        vad = SimpleVAD(
            energy_threshold=0.02,
            min_speech_duration=0.5,
            min_silence_duration=0.5,
            sample_rate=16000
        )

        manager = AudioCaptureManager(
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            enable_vad=True,
            vad=vad
        )

        manager.set_segment_callback(on_segment)

        success = manager.start_capture()
        if not success:
            pytest.skip("Failed to start audio capture")

        print("Recording for 5 seconds - speak to test transcription...")
        time.sleep(5)

        manager.stop_capture()

        print(f"\nTotal transcriptions: {len(transcriptions)}")
        for t in transcriptions:
            print(f"  [{t['confidence']:.2f}] {t['text']}")


# =============================================================================
# MULTI-BRIDGE INTEGRATION TESTS
# =============================================================================

class TestMultiBridgeIntegration:
    """Test multi-bridge functionality with real components."""

    def test_bridge_id_directory_creation(self):
        """Test that bridge-specific directories are created correctly."""
        from src.integration.game_recorder import GameRecorder
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch the base recording path
            with patch.object(Path, 'mkdir'):
                os.environ['ENABLE_AUDIO_CAPTURE'] = 'false'

                # Mock the client to avoid needing real server
                with patch('src.integration.game_recorder.StarshipHorizonsClient') as mock_client:
                    mock_client.return_value.test_connection.return_value = True
                    mock_client.return_value.get_game_status.return_value = {}

                    recorder = GameRecorder(bridge_id="Bridge-Alpha")
                    mission_id = recorder.start_recording("Test Mission")

                    # Verify mission ID format
                    assert mission_id.startswith("Bridge-Alpha_GAME_")

                    # Clean up
                    recorder.is_recording = False

    def test_multiple_bridge_ids_unique(self):
        """Test that different bridges get unique mission IDs."""
        from src.metrics.event_recorder import EventRecorder
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        recorder_alpha = EventRecorder(
            mission_id=f"Bridge-Alpha_GAME_{timestamp}",
            bridge_id="Bridge-Alpha"
        )

        recorder_beta = EventRecorder(
            mission_id=f"Bridge-Beta_GAME_{timestamp}",
            bridge_id="Bridge-Beta"
        )

        recorder_charlie = EventRecorder(
            mission_id=f"Bridge-Charlie_GAME_{timestamp}",
            bridge_id="Bridge-Charlie"
        )

        # All should have different mission IDs even with same timestamp
        assert recorder_alpha.mission_id != recorder_beta.mission_id
        assert recorder_beta.mission_id != recorder_charlie.mission_id
        assert recorder_alpha.bridge_id == "Bridge-Alpha"
        assert recorder_beta.bridge_id == "Bridge-Beta"
        assert recorder_charlie.bridge_id == "Bridge-Charlie"

    def test_bridge_metadata_in_exports(self):
        """Test that bridge metadata is correctly exported."""
        from src.metrics.event_recorder import EventRecorder
        from src.metrics.audio_transcript import AudioTranscriptService
        from src.metrics.mission_summarizer import MissionSummarizer
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test EventRecorder export
            event_recorder = EventRecorder(
                mission_id="Bridge-Alpha_GAME_20260114_120000",
                mission_name="Test Mission",
                bridge_id="Bridge-Alpha"
            )
            event_recorder.record_event("test", "test", {"data": "value"})

            events_file = Path(tmpdir) / "events.json"
            event_recorder.export_to_json(events_file)

            with open(events_file) as f:
                events_data = json.load(f)
            assert events_data['bridge_id'] == "Bridge-Alpha"

            # Test AudioTranscriptService export
            os.environ['ENABLE_AUDIO_CAPTURE'] = 'false'
            audio_service = AudioTranscriptService(
                mission_id="Bridge-Alpha_GAME_20260114_120000",
                bridge_id="Bridge-Alpha"
            )
            audio_service.transcripts.append({
                "timestamp": datetime.now(),
                "speaker": "Captain",
                "text": "Test",
                "confidence": 0.9
            })

            transcript_file = Path(tmpdir) / "transcripts.json"
            audio_service.export_transcript(transcript_file)

            with open(transcript_file) as f:
                transcript_data = json.load(f)
            assert transcript_data['bridge_id'] == "Bridge-Alpha"

            # Test MissionSummarizer export
            summarizer = MissionSummarizer(
                mission_id="Bridge-Alpha_GAME_20260114_120000",
                mission_name="Test Mission",
                bridge_id="Bridge-Alpha"
            )
            summarizer.load_events([{
                "event_id": "E001",
                "timestamp": datetime.now(),
                "event_type": "test",
                "category": "test",
                "data": {}
            }])

            report_file = Path(tmpdir) / "report.json"
            summarizer.export_report(report_file, format="json")

            with open(report_file) as f:
                report_data = json.load(f)
            assert report_data['bridge_id'] == "Bridge-Alpha"

    @requires_game_server
    def test_multi_bridge_concurrent_recording(self):
        """Test that multiple bridges can record concurrently (simulated)."""
        from src.metrics.event_recorder import EventRecorder
        import threading

        results = {}
        errors = []

        def record_bridge(bridge_id: str, duration: float):
            try:
                recorder = EventRecorder(
                    mission_id=f"{bridge_id}_GAME_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    mission_name=f"Test Mission - {bridge_id}",
                    bridge_id=bridge_id
                )

                # Simulate recording events
                for i in range(10):
                    recorder.record_event(
                        event_type="test_event",
                        category="test",
                        data={"bridge": bridge_id, "sequence": i}
                    )
                    time.sleep(duration / 10)

                stats = recorder.get_statistics()
                results[bridge_id] = {
                    'events': stats['total_events'],
                    'mission_id': recorder.mission_id
                }
            except Exception as e:
                errors.append(f"{bridge_id}: {e}")

        # Start three "bridges" concurrently
        threads = [
            threading.Thread(target=record_bridge, args=("Bridge-Alpha", 1.0)),
            threading.Thread(target=record_bridge, args=("Bridge-Beta", 1.0)),
            threading.Thread(target=record_bridge, args=("Bridge-Charlie", 1.0))
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify results
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 3

        for bridge_id, data in results.items():
            assert data['events'] == 10
            assert bridge_id in data['mission_id']
            print(f"{bridge_id}: {data['events']} events, ID={data['mission_id']}")


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestEndToEndIntegration:
    """Full end-to-end integration tests."""

    @requires_game_server
    @requires_ollama
    def test_full_recording_with_llm_report(self):
        """Test complete recording flow with LLM report generation."""
        from src.integration.game_recorder import GameRecorder

        os.environ['ENABLE_AUDIO_CAPTURE'] = 'false'
        os.environ['ENABLE_LLM_REPORTS'] = 'true'

        recorder = GameRecorder(bridge_id="Integration-Test")

        # Start recording
        mission_id = recorder.start_recording(mission_name="E2E Integration Test")
        print(f"Started recording: {mission_id}")

        # Record for 5 seconds
        time.sleep(5)

        # Stop recording
        summary = recorder.stop_recording()

        assert summary is not None
        assert 'mission_id' in summary
        print(f"Recording summary: {summary['statistics']}")

        # Generate LLM summary
        summarizer = recorder.generate_summary()
        if summarizer:
            # This will use real Ollama
            llm_summary = summarizer.generate_llm_summary()
            assert llm_summary is not None
            print(f"LLM Summary preview: {llm_summary[:200]}...")

    @requires_audio
    @requires_whisper
    def test_audio_to_transcript_pipeline(self):
        """Test complete audio capture to transcription pipeline."""
        from src.metrics.audio_transcript import AudioTranscriptService

        os.environ['ENABLE_AUDIO_CAPTURE'] = 'true'

        service = AudioTranscriptService(
            mission_id="Audio-Pipeline-Test",
            sample_rate=16000,
            channels=1,
            auto_transcribe=True,
            bridge_id="Test-Bridge"
        )

        # Start capture
        success = service.start_audio_capture()
        if not success:
            pytest.skip("Failed to start audio capture")

        print("Recording for 5 seconds - speak to test pipeline...")

        # Start transcription
        service.start_realtime_transcription()

        time.sleep(5)

        # Stop
        service.stop_realtime_transcription()
        service.stop_audio_capture()

        # Check results
        transcripts = service.get_all_transcripts()
        print(f"Captured {len(transcripts)} transcripts")

        for t in transcripts:
            print(f"  [{t.get('speaker', 'Unknown')}]: {t.get('text', '')[:50]}")

        # Export
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            service.export_transcript(Path(f.name))
            print(f"Exported to: {f.name}")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and stress tests."""

    def test_event_recorder_high_volume(self):
        """Test EventRecorder handles high event volume."""
        from src.metrics.event_recorder import EventRecorder
        import time

        recorder = EventRecorder(
            mission_id="Perf-Test",
            mission_name="Performance Test",
            bridge_id="Perf-Bridge"
        )

        # Record 1000 events
        start = time.time()
        for i in range(1000):
            recorder.record_event(
                event_type="perf_test",
                category="performance",
                data={"sequence": i, "payload": "x" * 100}
            )
        elapsed = time.time() - start

        stats = recorder.get_statistics()
        assert stats['total_events'] == 1000

        print(f"Recorded 1000 events in {elapsed:.3f}s ({1000/elapsed:.0f} events/sec)")
        assert elapsed < 5.0, "Should record 1000 events in under 5 seconds"

    def test_event_recorder_concurrent_writes(self):
        """Test EventRecorder thread safety under concurrent writes."""
        from src.metrics.event_recorder import EventRecorder
        import threading

        recorder = EventRecorder(
            mission_id="Concurrent-Test",
            mission_name="Concurrent Test",
            bridge_id="Concurrent-Bridge"
        )

        errors = []

        def write_events(thread_id: int, count: int):
            try:
                for i in range(count):
                    recorder.record_event(
                        event_type="concurrent_test",
                        category="test",
                        data={"thread": thread_id, "sequence": i}
                    )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # 10 threads, 100 events each
        threads = [
            threading.Thread(target=write_events, args=(i, 100))
            for i in range(10)
        ]

        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        assert len(errors) == 0, f"Errors: {errors}"

        stats = recorder.get_statistics()
        assert stats['total_events'] == 1000

        print(f"10 threads x 100 events = {stats['total_events']} in {elapsed:.3f}s")

    @requires_whisper
    def test_whisper_latency(self):
        """Measure Whisper transcription latency."""
        from src.audio.whisper_transcriber import WhisperTranscriber
        import numpy as np

        transcriber = WhisperTranscriber()
        transcriber.load_model()

        # Generate 3 seconds of test audio (silence with some noise)
        np.random.seed(42)
        test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.01

        # Warm up
        transcriber._transcribe_segment(test_audio[:16000], 0, None, {})

        # Measure
        times = []
        for i in range(3):
            start = time.time()
            transcriber._transcribe_segment(test_audio, float(i), None, {})
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        print(f"Whisper latency for 3s audio: {avg_time:.2f}s (avg of 3 runs)")
        print(f"Real-time factor: {avg_time / 3.0:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
