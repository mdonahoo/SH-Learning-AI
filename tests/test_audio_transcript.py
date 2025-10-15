#!/usr/bin/env python3
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import wave
import numpy as np


class TestAudioTranscriptService:
    """Test suite for audio recording and transcription service."""

    def test_audio_transcript_service_initialization(self):
        """Test initializing the audio transcript service."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(
            mission_id="AUDIO_TEST_001",
            sample_rate=16000,
            channels=1
        )

        assert service.mission_id == "AUDIO_TEST_001"
        assert service.sample_rate == 16000
        assert service.channels == 1
        assert service.is_recording == False
        assert len(service.audio_segments) == 0

    def test_start_stop_recording(self):
        """Test starting and stopping audio recording."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="REC_TEST")

        # Start recording
        service.start_recording()
        assert service.is_recording == True
        assert service.recording_start_time is not None

        # Stop recording
        duration = service.stop_recording()
        assert service.is_recording == False
        assert duration > 0

    def test_audio_buffer_management(self):
        """Test managing audio buffers for continuous recording."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(
            mission_id="BUFFER_TEST",
            buffer_duration=5  # 5 second buffers
        )

        service.start_recording()

        # Simulate adding audio data
        sample_data = np.random.random(16000).astype(np.float32)  # 1 second of audio
        for _ in range(10):  # 10 seconds of audio
            service.add_audio_chunk(sample_data)

        service.stop_recording()

        # Should have created 2 complete buffers
        assert len(service.audio_segments) >= 2

    def test_voice_activity_detection(self):
        """Test detecting voice activity in audio stream."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="VAD_TEST")

        # Create silent audio
        silent_audio = np.zeros(16000, dtype=np.float32)
        assert service.detect_voice_activity(silent_audio) == False

        # Create audio with signal
        voice_audio = np.random.normal(0, 0.3, 16000).astype(np.float32)
        assert service.detect_voice_activity(voice_audio) == True

    def test_speaker_diarization(self):
        """Test identifying different speakers in audio."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="SPEAKER_TEST")

        # Simulate audio segments from different speakers
        segment1 = {
            "timestamp": datetime.now(),
            "audio_data": np.random.random(16000),
            "duration": 1.0
        }

        segment2 = {
            "timestamp": datetime.now() + timedelta(seconds=2),
            "audio_data": np.random.random(16000),
            "duration": 1.0
        }

        # Process segments for speaker identification
        speaker1 = service.identify_speaker(segment1["audio_data"])
        speaker2 = service.identify_speaker(segment2["audio_data"])

        assert speaker1 is not None
        assert speaker2 is not None

    def test_audio_transcription(self):
        """Test transcribing audio to text."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="TRANSCRIPT_TEST")

        # Create mock audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Create a simple WAV file
            with wave.open(tmp.name, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.random.bytes(32000))  # 1 second

            # Transcribe audio
            transcript = service.transcribe_audio(tmp.name)

            assert transcript is not None
            assert "text" in transcript
            assert "confidence" in transcript
            assert "timestamp" in transcript

            Path(tmp.name).unlink()

    def test_realtime_transcription_queue(self):
        """Test queuing audio for realtime transcription."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="QUEUE_TEST")

        service.start_realtime_transcription()

        # Add audio chunks to queue
        for i in range(5):
            chunk = np.random.random(16000).astype(np.float32)
            service.queue_for_transcription(chunk, timestamp=datetime.now())

        # Get transcription results
        results = service.get_transcription_results(timeout=1)
        assert results is not None

        service.stop_realtime_transcription()

    def test_transcript_export(self):
        """Test exporting transcripts with timestamps."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="EXPORT_TEST")

        # Add mock transcripts
        service.add_transcript(
            timestamp=datetime.now(),
            speaker="Captain",
            text="Set course for the Neutral Zone",
            confidence=0.95
        )

        service.add_transcript(
            timestamp=datetime.now() + timedelta(seconds=5),
            speaker="Helm",
            text="Course laid in, Captain",
            confidence=0.92
        )

        # Export transcripts
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "transcript.json"
            service.export_transcript(filepath)

            assert filepath.exists()

            # Verify content
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert data["mission_id"] == "EXPORT_TEST"
            assert len(data["transcripts"]) == 2
            assert data["transcripts"][0]["speaker"] == "Captain"

    def test_audio_file_storage(self):
        """Test storing audio files with proper organization."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="STORAGE_TEST")

        with tempfile.TemporaryDirectory() as tmpdir:
            service.set_storage_path(tmpdir)

            # Save audio segment
            audio_data = np.random.random(16000).astype(np.float32)
            filepath = service.save_audio_segment(
                audio_data,
                timestamp=datetime.now(),
                segment_id="SEG_001"
            )

            assert filepath.exists()
            assert "STORAGE_TEST" in str(filepath)
            assert filepath.suffix == ".wav"

    def test_transcript_synchronization(self):
        """Test synchronizing transcripts with game events."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="SYNC_TEST")

        # Add transcript with event marker
        event_time = datetime.now()
        service.add_transcript(
            timestamp=event_time,
            speaker="Tactical",
            text="Firing photon torpedoes",
            confidence=0.90,
            event_id="EVENT_FIRE_001"
        )

        # Get transcript at specific time
        transcript = service.get_transcript_at_time(event_time)
        assert transcript is not None
        assert transcript["event_id"] == "EVENT_FIRE_001"

    def test_audio_metrics(self):
        """Test calculating audio metrics for quality assessment."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="METRICS_TEST")

        # Generate test audio
        audio_data = np.random.normal(0, 0.1, 16000).astype(np.float32)

        metrics = service.calculate_audio_metrics(audio_data)

        assert "signal_to_noise_ratio" in metrics
        assert "peak_amplitude" in metrics
        assert "rms_energy" in metrics
        assert "zero_crossing_rate" in metrics
        assert metrics["signal_to_noise_ratio"] > 0


class TestAudioTranscriptIntegration:
    """Integration tests for audio transcript service."""

    def test_continuous_recording_with_transcription(self):
        """Test continuous recording with automatic transcription."""
        from src.metrics.audio_transcript import AudioTranscriptService
        import time

        service = AudioTranscriptService(
            mission_id="CONTINUOUS_TEST",
            auto_transcribe=True
        )

        service.start_recording()
        service.start_realtime_transcription()

        # Simulate 5 seconds of recording
        for _ in range(5):
            chunk = np.random.normal(0, 0.1, 16000).astype(np.float32)
            service.add_audio_chunk(chunk)
            time.sleep(1)

        service.stop_recording()
        service.stop_realtime_transcription()

        # Should have audio and attempted transcription
        assert len(service.audio_segments) > 0
        assert service.get_total_duration() >= 5

    def test_multi_speaker_conversation(self):
        """Test handling multi-speaker conversations."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="CONVERSATION_TEST")

        # Simulate conversation
        conversation = [
            ("Captain", "Status report, all stations"),
            ("Helm", "Navigation systems operational"),
            ("Tactical", "Weapons systems online"),
            ("Science", "Sensors detecting no anomalies"),
            ("Engineering", "Warp core stable at 98%"),
        ]

        for speaker, text in conversation:
            service.add_transcript(
                timestamp=datetime.now(),
                speaker=speaker,
                text=text,
                confidence=0.9
            )

        # Get conversation summary
        summary = service.get_conversation_summary()
        assert summary["total_utterances"] == 5
        assert summary["unique_speakers"] == 5
        assert "Captain" in summary["speakers"]

    def test_audio_recovery_after_interruption(self):
        """Test recovering audio recording after interruption."""
        from src.metrics.audio_transcript import AudioTranscriptService

        service = AudioTranscriptService(mission_id="RECOVERY_TEST")

        # Start recording
        service.start_recording()
        service.add_audio_chunk(np.random.random(16000))

        # Simulate interruption
        service.pause_recording()
        assert service.is_paused == True

        # Resume recording
        service.resume_recording()
        service.add_audio_chunk(np.random.random(16000))

        service.stop_recording()

        # Should have both segments
        assert len(service.audio_segments) == 2