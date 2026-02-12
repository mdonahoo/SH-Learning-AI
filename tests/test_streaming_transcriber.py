"""Tests for streaming transcription module."""

import time
import threading
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.web.streaming_transcriber import (
    StreamingTranscriptionSession,
    StreamingTranscriptionManager,
    STREAMING_MIN_CHUNK_SECONDS,
    STREAMING_SESSION_TIMEOUT,
)


class TestStreamingTranscriptionSession:
    """Test suite for StreamingTranscriptionSession."""

    @pytest.fixture
    def session(self):
        """Create a test session."""
        return StreamingTranscriptionSession(
            session_id='test-session-123',
            workspace_id='test-workspace-456'
        )

    def test_initialization(self, session):
        """Test session initializes with correct default values."""
        assert session.session_id == 'test-session-123'
        assert session.workspace_id == 'test-workspace-456'
        assert session.audio_chunks == []
        assert session.segments == []
        assert session.last_transcribed_offset == 0.0
        assert session.is_finalized is False
        assert session.total_audio_duration == 0.0

    def test_add_chunk(self, session):
        """Test adding audio chunks."""
        session.add_chunk(b'\x00\x01\x02')
        session.add_chunk(b'\x03\x04\x05')
        assert len(session.audio_chunks) == 2

    def test_get_accumulated_audio(self, session):
        """Test retrieving accumulated audio bytes."""
        session.add_chunk(b'\x00\x01')
        session.add_chunk(b'\x02\x03')
        result = session.get_accumulated_audio()
        assert result == b'\x00\x01\x02\x03'

    def test_get_accumulated_audio_empty(self, session):
        """Test accumulated audio when no chunks added."""
        assert session.get_accumulated_audio() == b''

    def test_add_segments(self, session):
        """Test adding transcript segments."""
        segments = [
            {'start': 0.0, 'end': 2.0, 'text': 'Hello'},
            {'start': 2.0, 'end': 4.0, 'text': 'World'},
        ]
        session.add_segments(segments)
        assert len(session.segments) == 2
        assert session.segments[0]['text'] == 'Hello'

    def test_get_segments_returns_copy(self, session):
        """Test that get_segments returns a copy, not the original list."""
        session.add_segments([{'start': 0.0, 'end': 1.0, 'text': 'Test'}])
        segments = session.get_segments()
        segments.append({'start': 1.0, 'end': 2.0, 'text': 'Extra'})
        assert len(session.get_segments()) == 1

    def test_is_expired_fresh(self, session):
        """Test that fresh sessions are not expired."""
        assert session.is_expired(timeout=3600) is False

    def test_is_expired_timeout(self, session):
        """Test that old sessions are expired."""
        session.last_activity = time.time() - 7200  # 2 hours ago
        assert session.is_expired(timeout=3600) is True

    def test_is_expired_custom_timeout(self, session):
        """Test expiration with custom timeout."""
        session.last_activity = time.time() - 5
        assert session.is_expired(timeout=3) is True
        assert session.is_expired(timeout=10) is False

    def test_add_chunk_updates_activity(self, session):
        """Test that adding chunks updates last_activity."""
        session.last_activity = time.time() - 100
        old_activity = session.last_activity
        session.add_chunk(b'\x00')
        assert session.last_activity > old_activity

    def test_thread_safety_chunks(self, session):
        """Test concurrent chunk additions are thread-safe."""
        errors = []

        def add_chunks():
            try:
                for i in range(100):
                    session.add_chunk(bytes([i % 256]))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_chunks) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(session.audio_chunks) == 400


class TestStreamingTranscriptionManager:
    """Test suite for StreamingTranscriptionManager."""

    @pytest.fixture
    def manager(self):
        """Create a test manager."""
        return StreamingTranscriptionManager()

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.active_session_count == 0

    def test_create_session(self, manager):
        """Test creating a new session."""
        session_id = manager.create_session('workspace-1')
        assert session_id is not None
        assert len(session_id) > 0
        assert manager.active_session_count == 1

    def test_create_multiple_sessions(self, manager):
        """Test creating multiple sessions."""
        id1 = manager.create_session('workspace-1')
        id2 = manager.create_session('workspace-2')
        assert id1 != id2
        assert manager.active_session_count == 2

    def test_get_session(self, manager):
        """Test retrieving a session by ID."""
        session_id = manager.create_session('workspace-1')
        session = manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.workspace_id == 'workspace-1'

    def test_get_session_not_found(self, manager):
        """Test retrieving a nonexistent session."""
        assert manager.get_session('nonexistent') is None

    def test_add_audio_chunk(self, manager):
        """Test adding audio chunks to a session."""
        session_id = manager.create_session('workspace-1')
        result = manager.add_audio_chunk(session_id, b'\x00\x01\x02')
        assert result is True
        session = manager.get_session(session_id)
        assert len(session.audio_chunks) == 1

    def test_add_audio_chunk_not_found(self, manager):
        """Test adding chunk to nonexistent session."""
        result = manager.add_audio_chunk('nonexistent', b'\x00')
        assert result is False

    def test_finalize_session(self, manager):
        """Test finalizing a session."""
        session_id = manager.create_session('workspace-1')
        session = manager.get_session(session_id)
        session.add_segments([
            {'start': 0.0, 'end': 2.0, 'text': 'Hello'},
        ])
        session.total_audio_duration = 2.0

        result = manager.finalize_session(session_id)
        assert result is not None
        assert result['session_id'] == session_id
        assert len(result['segments']) == 1
        assert result['segments'][0]['text'] == 'Hello'
        assert result['info']['duration'] == 2.0
        assert result['info']['streaming'] is True

        # Session removed after finalization
        assert manager.get_session(session_id) is None
        assert manager.active_session_count == 0

    def test_finalize_session_not_found(self, manager):
        """Test finalizing a nonexistent session."""
        assert manager.finalize_session('nonexistent') is None

    def test_cleanup_expired(self, manager):
        """Test cleaning up expired sessions."""
        sid1 = manager.create_session('workspace-1')
        sid2 = manager.create_session('workspace-2')

        # Make one session expired
        session1 = manager.get_session(sid1)
        session1.last_activity = time.time() - 7200

        cleaned = manager.cleanup_expired(timeout=3600)
        assert cleaned == 1
        assert manager.get_session(sid1) is None
        assert manager.get_session(sid2) is not None
        assert manager.active_session_count == 1

    def test_cleanup_no_expired(self, manager):
        """Test cleanup when no sessions are expired."""
        manager.create_session('workspace-1')
        cleaned = manager.cleanup_expired(timeout=3600)
        assert cleaned == 0
        assert manager.active_session_count == 1

    @patch('src.web.streaming_transcriber.PYDUB_AVAILABLE', True)
    @patch('src.web.streaming_transcriber.AudioSegment')
    def test_process_new_audio_insufficient(self, mock_audio_seg, manager):
        """Test process_new_audio skips when not enough new audio."""
        session_id = manager.create_session('workspace-1')
        manager.add_audio_chunk(session_id, b'\x00' * 100)

        # Mock short audio (less than minimum threshold)
        mock_audio = MagicMock()
        mock_audio.__len__ = Mock(return_value=5000)  # 5 seconds
        mock_audio_seg.from_file.return_value = mock_audio

        mock_processor = Mock()
        result = manager.process_new_audio(session_id, mock_processor)
        assert result == []

    @patch('src.web.streaming_transcriber.PYDUB_AVAILABLE', True)
    @patch('src.web.streaming_transcriber.AudioSegment')
    def test_process_new_audio_success(self, mock_audio_seg, manager):
        """Test successful audio processing with mocked Whisper."""
        session_id = manager.create_session('workspace-1')
        manager.add_audio_chunk(session_id, b'\x00' * 1000)

        # Mock audio with enough duration
        mock_audio = MagicMock()
        mock_audio.__len__ = Mock(return_value=15000)  # 15 seconds
        mock_audio.__getitem__ = Mock(return_value=mock_audio)
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio_seg.from_file.return_value = mock_audio

        # Mock processor
        mock_processor = Mock()
        mock_processor.is_model_loaded = True
        mock_processor.transcribe_with_segments.return_value = (
            [
                {'start': 0.0, 'end': 5.0, 'text': 'Shields up', 'confidence': 0.9},
                {'start': 5.0, 'end': 10.0, 'text': 'Red alert', 'confidence': 0.85},
            ],
            {'language': 'en', 'duration': 15.0}
        )

        result = manager.process_new_audio(session_id, mock_processor)
        assert len(result) >= 1

        # Verify timestamps are adjusted
        session = manager.get_session(session_id)
        assert session.last_transcribed_offset == 15.0

    @patch('src.web.streaming_transcriber.PYDUB_AVAILABLE', True)
    @patch('src.web.streaming_transcriber.AudioSegment')
    def test_process_new_audio_force_short_tail(self, mock_audio_seg, manager):
        """Test that force=True transcribes audio shorter than minimum threshold."""
        session_id = manager.create_session('workspace-1')
        session = manager.get_session(session_id)
        # Simulate that previous pass already transcribed up to 60s
        session.last_transcribed_offset = 60.0
        manager.add_audio_chunk(session_id, b'\x00' * 100)

        # Mock audio with only 5s of new audio (total 65s, offset 60s)
        mock_audio = MagicMock()
        mock_audio.__len__ = Mock(return_value=65000)  # 65 seconds total
        mock_audio.__getitem__ = Mock(return_value=mock_audio)
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio_seg.from_file.return_value = mock_audio

        mock_processor = Mock()
        mock_processor.is_model_loaded = True
        mock_processor.transcribe_with_segments.return_value = (
            [{'start': 0.0, 'end': 5.0, 'text': 'Final words', 'confidence': 0.9}],
            {'language': 'en', 'duration': 5.0}
        )

        # Without force, this should return empty (5s < 10s minimum)
        result = manager.process_new_audio(session_id, mock_processor, force=False)
        assert result == []

        # With force, it should transcribe the short tail
        result = manager.process_new_audio(session_id, mock_processor, force=True)
        assert len(result) == 1
        assert result[0]['text'] == 'Final words'
        # Verify timestamp adjustment includes the 60s offset
        assert result[0]['start'] == 60.0
        assert result[0]['end'] == 65.0

    def test_process_new_audio_not_found(self, manager):
        """Test processing audio for nonexistent session."""
        mock_processor = Mock()
        result = manager.process_new_audio('nonexistent', mock_processor)
        assert result == []

    @patch('src.web.streaming_transcriber.PYDUB_AVAILABLE', False)
    def test_process_new_audio_no_pydub(self, manager):
        """Test processing when pydub is not available."""
        session_id = manager.create_session('workspace-1')
        manager.add_audio_chunk(session_id, b'\x00' * 100)
        mock_processor = Mock()
        result = manager.process_new_audio(session_id, mock_processor)
        assert result == []

    def test_process_new_audio_empty_session(self, manager):
        """Test processing when session has no audio data."""
        session_id = manager.create_session('workspace-1')
        mock_processor = Mock()

        with patch('src.web.streaming_transcriber.PYDUB_AVAILABLE', True):
            result = manager.process_new_audio(session_id, mock_processor)
        assert result == []

    def test_segment_format_matches_pipeline(self, manager):
        """Test that finalized segment format matches analysis pipeline expectations."""
        session_id = manager.create_session('workspace-1')
        session = manager.get_session(session_id)

        # Add segments in the expected format
        segments = [
            {
                'start': 0.0,
                'end': 3.5,
                'text': 'Captain, shields are at 80 percent.',
                'confidence': 0.92,
            },
            {
                'start': 3.5,
                'end': 7.0,
                'text': 'Divert power to forward shields.',
                'confidence': 0.88,
            },
        ]
        session.add_segments(segments)
        session.total_audio_duration = 7.0

        result = manager.finalize_session(session_id)

        # Verify structure matches what analyze_audio expects
        for seg in result['segments']:
            assert 'start' in seg
            assert 'end' in seg
            assert 'text' in seg
            assert 'confidence' in seg
            assert isinstance(seg['start'], float)
            assert isinstance(seg['end'], float)
            assert isinstance(seg['text'], str)

        # Verify info structure
        assert 'duration' in result['info']
        assert 'language' in result['info']

    def test_double_finalize(self, manager):
        """Test that finalizing twice returns None the second time."""
        session_id = manager.create_session('workspace-1')
        result1 = manager.finalize_session(session_id)
        assert result1 is not None
        result2 = manager.finalize_session(session_id)
        assert result2 is None
