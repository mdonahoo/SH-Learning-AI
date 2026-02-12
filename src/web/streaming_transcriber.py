"""
Streaming transcription module for real-time audio processing.

Manages WebSocket-based streaming sessions where audio chunks arrive
during recording and are incrementally transcribed, so that most
transcription work is complete by the time the user stops recording.
"""

import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not installed. Streaming transcription unavailable.")

# Hallucination filter
try:
    from src.audio.whisper_transcriber import is_hallucination
    HALLUCINATION_FILTER_AVAILABLE = True
except ImportError:
    HALLUCINATION_FILTER_AVAILABLE = False
    is_hallucination = None

# Domain post-correction
try:
    from src.audio.domain_postcorrector import DomainPostCorrector
    POSTCORRECTOR_AVAILABLE = True
except ImportError:
    POSTCORRECTOR_AVAILABLE = False
    DomainPostCorrector = None

# Configuration
STREAMING_ENABLED = os.getenv(
    'STREAMING_TRANSCRIBE_ENABLED', 'true'
).lower() == 'true'
STREAMING_MIN_CHUNK_SECONDS = int(os.getenv(
    'STREAMING_MIN_CHUNK_SECONDS', '10'
))
STREAMING_SESSION_TIMEOUT = int(os.getenv(
    'STREAMING_SESSION_TIMEOUT', '3600'
))


class StreamingTranscriptionSession:
    """
    Per-recording session state for streaming transcription.

    Accumulates raw audio chunks from the browser, tracks which
    portions have already been transcribed, and stores completed
    transcript segments.

    Attributes:
        session_id: Unique session identifier.
        workspace_id: Associated workspace identifier.
        audio_chunks: Accumulated raw WebM chunks.
        segments: Completed transcript segments.
        last_transcribed_offset: Seconds of audio already transcribed.
        created_at: Timestamp when the session was created.
    """

    def __init__(self, session_id: str, workspace_id: str):
        """
        Initialize a streaming transcription session.

        Args:
            session_id: Unique session identifier.
            workspace_id: Associated workspace identifier.
        """
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.audio_chunks: List[bytes] = []
        self.segments: List[Dict[str, Any]] = []
        self.last_transcribed_offset: float = 0.0
        self.created_at: float = time.time()
        self.last_activity: float = time.time()
        self.total_audio_duration: float = 0.0
        self.is_finalized: bool = False
        self._lock = threading.Lock()
        self._transcribing = False

    def add_chunk(self, chunk: bytes) -> None:
        """
        Append a binary audio chunk.

        Args:
            chunk: Raw audio bytes (WebM fragment).
        """
        with self._lock:
            self.audio_chunks.append(chunk)
            self.last_activity = time.time()

    def get_accumulated_audio(self) -> bytes:
        """
        Get all accumulated audio data.

        Returns:
            Concatenated audio bytes.
        """
        with self._lock:
            return b''.join(self.audio_chunks)

    def add_segments(self, new_segments: List[Dict[str, Any]]) -> None:
        """
        Add new transcript segments.

        Args:
            new_segments: List of segment dicts with start, end, text keys.
        """
        with self._lock:
            self.segments.extend(new_segments)

    def get_segments(self) -> List[Dict[str, Any]]:
        """
        Get all accumulated transcript segments.

        Returns:
            Copy of the segments list.
        """
        with self._lock:
            return list(self.segments)

    def is_expired(self, timeout: int = STREAMING_SESSION_TIMEOUT) -> bool:
        """
        Check if the session has expired.

        Args:
            timeout: Session timeout in seconds.

        Returns:
            True if the session has exceeded the timeout.
        """
        return (time.time() - self.last_activity) > timeout


class StreamingTranscriptionManager:
    """
    Manages all active streaming transcription sessions.

    Provides lifecycle management (create, process, finalize, cleanup)
    and delegates to the shared AudioProcessor Whisper model for
    transcription.

    Attributes:
        sessions: Active session mapping.
    """

    def __init__(self) -> None:
        """Initialize the streaming transcription manager."""
        self.sessions: Dict[str, StreamingTranscriptionSession] = {}
        self._lock = threading.Lock()
        self._postcorrector: Optional[Any] = None
        if POSTCORRECTOR_AVAILABLE and DomainPostCorrector:
            try:
                self._postcorrector = DomainPostCorrector()
            except Exception as e:
                logger.warning(f"Failed to initialize domain postcorrector: {e}")
        logger.info("StreamingTranscriptionManager initialized")

    def create_session(self, workspace_id: str) -> str:
        """
        Create a new streaming transcription session.

        Args:
            workspace_id: Associated workspace identifier.

        Returns:
            Unique session ID.
        """
        session_id = str(uuid.uuid4())
        session = StreamingTranscriptionSession(session_id, workspace_id)
        with self._lock:
            self.sessions[session_id] = session
        logger.info(
            f"Created streaming session {session_id} "
            f"for workspace {workspace_id}"
        )
        return session_id

    def get_session(
        self, session_id: str
    ) -> Optional[StreamingTranscriptionSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            The session if found, None otherwise.
        """
        with self._lock:
            return self.sessions.get(session_id)

    def add_audio_chunk(self, session_id: str, chunk: bytes) -> bool:
        """
        Append binary audio data to a session.

        Args:
            session_id: Session identifier.
            chunk: Raw audio bytes.

        Returns:
            True if the chunk was added, False if session not found.
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found for audio chunk")
            return False
        session.add_chunk(chunk)
        return True

    def process_new_audio(
        self,
        session_id: str,
        processor: Any,
        force: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Transcribe new audio since the last run.

        Decodes all accumulated WebM data, extracts audio beyond the
        last transcribed offset, converts to WAV, and runs Whisper.

        Args:
            session_id: Session identifier.
            processor: AudioProcessor instance with loaded Whisper model.
            force: If True, skip the minimum chunk duration check.
                Use this for the final transcription pass so that
                trailing audio shorter than the minimum is not lost.

        Returns:
            List of new transcript segment dicts, or empty list.
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found for processing")
            return []

        if session._transcribing:
            logger.debug(f"Session {session_id} already transcribing, skip")
            return []

        if not PYDUB_AVAILABLE:
            logger.error("pydub required for streaming transcription")
            return []

        session._transcribing = True
        tmp_webm = None
        tmp_wav = None
        try:
            # Write accumulated audio to temp WebM file
            audio_data = session.get_accumulated_audio()
            if not audio_data:
                return []

            tmp_webm = tempfile.NamedTemporaryFile(
                suffix='.webm', delete=False
            )
            tmp_webm.write(audio_data)
            tmp_webm.close()

            # Decode to get total duration
            try:
                full_audio = AudioSegment.from_file(tmp_webm.name, format='webm')
            except Exception as e:
                logger.warning(f"Failed to decode WebM audio: {e}")
                return []

            total_duration = len(full_audio) / 1000.0
            session.total_audio_duration = total_duration
            new_audio_seconds = total_duration - session.last_transcribed_offset

            # Skip if no meaningful new audio at all
            if new_audio_seconds < 0.5:
                return []

            # Only transcribe if enough new audio accumulated
            # (skip this check on force=True for the final pass)
            if not force and new_audio_seconds < STREAMING_MIN_CHUNK_SECONDS:
                logger.debug(
                    f"Session {session_id}: only {new_audio_seconds:.1f}s "
                    f"new audio, need {STREAMING_MIN_CHUNK_SECONDS}s"
                )
                return []

            # Extract only the new portion
            start_ms = int(session.last_transcribed_offset * 1000)
            new_portion = full_audio[start_ms:]

            # Convert to mono 16kHz WAV for Whisper
            new_portion = new_portion.set_channels(1).set_frame_rate(16000)
            tmp_wav = tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False
            )
            new_portion.export(tmp_wav.name, format='wav')
            tmp_wav.close()

            # Ensure Whisper model is loaded
            if not processor.is_model_loaded:
                logger.info("Loading Whisper model for streaming transcription")
                processor.load_model()

            if not processor.is_model_loaded:
                logger.error("Whisper model not available for streaming")
                return []

            # Transcribe the new portion
            segments, info = processor.transcribe_with_segments(tmp_wav.name)

            # Adjust timestamps by the offset
            offset = session.last_transcribed_offset
            new_segments = []
            for seg in segments:
                text = seg.get('text', '').strip()
                if not text:
                    continue

                # Filter hallucinations
                if HALLUCINATION_FILTER_AVAILABLE and is_hallucination(text):
                    continue

                adjusted_seg = {
                    'start': round(seg['start'] + offset, 3),
                    'end': round(seg['end'] + offset, 3),
                    'text': text,
                    'confidence': seg.get('confidence', 0.0),
                }
                if 'words' in seg:
                    adjusted_seg['words'] = seg['words']
                new_segments.append(adjusted_seg)

            # Apply domain corrections if available
            if self._postcorrector and new_segments:
                try:
                    new_segments, _correction_stats = (
                        self._postcorrector.correct_segments(new_segments)
                    )
                except Exception as e:
                    logger.warning(f"Domain post-correction failed: {e}")

            # Update session state
            session.last_transcribed_offset = total_duration
            session.add_segments(new_segments)

            logger.info(
                f"Session {session_id}: transcribed {new_audio_seconds:.1f}s "
                f"new audio -> {len(new_segments)} segments "
                f"(total offset now {total_duration:.1f}s)"
            )
            return new_segments

        except Exception as e:
            logger.error(
                f"Streaming transcription error for session {session_id}: {e}",
                exc_info=True
            )
            return []
        finally:
            session._transcribing = False
            # Clean up temp files
            for tmp in (tmp_webm, tmp_wav):
                if tmp is not None:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass

    def finalize_session(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize a session and return accumulated results.

        Marks the session as finalized and returns all segments plus
        metadata. The session is removed from active tracking.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with 'segments', 'info', and metadata, or None if not found.
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found for finalization")
            return None

        session.is_finalized = True
        segments = session.get_segments()

        result = {
            'segments': segments,
            'info': {
                'language': 'en',
                'duration': session.total_audio_duration,
                'streaming': True,
                'segment_count': len(segments),
            },
            'session_id': session_id,
            'workspace_id': session.workspace_id,
        }

        # Remove from active sessions
        with self._lock:
            self.sessions.pop(session_id, None)

        logger.info(
            f"Finalized session {session_id}: "
            f"{len(segments)} segments, "
            f"{session.total_audio_duration:.1f}s duration"
        )
        return result

    def cleanup_expired(
        self, timeout: int = STREAMING_SESSION_TIMEOUT
    ) -> int:
        """
        Remove expired sessions.

        Args:
            timeout: Session timeout in seconds.

        Returns:
            Number of sessions cleaned up.
        """
        expired = []
        with self._lock:
            for sid, session in self.sessions.items():
                if session.is_expired(timeout):
                    expired.append(sid)
            for sid in expired:
                del self.sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired streaming sessions")
        return len(expired)

    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self.sessions)
