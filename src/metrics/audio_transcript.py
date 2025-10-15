#!/usr/bin/env python3
"""
Audio Transcript Service for Starship Horizons Learning AI
Handles audio recording, transcription, and speaker identification.
"""

import json
import queue
import threading
import wave
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AudioTranscriptService:
    """Service for recording and transcribing mission audio."""

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

    def _initialize_audio_components(self):
        """Initialize audio capture and transcription components."""
        try:
            # Import components
            from src.audio.capture import AudioCaptureManager
            from src.audio.whisper_transcriber import WhisperTranscriber
            from src.audio.speaker_diarization import SpeakerDiarizer, EngagementAnalyzer

            # Initialize speaker diarizer
            enable_diarization = os.getenv('ENABLE_SPEAKER_DIARIZATION', 'true').lower() == 'true'
            use_neural = os.getenv('USE_NEURAL_DIARIZATION', 'false').lower() == 'true'

            if enable_diarization:
                if use_neural:
                    try:
                        from src.audio.neural_diarization import NeuralSpeakerDiarizer
                        self._speaker_diarizer = NeuralSpeakerDiarizer()
                        logger.info("Neural speaker diarizer initialized")
                    except Exception as e:
                        logger.warning(f"Failed to load neural diarizer, falling back to simple: {e}")
                        self._speaker_diarizer = SpeakerDiarizer()
                        logger.info("Speaker diarizer initialized (fallback)")
                else:
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

    def start_recording(self) -> None:
        """Start audio recording."""
        with self._lock:
            if not self.is_recording:
                self.is_recording = True
                self.is_paused = False
                self.recording_start_time = datetime.now()
                self.buffer_start_time = datetime.now()
                self.current_buffer = []

    def stop_recording(self) -> float:
        """
        Stop audio recording.

        Returns:
            Total recording duration in seconds
        """
        with self._lock:
            if self.is_recording:
                self.is_recording = False
                self.recording_end_time = datetime.now()

                # Save any remaining buffer
                if self.current_buffer:
                    self._save_buffer()

                # Calculate duration
                duration = (self.recording_end_time - self.recording_start_time).total_seconds()
                return duration
            return 0

    def pause_recording(self) -> None:
        """Pause audio recording."""
        with self._lock:
            if self.is_recording and not self.is_paused:
                self.is_paused = True
                # Save current buffer
                if self.current_buffer:
                    self._save_buffer()

    def resume_recording(self) -> None:
        """Resume audio recording."""
        with self._lock:
            if self.is_recording and self.is_paused:
                self.is_paused = False
                self.buffer_start_time = datetime.now()
                self.current_buffer = []

    def add_audio_chunk(self, audio_data: np.ndarray) -> None:
        """
        Add audio chunk to current buffer.

        Args:
            audio_data: Audio samples as numpy array
        """
        with self._lock:
            if self.is_recording and not self.is_paused:
                self.current_buffer.append(audio_data)

                # Check if buffer is full
                buffer_samples = sum(len(chunk) for chunk in self.current_buffer)
                buffer_duration = buffer_samples / self.sample_rate

                if buffer_duration >= self.buffer_duration:
                    self._save_buffer()
                    self.buffer_start_time = datetime.now()
                    self.current_buffer = []

    def _save_buffer(self) -> None:
        """Save current buffer as an audio segment."""
        if not self.current_buffer:
            return

        # Concatenate all chunks
        audio_data = np.concatenate(self.current_buffer)

        segment = {
            "timestamp": self.buffer_start_time,
            "audio_data": audio_data,
            "duration": len(audio_data) / self.sample_rate
        }

        self.audio_segments.append(segment)

    def detect_voice_activity(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio contains voice activity.

        Args:
            audio_data: Audio samples
            threshold: Energy threshold for voice detection

        Returns:
            True if voice activity detected
        """
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(audio_data ** 2))
        return energy > threshold

    def identify_speaker(self, audio_data: np.ndarray) -> str:
        """
        Identify speaker from audio.

        Args:
            audio_data: Audio samples

        Returns:
            Speaker identifier
        """
        # Simplified speaker identification
        # In production, this would use speaker embeddings
        audio_hash = hash(audio_data.tobytes()) % 1000

        if audio_hash not in self._speaker_profiles:
            speaker_id = f"Speaker_{len(self._speaker_profiles) + 1}"
            self._speaker_profiles[audio_hash] = speaker_id

        return self._speaker_profiles[audio_hash]

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription result with text and confidence
        """
        try:
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
                logger.warning("Whisper transcriber not initialized")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "error": "Transcriber not available"
                }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def start_realtime_transcription(self) -> None:
        """Start realtime transcription thread."""
        with self._lock:
            if not self._transcription_active:
                self._transcription_active = True
                self._transcription_thread = threading.Thread(
                    target=self._transcription_worker,
                    daemon=True
                )
                self._transcription_thread.start()

    def stop_realtime_transcription(self) -> None:
        """Stop realtime transcription thread."""
        with self._lock:
            self._transcription_active = False

        if self._transcription_thread:
            self._transcription_queue.put(None)  # Sentinel
            self._transcription_thread.join(timeout=1)

    def _transcription_worker(self) -> None:
        """Worker thread for processing transcription queue."""
        while self._transcription_active:
            try:
                item = self._transcription_queue.get(timeout=1)
                if item is None:  # Sentinel
                    break

                # Extract audio and metadata
                audio_chunk = item.get('audio')
                timestamp = item.get('timestamp')

                if audio_chunk is None or timestamp is None:
                    continue

                # Identify speaker if diarizer available
                speaker_id = None
                if self._speaker_diarizer:
                    try:
                        speaker_id, confidence = self._speaker_diarizer.identify_speaker(audio_chunk)
                        logger.debug(f"Speaker identified: {speaker_id} (confidence: {confidence:.2f})")
                    except Exception as e:
                        logger.warning(f"Speaker identification failed: {e}")

                # Queue for transcription if transcriber available
                if self._whisper_transcriber:
                    try:
                        self._whisper_transcriber.queue_audio(
                            audio_chunk,
                            timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp,
                            speaker_id=speaker_id
                        )
                    except Exception as e:
                        logger.error(f"Failed to queue audio for transcription: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription worker error: {e}")

    def queue_for_transcription(self, audio_chunk: np.ndarray, timestamp: datetime) -> None:
        """
        Queue audio chunk for transcription.

        Args:
            audio_chunk: Audio data
            timestamp: Timestamp of audio
        """
        # Save raw audio if enabled
        save_raw = os.getenv('SAVE_RAW_AUDIO', 'false').lower() == 'true'
        if save_raw and self.storage_path:
            try:
                segment_id = f"seg_{len(self.transcripts):04d}"
                audio_path = self.save_audio_segment(audio_chunk, timestamp, segment_id)
                logger.debug(f"Saved audio segment: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to save audio segment: {e}")

        self._transcription_queue.put({
            "audio": audio_chunk,
            "timestamp": timestamp
        })

    def get_transcription_results(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get transcription results.

        Args:
            timeout: Timeout in seconds

        Returns:
            Transcription results or None
        """
        if not self._whisper_transcriber:
            return {
                "results": [],
                "status": "unavailable",
                "error": "Transcriber not initialized"
            }

        try:
            # Get pending results from transcriber
            results = self._whisper_transcriber.get_results()

            # Add to our transcript storage
            for result in results:
                self.add_transcript(
                    timestamp=datetime.fromtimestamp(result['timestamp']),
                    speaker=result.get('speaker_id', 'unknown'),
                    text=result['text'],
                    confidence=result['confidence']
                )

            return {
                "results": results,
                "status": "success",
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Failed to get transcription results: {e}")
            return {
                "results": [],
                "status": "error",
                "error": str(e)
            }

    def add_transcript(self, timestamp: datetime, speaker: str, text: str,
                      confidence: float, event_id: str = None) -> None:
        """
        Add a transcript entry.

        Args:
            timestamp: When the speech occurred
            speaker: Who spoke
            text: What was said
            confidence: Transcription confidence
            event_id: Optional associated event ID
        """
        transcript = {
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text,
            "confidence": confidence
        }

        if event_id:
            transcript["event_id"] = event_id

        with self._lock:
            self.transcripts.append(transcript)

    def get_all_transcripts(self) -> List[Dict[str, Any]]:
        """Get all transcripts."""
        with self._lock:
            return self.transcripts.copy()

    def get_transcript_at_time(self, timestamp: datetime, window: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get transcript at specific time.

        Args:
            timestamp: Target timestamp
            window: Time window in seconds

        Returns:
            Transcript entry or None
        """
        with self._lock:
            for transcript in self.transcripts:
                time_diff = abs((transcript["timestamp"] - timestamp).total_seconds())
                if time_diff <= window:
                    return transcript
        return None

    def export_transcript(self, filepath: Path) -> None:
        """
        Export transcripts to file.

        Args:
            filepath: Export file path
        """
        with self._lock:
            export_data = {
                "mission_id": self.mission_id,
                "recording_start": self.recording_start_time.isoformat() if self.recording_start_time else None,
                "recording_end": self.recording_end_time.isoformat() if self.recording_end_time else None,
                "transcripts": []
            }

            for transcript in self.transcripts:
                t_copy = transcript.copy()
                t_copy["timestamp"] = t_copy["timestamp"].isoformat()
                export_data["transcripts"].append(t_copy)

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

    def set_storage_path(self, path: str) -> None:
        """Set storage path for audio files."""
        self.storage_path = Path(path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_audio_segment(self, audio_data: np.ndarray, timestamp: datetime,
                          segment_id: str) -> Path:
        """
        Save audio segment to file.

        Args:
            audio_data: Audio samples
            timestamp: Segment timestamp
            segment_id: Segment identifier

        Returns:
            Path to saved file
        """
        if not self.storage_path:
            self.set_storage_path(f"/tmp/audio/{self.mission_id}")

        # Create filename
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.mission_id}_{segment_id}_{timestamp_str}.wav"
        filepath = self.storage_path / filename

        # Save as WAV
        with wave.open(str(filepath), 'wb') as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)

            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav.writeframes(audio_int16.tobytes())

        return filepath

    def calculate_audio_metrics(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate audio quality metrics.

        Args:
            audio_data: Audio samples

        Returns:
            Dictionary of audio metrics
        """
        # RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))

        # Peak amplitude
        peak_amplitude = np.max(np.abs(audio_data))

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zero_crossing_rate = zero_crossings / len(audio_data)

        # Simple SNR estimate (assuming noise floor)
        noise_floor = np.percentile(np.abs(audio_data), 10)
        signal_power = rms_energy ** 2
        noise_power = noise_floor ** 2

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')

        return {
            "rms_energy": float(rms_energy),
            "peak_amplitude": float(peak_amplitude),
            "zero_crossing_rate": float(zero_crossing_rate),
            "signal_to_noise_ratio": float(snr) if snr != float('inf') else 100.0
        }

    def get_total_duration(self) -> float:
        """Get total recording duration in seconds."""
        with self._lock:
            total = sum(seg["duration"] for seg in self.audio_segments)

            # Add current buffer if recording
            if self.is_recording and self.current_buffer:
                buffer_samples = sum(len(chunk) for chunk in self.current_buffer)
                total += buffer_samples / self.sample_rate

            return total

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary statistics."""
        with self._lock:
            speaker_counts = defaultdict(int)

            for transcript in self.transcripts:
                speaker_counts[transcript["speaker"]] += 1

            return {
                "total_utterances": len(self.transcripts),
                "unique_speakers": len(speaker_counts),
                "speakers": dict(speaker_counts)
            }

    def start_audio_capture(self) -> bool:
        """
        Start real-time audio capture.

        Returns:
            True if capture started successfully
        """
        try:
            # Set recording start time if not already set
            if not self.recording_start_time:
                self.recording_start_time = datetime.now()

            # Initialize capture manager if needed
            if not self._capture_manager:
                from src.audio.capture import AudioCaptureManager
                self._capture_manager = AudioCaptureManager()

            # Set up segment callback to process captured audio
            def on_audio_segment(audio_data: np.ndarray, start_time: float, end_time: float):
                """Callback for audio segments from capture manager."""
                timestamp = datetime.fromtimestamp(self.recording_start_time.timestamp() + start_time)

                # Queue for transcription
                if self.auto_transcribe:
                    self.queue_for_transcription(audio_data, timestamp)

            self._capture_manager.set_segment_callback(on_audio_segment)

            # Start transcription workers if enabled
            if self._whisper_transcriber and self.auto_transcribe:
                self._whisper_transcriber.start_workers()

            # Start capture
            success = self._capture_manager.start_capture()
            if success:
                logger.info("✓ Audio capture started")
            return success

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False

    def stop_audio_capture(self) -> None:
        """Stop real-time audio capture and cleanup."""
        try:
            # Stop capture
            if self._capture_manager:
                self._capture_manager.stop_capture()
                logger.info("✓ Audio capture stopped")

            # Stop transcription workers
            if self._whisper_transcriber:
                self._whisper_transcriber.stop_workers()
                logger.info("✓ Transcription workers stopped")

        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")

    def get_engagement_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get engagement analytics summary.

        Returns:
            Engagement metrics dictionary or None if not available
        """
        if not self._engagement_analyzer:
            logger.warning("Engagement analyzer not initialized")
            return None

        try:
            # Use the mission communication summary
            return self._engagement_analyzer.get_mission_communication_summary()

        except Exception as e:
            logger.error(f"Failed to calculate engagement metrics: {e}")
            return None