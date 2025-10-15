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
                    chunk_duration = len(audio_float) / self.sample_rate
                    self._segment_callback(
                        audio_float,
                        current_time,
                        current_time + chunk_duration
                    )
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
