"""
Waveform amplitude envelope extractor.

Extracts RMS amplitude data from WAV files, producing a downsampled
amplitude envelope suitable for frontend waveform visualization.
Uses numpy for efficient windowed RMS computation.
"""

import logging
import os
import wave
from typing import Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default samples per second for the output envelope
DEFAULT_SAMPLES_PER_SECOND = int(os.getenv('WAVEFORM_SAMPLES_PER_SECOND', '10'))


class WaveformAnalyzer:
    """
    Extracts RMS amplitude envelope from WAV audio files.

    Produces a normalized amplitude array downsampled to a configurable
    rate (default 10 samples/sec) for lightweight frontend rendering.

    Attributes:
        samples_per_second: Output envelope resolution.
    """

    def __init__(self, samples_per_second: int = DEFAULT_SAMPLES_PER_SECOND):
        """
        Initialize the waveform analyzer.

        Args:
            samples_per_second: Number of amplitude samples per second
                in the output envelope. Higher values give more detail
                but larger payloads.
        """
        if samples_per_second < 1:
            raise ValueError("samples_per_second must be >= 1")
        self.samples_per_second = samples_per_second

    def extract_envelope(self, wav_path: str) -> Dict[str, Any]:
        """
        Extract RMS amplitude envelope from a WAV file.

        Reads the WAV file, converts to mono float samples, computes
        windowed RMS energy, and normalizes to 0-1.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            Dictionary containing:
                - sample_rate: Output samples per second
                - duration_seconds: Audio duration
                - amplitude: List of normalized 0-1 RMS values
                - peak_amplitude: Maximum amplitude value
                - average_amplitude: Mean amplitude value

        Raises:
            FileNotFoundError: If wav_path does not exist.
            ValueError: If file cannot be read as WAV.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        try:
            samples, sample_rate, duration, num_channels = self._load_wav(wav_path)
        except Exception as e:
            raise ValueError(f"Failed to read WAV file: {e}") from e

        if len(samples) == 0:
            logger.warning("WAV file contains no audio data")
            return {
                'sample_rate': self.samples_per_second,
                'duration_seconds': duration,
                'amplitude': [],
                'peak_amplitude': 0.0,
                'average_amplitude': 0.0,
            }

        # Compute RMS envelope
        window_size = max(1, sample_rate // self.samples_per_second)
        rms_values = self._compute_rms(samples, window_size)

        # Normalize to 0-1
        peak = float(np.max(rms_values)) if len(rms_values) > 0 else 0.0
        if peak > 0:
            normalized = (rms_values / peak).tolist()
        else:
            normalized = rms_values.tolist()

        avg_amplitude = float(np.mean(normalized)) if len(normalized) > 0 else 0.0

        logger.info(
            f"Waveform extracted: {len(normalized)} samples, "
            f"duration={duration:.1f}s, peak={peak:.4f}"
        )

        return {
            'sample_rate': self.samples_per_second,
            'duration_seconds': round(duration, 3),
            'amplitude': normalized,
            'peak_amplitude': round(peak, 6),
            'average_amplitude': round(avg_amplitude, 6),
        }

    def _load_wav(self, wav_path: str) -> tuple:
        """
        Load WAV file and return mono float samples.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            Tuple of (samples_array, sample_rate, duration, num_channels).
        """
        with wave.open(wav_path, 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            raw_data = wf.readframes(num_frames)

        duration = num_frames / sample_rate if sample_rate > 0 else 0.0

        # Convert raw bytes to numpy array based on sample width
        if sample_width == 1:
            # 8-bit unsigned
            samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0
            samples /= 128.0
        elif sample_width == 2:
            # 16-bit signed (most common)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            samples /= 32768.0
        elif sample_width == 3:
            # 24-bit signed - need manual conversion
            samples = self._read_24bit(raw_data)
        elif sample_width == 4:
            # 32-bit signed
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32)
            samples /= 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        # Convert stereo/multi-channel to mono by averaging channels
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels).mean(axis=1)

        return samples, sample_rate, duration, num_channels

    def _read_24bit(self, raw_data: bytes) -> np.ndarray:
        """
        Convert 24-bit audio data to float32 numpy array.

        Args:
            raw_data: Raw 24-bit PCM bytes.

        Returns:
            Normalized float32 numpy array.
        """
        # Pad each 3-byte sample to 4 bytes for int32 conversion
        num_samples = len(raw_data) // 3
        samples = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            # Little-endian 24-bit to signed int
            val = int.from_bytes(raw_data[i * 3:(i + 1) * 3], byteorder='little', signed=True)
            # Sign-extend from 24-bit
            if val >= 0x800000:
                val -= 0x1000000
            samples[i] = val / 8388608.0
        return samples

    def _compute_rms(self, samples: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute RMS energy per window.

        Args:
            samples: Audio samples as float32 array.
            window_size: Number of samples per window.

        Returns:
            Array of RMS values, one per window.
        """
        # Trim to complete windows
        num_windows = len(samples) // window_size
        if num_windows == 0:
            # File shorter than one window — return single RMS of entire signal
            return np.array([np.sqrt(np.mean(samples ** 2))], dtype=np.float32)

        trimmed = samples[:num_windows * window_size]
        windowed = trimmed.reshape(num_windows, window_size)
        rms = np.sqrt(np.mean(windowed ** 2, axis=1))
        return rms
