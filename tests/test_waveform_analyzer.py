"""
Tests for the waveform amplitude envelope extractor.

Tests cover:
- Correct number of output samples for duration
- Amplitude normalized to 0-1
- Mono and stereo WAV handling
- Silent audio produces near-zero amplitude
- Peak detection accuracy
- Missing/invalid file handling
- Edge cases (empty, very short files)
"""

import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from src.audio.waveform_analyzer import WaveformAnalyzer


@pytest.fixture
def analyzer():
    """Create a default WaveformAnalyzer."""
    return WaveformAnalyzer(samples_per_second=10)


def _create_wav(
    path: str,
    duration: float = 1.0,
    sample_rate: int = 16000,
    num_channels: int = 1,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    sample_width: int = 2,
) -> str:
    """
    Helper to create a WAV file with a sine wave.

    Args:
        path: Output file path.
        duration: Duration in seconds.
        sample_rate: Samples per second.
        num_channels: 1 for mono, 2 for stereo.
        frequency: Sine wave frequency in Hz.
        amplitude: Sine wave amplitude (0-1).
        sample_width: Bytes per sample (1, 2, or 4).

    Returns:
        The output path.
    """
    num_frames = int(sample_rate * duration)
    t = np.linspace(0, duration, num_frames, endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)

        if sample_width == 2:
            int_signal = (signal * 32767).astype(np.int16)
        elif sample_width == 1:
            int_signal = ((signal + 1.0) * 127.5).astype(np.uint8)
        elif sample_width == 4:
            int_signal = (signal * 2147483647).astype(np.int32)
        else:
            raise ValueError(f"Unsupported sample_width: {sample_width}")

        if num_channels == 2:
            # Interleave same signal on both channels
            stereo = np.column_stack([int_signal, int_signal]).flatten()
            wf.writeframes(stereo.tobytes())
        else:
            wf.writeframes(int_signal.tobytes())

    return path


def _create_silent_wav(path: str, duration: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a silent WAV file."""
    return _create_wav(path, duration=duration, sample_rate=sample_rate, amplitude=0.0)


class TestWaveformAnalyzerInit:
    """Tests for WaveformAnalyzer initialization."""

    def test_default_initialization(self):
        """Test default initialization uses env or fallback."""
        analyzer = WaveformAnalyzer()
        assert analyzer.samples_per_second >= 1

    def test_custom_samples_per_second(self):
        """Test custom samples_per_second."""
        analyzer = WaveformAnalyzer(samples_per_second=20)
        assert analyzer.samples_per_second == 20

    def test_invalid_samples_per_second(self):
        """Test that zero or negative samples_per_second raises."""
        with pytest.raises(ValueError, match="samples_per_second must be >= 1"):
            WaveformAnalyzer(samples_per_second=0)

        with pytest.raises(ValueError, match="samples_per_second must be >= 1"):
            WaveformAnalyzer(samples_per_second=-5)


class TestExtractEnvelope:
    """Tests for the extract_envelope method."""

    def test_correct_sample_count(self, analyzer):
        """Test that output has approximately correct number of samples."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=5.0, sample_rate=16000)
            result = analyzer.extract_envelope(path)

            # 5 seconds at 10 samples/sec = ~50 samples
            expected = 5.0 * analyzer.samples_per_second
            assert abs(len(result['amplitude']) - expected) <= 2
            assert result['sample_rate'] == 10
        finally:
            os.unlink(path)

    def test_amplitude_normalized_0_1(self, analyzer):
        """Test that all amplitude values are between 0 and 1."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=2.0, amplitude=0.8)
            result = analyzer.extract_envelope(path)

            for val in result['amplitude']:
                assert 0.0 <= val <= 1.0, f"Amplitude {val} out of range"
        finally:
            os.unlink(path)

    def test_peak_amplitude_is_one_when_normalized(self, analyzer):
        """Test peak amplitude in normalized output is 1.0."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=2.0, amplitude=0.5)
            result = analyzer.extract_envelope(path)

            # After normalization, the peak should be 1.0
            assert max(result['amplitude']) == pytest.approx(1.0, abs=0.01)
        finally:
            os.unlink(path)

    def test_silent_audio_near_zero(self, analyzer):
        """Test that silent audio produces near-zero amplitude."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_silent_wav(path, duration=2.0)
            result = analyzer.extract_envelope(path)

            assert result['peak_amplitude'] < 0.001
            assert result['average_amplitude'] < 0.001
        finally:
            os.unlink(path)

    def test_stereo_audio(self, analyzer):
        """Test that stereo audio is handled (downmixed to mono)."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=2.0, num_channels=2, amplitude=0.5)
            result = analyzer.extract_envelope(path)

            assert len(result['amplitude']) > 0
            assert result['duration_seconds'] == pytest.approx(2.0, abs=0.1)
        finally:
            os.unlink(path)

    def test_duration_reported_correctly(self, analyzer):
        """Test that duration matches input file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=3.5, sample_rate=16000)
            result = analyzer.extract_envelope(path)

            assert result['duration_seconds'] == pytest.approx(3.5, abs=0.01)
        finally:
            os.unlink(path)

    def test_8bit_audio(self, analyzer):
        """Test 8-bit WAV files are handled."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=1.0, sample_width=1, amplitude=0.5)
            result = analyzer.extract_envelope(path)

            assert len(result['amplitude']) > 0
            assert result['peak_amplitude'] > 0
        finally:
            os.unlink(path)

    def test_32bit_audio(self, analyzer):
        """Test 32-bit WAV files are handled."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=1.0, sample_width=4, amplitude=0.5)
            result = analyzer.extract_envelope(path)

            assert len(result['amplitude']) > 0
            assert result['peak_amplitude'] > 0
        finally:
            os.unlink(path)

    def test_high_resolution_output(self):
        """Test higher samples_per_second produces more data points."""
        high_res = WaveformAnalyzer(samples_per_second=50)
        low_res = WaveformAnalyzer(samples_per_second=5)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=2.0, sample_rate=16000)
            high_result = high_res.extract_envelope(path)
            low_result = low_res.extract_envelope(path)

            assert len(high_result['amplitude']) > len(low_result['amplitude'])
        finally:
            os.unlink(path)


class TestErrorHandling:
    """Tests for error conditions."""

    def test_missing_file(self, analyzer):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            analyzer.extract_envelope('/nonexistent/path/audio.wav')

    def test_invalid_file(self, analyzer):
        """Test ValueError for non-WAV file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, mode='w') as f:
            f.write("not a wav file")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Failed to read WAV"):
                analyzer.extract_envelope(path)
        finally:
            os.unlink(path)

    def test_output_dict_keys(self, analyzer):
        """Test that output contains all expected keys."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name
        try:
            _create_wav(path, duration=1.0)
            result = analyzer.extract_envelope(path)

            expected_keys = {
                'sample_rate', 'duration_seconds', 'amplitude',
                'peak_amplitude', 'average_amplitude'
            }
            assert set(result.keys()) == expected_keys
        finally:
            os.unlink(path)
