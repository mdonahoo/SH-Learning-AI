"""
Audio preprocessing for improved transcription and diarization quality.

Features:
- Peak and RMS normalization
- High-pass filtering (removes low-frequency rumble)
- Noise reduction using spectral gating
- Voice activity enhancement
"""

import numpy as np
import logging
import os
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from scipy.signal import butter, filtfilt, medfilt
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - some preprocessing features disabled")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.debug("noisereduce not available - using basic noise reduction")


def normalize_audio(
    audio: np.ndarray,
    target_level: float = -3.0,
    method: str = 'peak'
) -> np.ndarray:
    """
    Normalize audio to a target level.

    Args:
        audio: Input audio samples (float32, -1 to 1)
        target_level: Target level in dB (default -3 dB)
        method: 'peak' for peak normalization, 'rms' for RMS normalization

    Returns:
        Normalized audio
    """
    if len(audio) == 0:
        return audio

    # Convert target to linear
    target_linear = 10 ** (target_level / 20)

    if method == 'peak':
        # Peak normalization
        peak = np.abs(audio).max()
        if peak > 0:
            gain = target_linear / peak
            audio = audio * gain
            logger.debug(f"Peak normalized: gain={gain:.2f}")
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            gain = target_linear / rms
            # Limit gain to prevent clipping
            gain = min(gain, 1.0 / (np.abs(audio).max() + 1e-10))
            audio = audio * gain
            logger.debug(f"RMS normalized: gain={gain:.2f}")

    # Ensure no clipping
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_freq: float = 80.0,
    order: int = 5
) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency noise.

    Args:
        audio: Input audio samples
        sample_rate: Audio sample rate
        cutoff_freq: Cutoff frequency in Hz (default 80 Hz removes rumble)
        order: Filter order (default 5)

    Returns:
        Filtered audio
    """
    if not SCIPY_AVAILABLE:
        logger.debug("scipy not available, skipping high-pass filter")
        return audio

    if len(audio) < order * 3:
        return audio

    try:
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        if normalized_cutoff >= 1:
            logger.warning(f"Cutoff freq {cutoff_freq} Hz too high for sample rate {sample_rate}")
            return audio

        b, a = butter(order, normalized_cutoff, btype='high')
        filtered = filtfilt(b, a, audio)

        logger.debug(f"Applied high-pass filter: cutoff={cutoff_freq}Hz")
        return filtered.astype(np.float32)

    except Exception as e:
        logger.warning(f"High-pass filter failed: {e}")
        return audio


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    noise_reduce_strength: float = 0.5,
    use_advanced: bool = True
) -> np.ndarray:
    """
    Reduce background noise in audio.

    Args:
        audio: Input audio samples
        sample_rate: Audio sample rate
        noise_reduce_strength: Strength of noise reduction (0-1)
        use_advanced: Use noisereduce library if available

    Returns:
        Noise-reduced audio
    """
    if len(audio) == 0:
        return audio

    # Try advanced noise reduction first
    if use_advanced and NOISEREDUCE_AVAILABLE:
        try:
            # Use first 0.5 seconds as noise profile (or less if audio is short)
            noise_clip_length = min(int(0.5 * sample_rate), len(audio) // 4)

            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                prop_decrease=noise_reduce_strength,
                n_fft=2048,
                hop_length=512,
                stationary=False  # Non-stationary for better speech handling
            )
            logger.debug(f"Applied advanced noise reduction: strength={noise_reduce_strength}")
            return reduced.astype(np.float32)

        except Exception as e:
            logger.warning(f"Advanced noise reduction failed: {e}, falling back to basic")

    # Basic noise reduction using spectral gating (scipy-based)
    if SCIPY_AVAILABLE:
        try:
            return _basic_noise_reduction(audio, sample_rate, noise_reduce_strength)
        except Exception as e:
            logger.warning(f"Basic noise reduction failed: {e}")

    return audio


def _basic_noise_reduction(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.5
) -> np.ndarray:
    """
    Basic noise reduction using spectral subtraction.

    Args:
        audio: Input audio
        sample_rate: Sample rate
        strength: Reduction strength (0-1)

    Returns:
        Noise-reduced audio
    """
    from scipy.signal import stft, istft

    # Compute STFT
    nperseg = min(2048, len(audio) // 4)
    if nperseg < 256:
        return audio

    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg)

    # Estimate noise from quietest frames
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Use median of lowest 10% magnitudes as noise floor
    noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)

    # Spectral subtraction
    magnitude_cleaned = np.maximum(magnitude - strength * noise_floor, 0)

    # Reconstruct
    Zxx_cleaned = magnitude_cleaned * np.exp(1j * phase)
    _, audio_cleaned = istft(Zxx_cleaned, fs=sample_rate, nperseg=nperseg)

    # Match original length
    if len(audio_cleaned) > len(audio):
        audio_cleaned = audio_cleaned[:len(audio)]
    elif len(audio_cleaned) < len(audio):
        audio_cleaned = np.pad(audio_cleaned, (0, len(audio) - len(audio_cleaned)))

    logger.debug("Applied basic spectral subtraction noise reduction")
    return audio_cleaned.astype(np.float32)


def enhance_voice(
    audio: np.ndarray,
    sample_rate: int,
    low_freq: float = 100.0,
    high_freq: float = 8000.0
) -> np.ndarray:
    """
    Enhance voice frequencies by bandpass filtering.

    Args:
        audio: Input audio
        sample_rate: Sample rate
        low_freq: Low cutoff for voice band
        high_freq: High cutoff for voice band

    Returns:
        Voice-enhanced audio
    """
    if not SCIPY_AVAILABLE:
        return audio

    try:
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)

        if low >= high:
            return audio

        b, a = butter(4, [low, high], btype='band')
        enhanced = filtfilt(b, a, audio)

        logger.debug(f"Applied voice enhancement: {low_freq}-{high_freq}Hz")
        return enhanced.astype(np.float32)

    except Exception as e:
        logger.warning(f"Voice enhancement failed: {e}")
        return audio


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
    highpass: bool = True,
    noise_reduce: bool = True,
    noise_strength: float = 0.3
) -> np.ndarray:
    """
    Full preprocessing pipeline for audio.

    Args:
        audio: Input audio samples (float32)
        sample_rate: Audio sample rate
        normalize: Apply normalization
        highpass: Apply high-pass filter
        noise_reduce: Apply noise reduction
        noise_strength: Noise reduction strength (0-1)

    Returns:
        Preprocessed audio
    """
    logger.info(
        f"Preprocessing audio: normalize={normalize}, highpass={highpass}, "
        f"noise_reduce={noise_reduce}, strength={noise_strength}"
    )

    # Start with copy to avoid modifying original
    processed = audio.copy()

    # Step 1: High-pass filter to remove rumble
    if highpass:
        processed = highpass_filter(processed, sample_rate, cutoff_freq=80.0)

    # Step 2: Noise reduction
    if noise_reduce:
        processed = reduce_noise(
            processed, sample_rate,
            noise_reduce_strength=noise_strength,
            use_advanced=True
        )

    # Step 3: Normalize to consistent level
    if normalize:
        processed = normalize_audio(processed, target_level=-3.0, method='peak')

    return processed


def get_audio_stats(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Get statistics about audio quality.

    Args:
        audio: Audio samples
        sample_rate: Sample rate

    Returns:
        Dictionary with audio statistics
    """
    if len(audio) == 0:
        return {'duration': 0, 'peak': 0, 'rms': 0, 'snr_estimate': 0}

    duration = len(audio) / sample_rate
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Estimate SNR from quiet vs loud sections
    frame_size = int(0.02 * sample_rate)  # 20ms frames
    if len(audio) > frame_size * 10:
        frames = [
            audio[i:i+frame_size]
            for i in range(0, len(audio) - frame_size, frame_size)
        ]
        frame_energies = [np.sqrt(np.mean(f ** 2)) for f in frames]

        # Noise estimate: lowest 10% of frame energies
        noise_estimate = np.percentile(frame_energies, 10)
        # Signal estimate: highest 10% of frame energies
        signal_estimate = np.percentile(frame_energies, 90)

        if noise_estimate > 0:
            snr_estimate = 20 * np.log10(signal_estimate / noise_estimate)
        else:
            snr_estimate = 60.0  # Very clean audio
    else:
        snr_estimate = 0.0

    return {
        'duration': duration,
        'peak': peak,
        'rms': rms,
        'snr_estimate': snr_estimate
    }
