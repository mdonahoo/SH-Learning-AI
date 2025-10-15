"""
Audio processing module for Starship Horizons Learning AI.

This module provides audio input/output capabilities including:
- Microphone recording
- Speaker output
- Audio configuration
- WSL2 compatibility
- Real-time audio transcription
- Speaker diarization
- Engagement analytics
"""

from .config import AudioConfig

# Import new audio transcription components
try:
    from .speaker_diarization import (
        SpeakerDiarizer,
        SimpleVAD,
        SpeakerSegment,
        EngagementAnalyzer
    )
    from .capture import AudioCaptureManager
    from .whisper_transcriber import WhisperTranscriber

    __all__ = [
        'AudioConfig',
        'SpeakerDiarizer',
        'SimpleVAD',
        'SpeakerSegment',
        'EngagementAnalyzer',
        'AudioCaptureManager',
        'WhisperTranscriber'
    ]
except ImportError:
    # Transcription components not available
    __all__ = ['AudioConfig']