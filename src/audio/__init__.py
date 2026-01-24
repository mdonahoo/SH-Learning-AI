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
- Two-pass batch diarization for consistent speaker IDs
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

# Import two-pass batch diarization components
try:
    from .batch_diarizer import (
        BatchSpeakerDiarizer,
        DiarizationResult,
        SpeakerCluster,
        is_batch_diarizer_available
    )
    __all__.extend([
        'BatchSpeakerDiarizer',
        'DiarizationResult',
        'SpeakerCluster',
        'is_batch_diarizer_available'
    ])
except ImportError:
    # Batch diarization not available
    pass

# Import CPU diarization components
try:
    from .cpu_diarization import (
        CPUSpeakerDiarizer,
        CPUSpeakerProfile
    )
    __all__.extend([
        'CPUSpeakerDiarizer',
        'CPUSpeakerProfile'
    ])
except ImportError:
    # CPU diarization not available
    pass

# Import transcript post-processor
try:
    from .transcript_postprocessor import (
        TranscriptPostProcessor,
        merge_adjacent_fragments
    )
    __all__.extend([
        'TranscriptPostProcessor',
        'merge_adjacent_fragments'
    ])
except ImportError:
    # Post-processor not available
    pass