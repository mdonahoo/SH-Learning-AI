"""
Web server module for audio analysis API.

Provides a FastAPI-based web interface for audio transcription
and analysis using the existing Starship Horizons audio modules.
"""

from src.web.server import app, create_app
from src.web.audio_processor import AudioProcessor
from src.web.models import (
    TranscriptionSegment,
    SpeakerInfo,
    AnalysisResult,
    TranscriptionResult,
)

__all__ = [
    "app",
    "create_app",
    "AudioProcessor",
    "TranscriptionSegment",
    "SpeakerInfo",
    "AnalysisResult",
    "TranscriptionResult",
]
