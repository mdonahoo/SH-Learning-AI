"""
Metrics and evaluation modules for Starship Horizons Learning AI.

This package provides:
- Event recording and tracking
- Audio transcript processing
- Mission summarization and analysis
- Learning evaluation frameworks
- Report validation
"""

from src.metrics.event_recorder import EventRecorder
from src.metrics.audio_transcript import AudioTranscriptService
from src.metrics.mission_summarizer import MissionSummarizer
from src.metrics.learning_evaluator import LearningEvaluator
from src.metrics.report_validator import (
    ReportValidator,
    ValidationIssue,
    ValidationSeverity,
    validate_report,
)

__all__ = [
    'EventRecorder',
    'AudioTranscriptService',
    'MissionSummarizer',
    'LearningEvaluator',
    'ReportValidator',
    'ValidationIssue',
    'ValidationSeverity',
    'validate_report',
]
