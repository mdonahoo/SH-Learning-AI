"""
Metrics and evaluation modules for Starship Horizons Learning AI.

This package provides:
- Event recording and tracking
- Audio transcript processing
- Mission summarization and analysis
- Learning evaluation frameworks
- Report validation
- Role inference and analysis
- Confidence distribution analysis
- Mission phase detection
- Quality verification
- Speaker scorecards
- Communication quality analysis
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
from src.metrics.role_inference import (
    RoleInferenceEngine,
    BridgeRole,
    RolePatterns,
    SpeakerRoleAnalysis,
)
from src.metrics.confidence_analyzer import (
    ConfidenceAnalyzer,
    ConfidenceRange,
    CONFIDENCE_RANGES,
)
from src.metrics.phase_analyzer import (
    MissionPhaseAnalyzer,
    PhaseDefinition,
    PhaseAnalysis,
    PHASE_DEFINITIONS,
)
from src.metrics.quality_verifier import (
    QualityVerifier,
    VerificationCheck,
)
from src.metrics.speaker_scorecard import (
    SpeakerScorecardGenerator,
    SpeakerScorecard,
    SpeakerScore,
    SCORE_METRICS,
)
from src.metrics.communication_quality import (
    CommunicationQualityAnalyzer,
    CommunicationPattern,
    CommunicationAssessment,
    EFFECTIVE_PATTERNS,
    IMPROVEMENT_PATTERNS,
)
from src.metrics.enhanced_report_builder import EnhancedReportBuilder

__all__ = [
    # Core modules
    'EventRecorder',
    'AudioTranscriptService',
    'MissionSummarizer',
    'LearningEvaluator',
    'ReportValidator',
    'ValidationIssue',
    'ValidationSeverity',
    'validate_report',
    # Role inference
    'RoleInferenceEngine',
    'BridgeRole',
    'RolePatterns',
    'SpeakerRoleAnalysis',
    # Confidence analysis
    'ConfidenceAnalyzer',
    'ConfidenceRange',
    'CONFIDENCE_RANGES',
    # Phase analysis
    'MissionPhaseAnalyzer',
    'PhaseDefinition',
    'PhaseAnalysis',
    'PHASE_DEFINITIONS',
    # Quality verification
    'QualityVerifier',
    'VerificationCheck',
    # Speaker scorecards
    'SpeakerScorecardGenerator',
    'SpeakerScorecard',
    'SpeakerScore',
    'SCORE_METRICS',
    # Communication quality
    'CommunicationQualityAnalyzer',
    'CommunicationPattern',
    'CommunicationAssessment',
    'EFFECTIVE_PATTERNS',
    'IMPROVEMENT_PATTERNS',
    # Enhanced report builder
    'EnhancedReportBuilder',
]
