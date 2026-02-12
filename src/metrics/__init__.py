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
- 7 Habits of Highly Effective People analysis
- Comprehensive training recommendations for educational contexts
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

# Aggregate role inference with diarization confidence
try:
    from src.metrics.aggregate_role_inference import (
        AggregateRoleInferenceEngine,
        AggregateRoleAnalysis,
        is_aggregate_inference_available,
    )
except ImportError:
    AggregateRoleInferenceEngine = None
    AggregateRoleAnalysis = None
    is_aggregate_inference_available = None
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
from src.metrics.seven_habits import (
    SevenHabitsAnalyzer,
    SevenHabit,
    HabitAssessment,
    HabitIndicators,
)
from src.metrics.training_recommendations import (
    TrainingRecommendationEngine,
    TrainingRecommendation,
    DrillActivity,
    RecommendationPriority,
    SkillCategory,
)
from src.metrics.performance_tracker import PerformanceTracker, DependencyCall
from src.metrics.live_metrics import LiveMetricsComputer

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
    # Aggregate role inference (two-pass architecture)
    'AggregateRoleInferenceEngine',
    'AggregateRoleAnalysis',
    'is_aggregate_inference_available',
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
    # 7 Habits analysis
    'SevenHabitsAnalyzer',
    'SevenHabit',
    'HabitAssessment',
    'HabitIndicators',
    # Training recommendations
    'TrainingRecommendationEngine',
    'TrainingRecommendation',
    'DrillActivity',
    'RecommendationPriority',
    'SkillCategory',
    # Performance tracking
    'PerformanceTracker',
    'DependencyCall',
    # Live metrics
    'LiveMetricsComputer',
]
