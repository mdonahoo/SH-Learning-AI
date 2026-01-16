"""
Pydantic models for the audio analysis web API.

Defines request/response schemas for all API endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """Single transcription segment with speaker attribution."""

    start_time: float = Field(description="Segment start time in seconds")
    end_time: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed text")
    confidence: float = Field(
        description="Transcription confidence (0-1)", ge=0.0, le=1.0
    )
    speaker_id: Optional[str] = Field(
        default=None, description="Identified speaker ID"
    )
    speaker_role: Optional[str] = Field(
        default=None, description="Inferred bridge role"
    )


class WordTimestamp(BaseModel):
    """Word-level timestamp from Whisper."""

    word: str
    start: float
    end: float
    probability: float


class SpeakerInfo(BaseModel):
    """Speaker identification and engagement info."""

    speaker_id: str = Field(description="Unique speaker identifier")
    total_speaking_time: float = Field(description="Total speaking time in seconds")
    utterance_count: int = Field(description="Number of utterances")
    avg_utterance_duration: float = Field(
        description="Average utterance duration in seconds"
    )
    role: Optional[str] = Field(default=None, description="Inferred bridge role")
    engagement_score: Optional[float] = Field(
        default=None, description="Engagement score (0-100)"
    )


class CommunicationPatternMatch(BaseModel):
    """A matched communication pattern."""

    pattern_name: str
    category: str  # "effective" or "needs_improvement"
    description: str
    count: int
    examples: List[str] = Field(default_factory=list)


class CommunicationQuality(BaseModel):
    """Communication quality analysis results."""

    effective_count: int = Field(description="Number of effective patterns found")
    improvement_count: int = Field(
        description="Number of patterns needing improvement"
    )
    effective_percentage: float = Field(
        description="Percentage of effective communications"
    )
    patterns: List[CommunicationPatternMatch] = Field(
        default_factory=list, description="Matched patterns with details"
    )


class MetricScore(BaseModel):
    """Score for a single performance metric."""

    metric_name: str = Field(description="Name of the metric")
    display_name: str = Field(description="Human-readable name")
    score: int = Field(description="Score from 1-5", ge=1, le=5)
    evidence: str = Field(description="Supporting evidence for the score")


class SpeakerScorecard(BaseModel):
    """Detailed scorecard for a speaker with 1-5 ratings."""

    speaker_id: str = Field(description="Speaker identifier")
    inferred_role: str = Field(description="Inferred bridge role")
    utterance_count: int = Field(description="Number of utterances")
    overall_score: float = Field(description="Overall score (1-5 average)")
    scores: List[MetricScore] = Field(description="Individual metric scores")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    development_areas: List[str] = Field(
        default_factory=list, description="Areas for improvement"
    )


class RoleAssignment(BaseModel):
    """Role inference result for a speaker."""

    speaker_id: str = Field(description="Speaker identifier")
    role: str = Field(description="Inferred role name")
    confidence: float = Field(description="Confidence in role assignment (0-1)")
    keyword_matches: int = Field(description="Number of role-related keywords found")
    key_indicators: List[str] = Field(
        default_factory=list, description="Key phrases that indicate this role"
    )


class ConfidenceBucket(BaseModel):
    """Confidence distribution bucket."""

    label: str = Field(description="Bucket label (e.g., '90% and above')")
    range_name: str = Field(description="Range name (excellent/good/acceptable/marginal/poor)")
    count: int = Field(description="Number of utterances in this range")
    percentage: float = Field(description="Percentage of total utterances")


class ConfidenceDistribution(BaseModel):
    """Confidence score distribution analysis."""

    total_utterances: int = Field(description="Total number of utterances")
    average_confidence: float = Field(description="Average confidence score")
    buckets: List[ConfidenceBucket] = Field(description="Distribution buckets")
    speaker_averages: Dict[str, float] = Field(
        default_factory=dict, description="Per-speaker average confidence"
    )
    quality_assessment: str = Field(description="Overall audio quality assessment")


class KirkpatrickLevel(BaseModel):
    """Single level of Kirkpatrick evaluation."""

    level: int = Field(description="Level number (1-4)")
    name: str = Field(description="Level name")
    score: float = Field(description="Score for this level (0-100)")
    interpretation: str = Field(description="Interpretation of the score")


class LearningEvaluation(BaseModel):
    """Learning framework evaluation results."""

    kirkpatrick_levels: List[KirkpatrickLevel] = Field(
        description="Kirkpatrick 4-level evaluation"
    )
    blooms_level: str = Field(description="Highest Bloom's Taxonomy level observed")
    blooms_score: float = Field(description="Bloom's Taxonomy score (0-100)")
    nasa_teamwork_score: float = Field(description="NASA Teamwork Framework score (0-100)")
    overall_learning_score: float = Field(description="Combined learning score (0-100)")


# ============================================================================
# Educational Framework Models (7 Habits, Training Recommendations)
# ============================================================================


class HabitScore(BaseModel):
    """Score for a single habit from the 7 Habits framework."""

    habit_number: int = Field(description="Habit number (1-7)")
    habit_name: str = Field(description="Official habit name")
    youth_friendly_name: str = Field(description="Youth-friendly habit name")
    score: int = Field(description="Score from 1-5", ge=1, le=5)
    observation_count: int = Field(description="Number of observations")
    interpretation: str = Field(description="Assessment interpretation")
    development_tip: str = Field(description="Tip for developing this habit")
    examples: List[str] = Field(default_factory=list, description="Example quotes")


class SevenHabitsAssessment(BaseModel):
    """Complete 7 Habits of Highly Effective People assessment."""

    overall_score: float = Field(description="Overall effectiveness score (1-5)")
    habits: List[HabitScore] = Field(description="Individual habit scores")
    strengths: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top strength habits"
    )
    growth_areas: List[Dict[str, Any]] = Field(
        default_factory=list, description="Habits needing development"
    )


class TrainingRecommendationItem(BaseModel):
    """Single training recommendation."""

    title: str = Field(description="Recommendation title")
    description: str = Field(description="Detailed description")
    priority: str = Field(description="Priority level (CRITICAL/HIGH/MEDIUM/LOW)")
    category: str = Field(description="Skill category")
    frameworks: List[str] = Field(default_factory=list, description="Related frameworks")
    scout_connection: Optional[str] = Field(
        default=None, description="Connection to Scout Law"
    )
    habit_connection: Optional[str] = Field(
        default=None, description="Connection to 7 Habits"
    )
    success_criteria: str = Field(description="How to measure success")


class DrillActivity(BaseModel):
    """Training drill activity."""

    name: str = Field(description="Drill name")
    purpose: str = Field(description="Purpose of the drill")
    duration: str = Field(description="Estimated duration")
    participants: str = Field(description="Who should participate")
    steps: List[str] = Field(default_factory=list, description="Activity steps")
    debrief_questions: List[str] = Field(
        default_factory=list, description="Discussion questions"
    )
    frameworks_addressed: List[str] = Field(
        default_factory=list, description="Frameworks this drill addresses"
    )


class DiscussionTopic(BaseModel):
    """Team discussion topic."""

    topic: str = Field(description="Discussion topic title")
    question: str = Field(description="Main discussion question")
    scout_connection: Optional[str] = Field(
        default=None, description="Connection to Scout Law"
    )
    discussion_points: List[str] = Field(
        default_factory=list, description="Key points to cover"
    )


class TrainingRecommendations(BaseModel):
    """Complete training recommendations for educational contexts."""

    immediate_actions: List[TrainingRecommendationItem] = Field(
        default_factory=list, description="Immediate action items"
    )
    communication_improvements: List[TrainingRecommendationItem] = Field(
        default_factory=list, description="Communication skill recommendations"
    )
    leadership_development: List[TrainingRecommendationItem] = Field(
        default_factory=list, description="Leadership development recommendations"
    )
    teamwork_enhancements: List[TrainingRecommendationItem] = Field(
        default_factory=list, description="Teamwork improvement recommendations"
    )
    drills: List[DrillActivity] = Field(
        default_factory=list, description="Recommended training drills"
    )
    discussion_topics: List[DiscussionTopic] = Field(
        default_factory=list, description="Team discussion topics"
    )
    framework_alignment: Dict[str, List[str]] = Field(
        default_factory=dict, description="How frameworks align with each other"
    )
    total_recommendations: int = Field(
        default=0, description="Total number of recommendations"
    )


class AnalysisResult(BaseModel):
    """Complete audio analysis result."""

    transcription: List[TranscriptionSegment] = Field(
        description="Transcribed segments with speaker info"
    )
    full_text: str = Field(description="Full transcript as plain text")
    duration_seconds: float = Field(description="Audio duration in seconds")
    speakers: List[SpeakerInfo] = Field(description="Identified speakers")
    communication_quality: Optional[CommunicationQuality] = Field(
        default=None, description="Communication quality analysis"
    )
    # New detailed analysis fields
    speaker_scorecards: List[SpeakerScorecard] = Field(
        default_factory=list, description="Detailed scorecards per speaker"
    )
    role_assignments: List[RoleAssignment] = Field(
        default_factory=list, description="Inferred roles for each speaker"
    )
    confidence_distribution: Optional[ConfidenceDistribution] = Field(
        default=None, description="Confidence score distribution"
    )
    learning_evaluation: Optional[LearningEvaluation] = Field(
        default=None, description="Learning framework evaluation"
    )
    # Educational framework analysis
    seven_habits: Optional[SevenHabitsAssessment] = Field(
        default=None, description="7 Habits of Highly Effective People assessment"
    )
    training_recommendations: Optional[TrainingRecommendations] = Field(
        default=None, description="Training recommendations for educational contexts"
    )
    processing_time_seconds: float = Field(
        description="Total processing time in seconds"
    )


class TranscriptionResult(BaseModel):
    """Transcription-only result (no speaker/quality analysis)."""

    segments: List[TranscriptionSegment] = Field(description="Transcribed segments")
    full_text: str = Field(description="Full transcript as plain text")
    duration_seconds: float = Field(description="Audio duration in seconds")
    language: str = Field(description="Detected language")
    processing_time_seconds: float = Field(description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    whisper_loaded: bool = Field(description="Whether Whisper model is loaded")
    whisper_model: Optional[str] = Field(
        default=None, description="Loaded Whisper model size"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error info")
