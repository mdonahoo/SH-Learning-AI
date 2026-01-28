"""
Pydantic models for the audio analysis web API.

Defines request/response schemas for all API endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SegmentSentiment(BaseModel):
    """Per-segment stress and sentiment analysis result."""

    stress_level: float = Field(description="Composite stress score (0-1)")
    stress_label: str = Field(description="Stress category: low/moderate/high")
    emotion: str = Field(
        description="Emotion label: calm/focused/tense/urgent/critical"
    )
    sentiment: str = Field(description="Polarity: positive/neutral/negative")
    sentiment_score: float = Field(
        description="Sentiment polarity score (-1 to 1)"
    )
    signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of individual scoring signals"
    )


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
    sentiment: Optional[SegmentSentiment] = Field(
        default=None, description="Stress/sentiment analysis for this segment"
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


class PatternEvidence(BaseModel):
    """Evidence for a single pattern match."""

    text: str = Field(description="The text that matched the pattern")
    speaker: Optional[str] = Field(default=None, description="Speaker who said this")
    timestamp: Optional[float] = Field(default=None, description="Start time in seconds")
    matched_substring: Optional[str] = Field(
        default=None, description="The specific substring that triggered the match"
    )


class CommunicationPatternMatch(BaseModel):
    """A matched communication pattern."""

    pattern_name: str
    category: str  # "effective" or "needs_improvement"
    description: str
    count: int
    examples: List[Any] = Field(default_factory=list)
    # Evidence fields
    evidence_details: List[PatternEvidence] = Field(
        default_factory=list, description="Detailed evidence for each match"
    )


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
    # Evidence fields
    total_utterances_assessed: int = Field(
        default=0, description="Total number of utterances assessed"
    )
    calculation_summary: Optional[str] = Field(
        default=None,
        description="Summary of how the metrics were calculated"
    )


class MetricScore(BaseModel):
    """Score for a single performance metric."""

    metric_name: str = Field(description="Name of the metric")
    display_name: str = Field(description="Human-readable name")
    score: int = Field(description="Score from 1-5", ge=1, le=5)
    evidence: str = Field(description="Supporting evidence for the score")
    # Evidence fields
    supporting_quotes: List[str] = Field(
        default_factory=list, description="Supporting quotes from transcript"
    )
    threshold_info: Optional[str] = Field(
        default=None, description="Score thresholds used for this metric"
    )
    calculation_details: Optional[str] = Field(
        default=None, description="Details of how score was calculated"
    )


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
    confidence: float = Field(description="Combined confidence in role assignment (0-1)")
    keyword_matches: int = Field(description="Number of role-related keywords found")
    key_indicators: List[str] = Field(
        default_factory=list, description="Key phrases that indicate this role"
    )
    # Telemetry correlation fields
    voice_confidence: Optional[float] = Field(
        default=None, description="Base confidence from keyword/voice analysis (0-1)"
    )
    telemetry_confidence: Optional[float] = Field(
        default=None, description="Confidence boost from telemetry correlation (0-1)"
    )
    evidence_count: Optional[int] = Field(
        default=None, description="Number of supporting telemetry events"
    )
    methodology_note: Optional[str] = Field(
        default=None, description="Explanation of how confidence was calculated"
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
    examples: List[Any] = Field(default_factory=list, description="Example quotes with speaker info")
    # Evidence fields
    pattern_breakdown: Dict[str, int] = Field(
        default_factory=dict, description="Count of each pattern type matched"
    )
    speaker_contributions: Dict[str, int] = Field(
        default_factory=dict, description="Contributions per speaker"
    )
    gap_to_next_score: Optional[str] = Field(
        default=None, description="What's needed to reach the next score level"
    )


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
    # Evidence fields
    trigger_metrics: List[str] = Field(
        default_factory=list, description="Which metrics triggered this recommendation"
    )
    current_value: Optional[str] = Field(
        default=None, description="Current observed value"
    )
    target_value: Optional[str] = Field(
        default=None, description="Target value to achieve"
    )
    gap_explanation: Optional[str] = Field(
        default=None, description="Explanation of the gap to address"
    )


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


class WaveformData(BaseModel):
    """Audio waveform amplitude envelope for timeline visualization."""

    sample_rate: int = Field(description="Output samples per second")
    duration_seconds: float = Field(description="Audio duration in seconds")
    amplitude: List[float] = Field(
        default_factory=list, description="Normalized 0-1 RMS amplitude values"
    )
    peak_amplitude: float = Field(description="Peak raw amplitude value")
    average_amplitude: float = Field(description="Average raw amplitude value")


class SentimentSummary(BaseModel):
    """Aggregate stress/sentiment statistics across all segments."""

    average_stress: float = Field(description="Mean stress level (0-1)")
    peak_stress_time: float = Field(
        description="Timestamp of peak stress in seconds"
    )
    peak_stress_level: float = Field(description="Maximum stress level (0-1)")
    stress_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Percentage distribution: low/moderate/high"
    )
    speaker_stress: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-speaker stress statistics"
    )
    stress_timeline: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Downsampled stress timeline for frontend chart"
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
    # Sentiment spectrogram data
    waveform_data: Optional[WaveformData] = Field(
        default=None, description="Audio waveform amplitude envelope"
    )
    sentiment_summary: Optional[SentimentSummary] = Field(
        default=None, description="Aggregate stress/sentiment statistics"
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


class ServiceStatus(BaseModel):
    """Individual service status."""

    available: bool = Field(description="Whether service is available")
    status: str = Field(description="Status message")
    details: Optional[str] = Field(default=None, description="Additional details")


class ServicesStatusResponse(BaseModel):
    """Status of all external services."""

    whisper: ServiceStatus = Field(description="Whisper transcription service status")
    ollama: ServiceStatus = Field(description="Ollama LLM service status")
    diarization: ServiceStatus = Field(description="Speaker diarization service status")
    horizons: ServiceStatus = Field(description="Starship Horizons game server status")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error info")
