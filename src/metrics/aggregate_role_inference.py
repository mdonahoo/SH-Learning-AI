"""
Aggregate role inference engine with voice confidence integration.

This module provides enhanced role inference that combines:
1. Voice confidence from diarization (how tight is the speaker's voice cluster)
2. Role confidence from keyword matching (how dominant is one role pattern)
3. Evidence factor (more utterances = more reliable inference)

This is Pass 2 of the two-pass processing architecture, designed to work
with DiarizationResult from Pass 1 (batch diarization).
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# Import base role inference components
try:
    from src.metrics.role_inference import (
        RoleInferenceEngine,
        BridgeRole,
        RolePatterns,
        SpeakerRoleAnalysis
    )
    from src.audio.batch_diarizer import DiarizationResult, SpeakerCluster
    ROLE_INFERENCE_AVAILABLE = True
    BATCH_DIARIZER_AVAILABLE = True
except ImportError as e:
    ROLE_INFERENCE_AVAILABLE = False
    BATCH_DIARIZER_AVAILABLE = False
    logger.warning(f"Role inference or batch diarizer not available: {e}")


@dataclass
class AggregateRoleAnalysis:
    """
    Enhanced role analysis with combined confidence metrics.

    Attributes:
        speaker: Speaker identifier
        inferred_role: Detected bridge role
        voice_confidence: How consistent is this speaker's voice (from diarization)
        role_confidence: How dominant is one role in keyword matches
        combined_confidence: Overall confidence combining voice + role + evidence
        utterance_count: Number of utterances analyzed
        utterance_percentage: Percentage of total voice traffic
        keyword_matches: Dict of matched keywords and counts
        total_keyword_matches: Total keyword pattern matches
        key_indicators: Top keywords indicating the role
        example_utterances: Sample utterances showing role behavior
        methodology_notes: Explanation of how role was determined
        evidence_factor: min(1.0, utterance_count / 20) - more data = more reliable
    """
    speaker: str
    inferred_role: BridgeRole
    voice_confidence: float
    role_confidence: float
    combined_confidence: float
    utterance_count: int
    utterance_percentage: float
    keyword_matches: Dict[str, int]
    total_keyword_matches: int
    key_indicators: List[str]
    example_utterances: List[Dict[str, Any]]
    methodology_notes: str
    evidence_factor: float = 0.0


class AggregateRoleInferenceEngine:
    """
    Enhanced role inference using diarization confidence.

    This engine combines voice clustering confidence from Pass 1
    (batch diarization) with keyword-based role detection to produce
    more accurate role assignments.

    Combined Confidence Formula:
        combined = voice_confidence * 0.40 + role_confidence * 0.40 + evidence_factor * 0.20

    Where:
    - voice_confidence: How tight is the speaker's voice cluster (from Pass 1)
    - role_confidence: How dominant is one role in keyword matches (from Pass 2)
    - evidence_factor: min(1.0, utterance_count / 20) - more data = more reliable
    """

    # Weights for combined confidence calculation
    VOICE_WEIGHT = 0.40
    ROLE_WEIGHT = 0.40
    EVIDENCE_WEIGHT = 0.20

    # Minimum utterances for full evidence factor
    MIN_UTTERANCES_FOR_FULL_EVIDENCE = 20

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        diarization_result: Optional['DiarizationResult'] = None,
        patterns: Optional[RolePatterns] = None
    ):
        """
        Initialize aggregate role inference engine.

        Args:
            transcripts: List of transcript dictionaries with 'speaker' and 'text'
            diarization_result: Optional result from batch diarization (Pass 1)
            patterns: Optional custom role patterns
        """
        self.transcripts = transcripts
        self.diarization_result = diarization_result
        self.patterns = patterns or RolePatterns()

        # Initialize base engine for keyword analysis
        if ROLE_INFERENCE_AVAILABLE:
            self._base_engine = RoleInferenceEngine(transcripts, patterns)
        else:
            self._base_engine = None
            logger.warning("Base role inference not available")

    def infer_roles(self) -> Dict[str, AggregateRoleAnalysis]:
        """
        Perform aggregate role inference with combined confidence.

        Returns:
            Dictionary mapping speaker IDs to AggregateRoleAnalysis
        """
        if not self._base_engine:
            logger.error("Role inference engine not available")
            return {}

        # Get base keyword analysis
        base_results = self._base_engine.analyze_all_speakers()

        # Enhance with diarization confidence
        aggregate_results = {}
        for speaker_id, base_analysis in base_results.items():
            aggregate = self._analyze_speaker_aggregate(speaker_id, base_analysis)
            aggregate_results[speaker_id] = aggregate

        # Post-process to resolve conflicts
        aggregate_results = self._resolve_conflicts(aggregate_results)

        return aggregate_results

    def _analyze_speaker_aggregate(
        self,
        speaker_id: str,
        base_analysis: 'SpeakerRoleAnalysis'
    ) -> AggregateRoleAnalysis:
        """
        Analyze a single speaker with combined confidence.

        Args:
            speaker_id: Speaker identifier
            base_analysis: Base keyword analysis result

        Returns:
            AggregateRoleAnalysis with combined confidence
        """
        # Get voice confidence from diarization result
        voice_confidence = self._get_voice_confidence(speaker_id)

        # Calculate role confidence (how dominant is one role)
        role_confidence = self._calculate_role_confidence(base_analysis)

        # Calculate evidence factor
        evidence_factor = min(
            1.0,
            base_analysis.utterance_count / self.MIN_UTTERANCES_FOR_FULL_EVIDENCE
        )

        # Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(
            voice_confidence, role_confidence, evidence_factor
        )

        # Build methodology note
        methodology = self._build_methodology_note(
            speaker_id,
            base_analysis,
            voice_confidence,
            role_confidence,
            evidence_factor,
            combined_confidence
        )

        return AggregateRoleAnalysis(
            speaker=speaker_id,
            inferred_role=base_analysis.inferred_role,
            voice_confidence=round(voice_confidence, 3),
            role_confidence=round(role_confidence, 3),
            combined_confidence=round(combined_confidence, 3),
            utterance_count=base_analysis.utterance_count,
            utterance_percentage=base_analysis.utterance_percentage,
            keyword_matches=base_analysis.keyword_matches,
            total_keyword_matches=base_analysis.total_keyword_matches,
            key_indicators=base_analysis.key_indicators,
            example_utterances=base_analysis.example_utterances,
            methodology_notes=methodology,
            evidence_factor=round(evidence_factor, 3)
        )

    def _get_voice_confidence(self, speaker_id: str) -> float:
        """
        Get voice confidence from diarization result.

        Returns cluster tightness (0-1) indicating how consistent
        the speaker's voice is across all their segments.
        """
        if not self.diarization_result:
            return 0.5  # Default when no diarization data

        return self.diarization_result.get_speaker_voice_confidence(speaker_id)

    def _calculate_role_confidence(
        self,
        base_analysis: 'SpeakerRoleAnalysis'
    ) -> float:
        """
        Calculate role confidence from keyword analysis.

        Higher confidence when one role clearly dominates the keyword matches.
        """
        if base_analysis.total_keyword_matches == 0:
            return 0.0

        # The base analysis already calculates confidence based on
        # role dominance, so we can use that directly
        return base_analysis.confidence

    def _calculate_combined_confidence(
        self,
        voice_confidence: float,
        role_confidence: float,
        evidence_factor: float
    ) -> float:
        """
        Calculate combined confidence using weighted formula.

        Formula:
            combined = voice * 0.40 + role * 0.40 + evidence * 0.20
        """
        combined = (
            voice_confidence * self.VOICE_WEIGHT +
            role_confidence * self.ROLE_WEIGHT +
            evidence_factor * self.EVIDENCE_WEIGHT
        )
        return min(1.0, max(0.0, combined))

    def _build_methodology_note(
        self,
        speaker_id: str,
        base_analysis: 'SpeakerRoleAnalysis',
        voice_confidence: float,
        role_confidence: float,
        evidence_factor: float,
        combined_confidence: float
    ) -> str:
        """Build detailed methodology note explaining the assignment."""
        role_name = (
            base_analysis.inferred_role.value
            if hasattr(base_analysis.inferred_role, 'value')
            else str(base_analysis.inferred_role)
        )

        parts = [base_analysis.methodology_notes]

        # Add combined confidence breakdown
        parts.append(
            f"Combined confidence: {combined_confidence:.0%} "
            f"(voice={voice_confidence:.0%}, role={role_confidence:.0%}, "
            f"evidence={evidence_factor:.0%})"
        )

        # Add diarization note if available
        if self.diarization_result:
            cluster = self.diarization_result.speaker_clusters.get(speaker_id)
            if cluster:
                parts.append(
                    f"Voice cluster: {len(cluster.embeddings)} embeddings, "
                    f"tightness={cluster.cluster_tightness:.2f}"
                )

        return " ".join(parts)

    def _resolve_conflicts(
        self,
        results: Dict[str, AggregateRoleAnalysis]
    ) -> Dict[str, AggregateRoleAnalysis]:
        """
        Resolve conflicts when multiple speakers have same role.

        Speaker with highest combined confidence keeps the role.
        """
        # Group by inferred role
        role_assignments: Dict[BridgeRole, List[AggregateRoleAnalysis]] = defaultdict(list)
        for analysis in results.values():
            role_assignments[analysis.inferred_role].append(analysis)

        # For roles with multiple speakers, keep strongest match
        reassignments = {}
        for role, speakers in role_assignments.items():
            if role == BridgeRole.UNKNOWN:
                continue

            if len(speakers) > 1:
                # Sort by combined confidence
                sorted_speakers = sorted(
                    speakers,
                    key=lambda x: x.combined_confidence,
                    reverse=True
                )

                # First speaker keeps the role
                for secondary in sorted_speakers[1:]:
                    if secondary.utterance_percentage > 15:
                        reassignments[secondary.speaker] = BridgeRole.EXECUTIVE_OFFICER
                    else:
                        reassignments[secondary.speaker] = BridgeRole.UNKNOWN

        # Apply reassignments
        for speaker_id, new_role in reassignments.items():
            old_analysis = results[speaker_id]
            results[speaker_id] = AggregateRoleAnalysis(
                speaker=old_analysis.speaker,
                inferred_role=new_role,
                voice_confidence=old_analysis.voice_confidence,
                role_confidence=old_analysis.role_confidence * 0.7,
                combined_confidence=old_analysis.combined_confidence * 0.7,
                utterance_count=old_analysis.utterance_count,
                utterance_percentage=old_analysis.utterance_percentage,
                keyword_matches=old_analysis.keyword_matches,
                total_keyword_matches=old_analysis.total_keyword_matches,
                key_indicators=old_analysis.key_indicators,
                example_utterances=old_analysis.example_utterances,
                methodology_notes=(
                    old_analysis.methodology_notes +
                    f" (Reassigned from {old_analysis.inferred_role.value} due to role conflict.)"
                ),
                evidence_factor=old_analysis.evidence_factor
            )

        return results

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for API response.

        Returns:
            Dictionary with all role analysis data
        """
        results = self.infer_roles()

        speaker_roles = {}
        for speaker_id, analysis in results.items():
            role_value = (
                analysis.inferred_role.value
                if hasattr(analysis.inferred_role, 'value')
                else str(analysis.inferred_role)
            )
            speaker_roles[speaker_id] = {
                'role': role_value,
                'voice_confidence': analysis.voice_confidence,
                'role_confidence': analysis.role_confidence,
                'combined_confidence': analysis.combined_confidence,
                'utterance_count': analysis.utterance_count,
                'utterance_percentage': analysis.utterance_percentage,
                'keyword_matches': analysis.total_keyword_matches,
                'key_indicators': analysis.key_indicators,
                'example_utterances': analysis.example_utterances,
                'methodology_note': analysis.methodology_notes,
                'evidence_factor': analysis.evidence_factor
            }

        # Add diarization methodology if available
        diarization_methodology = ""
        if self.diarization_result:
            diarization_methodology = self.diarization_result.methodology_note

        return {
            'speaker_roles': speaker_roles,
            'diarization_methodology': diarization_methodology,
            'inference_weights': {
                'voice_weight': self.VOICE_WEIGHT,
                'role_weight': self.ROLE_WEIGHT,
                'evidence_weight': self.EVIDENCE_WEIGHT
            }
        }


def is_aggregate_inference_available() -> bool:
    """Check if aggregate role inference is available."""
    return ROLE_INFERENCE_AVAILABLE and BATCH_DIARIZER_AVAILABLE
