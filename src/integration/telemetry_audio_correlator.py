"""
Telemetry-Audio Correlator Module.

Correlates game telemetry events with audio segments to anchor speaker
identification to console positions, boosting role confidence from
keyword-only analysis.

The correlator matches telemetry events (e.g., throttle changes, weapons fire)
with nearby audio segments to validate voice patterns against console actions.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# Configuration from environment
CORRELATION_WINDOW_MS = int(os.getenv('CORRELATION_WINDOW_MS', '500'))
MIN_CONFIDENCE_BOOST = float(os.getenv('MIN_CONFIDENCE_BOOST', '0.1'))
MAX_CONFIDENCE_BOOST = float(os.getenv('MAX_CONFIDENCE_BOOST', '0.3'))


class BridgeRole(Enum):
    """Standard bridge roles in Starship Horizons."""
    CAPTAIN = "Captain/Command"
    HELM = "Helm/Navigation"
    TACTICAL = "Tactical/Weapons"
    SCIENCE = "Science/Sensors"
    ENGINEERING = "Engineering/Systems"
    OPERATIONS = "Operations/Monitoring"
    COMMUNICATIONS = "Communications"
    UNKNOWN = "Crew Member"


# Category to role mapping for telemetry events
CATEGORY_ROLE_MAP: Dict[str, BridgeRole] = {
    # Helm/Navigation categories
    "helm": BridgeRole.HELM,
    "navigation": BridgeRole.HELM,
    "course": BridgeRole.HELM,
    "heading": BridgeRole.HELM,
    "throttle": BridgeRole.HELM,
    "warp": BridgeRole.HELM,
    "impulse": BridgeRole.HELM,

    # Tactical/Weapons categories
    "tactical": BridgeRole.TACTICAL,
    "combat": BridgeRole.TACTICAL,
    "weapons": BridgeRole.TACTICAL,
    "defensive": BridgeRole.TACTICAL,
    "shields": BridgeRole.TACTICAL,
    "targeting": BridgeRole.TACTICAL,
    "fire": BridgeRole.TACTICAL,

    # Science categories
    "science": BridgeRole.SCIENCE,
    "sensors": BridgeRole.SCIENCE,
    "scan": BridgeRole.SCIENCE,
    "analysis": BridgeRole.SCIENCE,

    # Engineering categories
    "engineering": BridgeRole.ENGINEERING,
    "power": BridgeRole.ENGINEERING,
    "systems": BridgeRole.ENGINEERING,
    "repairs": BridgeRole.ENGINEERING,
    "reactor": BridgeRole.ENGINEERING,
    "damage_control": BridgeRole.ENGINEERING,

    # Operations categories
    "operations": BridgeRole.OPERATIONS,
    "monitoring": BridgeRole.OPERATIONS,
    "cargo": BridgeRole.OPERATIONS,
    "docking": BridgeRole.OPERATIONS,

    # Communications categories
    "communications": BridgeRole.COMMUNICATIONS,
    "hailing": BridgeRole.COMMUNICATIONS,
    "transmission": BridgeRole.COMMUNICATIONS,

    # System/telemetry (not role-specific)
    "telemetry": None,
    "critical": None,
    "system_alert": None,
    "crew_communication": None,
}


@dataclass
class CorrelationEvidence:
    """Evidence of correlation between a telemetry event and an audio segment."""
    event_id: str
    event_type: str
    event_category: str
    speaker_id: str
    expected_role: str
    time_delta_ms: float
    confidence_contribution: float
    event_timestamp: float
    segment_timestamp: float
    segment_text: str = ""


@dataclass
class RoleConfidenceUpdate:
    """Updated role confidence with telemetry-based boost."""
    speaker_id: str
    role: str
    base_confidence: float
    boosted_confidence: float
    evidence_count: int
    evidence_trail: List[CorrelationEvidence] = field(default_factory=list)
    telemetry_boost: float = 0.0
    methodology_note: str = ""


class TelemetryAudioCorrelator:
    """
    Correlates telemetry events with audio segments to enhance speaker
    role identification.

    The correlator anchors voice patterns to console positions by matching
    telemetry events (console actions) with nearby audio segments within
    a configurable time window.
    """

    def __init__(
        self,
        correlation_window_ms: Optional[int] = None,
        min_confidence_boost: Optional[float] = None,
        max_confidence_boost: Optional[float] = None
    ):
        """
        Initialize the correlator.

        Args:
            correlation_window_ms: Time window for event-audio matching (ms)
            min_confidence_boost: Minimum boost from telemetry correlation
            max_confidence_boost: Maximum boost from telemetry correlation
        """
        self.correlation_window_ms = correlation_window_ms or CORRELATION_WINDOW_MS
        self.min_confidence_boost = min_confidence_boost or MIN_CONFIDENCE_BOOST
        self.max_confidence_boost = max_confidence_boost or MAX_CONFIDENCE_BOOST

        self.events: List[Dict[str, Any]] = []
        self.transcripts: List[Dict[str, Any]] = []
        self.mission_start: Optional[datetime] = None

        self._correlations: List[CorrelationEvidence] = []
        self._speaker_evidence: Dict[str, List[CorrelationEvidence]] = defaultdict(list)

        logger.info(
            f"TelemetryAudioCorrelator initialized: "
            f"window={self.correlation_window_ms}ms, "
            f"boost_range=[{self.min_confidence_boost:.2f}, {self.max_confidence_boost:.2f}]"
        )

    def load_data(
        self,
        events: List[Dict[str, Any]],
        transcripts: List[Dict[str, Any]],
        mission_start: Optional[datetime] = None
    ) -> None:
        """
        Load telemetry events and audio transcripts for correlation.

        Args:
            events: List of telemetry events from EventRecorder
            transcripts: List of transcript segments with speaker_id and timestamps
            mission_start: Optional mission start time for absolute-to-relative conversion
        """
        self.events = events or []
        self.transcripts = transcripts or []
        self.mission_start = mission_start

        # Clear previous correlations
        self._correlations = []
        self._speaker_evidence = defaultdict(list)

        # Unify timeline if we have a mission start
        self._unify_timeline()

        logger.info(
            f"Loaded {len(self.events)} events and {len(self.transcripts)} transcripts"
        )

    def _unify_timeline(self) -> None:
        """Convert all timestamps to seconds from mission start."""
        if not self.mission_start:
            # Try to infer from first event
            if self.events and 'timestamp' in self.events[0]:
                ts = self.events[0]['timestamp']
                if isinstance(ts, datetime):
                    self.mission_start = ts
                elif isinstance(ts, str):
                    try:
                        self.mission_start = datetime.fromisoformat(ts)
                    except ValueError:
                        pass

        # Events already have relative timestamps (in seconds) in many cases
        # We'll normalize by checking the type
        for event in self.events:
            if 'timestamp' in event and 'relative_time' not in event:
                ts = event['timestamp']
                if isinstance(ts, datetime) and self.mission_start:
                    delta = ts - self.mission_start
                    event['relative_time'] = delta.total_seconds()
                elif isinstance(ts, (int, float)):
                    event['relative_time'] = float(ts)
                elif isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts)
                        if self.mission_start:
                            delta = dt - self.mission_start
                            event['relative_time'] = delta.total_seconds()
                    except ValueError:
                        event['relative_time'] = 0.0

        # Transcripts typically have start_time in seconds
        for transcript in self.transcripts:
            if 'relative_time' not in transcript:
                transcript['relative_time'] = transcript.get(
                    'start_time', transcript.get('timestamp', 0.0)
                )

    def correlate_all(self) -> List[CorrelationEvidence]:
        """
        Find all correlations between telemetry events and audio segments.

        Returns:
            List of CorrelationEvidence objects
        """
        self._correlations = []
        self._speaker_evidence = defaultdict(list)

        window_seconds = self.correlation_window_ms / 1000.0

        for event in self.events:
            category = event.get('category', '').lower()
            expected_role = self._category_to_role(category)

            if not expected_role:
                continue  # Skip events without role mapping

            event_time = event.get('relative_time', 0.0)
            event_id = event.get('event_id', str(id(event)))
            event_type = event.get('event_type', 'unknown')

            # Find audio segments within the correlation window
            for transcript in self.transcripts:
                segment_time = transcript.get('relative_time', 0.0)
                speaker_id = transcript.get('speaker_id') or transcript.get('speaker')

                if not speaker_id:
                    continue

                time_delta_ms = abs(event_time - segment_time) * 1000

                if time_delta_ms <= self.correlation_window_ms:
                    # Calculate confidence contribution based on time proximity
                    # Closer = higher contribution
                    proximity_factor = 1.0 - (time_delta_ms / self.correlation_window_ms)
                    confidence_contribution = (
                        self.min_confidence_boost +
                        (self.max_confidence_boost - self.min_confidence_boost) * proximity_factor
                    )

                    evidence = CorrelationEvidence(
                        event_id=event_id,
                        event_type=event_type,
                        event_category=category,
                        speaker_id=speaker_id,
                        expected_role=expected_role.value,
                        time_delta_ms=time_delta_ms,
                        confidence_contribution=confidence_contribution,
                        event_timestamp=event_time,
                        segment_timestamp=segment_time,
                        segment_text=transcript.get('text', '')[:100]
                    )

                    self._correlations.append(evidence)
                    self._speaker_evidence[speaker_id].append(evidence)

        logger.info(f"Found {len(self._correlations)} correlations")
        return self._correlations

    def _category_to_role(self, category: str) -> Optional[BridgeRole]:
        """Map event category to expected bridge role."""
        category_lower = category.lower().strip()

        # Direct lookup
        if category_lower in CATEGORY_ROLE_MAP:
            return CATEGORY_ROLE_MAP[category_lower]

        # Check for partial matches
        for key, role in CATEGORY_ROLE_MAP.items():
            if key in category_lower or category_lower in key:
                return role

        return None

    def update_role_confidences(
        self,
        existing_roles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, RoleConfidenceUpdate]:
        """
        Update role confidences based on telemetry correlations.

        Args:
            existing_roles: Dict mapping speaker_id to role info
                           (role, confidence, methodology_notes, etc.)

        Returns:
            Dict mapping speaker_id to RoleConfidenceUpdate
        """
        updates: Dict[str, RoleConfidenceUpdate] = {}

        for speaker_id, evidence_list in self._speaker_evidence.items():
            if speaker_id not in existing_roles:
                continue

            role_info = existing_roles[speaker_id]
            base_confidence = role_info.get('confidence', 0.5)
            assigned_role = role_info.get('role', 'Crew Member')

            # Calculate telemetry boost
            # Sum contributions, but apply diminishing returns
            total_contribution = 0.0
            matching_evidence = []

            for evidence in evidence_list:
                # Check if the telemetry role matches the assigned role
                if self._roles_match(assigned_role, evidence.expected_role):
                    total_contribution += evidence.confidence_contribution
                    matching_evidence.append(evidence)

            # Apply diminishing returns: sqrt scaling
            if len(matching_evidence) > 0:
                import math
                # Average contribution * sqrt(count) / sqrt(count) = average
                # But we want more evidence = more confidence, with diminishing returns
                avg_contribution = total_contribution / len(matching_evidence)
                # Scale factor: sqrt of count, capped
                scale_factor = min(math.sqrt(len(matching_evidence)), 3.0)
                telemetry_boost = min(
                    avg_contribution * scale_factor,
                    self.max_confidence_boost
                )
            else:
                telemetry_boost = 0.0

            # Calculate boosted confidence (capped at 1.0)
            boosted_confidence = min(1.0, base_confidence + telemetry_boost)

            methodology = (
                f"Base confidence {base_confidence:.2%} from keyword analysis. "
                f"Telemetry correlation added {telemetry_boost:.2%} boost from "
                f"{len(matching_evidence)} matching console actions within "
                f"\u00b1{self.correlation_window_ms}ms window."
            )

            updates[speaker_id] = RoleConfidenceUpdate(
                speaker_id=speaker_id,
                role=assigned_role,
                base_confidence=base_confidence,
                boosted_confidence=boosted_confidence,
                evidence_count=len(matching_evidence),
                evidence_trail=matching_evidence[:10],  # Limit to 10 examples
                telemetry_boost=telemetry_boost,
                methodology_note=methodology
            )

        return updates

    def _roles_match(self, assigned_role: str, telemetry_role: str) -> bool:
        """Check if assigned role matches the telemetry-inferred role."""
        assigned = assigned_role.lower()
        telemetry = telemetry_role.lower()

        # Direct match
        if assigned == telemetry:
            return True

        # Partial match (e.g., "Helm/Navigation" contains "helm")
        if "helm" in assigned and "helm" in telemetry:
            return True
        if "navigation" in assigned and "navigation" in telemetry:
            return True
        if "tactical" in assigned and "tactical" in telemetry:
            return True
        if "weapons" in assigned and ("tactical" in telemetry or "weapons" in telemetry):
            return True
        if "science" in assigned and "science" in telemetry:
            return True
        if "sensor" in assigned and "sensor" in telemetry:
            return True
        if "engineer" in assigned and ("engineer" in telemetry or "systems" in telemetry):
            return True
        if "operations" in assigned and "operations" in telemetry:
            return True
        if "communications" in assigned and "communications" in telemetry:
            return True

        return False

    def integrate_with_role_inference(
        self,
        role_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate telemetry correlations with RoleInferenceEngine results.

        Args:
            role_results: Results from RoleInferenceEngine.get_structured_results()

        Returns:
            Enhanced results with telemetry confidence boost
        """
        speaker_roles = role_results.get('speaker_roles', {})

        # Run correlation if not already done
        if not self._correlations:
            self.correlate_all()

        # Update confidences
        updates = self.update_role_confidences(speaker_roles)

        # Merge updates back into speaker_roles
        enhanced_roles = {}
        for speaker_id, role_data in speaker_roles.items():
            enhanced = dict(role_data)

            if speaker_id in updates:
                update = updates[speaker_id]
                enhanced['voice_confidence'] = update.base_confidence
                enhanced['telemetry_confidence'] = update.telemetry_boost
                enhanced['confidence'] = update.boosted_confidence
                enhanced['evidence_count'] = update.evidence_count
                enhanced['methodology_note'] = update.methodology_note
                enhanced['telemetry_evidence'] = [
                    {
                        'event_type': e.event_type,
                        'event_category': e.event_category,
                        'time_delta_ms': e.time_delta_ms,
                        'event_timestamp': e.event_timestamp,
                        'segment_text': e.segment_text[:50]
                    }
                    for e in update.evidence_trail[:5]
                ]

            enhanced_roles[speaker_id] = enhanced

        return {
            **role_results,
            'speaker_roles': enhanced_roles,
            'telemetry_correlation': {
                'total_correlations': len(self._correlations),
                'speakers_with_evidence': len(updates),
                'average_evidence_per_speaker': (
                    sum(u.evidence_count for u in updates.values()) / len(updates)
                    if updates else 0
                )
            }
        }

    def correlate_realtime(
        self,
        event: Dict[str, Any],
        recent_segments: List[Dict[str, Any]]
    ) -> Optional[CorrelationEvidence]:
        """
        Correlate a single event with recent audio segments in real-time.

        Args:
            event: A single telemetry event
            recent_segments: Recent audio segments within the correlation window

        Returns:
            CorrelationEvidence if a match is found, None otherwise
        """
        category = event.get('category', '').lower()
        expected_role = self._category_to_role(category)

        if not expected_role:
            return None

        event_time = event.get('relative_time', 0.0)
        if not event_time and 'timestamp' in event:
            ts = event['timestamp']
            if isinstance(ts, (int, float)):
                event_time = float(ts)

        event_id = event.get('event_id', str(id(event)))
        event_type = event.get('event_type', 'unknown')

        best_match = None
        best_delta = float('inf')

        for segment in recent_segments:
            segment_time = segment.get('start_time', segment.get('timestamp', 0.0))
            speaker_id = segment.get('speaker_id') or segment.get('speaker')

            if not speaker_id:
                continue

            time_delta_ms = abs(event_time - segment_time) * 1000

            if time_delta_ms <= self.correlation_window_ms and time_delta_ms < best_delta:
                proximity_factor = 1.0 - (time_delta_ms / self.correlation_window_ms)
                confidence_contribution = (
                    self.min_confidence_boost +
                    (self.max_confidence_boost - self.min_confidence_boost) * proximity_factor
                )

                best_match = CorrelationEvidence(
                    event_id=event_id,
                    event_type=event_type,
                    event_category=category,
                    speaker_id=speaker_id,
                    expected_role=expected_role.value,
                    time_delta_ms=time_delta_ms,
                    confidence_contribution=confidence_contribution,
                    event_timestamp=event_time,
                    segment_timestamp=segment_time,
                    segment_text=segment.get('text', '')[:100]
                )
                best_delta = time_delta_ms

        if best_match:
            self._correlations.append(best_match)
            self._speaker_evidence[best_match.speaker_id].append(best_match)

        return best_match

    def get_correlation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all correlations found.

        Returns:
            Dictionary with correlation statistics and evidence summary
        """
        if not self._correlations:
            return {
                'total_correlations': 0,
                'speakers_with_evidence': 0,
                'role_distribution': {},
                'event_types': {},
                'average_time_delta_ms': 0,
            }

        role_dist = defaultdict(int)
        event_types = defaultdict(int)
        total_delta = 0.0

        for corr in self._correlations:
            role_dist[corr.expected_role] += 1
            event_types[corr.event_type] += 1
            total_delta += corr.time_delta_ms

        return {
            'total_correlations': len(self._correlations),
            'speakers_with_evidence': len(self._speaker_evidence),
            'role_distribution': dict(role_dist),
            'event_types': dict(event_types),
            'average_time_delta_ms': total_delta / len(self._correlations),
            'speaker_evidence_counts': {
                speaker: len(evidence)
                for speaker, evidence in self._speaker_evidence.items()
            }
        }

    def get_speaker_evidence(self, speaker_id: str) -> List[CorrelationEvidence]:
        """
        Get all correlation evidence for a specific speaker.

        Args:
            speaker_id: The speaker to get evidence for

        Returns:
            List of CorrelationEvidence for this speaker
        """
        return self._speaker_evidence.get(speaker_id, [])

    def clear(self) -> None:
        """Clear all loaded data and correlations."""
        self.events = []
        self.transcripts = []
        self.mission_start = None
        self._correlations = []
        self._speaker_evidence = defaultdict(list)
