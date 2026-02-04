"""
Telemetry-Audio Correlator Module.

Correlates game telemetry events with audio segments to enhance speaker
role identification using multiple strategies:

1. EVENT DENSITY CORRELATION: Matches high activity in a category (e.g., many
   science scans) with speakers who frequently discuss that topic.

2. COMMAND vs REPORT DETECTION: Distinguishes between commanders (speech BEFORE
   action: "scan that") and operators (speech AFTER action: "scan complete").

3. NEGATIVE CORRELATION: Rules out roles when a speaker never mentions topics
   despite high activity (e.g., many helm events but speaker never says "heading").

This approach handles the reality that multiple consoles operate simultaneously
and the person pressing buttons may not be the one currently speaking.
"""

import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# Configuration from environment
CORRELATION_WINDOW_MS = int(os.getenv('CORRELATION_WINDOW_MS', '5000'))  # Wider window for density
MIN_CONFIDENCE_BOOST = float(os.getenv('MIN_CONFIDENCE_BOOST', '0.05'))
MAX_CONFIDENCE_BOOST = float(os.getenv('MAX_CONFIDENCE_BOOST', '0.25'))
DENSITY_CORRELATION_WEIGHT = float(os.getenv('DENSITY_CORRELATION_WEIGHT', '0.4'))
COMMAND_REPORT_WEIGHT = float(os.getenv('COMMAND_REPORT_WEIGHT', '0.4'))
NEGATIVE_CORRELATION_WEIGHT = float(os.getenv('NEGATIVE_CORRELATION_WEIGHT', '0.2'))


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


# Keywords for detecting speech TOPICS per role
# These are weighted towards Starship Horizons specific terminology
ROLE_TOPIC_KEYWORDS: Dict[BridgeRole, List[str]] = {
    BridgeRole.HELM: [
        "heading", "course", "speed", "throttle", "impulse", "warp",
        "bearing", "degrees", "port", "starboard", "come about", "full stop",
        "ahead", "reverse", "orbit", "intercept", "evasive", "maneuver",
        "navigation", "set course", "change heading", "slow", "faster"
    ],
    BridgeRole.TACTICAL: [
        "weapons", "fire", "target", "shields", "torpedo", "phaser",
        "lock", "arm", "disarm", "alert", "red alert", "yellow alert",
        "defensive", "attack", "engage", "disengage", "enemy", "hostile",
        "threat", "battle stations"
    ],
    BridgeRole.SCIENCE: [
        "scan", "sensor", "reading", "analysis", "detect", "anomaly",
        "signal", "frequency", "spectrum", "probe", "data", "contact",
        "life signs", "composition", "radiation", "scanning", "scanned",
        "object", "vessel", "ship", "container", "silkworm", "retrieve"
    ],
    BridgeRole.ENGINEERING: [
        "power", "reactor", "engine", "repair", "damage", "systems",
        "hull", "breach", "coolant", "warp core", "energy", "offline",
        "online", "reroute", "bypass", "overload", "operational", "status"
    ],
    BridgeRole.OPERATIONS: [
        "cargo", "transporter", "beam", "shuttle", "docking", "bay",
        "inventory", "supplies", "personnel", "away team", "transfer",
        "deploy", "retrieve", "aboard", "launched", "returned"
    ],
    BridgeRole.COMMUNICATIONS: [
        "hail", "channel", "frequency", "message", "transmission",
        "signal", "respond", "contact", "communication", "audio", "video",
        "incoming", "outgoing", "captain", "admiral", "starbase"
    ],
    BridgeRole.CAPTAIN: [
        "report", "status", "engage", "make it so", "on screen",
        "all hands", "battle stations", "stand down", "proceed",
        "acknowledged", "understood", "execute", "what do you see",
        "set a course", "fire", "evasive", "full speed"
    ],
}


# Command patterns - speech that indicates ORDERING an action (typically Captain)
COMMAND_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(scan|fire|engage|set|come to|bring us|take us|hail|open|raise|lower)\b', re.I),
    re.compile(r'\b(give me|show me|get me|find|locate|target|arm|disarm)\b', re.I),
    re.compile(r'\b(full|ahead|stop|reverse|evasive|intercept)\b', re.I),
    re.compile(r'\bwhat (do|is|are|can)\b', re.I),  # Questions often from command
]


# Report patterns - speech that indicates REPORTING a result (typically operator)
REPORT_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(complete|ready|done|finished|confirmed|standing by)\b', re.I),
    re.compile(r'\b(reading|showing|detecting|seeing|getting)\b', re.I),
    re.compile(r'\b(online|offline|operational|functional|damaged)\b', re.I),
    re.compile(r'\b(contact|signal|anomaly|vessel|ship) (detected|identified|on)\b', re.I),
    re.compile(r'\b(aye|copy|roger|understood|acknowledged)\b', re.I),
    re.compile(r'\b(captain|sir|commander)\b.*\b(we have|i have|there\'?s)\b', re.I),
]


# Station names that a Captain might address
STATION_NAMES: List[str] = [
    'science', 'helm', 'tactical', 'engineering', 'communications',
    'ops', 'operations', 'weapons', 'navigation', 'sensors'
]


# Captain-specific patterns - speech that indicates command authority
CAPTAIN_PATTERNS: List[re.Pattern] = [
    # Addressing stations directly (with optional words in between)
    re.compile(r'\b(science|helm|tactical|engineering|communications|ops)\b.{0,20}(what|scan|set|fire|hail|raise|lower|report)', re.I),
    # Asking for status/information
    re.compile(r'\bwhat.{0,10}(do you|can you|is|are|have|see)\b', re.I),
    re.compile(r'\b(give me|show me|get me|report|status)\b', re.I),
    # Giving direct orders (more flexible spacing)
    re.compile(r'\b(set\s+a?\s*course|full\s+speed|all\s+stop|engage|make\s+it\s+so|on\s+screen)\b', re.I),
    re.compile(r'\b(red\s+alert|yellow\s+alert|battle\s+stations|stand\s+down|all\s+hands)\b', re.I),
    # Directives to retrieve/deploy/need
    re.compile(r'\b(we.{0,10}(need|going)\s+to|retrieve|deploy|let.{0,5}s)\b', re.I),
    # Acknowledgment of reports (Captain receives reports)
    re.compile(r'\b(very\s+well|good|excellent|understood|proceed|carry\s+on|copy\s+that)\b', re.I),
]


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
class DensityCorrelation:
    """Correlation between event density and speaker topic density."""
    role: BridgeRole
    event_count: int
    event_density: float  # events per minute
    speaker_topic_counts: Dict[str, int] = field(default_factory=dict)
    speaker_topic_density: Dict[str, float] = field(default_factory=dict)
    speaker_correlations: Dict[str, float] = field(default_factory=dict)


@dataclass
class CommandReportEvidence:
    """Evidence of command vs report speech pattern."""
    speaker_id: str
    speech_type: str  # "command" or "report"
    text: str
    timestamp: float
    nearby_event_type: str
    nearby_event_time: float
    time_delta: float  # negative = speech before event (command), positive = after (report)
    inferred_role: str  # "Commander" for commands, actual role for reports


@dataclass
class NegativeEvidence:
    """Evidence that a speaker is NOT a particular role."""
    speaker_id: str
    ruled_out_role: BridgeRole
    reason: str
    event_count: int
    topic_mentions: int


@dataclass
class CaptainEvidence:
    """Evidence that a speaker is the Captain (no console, gives orders)."""
    speaker_id: str
    station_addresses: int  # Times they addressed other stations by name
    command_patterns: int   # Times they used command language
    questions_asked: int    # Times they asked for status/info
    utterance_count: int    # Total utterances (Captains talk a lot)
    has_console_correlation: bool  # False = likely Captain
    captain_score: float    # Composite likelihood score


@dataclass
class SmartCorrelationResult:
    """Combined result from all correlation strategies."""
    speaker_id: str
    density_boost: float
    density_evidence: List[str]
    command_report_boost: float
    command_report_evidence: List[CommandReportEvidence]
    negative_adjustments: Dict[str, float]
    negative_evidence: List[NegativeEvidence]
    captain_evidence: Optional[CaptainEvidence]
    captain_boost: float
    total_boost: float
    methodology_note: str


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
    # New fields for smart correlation
    density_boost: float = 0.0
    command_report_boost: float = 0.0
    negative_adjustment: float = 0.0
    captain_boost: float = 0.0
    smart_evidence: Optional[SmartCorrelationResult] = None


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

    # =========================================================================
    # SMART CORRELATION METHODS
    # =========================================================================

    def correlate_smart(self) -> Dict[str, SmartCorrelationResult]:
        """
        Run all smart correlation strategies and combine results.

        This method uses three complementary approaches:
        1. Event density correlation - matches event frequency with speech topics
        2. Command vs report detection - identifies commanders vs operators
        3. Negative correlation - rules out roles based on absent topics

        Returns:
            Dict mapping speaker_id to SmartCorrelationResult
        """
        if not self.transcripts:
            logger.warning("No transcripts loaded for smart correlation")
            return {}

        # Note: events may be empty (e.g., when recording without telemetry)
        # Captain detection still works with just transcripts

        # Calculate recording duration
        duration_seconds = self._get_recording_duration()
        if duration_seconds <= 0:
            duration_seconds = 120  # Default 2 minutes

        logger.info(
            f"Running smart correlation on {len(self.events)} events and "
            f"{len(self.transcripts)} transcripts ({duration_seconds:.0f}s duration)"
        )

        # Run each correlation strategy
        density_results = self._correlate_by_density(duration_seconds)
        command_report_results = self._detect_command_vs_report()
        negative_results = self._correlate_negative(duration_seconds)
        captain_results = self._detect_captain(density_results)

        # Combine results per speaker
        all_speakers = set()
        for t in self.transcripts:
            speaker = t.get('speaker_id') or t.get('speaker')
            if speaker:
                all_speakers.add(speaker)

        smart_results: Dict[str, SmartCorrelationResult] = {}

        for speaker_id in all_speakers:
            # Density boost
            density_boost = 0.0
            density_evidence = []
            if speaker_id in density_results:
                for role, correlation in density_results[speaker_id].items():
                    if correlation > 0.1:  # Lower threshold to capture partial correlations
                        boost = correlation * DENSITY_CORRELATION_WEIGHT * self.max_confidence_boost
                        density_boost = max(density_boost, boost)
                        density_evidence.append(
                            f"{role.value}: {correlation:.0%} topic-event correlation"
                        )

            # Command/report boost
            cr_boost = 0.0
            cr_evidence = []
            if speaker_id in command_report_results:
                cr_data = command_report_results[speaker_id]
                # More reports = likely operator, more commands = likely commander
                report_count = cr_data.get('reports', 0)
                command_count = cr_data.get('commands', 0)
                cr_evidence = cr_data.get('evidence', [])

                if report_count > command_count and report_count >= 2:
                    # Likely an operator - boost their detected role
                    cr_boost = min(report_count * 0.03, COMMAND_REPORT_WEIGHT * self.max_confidence_boost)
                elif command_count > report_count and command_count >= 2:
                    # Likely a commander - will be handled in role assignment
                    cr_boost = min(command_count * 0.02, COMMAND_REPORT_WEIGHT * self.max_confidence_boost)

            # Negative adjustments
            negative_adj = {}
            negative_ev = []
            if speaker_id in negative_results:
                for neg in negative_results[speaker_id]:
                    penalty = -0.1 * NEGATIVE_CORRELATION_WEIGHT
                    negative_adj[neg.ruled_out_role.value] = penalty
                    negative_ev.append(neg)

            # Captain detection
            captain_boost = 0.0
            captain_ev = None
            if speaker_id in captain_results:
                captain_ev = captain_results[speaker_id]
                # Captain boost based on score (max 15% boost)
                if captain_ev.captain_score >= 1.0:
                    captain_boost = 0.15
                elif captain_ev.captain_score >= 0.5:
                    captain_boost = captain_ev.captain_score * 0.10
                else:
                    captain_boost = captain_ev.captain_score * 0.05

            total_boost = density_boost + cr_boost + captain_boost
            # Apply negative adjustments later per-role

            methodology = self._build_smart_methodology(
                density_boost, density_evidence,
                cr_boost, cr_evidence,
                negative_adj, negative_ev,
                captain_boost, captain_ev
            )

            smart_results[speaker_id] = SmartCorrelationResult(
                speaker_id=speaker_id,
                density_boost=density_boost,
                density_evidence=density_evidence,
                command_report_boost=cr_boost,
                command_report_evidence=cr_evidence,
                negative_adjustments=negative_adj,
                negative_evidence=negative_ev,
                captain_evidence=captain_ev,
                captain_boost=captain_boost,
                total_boost=total_boost,
                methodology_note=methodology
            )

        logger.info(f"Smart correlation complete: {len(smart_results)} speakers analyzed")
        return smart_results

    def _get_recording_duration(self) -> float:
        """Calculate recording duration from transcripts."""
        if not self.transcripts:
            return 0.0

        max_time = 0.0
        for t in self.transcripts:
            end_time = t.get('end_time', t.get('start_time', 0))
            max_time = max(max_time, end_time)
        return max_time

    def _correlate_by_density(
        self,
        duration_seconds: float
    ) -> Dict[str, Dict[BridgeRole, float]]:
        """
        Correlate event density per category with speaker topic density.

        If there are many science scans and a speaker frequently uses science
        keywords, there's a correlation suggesting they're at the science console.

        Args:
            duration_seconds: Recording duration for density calculation

        Returns:
            Dict mapping speaker_id to {role: correlation_score}
        """
        # Count events per role category
        role_event_counts: Dict[BridgeRole, int] = defaultdict(int)
        for event in self.events:
            category = event.get('category', '').lower()
            role = self._category_to_role(category)
            if role:
                role_event_counts[role] += 1

        # Calculate event density (events per minute)
        duration_minutes = max(duration_seconds / 60, 0.5)
        role_event_density: Dict[BridgeRole, float] = {
            role: count / duration_minutes
            for role, count in role_event_counts.items()
        }

        # Count topic keywords per speaker
        speaker_topic_counts: Dict[str, Dict[BridgeRole, int]] = defaultdict(lambda: defaultdict(int))
        speaker_total_words: Dict[str, int] = defaultdict(int)

        for t in self.transcripts:
            speaker = t.get('speaker_id') or t.get('speaker')
            text = t.get('text', '').lower()
            if not speaker or not text:
                continue

            words = text.split()
            speaker_total_words[speaker] += len(words)

            for role, keywords in ROLE_TOPIC_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        speaker_topic_counts[speaker][role] += 1

        # Calculate correlation: speakers with high topic density for high-activity roles
        speaker_correlations: Dict[str, Dict[BridgeRole, float]] = defaultdict(dict)

        for speaker, topic_counts in speaker_topic_counts.items():
            total_words = max(speaker_total_words[speaker], 1)

            for role, topic_count in topic_counts.items():
                event_density = role_event_density.get(role, 0)
                if event_density < 0.3:  # Ignore very low-activity roles
                    continue

                # Topic density: mentions per 100 words
                topic_density = (topic_count / total_words) * 100

                # Correlation: high topic density + high event density = strong match
                # Normalize event density (cap at 10 events/min)
                normalized_event_density = min(event_density / 10, 1.0)
                # Normalize topic density (cap at 20%)
                normalized_topic_density = min(topic_density / 20, 1.0)

                # Correlation score: geometric mean
                correlation = math.sqrt(normalized_event_density * normalized_topic_density)
                speaker_correlations[speaker][role] = correlation

        logger.debug(f"Density correlation: {dict(speaker_correlations)}")
        return dict(speaker_correlations)

    def _detect_command_vs_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect command patterns (before events) vs report patterns (after events).

        Commands like "scan that target" followed by a scan event suggest the
        speaker is a commander. Reports like "scan complete" after a scan event
        suggest the speaker is the operator.

        Returns:
            Dict mapping speaker_id to {commands: count, reports: count, evidence: [...]}
        """
        results: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'commands': 0, 'reports': 0, 'evidence': []}
        )

        # Use a wider window for command/report detection (5 seconds)
        window_seconds = 5.0

        for event in self.events:
            event_time = event.get('relative_time', event.get('timestamp', 0))
            if isinstance(event_time, datetime):
                continue  # Skip non-numeric timestamps

            event_type = event.get('event_type', 'unknown')
            category = event.get('category', '').lower()
            role = self._category_to_role(category)
            if not role:
                continue

            # Find nearby speech
            for t in self.transcripts:
                speaker = t.get('speaker_id') or t.get('speaker')
                text = t.get('text', '')
                segment_time = t.get('relative_time', t.get('start_time', 0))

                if not speaker or not text:
                    continue

                time_delta = segment_time - event_time  # negative = speech before event

                if abs(time_delta) > window_seconds:
                    continue

                # Check for command patterns (speech BEFORE event)
                if -window_seconds <= time_delta <= 0.5:  # Speech before or just after
                    for pattern in COMMAND_PATTERNS:
                        if pattern.search(text):
                            results[speaker]['commands'] += 1
                            evidence = CommandReportEvidence(
                                speaker_id=speaker,
                                speech_type="command",
                                text=text[:80],
                                timestamp=segment_time,
                                nearby_event_type=event_type,
                                nearby_event_time=event_time,
                                time_delta=time_delta,
                                inferred_role="Commander"
                            )
                            results[speaker]['evidence'].append(evidence)
                            break

                # Check for report patterns (speech AFTER event)
                if -0.5 <= time_delta <= window_seconds:  # Speech after or just before
                    for pattern in REPORT_PATTERNS:
                        if pattern.search(text):
                            results[speaker]['reports'] += 1
                            evidence = CommandReportEvidence(
                                speaker_id=speaker,
                                speech_type="report",
                                text=text[:80],
                                timestamp=segment_time,
                                nearby_event_type=event_type,
                                nearby_event_time=event_time,
                                time_delta=time_delta,
                                inferred_role=role.value
                            )
                            results[speaker]['evidence'].append(evidence)
                            break

        logger.debug(f"Command/report detection: {len(results)} speakers analyzed")
        return dict(results)

    def _correlate_negative(
        self,
        duration_seconds: float
    ) -> Dict[str, List[NegativeEvidence]]:
        """
        Find negative correlations - roles that speakers are unlikely to have.

        If there are many helm events (heading changes, speed changes) but a
        speaker NEVER mentions helm-related topics, they're probably not the
        helm officer.

        Args:
            duration_seconds: Recording duration

        Returns:
            Dict mapping speaker_id to list of NegativeEvidence
        """
        results: Dict[str, List[NegativeEvidence]] = defaultdict(list)

        # Count events per role
        role_event_counts: Dict[BridgeRole, int] = defaultdict(int)
        for event in self.events:
            category = event.get('category', '').lower()
            role = self._category_to_role(category)
            if role:
                role_event_counts[role] += 1

        # Find high-activity roles (at least 3 events)
        active_roles = {role for role, count in role_event_counts.items() if count >= 3}

        if not active_roles:
            return dict(results)

        # Count topic mentions per speaker
        speaker_topic_counts: Dict[str, Dict[BridgeRole, int]] = defaultdict(lambda: defaultdict(int))
        speaker_utterances: Dict[str, int] = defaultdict(int)

        for t in self.transcripts:
            speaker = t.get('speaker_id') or t.get('speaker')
            text = t.get('text', '').lower()
            if not speaker or not text:
                continue

            speaker_utterances[speaker] += 1

            for role, keywords in ROLE_TOPIC_KEYWORDS.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        speaker_topic_counts[speaker][role] += 1
                        break  # Count once per utterance per role

        # Find speakers with significant speech but zero mentions of active roles
        for speaker, utterance_count in speaker_utterances.items():
            if utterance_count < 3:  # Need minimum speech sample
                continue

            for role in active_roles:
                if role == BridgeRole.CAPTAIN:
                    continue  # Captain doesn't have specific console events

                topic_mentions = speaker_topic_counts[speaker].get(role, 0)
                event_count = role_event_counts[role]

                # If high activity but zero mentions, negative correlation
                if topic_mentions == 0 and event_count >= 5:
                    evidence = NegativeEvidence(
                        speaker_id=speaker,
                        ruled_out_role=role,
                        reason=f"{event_count} {role.value} events but 0 topic mentions in {utterance_count} utterances",
                        event_count=event_count,
                        topic_mentions=0
                    )
                    results[speaker].append(evidence)

        logger.debug(f"Negative correlation: {sum(len(v) for v in results.values())} exclusions found")
        return dict(results)

    def _detect_captain(
        self,
        density_results: Dict[str, Dict[BridgeRole, float]]
    ) -> Dict[str, CaptainEvidence]:
        """
        Detect speakers who are likely the Captain.

        The Captain is unique: they give orders but don't press buttons.
        Indicators:
        1. Addresses other stations by name ("Science, scan that")
        2. Uses command/imperative language
        3. Asks questions for status ("What do you see?")
        4. High speech volume
        5. No console telemetry correlation (or very low)

        Args:
            density_results: Results from density correlation (to check console correlation)

        Returns:
            Dict mapping speaker_id to CaptainEvidence
        """
        results: Dict[str, CaptainEvidence] = {}

        # Gather speech data per speaker
        speaker_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'texts': [], 'station_addresses': 0, 'command_patterns': 0, 'questions': 0}
        )

        for t in self.transcripts:
            speaker = t.get('speaker_id') or t.get('speaker')
            text = t.get('text', '').lower()
            if not speaker or not text:
                continue

            speaker_data[speaker]['texts'].append(text)

            # Check for station addressing ("Science, scan that")
            for station in STATION_NAMES:
                if station in text:
                    # Check if followed by command-like text
                    station_idx = text.find(station)
                    after_station = text[station_idx + len(station):]
                    if any(cmd in after_station for cmd in ['scan', 'set', 'fire', 'hail', 'what', 'report', 'status']):
                        speaker_data[speaker]['station_addresses'] += 1
                        break

            # Check for Captain-specific patterns
            for pattern in CAPTAIN_PATTERNS:
                if pattern.search(text):
                    speaker_data[speaker]['command_patterns'] += 1
                    break

            # Check for questions (Captains ask for status)
            if '?' in text or any(text.startswith(q) for q in ['what ', 'where ', 'how ', 'is ', 'are ', 'do ', 'can ']):
                speaker_data[speaker]['questions'] += 1

        # Calculate Captain likelihood for each speaker
        for speaker, data in speaker_data.items():
            utterance_count = len(data['texts'])
            if utterance_count < 2:
                continue

            # Check if speaker has console correlation
            has_console_correlation = False
            max_correlation = 0.0
            if speaker in density_results:
                for role, corr in density_results[speaker].items():
                    if role != BridgeRole.CAPTAIN and corr > 0.15:
                        has_console_correlation = True
                        max_correlation = max(max_correlation, corr)

            # Captain score calculation
            # - Station addresses are very captain-like (weight: 3)
            # - Command patterns suggest leadership (weight: 2)
            # - Questions suggest information gathering (weight: 1)
            # - Penalty if they have strong console correlation
            raw_score = (
                data['station_addresses'] * 3.0 +
                data['command_patterns'] * 2.0 +
                data['questions'] * 1.0
            )

            # Normalize by utterance count
            normalized_score = raw_score / utterance_count

            # Apply penalty for console correlation (operators don't usually give orders)
            if has_console_correlation:
                normalized_score *= (1.0 - max_correlation * 0.5)

            # Boost for high speech volume (Captains talk a lot)
            if utterance_count >= 5:
                normalized_score *= 1.2
            elif utterance_count >= 8:
                normalized_score *= 1.4

            # Create evidence if score is meaningful
            if normalized_score > 0.3 or data['station_addresses'] >= 2:
                results[speaker] = CaptainEvidence(
                    speaker_id=speaker,
                    station_addresses=data['station_addresses'],
                    command_patterns=data['command_patterns'],
                    questions_asked=data['questions'],
                    utterance_count=utterance_count,
                    has_console_correlation=has_console_correlation,
                    captain_score=normalized_score
                )

        logger.debug(f"Captain detection: {len(results)} potential captains found")
        return results

    def _build_smart_methodology(
        self,
        density_boost: float,
        density_evidence: List[str],
        cr_boost: float,
        cr_evidence: List[CommandReportEvidence],
        negative_adj: Dict[str, float],
        negative_ev: List[NegativeEvidence],
        captain_boost: float = 0.0,
        captain_ev: Optional[CaptainEvidence] = None
    ) -> str:
        """Build a human-readable methodology note."""
        parts = []

        if captain_boost > 0 and captain_ev:
            parts.append(
                f"Captain indicators: +{captain_boost:.1%} "
                f"({captain_ev.station_addresses} station addresses, "
                f"{captain_ev.command_patterns} commands, "
                f"{captain_ev.questions_asked} questions)"
            )

        if density_boost > 0:
            parts.append(f"Density correlation: +{density_boost:.1%} ({', '.join(density_evidence[:2])})")

        if cr_boost > 0:
            report_count = len([e for e in cr_evidence if e.speech_type == 'report'])
            command_count = len([e for e in cr_evidence if e.speech_type == 'command'])
            parts.append(f"Speech patterns: +{cr_boost:.1%} ({command_count} commands, {report_count} reports)")

        if negative_ev:
            excluded = [e.ruled_out_role.value for e in negative_ev]
            parts.append(f"Ruled out: {', '.join(excluded)}")

        return " | ".join(parts) if parts else "No telemetry correlation evidence"

    def update_role_confidences_smart(
        self,
        existing_roles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, RoleConfidenceUpdate]:
        """
        Update role confidences using smart correlation strategies.

        This is the main entry point for the improved correlation approach.

        Args:
            existing_roles: Dict mapping speaker_id to role info

        Returns:
            Dict mapping speaker_id to RoleConfidenceUpdate
        """
        smart_results = self.correlate_smart()
        updates: Dict[str, RoleConfidenceUpdate] = {}

        for speaker_id, role_info in existing_roles.items():
            base_confidence = role_info.get('confidence', 0.5)
            assigned_role = role_info.get('role', 'Crew Member')

            smart = smart_results.get(speaker_id)
            if not smart:
                # No smart correlation data - use original confidence
                updates[speaker_id] = RoleConfidenceUpdate(
                    speaker_id=speaker_id,
                    role=assigned_role,
                    base_confidence=base_confidence,
                    boosted_confidence=base_confidence,
                    evidence_count=0,
                    telemetry_boost=0.0,
                    methodology_note="No telemetry correlation data"
                )
                continue

            # Calculate total boost
            total_boost = smart.total_boost

            # Apply negative adjustment if this role was ruled out
            if assigned_role in smart.negative_adjustments:
                total_boost += smart.negative_adjustments[assigned_role]

            # Check if this speaker should be promoted to Captain
            # Captain override: if captain score is high AND they don't have strong console correlation
            final_role = assigned_role
            captain_boost_applied = 0.0
            if smart.captain_evidence and smart.captain_boost > 0:
                captain_ev = smart.captain_evidence
                # Strong captain evidence: promote to Captain role
                if (captain_ev.captain_score >= 1.0 or
                    (captain_ev.station_addresses >= 2 and captain_ev.command_patterns >= 1)):
                    # Only promote if not already a strong operator match
                    if smart.density_boost < 0.05 or not captain_ev.has_console_correlation:
                        final_role = BridgeRole.CAPTAIN.value
                        captain_boost_applied = smart.captain_boost
                        # Recalculate confidence for Captain role
                        total_boost = smart.captain_boost + smart.command_report_boost
                        logger.info(
                            f"Promoting {speaker_id} to Captain (score={captain_ev.captain_score:.2f}, "
                            f"addresses={captain_ev.station_addresses}, commands={captain_ev.command_patterns})"
                        )

            # Cap the boost
            total_boost = max(-0.2, min(total_boost, self.max_confidence_boost + 0.1))  # Allow slightly higher for Captain

            boosted_confidence = min(1.0, max(0.1, base_confidence + total_boost))

            updates[speaker_id] = RoleConfidenceUpdate(
                speaker_id=speaker_id,
                role=final_role,
                base_confidence=base_confidence,
                boosted_confidence=boosted_confidence,
                evidence_count=len(smart.density_evidence) + len(smart.command_report_evidence),
                telemetry_boost=total_boost,
                methodology_note=smart.methodology_note,
                density_boost=smart.density_boost,
                command_report_boost=smart.command_report_boost,
                negative_adjustment=smart.negative_adjustments.get(assigned_role, 0.0),
                captain_boost=captain_boost_applied,
                smart_evidence=smart
            )

        return updates

    def cross_reference_speech_action(self) -> Dict[str, Any]:
        """
        Cross-reference crew speech with game actions to find discrepancies.

        Identifies cases where:
        - Crew said they'd do something but telemetry shows it didn't happen
        - Game actions occurred that no one discussed
        - Speech and action are well-aligned (positive reinforcement)

        Returns:
            Dictionary with aligned actions, speech-only intentions,
            action-only events, and an alignment score
        """
        if not self.transcripts:
            return {
                'aligned': [],
                'speech_only': [],
                'action_only': [],
                'alignment_score': 0.0,
                'total_speech_intentions': 0,
                'total_game_actions': 0,
            }

        # Detect speech intentions (crew stating they will do something)
        intention_patterns = [
            (re.compile(r'\b(set|setting)\s+(a\s+)?course\b', re.I), 'navigation', 'Set course'),
            (re.compile(r'\b(fire|firing|arm|arming)\s+(torpedo|phaser|weapon)', re.I), 'tactical', 'Weapons engagement'),
            (re.compile(r'\b(raise|raising|lower|lowering)\s+shields?\b', re.I), 'tactical', 'Shield adjustment'),
            (re.compile(r'\b(scan|scanning)\b', re.I), 'science', 'Sensor scan'),
            (re.compile(r'\b(hail|hailing|open.{0,10}channel)\b', re.I), 'communications', 'Communications'),
            (re.compile(r'\b(dock|docking)\b', re.I), 'operations', 'Docking operation'),
            (re.compile(r'\b(red|yellow)\s+alert\b', re.I), 'tactical', 'Alert change'),
            (re.compile(r'\b(warp|jump|engage)\b', re.I), 'helm', 'Warp/navigation'),
            (re.compile(r'\b(transfer|load|deploy)\b', re.I), 'operations', 'Resource operation'),
            (re.compile(r'\b(retrieve|pick.{0,5}up|collect)\b', re.I), 'operations', 'Retrieval operation'),
        ]

        # Extract speech intentions with timestamps
        speech_intentions = []
        for t in self.transcripts:
            text = t.get('text', '')
            timestamp = t.get('relative_time', t.get('start_time', 0))
            speaker = t.get('speaker_id') or t.get('speaker', 'unknown')
            if not text:
                continue

            for pattern, category, action_name in intention_patterns:
                if pattern.search(text):
                    speech_intentions.append({
                        'speaker': speaker,
                        'text': text[:120],
                        'timestamp': timestamp,
                        'category': category,
                        'action_name': action_name,
                    })
                    break  # One intention per utterance

        # Build game action list with timestamps
        game_actions = []
        for event in self.events:
            category = event.get('category', '').lower()
            event_type = event.get('event_type', 'unknown')
            timestamp = event.get('relative_time', event.get('timestamp', 0))
            if isinstance(timestamp, datetime):
                continue
            data = event.get('data', {})

            # Build description
            description = event_type.replace('_', ' ').title()
            message = data.get('message') or data.get('Message', '')
            if message:
                description = f"{description}: {str(message)[:80]}"

            game_actions.append({
                'event_type': event_type,
                'category': category,
                'timestamp': timestamp,
                'description': description,
            })

        # Match speech intentions to game actions (within 15-second window)
        match_window = 15.0  # seconds
        aligned = []
        matched_speech_indices = set()
        matched_action_indices = set()

        for si_idx, intention in enumerate(speech_intentions):
            for ga_idx, action in enumerate(game_actions):
                if ga_idx in matched_action_indices:
                    continue

                # Check category alignment
                cat_match = (
                    intention['category'] == action['category'] or
                    (intention['category'] in ('helm', 'navigation') and
                     action['category'] in ('helm', 'navigation')) or
                    (intention['category'] == 'tactical' and
                     action['category'] in ('tactical', 'combat', 'weapons', 'defensive'))
                )

                if not cat_match:
                    continue

                # Check time proximity
                time_delta = action['timestamp'] - intention['timestamp']
                if -5.0 <= time_delta <= match_window:
                    aligned.append({
                        'speech': intention['text'],
                        'speaker': intention['speaker'],
                        'speech_time': intention['timestamp'],
                        'action': action['description'],
                        'action_time': action['timestamp'],
                        'time_delta': round(time_delta, 1),
                        'category': intention['category'],
                    })
                    matched_speech_indices.add(si_idx)
                    matched_action_indices.add(ga_idx)
                    break

        # Find unmatched speech intentions
        speech_only = [
            intention for i, intention in enumerate(speech_intentions)
            if i not in matched_speech_indices
        ]

        # Find significant unmatched game actions
        significant_types = {
            'alert_change', 'weapons_fire', 'shield_change', 'damage_report',
            'warp_engage', 'warp_disengage', 'docking_complete', 'scan_complete',
        }
        action_only = [
            action for i, action in enumerate(game_actions)
            if i not in matched_action_indices and
            action['event_type'] in significant_types
        ]

        # Calculate alignment score
        total_checkable = len(speech_intentions) + len(
            [a for a in game_actions if a['event_type'] in significant_types]
        )
        total_aligned = len(aligned) * 2  # Each alignment covers one from each side
        alignment_score = (
            total_aligned / total_checkable if total_checkable > 0 else 0.0
        )

        return {
            'aligned': aligned[:20],
            'speech_only': speech_only[:10],
            'action_only': action_only[:10],
            'alignment_score': round(min(1.0, alignment_score), 3),
            'total_speech_intentions': len(speech_intentions),
            'total_game_actions': len(game_actions),
            'total_aligned': len(aligned),
        }

    def get_smart_correlation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of smart correlation analysis.

        Returns:
            Dictionary with correlation statistics
        """
        smart_results = self.correlate_smart()

        if not smart_results:
            return {
                'strategy': 'smart',
                'speakers_analyzed': 0,
                'density_correlations': 0,
                'command_patterns': 0,
                'report_patterns': 0,
                'negative_exclusions': 0,
                'captain_detections': 0,
            }

        total_commands = sum(
            len([e for e in r.command_report_evidence if e.speech_type == 'command'])
            for r in smart_results.values()
        )
        total_reports = sum(
            len([e for e in r.command_report_evidence if e.speech_type == 'report'])
            for r in smart_results.values()
        )
        total_negative = sum(len(r.negative_evidence) for r in smart_results.values())
        total_density = sum(1 for r in smart_results.values() if r.density_boost > 0)
        total_captains = sum(1 for r in smart_results.values() if r.captain_evidence and r.captain_boost > 0)

        return {
            'strategy': 'smart',
            'speakers_analyzed': len(smart_results),
            'density_correlations': total_density,
            'command_patterns': total_commands,
            'report_patterns': total_reports,
            'negative_exclusions': total_negative,
            'captain_detections': total_captains,
            'event_count': len(self.events),
            'transcript_count': len(self.transcripts),
        }
