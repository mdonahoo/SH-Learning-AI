"""
Mission phase analyzer for temporal breakdown of missions.

Detects mission phases based on communication patterns and content,
providing chronological analysis with per-phase statistics.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class PhaseDefinition:
    """Definition of a mission phase with detection patterns."""
    name: str
    display_name: str
    patterns: List[str]
    priority: int  # Lower = higher priority for phase assignment


# Standard mission phase definitions
PHASE_DEFINITIONS = [
    PhaseDefinition(
        name="initial_status",
        display_name="Initial Status and Departure",
        patterns=[
            r"(?i)(systems|status|check|ready|online|standing by)",
            r"(?i)(departure|leaving|undocking|launch)",
            r"(?i)(all stations|report|nominal)",
        ],
        priority=1
    ),
    PhaseDefinition(
        name="navigation",
        display_name="Navigation and Exploration",
        patterns=[
            r"(?i)(heading|course|set course|eta|approach)",
            r"(?i)(within \d+|kilometers|distance)",
            r"(?i)(navigate|navigation|transit|traveling)",
            r"(?i)(sector|coordinates|waypoint)",
        ],
        priority=2
    ),
    PhaseDefinition(
        name="objective_work",
        display_name="Objective Operations",
        patterns=[
            r"(?i)(scan|scanning|collect|retrieve|pickup)",
            r"(?i)(transferring|transfer|cargo|container)",
            r"(?i)(dock|docking|hailing|transmit)",
            r"(?i)(research|investigate|analyze)",
        ],
        priority=3
    ),
    PhaseDefinition(
        name="tactical_prep",
        display_name="Weapons Preparation and Contact",
        patterns=[
            r"(?i)(weapons|arm|arming|load|loading)",
            r"(?i)(shields|power to|energy)",
            r"(?i)(target|targeting|lock on)",
            r"(?i)(contact|detected|enemy|hostile)",
        ],
        priority=4
    ),
    PhaseDefinition(
        name="diplomatic",
        display_name="Diplomatic and Tactical Engagement",
        patterns=[
            r"(?i)(hailing|negotiat|trade|offer)",
            r"(?i)(dominion|federation|alliance)",
            r"(?i)(terms|demand|agree|refuse)",
            r"(?i)(standoff|tension|diplomatic)",
        ],
        priority=5
    ),
    PhaseDefinition(
        name="combat",
        display_name="Combat Engagement",
        patterns=[
            r"(?i)(fire|firing|launch|missile|torpedo)",
            r"(?i)(hit|damage|impact|taking fire)",
            r"(?i)(evasive|maneuver|shields up)",
            r"(?i)(attack|engage|destroy)",
        ],
        priority=6
    ),
    PhaseDefinition(
        name="escape",
        display_name="Extraction and Escape",
        patterns=[
            r"(?i)(retreat|escape|get out|run)",
            r"(?i)(warp|jump|hyperspace)",
            r"(?i)(fall back|withdraw|disengage)",
        ],
        priority=7
    ),
    PhaseDefinition(
        name="conclusion",
        display_name="Mission Conclusion",
        patterns=[
            r"(?i)(complete|finished|done|mission accomplished)",
            r"(?i)(return|dock|space dock|starbase)",
            r"(?i)(debrief|end|wrap up)",
        ],
        priority=8
    ),
]


@dataclass
class PhaseAnalysis:
    """Analysis results for a single mission phase."""
    phase_number: int
    phase_name: str
    display_name: str
    start_time: str
    end_time: str
    duration_minutes: float
    utterance_count: int
    primary_speakers: List[Tuple[str, int]]  # (speaker, count) pairs
    notable_communications: List[Dict[str, Any]]
    summary: str


class MissionPhaseAnalyzer:
    """
    Analyzes mission transcripts to identify and characterize mission phases.

    Uses keyword clustering and temporal analysis to detect phase transitions
    and generate per-phase statistics matching the example report format.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        events: List[Dict[str, Any]] = None,
        phase_definitions: List[PhaseDefinition] = None,
        min_phase_duration_minutes: float = 3.0,
        min_phase_utterances: int = 10
    ):
        """
        Initialize the phase analyzer.

        Args:
            transcripts: List of transcript dictionaries with 'timestamp', 'speaker', 'text'
            events: Optional list of game events
            phase_definitions: Optional custom phase definitions
            min_phase_duration_minutes: Minimum duration to consider a phase
            min_phase_utterances: Minimum utterances to form a phase
        """
        self.transcripts = sorted(
            transcripts,
            key=lambda x: self._parse_timestamp(x.get('timestamp', ''))
        )
        self.events = events or []
        self.phase_definitions = phase_definitions or PHASE_DEFINITIONS
        self.min_phase_duration = min_phase_duration_minutes
        self.min_phase_utterances = min_phase_utterances

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse various timestamp formats to datetime."""
        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, (int, float)):
            # Assume seconds since some epoch
            return datetime.fromtimestamp(ts)

        if isinstance(ts, str):
            # Try ISO format
            try:
                if 'T' in ts:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                # Try time-only format
                return datetime.strptime(ts, '%H:%M:%S')
            except (ValueError, TypeError):
                pass

        return datetime.min

    def _classify_utterance(self, text: str) -> Optional[str]:
        """Classify an utterance into a phase type based on patterns."""
        matches = {}

        for phase_def in self.phase_definitions:
            score = 0
            for pattern in phase_def.patterns:
                if re.search(pattern, text):
                    score += 1
            if score > 0:
                matches[phase_def.name] = (score, phase_def.priority)

        if not matches:
            return None

        # Return the phase with highest score, using priority as tiebreaker
        best = max(matches.items(), key=lambda x: (x[1][0], -x[1][1]))
        return best[0]

    def analyze_phases(self) -> List[PhaseAnalysis]:
        """
        Analyze transcripts and identify mission phases.

        Returns:
            List of PhaseAnalysis objects in chronological order
        """
        if not self.transcripts:
            return []

        # Classify each utterance
        classified = []
        for t in self.transcripts:
            phase_type = self._classify_utterance(t.get('text', ''))
            classified.append({
                **t,
                'phase_type': phase_type or 'unclassified'
            })

        # Use sliding window to detect phase transitions
        window_size = max(5, len(classified) // 20)  # ~5% of transcript
        phases = self._detect_phases(classified, window_size)

        # Generate phase analyses
        results = []
        for i, phase in enumerate(phases, 1):
            analysis = self._analyze_single_phase(i, phase)
            results.append(analysis)

        return results

    def _detect_phases(
        self,
        classified: List[Dict],
        window_size: int
    ) -> List[List[Dict]]:
        """Detect phase boundaries using sliding window analysis."""
        if len(classified) < window_size:
            # Too few utterances, treat as single phase
            return [classified]

        phases = []
        current_phase = []
        current_dominant = None

        for i, utterance in enumerate(classified):
            # Look at window around current position
            start = max(0, i - window_size // 2)
            end = min(len(classified), i + window_size // 2)
            window = classified[start:end]

            # Find dominant phase type in window
            type_counts = Counter(u['phase_type'] for u in window if u['phase_type'] != 'unclassified')
            if type_counts:
                dominant = type_counts.most_common(1)[0][0]
            else:
                dominant = current_dominant or 'unclassified'

            # Check for phase transition
            if current_dominant is None:
                current_dominant = dominant
            elif dominant != current_dominant:
                # Phase transition detected
                if (len(current_phase) >= self.min_phase_utterances):
                    phases.append(current_phase)
                    current_phase = []
                current_dominant = dominant

            current_phase.append(utterance)

        # Add final phase
        if current_phase:
            phases.append(current_phase)

        # Merge very short phases into neighbors
        phases = self._merge_short_phases(phases)

        return phases

    def _merge_short_phases(self, phases: List[List[Dict]]) -> List[List[Dict]]:
        """Merge phases that are too short into their neighbors."""
        if len(phases) <= 1:
            return phases

        merged = []
        i = 0

        while i < len(phases):
            phase = phases[i]

            # Check if phase is too short
            if len(phase) < self.min_phase_utterances and merged:
                # Merge into previous phase
                merged[-1].extend(phase)
            elif len(phase) < self.min_phase_utterances and i < len(phases) - 1:
                # Merge into next phase
                phases[i + 1] = phase + phases[i + 1]
            else:
                merged.append(phase)

            i += 1

        return merged

    def _analyze_single_phase(
        self,
        phase_number: int,
        utterances: List[Dict]
    ) -> PhaseAnalysis:
        """Analyze a single phase and generate statistics."""
        if not utterances:
            return self._empty_phase_analysis(phase_number)

        # Get timestamps
        timestamps = [self._parse_timestamp(u.get('timestamp', '')) for u in utterances]
        valid_timestamps = [t for t in timestamps if t != datetime.min]

        if valid_timestamps:
            start_time = min(valid_timestamps)
            end_time = max(valid_timestamps)
            duration = (end_time - start_time).total_seconds() / 60
        else:
            start_time = datetime.min
            end_time = datetime.min
            duration = 0

        # Count speakers
        speaker_counts = Counter(
            u.get('speaker') or u.get('speaker_id') or 'unknown'
            for u in utterances
        )
        primary_speakers = speaker_counts.most_common(3)

        # Determine phase type
        type_counts = Counter(
            u.get('phase_type', 'unclassified')
            for u in utterances
            if u.get('phase_type') != 'unclassified'
        )

        if type_counts:
            dominant_type = type_counts.most_common(1)[0][0]
            phase_def = next(
                (p for p in self.phase_definitions if p.name == dominant_type),
                None
            )
            display_name = phase_def.display_name if phase_def else dominant_type.replace('_', ' ').title()
        else:
            display_name = "General Operations"
            dominant_type = "general"

        # Select notable communications (high confidence, diverse speakers)
        notable = self._select_notable_communications(utterances)

        # Generate summary
        summary = self._generate_phase_summary(
            display_name, utterances, primary_speakers, duration
        )

        return PhaseAnalysis(
            phase_number=phase_number,
            phase_name=dominant_type,
            display_name=display_name,
            start_time=start_time.strftime('%H:%M:%S') if start_time != datetime.min else 'Unknown',
            end_time=end_time.strftime('%H:%M:%S') if end_time != datetime.min else 'Unknown',
            duration_minutes=round(duration, 1),
            utterance_count=len(utterances),
            primary_speakers=primary_speakers,
            notable_communications=notable,
            summary=summary
        )

    def _empty_phase_analysis(self, phase_number: int) -> PhaseAnalysis:
        """Create empty phase analysis."""
        return PhaseAnalysis(
            phase_number=phase_number,
            phase_name="unknown",
            display_name="Unknown Phase",
            start_time="Unknown",
            end_time="Unknown",
            duration_minutes=0,
            utterance_count=0,
            primary_speakers=[],
            notable_communications=[],
            summary="No data available for this phase."
        )

    def _select_notable_communications(
        self,
        utterances: List[Dict],
        max_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Select notable communications from phase."""
        # Sort by confidence
        sorted_utterances = sorted(
            utterances,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )

        notable = []
        seen_speakers = set()

        for u in sorted_utterances:
            if len(notable) >= max_count:
                break

            speaker = u.get('speaker') or u.get('speaker_id') or 'unknown'
            text = u.get('text', '')

            # Skip very short utterances
            if len(text) < 10:
                continue

            # Try to get diverse speakers
            if speaker in seen_speakers and len(notable) < max_count // 2:
                continue

            timestamp = u.get('timestamp', '')
            if isinstance(timestamp, datetime):
                timestamp = timestamp.strftime('%H:%M:%S')
            elif isinstance(timestamp, str) and 'T' in timestamp:
                timestamp = timestamp.split('T')[1][:8]

            notable.append({
                'timestamp': timestamp,
                'speaker': speaker,
                'text': text,
                'confidence': u.get('confidence', 0)
            })
            seen_speakers.add(speaker)

        return notable

    def _generate_phase_summary(
        self,
        display_name: str,
        utterances: List[Dict],
        primary_speakers: List[Tuple[str, int]],
        duration: float
    ) -> str:
        """Generate a brief summary of the phase."""
        speaker_text = ", ".join(
            f"{speaker} ({count})"
            for speaker, count in primary_speakers[:2]
        )

        return (
            f"Phase lasted {duration:.0f} minutes with {len(utterances)} utterances. "
            f"Primary speakers: {speaker_text}."
        )

    def generate_phase_analysis_section(self) -> str:
        """
        Generate the Mission Phase Analysis section for reports.

        Returns:
            Markdown formatted phase analysis
        """
        phases = self.analyze_phases()

        if not phases:
            return "## Mission Phase Analysis\n\nInsufficient data for phase analysis."

        lines = ["## Mission Phase Analysis", ""]

        for phase in phases:
            lines.append(f"### Phase {phase.phase_number}: {phase.display_name} "
                        f"({phase.start_time} to {phase.end_time})")
            lines.append("")
            lines.append(f"**Duration:** {phase.duration_minutes:.0f} minutes")
            lines.append(f"**Utterances:** {phase.utterance_count}")

            # Format primary speakers
            speaker_str = ", ".join(
                f"{speaker} ({count})"
                for speaker, count in phase.primary_speakers
            )
            lines.append(f"**Primary Speakers:** {speaker_str}")
            lines.append("")
            lines.append(phase.summary)
            lines.append("")

            # Notable communications
            if phase.notable_communications:
                lines.append("**Notable Communications:**")
                for comm in phase.notable_communications:
                    lines.append(f"- [{comm['timestamp']}] {comm['speaker']}: \"{comm['text']}\"")
                lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all phase analysis data
        """
        phases = self.analyze_phases()

        return {
            'phase_analysis_section': self.generate_phase_analysis_section(),
            'total_phases': len(phases),
            'phases': [
                {
                    'phase_number': p.phase_number,
                    'phase_name': p.phase_name,
                    'display_name': p.display_name,
                    'start_time': p.start_time,
                    'end_time': p.end_time,
                    'duration_minutes': p.duration_minutes,
                    'utterance_count': p.utterance_count,
                    'primary_speakers': [
                        {'speaker': s, 'count': c}
                        for s, c in p.primary_speakers
                    ],
                    'notable_communications': p.notable_communications,
                    'summary': p.summary
                }
                for p in phases
            ]
        }
