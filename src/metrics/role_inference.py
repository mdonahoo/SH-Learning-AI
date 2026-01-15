"""
Role inference engine for crew member analysis.

This module provides keyword-frequency-based role detection for bridge crew members,
matching the methodology described in the example mission debrief report.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class BridgeRole(Enum):
    """Standard bridge roles in Starship Horizons."""
    CAPTAIN = "Captain/Command"
    EXECUTIVE_OFFICER = "Executive Officer/Support"
    HELM = "Helm/Navigation"
    TACTICAL = "Tactical/Weapons"
    SCIENCE = "Science/Sensors"
    ENGINEERING = "Engineering/Systems"
    OPERATIONS = "Operations/Monitoring"
    COMMUNICATIONS = "Communications"
    UNKNOWN = "Crew Member"


@dataclass
class RolePatterns:
    """Keyword patterns for role detection."""

    CAPTAIN_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(set course|engage|execute|make it so|proceed)\b",
        r"(?i)\b(red alert|yellow alert|battle stations|stand down)\b",
        r"(?i)\b(all hands|attention|listen up|everyone)\b",
        r"(?i)\b(go ahead|stand by|alright|stop|wait|hold on)\b",
        r"(?i)\b(i want|we need|let's|should we)\b",
        r"(?i)\b(good work|well done|excellent|nice job)\b",
        r"(?i)\b(report|status report|give me|what's our)\b",
        r"(?i)\b(fire|launch|target|weapons free)\b",
    ])

    HELM_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(course laid in|course set|heading|bearing)\b",
        r"(?i)\b(impulse|warp|full stop|all stop)\b",
        r"(?i)\b(eta|arrival|distance to|kilometers away)\b",
        r"(?i)\b(approach|approaching|orbit|docking)\b",
        r"(?i)\b(evasive|maneuver|turn|rotate)\b",
        r"(?i)\b(navigation|nav|plotting|plot course)\b",
    ])

    TACTICAL_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(targeting|target locked|acquiring|locked on)\b",
        r"(?i)\b(weapons|torpedoes|phasers|missiles)\b",
        r"(?i)\b(shields|shield status|shields at|hull)\b",
        r"(?i)\b(firing|fire|launch|launching)\b",
        r"(?i)\b(enemy|hostile|threat|contact|bogey)\b",
        r"(?i)\b(damage|hit|impact|taking fire)\b",
    ])

    SCIENCE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(scanning|scan complete|sensors|detecting)\b",
        r"(?i)\b(reading|readings|analysis|analyzing)\b",
        r"(?i)\b(anomaly|signal|signature|emissions)\b",
        r"(?i)\b(life signs|life forms|biosigns)\b",
        r"(?i)\b(composition|spectrum|radiation)\b",
        r"(?i)\b(data|research|scientific)\b",
    ])

    ENGINEERING_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(reactor|power levels|power at|warp core)\b",
        r"(?i)\b(rerouting|diverting|transferring) power\b",
        r"(?i)\b(damage control|repairs|repairing)\b",
        r"(?i)\b(systems|subsystems|online|offline)\b",
        r"(?i)\b(coolant|overload|overheating|venting)\b",
        r"(?i)\b(efficiency|output|capacity)\b",
    ])

    OPERATIONS_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(sector|quadrant|grid|coordinates)\b",
        r"(?i)\b(monitoring|tracking|observing)\b",
        r"(?i)\b(status|nominal|operational|ready)\b",
        r"(?i)\b(cargo|supplies|inventory|manifest)\b",
        r"(?i)\b(docking|dock|bay|hangar)\b",
        r"(?i)\b(schedule|timing|countdown)\b",
    ])

    COMMUNICATIONS_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(hailing|hail|channel open|frequency)\b",
        r"(?i)\b(transmitting|transmission|receiving|signal)\b",
        r"(?i)\b(message|incoming|outgoing)\b",
        r"(?i)\b(audio|visual|subspace)\b",
        r"(?i)\b(broadcast|distress|mayday)\b",
    ])

    EXECUTIVE_OFFICER_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(aye|acknowledged|understood|copy|roger)\b",
        r"(?i)\b(sir|captain|ma'am)\b",
        r"(?i)\b(confirm|confirmed|affirmative|negative)\b",
        r"(?i)\b(ready|standing by|on it)\b",
        r"(?i)\b(i'll handle|got it|taking care)\b",
    ])


@dataclass
class SpeakerRoleAnalysis:
    """Analysis results for a single speaker's role."""
    speaker: str
    inferred_role: BridgeRole
    confidence: float
    utterance_count: int
    utterance_percentage: float
    keyword_matches: Dict[str, int]
    total_keyword_matches: int
    key_indicators: List[str]
    example_utterances: List[Dict[str, Any]]
    methodology_notes: str


class RoleInferenceEngine:
    """
    Engine for inferring bridge crew roles from transcript analysis.

    Uses keyword frequency analysis to assign probable roles to speakers,
    matching the methodology used in professional mission debrief reports.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        patterns: Optional[RolePatterns] = None
    ):
        """
        Initialize the role inference engine.

        Args:
            transcripts: List of transcript dictionaries with 'speaker' and 'text'
            patterns: Optional custom role patterns
        """
        self.transcripts = transcripts
        self.patterns = patterns or RolePatterns()
        self._role_pattern_map = self._build_role_pattern_map()

    def _build_role_pattern_map(self) -> Dict[BridgeRole, List[str]]:
        """Build mapping of roles to their detection patterns."""
        return {
            BridgeRole.CAPTAIN: self.patterns.CAPTAIN_PATTERNS,
            BridgeRole.HELM: self.patterns.HELM_PATTERNS,
            BridgeRole.TACTICAL: self.patterns.TACTICAL_PATTERNS,
            BridgeRole.SCIENCE: self.patterns.SCIENCE_PATTERNS,
            BridgeRole.ENGINEERING: self.patterns.ENGINEERING_PATTERNS,
            BridgeRole.OPERATIONS: self.patterns.OPERATIONS_PATTERNS,
            BridgeRole.COMMUNICATIONS: self.patterns.COMMUNICATIONS_PATTERNS,
            BridgeRole.EXECUTIVE_OFFICER: self.patterns.EXECUTIVE_OFFICER_PATTERNS,
        }

    def analyze_all_speakers(self) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Analyze all speakers and infer their roles.

        Returns:
            Dictionary mapping speaker IDs to their role analysis
        """
        # Count utterances per speaker
        speaker_utterances = defaultdict(list)
        for t in self.transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            speaker_utterances[speaker].append(t)

        total_utterances = len(self.transcripts)
        results = {}

        for speaker, utterances in speaker_utterances.items():
            results[speaker] = self._analyze_speaker(
                speaker, utterances, total_utterances
            )

        # Post-process to resolve role conflicts
        results = self._resolve_role_conflicts(results)

        return results

    def _analyze_speaker(
        self,
        speaker: str,
        utterances: List[Dict[str, Any]],
        total_utterances: int
    ) -> SpeakerRoleAnalysis:
        """Analyze a single speaker's communications."""
        # Count keyword matches per role
        role_scores: Dict[BridgeRole, int] = defaultdict(int)
        keyword_matches: Dict[str, int] = defaultdict(int)
        matched_keywords: Dict[BridgeRole, List[str]] = defaultdict(list)

        for utterance in utterances:
            text = utterance.get('text', '')

            for role, patterns in self._role_pattern_map.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        role_scores[role] += len(matches)
                        for match in matches:
                            # Get the matched text (could be tuple from groups)
                            match_text = match if isinstance(match, str) else match[0]
                            keyword_matches[match_text.lower()] += 1
                            if match_text.lower() not in matched_keywords[role]:
                                matched_keywords[role].append(match_text.lower())

        # Determine primary role
        total_matches = sum(role_scores.values())

        if total_matches == 0:
            inferred_role = BridgeRole.UNKNOWN
            confidence = 0.0
        else:
            # Find role with highest score
            sorted_roles = sorted(role_scores.items(), key=lambda x: -x[1])
            inferred_role = sorted_roles[0][0]

            # Calculate confidence based on score dominance
            top_score = sorted_roles[0][1]
            second_score = sorted_roles[1][1] if len(sorted_roles) > 1 else 0

            # Confidence is higher when one role clearly dominates
            if top_score > 0:
                dominance = (top_score - second_score) / top_score
                frequency = top_score / len(utterances)
                confidence = min(1.0, (dominance * 0.5 + frequency * 0.5))
            else:
                confidence = 0.0

        # Special case: High utterance count with XO patterns suggests command support
        utterance_pct = len(utterances) / total_utterances * 100
        if (inferred_role == BridgeRole.EXECUTIVE_OFFICER and
            utterance_pct > 25 and
            role_scores.get(BridgeRole.CAPTAIN, 0) > role_scores.get(BridgeRole.EXECUTIVE_OFFICER, 0) * 0.3):
            # This might be a captain or XO depending on context
            pass

        # Get top keywords as indicators
        top_keywords = sorted(keyword_matches.items(), key=lambda x: -x[1])[:5]
        key_indicators = [f'"{kw}" ({count})' for kw, count in top_keywords]

        # Select example utterances (high confidence, showing role indicators)
        example_utterances = []
        for u in sorted(utterances, key=lambda x: x.get('confidence', 0), reverse=True)[:3]:
            example_utterances.append({
                'timestamp': u.get('timestamp', ''),
                'text': u.get('text', ''),
                'confidence': u.get('confidence', 0)
            })

        # Generate methodology note
        methodology = self._generate_methodology_note(
            speaker, inferred_role, len(utterances), utterance_pct,
            role_scores, total_matches, matched_keywords
        )

        return SpeakerRoleAnalysis(
            speaker=speaker,
            inferred_role=inferred_role,
            confidence=round(confidence, 2),
            utterance_count=len(utterances),
            utterance_percentage=round(utterance_pct, 1),
            keyword_matches=dict(keyword_matches),
            total_keyword_matches=total_matches,
            key_indicators=key_indicators,
            example_utterances=example_utterances,
            methodology_notes=methodology
        )

    def _generate_methodology_note(
        self,
        speaker: str,
        role: BridgeRole,
        utterance_count: int,
        utterance_pct: float,
        role_scores: Dict[BridgeRole, int],
        total_matches: int,
        matched_keywords: Dict[BridgeRole, List[str]]
    ) -> str:
        """Generate explanation of role assignment methodology."""
        if role == BridgeRole.UNKNOWN:
            return (f"{speaker} had {utterance_count} utterances ({utterance_pct:.1f}% of traffic) "
                   f"but no clear role indicators were detected.")

        role_score = role_scores.get(role, 0)
        role_keywords = matched_keywords.get(role, [])
        keyword_sample = ', '.join(f'"{kw}"' for kw in role_keywords[:5])

        note = (f"{speaker} was assigned {role.value} based on {role_score} keyword pattern matches "
               f"(including {keyword_sample}) combined with {utterance_count} utterances "
               f"({utterance_pct:.1f}% of all voice traffic).")

        # Add context for high-volume speakers
        if utterance_pct > 40:
            note += f" This speaker dominated communications, suggesting a command role."
        elif utterance_pct > 20:
            note += f" Significant communication volume indicates an active role."

        return note

    def _resolve_role_conflicts(
        self,
        results: Dict[str, SpeakerRoleAnalysis]
    ) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Resolve conflicts when multiple speakers are assigned the same role.

        The speaker with the highest score for a role keeps it; others are
        reassigned to their next best role or marked as support.
        """
        # Group by inferred role
        role_assignments: Dict[BridgeRole, List[SpeakerRoleAnalysis]] = defaultdict(list)
        for analysis in results.values():
            role_assignments[analysis.inferred_role].append(analysis)

        # For roles with multiple speakers, keep the strongest match
        reassignments = {}
        for role, speakers in role_assignments.items():
            if role == BridgeRole.UNKNOWN:
                continue

            if len(speakers) > 1:
                # Sort by total keyword matches for this role
                sorted_speakers = sorted(
                    speakers,
                    key=lambda x: x.total_keyword_matches,
                    reverse=True
                )

                # First speaker keeps the role
                primary = sorted_speakers[0]

                # Others become support roles or unknown
                for secondary in sorted_speakers[1:]:
                    if secondary.utterance_percentage > 15:
                        # High volume speaker becomes XO/Support
                        reassignments[secondary.speaker] = BridgeRole.EXECUTIVE_OFFICER
                    else:
                        reassignments[secondary.speaker] = BridgeRole.UNKNOWN

        # Apply reassignments
        for speaker, new_role in reassignments.items():
            old_analysis = results[speaker]
            results[speaker] = SpeakerRoleAnalysis(
                speaker=old_analysis.speaker,
                inferred_role=new_role,
                confidence=old_analysis.confidence * 0.7,  # Reduce confidence for reassigned
                utterance_count=old_analysis.utterance_count,
                utterance_percentage=old_analysis.utterance_percentage,
                keyword_matches=old_analysis.keyword_matches,
                total_keyword_matches=old_analysis.total_keyword_matches,
                key_indicators=old_analysis.key_indicators,
                example_utterances=old_analysis.example_utterances,
                methodology_notes=old_analysis.methodology_notes +
                    f" (Reassigned from {old_analysis.inferred_role.value} due to role conflict.)"
            )

        return results

    def generate_role_analysis_table(self) -> str:
        """
        Generate a markdown table of role assignments.

        Returns:
            Markdown formatted table string
        """
        results = self.analyze_all_speakers()

        # Sort by utterance count (descending)
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.utterance_count,
            reverse=True
        )

        lines = [
            "| Speaker | Utterances | Likely Role | Key Indicators |",
            "| --- | --- | --- | --- |"
        ]

        for analysis in sorted_results:
            indicators = ", ".join(analysis.key_indicators[:3]) if analysis.key_indicators else "No clear indicators"
            lines.append(
                f"| {analysis.speaker} | {analysis.utterance_count} | "
                f"{analysis.inferred_role.value} | {indicators} |"
            )

        return "\n".join(lines)

    def generate_methodology_section(self) -> str:
        """
        Generate the Role Assignment Methodology section for reports.

        Returns:
            Markdown formatted methodology explanation
        """
        results = self.analyze_all_speakers()

        # Sort by utterance count
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.utterance_count,
            reverse=True
        )

        lines = ["### Role Assignment Methodology", ""]
        lines.append(
            "Role assignments are based on keyword frequency analysis across all utterances. "
            "Each speaker's communications were analyzed for patterns indicating specific bridge roles."
        )
        lines.append("")

        for analysis in sorted_results:
            if analysis.utterance_count > 0:
                lines.append(analysis.methodology_notes)
                lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all role analysis data
        """
        results = self.analyze_all_speakers()

        return {
            'role_table': self.generate_role_analysis_table(),
            'methodology': self.generate_methodology_section(),
            'speaker_roles': {
                speaker: {
                    'role': analysis.inferred_role.value,
                    'confidence': analysis.confidence,
                    'utterance_count': analysis.utterance_count,
                    'utterance_percentage': analysis.utterance_percentage,
                    'keyword_matches': analysis.total_keyword_matches,
                    'key_indicators': analysis.key_indicators,
                    'methodology_note': analysis.methodology_notes
                }
                for speaker, analysis in results.items()
            }
        }
