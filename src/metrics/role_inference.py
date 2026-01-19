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

    # Patterns that indicate someone is ADDRESSING a superior (NOT being the captain)
    # These should REDUCE captain score and INCREASE crew member score
    ADDRESSING_AUTHORITY_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)^(captain|sir|ma'am|commander)\b",  # Starting with title = addressing
        r"(?i)^(yes|aye|acknowledged|understood|copy|roger)\s+(sir|captain|ma'am)",
        r"(?i)\b(captain|sir),?\s+(we have|we've got|there's|i'm reading|i'm detecting)",
        r"(?i)\b(captain|sir),?\s+(what|should|do you want|shall)",
        r"(?i)\bpermission to\b",  # Asking permission = not captain
        r"(?i)\bwaiting (for|on) (your|the captain)",
        r"(?i)\b(your orders|awaiting orders)\b",
    ])

    CAPTAIN_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(set course|engage|execute|make it so|proceed)\b",
        r"(?i)\b(red alert|yellow alert|battle stations|stand down)\b",
        r"(?i)\b(all hands|attention|listen up|everyone)\b",
        r"(?i)\b(go ahead|stand by|alright|stop|wait|hold on)\b",
        r"(?i)\b(i want|we need|let's|should we)\b",
        r"(?i)\b(good work|well done|excellent|nice job)\b",
        r"(?i)\b(fire|launch|target|weapons free)\b",
        # More specific captain patterns - giving orders to specific stations
        r"(?i)\b(helm|tactical|science|engineering|ops),?\s+(report|status|what)",
        r"(?i)\bwhat('s| is) (the|our) (status|situation|position)\b",
        r"(?i)\bon screen\b",
        r"(?i)\bhail (them|the|that)\b",
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
        r"(?i)\b(confirm|confirmed|affirmative|negative)\b",
        r"(?i)\b(ready|standing by|on it)\b",
        r"(?i)\b(i'll handle|got it|taking care)\b",
        # Note: "sir/captain/ma'am" removed - now handled by addressing patterns
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
        self._addressing_patterns = self._build_addressing_patterns()

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

    def _build_addressing_patterns(self) -> List[re.Pattern]:
        """Build compiled patterns for detecting when someone is addressing authority."""
        return [re.compile(p) for p in self.patterns.ADDRESSING_AUTHORITY_PATTERNS]

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
        addressing_count = 0  # Count how often this speaker addresses authority

        for utterance in utterances:
            text = utterance.get('text', '')

            # Check if this utterance is addressing authority (e.g., "Captain, we have...")
            # This REDUCES the likelihood that this speaker is the captain
            is_addressing = False
            for pattern in self._addressing_patterns:
                if pattern.search(text):
                    is_addressing = True
                    addressing_count += 1
                    break

            for role, patterns in self._role_pattern_map.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        # If addressing authority, don't count captain patterns
                        # The speaker is talking TO the captain, not BEING the captain
                        if is_addressing and role == BridgeRole.CAPTAIN:
                            continue

                        role_scores[role] += len(matches)
                        for match in matches:
                            # Get the matched text (could be tuple from groups)
                            match_text = match if isinstance(match, str) else match[0]
                            keyword_matches[match_text.lower()] += 1
                            if match_text.lower() not in matched_keywords[role]:
                                matched_keywords[role].append(match_text.lower())

        # Determine primary role
        total_matches = sum(role_scores.values())

        # Calculate addressing ratio - how often does this speaker defer to authority?
        addressing_ratio = addressing_count / len(utterances) if utterances else 0

        utterance_pct = len(utterances) / total_utterances * 100

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

            # Special case: High addressing ratio means this person is likely NOT the captain
            # They're consistently deferring to authority
            if addressing_ratio > 0.3 and inferred_role == BridgeRole.CAPTAIN:
                # This person frequently addresses authority - they're probably NOT the captain
                # Reassign to XO or second-best role
                if len(sorted_roles) > 1:
                    inferred_role = sorted_roles[1][0]
                    confidence *= 0.7  # Reduce confidence since this was a reassignment
                else:
                    inferred_role = BridgeRole.EXECUTIVE_OFFICER
                    confidence = 0.4

            # Special case: High utterance count with XO patterns suggests command support
            if (inferred_role == BridgeRole.EXECUTIVE_OFFICER and
                utterance_pct > 25 and
                role_scores.get(BridgeRole.CAPTAIN, 0) > role_scores.get(BridgeRole.EXECUTIVE_OFFICER, 0) * 0.3):
                # This might be a captain or XO depending on context
                pass

        # Get top keywords as indicators
        top_keywords = sorted(keyword_matches.items(), key=lambda x: -x[1])[:5]
        key_indicators = [f'"{kw}" ({count})' for kw, count in top_keywords]

        # Add addressing indicator if significant
        if addressing_count > 2:
            key_indicators.append(f"addresses authority ({addressing_count}x)")

        # Select example utterances (high confidence, showing role indicators)
        example_utterances = []
        for u in sorted(utterances, key=lambda x: x.get('confidence', 0), reverse=True)[:5]:
            example_utterances.append({
                'timestamp': u.get('timestamp', ''),
                'text': u.get('text', ''),
                'confidence': u.get('confidence', 0)
            })

        # Generate methodology note
        methodology = self._generate_methodology_note(
            speaker, inferred_role, len(utterances), utterance_pct,
            role_scores, total_matches, matched_keywords, addressing_count
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
        matched_keywords: Dict[BridgeRole, List[str]],
        addressing_count: int = 0
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

        # Add context for addressing patterns
        if addressing_count > 0:
            addressing_pct = addressing_count / utterance_count * 100 if utterance_count > 0 else 0
            if addressing_pct > 30:
                note += f" Frequently addresses authority ({addressing_count} times), indicating crew member role."
            elif addressing_pct > 10:
                note += f" Sometimes addresses superiors ({addressing_count} times)."

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


@dataclass
class VoicePatternMetrics:
    """Metrics derived from speaking patterns."""
    speaker: str
    avg_words_per_utterance: float
    utterance_count: int
    speaking_percentage: float
    command_ratio: float  # Ratio of imperative sentences
    question_ratio: float  # Ratio of questions asked
    avg_utterance_duration: float
    is_dominant_speaker: bool
    response_pattern: str  # 'initiator', 'responder', 'balanced'


class VoicePatternAnalyzer:
    """
    Analyzes speaking patterns to help identify bridge roles.

    Uses speech characteristics like command frequency, question ratio,
    and speaking dominance to infer roles without requiring keywords.
    """

    # Role-specific voice pattern expectations
    ROLE_PATTERNS = {
        BridgeRole.CAPTAIN: {
            'min_utterance_pct': 20,  # Captains speak a lot
            'command_ratio_weight': 0.8,  # High command ratio expected
            'question_ratio_weight': 0.3,  # Moderate questions
            'dominant_bonus': 0.2,  # Bonus for being dominant speaker
            'initiator_bonus': 0.15,  # Bonus for initiating conversations
        },
        BridgeRole.HELM: {
            'min_utterance_pct': 10,
            'command_ratio_weight': 0.2,  # Few commands
            'question_ratio_weight': 0.2,  # Few questions
            'confirmation_weight': 0.6,  # High confirmation rate
        },
        BridgeRole.TACTICAL: {
            'min_utterance_pct': 8,
            'command_ratio_weight': 0.4,  # Some commands
            'short_utterance_weight': 0.5,  # Short, decisive
            'alert_pattern_weight': 0.6,
        },
        BridgeRole.SCIENCE: {
            'min_utterance_pct': 8,
            'long_utterance_weight': 0.5,  # Detailed explanations
            'question_ratio_weight': 0.4,  # Asks about data
            'report_pattern_weight': 0.6,
        },
        BridgeRole.ENGINEERING: {
            'min_utterance_pct': 8,
            'report_pattern_weight': 0.6,  # Status reports
            'technical_weight': 0.5,
        },
    }

    # Patterns for command detection (imperative sentences)
    COMMAND_PATTERNS = [
        r"^(?:set|engage|fire|launch|raise|lower|divert|transfer|scan|target|lock|hold|stop|go|execute|initiate)\b",
        r"^(?:all hands|attention|battle stations|red alert|yellow alert)\b",
        r"\b(?:now|immediately|at once)$",
    ]

    # Patterns for confirmation/acknowledgment
    CONFIRMATION_PATTERNS = [
        r"^(?:aye|yes|affirmative|confirmed|acknowledged|copy|roger|understood)\b",
        r"^(?:course|heading|target|shields?|weapons?|power)\s+(?:set|locked|ready|online)\b",
    ]

    # Patterns for status reports
    REPORT_PATTERNS = [
        r"^(?:captain|sir|ma'am),?\s",
        r"\b(?:reading|detecting|showing|at|levels?)\s+\d",
        r"\b(?:status|report|analysis|scan)\s+(?:complete|ready|shows)\b",
    ]

    def __init__(self, transcripts: List[Dict[str, Any]]):
        """
        Initialize voice pattern analyzer.

        Args:
            transcripts: List of transcript dictionaries with speaker and text
        """
        self.transcripts = transcripts
        self._compiled_commands = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_PATTERNS]
        self._compiled_confirms = [re.compile(p, re.IGNORECASE) for p in self.CONFIRMATION_PATTERNS]
        self._compiled_reports = [re.compile(p, re.IGNORECASE) for p in self.REPORT_PATTERNS]

    def analyze_speaker_patterns(self) -> Dict[str, VoicePatternMetrics]:
        """
        Analyze speaking patterns for all speakers.

        Returns:
            Dict mapping speaker ID to VoicePatternMetrics
        """
        speaker_utterances = defaultdict(list)
        for t in self.transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            speaker_utterances[speaker].append(t)

        total_utterances = len(self.transcripts)
        total_speakers = len(speaker_utterances)

        # Find dominant speaker
        max_utterances = max(len(u) for u in speaker_utterances.values()) if speaker_utterances else 0

        results = {}
        for speaker, utterances in speaker_utterances.items():
            results[speaker] = self._analyze_single_speaker(
                speaker, utterances, total_utterances, max_utterances
            )

        return results

    def _analyze_single_speaker(
        self,
        speaker: str,
        utterances: List[Dict[str, Any]],
        total_utterances: int,
        max_utterances: int
    ) -> VoicePatternMetrics:
        """Analyze patterns for a single speaker."""
        texts = [u.get('text', '') for u in utterances]
        utterance_count = len(utterances)

        # Calculate average words per utterance
        word_counts = [len(t.split()) for t in texts]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        # Calculate speaking percentage
        speaking_pct = (utterance_count / total_utterances * 100) if total_utterances > 0 else 0

        # Count commands and questions
        command_count = 0
        question_count = 0
        confirm_count = 0
        report_count = 0

        for text in texts:
            # Check for commands
            for pattern in self._compiled_commands:
                if pattern.search(text):
                    command_count += 1
                    break

            # Check for questions
            if '?' in text or text.lower().startswith(('what', 'where', 'when', 'how', 'why', 'is', 'are', 'do', 'does', 'can', 'could', 'should')):
                question_count += 1

            # Check for confirmations
            for pattern in self._compiled_confirms:
                if pattern.search(text):
                    confirm_count += 1
                    break

            # Check for reports
            for pattern in self._compiled_reports:
                if pattern.search(text):
                    report_count += 1
                    break

        command_ratio = command_count / utterance_count if utterance_count > 0 else 0
        question_ratio = question_count / utterance_count if utterance_count > 0 else 0

        # Calculate average utterance duration if available
        total_duration = 0
        duration_count = 0
        for u in utterances:
            start = u.get('start_time', 0)
            end = u.get('end_time', 0)
            if end > start:
                total_duration += end - start
                duration_count += 1
        avg_duration = total_duration / duration_count if duration_count > 0 else 0

        # Determine if dominant speaker
        is_dominant = utterance_count == max_utterances and speaking_pct > 25

        # Determine response pattern based on conversation position
        # (This is simplified - a full analysis would look at timing)
        if command_ratio > 0.3:
            response_pattern = 'initiator'
        elif confirm_count / utterance_count > 0.3 if utterance_count > 0 else False:
            response_pattern = 'responder'
        else:
            response_pattern = 'balanced'

        return VoicePatternMetrics(
            speaker=speaker,
            avg_words_per_utterance=round(avg_words, 1),
            utterance_count=utterance_count,
            speaking_percentage=round(speaking_pct, 1),
            command_ratio=round(command_ratio, 2),
            question_ratio=round(question_ratio, 2),
            avg_utterance_duration=round(avg_duration, 2),
            is_dominant_speaker=is_dominant,
            response_pattern=response_pattern
        )

    def get_role_hints(self) -> Dict[str, List[Tuple[BridgeRole, float]]]:
        """
        Get role hints based on voice patterns.

        Returns:
            Dict mapping speaker ID to list of (role, confidence) tuples
        """
        patterns = self.analyze_speaker_patterns()
        hints = {}

        for speaker, metrics in patterns.items():
            role_scores = []

            # Captain hints
            captain_score = 0.0
            if metrics.is_dominant_speaker:
                captain_score += 0.2
            if metrics.command_ratio > 0.2:
                captain_score += min(0.3, metrics.command_ratio)
            if metrics.speaking_percentage > 25:
                captain_score += 0.15
            if metrics.response_pattern == 'initiator':
                captain_score += 0.1
            if captain_score > 0.2:
                role_scores.append((BridgeRole.CAPTAIN, min(0.6, captain_score)))

            # Helm hints
            helm_score = 0.0
            if metrics.speaking_percentage > 8 and metrics.command_ratio < 0.2:
                helm_score += 0.2
            if metrics.avg_words_per_utterance < 10:
                helm_score += 0.1
            if metrics.response_pattern == 'responder':
                helm_score += 0.15
            if helm_score > 0.2:
                role_scores.append((BridgeRole.HELM, min(0.5, helm_score)))

            # Tactical hints
            tactical_score = 0.0
            if metrics.avg_words_per_utterance < 8:  # Short, decisive
                tactical_score += 0.15
            if metrics.command_ratio > 0.1 and metrics.command_ratio < 0.3:
                tactical_score += 0.15
            if tactical_score > 0.2:
                role_scores.append((BridgeRole.TACTICAL, min(0.5, tactical_score)))

            # Science hints
            science_score = 0.0
            if metrics.avg_words_per_utterance > 12:  # Longer explanations
                science_score += 0.2
            if metrics.question_ratio > 0.15:
                science_score += 0.15
            if science_score > 0.2:
                role_scores.append((BridgeRole.SCIENCE, min(0.5, science_score)))

            # Engineering hints
            eng_score = 0.0
            if metrics.speaking_percentage > 5 and metrics.speaking_percentage < 20:
                eng_score += 0.1
            if metrics.response_pattern == 'responder':
                eng_score += 0.1
            if eng_score > 0.15:
                role_scores.append((BridgeRole.ENGINEERING, min(0.4, eng_score)))

            # Sort by score descending
            role_scores.sort(key=lambda x: -x[1])
            hints[speaker] = role_scores

        return hints


class EnhancedRoleInferenceEngine(RoleInferenceEngine):
    """
    Enhanced role inference using both keywords and voice patterns.

    Combines keyword-based role detection with voice pattern analysis
    for more accurate role identification from audio alone.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        patterns: Optional[RolePatterns] = None,
        use_voice_patterns: bool = True
    ):
        """
        Initialize enhanced role inference engine.

        Args:
            transcripts: List of transcript dictionaries
            patterns: Optional custom role patterns
            use_voice_patterns: Whether to use voice pattern analysis
        """
        super().__init__(transcripts, patterns)
        self.use_voice_patterns = use_voice_patterns
        self._voice_analyzer = VoicePatternAnalyzer(transcripts) if use_voice_patterns else None

    def analyze_all_speakers(self) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Analyze all speakers using keywords and voice patterns.

        Returns:
            Dictionary mapping speaker IDs to their role analysis
        """
        # Get base keyword analysis
        results = super().analyze_all_speakers()

        # Enhance with voice patterns if enabled
        if self.use_voice_patterns and self._voice_analyzer:
            voice_hints = self._voice_analyzer.get_role_hints()
            voice_metrics = self._voice_analyzer.analyze_speaker_patterns()

            for speaker, analysis in results.items():
                hints = voice_hints.get(speaker, [])
                metrics = voice_metrics.get(speaker)

                if hints and metrics:
                    # Boost confidence if voice patterns agree with keyword role
                    keyword_role = analysis.inferred_role
                    for hint_role, hint_score in hints:
                        if hint_role == keyword_role:
                            # Voice pattern agrees - boost confidence
                            boost = min(0.15, hint_score * 0.3)
                            new_confidence = min(1.0, analysis.confidence + boost)

                            # Update methodology note
                            pattern_note = self._generate_voice_pattern_note(metrics)
                            new_methodology = (
                                f"{analysis.methodology_notes} "
                                f"Voice pattern analysis (+{boost:.0%}): {pattern_note}"
                            )

                            # Create updated analysis
                            results[speaker] = SpeakerRoleAnalysis(
                                speaker=analysis.speaker,
                                inferred_role=analysis.inferred_role,
                                confidence=round(new_confidence, 2),
                                utterance_count=analysis.utterance_count,
                                utterance_percentage=analysis.utterance_percentage,
                                keyword_matches=analysis.keyword_matches,
                                total_keyword_matches=analysis.total_keyword_matches,
                                key_indicators=analysis.key_indicators,
                                example_utterances=analysis.example_utterances,
                                methodology_notes=new_methodology
                            )
                            break

                    # If keyword analysis found UNKNOWN but voice patterns suggest a role
                    if analysis.inferred_role == BridgeRole.UNKNOWN and hints:
                        top_hint = hints[0]
                        if top_hint[1] >= 0.3:  # Reasonable voice pattern confidence
                            new_methodology = (
                                f"{analysis.methodology_notes} "
                                f"Voice pattern suggests {top_hint[0].value} "
                                f"(confidence: {top_hint[1]:.0%})."
                            )
                            results[speaker] = SpeakerRoleAnalysis(
                                speaker=analysis.speaker,
                                inferred_role=top_hint[0],
                                confidence=round(top_hint[1] * 0.7, 2),  # Reduced since no keyword support
                                utterance_count=analysis.utterance_count,
                                utterance_percentage=analysis.utterance_percentage,
                                keyword_matches=analysis.keyword_matches,
                                total_keyword_matches=analysis.total_keyword_matches,
                                key_indicators=analysis.key_indicators + [f"voice:{top_hint[0].value.split('/')[0].lower()}"],
                                example_utterances=analysis.example_utterances,
                                methodology_notes=new_methodology
                            )

        return results

    def _generate_voice_pattern_note(self, metrics: VoicePatternMetrics) -> str:
        """Generate a note describing the voice patterns observed."""
        parts = []

        if metrics.is_dominant_speaker:
            parts.append("dominant speaker")

        if metrics.command_ratio > 0.2:
            parts.append(f"{metrics.command_ratio:.0%} commands")
        elif metrics.command_ratio < 0.1:
            parts.append("few commands")

        if metrics.avg_words_per_utterance > 12:
            parts.append("detailed explanations")
        elif metrics.avg_words_per_utterance < 6:
            parts.append("brief responses")

        if metrics.response_pattern == 'initiator':
            parts.append("initiates conversations")
        elif metrics.response_pattern == 'responder':
            parts.append("responds to others")

        return ", ".join(parts) if parts else "standard speaking pattern"

    def get_voice_pattern_summary(self) -> Dict[str, Any]:
        """
        Get a summary of voice patterns for all speakers.

        Returns:
            Dictionary with voice pattern analysis
        """
        if not self._voice_analyzer:
            return {}

        metrics = self._voice_analyzer.analyze_speaker_patterns()
        hints = self._voice_analyzer.get_role_hints()

        return {
            'speaker_patterns': {
                speaker: {
                    'avg_words': m.avg_words_per_utterance,
                    'speaking_pct': m.speaking_percentage,
                    'command_ratio': m.command_ratio,
                    'question_ratio': m.question_ratio,
                    'is_dominant': m.is_dominant_speaker,
                    'pattern_type': m.response_pattern,
                    'role_hints': [(r.value, s) for r, s in hints.get(speaker, [])]
                }
                for speaker, m in metrics.items()
            }
        }
