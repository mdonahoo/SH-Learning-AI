"""
Communication quality analyzer for identifying effective and problematic patterns.

Detects filler words, incomplete sentences, and categorizes communications
as effective or needing improvement with supporting evidence.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CommunicationPattern:
    """A detected communication pattern."""
    name: str
    category: str  # "effective" or "needs_improvement"
    description: str
    patterns: List[str]


# Effective communication patterns
EFFECTIVE_PATTERNS = [
    CommunicationPattern(
        name="clear_command",
        category="effective",
        description="Clear, specific command with action and target",
        patterns=[
            r"(?i)(set course|engage|execute|fire|launch) .{5,}",
            r"(?i)(stop|hold|wait) (us )?(at|within) \d+",
            r"(?i)once we.{5,}(i'll|we'll|then)",
        ]
    ),
    CommunicationPattern(
        name="explicit_threshold",
        category="effective",
        description="Communication with explicit numeric threshold",
        patterns=[
            r"(?i)within \d+",
            r"(?i)(at|to) \d+ (kilometers|km|meters|percent|%)",
            r"(?i)\d+ (seconds|minutes|hours)",
        ]
    ),
    CommunicationPattern(
        name="proper_acknowledgment",
        category="effective",
        description="Proper acknowledgment of orders",
        patterns=[
            r"(?i)\b(aye|aye aye|acknowledged|understood|copy that|roger that)\b",
            r"(?i)\b(yes sir|yes ma'am|yes captain)\b",
            r"(?i)\b(on it|doing it now|executing)\b",
        ]
    ),
    CommunicationPattern(
        name="readiness_verification",
        category="effective",
        description="Verification of readiness before action",
        patterns=[
            r"(?i)(are you|is everyone|all stations) ready",
            r"(?i)ready to (fire|launch|engage|proceed)",
            r"(?i)standing by (for|to)",
        ]
    ),
    CommunicationPattern(
        name="status_report",
        category="effective",
        description="Clear status report with metrics",
        patterns=[
            r"(?i)(shields|hull|power|systems) (at|is|are) \d+%?",
            r"(?i)(distance|range|eta).{1,10}\d+",
            r"(?i)(all|systems) (nominal|operational|online|ready)",
        ]
    ),
    CommunicationPattern(
        name="conditional_action",
        category="effective",
        description="Clear conditional action statement",
        patterns=[
            r"(?i)once (we|you|they).{5,}(then|i'll|we'll)",
            r"(?i)when (we|you).{5,}(fire|launch|engage)",
            r"(?i)if .{5,}then",
        ]
    ),
]

# Patterns needing improvement
IMPROVEMENT_PATTERNS = [
    CommunicationPattern(
        name="filler_words",
        category="needs_improvement",
        description="Excessive filler words",
        patterns=[
            r"(?i)\buh+\b",
            r"(?i)\bum+\b",
            r"(?i)\blike,?\s",
            r"(?i)\byou know\b",
            r"(?i)\bso,?\s+(uh|um)",
        ]
    ),
    CommunicationPattern(
        name="incomplete_sentence",
        category="needs_improvement",
        description="Trailing off or incomplete thought",
        patterns=[
            r"\.\.\.$",
            r"(?i)where is the.{0,10}$",
            r"(?i)what about.{0,10}$",
            r"(?i)so,?\s*$",
            r"(?i)and then.{0,5}$",
        ]
    ),
    CommunicationPattern(
        name="vague_destination",
        category="needs_improvement",
        description="Vague or unclear destination/target",
        patterns=[
            r"(?i)get us (out|there|over)",
            r"(?i)go (there|that way|over there)",
            r"(?i)head (that way|over)",
        ]
    ),
    CommunicationPattern(
        name="uncertainty_marker",
        category="needs_improvement",
        description="Expression of uncertainty without context",
        patterns=[
            r"(?i)i('m| am) not (sure|confident|certain)",
            r"(?i)i (don't|do not) know",
            r"(?i)maybe we (should|could)",
            r"(?i)i (think|guess) (so|maybe)",
        ]
    ),
    CommunicationPattern(
        name="repetitive_alert",
        category="needs_improvement",
        description="Repetitive alerts without detail",
        patterns=[
            r"(.{5,})\1{2,}",  # Same phrase repeated 3+ times
            r"(?i)(incoming|alert|warning)[!.]+\s*(incoming|alert|warning)",
        ]
    ),
    CommunicationPattern(
        name="single_word_response",
        category="needs_improvement",
        description="Single word response lacking detail",
        patterns=[
            r"^(okay|ok|sure|yeah|yep|right|fine)\.?$",
            r"^(yes|no|maybe)\.?$",
        ]
    ),
]


@dataclass
class CommunicationAssessment:
    """Assessment of a single communication."""
    timestamp: str
    speaker: str
    text: str
    confidence: float
    category: str  # "effective" or "needs_improvement"
    pattern_name: str
    pattern_description: str
    assessment: str


class CommunicationQualityAnalyzer:
    """
    Analyzes communication quality and categorizes patterns.

    Identifies effective communications and those needing improvement
    with specific evidence and recommendations.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        effective_patterns: List[CommunicationPattern] = None,
        improvement_patterns: List[CommunicationPattern] = None
    ):
        """
        Initialize the communication quality analyzer.

        Args:
            transcripts: List of transcript dictionaries
            effective_patterns: Optional custom effective patterns
            improvement_patterns: Optional custom improvement patterns
        """
        self.transcripts = transcripts
        self.effective_patterns = effective_patterns or EFFECTIVE_PATTERNS
        self.improvement_patterns = improvement_patterns or IMPROVEMENT_PATTERNS

    def analyze_all(self) -> Dict[str, Any]:
        """
        Analyze all communications.

        Returns:
            Dictionary with categorized communications and statistics
        """
        effective = []
        needs_improvement = []
        uncategorized = []

        for t in self.transcripts:
            assessment = self._assess_communication(t)
            if assessment:
                if assessment.category == "effective":
                    effective.append(assessment)
                else:
                    needs_improvement.append(assessment)
            else:
                uncategorized.append(t)

        # Calculate pattern frequency
        effective_by_pattern = defaultdict(list)
        improvement_by_pattern = defaultdict(list)

        for a in effective:
            effective_by_pattern[a.pattern_name].append(a)
        for a in needs_improvement:
            improvement_by_pattern[a.pattern_name].append(a)

        return {
            'effective': effective,
            'needs_improvement': needs_improvement,
            'uncategorized_count': len(uncategorized),
            'total_analyzed': len(self.transcripts),
            'effective_by_pattern': dict(effective_by_pattern),
            'improvement_by_pattern': dict(improvement_by_pattern),
            'statistics': self._calculate_statistics(effective, needs_improvement)
        }

    def _assess_communication(self, transcript: Dict) -> Optional[CommunicationAssessment]:
        """Assess a single communication."""
        text = transcript.get('text', '')
        speaker = transcript.get('speaker') or transcript.get('speaker_id') or 'unknown'
        confidence = transcript.get('confidence', 0)
        timestamp = transcript.get('timestamp', '')

        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime('%H:%M:%S')
        elif isinstance(timestamp, str) and 'T' in timestamp:
            timestamp = timestamp.split('T')[1][:8]

        # Check for improvement patterns first (they're more specific)
        for pattern in self.improvement_patterns:
            for regex in pattern.patterns:
                if re.search(regex, text):
                    return CommunicationAssessment(
                        timestamp=timestamp,
                        speaker=speaker,
                        text=text,
                        confidence=confidence,
                        category="needs_improvement",
                        pattern_name=pattern.name,
                        pattern_description=pattern.description,
                        assessment=self._generate_improvement_assessment(pattern, text)
                    )

        # Check for effective patterns
        for pattern in self.effective_patterns:
            for regex in pattern.patterns:
                if re.search(regex, text):
                    return CommunicationAssessment(
                        timestamp=timestamp,
                        speaker=speaker,
                        text=text,
                        confidence=confidence,
                        category="effective",
                        pattern_name=pattern.name,
                        pattern_description=pattern.description,
                        assessment=self._generate_effective_assessment(pattern, text)
                    )

        return None

    def _generate_improvement_assessment(
        self,
        pattern: CommunicationPattern,
        text: str
    ) -> str:
        """Generate assessment text for improvement pattern."""
        assessments = {
            "filler_words": "Filler words reduce clarity and confidence",
            "incomplete_sentence": "Incomplete thought disrupts information flow",
            "vague_destination": "Vague destination, needs specific coordinates or target",
            "uncertainty_marker": "Uncertainty without specific context",
            "repetitive_alert": "Repetitive alert without actionable detail",
            "single_word_response": "Single word response lacks necessary detail",
        }
        return assessments.get(pattern.name, pattern.description)

    def _generate_effective_assessment(
        self,
        pattern: CommunicationPattern,
        text: str
    ) -> str:
        """Generate assessment text for effective pattern."""
        assessments = {
            "clear_command": "Clear command with specific action",
            "explicit_threshold": "Explicit numeric threshold provided",
            "proper_acknowledgment": "Proper acknowledgment of orders",
            "readiness_verification": "Verification before action",
            "status_report": "Clear status report with metrics",
            "conditional_action": "Clear conditional action statement",
        }
        return assessments.get(pattern.name, pattern.description)

    def _calculate_statistics(
        self,
        effective: List[CommunicationAssessment],
        needs_improvement: List[CommunicationAssessment]
    ) -> Dict[str, Any]:
        """Calculate quality statistics."""
        total = len(self.transcripts)
        effective_count = len(effective)
        improvement_count = len(needs_improvement)

        # By speaker
        speaker_effective = defaultdict(int)
        speaker_improvement = defaultdict(int)

        for a in effective:
            speaker_effective[a.speaker] += 1
        for a in needs_improvement:
            speaker_improvement[a.speaker] += 1

        return {
            'total_utterances': total,
            'effective_count': effective_count,
            'effective_percentage': round(effective_count / total * 100, 1) if total > 0 else 0,
            'improvement_count': improvement_count,
            'improvement_percentage': round(improvement_count / total * 100, 1) if total > 0 else 0,
            'speaker_effective': dict(speaker_effective),
            'speaker_improvement': dict(speaker_improvement),
        }

    def generate_command_control_section(self) -> str:
        """
        Generate Command and Control Assessment section.

        Returns:
            Markdown formatted assessment section
        """
        results = self.analyze_all()

        lines = [
            "## Command and Control Assessment",
            "",
            "### Command Clarity Analysis",
            "",
        ]

        # Effective examples table
        lines.append("**Effective Command Examples:**")
        lines.append("")
        lines.append("| Timestamp | Speaker | Communication | Assessment |")
        lines.append("| --- | --- | --- | --- |")

        effective = results['effective']
        # Sort by confidence and take top 5
        sorted_effective = sorted(effective, key=lambda x: x.confidence, reverse=True)[:5]

        for a in sorted_effective:
            # Truncate long text
            text = a.text[:60] + "..." if len(a.text) > 60 else a.text
            lines.append(f"| {a.timestamp} | {a.speaker} | \"{text}\" | {a.assessment} |")

        lines.append("")

        # Improvement examples table
        lines.append("**Communications Requiring Improvement:**")
        lines.append("")
        lines.append("| Timestamp | Speaker | Communication | Issue |")
        lines.append("| --- | --- | --- | --- |")

        needs_improvement = results['needs_improvement']
        sorted_improvement = sorted(needs_improvement, key=lambda x: x.confidence, reverse=True)[:5]

        for a in sorted_improvement:
            text = a.text[:60] + "..." if len(a.text) > 60 else a.text
            lines.append(f"| {a.timestamp} | {a.speaker} | \"{text}\" | {a.assessment} |")

        lines.append("")

        # Statistics summary
        stats = results['statistics']
        lines.append("### Communication Quality Statistics")
        lines.append("")
        lines.append(f"- **Effective Communications:** {stats['effective_count']} ({stats['effective_percentage']}%)")
        lines.append(f"- **Communications Needing Improvement:** {stats['improvement_count']} ({stats['improvement_percentage']}%)")
        lines.append("")

        # Pattern breakdown
        if results['improvement_by_pattern']:
            lines.append("**Issue Breakdown:**")
            for pattern_name, assessments in sorted(
                results['improvement_by_pattern'].items(),
                key=lambda x: -len(x[1])
            ):
                pattern = next(
                    (p for p in self.improvement_patterns if p.name == pattern_name),
                    None
                )
                desc = pattern.description if pattern else pattern_name
                lines.append(f"- {desc}: {len(assessments)} occurrences")
            lines.append("")

        return "\n".join(lines)

    def generate_notable_communications_section(self) -> str:
        """
        Generate Notable Communications section.

        Returns:
            Markdown formatted notable communications
        """
        results = self.analyze_all()

        lines = [
            "## Notable Communications",
            "",
            "### Exemplary Communications",
            ""
        ]

        # Top effective communications
        effective = sorted(results['effective'], key=lambda x: x.confidence, reverse=True)[:5]
        for a in effective:
            lines.append(f"- **[{a.timestamp}] {a.speaker}:** \"{a.text}\"")
            lines.append(f"  - *{a.assessment}*")
            lines.append("")

        lines.append("### Communications Needing Improvement")
        lines.append("")

        # Top improvement areas
        needs_improvement = sorted(
            results['needs_improvement'],
            key=lambda x: x.confidence,
            reverse=True
        )[:5]

        for a in needs_improvement:
            lines.append(f"- **[{a.timestamp}] {a.speaker}:** \"{a.text}\"")
            lines.append(f"  - *Issue: {a.assessment}*")
            lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all communication quality data
        """
        results = self.analyze_all()

        return {
            'command_control_section': self.generate_command_control_section(),
            'notable_communications_section': self.generate_notable_communications_section(),
            'statistics': results['statistics'],
            'effective_examples': [
                {
                    'timestamp': a.timestamp,
                    'speaker': a.speaker,
                    'text': a.text,
                    'assessment': a.assessment,
                    'pattern': a.pattern_name
                }
                for a in sorted(results['effective'], key=lambda x: x.confidence, reverse=True)[:10]
            ],
            'improvement_examples': [
                {
                    'timestamp': a.timestamp,
                    'speaker': a.speaker,
                    'text': a.text,
                    'issue': a.assessment,
                    'pattern': a.pattern_name
                }
                for a in sorted(results['needs_improvement'], key=lambda x: x.confidence, reverse=True)[:10]
            ],
            'pattern_counts': {
                'effective': {k: len(v) for k, v in results['effective_by_pattern'].items()},
                'needs_improvement': {k: len(v) for k, v in results['improvement_by_pattern'].items()}
            }
        }
