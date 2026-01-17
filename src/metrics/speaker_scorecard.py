"""
Speaker scorecard generator for individual crew performance assessment.

Generates evidence-based 1-5 ratings for each crew member across
multiple performance dimensions with supporting quotes.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScoreMetric:
    """Definition of a scoring metric."""
    name: str
    display_name: str
    description: str
    score_1: str
    score_3: str
    score_5: str


# Standard scoring metrics
SCORE_METRICS = [
    ScoreMetric(
        name="protocol_adherence",
        display_name="Protocol Adherence",
        description="Use of standard bridge protocols and formal communication",
        score_1="No protocol language detected",
        score_3="Occasional use of protocols",
        score_5="Consistent, proper protocol usage"
    ),
    ScoreMetric(
        name="communication_clarity",
        display_name="Communication Clarity",
        description="Clear, complete, and unambiguous communications",
        score_1="Frequent unclear or incomplete communications",
        score_3="Generally clear with some issues",
        score_5="Consistently clear and complete"
    ),
    ScoreMetric(
        name="response_time",
        display_name="Response Time",
        description="Speed of response to commands and queries",
        score_1="Slow or unresponsive",
        score_3="Adequate response times",
        score_5="Quick and consistent responses"
    ),
    ScoreMetric(
        name="technical_accuracy",
        display_name="Technical Accuracy",
        description="Correct use of technical terminology and procedures",
        score_1="Frequent technical errors",
        score_3="Generally accurate",
        score_5="Consistently accurate terminology"
    ),
    ScoreMetric(
        name="team_coordination",
        display_name="Team Coordination",
        description="Coordination with other crew members",
        score_1="Little coordination observed",
        score_3="Some coordination patterns",
        score_5="Strong coordination and backup"
    ),
]


@dataclass
class SpeakerScore:
    """Score for a single metric."""
    metric_name: str
    score: int  # 1-5
    evidence: str
    raw_value: float  # The underlying measurement
    supporting_quotes: List[str] = field(default_factory=list)  # Actual quotes
    threshold_info: str = ""  # Score thresholds used for this metric
    pattern_breakdown: Dict[str, int] = field(default_factory=dict)  # Pattern match counts
    calculation_details: str = ""  # Details of how score was calculated


@dataclass
class SpeakerScorecard:
    """Complete scorecard for a speaker."""
    speaker: str
    inferred_role: str
    utterance_count: int
    utterance_percentage: float
    scores: List[SpeakerScore]
    overall_score: float
    strengths: List[str]
    development_areas: List[str]
    example_quotes: List[Dict[str, Any]]


class SpeakerScorecardGenerator:
    """
    Generates evidence-based scorecards for each speaker.

    Calculates metrics per speaker and provides supporting evidence
    for each score assigned.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        role_assignments: Dict[str, str] = None,
        metrics: List[ScoreMetric] = None
    ):
        """
        Initialize the scorecard generator.

        Args:
            transcripts: List of transcript dictionaries
            role_assignments: Optional mapping of speaker to role
            metrics: Optional custom scoring metrics
        """
        self.transcripts = transcripts
        self.role_assignments = role_assignments or {}
        self.metrics = metrics or SCORE_METRICS

        # Pre-compute speaker utterances
        self.speaker_utterances: Dict[str, List[Dict]] = defaultdict(list)
        for t in transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            self.speaker_utterances[speaker].append(t)

        self.total_utterances = len(transcripts)

    def generate_all_scorecards(self) -> Dict[str, SpeakerScorecard]:
        """
        Generate scorecards for all speakers.

        Returns:
            Dictionary mapping speaker ID to their scorecard
        """
        results = {}

        for speaker, utterances in self.speaker_utterances.items():
            results[speaker] = self._generate_scorecard(speaker, utterances)

        return results

    def _generate_scorecard(
        self,
        speaker: str,
        utterances: List[Dict]
    ) -> SpeakerScorecard:
        """Generate scorecard for a single speaker."""
        scores = []

        # Calculate each metric
        scores.append(self._score_protocol_adherence(utterances))
        scores.append(self._score_communication_clarity(utterances))
        scores.append(self._score_response_time(speaker, utterances))
        scores.append(self._score_technical_accuracy(utterances))
        scores.append(self._score_team_coordination(utterances))

        # Calculate overall score
        overall = sum(s.score for s in scores) / len(scores) if scores else 0

        # Identify strengths and development areas
        strengths = self._identify_strengths(scores, utterances)
        development = self._identify_development_areas(scores, utterances)

        # Get example quotes
        examples = self._get_example_quotes(utterances)

        return SpeakerScorecard(
            speaker=speaker,
            inferred_role=self.role_assignments.get(speaker, "Crew Member"),
            utterance_count=len(utterances),
            utterance_percentage=round(len(utterances) / self.total_utterances * 100, 1),
            scores=scores,
            overall_score=round(overall, 1),
            strengths=strengths,
            development_areas=development,
            example_quotes=examples
        )

    def _score_protocol_adherence(self, utterances: List[Dict]) -> SpeakerScore:
        """Score protocol adherence based on formal language usage."""
        protocol_patterns = {
            "acknowledgments": r"(?i)\b(aye|aye aye|acknowledged|understood|copy|roger|affirmative)\b",
            "titles": r"(?i)\b(sir|captain|ma'am)\b",
            "status": r"(?i)\b(reporting|standing by|ready|on station)\b",
            "confirmations": r"(?i)\b(confirm|confirmed|negative)\b",
        }

        matches = 0
        total = len(utterances)
        matching_quotes = []
        pattern_counts = {name: 0 for name in protocol_patterns}

        for u in utterances:
            text = u.get('text', '')
            for pattern_name, pattern in protocol_patterns.items():
                if re.search(pattern, text):
                    matches += 1
                    pattern_counts[pattern_name] += 1
                    # Collect quote with timestamp if available
                    ts = u.get('timestamp', u.get('start_time', ''))
                    if isinstance(ts, (int, float)):
                        ts = f"{int(ts // 60)}:{int(ts % 60):02d}"
                    quote = f'"{text[:100]}..."' if len(text) > 100 else f'"{text}"'
                    if ts:
                        quote = f"[{ts}] {quote}"
                    matching_quotes.append(quote)
                    break

        if total == 0:
            rate = 0
        else:
            rate = matches / total

        # Score thresholds
        threshold_info = "Score 5: ≥30% | Score 4: ≥20% | Score 3: ≥10% | Score 2: ≥5% | Score 1: <5%"

        # Convert rate to 1-5 score
        if rate >= 0.30:
            score = 5
            evidence = "Consistent use of protocols"
        elif rate >= 0.20:
            score = 4
            evidence = "Frequent protocol usage"
        elif rate >= 0.10:
            score = 3
            evidence = "Occasional use of protocols"
        elif rate >= 0.05:
            score = 2
            evidence = "Minimal protocol language"
        else:
            score = 1
            evidence = "No protocol language detected"

        calculation_details = (
            f"Counted utterances containing protocol keywords. "
            f"Rate = {matches}/{total} = {rate*100:.1f}%. "
            f"Threshold for score {score}: {'>=' if score > 1 else '<'}"
            f"{[0.30, 0.20, 0.10, 0.05, 0][5-score]*100:.0f}%"
        )

        return SpeakerScore(
            metric_name="protocol_adherence",
            score=score,
            evidence=f"{evidence} ({matches}/{total} utterances, {rate*100:.1f}%)",
            raw_value=rate,
            supporting_quotes=matching_quotes[:5],
            threshold_info=threshold_info,
            pattern_breakdown=pattern_counts,
            calculation_details=calculation_details
        )

    def _score_communication_clarity(self, utterances: List[Dict]) -> SpeakerScore:
        """Score communication clarity based on confidence and completeness."""
        threshold_info = "Score 5: ≥80% | Score 4: ≥70% | Score 3: ≥60% | Score 2: ≥50% | Score 1: <50%"

        if not utterances:
            return SpeakerScore(
                metric_name="communication_clarity",
                score=1,
                evidence="No communications to assess",
                raw_value=0,
                supporting_quotes=[],
                threshold_info=threshold_info,
                pattern_breakdown={},
                calculation_details="No utterances to analyze"
            )

        # Factors: confidence, sentence completeness, filler words
        confidences = [u.get('confidence', 0) for u in utterances]
        avg_confidence = sum(confidences) / len(confidences)

        # Check for incomplete sentences
        incomplete_patterns = [
            r'\.\.\.$',  # trailing ellipsis
            r',\s*$',    # trailing comma
            r'uh\s*$',   # trailing filler
            r'\?\s*\.\.\.',  # question with ellipsis
        ]

        filler_patterns = [
            r'(?i)\buh+\b',
            r'(?i)\bum+\b',
            r'(?i)\blike\b',
            r'(?i)\byou know\b',
        ]

        incomplete_count = 0
        filler_count = 0
        clear_quotes = []  # High confidence, complete utterances
        unclear_quotes = []  # Low confidence or fillers

        for u in utterances:
            text = u.get('text', '')
            conf = u.get('confidence', 0)
            ts = u.get('timestamp', u.get('start_time', ''))
            if isinstance(ts, (int, float)):
                ts = f"{int(ts // 60)}:{int(ts % 60):02d}"

            is_incomplete = False
            has_filler = False

            for pattern in incomplete_patterns:
                if re.search(pattern, text):
                    incomplete_count += 1
                    is_incomplete = True
                    break
            for pattern in filler_patterns:
                if re.search(pattern, text):
                    filler_count += 1
                    has_filler = True
                    break

            # Collect examples
            quote = f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"'
            if ts:
                quote = f"[{ts}] {quote}"

            if conf >= 0.7 and not is_incomplete and not has_filler:
                clear_quotes.append(quote)
            elif is_incomplete or has_filler or conf < 0.5:
                unclear_quotes.append(quote)

        total = len(utterances)
        incomplete_rate = incomplete_count / total if total > 0 else 0
        filler_rate = filler_count / total if total > 0 else 0

        # Combined clarity score (confidence weighted most heavily)
        clarity = avg_confidence * 0.6 + (1 - incomplete_rate) * 0.2 + (1 - filler_rate) * 0.2

        pattern_breakdown = {
            "incomplete_sentences": incomplete_count,
            "filler_words": filler_count,
            "clear_utterances": len(clear_quotes),
            "unclear_utterances": len(unclear_quotes)
        }

        calculation_details = (
            f"Clarity = (confidence × 0.6) + (completeness × 0.2) + (no-fillers × 0.2). "
            f"Confidence: {avg_confidence:.2f}, Incomplete: {incomplete_rate*100:.1f}%, "
            f"Fillers: {filler_rate*100:.1f}%. Final clarity: {clarity*100:.1f}%"
        )

        if clarity >= 0.80:
            score = 5
            evidence = f"Consistently clear (confidence: {avg_confidence:.2f})"
            quotes = clear_quotes[:3]
        elif clarity >= 0.70:
            score = 4
            evidence = f"Generally clear (confidence: {avg_confidence:.2f})"
            quotes = clear_quotes[:3]
        elif clarity >= 0.60:
            score = 3
            evidence = f"Some clarity issues (confidence: {avg_confidence:.2f}, {incomplete_count} incomplete)"
            quotes = unclear_quotes[:3] if unclear_quotes else clear_quotes[:3]
        elif clarity >= 0.50:
            score = 2
            evidence = f"Clarity needs work (confidence: {avg_confidence:.2f}, {filler_count} fillers)"
            quotes = unclear_quotes[:3]
        else:
            score = 1
            evidence = f"Frequent unclear communications (confidence: {avg_confidence:.2f})"
            quotes = unclear_quotes[:3]

        return SpeakerScore(
            metric_name="communication_clarity",
            score=score,
            evidence=evidence,
            raw_value=clarity,
            supporting_quotes=quotes,
            threshold_info=threshold_info,
            pattern_breakdown=pattern_breakdown,
            calculation_details=calculation_details
        )

    def _score_response_time(self, speaker: str, utterances: List[Dict]) -> SpeakerScore:
        """Score response time based on command-response patterns."""
        threshold_info = "Score 5: <3s | Score 4: <5s | Score 3: <8s | Score 2: <12s | Score 1: ≥12s"

        # Find responses from this speaker to other speakers' commands
        response_times = []
        response_examples = []

        command_patterns = {
            "station_calls": r"(?i)(helm|tactical|science|engineering|operations),?\s",
            "status_requests": r"(?i)(report|status|what's|how's)",
            "action_commands": r"(?i)(set course|engage|fire|launch)",
        }

        pattern_matches = {name: 0 for name in command_patterns}

        for i, t in enumerate(self.transcripts):
            # Check if this is a command to our speaker
            t_speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            if t_speaker == speaker:
                continue

            text = t.get('text', '')
            matched_pattern = None
            for pattern_name, pattern in command_patterns.items():
                if re.search(pattern, text):
                    matched_pattern = pattern_name
                    break

            if not matched_pattern:
                continue

            # Look for response from our speaker in next few utterances
            for j in range(i + 1, min(i + 5, len(self.transcripts))):
                next_t = self.transcripts[j]
                next_speaker = next_t.get('speaker') or next_t.get('speaker_id') or 'unknown'

                if next_speaker == speaker:
                    # Calculate time delta
                    try:
                        t1 = self._parse_timestamp(t.get('timestamp', ''))
                        t2 = self._parse_timestamp(next_t.get('timestamp', ''))
                        if t1 and t2:
                            delta = (t2 - t1).total_seconds()
                            if 0 < delta < 30:
                                response_times.append(delta)
                                pattern_matches[matched_pattern] += 1
                                # Record the command-response pair
                                cmd_text = text[:50] + "..." if len(text) > 50 else text
                                resp_text = next_t.get('text', '')[:50]
                                response_examples.append(
                                    f'Command: "{cmd_text}" → Response ({delta:.1f}s): "{resp_text}"'
                                )
                    except (ValueError, TypeError):
                        pass
                    break

        if not response_times:
            # If no clear command-response patterns, use utterance frequency
            utterance_rate = len(utterances) / self.total_utterances if self.total_utterances > 0 else 0
            # Get some example utterances
            example_quotes = []
            for u in utterances[:3]:
                text = u.get('text', '')
                ts = u.get('timestamp', u.get('start_time', ''))
                if isinstance(ts, (int, float)):
                    ts = f"{int(ts // 60)}:{int(ts % 60):02d}"
                quote = f'"{text[:60]}..."' if len(text) > 60 else f'"{text}"'
                if ts:
                    quote = f"[{ts}] {quote}"
                example_quotes.append(quote)

            calculation_details = (
                f"No command-response pairs detected. Using engagement rate: "
                f"{len(utterances)}/{self.total_utterances} = {utterance_rate*100:.1f}%"
            )

            if utterance_rate >= 0.20:
                score = 4
                evidence = f"High engagement ({len(utterances)} utterances, no command-response pairs detected)"
            elif utterance_rate >= 0.10:
                score = 3
                evidence = f"Moderate engagement ({len(utterances)} utterances)"
            else:
                score = 2
                evidence = f"Limited engagement ({len(utterances)} utterances)"

            return SpeakerScore(
                metric_name="response_time",
                score=score,
                evidence=evidence,
                raw_value=utterance_rate,
                supporting_quotes=example_quotes,
                threshold_info="Engagement-based: Score 4: ≥20% | Score 3: ≥10% | Score 2: <10%",
                pattern_breakdown={"utterance_count": len(utterances), "total_utterances": self.total_utterances},
                calculation_details=calculation_details
            )

        avg_response = sum(response_times) / len(response_times)

        calculation_details = (
            f"Measured {len(response_times)} command-response pairs. "
            f"Response times: min={min(response_times):.1f}s, max={max(response_times):.1f}s, "
            f"avg={avg_response:.1f}s"
        )

        if avg_response < 3:
            score = 5
            evidence = f"Quick responses (avg {avg_response:.1f}s)"
        elif avg_response < 5:
            score = 4
            evidence = f"Good response times (avg {avg_response:.1f}s)"
        elif avg_response < 8:
            score = 3
            evidence = f"Adequate responses (avg {avg_response:.1f}s)"
        elif avg_response < 12:
            score = 2
            evidence = f"Slow responses (avg {avg_response:.1f}s)"
        else:
            score = 1
            evidence = f"Very slow responses (avg {avg_response:.1f}s)"

        return SpeakerScore(
            metric_name="response_time",
            score=score,
            evidence=evidence,
            raw_value=avg_response,
            supporting_quotes=response_examples[:3],
            threshold_info=threshold_info,
            pattern_breakdown=pattern_matches,
            calculation_details=calculation_details
        )

    def _score_technical_accuracy(self, utterances: List[Dict]) -> SpeakerScore:
        """Score technical accuracy based on terminology usage."""
        technical_patterns = {
            "navigation": r"(?i)\b(kilometers|km|meters|range|bearing|heading)\b",
            "systems": r"(?i)\b(shields?|hull|power|reactor|warp|impulse)\b",
            "weapons": r"(?i)\b(phasers?|torpedoes?|weapons?|missiles?)\b",
            "sensors": r"(?i)\b(sensors?|scanning|detecting|readings?)\b",
            "spatial": r"(?i)\b(coordinates?|sector|quadrant|orbit)\b",
            "timing": r"(?i)\b(eta|arrival|departure|docking)\b",
        }

        threshold_info = "Score 5: ≥40% | Score 4: ≥25% | Score 3: ≥15% | Score 2: ≥5% | Score 1: <5%"

        matches = 0
        total = len(utterances)
        technical_quotes = []
        pattern_counts = {name: 0 for name in technical_patterns}

        for u in utterances:
            text = u.get('text', '')
            for pattern_name, pattern in technical_patterns.items():
                if re.search(pattern, text):
                    matches += 1
                    pattern_counts[pattern_name] += 1
                    # Collect quote with timestamp
                    ts = u.get('timestamp', u.get('start_time', ''))
                    if isinstance(ts, (int, float)):
                        ts = f"{int(ts // 60)}:{int(ts % 60):02d}"
                    quote = f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"'
                    if ts:
                        quote = f"[{ts}] {quote}"
                    technical_quotes.append(quote)
                    break

        if total == 0:
            rate = 0
        else:
            rate = matches / total

        calculation_details = (
            f"Counted utterances with technical terminology. "
            f"Rate = {matches}/{total} = {rate*100:.1f}%. "
            f"Categories: {', '.join(f'{k}={v}' for k, v in pattern_counts.items() if v > 0)}"
        )

        if rate >= 0.40:
            score = 5
            evidence = f"Rich technical vocabulary ({matches} technical utterances)"
        elif rate >= 0.25:
            score = 4
            evidence = f"Good technical language ({matches} technical utterances)"
        elif rate >= 0.15:
            score = 3
            evidence = f"Adequate terminology ({matches} technical utterances)"
        elif rate >= 0.05:
            score = 2
            evidence = f"Limited technical language ({matches} technical utterances)"
        else:
            score = 1
            evidence = "Minimal technical terminology"

        return SpeakerScore(
            metric_name="technical_accuracy",
            score=score,
            evidence=evidence,
            raw_value=rate,
            supporting_quotes=technical_quotes[:5],
            threshold_info=threshold_info,
            pattern_breakdown=pattern_counts,
            calculation_details=calculation_details
        )

    def _score_team_coordination(self, utterances: List[Dict]) -> SpeakerScore:
        """Score team coordination based on collaborative patterns."""
        coordination_patterns = {
            "assistance": r"(?i)\b(help|assist|support|backup|cover)\b",
            "readiness": r"(?i)\b(ready|standing by|on it|got it)\b",
            "acknowledgment": r"(?i)\b(confirm|acknowledged|copy|roger)\b",
            "team_language": r"(?i)\b(together|team|we need|let's)\b",
            "status_sharing": r"(?i)\b(status|report|update)\b",
        }

        threshold_info = "Score 5: ≥35% | Score 4: ≥25% | Score 3: ≥15% | Score 2: ≥5% | Score 1: <5%"

        matches = 0
        total = len(utterances)
        coordination_quotes = []
        pattern_counts = {name: 0 for name in coordination_patterns}

        for u in utterances:
            text = u.get('text', '')
            for pattern_name, pattern in coordination_patterns.items():
                if re.search(pattern, text):
                    matches += 1
                    pattern_counts[pattern_name] += 1
                    # Collect quote with timestamp
                    ts = u.get('timestamp', u.get('start_time', ''))
                    if isinstance(ts, (int, float)):
                        ts = f"{int(ts // 60)}:{int(ts % 60):02d}"
                    quote = f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"'
                    if ts:
                        quote = f"[{ts}] {quote}"
                    coordination_quotes.append(quote)
                    break

        if total == 0:
            rate = 0
        else:
            rate = matches / total

        calculation_details = (
            f"Counted utterances with coordination language. "
            f"Rate = {matches}/{total} = {rate*100:.1f}%. "
            f"Types: {', '.join(f'{k}={v}' for k, v in pattern_counts.items() if v > 0)}"
        )

        if rate >= 0.35:
            score = 5
            evidence = f"Strong coordination ({matches} coordination utterances)"
        elif rate >= 0.25:
            score = 4
            evidence = f"Good coordination ({matches} coordination utterances)"
        elif rate >= 0.15:
            score = 3
            evidence = f"Some coordination ({matches} coordination utterances)"
        elif rate >= 0.05:
            score = 2
            evidence = f"Limited coordination ({matches} coordination utterances)"
        else:
            score = 1
            evidence = "Little coordination observed"

        return SpeakerScore(
            metric_name="team_coordination",
            score=score,
            evidence=evidence,
            raw_value=rate,
            supporting_quotes=coordination_quotes[:5],
            threshold_info=threshold_info,
            pattern_breakdown=pattern_counts,
            calculation_details=calculation_details
        )

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp to datetime."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                if 'T' in ts:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return datetime.strptime(ts, '%H:%M:%S')
            except (ValueError, TypeError):
                pass
        return None

    def _identify_strengths(
        self,
        scores: List[SpeakerScore],
        utterances: List[Dict]
    ) -> List[str]:
        """Identify speaker strengths based on scores."""
        strengths = []

        for score in scores:
            if score.score >= 4:
                metric = next(
                    (m for m in self.metrics if m.name == score.metric_name),
                    None
                )
                if metric:
                    strengths.append(f"{metric.display_name}: {score.evidence}")

        # Add volume-based strength if applicable
        if len(utterances) / self.total_utterances > 0.20:
            strengths.append(f"High engagement with {len(utterances)} utterances")

        return strengths[:3]  # Limit to top 3

    def _identify_development_areas(
        self,
        scores: List[SpeakerScore],
        utterances: List[Dict]
    ) -> List[str]:
        """Identify development areas based on scores."""
        areas = []

        for score in scores:
            if score.score <= 2:
                metric = next(
                    (m for m in self.metrics if m.name == score.metric_name),
                    None
                )
                if metric:
                    areas.append(f"{metric.display_name}: {score.evidence}")

        return areas[:3]  # Limit to top 3

    def _get_example_quotes(self, utterances: List[Dict]) -> List[Dict[str, Any]]:
        """Get representative example quotes."""
        # Sort by confidence
        sorted_u = sorted(utterances, key=lambda x: x.get('confidence', 0), reverse=True)

        examples = []
        for u in sorted_u[:5]:
            ts = u.get('timestamp', '')
            if isinstance(ts, datetime):
                ts = ts.strftime('%H:%M:%S')
            elif isinstance(ts, str) and 'T' in ts:
                ts = ts.split('T')[1][:8]

            examples.append({
                'timestamp': ts,
                'text': u.get('text', ''),
                'confidence': u.get('confidence', 0)
            })

        return examples

    def generate_scorecard_section(self, speaker: str) -> str:
        """
        Generate markdown scorecard section for a speaker.

        Args:
            speaker: Speaker ID

        Returns:
            Markdown formatted scorecard
        """
        scorecards = self.generate_all_scorecards()
        if speaker not in scorecards:
            return f"### {speaker}\n\nNo data available."

        sc = scorecards[speaker]

        lines = [
            f"### {sc.speaker} ({sc.inferred_role})",
            "",
            f"**Utterances:** {sc.utterance_count} ({sc.utterance_percentage}% of total)",
            "",
            "| Metric | Score | Evidence |",
            "| --- | --- | --- |"
        ]

        for score in sc.scores:
            metric = next(
                (m for m in self.metrics if m.name == score.metric_name),
                None
            )
            name = metric.display_name if metric else score.metric_name
            lines.append(f"| {name} | {score.score}/5 | {score.evidence} |")

        lines.append("")
        lines.append(f"**Overall Score:** {sc.overall_score}/5")
        lines.append("")

        if sc.strengths:
            lines.append("**Strengths:**")
            for s in sc.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if sc.development_areas:
            lines.append("**Development Areas:**")
            for d in sc.development_areas:
                lines.append(f"- {d}")
            lines.append("")

        return "\n".join(lines)

    def generate_all_scorecards_section(self) -> str:
        """
        Generate markdown section with all speaker scorecards.

        Returns:
            Markdown formatted scorecards for all speakers
        """
        scorecards = self.generate_all_scorecards()

        # Sort by utterance count
        sorted_speakers = sorted(
            scorecards.keys(),
            key=lambda x: scorecards[x].utterance_count,
            reverse=True
        )

        lines = ["## Crew Performance Scorecards", ""]

        for speaker in sorted_speakers:
            lines.append(self.generate_scorecard_section(speaker))
            lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all scorecard data
        """
        scorecards = self.generate_all_scorecards()

        return {
            'scorecards_section': self.generate_all_scorecards_section(),
            'speaker_scorecards': {
                speaker: {
                    'role': sc.inferred_role,
                    'utterance_count': sc.utterance_count,
                    'utterance_percentage': sc.utterance_percentage,
                    'overall_score': sc.overall_score,
                    'scores': {
                        s.metric_name: {
                            'score': s.score,
                            'evidence': s.evidence,
                            'raw_value': s.raw_value,
                            'supporting_quotes': s.supporting_quotes,
                            'threshold_info': s.threshold_info,
                            'pattern_breakdown': s.pattern_breakdown,
                            'calculation_details': s.calculation_details
                        }
                        for s in sc.scores
                    },
                    'strengths': sc.strengths,
                    'development_areas': sc.development_areas
                }
                for speaker, sc in scorecards.items()
            }
        }
