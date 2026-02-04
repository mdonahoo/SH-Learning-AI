"""
Captain-specific leadership assessment module.

Assesses the captain's leadership across multiple dimensions using
transcript analysis and optional game telemetry data. Unlike the
generic speaker scorecard, this module evaluates behaviors unique
to the command role: delegation, crew engagement, information flow,
crisis response, and praise/feedback.

Reference: Bridge Resource Management (BRM) and Crew Resource
Management (CRM) frameworks adapted for bridge simulator training.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LeadershipDimension:
    """Assessment result for a single leadership dimension."""
    name: str
    display_name: str
    score: int  # 1-5, or 0 for insufficient data
    count: int
    evidence: str
    examples: List[str]
    description: str


@dataclass
class CaptainAssessment:
    """Complete captain leadership assessment."""
    captain_speaker: str
    captain_role: str
    dimensions: Dict[str, LeadershipDimension]
    overall_score: float
    strengths: List[str]
    development_areas: List[str]
    utterance_count: int


class CaptainLeadershipAssessor:
    """
    Assesses captain-specific leadership behaviors from transcripts.

    Identifies the captain speaker and evaluates their leadership across
    dimensions that are specific to the command role:

    - Delegation: Does the captain address stations by name and let crew
      handle their areas vs. micromanaging?
    - Crew Engagement: Does the captain solicit input before decisions?
    - Information Flow: Does the captain request and share status updates?
    - Praise/Feedback: Does the captain acknowledge crew contributions?
    - Crisis Response: How does communication change during high-intensity
      moments (if telemetry shows alerts/combat)?

    Attributes:
        transcripts: List of transcript dictionaries
        role_assignments: Mapping of speaker_id to role
        telemetry_events: Optional game telemetry events
    """

    # Patterns for each leadership dimension
    DELEGATION_PATTERNS = {
        "station_addressing": r"(?i)\b(helm|tactical|science|engineering|operations|communications)\b,?\s",
        "task_assignment": r"(?i)\b(you handle|take care of|your call|you decide|i need you to)\b",
        "authority_granting": r"(?i)\b(at your discretion|when you're ready|your judgment)\b",
    }

    CREW_ENGAGEMENT_PATTERNS = {
        "soliciting_input": r"(?i)\b(what do you think|any ideas|suggestions|recommendations|your thoughts)\b",
        "asking_assessment": r"(?i)\b(your assessment|what do you see|what are we looking at|what's your read)\b",
        "inviting_options": r"(?i)\b(options|alternatives|what can we do|how should we)\b",
    }

    INFORMATION_FLOW_PATTERNS = {
        "requesting_status": r"(?i)\b(status|report|update|what's the|how's the|where are we)\b",
        "sharing_intent": r"(?i)\b(here's the plan|our plan is|we're going to|the plan is)\b",
        "situational_briefing": r"(?i)\b(situation is|we have|we're facing|here's what)\b",
    }

    PRAISE_FEEDBACK_PATTERNS = {
        "acknowledgment": r"(?i)\b(good work|well done|nice|great job|excellent|thank you|thanks)\b",
        "specific_praise": r"(?i)\b(nice (shot|work|call|move)|good (call|thinking|catch))\b",
        "encouragement": r"(?i)\b(keep it up|stay sharp|stay focused|you got this|we can do this)\b",
    }

    DIRECTIVE_PATTERNS = {
        "direct_commands": r"(?i)\b(set course|engage|fire|launch|raise shields|red alert|yellow alert|battle stations)\b",
        "orders": r"(?i)\b(make it so|execute|do it|go ahead|proceed)\b",
        "halting": r"(?i)\b(hold|stop|cease fire|stand down|belay that|wait)\b",
    }

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        role_assignments: Optional[Dict[str, str]] = None,
        telemetry_events: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize captain leadership assessor.

        Args:
            transcripts: List of transcript dictionaries with 'text',
                'speaker'/'speaker_id', 'timestamp'/'start_time'
            role_assignments: Mapping of speaker_id to inferred role
            telemetry_events: Optional game telemetry events for
                crisis response assessment
        """
        self.transcripts = transcripts
        self.role_assignments = role_assignments or {}
        self.telemetry_events = telemetry_events or []

        # Pre-compute speaker utterances
        self.speaker_utterances: Dict[str, List[Dict]] = defaultdict(list)
        for t in transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            self.speaker_utterances[speaker].append(t)

    def identify_captain(self) -> Optional[str]:
        """
        Identify the captain speaker from role assignments.

        Returns:
            Speaker ID of the captain, or None if not identified
        """
        captain_roles = {'Captain', 'Captain/Command', 'Command', 'CO'}
        for speaker_id, role in self.role_assignments.items():
            if role in captain_roles:
                return speaker_id

        # Fallback: look for speaker whose role contains 'captain' or 'command'
        for speaker_id, role in self.role_assignments.items():
            if 'captain' in role.lower() or 'command' in role.lower():
                return speaker_id

        return None

    def assess(self) -> Optional[CaptainAssessment]:
        """
        Perform captain leadership assessment.

        Returns:
            CaptainAssessment or None if captain cannot be identified
        """
        captain_id = self.identify_captain()
        if not captain_id:
            logger.info("Captain not identified — skipping captain leadership assessment")
            return None

        captain_utterances = self.speaker_utterances.get(captain_id, [])
        if not captain_utterances:
            logger.info(f"Captain {captain_id} has no utterances — skipping assessment")
            return None

        captain_role = self.role_assignments.get(captain_id, 'Captain')

        # Assess each dimension
        dimensions = {}
        dimensions['delegation'] = self._assess_delegation(captain_utterances)
        dimensions['crew_engagement'] = self._assess_crew_engagement(captain_utterances)
        dimensions['information_flow'] = self._assess_information_flow(captain_utterances)
        dimensions['praise_feedback'] = self._assess_praise_feedback(captain_utterances)
        dimensions['crisis_response'] = self._assess_crisis_response(
            captain_id, captain_utterances
        )

        # Calculate overall score (exclude 0 = insufficient data)
        scored = [d for d in dimensions.values() if d.score > 0]
        overall = sum(d.score for d in scored) / len(scored) if scored else 0

        # Identify strengths and development areas
        strengths = [
            f"{d.display_name}: {d.evidence}"
            for d in sorted(dimensions.values(), key=lambda x: x.score, reverse=True)
            if d.score >= 4
        ][:3]

        development_areas = [
            f"{d.display_name}: {d.evidence}"
            for d in sorted(dimensions.values(), key=lambda x: x.score)
            if 1 <= d.score <= 2
        ][:3]

        return CaptainAssessment(
            captain_speaker=captain_id,
            captain_role=captain_role,
            dimensions=dimensions,
            overall_score=round(overall, 1),
            strengths=strengths,
            development_areas=development_areas,
            utterance_count=len(captain_utterances),
        )

    def _assess_dimension(
        self,
        utterances: List[Dict],
        patterns: Dict[str, str],
        dimension_name: str,
        display_name: str,
        description: str
    ) -> LeadershipDimension:
        """
        Generic dimension assessment using pattern matching.

        Args:
            utterances: Captain's utterances
            patterns: Dict of pattern_name -> regex pattern
            dimension_name: Internal dimension name
            display_name: Human-readable dimension name
            description: Dimension description

        Returns:
            LeadershipDimension with score and evidence
        """
        total = len(utterances)
        if total == 0:
            return LeadershipDimension(
                name=dimension_name,
                display_name=display_name,
                score=0,
                count=0,
                evidence="No captain utterances to assess",
                examples=[],
                description=description,
            )

        match_count = 0
        examples = []

        for u in utterances:
            text = u.get('text', '')
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, text):
                    match_count += 1
                    if len(examples) < 5:
                        examples.append(text[:120])
                    break  # Count each utterance once

        rate = match_count / total

        # Score thresholds — consistent across dimensions
        if rate >= 0.25:
            score = 5
        elif rate >= 0.15:
            score = 4
        elif rate >= 0.08:
            score = 3
        elif rate >= 0.03:
            score = 2
        else:
            score = 1

        evidence = f"{match_count} of {total} utterances ({rate*100:.0f}%)"

        return LeadershipDimension(
            name=dimension_name,
            display_name=display_name,
            score=score,
            count=match_count,
            evidence=evidence,
            examples=examples,
            description=description,
        )

    def _assess_delegation(self, utterances: List[Dict]) -> LeadershipDimension:
        """Assess captain's delegation effectiveness."""
        return self._assess_dimension(
            utterances,
            self.DELEGATION_PATTERNS,
            "delegation",
            "Delegation",
            "Addresses stations by name, assigns tasks, grants authority to crew"
        )

    def _assess_crew_engagement(self, utterances: List[Dict]) -> LeadershipDimension:
        """Assess captain's crew engagement — soliciting input before decisions."""
        return self._assess_dimension(
            utterances,
            self.CREW_ENGAGEMENT_PATTERNS,
            "crew_engagement",
            "Crew Engagement",
            "Solicits input, asks for assessments, invites options before major decisions"
        )

    def _assess_information_flow(self, utterances: List[Dict]) -> LeadershipDimension:
        """Assess captain's information flow management."""
        return self._assess_dimension(
            utterances,
            self.INFORMATION_FLOW_PATTERNS,
            "information_flow",
            "Information Flow",
            "Requests status updates, shares plans, briefs crew on situation"
        )

    def _assess_praise_feedback(self, utterances: List[Dict]) -> LeadershipDimension:
        """Assess captain's praise and feedback to crew."""
        return self._assess_dimension(
            utterances,
            self.PRAISE_FEEDBACK_PATTERNS,
            "praise_feedback",
            "Praise & Feedback",
            "Acknowledges crew contributions, provides specific praise, encourages"
        )

    def _assess_crisis_response(
        self,
        captain_id: str,
        captain_utterances: List[Dict]
    ) -> LeadershipDimension:
        """
        Assess captain's communication during high-intensity moments.

        Compares directive vs. collaborative language during crisis
        events (red alert, combat, damage) detected from telemetry.
        If no telemetry, uses general directive pattern frequency.

        Args:
            captain_id: Captain speaker ID
            captain_utterances: Captain's utterances

        Returns:
            LeadershipDimension for crisis response
        """
        # If we have telemetry, find crisis periods
        crisis_utterances = self._get_crisis_utterances(
            captain_id, captain_utterances
        )

        if crisis_utterances:
            # During crisis, we want to see MORE directives (decisive leadership)
            # but also some engagement (not just barking orders)
            target = crisis_utterances
            description = (
                "Communication during high-intensity events — measures decisiveness "
                "and crew coordination under pressure"
            )
        else:
            # No crisis detected — use general directives as proxy
            target = captain_utterances
            description = (
                "Command presence — frequency of clear directives and orders. "
                "No crisis events detected in telemetry."
            )

        return self._assess_dimension(
            target,
            self.DIRECTIVE_PATTERNS,
            "crisis_response",
            "Command Presence",
            description
        )

    def _get_crisis_utterances(
        self,
        captain_id: str,
        captain_utterances: List[Dict]
    ) -> List[Dict]:
        """
        Get captain utterances that occurred during crisis events.

        Args:
            captain_id: Captain speaker ID
            captain_utterances: Captain's utterances

        Returns:
            List of utterances during crisis periods (empty if no telemetry)
        """
        if not self.telemetry_events:
            return []

        # Find crisis event timestamps
        crisis_keywords = {
            'red_alert', 'combat', 'damage', 'hull_breach', 'shields_down',
            'enemy_detected', 'hostile', 'attack', 'weapon_fire'
        }

        crisis_times: List[Tuple[float, float]] = []
        for event in self.telemetry_events:
            event_type = str(event.get('event_type', '')).lower()
            category = str(event.get('category', '')).lower()
            description = str(event.get('description', '')).lower()

            is_crisis = (
                event_type in crisis_keywords
                or category in {'combat', 'damage', 'alert'}
                or any(kw in description for kw in ['red alert', 'combat', 'damage', 'hostile'])
            )

            if is_crisis:
                event_time = event.get('relative_time', event.get('time', 0))
                if isinstance(event_time, (int, float)):
                    # Crisis window: 30 seconds before to 60 seconds after event
                    crisis_times.append((event_time - 30, event_time + 60))

        if not crisis_times:
            return []

        # Find captain utterances within crisis windows
        crisis_utterances = []
        for u in captain_utterances:
            u_time = u.get('start_time', u.get('timestamp', 0))
            if isinstance(u_time, (int, float)):
                for start, end in crisis_times:
                    if start <= u_time <= end:
                        crisis_utterances.append(u)
                        break

        return crisis_utterances

    def get_structured_results(self) -> Optional[Dict[str, Any]]:
        """
        Get structured results suitable for JSON serialization and LLM prompts.

        Returns:
            Dictionary with captain leadership data, or None if assessment
            could not be performed
        """
        assessment = self.assess()
        if not assessment:
            return None

        return {
            'captain_speaker': assessment.captain_speaker,
            'captain_role': assessment.captain_role,
            'overall_score': assessment.overall_score,
            'utterance_count': assessment.utterance_count,
            'dimensions': {
                name: {
                    'display_name': dim.display_name,
                    'score': dim.score,
                    'count': dim.count,
                    'evidence': dim.evidence,
                    'examples': dim.examples,
                    'description': dim.description,
                }
                for name, dim in assessment.dimensions.items()
            },
            'strengths': assessment.strengths,
            'development_areas': assessment.development_areas,
        }
