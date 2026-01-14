"""
Character voice analysis and personality inference.

Analyzes transcript patterns and behavior to create distinct
character voices for narrative generation.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CharacterArchetype(Enum):
    """Character archetypes for crew members."""

    # Leadership archetypes
    THE_COMMANDER = "commander"  # Decisive, takes charge
    THE_DIPLOMAT = "diplomat"  # Measured, seeks consensus
    THE_MAVERICK = "maverick"  # Unconventional, bold

    # Crew archetypes
    THE_VETERAN = "veteran"  # Experienced, steady
    THE_ROOKIE = "rookie"  # Eager, learning
    THE_SPECIALIST = "specialist"  # Technical expert
    THE_HOTHEAD = "hothead"  # Quick to act, emotional
    THE_ANALYST = "analyst"  # Data-driven, cautious
    THE_STEADY_HAND = "steady_hand"  # Calm under pressure

    # Support archetypes
    THE_VOICE_OF_REASON = "voice_of_reason"  # Provides perspective
    THE_COMIC_RELIEF = "comic_relief"  # Uses humor
    THE_WORRIER = "worrier"  # Anticipates problems


class StressResponse(Enum):
    """How a character responds to stress."""

    CALM = "calm"  # Maintains composure
    FOCUSED = "focused"  # Narrows attention
    ENERGIZED = "energized"  # Becomes more active
    TERSE = "terse"  # Shorter communications
    VERBOSE = "verbose"  # Over-explains
    PANICKED = "panicked"  # Loses composure


class CommunicationStyle(Enum):
    """Communication style patterns."""

    FORMAL = "formal"  # Uses protocol, titles
    CASUAL = "casual"  # Informal, friendly
    TECHNICAL = "technical"  # Heavy jargon
    DIRECT = "direct"  # Short, to the point
    EXPLANATORY = "explanatory"  # Provides context
    QUESTIONING = "questioning"  # Asks more than states


@dataclass
class CharacterVoice:
    """Complete voice profile for a character."""

    # Identity
    speaker_id: str
    role: str  # Captain, Helm, Tactical, etc.
    name: Optional[str] = None  # If we can infer a name

    # Personality
    archetype: CharacterArchetype = CharacterArchetype.THE_VETERAN
    stress_response: StressResponse = StressResponse.FOCUSED
    communication_style: CommunicationStyle = CommunicationStyle.DIRECT

    # Speech patterns (actual phrases they use)
    signature_phrases: List[str] = field(default_factory=list)
    vocabulary_markers: List[str] = field(default_factory=list)

    # Statistics
    total_utterances: int = 0
    avg_utterance_length: float = 0.0
    question_ratio: float = 0.0  # Portion of utterances that are questions
    exclamation_ratio: float = 0.0  # Portion with exclamations
    protocol_usage: float = 0.0  # How often they use formal protocol

    # Behavioral patterns
    speaks_first_in_crisis: bool = False
    asks_before_acting: bool = False
    uses_humor: bool = False
    gives_orders: bool = False
    requests_confirmation: bool = False

    # Narrative description
    voice_description: str = ""

    def get_narrative_intro(self) -> str:
        """Get a narrative introduction for this character."""
        archetype_intros = {
            CharacterArchetype.THE_COMMANDER: "commanding presence",
            CharacterArchetype.THE_DIPLOMAT: "measured diplomat",
            CharacterArchetype.THE_MAVERICK: "unconventional thinker",
            CharacterArchetype.THE_VETERAN: "seasoned veteran",
            CharacterArchetype.THE_ROOKIE: "eager newcomer",
            CharacterArchetype.THE_SPECIALIST: "technical expert",
            CharacterArchetype.THE_HOTHEAD: "quick-tempered officer",
            CharacterArchetype.THE_ANALYST: "analytical mind",
            CharacterArchetype.THE_STEADY_HAND: "unflappable presence",
            CharacterArchetype.THE_VOICE_OF_REASON: "voice of reason",
            CharacterArchetype.THE_COMIC_RELIEF: "resident wit",
            CharacterArchetype.THE_WORRIER: "cautious planner",
        }

        return archetype_intros.get(self.archetype, "crew member")


@dataclass
class CharacterRelationship:
    """Relationship between two characters."""

    character_a: str
    character_b: str
    interaction_count: int = 0
    response_pattern: str = ""  # "A often responds to B"
    dynamic: str = ""  # "mentor-student", "rivals", "partners"


class CharacterAnalyzer:
    """
    Analyzes transcripts and behavior to build character profiles.

    Uses linguistic patterns, timing, and behavior to infer
    personality traits for narrative generation.
    """

    # Protocol indicators
    PROTOCOL_PHRASES = {
        "aye", "aye aye", "acknowledged", "confirmed", "affirmative",
        "negative", "copy that", "roger", "understood", "sir", "ma'am",
        "captain", "commander", "reporting", "status report",
    }

    # Question indicators
    QUESTION_WORDS = {"what", "where", "when", "why", "how", "who", "which", "can", "could", "should", "would", "is", "are", "do", "does"}

    # Order indicators
    ORDER_PHRASES = {
        "engage", "fire", "evasive", "raise shields", "red alert",
        "set course", "on screen", "report", "status", "disengage",
        "all hands", "battle stations", "stand down",
    }

    # Humor indicators
    HUMOR_MARKERS = {"haha", "lol", "heh", "joke", "kidding", "funny", "laugh"}

    def __init__(self) -> None:
        """Initialize the character analyzer."""
        self.characters: Dict[str, CharacterVoice] = {}
        self.relationships: List[CharacterRelationship] = []

    def analyze(
        self,
        transcripts: List[Dict[str, Any]],
        events: Optional[List[Dict[str, Any]]] = None,
        station_assignments: Optional[Dict[str, str]] = None,
    ) -> Dict[str, CharacterVoice]:
        """
        Analyze transcripts to build character profiles.

        Args:
            transcripts: List of transcript entries with speaker, text, timestamp
            events: Optional telemetry events for behavioral context
            station_assignments: Optional mapping of speaker_id to station

        Returns:
            Dictionary of speaker_id to CharacterVoice
        """
        self.characters = {}
        events = events or []
        station_assignments = station_assignments or {}

        # Group transcripts by speaker
        by_speaker = self._group_by_speaker(transcripts)

        # Analyze each speaker
        for speaker_id, utterances in by_speaker.items():
            voice = self._analyze_speaker(
                speaker_id,
                utterances,
                station_assignments.get(speaker_id, "Unknown"),
            )

            # Enhance with behavioral data if available
            if events:
                voice = self._enhance_with_behavior(voice, events, transcripts)

            self.characters[speaker_id] = voice

        # Analyze relationships
        self.relationships = self._analyze_relationships(transcripts)

        # Assign archetypes based on full analysis
        self._assign_archetypes()

        # Generate voice descriptions
        for voice in self.characters.values():
            voice.voice_description = self._generate_voice_description(voice)

        logger.info(f"Analyzed {len(self.characters)} characters")
        return self.characters

    def _group_by_speaker(
        self,
        transcripts: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group transcripts by speaker ID."""
        by_speaker: Dict[str, List[Dict[str, Any]]] = {}

        for t in transcripts:
            speaker = t.get("speaker", "Unknown")
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(t)

        return by_speaker

    def _analyze_speaker(
        self,
        speaker_id: str,
        utterances: List[Dict[str, Any]],
        role: str,
    ) -> CharacterVoice:
        """
        Analyze a single speaker's utterances.

        Args:
            speaker_id: Speaker identifier
            utterances: All utterances from this speaker
            role: Station/role assignment

        Returns:
            CharacterVoice profile
        """
        voice = CharacterVoice(
            speaker_id=speaker_id,
            role=role,
            total_utterances=len(utterances),
        )

        if not utterances:
            return voice

        # Extract all text
        texts = [u.get("text", "") for u in utterances]
        all_text = " ".join(texts).lower()

        # Calculate basic statistics
        lengths = [len(t.split()) for t in texts]
        voice.avg_utterance_length = sum(lengths) / len(lengths) if lengths else 0

        # Question ratio
        questions = sum(1 for t in texts if "?" in t or self._starts_with_question(t))
        voice.question_ratio = questions / len(texts)

        # Exclamation ratio
        exclamations = sum(1 for t in texts if "!" in t)
        voice.exclamation_ratio = exclamations / len(texts)

        # Protocol usage
        protocol_count = sum(
            1 for t in texts
            if any(p in t.lower() for p in self.PROTOCOL_PHRASES)
        )
        voice.protocol_usage = protocol_count / len(texts)

        # Check for order-giving
        order_count = sum(
            1 for t in texts
            if any(o in t.lower() for o in self.ORDER_PHRASES)
        )
        voice.gives_orders = order_count > len(texts) * 0.1

        # Check for humor
        voice.uses_humor = any(h in all_text for h in self.HUMOR_MARKERS)

        # Check for confirmation requests
        confirm_patterns = ["right", "correct", "understood", "copy", "confirm"]
        confirm_count = sum(
            1 for t in texts
            if any(c in t.lower() and "?" in t for c in confirm_patterns)
        )
        voice.requests_confirmation = confirm_count > 2

        # Asks before acting
        voice.asks_before_acting = voice.question_ratio > 0.3

        # Extract signature phrases
        voice.signature_phrases = self._extract_signature_phrases(texts)

        # Determine communication style
        voice.communication_style = self._determine_communication_style(
            voice.avg_utterance_length,
            voice.question_ratio,
            voice.protocol_usage,
        )

        return voice

    def _starts_with_question(self, text: str) -> bool:
        """Check if text starts with a question word."""
        first_word = text.lower().split()[0] if text.split() else ""
        return first_word in self.QUESTION_WORDS

    def _extract_signature_phrases(
        self,
        texts: List[str],
        min_occurrences: int = 2,
    ) -> List[str]:
        """Extract frequently used phrases."""
        # Look for 2-4 word phrases
        phrases: Counter = Counter()

        for text in texts:
            words = text.lower().split()
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i + n])
                    # Filter out common filler
                    if not self._is_filler_phrase(phrase):
                        phrases[phrase] += 1

        # Return phrases that appear multiple times
        return [
            phrase for phrase, count in phrases.most_common(5)
            if count >= min_occurrences
        ]

    def _is_filler_phrase(self, phrase: str) -> bool:
        """Check if a phrase is common filler."""
        fillers = {
            "i think", "you know", "kind of", "sort of",
            "i mean", "like the", "to the", "in the",
            "on the", "at the", "is the", "and the",
        }
        return phrase in fillers

    def _determine_communication_style(
        self,
        avg_length: float,
        question_ratio: float,
        protocol_usage: float,
    ) -> CommunicationStyle:
        """Determine communication style from statistics."""
        if protocol_usage > 0.3:
            return CommunicationStyle.FORMAL
        if question_ratio > 0.4:
            return CommunicationStyle.QUESTIONING
        if avg_length < 5:
            return CommunicationStyle.DIRECT
        if avg_length > 15:
            return CommunicationStyle.EXPLANATORY

        return CommunicationStyle.CASUAL

    def _enhance_with_behavior(
        self,
        voice: CharacterVoice,
        events: List[Dict[str, Any]],
        transcripts: List[Dict[str, Any]],
    ) -> CharacterVoice:
        """
        Enhance character profile with behavioral observations.

        Args:
            voice: Base character voice
            events: Telemetry events
            transcripts: All transcripts

        Returns:
            Enhanced CharacterVoice
        """
        # Find crisis moments (red alerts, combat)
        crisis_times: List[datetime] = []
        for event in events:
            event_type = event.get("type", "").lower()
            data = event.get("data", {})

            if event_type == "alert" and data.get("level", 0) >= 4:
                try:
                    ts = datetime.fromisoformat(
                        event.get("timestamp", "").replace("Z", "+00:00")
                    )
                    crisis_times.append(ts)
                except (ValueError, TypeError):
                    pass

        # Check if this character speaks first during crises
        if crisis_times:
            first_speaker_count = 0
            for crisis_time in crisis_times:
                # Find first speaker within 30 seconds of crisis
                nearby = [
                    t for t in transcripts
                    if self._is_within_window(t, crisis_time, 30)
                ]
                if nearby:
                    nearby.sort(key=lambda t: t.get("timestamp", ""))
                    if nearby[0].get("speaker") == voice.speaker_id:
                        first_speaker_count += 1

            voice.speaks_first_in_crisis = first_speaker_count >= len(crisis_times) * 0.5

        # Determine stress response based on utterance patterns during crisis
        crisis_utterances = [
            t for t in transcripts
            if t.get("speaker") == voice.speaker_id
            and any(self._is_within_window(t, ct, 60) for ct in crisis_times)
        ]

        if crisis_utterances:
            crisis_lengths = [len(t.get("text", "").split()) for t in crisis_utterances]
            normal_length = voice.avg_utterance_length

            if crisis_lengths:
                crisis_avg = sum(crisis_lengths) / len(crisis_lengths)

                if crisis_avg < normal_length * 0.7:
                    voice.stress_response = StressResponse.TERSE
                elif crisis_avg > normal_length * 1.3:
                    voice.stress_response = StressResponse.VERBOSE
                elif voice.exclamation_ratio > 0.3:
                    voice.stress_response = StressResponse.ENERGIZED
                else:
                    voice.stress_response = StressResponse.FOCUSED

        return voice

    def _is_within_window(
        self,
        transcript: Dict[str, Any],
        target_time: datetime,
        window_seconds: float,
    ) -> bool:
        """Check if transcript is within time window of target."""
        try:
            ts = datetime.fromisoformat(
                transcript.get("timestamp", "").replace("Z", "+00:00")
            )
            return abs((ts - target_time).total_seconds()) <= window_seconds
        except (ValueError, TypeError):
            return False

    def _analyze_relationships(
        self,
        transcripts: List[Dict[str, Any]],
    ) -> List[CharacterRelationship]:
        """Analyze interaction patterns between characters."""
        relationships = []

        # Sort by timestamp
        sorted_transcripts = sorted(
            transcripts,
            key=lambda t: t.get("timestamp", ""),
        )

        # Track who responds to whom
        response_counts: Dict[tuple, int] = {}

        for i in range(1, len(sorted_transcripts)):
            prev = sorted_transcripts[i - 1]
            curr = sorted_transcripts[i]

            prev_speaker = prev.get("speaker", "")
            curr_speaker = curr.get("speaker", "")

            if prev_speaker and curr_speaker and prev_speaker != curr_speaker:
                key = (prev_speaker, curr_speaker)
                response_counts[key] = response_counts.get(key, 0) + 1

        # Create relationship objects for significant interactions
        for (speaker_a, speaker_b), count in response_counts.items():
            if count >= 3:  # Minimum interaction threshold
                rel = CharacterRelationship(
                    character_a=speaker_a,
                    character_b=speaker_b,
                    interaction_count=count,
                    response_pattern=f"{speaker_b} often responds to {speaker_a}",
                )

                # Infer dynamic
                voice_a = self.characters.get(speaker_a)
                voice_b = self.characters.get(speaker_b)

                if voice_a and voice_b:
                    if voice_a.gives_orders and not voice_b.gives_orders:
                        rel.dynamic = "commander-subordinate"
                    elif voice_a.asks_before_acting and voice_b.gives_orders:
                        rel.dynamic = "advisor-leader"
                    else:
                        rel.dynamic = "peers"

                relationships.append(rel)

        return relationships

    def _assign_archetypes(self) -> None:
        """Assign character archetypes based on full analysis."""
        # Sort characters by utterance count (most active first)
        sorted_chars = sorted(
            self.characters.values(),
            key=lambda v: v.total_utterances,
            reverse=True,
        )

        for i, voice in enumerate(sorted_chars):
            # Most active speaker with orders = Commander
            if i == 0 and voice.gives_orders:
                voice.archetype = CharacterArchetype.THE_COMMANDER
            # High protocol, measured responses = Diplomat
            elif voice.protocol_usage > 0.4 and voice.stress_response == StressResponse.CALM:
                voice.archetype = CharacterArchetype.THE_DIPLOMAT
            # Speaks first in crisis, short utterances = Maverick
            elif voice.speaks_first_in_crisis and voice.avg_utterance_length < 8:
                voice.archetype = CharacterArchetype.THE_MAVERICK
            # High question ratio = Analyst
            elif voice.question_ratio > 0.4:
                voice.archetype = CharacterArchetype.THE_ANALYST
            # Uses humor = Comic Relief
            elif voice.uses_humor:
                voice.archetype = CharacterArchetype.THE_COMIC_RELIEF
            # Very terse under stress = Steady Hand
            elif voice.stress_response == StressResponse.TERSE:
                voice.archetype = CharacterArchetype.THE_STEADY_HAND
            # Verbose under stress = Worrier
            elif voice.stress_response == StressResponse.VERBOSE:
                voice.archetype = CharacterArchetype.THE_WORRIER
            # Few utterances = Rookie (newer to speaking up)
            elif voice.total_utterances < 5:
                voice.archetype = CharacterArchetype.THE_ROOKIE
            # Default to Veteran
            else:
                voice.archetype = CharacterArchetype.THE_VETERAN

    def _generate_voice_description(self, voice: CharacterVoice) -> str:
        """Generate a narrative description of the character's voice."""
        parts = []

        # Role and archetype
        parts.append(f"The {voice.role}, a {voice.get_narrative_intro()}")

        # Communication style
        style_desc = {
            CommunicationStyle.FORMAL: "speaks with military precision",
            CommunicationStyle.CASUAL: "keeps things relaxed and informal",
            CommunicationStyle.TECHNICAL: "peppers speech with technical jargon",
            CommunicationStyle.DIRECT: "wastes no words",
            CommunicationStyle.EXPLANATORY: "always provides context",
            CommunicationStyle.QUESTIONING: "prefers to ask rather than assume",
        }
        parts.append(style_desc.get(voice.communication_style, ""))

        # Stress response
        stress_desc = {
            StressResponse.CALM: "remains unflappable under pressure",
            StressResponse.FOCUSED: "narrows focus when things get intense",
            StressResponse.ENERGIZED: "comes alive during crisis",
            StressResponse.TERSE: "becomes clipped and efficient under stress",
            StressResponse.VERBOSE: "talks through problems when stressed",
            StressResponse.PANICKED: "struggles to maintain composure",
        }
        if voice.stress_response != StressResponse.FOCUSED:
            parts.append(stress_desc.get(voice.stress_response, ""))

        # Signature traits
        if voice.speaks_first_in_crisis:
            parts.append("often the first voice heard when alarms sound")
        if voice.requests_confirmation:
            parts.append("always confirms understanding")
        if voice.uses_humor:
            parts.append("uses humor to lighten tension")

        return ". ".join(p for p in parts if p) + "."


def get_role_from_participation(
    speakers: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Assign roles based on participation order.

    Args:
        speakers: List of speaker stats with utterance counts

    Returns:
        Mapping of speaker_id to role
    """
    roles = {}

    # Sort by utterance count
    sorted_speakers = sorted(
        speakers,
        key=lambda s: s.get("utterances", 0),
        reverse=True,
    )

    role_order = [
        "Captain",
        "First Officer",
        "Tactical Officer",
        "Science Officer",
        "Helm Officer",
        "Operations Officer",
        "Engineering Officer",
    ]

    for i, speaker in enumerate(sorted_speakers):
        speaker_id = speaker.get("speaker", f"Speaker_{i}")
        role = role_order[i] if i < len(role_order) else f"Officer {i + 1}"
        roles[speaker_id] = role

    return roles
