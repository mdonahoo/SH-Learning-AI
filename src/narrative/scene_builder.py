"""
Scene construction from dramatic beats and dialogue.

Builds structured scenes that combine telemetry events, character
dialogue, and narrative direction for Star Trek-style episodes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.narrative.beat_detector import BeatType, DramaticBeat
from src.narrative.character_voice import CharacterVoice
from src.narrative.tension_analyzer import ActType, TensionCurve

logger = logging.getLogger(__name__)


class SceneType(Enum):
    """Types of scenes in an episode."""

    COLD_OPEN = "cold_open"
    ESTABLISHING = "establishing"
    TRANSITION = "transition"
    CRISIS = "crisis"
    COMBAT = "combat"
    TACTICAL = "tactical"
    DIALOGUE = "dialogue"
    CHARACTER_MOMENT = "character_moment"
    BRIEFING = "briefing"
    CLIMAX = "climax"
    DENOUEMENT = "denouement"
    CAPTAINS_LOG = "captains_log"


class AtmosphereType(Enum):
    """Atmospheric tone for scenes."""

    CALM = "calm"
    TENSE = "tense"
    URGENT = "urgent"
    CHAOTIC = "chaotic"
    TRIUMPHANT = "triumphant"
    SOMBER = "somber"


@dataclass
class DialogueLine:
    """A single line of dialogue with context."""

    speaker_id: str
    role: str
    text: str
    timestamp: datetime
    confidence: float = 1.0
    delivery: str = ""
    action: str = ""

    def format_screenplay(self) -> str:
        """Format as screenplay dialogue."""
        header = self.role.upper()
        if self.delivery:
            header += f"\n{self.delivery}"
        lines = [header]
        if self.action:
            lines.append(f"({self.action})")
        lines.append(self.text)
        return "\n".join(lines)


@dataclass
class StageDirection:
    """Stage direction/action description."""

    text: str
    timestamp: Optional[datetime] = None
    is_visual: bool = True


@dataclass
class Scene:
    """A complete scene in the episode."""

    scene_number: int
    scene_type: SceneType
    act: ActType
    start_time: datetime
    end_time: datetime
    dialogue: List[DialogueLine] = field(default_factory=list)
    directions: List[StageDirection] = field(default_factory=list)
    beats: List[DramaticBeat] = field(default_factory=list)
    atmosphere: AtmosphereType = AtmosphereType.CALM
    tension_level: float = 0.0
    location: str = "Bridge"
    lighting: str = "Standard"
    sound_effects: List[str] = field(default_factory=list)
    purpose: str = ""
    pov_character: Optional[str] = None
    prose: str = ""
    internal_monologue: str = ""

    def get_duration_seconds(self) -> float:
        """Get scene duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def format_header(self) -> str:
        """Format scene header for screenplay."""
        return f"SCENE {self.scene_number} - {self.location.upper()} - {self.lighting.upper()}"


@dataclass
class CaptainsLog:
    """A Captain's Log entry."""

    stardate: str
    log_type: str
    situation: str
    reflection: str
    timestamp: datetime
    tone: str = ""
    key_facts: List[str] = field(default_factory=list)


class SceneBuilder:
    """
    Constructs scenes from analyzed mission data.
    """

    TENSION_ATMOSPHERE = [
        (0.0, 0.2, AtmosphereType.CALM),
        (0.2, 0.4, AtmosphereType.TENSE),
        (0.4, 0.6, AtmosphereType.URGENT),
        (0.6, 0.8, AtmosphereType.CHAOTIC),
        (0.8, 1.0, AtmosphereType.CHAOTIC),
    ]

    ALERT_LIGHTING = {
        "green": "Standard bridge lighting",
        "yellow": "Amber alert lighting",
        "red": "Red alert - emergency lighting",
    }

    SOUND_EFFECTS = {
        "alert_yellow": ["Alert klaxon (two-tone)"],
        "alert_red": ["Red alert klaxon", "Battle stations announcement"],
        "weapon_fire": ["Phaser discharge", "Torpedo launch"],
        "impact": ["Hull impact rumble", "Sparks crackling"],
        "shields": ["Shield harmonic hum"],
        "calm": ["Bridge ambient hum"],
    }

    def __init__(self) -> None:
        """Initialize the scene builder."""
        self.scenes: List[Scene] = []
        self.logs: List[CaptainsLog] = []

    def build_scenes(
        self,
        beats: List[DramaticBeat],
        tension_curve: TensionCurve,
        characters: Dict[str, CharacterVoice],
        transcripts: List[Dict[str, Any]],
        mission_name: str = "Unknown Mission",
    ) -> List[Scene]:
        """
        Build scenes from mission data.

        Args:
            beats: Dramatic beats in chronological order
            tension_curve: Analyzed tension curve with act breaks
            characters: Character voice profiles
            transcripts: Raw transcript data
            mission_name: Name of the mission

        Returns:
            List of constructed scenes
        """
        self.scenes = []

        if not beats:
            logger.warning("No beats provided for scene construction")
            return []

        act_content = self._group_by_act(beats, tension_curve, transcripts)
        scene_number = 1

        for act, content in act_content.items():
            act_beats = content["beats"]
            act_transcripts = content["transcripts"]

            if not act_beats and not act_transcripts:
                continue

            scene_specs = self._plan_act_scenes(act, act_beats, tension_curve)

            for spec in scene_specs:
                scene = self._build_scene(
                    scene_number=scene_number,
                    spec=spec,
                    beats=act_beats,
                    transcripts=act_transcripts,
                    characters=characters,
                    tension_curve=tension_curve,
                )
                self.scenes.append(scene)
                scene_number += 1

        self.logs = self._generate_log_entries(tension_curve, mission_name, characters)

        logger.info(f"Built {len(self.scenes)} scenes and {len(self.logs)} log entries")
        return self.scenes

    def _group_by_act(
        self,
        beats: List[DramaticBeat],
        tension_curve: TensionCurve,
        transcripts: List[Dict[str, Any]],
    ) -> Dict[ActType, Dict[str, List]]:
        """Group beats and transcripts by act."""
        act_content = {
            ActType.COLD_OPEN: {"beats": [], "transcripts": []},
            ActType.ACT_ONE: {"beats": [], "transcripts": []},
            ActType.ACT_TWO: {"beats": [], "transcripts": []},
            ActType.ACT_THREE: {"beats": [], "transcripts": []},
            ActType.ACT_FOUR: {"beats": [], "transcripts": []},
        }

        act_ranges = self._get_act_time_ranges(beats, tension_curve)

        for beat in beats:
            assigned_act = self._get_act_for_time(beat.timestamp, act_ranges)
            if assigned_act:
                act_content[assigned_act]["beats"].append(beat)

        for t in transcripts:
            try:
                ts = datetime.fromisoformat(t.get("timestamp", "").replace("Z", "+00:00"))
                assigned_act = self._get_act_for_time(ts, act_ranges)
                if assigned_act:
                    act_content[assigned_act]["transcripts"].append(t)
            except (ValueError, TypeError):
                act_content[ActType.ACT_ONE]["transcripts"].append(t)

        return act_content

    def _get_act_time_ranges(
        self,
        beats: List[DramaticBeat],
        tension_curve: TensionCurve,
    ) -> Dict[ActType, tuple]:
        """Get time ranges for each act."""
        if not beats:
            return {}

        mission_start = beats[0].timestamp
        mission_end = beats[-1].timestamp
        ranges = {}
        current_act = ActType.COLD_OPEN
        current_start = mission_start

        for ab in tension_curve.act_breaks:
            ranges[current_act] = (current_start, ab.timestamp)
            current_act = ab.to_act
            current_start = ab.timestamp

        ranges[current_act] = (current_start, mission_end)
        return ranges

    def _get_act_for_time(
        self,
        timestamp: datetime,
        act_ranges: Dict[ActType, tuple],
    ) -> Optional[ActType]:
        """Determine which act a timestamp belongs to."""
        for act, (start, end) in act_ranges.items():
            if start <= timestamp <= end:
                return act
        return ActType.ACT_ONE

    def _plan_act_scenes(
        self,
        act: ActType,
        beats: List[DramaticBeat],
        tension_curve: TensionCurve,
    ) -> List[Dict[str, Any]]:
        """Plan scene structure for an act."""
        specs = []

        if act == ActType.COLD_OPEN:
            specs.append({
                "type": SceneType.COLD_OPEN,
                "act": act,
                "purpose": "Hook the audience",
                "beat_filter": lambda b: True,
            })
        elif act == ActType.ACT_ONE:
            specs.append({
                "type": SceneType.ESTABLISHING,
                "act": act,
                "purpose": "Introduce characters and mission",
                "beat_filter": lambda b: b.beat_type not in (BeatType.INCITING_INCIDENT, BeatType.ESCALATION),
            })
            if tension_curve.inciting_incident:
                specs.append({
                    "type": SceneType.TACTICAL,
                    "act": act,
                    "purpose": "The problem emerges",
                    "beat_filter": lambda b: b.beat_type in (BeatType.INCITING_INCIDENT, BeatType.DISCOVERY),
                })
        elif act == ActType.ACT_TWO:
            specs.append({
                "type": SceneType.TACTICAL,
                "act": act,
                "purpose": "First response to threat",
                "beat_filter": lambda b: b.beat_type in (BeatType.TACTICAL_DECISION, BeatType.ESCALATION),
            })
            combat_beats = [b for b in beats if b.beat_type == BeatType.ESCALATION]
            if combat_beats:
                specs.append({
                    "type": SceneType.COMBAT,
                    "act": act,
                    "purpose": "Combat engagement",
                    "beat_filter": lambda b: b.beat_type == BeatType.ESCALATION,
                })
        elif act == ActType.ACT_THREE:
            specs.append({
                "type": SceneType.CRISIS,
                "act": act,
                "purpose": "The darkest moment",
                "beat_filter": lambda b: b.beat_type == BeatType.CRISIS_POINT,
            })
        elif act == ActType.ACT_FOUR:
            specs.append({
                "type": SceneType.CLIMAX,
                "act": act,
                "purpose": "The decisive moment",
                "beat_filter": lambda b: b.beat_type == BeatType.CLIMAX,
            })
            specs.append({
                "type": SceneType.DENOUEMENT,
                "act": act,
                "purpose": "Aftermath and reflection",
                "beat_filter": lambda b: b.beat_type in (BeatType.RESOLUTION, BeatType.TRAGIC_RESOLUTION),
            })

        return specs

    def _build_scene(
        self,
        scene_number: int,
        spec: Dict[str, Any],
        beats: List[DramaticBeat],
        transcripts: List[Dict[str, Any]],
        characters: Dict[str, CharacterVoice],
        tension_curve: TensionCurve,
    ) -> Scene:
        """Build a single scene from specification."""
        scene_type = spec["type"]
        act = spec["act"]
        beat_filter = spec.get("beat_filter", lambda b: True)

        scene_beats = [b for b in beats if beat_filter(b)]

        if scene_beats:
            start_time = min(b.timestamp for b in scene_beats)
            end_time = max(b.timestamp for b in scene_beats)
        elif transcripts:
            try:
                times = [datetime.fromisoformat(t.get("timestamp", "").replace("Z", "+00:00")) for t in transcripts]
                start_time = min(times) if times else datetime.now()
                end_time = max(times) if times else datetime.now()
            except (ValueError, TypeError):
                start_time = datetime.now()
                end_time = datetime.now()
        else:
            start_time = datetime.now()
            end_time = datetime.now()

        scene_transcripts = self._get_transcripts_in_range(transcripts, start_time, end_time)
        dialogue = self._build_dialogue(scene_transcripts, characters, scene_type)

        avg_tension = sum(b.tension_level for b in scene_beats) / len(scene_beats) if scene_beats else 0.3
        atmosphere = self._get_atmosphere(avg_tension)
        directions = self._build_directions(scene_beats, scene_type, atmosphere)
        lighting = self._get_lighting(scene_beats)
        sounds = self._get_sound_effects(scene_type, scene_beats)
        pov = self._determine_pov(dialogue, characters, scene_type)

        return Scene(
            scene_number=scene_number,
            scene_type=scene_type,
            act=act,
            start_time=start_time,
            end_time=end_time,
            dialogue=dialogue,
            directions=directions,
            beats=scene_beats,
            atmosphere=atmosphere,
            tension_level=avg_tension,
            location="Bridge",
            lighting=lighting,
            sound_effects=sounds,
            purpose=spec.get("purpose", ""),
            pov_character=pov,
        )

    def _get_transcripts_in_range(
        self,
        transcripts: List[Dict[str, Any]],
        start: datetime,
        end: datetime,
        buffer_seconds: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Get transcripts within a time range."""
        result = []
        for t in transcripts:
            try:
                ts = datetime.fromisoformat(t.get("timestamp", "").replace("Z", "+00:00"))
                if (start - timedelta(seconds=buffer_seconds)) <= ts <= (end + timedelta(seconds=buffer_seconds)):
                    result.append(t)
            except (ValueError, TypeError):
                continue
        return sorted(result, key=lambda t: t.get("timestamp", ""))

    def _build_dialogue(
        self,
        transcripts: List[Dict[str, Any]],
        characters: Dict[str, CharacterVoice],
        scene_type: SceneType,
    ) -> List[DialogueLine]:
        """Build dialogue lines from transcripts."""
        dialogue = []
        for t in transcripts:
            speaker_id = t.get("speaker", "Unknown")
            text = t.get("text", "")
            confidence = t.get("confidence", 1.0)

            if not text.strip():
                continue

            voice = characters.get(speaker_id)
            role = voice.role if voice else speaker_id
            delivery = self._determine_delivery(voice, scene_type)

            try:
                timestamp = datetime.fromisoformat(t.get("timestamp", "").replace("Z", "+00:00"))
            except (ValueError, TypeError):
                timestamp = datetime.now()

            dialogue.append(DialogueLine(
                speaker_id=speaker_id,
                role=role,
                text=text,
                timestamp=timestamp,
                confidence=confidence,
                delivery=delivery,
            ))
        return dialogue

    def _determine_delivery(self, voice: Optional[CharacterVoice], scene_type: SceneType) -> str:
        """Determine dialogue delivery direction."""
        if not voice:
            return ""
        if scene_type in (SceneType.CRISIS, SceneType.COMBAT):
            from src.narrative.character_voice import StressResponse
            if voice.stress_response == StressResponse.CALM:
                return "(measured, controlled)"
            elif voice.stress_response == StressResponse.TERSE:
                return "(clipped, urgent)"
            else:
                return "(focused)"
        return ""

    def _get_atmosphere(self, tension: float) -> AtmosphereType:
        """Get atmosphere type from tension level."""
        for low, high, atmosphere in self.TENSION_ATMOSPHERE:
            if low <= tension < high:
                return atmosphere
        return AtmosphereType.CHAOTIC

    def _build_directions(
        self,
        beats: List[DramaticBeat],
        scene_type: SceneType,
        atmosphere: AtmosphereType,
    ) -> List[StageDirection]:
        """Build stage directions for the scene."""
        directions = []
        atmosphere_openings = {
            AtmosphereType.CALM: "The bridge hums with routine activity.",
            AtmosphereType.TENSE: "Tension hangs in the recycled air.",
            AtmosphereType.URGENT: "The bridge erupts into controlled chaos.",
            AtmosphereType.CHAOTIC: "Alarms blare. Consoles spark. The deck shudders.",
            AtmosphereType.TRIUMPHANT: "Relief washes across the bridge.",
            AtmosphereType.SOMBER: "A heavy silence settles over the crew.",
        }

        if scene_type == SceneType.COLD_OPEN:
            directions.append(StageDirection(text="FADE IN:", is_visual=True))

        directions.append(StageDirection(text=atmosphere_openings.get(atmosphere, ""), is_visual=True))

        for beat in beats:
            if beat.beat_type == BeatType.CRISIS_POINT:
                directions.append(StageDirection(
                    text="The viewscreen flickers. Warning lights cascade.",
                    timestamp=beat.timestamp,
                ))
            elif beat.beat_type == BeatType.ESCALATION:
                directions.append(StageDirection(
                    text="Red alert klaxons pierce the air.",
                    timestamp=beat.timestamp,
                ))
            elif beat.beat_type == BeatType.RESOLUTION:
                directions.append(StageDirection(
                    text="The alarms fall silent. The crew exhales.",
                    timestamp=beat.timestamp,
                ))

        return directions

    def _get_lighting(self, beats: List[DramaticBeat]) -> str:
        """Determine lighting from beats."""
        for beat in beats:
            for event in beat.telemetry_events:
                data = event.get("data", {})
                alert = data.get("level", data.get("alert", 2))
                if alert >= 4:
                    return self.ALERT_LIGHTING["red"]
                elif alert >= 3:
                    return self.ALERT_LIGHTING["yellow"]
        return self.ALERT_LIGHTING["green"]

    def _get_sound_effects(self, scene_type: SceneType, beats: List[DramaticBeat]) -> List[str]:
        """Get appropriate sound effects for the scene."""
        sounds = []
        if scene_type == SceneType.COMBAT:
            sounds.extend(self.SOUND_EFFECTS["weapon_fire"])
            sounds.extend(self.SOUND_EFFECTS["impact"])
        elif scene_type == SceneType.CRISIS:
            sounds.extend(self.SOUND_EFFECTS["alert_red"])
        elif scene_type in (SceneType.COLD_OPEN, SceneType.ESTABLISHING):
            sounds.extend(self.SOUND_EFFECTS["calm"])
        return list(set(sounds))

    def _determine_pov(
        self,
        dialogue: List[DialogueLine],
        characters: Dict[str, CharacterVoice],
        scene_type: SceneType,
    ) -> Optional[str]:
        """Determine POV character for the scene."""
        if not dialogue:
            return None
        if scene_type in (SceneType.CRISIS, SceneType.COMBAT):
            return dialogue[0].speaker_id
        speaker_counts: Dict[str, int] = {}
        for line in dialogue:
            speaker_counts[line.speaker_id] = speaker_counts.get(line.speaker_id, 0) + 1
        if speaker_counts:
            return max(speaker_counts, key=speaker_counts.get)
        return None

    def _generate_log_entries(
        self,
        tension_curve: TensionCurve,
        mission_name: str,
        characters: Dict[str, CharacterVoice],
    ) -> List[CaptainsLog]:
        """Generate Captain's Log entries."""
        logs = []
        captain = None
        for voice in characters.values():
            if voice.role.lower() == "captain" or voice.archetype.value == "commander":
                captain = voice
                break

        captain_tone = "measured" if captain and captain.archetype.value == "diplomat" else "personal"

        if tension_curve.points:
            logs.append(CaptainsLog(
                stardate=self._generate_stardate(tension_curve.points[0].timestamp),
                log_type="opening",
                situation=f"Mission: {mission_name}",
                reflection="",
                timestamp=tension_curve.points[0].timestamp,
                tone=captain_tone,
                key_facts=[mission_name],
            ))

        if tension_curve.resolution:
            outcome = "successful" if tension_curve.resolution.beat_type == BeatType.RESOLUTION else "costly"
            logs.append(CaptainsLog(
                stardate=self._generate_stardate(tension_curve.resolution.timestamp),
                log_type="closing",
                situation=f"Mission {outcome}",
                reflection="Lessons learned",
                timestamp=tension_curve.resolution.timestamp,
                tone=captain_tone,
                key_facts=[outcome],
            ))

        return logs

    def _generate_stardate(self, timestamp: datetime) -> str:
        """Generate a Star Trek-style stardate."""
        year = timestamp.year
        day_of_year = timestamp.timetuple().tm_yday
        fraction = (timestamp.hour * 60 + timestamp.minute) / 1440
        return f"{year}.{day_of_year + fraction:.1f}"
