"""
Episode generator - main orchestration pipeline.

Coordinates all narrative components to generate complete
Star Trek-style episode narratives from mission data.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from src.narrative.beat_detector import BeatDetector, BeatType, DramaticBeat
from src.narrative.character_voice import (
    CharacterAnalyzer,
    CharacterVoice,
    get_role_from_participation,
)
from src.narrative.scene_builder import CaptainsLog, Scene, SceneBuilder, SceneType
from src.narrative.tension_analyzer import (
    TensionAnalyzer,
    TensionCurve,
    format_tension_curve_ascii,
)
from src.narrative.trek_prompts import (
    build_captains_log_prompt,
    build_cold_open_prompt,
    build_crisis_scene_prompt,
    build_episode_title_prompt,
    build_full_episode_assembly_prompt,
    build_resolution_scene_prompt,
    build_scene_prompt,
)

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetadata:
    """Metadata for a generated episode."""

    title: str
    stardate: str
    mission_name: str
    mission_id: str
    duration: str
    outcome: str
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Episode:
    """A complete Star Trek-style episode."""

    metadata: EpisodeMetadata
    opening_log: str
    closing_log: str
    scenes: List[Scene]
    characters: Dict[str, CharacterVoice]
    tension_curve: TensionCurve

    # Generated content
    scene_prose: List[str] = field(default_factory=list)
    full_episode: str = ""

    # Analysis
    captain_archetype: str = ""
    key_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            "metadata": {
                "title": self.metadata.title,
                "stardate": self.metadata.stardate,
                "mission_name": self.metadata.mission_name,
                "mission_id": self.metadata.mission_id,
                "duration": self.metadata.duration,
                "outcome": self.metadata.outcome,
                "generated_at": self.metadata.generated_at.isoformat(),
            },
            "opening_log": self.opening_log,
            "closing_log": self.closing_log,
            "scene_count": len(self.scenes),
            "character_count": len(self.characters),
            "captain_archetype": self.captain_archetype,
            "key_themes": self.key_themes,
            "tension_peak": self.tension_curve.peak_tension,
            "full_episode": self.full_episode,
        }

    def save(self, output_path: Path) -> None:
        """Save episode to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save full episode text
        episode_file = output_path.with_suffix(".md")
        with open(episode_file, "w") as f:
            f.write(self.full_episode)

        # Save metadata as JSON
        meta_file = output_path.with_suffix(".json")
        with open(meta_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Episode saved to {episode_file}")


class EpisodeGenerator:
    """
    Main orchestrator for episode generation.

    Coordinates beat detection, tension analysis, character profiling,
    scene building, and LLM-based prose generation.
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize the episode generator.

        Args:
            llm_callback: Optional function to call LLM for prose generation.
                          Signature: (prompt: str) -> response: str
                          If None, returns structured data without prose.
        """
        self.llm_callback = llm_callback

        # Initialize components
        self.beat_detector = BeatDetector()
        self.tension_analyzer = TensionAnalyzer()
        self.character_analyzer = CharacterAnalyzer()
        self.scene_builder = SceneBuilder()

    def generate(
        self,
        events: List[Dict[str, Any]],
        transcripts: List[Dict[str, Any]],
        mission_name: str = "Unknown Mission",
        mission_id: str = "",
        speaker_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> Episode:
        """
        Generate a complete episode from mission data.

        Args:
            events: Telemetry events from the mission
            transcripts: Voice transcripts from the mission
            mission_name: Name of the mission
            mission_id: Mission identifier
            speaker_stats: Optional speaker statistics for role assignment

        Returns:
            Complete Episode object
        """
        logger.info(f"Generating episode for mission: {mission_name}")

        # Step 1: Detect dramatic beats
        logger.info("Step 1: Detecting dramatic beats...")
        beats = self.beat_detector.detect_beats(events, transcripts)
        logger.info(f"  Detected {len(beats)} beats")

        # Step 2: Analyze tension curve
        logger.info("Step 2: Analyzing tension curve...")
        tension_curve = self.tension_analyzer.analyze(beats)
        logger.info(f"  Peak tension: {tension_curve.peak_tension:.2f}")
        logger.info(f"  Act breaks: {len(tension_curve.act_breaks)}")

        # Step 3: Build character profiles
        logger.info("Step 3: Building character profiles...")
        station_assignments = {}
        if speaker_stats:
            station_assignments = get_role_from_participation(speaker_stats)
        characters = self.character_analyzer.analyze(
            transcripts, events, station_assignments
        )
        logger.info(f"  Profiled {len(characters)} characters")

        # Step 4: Build scenes
        logger.info("Step 4: Building scenes...")
        scenes = self.scene_builder.build_scenes(
            beats, tension_curve, characters, transcripts, mission_name
        )
        logs = self.scene_builder.logs
        logger.info(f"  Built {len(scenes)} scenes, {len(logs)} log entries")

        # Step 5: Determine metadata
        metadata = self._build_metadata(
            mission_name, mission_id, beats, tension_curve
        )

        # Step 6: Generate prose (if LLM available)
        opening_log = ""
        closing_log = ""
        scene_prose = []
        full_episode = ""

        if self.llm_callback:
            logger.info("Step 5: Generating prose with LLM...")
            opening_log, closing_log, scene_prose, full_episode = self._generate_prose(
                scenes, logs, characters, tension_curve, metadata
            )
        else:
            logger.info("Step 5: Skipping prose generation (no LLM callback)")
            full_episode = self._build_structured_output(
                scenes, logs, characters, tension_curve, metadata
            )

        # Step 7: Build final episode
        episode = Episode(
            metadata=metadata,
            opening_log=opening_log,
            closing_log=closing_log,
            scenes=scenes,
            characters=characters,
            tension_curve=tension_curve,
            scene_prose=scene_prose,
            full_episode=full_episode,
            captain_archetype=self._get_captain_archetype(characters),
            key_themes=self._identify_themes(beats),
        )

        logger.info("Episode generation complete!")
        return episode

    def _build_metadata(
        self,
        mission_name: str,
        mission_id: str,
        beats: List[DramaticBeat],
        tension_curve: TensionCurve,
    ) -> EpisodeMetadata:
        """Build episode metadata."""
        # Calculate duration
        if beats:
            duration_seconds = (beats[-1].timestamp - beats[0].timestamp).total_seconds()
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration = f"{minutes}m {seconds}s"
            stardate = self._generate_stardate(beats[0].timestamp)
        else:
            duration = "Unknown"
            stardate = self._generate_stardate(datetime.now())

        # Determine outcome
        if tension_curve.resolution:
            if tension_curve.resolution.beat_type == BeatType.RESOLUTION:
                outcome = "Mission Successful"
            else:
                outcome = "Mission Failed"
        else:
            outcome = "Unknown"

        return EpisodeMetadata(
            title=self._generate_title(mission_name, beats),
            stardate=stardate,
            mission_name=mission_name,
            mission_id=mission_id or "UNKNOWN",
            duration=duration,
            outcome=outcome,
        )

    def _generate_stardate(self, timestamp: datetime) -> str:
        """Generate Star Trek-style stardate."""
        year = timestamp.year
        day_of_year = timestamp.timetuple().tm_yday
        fraction = (timestamp.hour * 60 + timestamp.minute) / 1440
        return f"{year}.{day_of_year + fraction:.1f}"

    def _generate_title(
        self,
        mission_name: str,
        beats: List[DramaticBeat],
    ) -> str:
        """Generate episode title (or use mission name)."""
        # For now, use mission name as title
        # Could be enhanced with LLM call for creative titles
        return mission_name

    def _get_captain_archetype(
        self,
        characters: Dict[str, CharacterVoice],
    ) -> str:
        """Get the captain's archetype."""
        for voice in characters.values():
            if voice.role.lower() == "captain":
                return voice.archetype.value
        # Return first commander-type
        for voice in characters.values():
            if voice.archetype.value == "commander":
                return voice.archetype.value
        return "unknown"

    def _identify_themes(self, beats: List[DramaticBeat]) -> List[str]:
        """Identify key themes from dramatic beats."""
        themes = set()

        beat_themes = {
            BeatType.CRISIS_POINT: "survival",
            BeatType.TACTICAL_DECISION: "strategy",
            BeatType.DIPLOMATIC_BEAT: "diplomacy",
            BeatType.DISCOVERY: "exploration",
            BeatType.CHARACTER_MOMENT: "character",
            BeatType.TRAGIC_RESOLUTION: "sacrifice",
            BeatType.RESOLUTION: "triumph",
        }

        for beat in beats:
            if beat.beat_type in beat_themes:
                themes.add(beat_themes[beat.beat_type])

        return list(themes)

    def _generate_prose(
        self,
        scenes: List[Scene],
        logs: List[CaptainsLog],
        characters: Dict[str, CharacterVoice],
        tension_curve: TensionCurve,
        metadata: EpisodeMetadata,
    ) -> tuple:
        """
        Generate prose for all episode components using LLM.

        Returns:
            Tuple of (opening_log, closing_log, scene_prose_list, full_episode)
        """
        if not self.llm_callback:
            return "", "", [], ""

        scene_prose = []
        opening_log = ""
        closing_log = ""

        # Find captain for log style
        captain = None
        for voice in characters.values():
            if voice.role.lower() == "captain":
                captain = voice
                break

        mission_context = {
            "mission_name": metadata.mission_name,
            "duration": metadata.duration,
            "outcome": metadata.outcome,
        }

        # Generate logs
        if logs:
            opening_prompt = build_captains_log_prompt(
                logs[0], captain, mission_context
            )
            opening_log = self.llm_callback(opening_prompt)

            if len(logs) > 1:
                closing_prompt = build_captains_log_prompt(
                    logs[-1], captain, mission_context
                )
                closing_log = self.llm_callback(closing_prompt)

        # Generate scene prose
        for i, scene in enumerate(scenes):
            prev_summary = scene_prose[-1][:200] if scene_prose else ""
            next_hint = scenes[i + 1].purpose if i < len(scenes) - 1 else "Episode conclusion"

            if scene.scene_type == SceneType.COLD_OPEN:
                prompt = build_cold_open_prompt(
                    scene, characters, metadata.title
                )
            elif scene.scene_type == SceneType.CRISIS:
                crisis_details = {
                    "threat": "Enemy forces",
                    "stakes": scene.purpose,
                    "ship_status": "Critical damage",
                }
                prompt = build_crisis_scene_prompt(scene, characters, crisis_details)
            elif scene.scene_type == SceneType.DENOUEMENT:
                outcome = "victory" if "success" in metadata.outcome.lower() else "defeat"
                prompt = build_resolution_scene_prompt(scene, characters, outcome)
            else:
                prompt = build_scene_prompt(
                    scene, characters, prev_summary, next_hint
                )

            prose = self.llm_callback(prompt)
            scene_prose.append(prose)
            scene.prose = prose

        # Assemble full episode
        assembly_prompt = build_full_episode_assembly_prompt(
            scene_prose,
            [opening_log, closing_log],
            metadata.title,
            {
                "duration": metadata.duration,
                "outcome": metadata.outcome,
                "objectives_completed": "?",
                "objectives_total": "?",
            },
        )
        full_episode = self.llm_callback(assembly_prompt)

        return opening_log, closing_log, scene_prose, full_episode

    def _build_structured_output(
        self,
        scenes: List[Scene],
        logs: List[CaptainsLog],
        characters: Dict[str, CharacterVoice],
        tension_curve: TensionCurve,
        metadata: EpisodeMetadata,
    ) -> str:
        """
        Build structured output when no LLM is available.

        Returns episode as formatted text with structure visible.
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"  {metadata.title.upper()}")
        lines.append(f"  Stardate: {metadata.stardate}")
        lines.append("=" * 70)
        lines.append("")

        # Tension curve visualization
        lines.append("TENSION CURVE:")
        lines.append(format_tension_curve_ascii(tension_curve))
        lines.append("")

        # Character profiles
        lines.append("CHARACTERS:")
        for speaker_id, voice in characters.items():
            lines.append(f"  {voice.role} ({speaker_id})")
            lines.append(f"    Archetype: {voice.archetype.value}")
            lines.append(f"    Voice: {voice.voice_description}")
        lines.append("")

        # Opening log placeholder
        if logs:
            lines.append("CAPTAIN'S LOG - OPENING:")
            lines.append(f"  Stardate: {logs[0].stardate}")
            lines.append(f"  Situation: {logs[0].situation}")
            lines.append(f"  [Generate with: build_captains_log_prompt()]")
            lines.append("")

        # Scenes
        for scene in scenes:
            lines.append("-" * 70)
            lines.append(scene.format_header())
            lines.append(f"Type: {scene.scene_type.value}")
            lines.append(f"Act: {scene.act.value}")
            lines.append(f"Tension: {scene.tension_level:.0%}")
            lines.append(f"Atmosphere: {scene.atmosphere.value}")
            lines.append(f"Purpose: {scene.purpose}")
            lines.append("")

            # Stage directions
            for direction in scene.directions:
                lines.append(f"  [{direction.text}]")
            lines.append("")

            # Dialogue
            for dl in scene.dialogue:
                if dl.delivery:
                    lines.append(f"  {dl.role.upper()} {dl.delivery}")
                else:
                    lines.append(f"  {dl.role.upper()}")
                lines.append(f"    \"{dl.text}\"")
                lines.append("")

            # Sound effects
            if scene.sound_effects:
                lines.append(f"  SFX: {', '.join(scene.sound_effects)}")
            lines.append("")

        # Closing log placeholder
        if len(logs) > 1:
            lines.append("CAPTAIN'S LOG - CLOSING:")
            lines.append(f"  Stardate: {logs[-1].stardate}")
            lines.append(f"  Situation: {logs[-1].situation}")
            lines.append(f"  [Generate with: build_captains_log_prompt()]")
            lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"  Duration: {metadata.duration}")
        lines.append(f"  Outcome: {metadata.outcome}")
        lines.append(f"  Generated: {metadata.generated_at.isoformat()}")
        lines.append("=" * 70)

        return "\n".join(lines)


def create_episode_from_files(
    events_file: Path,
    transcripts_file: Path,
    mission_name: str = "Unknown Mission",
    llm_callback: Optional[Callable[[str], str]] = None,
) -> Episode:
    """
    Convenience function to create episode from JSON files.

    Args:
        events_file: Path to events JSON file
        transcripts_file: Path to transcripts JSON file
        mission_name: Name of the mission
        llm_callback: Optional LLM callback for prose generation

    Returns:
        Generated Episode
    """
    with open(events_file) as f:
        events = json.load(f)

    with open(transcripts_file) as f:
        transcripts = json.load(f)

    generator = EpisodeGenerator(llm_callback=llm_callback)
    return generator.generate(events, transcripts, mission_name)
