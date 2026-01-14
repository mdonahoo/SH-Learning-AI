"""
Narrative generation engine for Star Trek-style mission episodes.

This module transforms telemetry data and voice transcripts into
engaging episodic narratives with proper dramatic structure.

Modules:
    beat_detector: Identifies dramatic beats from telemetry events
    tension_analyzer: Maps beats to tension curves and act structure
    character_voice: Infers character personalities from behavior
    scene_builder: Constructs scenes with dialogue and action
    trek_prompts: Specialized prompts for Trek-style narrative
    episode_generator: Orchestrates the full generation pipeline
"""

from src.narrative.beat_detector import (
    BeatType,
    DramaticBeat,
    BeatDetector,
)
from src.narrative.tension_analyzer import (
    TensionPoint,
    ActBreak,
    TensionCurve,
    TensionAnalyzer,
)
from src.narrative.character_voice import (
    CharacterVoice,
    CharacterArchetype,
    CharacterAnalyzer,
)
from src.narrative.scene_builder import (
    Scene,
    SceneType,
    SceneBuilder,
)
from src.narrative.episode_generator import (
    Episode,
    EpisodeGenerator,
)

__all__ = [
    # Beat detection
    "BeatType",
    "DramaticBeat",
    "BeatDetector",
    # Tension analysis
    "TensionPoint",
    "ActBreak",
    "TensionCurve",
    "TensionAnalyzer",
    # Character voice
    "CharacterVoice",
    "CharacterArchetype",
    "CharacterAnalyzer",
    # Scene building
    "Scene",
    "SceneType",
    "SceneBuilder",
    # Episode generation
    "Episode",
    "EpisodeGenerator",
]
