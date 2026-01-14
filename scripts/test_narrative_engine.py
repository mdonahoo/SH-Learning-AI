#!/usr/bin/env python3
"""
Test script for the narrative engine.

Runs the narrative engine on a game recording to verify it works.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.narrative.episode_generator import EpisodeGenerator
from src.narrative.tension_analyzer import format_tension_curve_ascii

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_recording(recording_dir: Path) -> tuple:
    """Load events and transcripts from a recording directory."""
    events_file = recording_dir / "game_events.json"
    transcripts_file = recording_dir / "transcripts.json"

    with open(events_file) as f:
        events_data = json.load(f)

    with open(transcripts_file) as f:
        transcripts_data = json.load(f)

    # Extract the nested data
    events = events_data.get("events", [])
    transcripts = transcripts_data.get("transcripts", [])
    mission_name = events_data.get("mission_name", "Unknown Mission")
    mission_id = events_data.get("mission_id", "")

    # Check if there's a more specific mission name in events
    for event in events:
        if event.get("event_type") == "game_connected":
            mission = event.get("data", {}).get("Mission")
            if mission:
                mission_name = mission
                break

    return events, transcripts, mission_name, mission_id


def main():
    """Main script logic."""
    parser = argparse.ArgumentParser(description="Test narrative engine on a game recording")
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Path to recording directory (defaults to most recent)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (defaults to stdout)"
    )
    args = parser.parse_args()

    # Find recording directory
    recordings_dir = Path(__file__).parent.parent / "game_recordings"

    if args.recording:
        recording_dir = Path(args.recording)
    else:
        # Use most recent recording
        recordings = sorted(recordings_dir.iterdir(), reverse=True)
        if not recordings:
            logger.error("No recordings found")
            sys.exit(1)
        recording_dir = recordings[0]

    logger.info(f"Loading recording from: {recording_dir}")

    # Load data
    events, transcripts, mission_name, mission_id = load_recording(recording_dir)

    logger.info(f"Mission: {mission_name}")
    logger.info(f"Events: {len(events)}")
    logger.info(f"Transcripts: {len(transcripts)}")

    # Generate episode (without LLM for now)
    generator = EpisodeGenerator(llm_callback=None)
    episode = generator.generate(
        events=events,
        transcripts=transcripts,
        mission_name=mission_name,
        mission_id=mission_id,
    )

    # Print results
    print("\n" + "=" * 70)
    print("NARRATIVE ENGINE TEST RESULTS")
    print("=" * 70)

    print(f"\nEpisode Title: {episode.metadata.title}")
    print(f"Stardate: {episode.metadata.stardate}")
    print(f"Duration: {episode.metadata.duration}")
    print(f"Outcome: {episode.metadata.outcome}")

    print(f"\n--- TENSION CURVE ---")
    print(format_tension_curve_ascii(episode.tension_curve))

    print(f"\n--- CHARACTERS ({len(episode.characters)}) ---")
    for speaker_id, voice in episode.characters.items():
        print(f"\n  {voice.role} ({speaker_id})")
        print(f"    Archetype: {voice.archetype.value}")
        print(f"    Style: {voice.communication_style.value}")
        print(f"    Stress Response: {voice.stress_response.value}")
        print(f"    Utterances: {voice.total_utterances}")
        if voice.signature_phrases:
            print(f"    Signature Phrases: {voice.signature_phrases[:3]}")

    print(f"\n--- DRAMATIC BEATS ({len(episode.tension_curve.points)}) ---")
    for point in episode.tension_curve.points[:10]:
        if point.beat:
            print(f"  [{point.tension:.2f}] {point.beat.beat_type.value}: {point.beat.description}")

    print(f"\n--- SCENES ({len(episode.scenes)}) ---")
    for scene in episode.scenes:
        print(f"\n  Scene {scene.scene_number}: {scene.scene_type.value}")
        print(f"    Act: {scene.act.value}")
        print(f"    Tension: {scene.tension_level:.0%}")
        print(f"    Atmosphere: {scene.atmosphere.value}")
        print(f"    Dialogue Lines: {len(scene.dialogue)}")
        print(f"    Purpose: {scene.purpose}")

    print(f"\n--- KEY THEMES ---")
    for theme in episode.key_themes:
        print(f"  - {theme}")

    print(f"\n--- CAPTAIN ARCHETYPE ---")
    print(f"  {episode.captain_archetype}")

    # Print the structured output
    print("\n" + "=" * 70)
    print("FULL STRUCTURED EPISODE")
    print("=" * 70)
    print(episode.full_episode)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        episode.save(output_path)
        logger.info(f"Episode saved to {output_path}")

    logger.info("Narrative engine test complete!")


if __name__ == "__main__":
    main()
