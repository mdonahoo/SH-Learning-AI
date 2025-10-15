#!/usr/bin/env python3
"""
Record a Starship Horizons mission with audio transcription.

This script records game telemetry along with bridge crew audio,
transcribing and synchronizing everything into a unified timeline.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.game_recorder import GameRecorder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_live_stats(recorder: GameRecorder):
    """Display live recording statistics."""
    stats = recorder.get_live_stats()

    if stats.get('status') == 'not_recording':
        logger.warning("Not currently recording")
        return

    logger.info("=" * 70)
    logger.info("Live Recording Statistics")
    logger.info("=" * 70)
    logger.info(f"Mission ID: {stats.get('mission_id')}")
    logger.info(f"Game State: {stats.get('game_state')}")
    logger.info(f"Game Mode: {stats.get('game_mode')}")
    logger.info(f"Events Recorded: {stats.get('events_recorded')}")
    logger.info(f"Recording Duration: {stats.get('duration')}")
    logger.info(f"Events/Minute: {stats.get('events_per_minute')}")

    # Audio stats
    if 'transcripts_count' in stats:
        logger.info("")
        logger.info("Audio Transcription:")
        logger.info(f"  Transcripts: {stats.get('transcripts_count')}")
        logger.info(f"  Audio Duration: {stats.get('audio_duration', 0):.1f}s")

        conv_summary = stats.get('conversation_summary', {})
        if conv_summary:
            logger.info(f"  Utterances: {conv_summary.get('total_utterances', 0)}")
            logger.info(f"  Unique Speakers: {conv_summary.get('unique_speakers', 0)}")

    # Engagement metrics
    if 'engagement_metrics' in stats:
        engagement = stats['engagement_metrics']
        logger.info("")
        logger.info("Engagement Metrics:")
        logger.info(f"  Turn-taking rate: {engagement.get('turn_taking_rate', 0):.2f} turns/min")
        logger.info(f"  Avg response time: {engagement.get('avg_response_time', 0):.2f}s")

        speaker_stats = engagement.get('speaker_stats', {})
        if speaker_stats:
            logger.info("")
            logger.info("  Speaker Activity:")
            for speaker_id, speaker_data in speaker_stats.items():
                logger.info(
                    f"    {speaker_id}: {speaker_data.get('utterances', 0)} utterances, "
                    f"{speaker_data.get('total_time', 0):.1f}s"
                )

    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Record Starship Horizons mission with audio transcription'
    )
    parser.add_argument(
        '--host',
        default=os.getenv('GAME_HOST'),
        help='Game server hostname (default from .env)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Recording duration in seconds (default: record until interrupted)'
    )
    parser.add_argument(
        '--mission-name',
        help='Mission name (default: auto-detect from game)'
    )
    parser.add_argument(
        '--stats-interval',
        type=int,
        default=10,
        help='Live stats display interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio capture (only record game events)'
    )

    args = parser.parse_args()

    # Check configuration
    if not args.host:
        logger.error("No game host specified. Set GAME_HOST in .env or use --host")
        sys.exit(1)

    # Check audio configuration
    audio_enabled = os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() == 'true'
    if args.no_audio:
        logger.info("Audio capture disabled by --no-audio flag")
        os.environ['ENABLE_AUDIO_CAPTURE'] = 'false'
        audio_enabled = False
    elif not audio_enabled:
        logger.warning("ENABLE_AUDIO_CAPTURE not set to 'true' in .env")
        logger.warning("Recording will proceed without audio transcription")

    # Create recorder
    logger.info(f"Connecting to game at: {args.host}")
    recorder = GameRecorder(game_host=args.host)

    try:
        # Start recording
        mission_id = recorder.start_recording(mission_name=args.mission_name)

        logger.info("=" * 70)
        logger.info("Recording Started")
        logger.info("=" * 70)
        logger.info(f"Mission ID: {mission_id}")
        if audio_enabled:
            logger.info("✓ Audio transcription enabled")
            logger.info("Speak into your microphone - audio will be transcribed in real-time")
        else:
            logger.info("ℹ Audio transcription disabled")
        logger.info("")
        logger.info("Press Ctrl+C to stop recording")
        logger.info("=" * 70)

        # Monitor recording
        start_time = time.time()
        last_stats_time = start_time

        while True:
            current_time = time.time()

            # Display stats periodically
            if current_time - last_stats_time >= args.stats_interval:
                display_live_stats(recorder)
                last_stats_time = current_time

            # Check if we've reached duration limit
            if args.duration and (current_time - start_time) >= args.duration:
                logger.info(f"\nRecording duration limit ({args.duration}s) reached")
                break

            # Check for new transcriptions if audio enabled
            if audio_enabled and recorder.audio_service:
                results = recorder.audio_service.get_transcription_results()
                if results.get('status') == 'success' and results.get('results'):
                    for result in results['results']:
                        logger.info(
                            f"📝 [{result.get('speaker_id', 'Unknown')}] "
                            f"({result.get('confidence', 0):.2f}): {result.get('text', '')}"
                        )

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("\n\nRecording interrupted by user")

    except Exception as e:
        logger.error(f"Recording error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Stop recording
        logger.info("Stopping recording...")
        summary = recorder.stop_recording()

        if summary:
            logger.info("=" * 70)
            logger.info("Recording Complete")
            logger.info("=" * 70)
            logger.info(f"Mission ID: {summary.get('mission_id')}")
            logger.info(f"Export Path: {summary.get('export_path')}")

            stats = summary.get('statistics', {})
            logger.info(f"Total Events: {stats.get('total_events', 0)}")
            logger.info(f"Duration: {stats.get('duration', '0:00:00')}")
            logger.info(f"Events/Minute: {stats.get('events_per_minute', 0):.1f}")

            # Get combined timeline
            timeline_data = recorder.get_combined_timeline()
            if timeline_data and timeline_data.get('total_items', 0) > 0:
                logger.info(f"Timeline Items: {timeline_data['total_items']}")

                # Export combined timeline
                export_path = Path(summary['export_path'])
                timeline_file = export_path / "combined_timeline.json"

                import json
                with open(timeline_file, 'w') as f:
                    # Convert datetime objects to ISO strings for JSON
                    timeline_json = []
                    for item in timeline_data['timeline']:
                        item_copy = item.copy()
                        if hasattr(item_copy['timestamp'], 'isoformat'):
                            item_copy['timestamp'] = item_copy['timestamp'].isoformat()
                        timeline_json.append(item_copy)

                    json.dump({
                        'mission_id': timeline_data['mission_id'],
                        'total_items': timeline_data['total_items'],
                        'timeline': timeline_json
                    }, f, indent=2)

                logger.info(f"Combined timeline saved to: {timeline_file}")

            logger.info("=" * 70)
            logger.info("✓ Recording saved successfully!")

        # Generate summary
        summarizer = recorder.generate_summary()
        if summarizer:
            logger.info("\nGenerating mission summary...")
            summary_text = summarizer.generate_llm_summary()
            logger.info(f"\n{summary_text}")


if __name__ == "__main__":
    main()
