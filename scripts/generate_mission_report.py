#!/usr/bin/env python3
"""
Generate LLM-powered mission reports from recorded game sessions.

This script loads existing mission recordings and generates comprehensive
markdown reports using Ollama LLM integration.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.mission_summarizer import MissionSummarizer
from src.llm.ollama_client import OllamaClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mission_data(mission_dir: Path) -> dict:
    """
    Load mission data from directory.

    Args:
        mission_dir: Path to mission recording directory

    Returns:
        Dictionary with mission data
    """
    mission_data = {
        'mission_id': mission_dir.name,
        'events': [],
        'transcripts': []
    }

    # Load events
    events_file = mission_dir / "game_events.json"
    if events_file.exists():
        with open(events_file, 'r') as f:
            events_data = json.load(f)
            mission_data['events'] = events_data.get('events', [])
            mission_data['mission_name'] = events_data.get('mission_name', mission_dir.name)
    else:
        logger.warning(f"No events file found in {mission_dir}")

    # Load transcripts
    transcript_file = mission_dir / "transcripts.json"
    if transcript_file.exists():
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
            # Handle both formats: {"transcripts": [...]} or flat [...]
            if isinstance(transcript_data, list):
                mission_data['transcripts'] = transcript_data
            else:
                mission_data['transcripts'] = transcript_data.get('transcripts', [])
    else:
        logger.info(f"No transcripts file found in {mission_dir}")

    return mission_data


def generate_report(mission_dir: Path, style: str = "entertaining",
                   output_file: Path = None, force: bool = False) -> bool:
    """
    Generate mission report for a recording.

    Args:
        mission_dir: Path to mission recording directory
        style: Report style (entertaining, professional, technical, casual)
        output_file: Output file path (default: mission_dir/mission_report_llm.md)
        force: Overwrite existing report

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing mission: {mission_dir.name}")

    # Check if report already exists
    if not output_file:
        output_file = mission_dir / "mission_report_llm.md"

    if output_file.exists() and not force:
        logger.info(f"Report already exists: {output_file} (use --force to overwrite)")
        return True

    # Load mission data
    try:
        mission_data = load_mission_data(mission_dir)
    except Exception as e:
        logger.error(f"Failed to load mission data: {e}")
        return False

    # Check if we have any data to process
    has_events = bool(mission_data.get('events'))
    has_transcripts = bool(mission_data.get('transcripts'))

    if not has_events and not has_transcripts:
        logger.warning("No events or transcripts found - skipping")
        return False

    if not has_events:
        logger.info("No game events - generating transcript-only report")

    # Create summarizer
    mission_name = mission_data.get('mission_name', mission_dir.name)
    summarizer = MissionSummarizer(
        mission_id=mission_data['mission_id'],
        mission_name=mission_name
    )

    # Load data
    if has_events:
        summarizer.load_events(mission_data['events'])
    if has_transcripts:
        summarizer.load_transcripts(mission_data['transcripts'])

    # Generate report
    try:
        logger.info(f"Generating {style} report...")
        report = summarizer.generate_llm_report(style=style, output_file=output_file)

        if report:
            logger.info(f"✓ Report generated: {output_file}")
            return True
        else:
            logger.error("Report generation returned empty")
            return False

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return False


def batch_process(recordings_dir: Path, style: str = "entertaining",
                 force: bool = False, limit: int = None) -> dict:
    """
    Process multiple mission recordings.

    Args:
        recordings_dir: Directory containing mission recordings
        style: Report style
        force: Overwrite existing reports
        limit: Maximum number of missions to process

    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }

    # Find all mission directories
    mission_dirs = sorted(
        [d for d in recordings_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True  # Process newest first
    )

    if limit:
        mission_dirs = mission_dirs[:limit]

    stats['total'] = len(mission_dirs)

    logger.info(f"Found {stats['total']} mission recordings")

    for mission_dir in mission_dirs:
        try:
            if generate_report(mission_dir, style=style, force=force):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        except Exception as e:
            logger.error(f"Error processing {mission_dir.name}: {e}")
            stats['failed'] += 1

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate LLM-powered mission reports from recordings'
    )
    parser.add_argument(
        'mission',
        nargs='?',
        help='Mission directory or ID to process (omit for batch mode)'
    )
    parser.add_argument(
        '--recordings-dir',
        type=Path,
        default=Path('game_recordings'),
        help='Directory containing mission recordings (default: game_recordings)'
    )
    parser.add_argument(
        '--style',
        choices=['entertaining', 'professional', 'technical', 'casual'],
        default=os.getenv('LLM_REPORT_STYLE', 'entertaining'),
        help='Report style (default: from LLM_REPORT_STYLE env var or "entertaining")'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: mission_dir/mission_report_llm.md)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing reports'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all missions in recordings directory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of missions to process in batch mode'
    )
    parser.add_argument(
        '--check-connection',
        action='store_true',
        help='Check Ollama connection and list available models'
    )

    args = parser.parse_args()

    # Check Ollama connection if requested
    if args.check_connection:
        client = OllamaClient()
        logger.info(f"Checking connection to Ollama at {client.host}")

        if client.check_connection():
            logger.info("✓ Ollama server is accessible")
            models = client.list_models()
            if models:
                logger.info(f"Available models: {', '.join(models)}")
            else:
                logger.warning("No models found")
        else:
            logger.error("✗ Cannot connect to Ollama server")
            sys.exit(1)
        return

    # Batch mode
    if args.batch:
        if not args.recordings_dir.exists():
            logger.error(f"Recordings directory not found: {args.recordings_dir}")
            sys.exit(1)

        logger.info(f"Batch processing missions in {args.recordings_dir}")
        stats = batch_process(
            args.recordings_dir,
            style=args.style,
            force=args.force,
            limit=args.limit
        )

        logger.info("=" * 70)
        logger.info("Batch Processing Complete")
        logger.info("=" * 70)
        logger.info(f"Total missions: {stats['total']}")
        logger.info(f"Success: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("=" * 70)

        return

    # Single mission mode
    if not args.mission:
        logger.error("Mission directory or ID required (or use --batch)")
        parser.print_help()
        sys.exit(1)

    # Find mission directory
    mission_path = Path(args.mission)
    if not mission_path.exists():
        # Try in recordings dir
        mission_path = args.recordings_dir / args.mission
        if not mission_path.exists():
            logger.error(f"Mission not found: {args.mission}")
            sys.exit(1)

    if not mission_path.is_dir():
        logger.error(f"Not a directory: {mission_path}")
        sys.exit(1)

    # Generate report
    success = generate_report(
        mission_path,
        style=args.style,
        output_file=args.output,
        force=args.force
    )

    if success:
        logger.info("✓ Report generation complete")
    else:
        logger.error("✗ Report generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
