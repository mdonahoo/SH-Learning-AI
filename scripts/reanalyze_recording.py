#!/usr/bin/env python3
"""
Re-analyze a recording with the improved sub-segment diarization.

This script runs the full audio analysis pipeline on a recording
and saves the results to a new analysis file.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(step_id: str, step_label: str, progress: int):
    """Print progress updates."""
    logger.info(f"[{progress:3d}%] {step_label}")


def find_matching_telemetry(recording_path: Path) -> Optional[str]:
    """
    Find telemetry session that matches the recording timestamp.

    Args:
        recording_path: Path to the recording file

    Returns:
        Telemetry session ID if found, None otherwise
    """
    import os
    import glob

    # Extract timestamp from recording filename
    # Format: recording_YYYYMMDD_HHMMSS.wav
    recording_name = recording_path.stem
    if not recording_name.startswith('recording_'):
        return None

    try:
        # Parse recording timestamp
        timestamp_str = recording_name.replace('recording_', '')
        recording_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except ValueError:
        logger.warning(f"Could not parse timestamp from {recording_name}")
        return None

    # Find telemetry files and check their timestamps
    telemetry_files = glob.glob('data/telemetry/telemetry_*.json')

    best_match = None
    best_diff = float('inf')

    for telem_file in telemetry_files:
        telem_path = Path(telem_file)
        # Get file modification time
        telem_mtime = datetime.fromtimestamp(os.path.getmtime(telem_path))

        # Calculate time difference
        diff = abs((telem_mtime - recording_time).total_seconds())

        # Accept if within 5 minutes
        if diff < 300 and diff < best_diff:
            best_diff = diff
            # Extract session ID from filename
            session_id = telem_path.stem.replace('telemetry_', '')
            best_match = session_id

    if best_match:
        logger.info(f"Found telemetry session {best_match} ({best_diff:.0f}s difference)")

    return best_match


def main():
    """Run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Re-analyze a recording with improved diarization'
    )
    parser.add_argument(
        'recording',
        nargs='?',
        default='data/recordings/recording_20260120_214001.wav',
        help='Path to the recording file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (default: auto-generated in data/analyses/)'
    )
    parser.add_argument(
        '--no-narrative',
        action='store_true',
        help='Skip LLM narrative generation'
    )
    parser.add_argument(
        '--no-story',
        action='store_true',
        help='Skip LLM story generation'
    )
    parser.add_argument(
        '--telemetry', '-t',
        help='Telemetry session ID to link (e.g., fa4ab76e)'
    )
    parser.add_argument(
        '--auto-telemetry',
        action='store_true',
        help='Auto-detect telemetry session based on timestamp matching'
    )
    args = parser.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.exists():
        logger.error(f"Recording not found: {recording_path}")
        return 1

    logger.info(f"Re-analyzing recording: {recording_path}")

    # Determine telemetry session ID
    telemetry_session_id = args.telemetry

    if args.auto_telemetry and not telemetry_session_id:
        # Auto-detect telemetry based on timestamp matching
        telemetry_session_id = find_matching_telemetry(recording_path)
        if telemetry_session_id:
            logger.info(f"Auto-detected telemetry session: {telemetry_session_id}")
        else:
            logger.warning("No matching telemetry session found")

    telemetry_dir = None
    if telemetry_session_id:
        telemetry_file = Path(f"data/telemetry/telemetry_{telemetry_session_id}.json")
        if telemetry_file.exists():
            logger.info(f"Using telemetry: {telemetry_file}")
            telemetry_dir = str(telemetry_file.parent)
        else:
            # Search workspace directories
            workspace_matches = list(Path("data/workspaces").glob(
                f"*/telemetry/telemetry_{telemetry_session_id}.json"
            ))
            if workspace_matches:
                telemetry_file = workspace_matches[0]
                telemetry_dir = str(telemetry_file.parent)
                logger.info(f"Using telemetry from workspace: {telemetry_file}")
            else:
                logger.warning(f"Telemetry file not found: {telemetry_file}")
                telemetry_session_id = None

    # Import audio processor
    try:
        from src.web.audio_processor import AudioProcessor
    except ImportError as e:
        logger.error(f"Failed to import AudioProcessor: {e}")
        return 1

    # Initialize processor
    logger.info("Initializing AudioProcessor...")
    processor = AudioProcessor()

    # Run analysis
    logger.info("Running full analysis pipeline...")
    try:
        results = processor.analyze_audio(
            str(recording_path),
            include_diarization=True,
            include_quality=True,
            include_detailed=True,
            include_narrative=not args.no_narrative,
            include_story=not args.no_story,
            progress_callback=progress_callback,
            telemetry_session_id=telemetry_session_id,
            telemetry_dir=telemetry_dir
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("data/analyses")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"analysis_{timestamp}_improved.json"

    # Build full analysis structure
    analysis = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "recording_file": str(recording_path.name),
            "duration_seconds": results.get('duration_seconds', 0),
            "speaker_count": len(results.get('speakers', [])),
            "segment_count": len(results.get('transcription', [])),
            "analysis_version": "2.0_subsegment_diarization"
        },
        "results": results
    }

    # Save results
    logger.info(f"Saving analysis to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Duration: {results.get('duration_seconds', 0):.1f}s")
    logger.info(f"Segments: {len(results.get('transcription', []))}")
    logger.info(f"Speakers: {len(results.get('speakers', []))}")

    if results.get('diarization_methodology'):
        logger.info(f"Diarization: {results['diarization_methodology']}")

    # Show speaker breakdown
    if results.get('speakers'):
        logger.info("\nSpeaker breakdown:")
        for speaker in results['speakers']:
            logger.info(
                f"  {speaker['speaker_id']}: "
                f"{speaker['utterance_count']} utterances, "
                f"{speaker['total_speaking_time']:.1f}s"
            )

    # Show role assignments
    if results.get('role_assignments'):
        logger.info("\nRole assignments:")
        for role in results['role_assignments']:
            logger.info(
                f"  {role['speaker_id']} -> {role['role']} "
                f"(confidence: {role['confidence']:.2f})"
            )

    logger.info(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
