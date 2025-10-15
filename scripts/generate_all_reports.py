#!/usr/bin/env python3
"""
Generate all mission reports and stories from recorded game sessions.

Generates:
1. Hybrid Report - Training assessment (Kirkpatrick/Bloom's/NASA frameworks)
2. Mission Story - Chronological narrative with real dialogue
3. Factual Summary - Pure data, zero interpretation
4. Real Story - What actually happened (training reality)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.mission_summarizer import MissionSummarizer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mission_data(mission_dir: Path) -> dict:
    """Load mission data from directory."""
    mission_data = {
        'mission_id': mission_dir.name,
        'events': [],
        'transcripts': [],
        'mission_name': mission_dir.name
    }

    # Load events
    events_file = mission_dir / "game_events.json"
    if events_file.exists():
        with open(events_file, 'r') as f:
            events_data = json.load(f)
            mission_data['events'] = events_data.get('events', [])
            mission_data['mission_name'] = events_data.get('mission_name', mission_dir.name)
            mission_data['start_time'] = events_data.get('start_time')
            mission_data['end_time'] = events_data.get('end_time')

    # Load transcripts
    transcript_file = mission_dir / "transcripts.json"
    if transcript_file.exists():
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
            mission_data['transcripts'] = transcript_data.get('transcripts', [])

    return mission_data


def generate_factual_summary(mission_dir: Path, mission_data: dict) -> bool:
    """Generate pure factual summary with zero narrative."""
    try:
        output_file = mission_dir / "mission_summary_FACTUAL.md"

        events = mission_data['events']
        transcripts = mission_data['transcripts']

        # Calculate mission duration
        if mission_data.get('start_time') and mission_data.get('end_time'):
            start_time = datetime.fromisoformat(mission_data['start_time'])
            end_time = datetime.fromisoformat(mission_data['end_time'])
            duration = end_time - start_time
        else:
            duration = None

        # Speaker analysis
        speaker_counts = Counter(t['speaker'] for t in transcripts)
        total_comms = len(transcripts)

        # Extract objectives
        objectives = {}
        for event in events:
            if event.get('event_type') == 'mission_update' and 'Objectives' in event.get('data', {}):
                obj_data = event['data']['Objectives']
                for obj_name, obj_details in obj_data.items():
                    if isinstance(obj_details, dict):
                        objectives[obj_name] = obj_details

        # Generate summary
        summary = f"""# Mission Factual Summary

**Mission ID:** {mission_data['mission_id']}
**Mission Name:** {mission_data['mission_name']}
"""

        if duration:
            summary += f"""**Date:** {start_time.strftime('%Y-%m-%d')}
**Start Time:** {start_time.strftime('%H:%M:%S')}
**End Time:** {end_time.strftime('%H:%M:%S')}
**Duration:** {str(duration).split('.')[0]}
"""

        summary += f"""
## Crew Participation

**Total Participants:** {len(speaker_counts)}
**Total Communications:** {total_comms}
"""

        if total_comms > 0:
            avg_conf = sum(t['confidence'] for t in transcripts) / total_comms
            summary += f"**Average Confidence:** {avg_conf:.1%}\n"

        summary += "\n### Speaker Breakdown\n"

        for speaker, count in speaker_counts.most_common():
            pct = count / total_comms * 100
            avg_conf = sum(t['confidence'] for t in transcripts if t['speaker'] == speaker) / count
            summary += f"- {speaker}: {count} utterances ({pct:.1f}%), avg confidence {avg_conf:.1%}\n"

        # Objectives
        if objectives:
            comp_count = sum(1 for obj in objectives.values() if obj.get('Complete', False))
            summary += f"""
## Mission Objectives

**Total Objectives:** {len(objectives)}
**Completed:** {comp_count}
**Incomplete:** {len(objectives) - comp_count}

### Objective Details
"""

            for obj_name, obj_details in objectives.items():
                status = '✓ COMPLETE' if obj_details.get('Complete', False) else '✗ INCOMPLETE'
                current = obj_details.get('CurrentCount', 0)
                total_obj = obj_details.get('Count', 0)
                rank = obj_details.get('Rank', 'Unknown')
                desc = obj_details.get('Description', obj_name)

                summary += f"\n**[{rank}] {obj_name}** - {status}\n"
                summary += f"- Description: {desc}\n"
                summary += f"- Progress: {current}/{total_obj}\n"

        # Events
        event_types = Counter(e.get('event_type', 'unknown') for e in events)

        summary += f"""
## Events Recorded

**Total Events:** {len(events)}

### Event Type Distribution
"""

        for event_type, count in event_types.most_common(10):
            summary += f"- {event_type}: {count}\n"

        # Communication timeline
        if transcripts:
            summary += "\n## Communication Timeline\n\n### Mission Start (First 10 Communications)\n"

            for i, t in enumerate(transcripts[:10], 1):
                timestamp = t['timestamp'].split('T')[1][:8] if 'T' in t['timestamp'] else t['timestamp']
                summary += f'{i}. [{timestamp}] {t["speaker"]}: "{t["text"]}" (conf: {t["confidence"]:.2f})\n'

            summary += "\n### Mission End (Last 10 Communications)\n"

            for i, t in enumerate(transcripts[-10:], len(transcripts)-9):
                timestamp = t['timestamp'].split('T')[1][:8] if 'T' in t['timestamp'] else t['timestamp']
                summary += f'{i}. [{timestamp}] {t["speaker"]}: "{t["text"]}" (conf: {t["confidence"]:.2f})\n'

        # Final outcome
        if objectives:
            comp_rate = comp_count/len(objectives)*100
            summary += f"""
## Mission Outcome

**Status:** Mission completed with partial objective completion
**Completion Rate:** {comp_count}/{len(objectives)} objectives ({comp_rate:.1f}%)
"""

        if duration:
            summary += f"**Total Duration:** {str(duration).split('.')[0]}\n"

        if total_comms > 0:
            summary += f"**Data Quality:** {avg_conf:.1%} average transcription confidence\n"

        summary += """
---

*This summary contains only factual data extracted from mission telemetry and audio transcripts. No narrative interpretation or embellishment has been added.*
"""

        with open(output_file, 'w') as f:
            f.write(summary)

        logger.info(f"✓ Factual summary: {output_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate factual summary: {e}")
        return False


def generate_reports(mission_dir: Path, report_types: list = None, force: bool = False) -> dict:
    """
    Generate all requested report types for a mission.

    Args:
        mission_dir: Path to mission recording directory
        report_types: List of report types to generate (hybrid, story, factual, all)
        force: Overwrite existing reports

    Returns:
        Dictionary with generation results
    """
    results = {
        'hybrid': False,
        'story': False,
        'factual': False
    }

    if not report_types:
        report_types = ['all']

    logger.info(f"Processing mission: {mission_dir.name}")

    # Load mission data
    try:
        mission_data = load_mission_data(mission_dir)
    except Exception as e:
        logger.error(f"Failed to load mission data: {e}")
        return results

    if not mission_data['events']:
        logger.warning("No events found - skipping")
        return results

    # Create summarizer
    summarizer = MissionSummarizer(
        mission_id=mission_data['mission_id'],
        mission_name=mission_data['mission_name']
    )

    # Load data
    summarizer.load_events(mission_data['events'])
    summarizer.load_transcripts(mission_data['transcripts'])

    # Generate Hybrid Report (Training Assessment)
    if 'all' in report_types or 'hybrid' in report_types:
        output_file = mission_dir / "mission_report_HYBRID.md"

        if not output_file.exists() or force:
            try:
                logger.info("Generating hybrid training report...")
                report = summarizer.generate_hybrid_report(
                    style='professional',
                    output_file=output_file
                )
                if report:
                    results['hybrid'] = True
                    logger.info(f"✓ Hybrid report: {output_file}")
            except Exception as e:
                logger.error(f"Failed to generate hybrid report: {e}")
        else:
            logger.info(f"Hybrid report exists (use --force to overwrite)")
            results['hybrid'] = True

    # Generate Mission Story (Chronological Narrative)
    if 'all' in report_types or 'story' in report_types:
        output_file = mission_dir / "mission_story_TIMELINE.md"

        if not output_file.exists() or force:
            try:
                logger.info("Generating mission story...")
                story = summarizer.generate_mission_story(output_file=output_file)
                if story:
                    results['story'] = True
                    logger.info(f"✓ Mission story: {output_file}")
            except Exception as e:
                logger.error(f"Failed to generate story: {e}")
        else:
            logger.info(f"Mission story exists (use --force to overwrite)")
            results['story'] = True

    # Generate Factual Summary (Pure Data)
    if 'all' in report_types or 'factual' in report_types:
        if generate_factual_summary(mission_dir, mission_data):
            results['factual'] = True

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate all mission reports and stories from recordings'
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
        '--type',
        choices=['hybrid', 'story', 'factual', 'all'],
        action='append',
        dest='report_types',
        help='Report types to generate (can specify multiple, default: all)'
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

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        if not args.recordings_dir.exists():
            logger.error(f"Recordings directory not found: {args.recordings_dir}")
            sys.exit(1)

        # Find all mission directories
        mission_dirs = sorted(
            [d for d in args.recordings_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True  # Process newest first
        )

        if args.limit:
            mission_dirs = mission_dirs[:args.limit]

        logger.info(f"Found {len(mission_dirs)} mission recordings")

        total_success = 0
        for mission_dir in mission_dirs:
            results = generate_reports(
                mission_dir,
                report_types=args.report_types,
                force=args.force
            )
            if any(results.values()):
                total_success += 1

        logger.info(f"✓ Processed {total_success}/{len(mission_dirs)} missions")
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

    # Generate reports
    results = generate_reports(
        mission_path,
        report_types=args.report_types,
        force=args.force
    )

    # Report results
    logger.info("=" * 70)
    logger.info("Report Generation Complete")
    logger.info("=" * 70)
    logger.info(f"Hybrid Report (Training): {'✓' if results['hybrid'] else '✗'}")
    logger.info(f"Mission Story (Narrative): {'✓' if results['story'] else '✗'}")
    logger.info(f"Factual Summary (Data): {'✓' if results['factual'] else '✗'}")
    logger.info("=" * 70)

    if not any(results.values()):
        logger.error("✗ No reports generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
