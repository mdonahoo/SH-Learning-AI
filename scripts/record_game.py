#!/usr/bin/env python3
"""
Live Game Recording Script for Starship Horizons
Records real game events and generates mission summaries.
"""

import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.integration.game_recorder import GameRecorder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global recorder for signal handling
recorder = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global recorder
    logger.info("\n\nReceived interrupt signal...")
    if recorder and recorder.is_recording:
        summary = recorder.stop_recording()
        print("\n" + "=" * 60)
        print("RECORDING STOPPED")
        print("=" * 60)
        print(f"Mission ID: {summary.get('mission_id')}")
        print(f"Events recorded: {summary.get('statistics', {}).get('total_events', 0)}")
        print(f"Data saved to: {summary.get('export_path')}")
    sys.exit(0)


def main():
    """Main recording function."""
    global recorder

    print("=" * 60)
    print("STARSHIP HORIZONS - LIVE GAME RECORDER")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create recorder (uses environment variables)
    recorder = GameRecorder()

    print(f"Connecting to game at: {recorder.game_host}")

    # Test connection
    print("\nTesting connection to game...")
    if not recorder.client.test_connection():
        print("❌ Failed to connect to game!")
        print(f"Make sure Starship Horizons is running at {recorder.game_host}")
        return

    print("✅ Successfully connected to game!")

    # Get initial status
    status = recorder.client.get_game_status()
    if status:
        print(f"\nCurrent Game Status:")
        print(f"  State: {status.get('State')}")
        print(f"  Mode: {status.get('Mode')}")
        print(f"  Mission: {status.get('Mission') or 'None'}")

    # Start recording
    print("\n" + "-" * 60)
    print("Starting recording...")
    mission_id = recorder.start_recording()
    print(f"✅ Recording started with Mission ID: {mission_id}")
    print("\nPress Ctrl+C to stop recording and generate summary")
    print("-" * 60)

    # Main recording loop
    try:
        update_counter = 0
        while True:
            # Get live stats every 10 seconds
            if update_counter % 10 == 0:
                stats = recorder.get_live_stats()
                print(f"\n📊 Live Stats [{datetime.now().strftime('%H:%M:%S')}]")
                print(f"  Game State: {stats.get('game_state')}")
                print(f"  Events Recorded: {stats.get('events_recorded')}")
                print(f"  Recording Duration: {stats.get('recording_duration')}")
                print(f"  Events/Minute: {stats.get('events_per_minute', 0):.1f}")

            time.sleep(1)
            update_counter += 1

    except KeyboardInterrupt:
        # Handled by signal handler
        pass

    except Exception as e:
        logger.error(f"Recording error: {e}")
        if recorder and recorder.is_recording:
            recorder.stop_recording()


def quick_test():
    """Quick test to record for 30 seconds."""
    global recorder

    print("=" * 60)
    print("QUICK TEST - 30 SECOND RECORDING")
    print("=" * 60)

    recorder = GameRecorder()  # Uses environment variables

    if not recorder.client.test_connection():
        print("❌ Cannot connect to game")
        return

    print("✅ Connected to game")

    # Start recording
    mission_id = recorder.start_recording("Quick Test Mission")
    print(f"📹 Recording started: {mission_id}")

    # Record for 30 seconds
    for i in range(30, 0, -1):
        if i % 5 == 0:
            stats = recorder.get_live_stats()
            print(f"\n⏱️  {i} seconds remaining... Events: {stats.get('events_recorded', 0)}")
        time.sleep(1)

    # Stop and summarize
    print("\n⏹️  Stopping recording...")
    summary = recorder.stop_recording()

    print("\n" + "=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    print(f"Mission ID: {summary.get('mission_id')}")
    print(f"Total Events: {summary.get('statistics', {}).get('total_events', 0)}")
    print(f"Event Types: {summary.get('statistics', {}).get('event_types', {})}")
    print(f"Duration: {summary.get('statistics', {}).get('duration')}")
    print(f"Data saved to: {summary.get('export_path')}")

    # Generate summary
    print("\n📊 Generating mission summary...")
    summarizer = recorder.generate_summary()
    if summarizer:
        timeline = summarizer.generate_timeline()
        print(f"Timeline entries: {len(timeline)}")

        # Export report
        export_dir = Path(summary.get('export_path'))
        report_file = export_dir / "mission_report.md"
        summarizer.export_report(report_file, format="markdown")
        print(f"📄 Report saved to: {report_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record Starship Horizons game sessions")
    parser.add_argument("--test", action="store_true", help="Run a 30-second test recording")
    parser.add_argument("--host", help="Game host URL (overrides .env)")

    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        main()