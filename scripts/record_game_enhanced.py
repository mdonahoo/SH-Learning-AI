#!/usr/bin/env python3
"""
Enhanced Game Recorder - Records Meaningful Game Events Only
"""

import sys
import time
import signal
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.metrics.event_recorder import EventRecorder
from src.metrics.mission_summarizer import MissionSummarizer
from src.integration.enhanced_game_client import EnhancedGameClient

# Global for signal handling
recorder = None
client = None


def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    global recorder, client
    print("\n\n⏹️  Stopping recording...")

    if client:
        summary = client.get_summary()
        client.stop_monitoring()

        print("\n" + "=" * 60)
        print("RECORDING SUMMARY")
        print("=" * 60)
        print(f"Total meaningful events: {summary['total_events']}")
        print(f"Objectives completed: {summary['objectives_completed']}")
        print(f"Current grade: {summary['current_grade']:.1f}%")
        print("\nEvent breakdown:")
        for event_type, count in summary['event_types'].items():
            print(f"  {event_type}: {count}")

    if recorder:
        # Export data
        export_dir = Path("game_recordings") / recorder.mission_id
        export_dir.mkdir(parents=True, exist_ok=True)

        events_file = export_dir / "meaningful_events.json"
        recorder.export_to_json(events_file)
        print(f"\n📁 Events saved to: {events_file}")

        # Generate summary
        summarizer = MissionSummarizer(
            mission_id=recorder.mission_id,
            mission_name=recorder.mission_name
        )
        summarizer.load_events(recorder.events)

        # Create report
        report_file = export_dir / "mission_analysis.md"
        summarizer.export_report(report_file, format="markdown")
        print(f"📄 Analysis saved to: {report_file}")

    sys.exit(0)


def main():
    """Main recording function."""
    global recorder, client

    print("=" * 60)
    print("🎮 STARSHIP HORIZONS - ENHANCED EVENT RECORDER")
    print("=" * 60)
    print("Recording meaningful game events only:")
    print("  ✓ Mission start/end")
    print("  ✓ Objective discovery and completion")
    print("  ✓ Grade changes")
    print("  ✓ Progress milestones")
    print("-" * 60)

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create enhanced client (uses environment variables)
    client = EnhancedGameClient()

    # Test connection
    print("\n🔌 Connecting to game...")
    if not client.test_connection():
        print(f"❌ Cannot connect to game at {client.host}")
        return

    print("✅ Connected successfully!\n")

    # Create event recorder
    mission_id = f"ENHANCED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recorder = EventRecorder(
        mission_id=mission_id,
        mission_name="Enhanced Recording Session"
    )

    # Setup event handler
    def handle_game_event(event):
        """Record meaningful game events."""
        # Map event types to categories
        category_map = {
            "session_start": "session",
            "mission_started": "mission",
            "mission_ended": "mission",
            "mission_changed": "mission",
            "mission_info": "mission",
            "mission_completed": "mission",
            "objective_discovered": "objectives",
            "objective_progress": "objectives",
            "objective_completed": "objectives",
            "objective_failed": "objectives",
            "grade_changed": "performance",
            "time_milestone": "progress"
        }

        event_type = event.get("type", "unknown")
        category = category_map.get(event_type, "game")

        # Special handling for different event types
        if event_type == "objective_completed":
            # Record as achievement
            recorder.record_event(
                event_type="achievement",
                category="objectives",
                data={
                    **event.get("data", {}),
                    "event_type": event_type,
                    "timestamp": event.get("timestamp")
                }
            )
            print(f"  🎯 Objective Completed: {event['data']['name']}")

        elif event_type == "grade_changed":
            # Record performance change
            recorder.record_event(
                event_type="performance_update",
                category="performance",
                data=event.get("data", {})
            )
            print(f"  📊 Grade: {event['data']['percentage']}")

        elif event_type == "mission_started":
            print(f"  🚀 Mission Started: {event['data']['mission']}")
            recorder.record_event(event_type, category, event.get("data", {}))

        elif event_type == "mission_completed":
            success = "✅ SUCCESS" if event['data'].get('success') else "❌ FAILED"
            print(f"  🏁 Mission Complete: {success} - Grade: {event['data']['final_grade']*100:.1f}%")
            recorder.record_event(event_type, category, event.get("data", {}))

        elif event_type == "objective_discovered":
            print(f"  📋 New Objective: {event['data']['name']}")
            recorder.record_event(event_type, category, event.get("data", {}))

        elif event_type == "objective_progress":
            print(f"  📈 Progress: {event['data']['name']} - {event['data']['percentage']}")
            recorder.record_event(event_type, category, event.get("data", {}))

        elif event_type == "time_milestone":
            print(f"  ⏱️  {event['data']['elapsed_time']} elapsed")
            recorder.record_event(event_type, category, event.get("data", {}))

        else:
            # Record all other events
            recorder.record_event(event_type, category, event.get("data", {}))

    client.add_event_callback(handle_game_event)

    # Start monitoring
    print("\n📡 Starting monitoring...")
    print("   (Only meaningful changes will be recorded)")
    print("\n" + "-" * 60)
    print("LIVE EVENT FEED:")
    print("-" * 60)

    client.start_monitoring(interval=0.5)

    # Keep running
    print("\nPress Ctrl+C to stop recording and generate analysis\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def test_run():
    """Test run for 60 seconds."""
    global recorder, client

    print("=" * 60)
    print("🧪 TEST RUN - 60 SECOND RECORDING")
    print("=" * 60)

    client = EnhancedGameClient()  # Uses environment variables

    if not client.test_connection():
        print("❌ Cannot connect to game")
        return

    print("✅ Connected\n")

    # Create recorder
    mission_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recorder = EventRecorder(mission_id=mission_id, mission_name="Test Recording")

    # Count events
    event_count = 0

    def count_events(event):
        nonlocal event_count
        event_count += 1
        event_type = event.get("type", "unknown")

        # Only print important events
        important_events = [
            "mission_started", "mission_ended", "mission_completed",
            "objective_discovered", "objective_completed", "objective_progress",
            "grade_changed"
        ]

        if event_type in important_events:
            print(f"  [{event_count}] {event_type}: {event.get('data', {})}")

        # Record event
        recorder.record_event(
            event_type=event_type,
            category=event.get("category", "game"),
            data=event.get("data", {})
        )

    client.add_event_callback(count_events)
    client.start_monitoring(interval=0.5)

    print("Recording for 60 seconds...\n")

    for i in range(60, 0, -1):
        if i % 10 == 0:
            print(f"  ⏱️  {i} seconds remaining... ({event_count} meaningful events so far)")
        time.sleep(1)

    client.stop_monitoring()

    # Summary
    summary = client.get_summary()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Meaningful events recorded: {event_count}")
    print(f"Objectives completed: {summary['objectives_completed']}")
    print(f"Current grade: {summary['current_grade']:.1f}%")

    if summary['event_types']:
        print("\nEvent breakdown:")
        for event_type, count in summary['event_types'].items():
            print(f"  {event_type}: {count}")
    else:
        print("\n⚠️  No meaningful events detected during test period")
        print("   Make sure the game is running and in an active mission")

    # Save events
    if recorder and recorder.events:
        export_dir = Path("game_recordings") / recorder.mission_id
        export_dir.mkdir(parents=True, exist_ok=True)
        events_file = export_dir / "test_events.json"
        recorder.export_to_json(events_file)
        print(f"\n📁 Events saved to: {events_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Starship Horizons Event Recorder")
    parser.add_argument("--test", action="store_true", help="Run 60-second test")

    args = parser.parse_args()

    if args.test:
        test_run()
    else:
        main()