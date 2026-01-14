#!/usr/bin/env python3
"""
Complete Telemetry Recorder for Starship Horizons
Records mission events, objectives, AND ship system telemetry.
Supports extended recording sessions (2-10 minutes).
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent))

from src.metrics.event_recorder import EventRecorder
from src.metrics.mission_summarizer import MissionSummarizer
from src.integration.enhanced_game_client import EnhancedGameClient
from src.integration.websocket_telemetry_client import ShipTelemetryClient

# Globals for signal handling
event_recorder = None
game_client = None
telemetry_client = None
start_time = None
max_duration = None
bridge_id = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global event_recorder, game_client, telemetry_client

    print("\n\n⏹️  Stopping recording...")
    stop_recording()
    sys.exit(0)


def stop_recording():
    """Stop all recording and generate summary."""
    global event_recorder, game_client, telemetry_client, start_time

    recording_duration = (datetime.now() - start_time).total_seconds() if start_time else 0

    # Stop clients
    if game_client:
        game_summary = game_client.get_summary()
        game_client.stop_monitoring()

    if telemetry_client:
        ship_state = telemetry_client.get_ship_state()
        packet_stats = telemetry_client.get_packet_stats()
        telemetry_client.disconnect()

    print("\n" + "=" * 70)
    print("📊 RECORDING SUMMARY")
    print("=" * 70)

    # Recording stats
    print(f"\n⏱️  Duration: {recording_duration:.1f} seconds ({recording_duration/60:.1f} minutes)")

    # Game events summary
    if game_client:
        print(f"\n🎮 Game Events:")
        print(f"   Total meaningful events: {game_summary['total_events']}")
        print(f"   Objectives completed: {game_summary['objectives_completed']}")
        print(f"   Current grade: {game_summary['current_grade']:.1f}%")

    # Ship telemetry summary
    if telemetry_client:
        print(f"\n🚀 Ship Telemetry:")
        print(f"   Alert Level: {ship_state['alert_level']}")
        print(f"   Shields: {ship_state['shields']['percent']}%")
        print(f"   Hull: {ship_state['hull']['percent']}%")
        print(f"   Engines: {ship_state['engines']['speed']} speed")
        print(f"   Power: {ship_state['power']['total']}%")

        if packet_stats:
            print(f"\n📡 WebSocket Packets Received:")
            for packet_type, count in sorted(packet_stats.items())[:10]:
                print(f"   {packet_type}: {count}")

    # Save data
    if event_recorder:
        export_dir = Path("telemetry_recordings") / event_recorder.mission_id
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export events
        events_file = export_dir / "complete_telemetry.json"
        event_recorder.export_to_json(events_file)
        print(f"\n📁 Complete telemetry saved to: {events_file}")

        # Generate report
        summarizer = MissionSummarizer(
            mission_id=event_recorder.mission_id,
            mission_name=event_recorder.mission_name
        )
        summarizer.load_events(event_recorder.events)

        report_file = export_dir / "telemetry_report.md"
        summarizer.export_report(report_file, format="markdown")
        print(f"📄 Report saved to: {report_file}")

        # Save final ship state
        if telemetry_client:
            import json
            state_file = export_dir / "final_ship_state.json"
            with open(state_file, 'w') as f:
                json.dump(ship_state, f, indent=2)
            print(f"🚢 Ship state saved to: {state_file}")


def main(duration_minutes: int = 2):
    """
    Main recording function.

    Args:
        duration_minutes: Recording duration in minutes (2-10)
    """
    global event_recorder, game_client, telemetry_client, start_time, max_duration

    print("=" * 70)
    print("🚀 STARSHIP HORIZONS - COMPLETE TELEMETRY RECORDER")
    print("=" * 70)
    print(f"Recording Duration: {duration_minutes} minutes")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print("Recording:")
    print("  ✓ Mission objectives and progress")
    print("  ✓ Ship system telemetry (shields, hull, engines)")
    print("  ✓ Alert level changes")
    print("  ✓ Damage reports")
    print("  ✓ Combat events")
    print("-" * 70)

    # Setup
    signal.signal(signal.SIGINT, signal_handler)
    start_time = datetime.now()
    max_duration = timedelta(minutes=duration_minutes)

    # Initialize event recorder with optional bridge_id prefix
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    if bridge_id:
        mission_id = f"{bridge_id}_TELEMETRY_{timestamp}"
    else:
        mission_id = f"TELEMETRY_{timestamp}"
    event_recorder = EventRecorder(
        mission_id=mission_id,
        mission_name=f"Complete Telemetry Recording - {duration_minutes} min",
        bridge_id=bridge_id
    )

    # 1. Connect to game API
    print("\n📡 Connecting to game API...")
    game_client = EnhancedGameClient()  # Uses environment variables

    if not game_client.test_connection():
        print("❌ Cannot connect to game API")
        return

    print("✅ Game API connected")

    # 2. Connect to WebSocket for telemetry
    print("\n🔌 Connecting to telemetry WebSocket...")
    telemetry_client = ShipTelemetryClient()  # Uses environment variables

    ws_connected = telemetry_client.connect()
    if ws_connected:
        print("✅ WebSocket connected - receiving real-time telemetry")
    else:
        print("⚠️  WebSocket connection failed - telemetry limited")

    # 3. Setup event handlers
    event_count = {"game": 0, "telemetry": 0}

    def handle_game_event(event):
        """Handle game events (objectives, mission status)."""
        nonlocal event_count
        event_count["game"] += 1

        event_type = event.get("type", "unknown")

        # Important events to display
        display_events = [
            "mission_started", "mission_ended", "mission_completed",
            "objective_discovered", "objective_completed", "objective_progress",
            "grade_changed", "alert_level_changed"
        ]

        if event_type in display_events:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"  [{timestamp}] 🎮 {event_type}: {event.get('data', {})}")

        # Record event
        event_recorder.record_event(
            event_type=event_type,
            category=event.get("category", "game"),
            data=event.get("data", {})
        )

    def handle_telemetry_event(event):
        """Handle telemetry events (ship systems)."""
        nonlocal event_count
        event_count["telemetry"] += 1

        event_type = event.get("type", "unknown")

        # Important telemetry to display
        display_telemetry = [
            "alert_level_changed", "shields_changed", "hull_changed",
            "damage_report", "system_failure", "weapons_fired"
        ]

        if event_type in display_telemetry:
            timestamp = datetime.now().strftime("%H:%M:%S")
            data = event.get("data", {})

            # Format display based on type
            if "changed" in event_type:
                system = data.get("system", event_type.replace("_changed", ""))
                old_val = data.get("old_value", "?")
                new_val = data.get("new_value", "?")
                print(f"  [{timestamp}] 🚀 {system}: {old_val} → {new_val}")
            else:
                print(f"  [{timestamp}] 🚀 {event_type}: {data}")

        # Record event
        event_recorder.record_event(
            event_type=event_type,
            category=event.get("category", "telemetry"),
            data=event.get("data", {})
        )

    # Register handlers
    game_client.add_event_callback(handle_game_event)
    if telemetry_client:
        telemetry_client.add_callback(handle_telemetry_event)

    # 4. Start monitoring
    print("\n📊 Starting recording...")
    print("   (Press Ctrl+C to stop early)")
    print("\n" + "-" * 70)
    print("LIVE FEED:")
    print("-" * 70)

    game_client.start_monitoring(interval=0.5)

    # 5. Recording loop with progress updates
    elapsed = timedelta(0)
    last_update = datetime.now()
    update_interval = 30  # Update every 30 seconds

    try:
        while elapsed < max_duration:
            current_time = datetime.now()
            elapsed = current_time - start_time

            # Progress update
            if (current_time - last_update).total_seconds() >= update_interval:
                remaining = max_duration - elapsed
                minutes_elapsed = elapsed.total_seconds() / 60
                minutes_remaining = remaining.total_seconds() / 60

                print(f"\n  ⏱️  [{current_time.strftime('%H:%M:%S')}] "
                      f"Elapsed: {minutes_elapsed:.1f} min, "
                      f"Remaining: {minutes_remaining:.1f} min")
                print(f"     Events recorded - Game: {event_count['game']}, "
                      f"Telemetry: {event_count['telemetry']}")

                # Show current ship status if available
                if telemetry_client:
                    ship_state = telemetry_client.get_ship_state()
                    print(f"     Ship Status - Alert: {ship_state['alert_level']}, "
                          f"Shields: {ship_state['shields']['percent']}%, "
                          f"Hull: {ship_state['hull']['percent']}%")

                last_update = current_time

            time.sleep(1)

        print(f"\n⏰ Recording duration reached ({duration_minutes} minutes)")

    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")

    # Stop and summarize
    stop_recording()


def test_telemetry():
    """Quick test of telemetry recording (30 seconds)."""
    global event_recorder, game_client, telemetry_client, start_time

    print("=" * 70)
    print("🧪 TELEMETRY TEST - 30 SECONDS")
    print("=" * 70)

    start_time = datetime.now()

    # Initialize
    mission_id = f"TEST_{start_time.strftime('%Y%m%d_%H%M%S')}"
    event_recorder = EventRecorder(mission_id=mission_id, mission_name="Telemetry Test")

    # Connect clients (uses environment variables)
    game_client = EnhancedGameClient()
    telemetry_client = ShipTelemetryClient()

    if not game_client.test_connection():
        print("❌ Cannot connect to game")
        return

    print("✅ Game connected")

    ws_connected = telemetry_client.connect()
    print(f"{'✅' if ws_connected else '⚠️'} WebSocket {'connected' if ws_connected else 'failed'}")

    # Simple event counting
    event_counts = {"game": 0, "telemetry": 0}

    def count_game(event):
        event_counts["game"] += 1
        event_recorder.record_event(
            event.get("type", "unknown"),
            event.get("category", "game"),
            event.get("data", {})
        )

    def count_telemetry(event):
        event_counts["telemetry"] += 1
        event_recorder.record_event(
            event.get("type", "unknown"),
            event.get("category", "telemetry"),
            event.get("data", {})
        )

    game_client.add_event_callback(count_game)
    if telemetry_client:
        telemetry_client.add_callback(count_telemetry)

    # Monitor
    game_client.start_monitoring(interval=0.5)

    print("\nRecording for 30 seconds...")
    for i in range(30, 0, -1):
        if i % 5 == 0:
            print(f"  {i} sec - Game events: {event_counts['game']}, "
                  f"Telemetry: {event_counts['telemetry']}")
        time.sleep(1)

    # Stop
    game_client.stop_monitoring()
    if telemetry_client:
        ship_state = telemetry_client.get_ship_state()
        telemetry_client.disconnect()

        print(f"\nFinal Ship State:")
        print(f"  Alert: {ship_state['alert_level']}")
        print(f"  Shields: {ship_state['shields']['percent']}%")
        print(f"  Hull: {ship_state['hull']['percent']}%")

    print(f"\nTotal Events: {event_counts['game'] + event_counts['telemetry']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Telemetry Recorder for Starship Horizons")
    parser.add_argument("--duration", type=int, default=2,
                       help="Recording duration in minutes (2-10)")
    parser.add_argument("--test", action="store_true",
                       help="Run 30-second test")
    parser.add_argument(
        "--bridge-id",
        default=os.getenv('BRIDGE_ID'),
        help="Bridge identifier for multi-bridge deployments (default from .env)"
    )

    args = parser.parse_args()
    bridge_id = args.bridge_id

    if args.test:
        test_telemetry()
    else:
        # Validate duration
        duration = min(max(args.duration, 2), 10)  # Clamp between 2 and 10
        main(duration_minutes=duration)