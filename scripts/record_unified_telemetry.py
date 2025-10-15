#!/usr/bin/env python3
"""
Unified Telemetry Recorder for Starship Horizons
Combines all working telemetry sources:
- Mission objectives via HTTP API
- Alert changes via WebSocket
- Game variables via WebSocket
- Damage reports via WebSocket
- System events via WebSocket
"""

import sys
import time
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))

from src.metrics.event_recorder import EventRecorder
from src.metrics.mission_summarizer import MissionSummarizer
from src.integration.enhanced_game_client import EnhancedGameClient
from src.integration.browser_mimic_websocket import BrowserMimicWebSocket

# Globals for signal handling
event_recorder = None
game_client = None
websocket_client = None
start_time = None
max_duration = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⏹️  Stopping recording...")
    stop_recording()
    sys.exit(0)


def stop_recording():
    """Stop all recording and generate summary."""
    global event_recorder, game_client, websocket_client, start_time

    recording_duration = (datetime.now() - start_time).total_seconds() if start_time else 0

    # Stop clients
    if game_client:
        game_summary = game_client.get_summary()
        game_client.stop_monitoring()

    ws_data = {}
    final_ship_status = None
    all_vessels_status = {}
    if websocket_client:
        final_ship_status = websocket_client.get_ship_status_summary()
        all_vessels_status = websocket_client.get_all_vessels_status()
        ws_data = {
            "vessel_data": websocket_client.get_vessel_data(),
            "packet_stats": websocket_client.get_packet_stats(),
            "all_vessels": all_vessels_status
        }
        websocket_client.disconnect()

    print("\n" + "=" * 70)
    print("📊 UNIFIED RECORDING SUMMARY")
    print("=" * 70)

    # Recording stats
    print(f"\n⏱️  Duration: {recording_duration:.1f} seconds ({recording_duration/60:.1f} minutes)")

    # Game events summary
    if game_client:
        print(f"\n🎮 Mission Events (HTTP API):")
        print(f"   Total meaningful events: {game_summary['total_events']}")
        print(f"   Objectives completed: {game_summary['objectives_completed']}")
        print(f"   Current grade: {game_summary['current_grade']:.1f}%")

    # WebSocket telemetry summary
    if ws_data.get("vessel_data"):
        print(f"\n🚀 Ship Telemetry (WebSocket):")
        vessel_data = ws_data["vessel_data"]

        # Show alert level
        if "alert" in vessel_data:
            alert_level = vessel_data["alert"]
            alert_names = {1: "Condition 5 (Docked)", 2: "Green", 3: "Yellow", 4: "Red", 5: "Red Alert"}
            print(f"   Alert Level: {alert_names.get(alert_level, alert_level)}")

        # Show comprehensive ship status
        if final_ship_status:
            print(f"\n🛡️ FINAL SHIP STATUS:")
            print(f"   Hull: {final_ship_status.get('hull', 'Unknown')}%")
            print(f"   Shields: {final_ship_status.get('shields', 'Unknown')}%")

            # Show shield quadrants if damaged
            shield_sections = final_ship_status.get('shield_sections', {})
            damaged_shields = {sec: str for sec, str in shield_sections.items() if str < 100}
            if damaged_shields:
                print("   Shield Quadrants:")
                for section, strength in damaged_shields.items():
                    status = "🔴" if strength < 30 else "🟡" if strength < 70 else "🟢"
                    print(f"     {status} {section.capitalize()}: {strength}%")

            # Show damaged systems
            systems = final_ship_status.get('systems_health', {})
            damaged_systems = {sys: health for sys, health in systems.items() if health < 100}
            if damaged_systems:
                print("   Damaged Systems:")
                for sys, health in damaged_systems.items():
                    status = "🔴" if health < 30 else "🟡" if health < 70 else "🟢"
                    print(f"     {status} {sys.replace('_', ' ').title()}: {health}%")

            print(f"   Total Damage Taken: {final_ship_status.get('cumulative_damage', 0)}")
            print(f"   Damage Events: {final_ship_status.get('damage_events', 0)}")

        # Show other vessels' status if available
        if all_vessels_status:
            print(f"\n🎯 OTHER VESSELS STATUS ({len(all_vessels_status)} contacts):")
            for vessel_id, status in list(all_vessels_status.items())[:5]:  # Show first 5
                name = status.get('name', 'Unknown')
                faction = status.get('faction', 'Unknown')
                hull = status.get('hull', 100)
                shields = status.get('shields', 0)
                vtype = status.get('type', 'Unknown')

                # Status indicators
                hull_status = "🔴" if hull < 30 else "🟡" if hull < 70 else "🟢"
                shield_status = "🔴" if shields < 30 else "🟡" if shields < 70 else "🟢"

                print(f"   [{vessel_id}] {name} ({faction}/{vtype})")
                print(f"      {hull_status} Hull: {hull:.0f}% | {shield_status} Shields: {shields:.0f}%")

        # Show variables
        var_count = sum(1 for k in vessel_data.keys() if k.startswith("var_"))
        if var_count > 0:
            print(f"   Game Variables: {var_count} captured")
            for key, value in vessel_data.items():
                if key.startswith("var_"):
                    print(f"     - {key[4:]}: {value}")

    if ws_data.get("packet_stats"):
        print(f"\n📡 WebSocket Packets Received:")
        for packet_type, count in sorted(ws_data["packet_stats"].items(), key=lambda x: x[1], reverse=True)[:8]:
            if packet_type != "PING":
                print(f"   {packet_type}: {count}")

    # Save data
    if event_recorder:
        export_dir = Path("unified_recordings") / event_recorder.mission_id
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export events
        events_file = export_dir / "unified_telemetry.json"
        event_recorder.export_to_json(events_file, ship_status=final_ship_status)
        print(f"\n📁 Complete telemetry saved to: {events_file}")

        # Generate report
        summarizer = MissionSummarizer(
            mission_id=event_recorder.mission_id,
            mission_name=event_recorder.mission_name
        )
        summarizer.load_events(event_recorder.events)

        report_file = export_dir / "unified_report.md"
        summarizer.export_report(report_file, format="markdown")
        print(f"📄 Report saved to: {report_file}")

        # Save WebSocket data
        if ws_data.get("vessel_data"):
            import json
            ws_file = export_dir / "websocket_telemetry.json"
            with open(ws_file, 'w') as f:
                json.dump({
                    "vessel_data": ws_data["vessel_data"],
                    "packet_stats": ws_data["packet_stats"]
                }, f, indent=2)
            print(f"🔌 WebSocket data saved to: {ws_file}")


def main(duration_minutes: int = 2):
    """
    Main recording function.

    Args:
        duration_minutes: Recording duration in minutes (2-10)
    """
    global event_recorder, game_client, websocket_client, start_time, max_duration

    print("=" * 70)
    print("🚀 STARSHIP HORIZONS - UNIFIED TELEMETRY RECORDER")
    print("=" * 70)
    print(f"Recording Duration: {duration_minutes} minutes")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print("Recording ALL Available Data:")
    print("  ✅ Mission objectives and progress (HTTP API)")
    print("  ✅ Alert level changes (WebSocket)")
    print("  ✅ Game variables (WebSocket)")
    print("  ✅ Damage reports (WebSocket)")
    print("  ✅ System events (WebSocket)")
    print("-" * 70)

    # Setup
    signal.signal(signal.SIGINT, signal_handler)
    start_time = datetime.now()
    max_duration = timedelta(minutes=duration_minutes)

    # Initialize event recorder
    mission_id = f"UNIFIED_{start_time.strftime('%Y%m%d_%H%M%S')}"
    event_recorder = EventRecorder(
        mission_id=mission_id,
        mission_name=f"Unified Recording - {duration_minutes} min"
    )

    # 1. Connect to game API (HTTP)
    print("\n📡 Connecting to game API...")
    game_client = EnhancedGameClient(host="http://66.68.47.235:1864")

    if not game_client.test_connection():
        print("❌ Cannot connect to game API")
        return

    print("✅ Game API connected")

    # 2. Connect to WebSocket with browser protocol
    print("\n🔌 Connecting to WebSocket (browser protocol)...")
    websocket_client = BrowserMimicWebSocket(host="66.68.47.235", port=1865)

    ws_connected = websocket_client.connect(screen_name="TelemetryRecorder", is_main_viewer=True)
    if ws_connected:
        print("✅ WebSocket connected - receiving real-time telemetry")
    else:
        print("⚠️  WebSocket connection failed - limited telemetry")

    # 3. Setup event handlers
    event_counts = {"http": 0, "websocket": 0}
    last_alert = None

    def handle_game_event(event):
        """Handle HTTP API events (objectives, mission status)."""
        nonlocal event_counts
        event_counts["http"] += 1

        event_type = event.get("type", "unknown")

        # Important events to display
        display_events = [
            "mission_started", "mission_ended", "mission_completed",
            "objective_discovered", "objective_completed", "objective_progress",
            "grade_changed"
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

    def handle_websocket_event(event):
        """Handle WebSocket events (telemetry, alerts)."""
        nonlocal event_counts, last_alert
        event_counts["websocket"] += 1

        event_type = event.get("type", "unknown")
        data = event.get("data", {})

        # Display important telemetry
        if event_type == "alert_change":
            alert_value = data
            alert_names = {1: "Docked", 2: "Green", 3: "Yellow", 4: "Red", 5: "Red Alert"}
            alert_name = alert_names.get(alert_value, f"Level {alert_value}")

            if alert_value != last_alert:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"  [{timestamp}] 🚨 Alert Level: {alert_name}")
                last_alert = alert_value

        elif event_type == "variable_update":
            key = data.get("key", "")
            value = data.get("value", "")
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"  [{timestamp}] 📊 Variable: {key} = {value}")

        elif event_type == "damage_report":
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"  [{timestamp}] 💥 Damage: {data}")

        elif event_type in ["shields_changed", "hull_changed", "power_changed"]:
            system = data.get("system", event_type.replace("_changed", ""))
            old_val = data.get("old_value", "?")
            new_val = data.get("new_value", "?")
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"  [{timestamp}] 🚀 {system}: {old_val} → {new_val}")

        # Record all events
        event_recorder.record_event(
            event_type=event_type,
            category=event.get("category", "telemetry"),
            data=data
        )

    # Register handlers
    game_client.add_event_callback(handle_game_event)
    if websocket_client:
        websocket_client.add_callback(handle_websocket_event)

    # 4. Start monitoring
    print("\n📊 Starting unified recording...")
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
                print(f"     Events - HTTP: {event_counts['http']}, "
                      f"WebSocket: {event_counts['websocket']}")

                # Request fresh data
                if websocket_client:
                    try:
                        websocket_client.request_vessel_data()
                    except:
                        pass  # Ignore errors during data request

                last_update = current_time

            time.sleep(1)

        print(f"\n⏰ Recording duration reached ({duration_minutes} minutes)")

    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")
    except Exception as e:
        print(f"\n❌ Recording error: {e}")

    # Stop and summarize
    try:
        stop_recording()
    except Exception as e:
        print(f"❌ Error during stop: {e}")
        # Force cleanup
        if game_client:
            try:
                game_client.stop_monitoring()
            except:
                pass
        if websocket_client:
            try:
                websocket_client.disconnect()
            except:
                pass


def quick_test():
    """Quick 30-second test of unified recording."""
    global event_recorder, game_client, websocket_client, start_time

    print("=" * 70)
    print("🧪 UNIFIED TELEMETRY TEST - 30 SECONDS")
    print("=" * 70)

    start_time = datetime.now()

    # Initialize
    mission_id = f"TEST_{start_time.strftime('%Y%m%d_%H%M%S')}"
    event_recorder = EventRecorder(mission_id=mission_id, mission_name="Unified Test")

    # Connect clients
    game_client = EnhancedGameClient(host="http://66.68.47.235:1864")
    websocket_client = BrowserMimicWebSocket(host="66.68.47.235", port=1865)

    if not game_client.test_connection():
        print("❌ Cannot connect to game")
        return

    print("✅ Game API connected")

    ws_connected = websocket_client.connect(screen_name="TestRecorder", is_main_viewer=True)
    print(f"{'✅' if ws_connected else '⚠️'} WebSocket {'connected' if ws_connected else 'failed'}")

    # Simple event counting
    event_counts = {"http": 0, "websocket": 0}

    def count_http(event):
        event_counts["http"] += 1
        event_recorder.record_event(
            event.get("type", "unknown"),
            event.get("category", "game"),
            event.get("data", {})
        )

    def count_websocket(event):
        event_counts["websocket"] += 1
        event_type = event.get("type", "unknown")

        # Show important events
        if event_type in ["alert_change", "variable_update", "damage_report"]:
            print(f"  WebSocket: {event_type} - {event.get('data', {})}")

        event_recorder.record_event(
            event_type,
            event.get("category", "telemetry"),
            event.get("data", {})
        )

    game_client.add_event_callback(count_http)
    if websocket_client:
        websocket_client.add_callback(count_websocket)

    # Monitor
    game_client.start_monitoring(interval=0.5)

    print("\nRecording for 30 seconds...")
    print("(Try changing alert levels or ship systems)")
    print("-" * 70)

    for i in range(30, 0, -1):
        if i % 5 == 0:
            print(f"  {i} sec - HTTP events: {event_counts['http']}, "
                  f"WebSocket: {event_counts['websocket']}")

            if websocket_client and i % 10 == 0:
                websocket_client.request_vessel_data()

        time.sleep(1)

    # Stop
    game_client.stop_monitoring()
    if websocket_client:
        vessel_data = websocket_client.get_vessel_data()
        websocket_client.disconnect()

        print(f"\nCaptured WebSocket data:")
        for key, value in vessel_data.items():
            print(f"  {key}: {value}")

    print(f"\nTotal Events - HTTP: {event_counts['http']}, WebSocket: {event_counts['websocket']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Telemetry Recorder for Starship Horizons")
    parser.add_argument("--duration", type=int, default=2,
                       help="Recording duration in minutes (2-10)")
    parser.add_argument("--test", action="store_true",
                       help="Run 30-second test")

    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        # Validate duration
        duration = min(max(args.duration, 2), 10)  # Clamp between 2 and 10
        main(duration_minutes=duration)