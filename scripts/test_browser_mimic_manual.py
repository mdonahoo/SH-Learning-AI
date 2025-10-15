#!/usr/bin/env python3
"""
Test Browser-Mimicking WebSocket Client
Monitor ship telemetry with exact browser protocol.
"""

import os
import time
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

from src.integration.browser_mimic_websocket import BrowserMimicWebSocket

# Load environment variables
load_dotenv()

print("=" * 70)
print("🚀 BROWSER-MIMIC WEBSOCKET TEST")
print("=" * 70)
print("This mimics the browser's exact WebSocket protocol")
print("Try changing ship systems (shields, alert level, speed, etc.)")
print("-" * 70)

# Get connection details from environment
host = os.getenv('GAME_HOST', 'localhost')
port = int(os.getenv('GAME_PORT_WS', '1865'))

print(f"Connecting to {host}:{port}")

# Connect as MainViewer (like browser)
client = BrowserMimicWebSocket(host=host, port=port)

# Track events
event_count = 0
telemetry_events = []

def handle_event(event):
    global event_count
    event_count += 1

    event_type = event.get("type", "unknown")
    category = event.get("category", "")

    # Show telemetry events
    if category == "telemetry" or "changed" in event_type:
        timestamp = datetime.now().strftime("%H:%M:%S")
        data = event.get("data", {})
        print(f"[{timestamp}] 🎯 {event_type}: {data}")
        telemetry_events.append(event)

    # Show other important events
    elif event_type in ["damage_report", "alert_change", "variable_update"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 📍 {event_type}: {event.get('data', {})}")

client.add_callback(handle_event)

# Connect
if not client.connect(screen_name="Observer", is_main_viewer=True):
    print("❌ Failed to connect")
    exit(1)

print("✅ Connected with browser protocol")
time.sleep(2)  # Give time for packet registration

print("\n📡 Monitoring for 60 seconds...")
print("   (Change ship systems in game to see telemetry)")
print("-" * 70)

# Monitor and request updates
for i in range(60):
    if i % 10 == 0:
        # Request fresh data every 10 seconds
        client.request_vessel_data()

        remaining = 60 - i
        print(f"\n⏱️  {remaining} sec - Events: {event_count}, Telemetry: {len(telemetry_events)}")

        # Show packet stats
        stats = client.get_packet_stats()
        if stats:
            received_types = [f"{k}({v})" for k, v in stats.items() if k != "PING"]
            if received_types:
                print(f"   Packets: {', '.join(received_types[:5])}")

        # Show vessel data if available
        vessel_data = client.get_vessel_data()
        if vessel_data:
            # Show key telemetry fields
            telemetry = []
            for field in ["shields", "hull", "alert", "speed", "energy", "power"]:
                if field in vessel_data:
                    telemetry.append(f"{field}={vessel_data[field]}")

            if telemetry:
                print(f"   Telemetry: {', '.join(telemetry[:4])}")

    time.sleep(1)

# Disconnect and summarize
client.disconnect()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTotal events: {event_count}")
print(f"Telemetry events: {len(telemetry_events)}")

# Packet statistics
stats = client.get_packet_stats()
print(f"\nPacket types received:")
for packet_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {packet_type}: {count}")

# Vessel data
vessel_data = client.get_vessel_data()
if vessel_data:
    print(f"\n🚢 Vessel data captured ({len(vessel_data)} fields):")
    for key, value in list(vessel_data.items())[:10]:
        print(f"  {key}: {value}")

# Telemetry events
if telemetry_events:
    print(f"\n🎯 Telemetry changes detected:")
    for event in telemetry_events[:10]:
        data = event.get("data", {})
        print(f"  {event['type']}: {data.get('old_value')} → {data.get('new_value')}")
else:
    print("\n⚠️  No telemetry changes detected")
    print("  Make sure to change ship systems while the test is running")
    print("  (shields on/off, alert levels, engine power, etc.)")