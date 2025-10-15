#!/usr/bin/env python3
"""
Test WebSocket with Live Game
Monitor for ship status changes while you play.
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from src.integration.working_websocket_client import StarshipWebSocketClient

# Load environment variables
load_dotenv()

print("=" * 70)
print("🚀 WEBSOCKET LIVE TEST")
print("=" * 70)
print("This will monitor the game for 60 seconds")
print("Try changing ship systems (shields, alert level, etc.)")
print("-" * 70)

# Get connection details from environment
host = os.getenv('GAME_HOST', 'localhost')
port = int(os.getenv('GAME_PORT_WS', '1865'))

print(f"Connecting to {host}:{port}")

# Connect
client = StarshipWebSocketClient(host=host, port=port)

if not client.connect():
    print("❌ Failed to connect")
    exit(1)

print("✅ Connected to WebSocket")
time.sleep(1)

# Track events
event_count = 0
significant_events = []

def handle_event(event):
    global event_count
    event_count += 1

    event_type = event.get("type", "unknown")

    # Show significant events
    if "changed" in event_type or event_type in ["damage_report", "alert_changed", "contacts_update"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {event_type}: {event.get('data', {})}")
        significant_events.append(event)

client.add_callback(handle_event)

print("\nMonitoring for 60 seconds...")
print("(Change ship systems in the game to see events)")
print("-" * 70)

# Monitor and periodically request updates
for i in range(60):
    if i % 10 == 0:
        # Request fresh data every 10 seconds
        client.request_update()
        remaining = 60 - i
        print(f"\n⏱️  {remaining} seconds remaining... Total events: {event_count}")

        # Show current ship data if available
        if client.ship_data:
            print(f"   Ship data fields: {list(client.ship_data.keys())}")

    time.sleep(1)

# Disconnect and summarize
client.disconnect()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = client.get_summary()

print(f"\nTotal events: {event_count}")
print(f"Significant events: {len(significant_events)}")

print(f"\nPacket types received: {summary['packet_types']}")

print(f"\nPacket counts:")
for pkt_type, count in summary['packet_counts'].items():
    print(f"  {pkt_type}: {count}")

if client.ship_data:
    print(f"\nShip data captured:")
    for key, value in client.ship_data.items():
        print(f"  {key}: {value}")
else:
    print("\n⚠️  No ship data captured")
    print("  Make sure the game is running and you're in a mission")

if significant_events:
    print(f"\n🎯 Significant events detected:")
    for event in significant_events[:5]:
        print(f"  - {event['type']}: {event.get('data', {})}")