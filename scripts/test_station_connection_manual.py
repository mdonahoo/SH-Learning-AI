#!/usr/bin/env python3
"""
Test connecting as different bridge stations
Try to receive telemetry data by joining as a crew member.
"""

import os
import json
import time
import websocket
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_station(station_name: str, role: str = None):
    """Test connection as a specific station."""
    print(f"\n{'='*60}")
    print(f"Testing as: {station_name}")
    print("="*60)

    # Get connection details from environment
    host = os.getenv('GAME_HOST', 'localhost')
    port = int(os.getenv('GAME_PORT_WS', '1865'))
    ws_url = f"ws://{host}:{port}"
    received_packets = []

    def on_message(ws, message):
        try:
            data = json.loads(message)
            cmd = data.get("Cmd") or data.get("cmd")
            if cmd and cmd not in ["PING"]:
                print(f"  Received: {cmd}")
                received_packets.append(cmd)
        except:
            pass

    def on_error(ws, error):
        print(f"  Error: {error}")

    def on_close(ws, close_code, close_msg):
        pass

    def on_open(ws):
        print(f"  Connected")

        # Try different authentication methods
        identifications = [
            # As a station
            {
                "cmd": "IDENTIFY",
                "Station": station_name,
                "Role": role or station_name,
                "Type": "Station"
            },
            # Join station
            {
                "cmd": "JOIN",
                "Station": station_name
            },
            # Set role
            {
                "cmd": "ROLE",
                "Role": role or station_name
            },
            # Subscribe to data
            {
                "cmd": "SUBSCRIBE",
                "Station": station_name,
                "Types": ["VESSEL", "DAMAGE", "ALERT"]
            }
        ]

        for ident in identifications:
            ws.send(json.dumps(ident))
            time.sleep(0.2)

        # Request vessel data
        ws.send(json.dumps({"cmd": "GET", "type": "VESSEL"}))

    try:
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Run for 5 seconds
        from threading import Thread, Event
        stop_event = Event()

        def run_ws():
            ws.run_forever()

        ws_thread = Thread(target=run_ws, daemon=True)
        ws_thread.start()

        time.sleep(5)
        ws.close()

        # Report results
        unique_packets = set(received_packets)
        if len(unique_packets) > 1:  # More than just SESSION
            print(f"  ✅ Received packet types: {unique_packets}")
            return True
        else:
            print(f"  ❌ Only basic packets received")
            return False

    except Exception as e:
        print(f"  Failed: {e}")
        return False


# Test different stations
stations = [
    ("Helm", "Pilot"),
    ("Tactical", "Weapons"),
    ("Science", "Sensors"),
    ("Engineering", "Engineer"),
    ("Communications", "Comms"),
    ("Captain", "Command"),
    ("MainViewer", "Observer"),
    ("GM", "GameMaster")
]

print("Testing different station connections...")
print("Looking for telemetry data access...")

successful = []

for station, role in stations:
    if test_station(station, role):
        successful.append(station)
    time.sleep(1)

print("\n" + "="*60)
print("RESULTS")
print("="*60)

if successful:
    print(f"✅ Successful connections: {successful}")
else:
    print("❌ No stations received telemetry data")
    print("\nPossible reasons:")
    print("  1. Game might need active players at stations")
    print("  2. Telemetry might be sent via HTTP polling instead")
    print("  3. Different authentication method required")
    print("  4. Game might be in wrong mode/state")