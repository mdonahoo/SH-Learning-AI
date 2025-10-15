#!/usr/bin/env python3
"""
Test the enhanced telemetry capture with comprehensive packet collection.
"""

import json
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.integration.browser_mimic_websocket import BrowserMimicWebSocket

# Global for signal handling
websocket_client = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⏹️  Stopping enhanced telemetry test...")
    if websocket_client:
        print_comprehensive_summary()
        websocket_client.disconnect()
    sys.exit(0)

def print_comprehensive_summary():
    """Print comprehensive summary of all telemetry collected."""
    if not websocket_client:
        return

    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TELEMETRY SUMMARY")
    print("=" * 80)

    # Get all telemetry
    telemetry = websocket_client.get_comprehensive_telemetry()

    # Packet statistics
    packet_stats = telemetry['packet_stats']
    total_packets = sum(packet_stats.values())
    print(f"\n📦 PACKET STATISTICS:")
    print(f"   Total packets received: {total_packets}")
    print(f"   Unique packet types: {len(packet_stats)}")

    # Show top packet types
    sorted_packets = sorted(packet_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n   Top 10 packet types:")
    for ptype, count in sorted_packets:
        print(f"     {ptype}: {count}")

    # Mission data
    mission_data = telemetry['mission']
    print(f"\n🎯 MISSION DATA:")
    print(f"   Current mission: {mission_data.get('current_mission', 'None')}")
    print(f"   Objectives: {len(mission_data.get('objectives', []))}")
    print(f"   Player objectives: {len(mission_data.get('player_objectives', []))}")
    print(f"   GM objectives: {len(mission_data.get('gm_objectives', []))}")
    print(f"   Encounters: {len(mission_data.get('encounters', []))}")

    if mission_data.get('mission_briefing'):
        print(f"   Mission briefing: {str(mission_data['mission_briefing'])[:100]}...")

    # Ship internals
    internals = telemetry['ship_internals']
    print(f"\n🏢 SHIP INTERNALS:")
    print(f"   Decks mapped: {len(internals.get('decks', {}))}")
    print(f"   Current location: {internals.get('current_location', 'Unknown')}")
    print(f"   Personnel tracked: {len(internals.get('personnel', []))}")
    print(f"   Cameras: {len(internals.get('cameras', {}))}")
    print(f"   Devices: {len(internals.get('devices', {}))}")

    # Advanced systems
    advanced = telemetry['advanced_systems']
    print(f"\n⚙️ ADVANCED SYSTEMS:")
    print(f"   Components: {len(advanced.get('components', {}))}")
    print(f"   Factions: {len(advanced.get('factions', {}))}")
    if advanced.get('factions'):
        print(f"   Faction data: {advanced['factions']}")
    print(f"   Planetary systems: {len(advanced.get('planetary_systems', {}))}")
    print(f"   Map data: {'Yes' if advanced.get('map_data') else 'No'}")

    # Enhanced combat
    combat = telemetry['combat']
    print(f"\n💣 ENHANCED COMBAT:")
    print(f"   Ordnance types: {len(combat.get('ordnance', {}))}")
    if combat.get('ordnance'):
        print(f"   Ordnance inventory: {combat['ordnance']}")
    print(f"   Selected ordnance: {combat.get('ordnance_selected', 'None')}")
    print(f"   Active projectiles: {len(combat.get('projectiles', []))}")
    print(f"   Drones: {len(combat.get('drones', {}))}")

    # Multiplayer/Crew
    multiplayer = telemetry['multiplayer']
    print(f"\n👥 MULTIPLAYER/CREW:")
    print(f"   Cast mode: {multiplayer.get('cast_mode', False)}")
    print(f"   Channels: {len(multiplayer.get('channels', {}))}")
    print(f"   Broadcasts: {len(multiplayer.get('broadcasts', []))}")
    print(f"   Player vessels: {multiplayer.get('player_vessels', {})}")
    print(f"   Callsigns: {multiplayer.get('callsigns', {})}")

    # UI State
    ui = telemetry['ui_state']
    print(f"\n🖥️ UI STATE:")
    print(f"   Game messages: {len(ui.get('game_messages', []))}")
    if ui.get('game_messages'):
        print(f"   Last message: {ui['game_messages'][-1]}")
    print(f"   Media playing: {len(ui.get('media_playing', []))}")
    print(f"   Event states: {len(ui.get('event_states', {}))}")

    # Ship status
    ship_status = telemetry['ship_status']
    print(f"\n🚢 SHIP STATUS:")
    print(f"   Hull: {ship_status.get('hull', 'Unknown')}%")
    print(f"   Shields: {ship_status.get('shields', 'Unknown')}%")
    print(f"   Alert level: {ship_status.get('alert_level', 'Unknown')}")

def on_telemetry_event(event):
    """Handle telemetry events."""
    category = event.get('category', 'unknown')
    event_type = event.get('type', 'unknown')

    # Log significant events
    if category in ['mission', 'ship_internal', 'advanced', 'combat', 'multiplayer', 'ui']:
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color code by category
        colors = {
            'mission': '🎯',
            'ship_internal': '🏢',
            'advanced': '⚙️',
            'combat': '💣',
            'multiplayer': '👥',
            'ui': '🖥️'
        }

        icon = colors.get(category, '📦')
        print(f"[{timestamp}] {icon} {category.upper()}: {event_type}")

        # Show data preview for interesting packets
        if event_type in ['mission_briefing', 'player_objectives', 'ordnance',
                         'factions', 'personnel', 'game_message', 'broadcast']:
            data = event.get('data')
            if data:
                print(f"    → {str(data)[:150]}...")

def main():
    """Main test function."""
    global websocket_client

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 80)
    print("🚀 ENHANCED TELEMETRY CAPTURE TEST")
    print("=" * 80)
    print("This test captures ALL available telemetry including:")
    print("  - Mission briefings and objectives")
    print("  - Ship internal systems and personnel")
    print("  - Advanced systems and factions")
    print("  - Enhanced combat data")
    print("  - Multiplayer/crew coordination")
    print("  - UI and game messages")
    print("\nPress Ctrl+C to stop and see summary.")
    print("=" * 80)

    # Create WebSocket client
    websocket_client = BrowserMimicWebSocket(host=os.getenv("GAME_HOST", "localhost"))

    # Register callback
    websocket_client.add_callback(on_telemetry_event)

    # Connect
    print("\n🔌 Connecting to game server...")
    if websocket_client.connect(screen_name="TelemetryTest"):
        print("✅ Connected successfully!")
        print("\n📡 Listening for telemetry...")
        print("-" * 80)

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("❌ Connection failed!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())