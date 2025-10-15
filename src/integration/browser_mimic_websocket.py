#!/usr/bin/env python3
"""
Browser-Mimicking WebSocket Client for Starship Horizons
Exactly replicates browser communication protocol to receive ship telemetry.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
import websocket

from .smart_filters import MissileTracker, ContactSignificanceFilter, EventPrioritizer
from .station_handlers import StationHandlers, CrewPerformanceTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrowserMimicWebSocket:
    """WebSocket client that mimics browser behavior exactly."""

    def __init__(self, host: str = "66.68.47.235", port: int = 1865):
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}/"  # Note the trailing slash like browser

        self.ws = None
        self.connected = False
        self.session_id = None
        self.vessel_id = None
        self.guid = str(uuid.uuid4())

        # Callbacks
        self._callbacks = []

        # Packet tracking
        self.packets_accepted = []
        self.last_packets = {}
        self.packet_counts = {}

        # Ship data - ENHANCED for comprehensive telemetry
        self.vessel_data = {
            # Core ship status
            'hull_percentage': 100,
            'shield_percentage': 100,  # Overall shield strength
            'shield_frequency': 0,
            'shield_harmonics': 0,

            # Shield quadrants/sections
            'shield_sections': {
                'forward': 100,
                'aft': 100,
                'port': 100,
                'starboard': 100,
                'dorsal': 100,
                'ventral': 100
            },
            'shield_distribution': {
                'forward': 25,     # Percentage of power to this quadrant
                'aft': 25,
                'port': 25,
                'starboard': 25
            },

            # Damage tracking
            'damage_events': [],
            'cumulative_damage': 0,
            'last_damage_time': None,
            'damage_sources': {},

            # System status (percentage health)
            'systems': {
                'engines': 100,
                'weapons': 100,
                'shields': 100,
                'sensors': 100,
                'life_support': 100,
                'warp_drive': 100,
                'impulse_drive': 100,
                'communications': 100,
                'transporters': 100,
                'tractor_beam': 100
            },

            # Power distribution
            'power': {
                'total_available': 100,
                'engines': 0,
                'weapons': 0,
                'shields': 0,
                'auxiliary': 0,
                'life_support': 0
            },

            # Engineering
            'engineering': {
                'reactor_output': 100,
                'reactor_efficiency': 100,
                'coolant_level': 100,
                'coolant_pressure': 0,
                'repair_teams': [],
                'damage_control_teams': []
            },

            # Navigation
            'navigation': {
                'position': {'x': 0, 'y': 0, 'z': 0},
                'heading': 0,
                'pitch': 0,
                'roll': 0,
                'speed': 0,
                'warp_speed': 0,
                'impulse_speed': 0
            },

            # Combat readiness
            'combat': {
                'weapons_armed': False,
                'weapons_locked': False,
                'current_target': None,
                'torpedo_count': {},
                'phaser_charge': 100
            }
        }
        self.last_vessel_update = None

        # Mission and strategic data
        self.mission_data = {
            'current_mission': None,
            'mission_briefing': {},
            'mission_summary': {},
            'objectives': [],
            'player_objectives': [],
            'gm_objectives': [],
            'encounters': [],
            'scenario': {},
            'npcs': {}
        }

        # Ship internal data
        self.ship_internals = {
            'decks': {},
            'current_location': None,
            'location_details': {},
            'personnel': [],
            'cameras': {},
            'devices': {},
            'device_status': {}
        }

        # Advanced systems
        self.advanced_systems = {
            'components': {},
            'component_parts': {},
            'component_properties': {},
            'models': {},
            'factions': {},
            'planetary_systems': {},
            'map_data': {},
            'pre_flight': {}
        }

        # Enhanced combat data
        self.combat_enhanced = {
            'ordnance': {},
            'ordnance_selected': None,
            'projectiles': [],
            'drones': {},
            'drone_targets': {}
        }

        # Multiplayer/Crew
        self.multiplayer_data = {
            'cast_mode': False,
            'cast_host': None,
            'broadcasts': [],
            'channels': {},
            'channel_messages': [],
            'contacts': [],
            'player_vessels': {},
            'callsigns': {}
        }

        # UI/Media state
        self.ui_state = {
            'html_media': {},
            'media_playing': [],
            'controllers': {},
            'event_states': {},
            'game_messages': []
        }

        # Smart filtering components
        self.missile_tracker = MissileTracker()
        self.contact_filter = ContactSignificanceFilter()
        self.event_prioritizer = EventPrioritizer()

        # Station handlers and performance tracking
        self.station_handlers = StationHandlers()
        self.crew_tracker = CrewPerformanceTracker()

    def connect(self, screen_name: str = "Observer", is_main_viewer: bool = True) -> bool:
        """
        Connect to WebSocket mimicking browser.

        Args:
            screen_name: Screen/station name
            is_main_viewer: Whether this is a main viewer
        """
        try:
            logger.info(f"Connecting to {self.ws_url} as {screen_name}")

            def on_message(ws, message):
                self._handle_message(message)

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                self.connected = False

            def on_close(ws, close_code, close_msg):
                logger.info(f"WebSocket closed: {close_code}")
                self.connected = False

            def on_open(ws):
                logger.info("WebSocket connected - mimicking browser protocol")
                self.connected = True

                # Immediately identify like the browser does
                self._send_identification(screen_name, is_main_viewer)

                # Register for packets after short delay (like browser)
                threading.Timer(0.1, self._register_packets).start()

            # Create WebSocket with browser-like headers
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
                header={
                    "Origin": f"http://{self.host}:1864",
                    "User-Agent": "Mozilla/5.0"
                }
            )

            # Run in thread
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            self.ws_thread.start()

            # Wait for connection
            for i in range(10):
                if self.connected:
                    logger.info("✅ Successfully connected and identified")
                    return True
                time.sleep(0.5)

            logger.warning("Connection timeout")
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _send_identification(self, screen_name: str, is_main_viewer: bool):
        """Send identification exactly like browser."""
        # Match the browser's Identify() function format
        identify_data = {
            "ScreenName": screen_name,
            "Location": "Bridge",
            "ServerID": self.session_id or "",
            "Guid": self.guid,
            "IsMainViewer": is_main_viewer,
            "UserInfo": {
                "Name": "AI Crew Member",
                "CallSign": "AI-CREW"
            }
        }

        # Send as JSON string in Value field
        self._send("IDENTIFY", json.dumps(identify_data))
        logger.info(f"Sent IDENTIFY as {screen_name}")

    def _register_packets(self):
        """Register for packet types like browser's SendPacketTypes()."""
        # Comprehensive packet types for all station telemetry
        packet_types = [
            # Core vessel and system data
            "VESSEL",           # Main vessel data
            "VESSEL-ID",        # Vessel identification
            "VESSEL-NO",        # Vessel number assignment
            "VESSEL-VALUES",    # Vessel values/variables
            "VESSEL-COMPONENTS", # Individual ship components (reactors, engines, etc.)
            "VESSEL-COMPONENTS-REMOVED", # Component destruction events
            "VESSEL-CLASSES",   # Ship class definitions
            "VESSEL-OCCUPANTS", # Crew locations on ship
            "VESSEL-WAYPOINTS", # Navigation waypoint data
            "WEAPON-GROUPS",    # Weapon grouping configuration
            "DAMAGE",           # Damage reports
            "DAMAGE-TEAMS",     # Damage control teams
            "SHIELDS",          # Shield status
            "HULL",             # Hull status
            "ALERT",            # Alert status
            "STATUS",           # General status
            "BC",               # Batch commands
            "BATCH",            # Batch updates containing vessel data
            "BATCH-COMPLETE",   # Batch update completion

            # Tactical station packets
            "WEAPONS",          # Weapons status
            "WEAPON-FIRE",      # Weapon discharge events
            "TARGET-TACTICAL",  # Tactical targeting
            "BEAM-FREQUENCY",   # Beam weapon settings

            # Science station packets
            "CONTACTS",         # Nearby contacts
            "SCAN",             # Scan operations
            "SCAN-RESULT",      # Scan results
            "TARGET-SCIENCE",   # Science targeting
            "PROBE",            # Probe launches

            # Operations/Communications packets
            "HAIL",             # Hailing attempts
            "HAIL-RESPONSE",    # Hail responses
            "COMM",             # Communications
            "TRANSPORTER",      # Transporter operations
            "DOCKING",          # Docking operations
            "CARGO",            # Cargo transfers
            "CARGO-BAY",        # Cargo bay status
            "CARGO-TRANSFER",   # Cargo movement events
            "CREW",             # Crew management

            # Shuttle and fighter operations
            "SHUTTLES",         # Shuttle bay status
            "SHUTTLE-LAUNCH",   # Shuttle launch events
            "SHUTTLE-DOCK",     # Shuttle docking events
            "FIGHTER-BAY",      # Fighter squadron status
            "FIGHTER-LAUNCH",   # Fighter launch orders
            "HANGAR",           # Hangar bay operations

            # Helm/Navigation packets
            "NAVIGATION",       # Navigation data
            "HELM",             # Helm controls
            "THROTTLE",         # Speed changes
            "COURSE",           # Course changes
            "TARGET-FLIGHT",    # Flight targeting
            "AUTOPILOT",        # Autopilot status

            # Engineering packets
            "POWER",            # Power distribution
            "REACTOR",          # Reactor status
            "SYSTEMS",          # System status
            "REPAIR",           # Repair operations
            "COOLANT",          # Coolant distribution

            # Mission/GM packets
            "MISSION",          # Mission status
            "OBJECTIVES",       # Mission objectives
            "GM-EVENT",         # Game Master events
            "SCENARIO",         # Scenario changes
            "NPC",              # NPC actions
            "SPAWN",            # Entity spawning
            "MISSION-BRIEFING", # Pre-mission briefings
            "MISSION-SUMMARY",  # Post-mission summaries
            "MISSIONS",         # Available missions list
            "PLAYER-OBJECTIVES",# Player-specific objectives
            "GM-OBJECTIVES",    # Game Master objectives
            "ENCOUNTERS",       # Random encounters
            "ENCOUNTERS-UPDATE",# Encounter status changes

            # Ship Internal Systems
            "DECKS",            # Deck layouts and maps
            "LOCATION-CURRENT", # Current location on ship
            "LOCATION-DETAIL",  # Detailed location info
            "PERSONNEL",        # Crew positions and assignments
            "CAMERAS",          # Internal camera feeds
            "DEVICES",          # Ship devices and equipment
            "DEVICE-STATUS",    # Device operational status
            "DEVICE-DELETE",    # Device removal events

            # Advanced Systems
            "COMPONENTS",       # Ship component details
            "COMPONENT-PARTS",  # Component sub-parts
            "COMPONENT-PROPERTY",# Component properties
            "MODELS",           # 3D model data
            "FACTIONS",         # Faction standings and relations
            "PLANETARY-SYSTEM-DETAIL", # Planetary system data
            "MAP",              # Map and navigation data
            "PRE-FLIGHT",       # Pre-flight check data

            # Combat Enhancement
            "ORDNANCE",         # Available ordnance
            "ORDNANCE-SELECTED",# Currently selected ordnance
            "PROJECTILES",      # Active projectiles in space
            "DRONES",           # Drone status
            "DRONE-TARGETS",    # Drone targeting data

            # Multiplayer/Crew Coordination
            "CAST",             # Streaming/spectating mode
            "CAST-HOST",        # Host information
            "BROADCAST",        # Broadcast messages
            "CHANNELS",         # Available communication channels
            "CHANNEL-UPDATE",   # Channel status updates
            "CHANNEL-MESSAGE",  # Channel messages
            "CHANNEL-CLOSE",    # Channel closure events
            "CHANNEL-TOPIC",    # Channel topics
            "CONTACT-REQUEST",  # Contact/friend requests
            "CONTACT-REMOVE",   # Contact removal
            "CALLSIGN",         # Callsign assignments
            "PLAYER-VESSELS",   # Player vessel assignments

            # UI and Media
            "HTMLMEDIA",        # HTML media elements
            "PLAY-MEDIA",       # Media playback events
            "CONTROLLERS",      # Input controller data
            "EVENTSTATE",       # Event state management
            "EVENT-TOGGLE",     # Event toggling
            "GAME-MESSAGE",     # Game-wide messages

            # Additional Telemetry
            "AUTO-PILOT",       # Autopilot engagement details
            "CONTACTS-CLEAR",   # Contact list clearing
            "OBJECT-REMOVE",    # Object removal from space
            "OBJECT-PROPERTY",  # Object property updates
            "CARGO-REPAIR",     # Cargo repair operations

            # Metadata packets
            "VARIABLES",        # Game variables
            "SYSTEM-LOG",       # System events
            "EVENT",            # Generic events
            "MESSAGE",          # System messages
            "PLAYERS",          # Player/crew info
            "EVENTS",           # Event list
            "GM-OBJECTS",       # Game Master objects
            "GM-OID",           # Game Master object IDs

            # Session and game state packets
            "SESSION",          # Game session information
            "SESSION-CLEAR",    # Session reset
            "SETTINGS",         # Game settings
            "SVRS",             # Server information
            "USER-INFO",        # User/player information
            "ROLES",            # Station role assignments
            "SCREENS",          # Available screen/console list

            # Control and heartbeat packets
            "PING",             # Heartbeat request
            "PONG",             # Heartbeat response
            "IDENTIFY",         # Client identification response
            "RESET",            # Game reset
            "CLEAR",            # Clear data
            "CNTL"              # Controller data (alternate name)
        ]

        for packet_type in packet_types:
            self._accept_packet(packet_type)
            time.sleep(0.05)  # Small delay between registrations

        logger.info(f"Registered for {len(packet_types)} packet types")

    def _accept_packet(self, packet_type: str):
        """Accept a packet type (browser's AcceptPacket function)."""
        if packet_type not in self.packets_accepted:
            self.packets_accepted.append(packet_type)
            self._send("ACCEPT-PACKET", packet_type)
            logger.debug(f"Accepted packet type: {packet_type}")

    def _send(self, cmd: str, value: Any = ""):
        """Send message in exact browser format."""
        if not self.ws or not self.connected:
            return

        try:
            # Match browser's Send() format exactly
            data = json.dumps({"Cmd": cmd, "Value": value})
            self.ws.send(data)
            logger.debug(f"Sent: Cmd={cmd}, Value={str(value)[:50]}...")
        except Exception as e:
            logger.error(f"Send failed: {e}")

    def _handle_message(self, message):
        """Handle incoming message like browser's ProcessCMD."""
        try:
            data = json.loads(message)

            # Extract command and value (handle both cases)
            cmd = data.get("Cmd") or data.get("cmd")
            value = data.get("Value") or data.get("value")

            if not cmd:
                return

            # Track packet
            self.packet_counts[cmd] = self.packet_counts.get(cmd, 0) + 1

            # Store last packet
            self.last_packets[cmd] = value

            # Log first occurrence of each packet type
            if self.packet_counts[cmd] == 1:
                logger.info(f"📦 New packet type received: {cmd}")
                if value:
                    logger.info(f"   Sample: {str(value)[:200]}...")

            # Process specific commands
            if cmd == "SESSION":
                self._handle_session(value)
            elif cmd == "VESSEL-ID":
                self._handle_vessel_id(value)
            elif cmd == "VESSEL":
                self._handle_vessel(value)
            elif cmd == "VESSEL-VALUES":
                self._handle_vessel_values(value)
            elif cmd == "DAMAGE":
                self._handle_damage(value)
            elif cmd == "VARIABLES":
                self._handle_variables(value)
            elif cmd == "SHIELDS" or cmd == "HULL":
                self._handle_system_status(cmd, value)
            elif cmd == "ALERT":
                self._handle_alert(value)
            elif cmd == "PING":
                # Respond to ping
                self._send("PONG")
            elif cmd == "BC" or cmd == "BATCH":  # Batch command/updates
                self._handle_batch(value)
            elif cmd in ["WEAPONS", "TARGET-TACTICAL", "TARGET-FLIGHT", "DAMAGE-TEAMS", "WEAPON-GROUPS"]:
                self._handle_unhandled_packet(cmd, value)
            elif cmd == "VESSEL-COMPONENTS":
                self._handle_vessel_components(value)
            elif cmd == "VESSEL-WAYPOINTS":
                self._handle_vessel_waypoints(value)
            elif cmd == "VESSEL-OCCUPANTS":
                self._handle_vessel_occupants(value)
            elif cmd in ["SETTINGS", "SVRS", "USER-INFO", "ROLES", "SCREENS"]:
                self._handle_game_metadata(cmd, value)
            elif cmd in ["RESET", "CLEAR", "SESSION-CLEAR"]:
                self._handle_control_packet(cmd, value)
            elif cmd == "CONTACTS":
                self._handle_contacts(value)
            elif cmd == "CONTACT-REMOVE":
                self._handle_contact_remove(value)
            elif cmd == "OBJECT-REMOVE":
                self._handle_object_remove(value)
            elif cmd == "TARGET-SCIENCE":
                self._handle_target_science(value)
            elif cmd == "PLAYERS":
                self._handle_players(value)
            elif cmd == "SYSTEM-LOG":
                self._handle_system_log(value)
            elif cmd == "MISSION":
                self._handle_mission(value)
            elif cmd in ["SHIELDS", "HULL", "POWER", "REACTOR", "LIFE-SUPPORT", "ENGINES", "PROPULSION", "POWER-GRID", "SYSTEMS", "WARP", "IMPULSE", "DAMAGE-CONTROL", "REPAIR-TEAMS", "COOLANT"]:
                self._handle_engineering_systems(cmd, value)
            # Operations/Communications station handlers
            elif cmd == "HAIL":
                event = self.station_handlers.handle_hail(value)
                if event:
                    self._emit_event(event)
                    self.crew_tracker.track_action("operations", "hail", time.time())
            elif cmd == "HAIL-RESPONSE":
                event = self.station_handlers.handle_hail_response(value)
                if event:
                    self._emit_event(event)
            elif cmd in ["COMM", "COMM-MESSAGE"]:
                event = self.station_handlers.handle_communications(cmd, value)
                if event:
                    self._emit_event(event)
            elif cmd == "TRANSPORTER":
                event = self.station_handlers.handle_transporter(value)
                if event:
                    self._emit_event(event)
                    self.crew_tracker.track_action("operations", "transporter", time.time())
            elif cmd == "DOCKING":
                event = self.station_handlers.handle_docking(value)
                if event:
                    self._emit_event(event)
            elif cmd in ["CARGO", "CARGO-BAY", "CARGO-TRANSFER"]:
                event = self.station_handlers.handle_cargo(cmd, value)
                if event:
                    self._emit_event(event)
                    if cmd == "CARGO-TRANSFER":
                        self.crew_tracker.track_action("operations", "cargo_transfer", time.time())
            elif cmd in ["SHUTTLES", "SHUTTLE-LAUNCH", "SHUTTLE-DOCK"]:
                event = self.station_handlers.handle_shuttles(cmd, value)
                if event:
                    self._emit_event(event)
                    self.crew_tracker.track_action("operations", "shuttle", time.time())
            elif cmd in ["FIGHTER-BAY", "FIGHTER-LAUNCH", "HANGAR"]:
                event = self.station_handlers.handle_fighters(cmd, value)
                if event:
                    self._emit_event(event)
                    if cmd == "FIGHTER-LAUNCH":
                        self.crew_tracker.track_action("tactical", "fighter_launch", time.time())
            # Helm station handlers
            elif cmd in ["NAVIGATION", "HELM", "THROTTLE", "COURSE", "AUTOPILOT"]:
                event = self.station_handlers.handle_helm_data(cmd, value)
                if event:
                    self._emit_event(event)
                    self.crew_tracker.track_action("helm", cmd.lower(), time.time())
            # Science station handlers
            elif cmd in ["SCAN", "SCAN-RESULT", "PROBE"]:
                event = self.station_handlers.handle_science_operations(cmd, value)
                if event:
                    self._emit_event(event)
                    self.crew_tracker.track_action("science", cmd.lower(), time.time())
            # Mission and strategic data handlers
            elif cmd == "MISSION-BRIEFING":
                self.mission_data['mission_briefing'] = value
                self._emit_event({"type": "mission_briefing", "category": "mission", "data": value})
                logger.info(f"📋 Mission Briefing received: {str(value)[:100]}...")
            elif cmd == "MISSION-SUMMARY":
                self.mission_data['mission_summary'] = value
                self._emit_event({"type": "mission_summary", "category": "mission", "data": value})
            elif cmd == "MISSIONS":
                self.mission_data['available_missions'] = value
                self._emit_event({"type": "missions_list", "category": "mission", "data": value})
            elif cmd == "PLAYER-OBJECTIVES":
                self.mission_data['player_objectives'] = value
                self._emit_event({"type": "player_objectives", "category": "mission", "data": value})
                logger.info(f"🎯 Player Objectives: {value}")
            elif cmd == "GM-OBJECTIVES":
                self.mission_data['gm_objectives'] = value
                self._emit_event({"type": "gm_objectives", "category": "mission", "data": value})
            elif cmd == "ENCOUNTERS":
                self.mission_data['encounters'] = value
                self._emit_event({"type": "encounters", "category": "mission", "data": value})
            elif cmd == "ENCOUNTERS-UPDATE":
                self.mission_data['encounters'] = value
                self._emit_event({"type": "encounters_update", "category": "mission", "data": value})

            # Ship internal systems
            elif cmd == "DECKS":
                self.ship_internals['decks'] = value
                self._emit_event({"type": "deck_layout", "category": "ship_internal", "data": value})
                logger.info(f"🏢 Deck layout received: {len(value) if isinstance(value, list) else 'data'}")
            elif cmd == "LOCATION-CURRENT":
                self.ship_internals['current_location'] = value
                self._emit_event({"type": "location_current", "category": "ship_internal", "data": value})
            elif cmd == "LOCATION-DETAIL":
                self.ship_internals['location_details'] = value
                self._emit_event({"type": "location_detail", "category": "ship_internal", "data": value})
            elif cmd == "PERSONNEL":
                self.ship_internals['personnel'] = value
                self._emit_event({"type": "personnel", "category": "ship_internal", "data": value})
                logger.info(f"👥 Personnel data: {len(value) if isinstance(value, list) else 'data'} crew")
            elif cmd == "CAMERAS":
                self.ship_internals['cameras'] = value
                self._emit_event({"type": "cameras", "category": "ship_internal", "data": value})
            elif cmd == "DEVICES":
                self.ship_internals['devices'] = value
                self._emit_event({"type": "devices", "category": "ship_internal", "data": value})
            elif cmd == "DEVICE-STATUS":
                self.ship_internals['device_status'] = value
                self._emit_event({"type": "device_status", "category": "ship_internal", "data": value})
            elif cmd == "DEVICE-DELETE":
                self._emit_event({"type": "device_delete", "category": "ship_internal", "data": value})

            # Advanced systems
            elif cmd == "COMPONENTS":
                self.advanced_systems['components'] = value
                self._emit_event({"type": "components", "category": "advanced", "data": value})
                logger.info(f"⚙️ Components data received")
            elif cmd == "COMPONENT-PARTS":
                self.advanced_systems['component_parts'] = value
                self._emit_event({"type": "component_parts", "category": "advanced", "data": value})
            elif cmd == "COMPONENT-PROPERTY":
                self.advanced_systems['component_properties'] = value
                self._emit_event({"type": "component_property", "category": "advanced", "data": value})
            elif cmd == "MODELS":
                self.advanced_systems['models'] = value
                self._emit_event({"type": "models", "category": "advanced", "data": value})
            elif cmd == "FACTIONS":
                self.advanced_systems['factions'] = value
                self._emit_event({"type": "factions", "category": "advanced", "data": value})
                logger.info(f"🏛️ Factions data: {value}")
            elif cmd == "PLANETARY-SYSTEM-DETAIL":
                self.advanced_systems['planetary_systems'] = value
                self._emit_event({"type": "planetary_system", "category": "advanced", "data": value})
            elif cmd == "MAP":
                self.advanced_systems['map_data'] = value
                self._emit_event({"type": "map", "category": "advanced", "data": value})
            elif cmd == "PRE-FLIGHT":
                self.advanced_systems['pre_flight'] = value
                self._emit_event({"type": "pre_flight", "category": "advanced", "data": value})

            # Enhanced combat
            elif cmd == "ORDNANCE":
                self.combat_enhanced['ordnance'] = value
                self._emit_event({"type": "ordnance", "category": "combat", "data": value})
                logger.info(f"💣 Ordnance inventory: {value}")
            elif cmd == "ORDNANCE-SELECTED":
                self.combat_enhanced['ordnance_selected'] = value
                self._emit_event({"type": "ordnance_selected", "category": "combat", "data": value})
            elif cmd == "PROJECTILES":
                self.combat_enhanced['projectiles'] = value
                self._emit_event({"type": "projectiles", "category": "combat", "data": value})
                logger.info(f"🚀 Active projectiles: {len(value) if isinstance(value, list) else value}")
            elif cmd == "DRONES":
                self.combat_enhanced['drones'] = value
                self._emit_event({"type": "drones", "category": "combat", "data": value})
            elif cmd == "DRONE-TARGETS":
                self.combat_enhanced['drone_targets'] = value
                self._emit_event({"type": "drone_targets", "category": "combat", "data": value})

            # Multiplayer/Crew coordination
            elif cmd == "CAST":
                self.multiplayer_data['cast_mode'] = value
                self._emit_event({"type": "cast", "category": "multiplayer", "data": value})
            elif cmd == "CAST-HOST":
                self.multiplayer_data['cast_host'] = value
                self._emit_event({"type": "cast_host", "category": "multiplayer", "data": value})
            elif cmd == "BROADCAST":
                self.multiplayer_data['broadcasts'].append(value)
                self._emit_event({"type": "broadcast", "category": "multiplayer", "data": value})
                logger.info(f"📡 Broadcast: {value}")
            elif cmd == "CHANNELS":
                self.multiplayer_data['channels'] = value
                self._emit_event({"type": "channels", "category": "multiplayer", "data": value})
            elif cmd == "CHANNEL-MESSAGE":
                self.multiplayer_data['channel_messages'].append(value)
                self._emit_event({"type": "channel_message", "category": "multiplayer", "data": value})
            elif cmd == "CHANNEL-UPDATE":
                self._emit_event({"type": "channel_update", "category": "multiplayer", "data": value})
            elif cmd == "CHANNEL-CLOSE":
                self._emit_event({"type": "channel_close", "category": "multiplayer", "data": value})
            elif cmd == "CHANNEL-TOPIC":
                self._emit_event({"type": "channel_topic", "category": "multiplayer", "data": value})
            elif cmd == "CONTACT-REQUEST":
                self.multiplayer_data['contacts'].append(value)
                self._emit_event({"type": "contact_request", "category": "multiplayer", "data": value})
            elif cmd == "CALLSIGN":
                self.multiplayer_data['callsigns'] = value
                self._emit_event({"type": "callsign", "category": "multiplayer", "data": value})
            elif cmd == "PLAYER-VESSELS":
                self.multiplayer_data['player_vessels'] = value
                self._emit_event({"type": "player_vessels", "category": "multiplayer", "data": value})
                logger.info(f"🚢 Player vessels: {value}")

            # UI and Media
            elif cmd == "HTMLMEDIA":
                self.ui_state['html_media'] = value
                self._emit_event({"type": "html_media", "category": "ui", "data": value})
            elif cmd == "PLAY-MEDIA":
                self.ui_state['media_playing'].append(value)
                self._emit_event({"type": "play_media", "category": "ui", "data": value})
            elif cmd == "CONTROLLERS":
                self.ui_state['controllers'] = value
                self._emit_event({"type": "controllers", "category": "ui", "data": value})
            elif cmd == "EVENTSTATE":
                self.ui_state['event_states'] = value
                self._emit_event({"type": "event_state", "category": "ui", "data": value})
            elif cmd == "EVENT-TOGGLE":
                self._emit_event({"type": "event_toggle", "category": "ui", "data": value})
            elif cmd == "GAME-MESSAGE":
                self.ui_state['game_messages'].append(value)
                self._emit_event({"type": "game_message", "category": "ui", "data": value})
                logger.info(f"📢 Game message: {value}")

            # Additional telemetry
            elif cmd == "AUTO-PILOT":
                self._emit_event({"type": "auto_pilot", "category": "navigation", "data": value})
            elif cmd == "CONTACTS-CLEAR":
                self._emit_event({"type": "contacts_clear", "category": "sensors", "data": value})
            elif cmd == "OBJECT-PROPERTY":
                self._emit_event({"type": "object_property", "category": "world", "data": value})
            elif cmd == "CARGO-REPAIR":
                self._emit_event({"type": "cargo_repair", "category": "operations", "data": value})
            elif cmd == "EVENTS":
                self._emit_event({"type": "events_list", "category": "mission", "data": value})
            elif cmd == "GM-OBJECTS":
                self._emit_event({"type": "gm_objects", "category": "gm", "data": value})
            elif cmd == "GM-OID":
                self._emit_event({"type": "gm_object_id", "category": "gm", "data": value})

            # Game Master/Mission handlers (original)
            elif cmd in ["GM-EVENT", "SCENARIO", "NPC", "SPAWN", "OBJECTIVES"]:
                event = self.station_handlers.handle_gm_events(cmd, value)
                if event:
                    self._emit_event(event)
            # Crew coordination handlers
            elif cmd in ["CREW", "CREW-ACTION", "STATION-REPORT", "CAPTAIN-ORDER"]:
                event = self.station_handlers.handle_crew_coordination(cmd, value)
                if event:
                    self._emit_event(event)
                    # Track coordination if it's a captain order
                    if cmd == "CAPTAIN-ORDER":
                        self.crew_tracker.last_order_time = time.time()
            else:
                # Log unhandled packets for debugging
                logger.debug(f"🤷 Unhandled packet: {cmd} = {str(value)[:100]}...")

        except Exception as e:
            logger.debug(f"Message parse error: {e}")

    def _handle_session(self, value):
        """Handle SESSION packet."""
        if value:
            self.session_id = value.get("ID")
            state = value.get("State")
            mode = value.get("Mode")

            logger.info(f"📍 Session: ID={self.session_id}, State={state}, Mode={mode}")

            self._emit_event({
                "type": "session_info",
                "category": "session",
                "data": value
            })

    def _handle_vessel_id(self, value):
        """Handle VESSEL-ID packet."""
        if isinstance(value, (int, str)):
            self.vessel_id = value
            logger.info(f"🚢 Vessel ID set: {self.vessel_id}")

            # Now request all vessel data
            self._send("GET", "VESSEL")
            self._send("GET", "VESSEL-VALUES")
            self._send("GET", "SHIELDS")
            self._send("GET", "HULL")
            self._send("GET", "STATUS")
            self._send("GET", "SYSTEMS")
            self._send("GET", "POWER")

    def _handle_vessel_values(self, value):
        """Handle VESSEL-VALUES packet which may contain hull/shield data."""
        if value:
            logger.info(f"🚢 Vessel Values: {value}")

            # Check for hull/shield data in various formats
            if isinstance(value, dict):
                for key, val in value.items():
                    if 'hull' in key.lower():
                        self.vessel_data['hull_percentage'] = val
                        logger.info(f"   Hull: {val}%")
                    elif 'shield' in key.lower():
                        self.vessel_data['shield_percentage'] = val
                        logger.info(f"   Shields: {val}%")
                    elif 'integrity' in key.lower():
                        self.vessel_data['hull_percentage'] = val
                        logger.info(f"   Integrity: {val}%")

                    # Store all values
                    self.vessel_data[f"vessel_value_{key}"] = val

            self._emit_event({
                "type": "vessel_values",
                "category": "telemetry",
                "data": value
            })

    def _handle_vessel(self, value):
        """Handle VESSEL packet - ENHANCED for comprehensive telemetry."""
        if not value:
            return

        # Check if this is our vessel
        vessel_id = value.get("ID")
        if self.vessel_id and vessel_id != self.vessel_id:
            return  # Not our vessel

        # Track changes
        changes = []

        # Enhanced telemetry fields to monitor
        telemetry_fields = [
            # Hull and Shields
            ("Shields", "shield_percentage"),
            ("Hull", "hull_percentage"),
            ("ShieldStrength", "shield_percentage"),
            ("HullIntegrity", "hull_percentage"),
            ("Integrity", "hull_percentage"),

            # Power and Energy
            ("Energy", "energy"),
            ("Power", "power"),
            ("SystemPower", "system_power"),
            ("ReactorOutput", "reactor_output"),

            # Navigation
            ("Speed", "speed"),
            ("Heading", "heading"),
            ("Position", "position"),
            ("Orientation", "orientation"),
            ("WarpSpeed", "warp_speed"),
            ("ImpulseSpeed", "impulse_speed"),
            ("Velocity", "velocity"),

            # Combat
            ("Alert", "alert"),
            ("WeaponsArmed", "weapons_armed"),
            ("TargetLock", "target_lock"),
            ("RedAlert", "red_alert"),

            # Systems
            ("LifeSupport", "life_support"),
            ("Sensors", "sensors"),
            ("Communications", "communications")
        ]

        for field, name in telemetry_fields:
            if field in value:
                new_value = value[field]

                # Special handling for complex fields
                if field == "Position" and isinstance(new_value, dict):
                    # Update navigation position
                    self.vessel_data['navigation']['position'] = {
                        'x': new_value.get('X', 0),
                        'y': new_value.get('Y', 0),
                        'z': new_value.get('Z', 0)
                    }
                    changes.append(("position", None, new_value))

                elif field == "Orientation" and isinstance(new_value, dict):
                    # Calculate heading, pitch, roll from quaternion
                    self.vessel_data['navigation']['orientation'] = new_value
                    changes.append(("orientation", None, new_value))

                elif field in ["Shields", "ShieldStrength"] and isinstance(new_value, dict):
                    # Shield quadrant data
                    for quadrant, strength in new_value.items():
                        quad_key = quadrant.lower()
                        if quad_key in self.vessel_data['shield_sections']:
                            old_val = self.vessel_data['shield_sections'][quad_key]
                            if old_val != strength:
                                self.vessel_data['shield_sections'][quad_key] = strength
                                changes.append((f"shield_{quad_key}", old_val, strength))

                elif field == "Integrity" and isinstance(new_value, (int, float)):
                    # Some packets use 'Integrity' for hull percentage
                    # Convert from raw value to percentage if needed
                    if new_value > 100:  # Raw hull points
                        max_hull = value.get('MaxIntegrity', new_value)
                        new_value = (new_value / max_hull) * 100 if max_hull > 0 else 100

                    old_value = self.vessel_data.get('hull_percentage')
                    if old_value != new_value:
                        self.vessel_data['hull_percentage'] = new_value
                        changes.append(("hull_percentage", old_value, new_value))

                else:
                    # Standard field update
                    old_value = self.vessel_data.get(name)
                    if old_value != new_value:
                        self.vessel_data[name] = new_value
                        changes.append((name, old_value, new_value))

                        # Update specific subsystems
                        if name == 'weapons_armed':
                            self.vessel_data['combat']['weapons_armed'] = new_value
                        elif name == 'target_lock':
                            self.vessel_data['combat']['current_target'] = new_value
                        elif name in ['speed', 'warp_speed', 'impulse_speed', 'heading']:
                            self.vessel_data['navigation'][name] = new_value

                        # Emit change event for significant changes
                        if self._is_significant_vessel_change(name, old_value, new_value):
                            self._emit_event({
                                "type": f"{name}_changed",
                                "category": "telemetry",
                                "data": {
                                    "system": name,
                                    "old_value": old_value,
                                    "new_value": new_value,
                                    "vessel_id": vessel_id
                                },
                                "priority": self._get_change_priority(name, new_value)
                            })

        # Log significant changes
        if changes:
            logger.info(f"🚀 Vessel telemetry updated:")
            for name, old_val, new_val in changes:
                logger.info(f"   {name}: {old_val} → {new_val}")

        # Store full vessel data
        self.vessel_data.update(value)
        self.last_vessel_update = datetime.now()

    def _handle_damage(self, value):
        """Handle DAMAGE packet - ENHANCED for detailed tracking."""
        if value:
            logger.info(f"💥 Damage report: {value}")

            # Parse damage data
            damage_amount = 0
            damage_type = "unknown"
            affected_system = None

            if isinstance(value, dict):
                # Extract damage details
                damage_amount = value.get('Amount', value.get('Damage', 0))
                damage_type = value.get('Type', 'kinetic')
                affected_system = value.get('System', value.get('Location', 'hull'))

                # Update hull/shield percentages if provided
                if 'HullPercent' in value:
                    self.vessel_data['hull_percentage'] = value['HullPercent']
                if 'ShieldPercent' in value:
                    self.vessel_data['shield_percentage'] = value['ShieldPercent']
                if 'Hull' in value:
                    self.vessel_data['hull_percentage'] = value['Hull']
                if 'Shields' in value:
                    # Could be overall percentage or quadrant data
                    if isinstance(value['Shields'], (int, float)):
                        self.vessel_data['shield_percentage'] = value['Shields']
                    elif isinstance(value['Shields'], dict):
                        # Quadrant data in damage packet
                        for quadrant, strength in value['Shields'].items():
                            quad_key = quadrant.lower().replace('-', '_')
                            if quad_key in self.vessel_data['shield_sections']:
                                self.vessel_data['shield_sections'][quad_key] = strength

                # Check for shield quadrant damage
                if 'ShieldQuadrant' in value:
                    quadrant = value['ShieldQuadrant'].lower()
                    if quadrant in self.vessel_data['shield_sections']:
                        # Reduce that quadrant's strength
                        current = self.vessel_data['shield_sections'][quadrant]
                        self.vessel_data['shield_sections'][quadrant] = max(0, current - damage_amount)
                        affected_system = f"shields_{quadrant}"

                # Track damage to specific systems
                if affected_system and affected_system in self.vessel_data['systems']:
                    current_health = self.vessel_data['systems'][affected_system]
                    self.vessel_data['systems'][affected_system] = max(0, current_health - damage_amount)
            elif isinstance(value, (int, float)):
                damage_amount = value

            # Track damage event
            damage_event = {
                'timestamp': datetime.now().isoformat(),
                'amount': damage_amount,
                'type': damage_type,
                'system': affected_system,
                'hull_after': self.vessel_data.get('hull_percentage', 100),
                'shield_after': self.vessel_data.get('shield_percentage', 100)
            }

            self.vessel_data['damage_events'].append(damage_event)
            self.vessel_data['cumulative_damage'] += damage_amount
            self.vessel_data['last_damage_time'] = datetime.now().isoformat()

            # Track damage sources
            if damage_type not in self.vessel_data['damage_sources']:
                self.vessel_data['damage_sources'][damage_type] = 0
            self.vessel_data['damage_sources'][damage_type] += damage_amount

            # Log critical damage
            if self.vessel_data['hull_percentage'] < 30:
                logger.warning(f"⚠️ CRITICAL: Hull at {self.vessel_data['hull_percentage']}%")
            if self.vessel_data['shield_percentage'] < 20:
                logger.warning(f"⚠️ CRITICAL: Shields at {self.vessel_data['shield_percentage']}%")

            self._emit_event({
                "type": "damage_report",
                "category": "damage",
                "data": damage_event,
                "priority": "CRITICAL" if damage_amount > 10 else "HIGH"
            })

    def _handle_variables(self, value):
        """Handle VARIABLES packet - contains ALL game variables and ship status."""
        if value and isinstance(value, dict):
            logger.info(f"📊 Variables update: {len(value)} game variables")

            # Log important variables for visibility
            important_vars = {}
            for key, val in value.items():
                key_lower = key.lower()

                # Store ALL variables in vessel data
                self.vessel_data[f"var_{key}"] = val

                # Log important ship systems
                if any(term in key_lower for term in ["shield", "hull", "power", "energy", "damage", "cargo", "warp", "impulse", "reactor", "life", "gravity"]):
                    logger.info(f"   🔧 {key}: {val}")
                    important_vars[key] = val

            # Emit event for ALL variables
            self._emit_event({
                "type": "variables_update",
                "category": "variables",
                "data": value
            })

            # Emit separate events for critical ship systems
            if important_vars:
                self._emit_event({
                    "type": "ship_systems_update",
                    "category": "systems",
                    "data": important_vars
                })

    def _handle_system_status(self, system: str, value):
        """Handle SHIELDS, HULL, etc packets - ENHANCED with quadrant support."""
        if value:
            logger.info(f"🛡️ {system}: {value}")

            # Parse percentage values if present
            if isinstance(value, dict):
                if system == "SHIELDS":
                    # Overall shield strength
                    if 'Percent' in value:
                        self.vessel_data['shield_percentage'] = value['Percent']
                    if 'Strength' in value:
                        self.vessel_data['shield_percentage'] = value['Strength']
                    if 'Frequency' in value:
                        self.vessel_data['shield_frequency'] = value['Frequency']
                    if 'Harmonics' in value:
                        self.vessel_data['shield_harmonics'] = value['Harmonics']

                    # Shield quadrants/sections
                    quadrant_keys = ['Forward', 'Aft', 'Port', 'Starboard', 'Dorsal', 'Ventral',
                                   'Fore', 'Rear', 'Left', 'Right', 'Top', 'Bottom']

                    for key in value:
                        key_lower = key.lower()
                        # Check for quadrant data
                        if 'forward' in key_lower or 'fore' in key_lower:
                            self.vessel_data['shield_sections']['forward'] = value[key]
                        elif 'aft' in key_lower or 'rear' in key_lower:
                            self.vessel_data['shield_sections']['aft'] = value[key]
                        elif 'port' in key_lower or 'left' in key_lower:
                            self.vessel_data['shield_sections']['port'] = value[key]
                        elif 'starboard' in key_lower or 'right' in key_lower:
                            self.vessel_data['shield_sections']['starboard'] = value[key]
                        elif 'dorsal' in key_lower or 'top' in key_lower:
                            self.vessel_data['shield_sections']['dorsal'] = value[key]
                        elif 'ventral' in key_lower or 'bottom' in key_lower:
                            self.vessel_data['shield_sections']['ventral'] = value[key]

                    # Check for shield sections/quadrants array
                    if 'Sections' in value and isinstance(value['Sections'], (list, dict)):
                        if isinstance(value['Sections'], list):
                            # Array format: [forward, aft, port, starboard]
                            if len(value['Sections']) >= 4:
                                self.vessel_data['shield_sections']['forward'] = value['Sections'][0]
                                self.vessel_data['shield_sections']['aft'] = value['Sections'][1]
                                self.vessel_data['shield_sections']['port'] = value['Sections'][2]
                                self.vessel_data['shield_sections']['starboard'] = value['Sections'][3]
                        else:
                            # Dict format
                            self.vessel_data['shield_sections'].update(value['Sections'])

                    # Shield power distribution
                    if 'Distribution' in value:
                        self.vessel_data['shield_distribution'] = value['Distribution']

                    # Log critical shield sections
                    for section, strength in self.vessel_data['shield_sections'].items():
                        if strength < 30:
                            logger.warning(f"⚠️ CRITICAL: {section} shields at {strength}%")
                elif system == "HULL":
                    if 'Percent' in value:
                        self.vessel_data['hull_percentage'] = value['Percent']
                    if 'Integrity' in value:
                        self.vessel_data['hull_percentage'] = value['Integrity']
            elif isinstance(value, (int, float)):
                # Direct percentage value
                if system == "SHIELDS":
                    self.vessel_data['shield_percentage'] = value
                elif system == "HULL":
                    self.vessel_data['hull_percentage'] = value

            self.vessel_data[system.lower()] = value

            self._emit_event({
                "type": f"{system.lower()}_status",
                "category": "systems",
                "data": value,
                "processed": {
                    'hull_percentage': self.vessel_data.get('hull_percentage'),
                    'shield_percentage': self.vessel_data.get('shield_percentage')
                }
            })

    def _handle_alert(self, value):
        """Handle ALERT packet."""
        if value:
            logger.info(f"🚨 Alert: {value}")

            # Track alert time for response time measurements
            alert_level_names = {1: "Docked", 2: "Green", 3: "Yellow", 4: "Red", 5: "Red Alert"}
            if value in [3, 4, 5]:  # Yellow or Red alert
                self.crew_tracker.last_alert_time = time.time()

            self._emit_event({
                "type": "alert_change",
                "category": "alert",
                "data": value,
                "priority": "CRITICAL" if value >= 4 else "HIGH"
            })

    def _handle_batch(self, value):
        """Handle batch command (BC/BATCH) packet - contains vessel hull/shield data."""
        # BATCH packets can be either a dict with VESSEL key or a list of commands
        if value and isinstance(value, dict):
            # Handle BATCH format: {"VESSEL": {...}, "OTHER_KEY": {...}}
            if "VESSEL" in value:
                vessel_data = value["VESSEL"]
                if isinstance(vessel_data, dict):
                    # Extract hull/shield data from VESSEL
                    self._handle_batch_vessel_data(vessel_data)

            # Process other keys in the batch
            for key, data in value.items():
                if key != "VESSEL" and data:
                    # Simulate a packet for other data types
                    self._handle_message(json.dumps({"Cmd": key, "Value": data}))

        elif value and isinstance(value, list):
            # Handle BC format: list of command objects
            logger.debug(f"Batch command with {len(value)} items")
            for item in value:
                if isinstance(item, dict):
                    # Recursively handle each item
                    self._handle_message(json.dumps(item))

    def _handle_batch_vessel_data(self, vessel_data):
        """Handle VESSEL data from BATCH packet - contains hull/shield info."""
        if not vessel_data or not isinstance(vessel_data, dict):
            return

        # Check if this is our vessel
        vessel_id = vessel_data.get("ID")
        if self.vessel_id and vessel_id != self.vessel_id:
            return  # Not our vessel

        # Extract hull/shield data
        changes = []

        # Hull integrity data
        if "Integrity" in vessel_data:
            integrity = vessel_data["Integrity"]
            max_integrity = vessel_data.get("MaxIntegrity", integrity)
            heal_integrity = vessel_data.get("HealIntegrity", integrity)

            # Calculate hull percentage
            hull_percentage = (integrity / max_integrity * 100) if max_integrity > 0 else 100

            # Check for changes
            old_hull = self.vessel_data.get("hull_percentage")
            if old_hull != hull_percentage:
                self.vessel_data["hull_percentage"] = hull_percentage
                self.vessel_data["hull_integrity"] = integrity
                self.vessel_data["hull_max"] = max_integrity
                self.vessel_data["hull_heal"] = heal_integrity
                changes.append(("hull", old_hull, hull_percentage))

                # Emit hull change event
                self._emit_event({
                    "type": "hull_update",
                    "category": "telemetry",
                    "data": {
                        "hull_percentage": hull_percentage,
                        "hull_integrity": integrity,
                        "hull_max": max_integrity,
                        "hull_heal": heal_integrity,
                        "vessel_id": vessel_id
                    },
                    "priority": "high" if hull_percentage < 50 else "medium"
                })

        # Shield data
        if "Shields" in vessel_data:
            shields = vessel_data["Shields"]

            # Shields can be a simple value (0-1) or a complex object
            if isinstance(shields, (int, float)):
                shield_percentage = shields * 100
                old_shield = self.vessel_data.get("shield_percentage")

                if old_shield != shield_percentage:
                    self.vessel_data["shield_percentage"] = shield_percentage
                    changes.append(("shield", old_shield, shield_percentage))

                    # Emit shield change event
                    self._emit_event({
                        "type": "shield_update",
                        "category": "telemetry",
                        "data": {
                            "shield_percentage": shield_percentage,
                            "vessel_id": vessel_id
                        },
                        "priority": "high" if shield_percentage < 25 else "medium"
                    })
            elif isinstance(shields, dict):
                # Complex shield object with quadrants
                self._handle_shield_data(shields)

        # Log significant changes
        if changes:
            logger.info(f"🚢 Hull/Shield update from BATCH:")
            for name, old_val, new_val in changes:
                old_str = f"{old_val:.1f}%" if old_val is not None else "N/A"
                new_str = f"{new_val:.1f}%" if new_val is not None else "N/A"
                logger.info(f"   {name}: {old_str} → {new_str}")

        # Update last vessel update time
        self.last_vessel_update = datetime.now()

    def _handle_shield_data(self, shield_data):
        """Handle complex shield data with quadrants."""
        if isinstance(shield_data, dict):
            # Check for shield quadrants
            if "Details" in shield_data and isinstance(shield_data["Details"], dict):
                segments = shield_data["Details"].get("Segments", {})
                for segment_id, segment_data in segments.items():
                    if isinstance(segment_data, dict) and "Value" in segment_data:
                        # Update quadrant shield data
                        quadrant = f"shield_segment_{segment_id}"
                        self.vessel_data[quadrant] = segment_data["Value"]

            # Handle overall shield level
            if "Level" in shield_data:
                level = shield_data["Level"]
                optimal = shield_data.get("Optimal", level)
                shield_percentage = (level / optimal * 100) if optimal > 0 else 0

                old_shield = self.vessel_data.get("shield_percentage")
                if old_shield != shield_percentage:
                    self.vessel_data["shield_percentage"] = shield_percentage
                    self.vessel_data["shield_level"] = level
                    self.vessel_data["shield_optimal"] = optimal

                    self._emit_event({
                        "type": "shield_update",
                        "category": "telemetry",
                        "data": {
                            "shield_percentage": shield_percentage,
                            "shield_level": level,
                            "shield_optimal": optimal
                        },
                        "priority": "high" if shield_percentage < 25 else "medium"
                    })

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add event callback."""
        self._callbacks.append(callback)

    def _emit_event(self, event: Dict[str, Any]):
        """Emit event to callbacks."""
        event["timestamp"] = datetime.now().isoformat()

        # Add priority if not already set
        if "priority" not in event:
            event["priority"] = self.event_prioritizer.get_event_priority(event)

        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def request_vessel_data(self):
        """Request fresh vessel data."""
        if self.vessel_id:
            self._send("GET", "VESSEL")
            self._send("GET", "VESSEL-VALUES")
            self._send("GET", "VARIABLES")
            self._send("GET", "STATUS")
            self._send("GET", "SHIELDS")
            self._send("GET", "HULL")
            self._send("GET", "SYSTEMS")
            self._send("GET", "POWER")

    def get_vessel_data(self) -> Dict[str, Any]:
        """Get current vessel data."""
        return self.vessel_data.copy()

    def _is_system_critical(self, system_name: str, value) -> bool:
        """Check if a system is in critical state."""
        if isinstance(value, (int, float)):
            return value < 30  # Below 30% is critical
        elif isinstance(value, dict):
            # Check for critical indicators in dict
            if 'Status' in value and value['Status'] in ['Critical', 'Failed', 'Offline']:
                return True
            if 'Health' in value and isinstance(value['Health'], (int, float)) and value['Health'] < 30:
                return True
            if 'Percent' in value and isinstance(value['Percent'], (int, float)) and value['Percent'] < 30:
                return True
        return False

    def get_ship_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive ship status summary."""
        return {
            'hull': self.vessel_data.get('hull_percentage', 100),
            'shields': self.vessel_data.get('shield_percentage', 100),
            'systems_health': self.vessel_data.get('systems', {}),
            'power_distribution': self.vessel_data.get('power', {}),
            'engineering': self.vessel_data.get('engineering', {}),
            'damage_events': len(self.vessel_data.get('damage_events', [])),
            'cumulative_damage': self.vessel_data.get('cumulative_damage', 0),
            'alert_level': self.vessel_data.get('alert', 2),
            'combat_readiness': self.vessel_data.get('combat', {}),
            'navigation': self.vessel_data.get('navigation', {}),
            'shield_sections': self.vessel_data.get('shield_sections', {}),
            'shield_distribution': self.vessel_data.get('shield_distribution', {})
        }

    def get_all_vessels_status(self) -> Dict[int, Dict[str, Any]]:
        """Get hull and shield status for all detected vessels."""
        return self.vessel_data.get('all_vessel_status', {})

    def _is_significant_vessel_change(self, field: str, old_value, new_value) -> bool:
        """Determine if a vessel data change is significant."""
        if old_value is None:
            return True

        # Critical fields always report
        if field in ['hull_percentage', 'shield_percentage', 'alert', 'weapons_armed', 'red_alert']:
            return True

        # Percentage changes > 5%
        if 'percentage' in field or 'percent' in field:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                return abs(new_value - old_value) > 5

        # Position changes
        if field == 'position':
            return True  # Always track position changes

        return old_value != new_value

    def _get_change_priority(self, field: str, value) -> str:
        """Get priority level for a field change."""
        # Critical if hull/shields below 30%
        if field in ['hull_percentage', 'shield_percentage']:
            if isinstance(value, (int, float)) and value < 30:
                return "CRITICAL"
            elif isinstance(value, (int, float)) and value < 50:
                return "HIGH"

        # Alert changes are high priority
        if field == 'alert' and value >= 4:
            return "CRITICAL"
        elif field == 'alert':
            return "HIGH"

        # Combat changes are high priority
        if field in ['weapons_armed', 'target_lock', 'red_alert']:
            return "HIGH"

        return "MEDIUM"

    def get_packet_stats(self) -> Dict[str, int]:
        """Get packet statistics."""
        return self.packet_counts.copy()

    def get_mission_data(self) -> Dict[str, Any]:
        """Get all mission and strategic data."""
        return self.mission_data.copy()

    def get_ship_internals(self) -> Dict[str, Any]:
        """Get ship internal systems data."""
        return self.ship_internals.copy()

    def get_advanced_systems(self) -> Dict[str, Any]:
        """Get advanced systems data (components, factions, etc)."""
        return self.advanced_systems.copy()

    def get_combat_enhanced(self) -> Dict[str, Any]:
        """Get enhanced combat data (ordnance, projectiles, drones)."""
        return self.combat_enhanced.copy()

    def get_multiplayer_data(self) -> Dict[str, Any]:
        """Get multiplayer/crew coordination data."""
        return self.multiplayer_data.copy()

    def get_ui_state(self) -> Dict[str, Any]:
        """Get UI and media state."""
        return self.ui_state.copy()

    def get_comprehensive_telemetry(self) -> Dict[str, Any]:
        """Get all telemetry data in one comprehensive report."""
        return {
            'vessel': self.get_vessel_data(),
            'mission': self.get_mission_data(),
            'ship_internals': self.get_ship_internals(),
            'advanced_systems': self.get_advanced_systems(),
            'combat': self.get_combat_enhanced(),
            'multiplayer': self.get_multiplayer_data(),
            'ui_state': self.get_ui_state(),
            'packet_stats': self.get_packet_stats(),
            'ship_status': self.get_ship_status_summary()
        }

    def _handle_unhandled_packet(self, cmd: str, value):
        """Handle previously unhandled packets like WEAPONS, TARGET-*, DAMAGE-TEAMS."""
        logger.info(f"🔧 {cmd} packet: {str(value)[:200]}...")

        # Emit event for recording
        self._emit_event({
            "type": f"{cmd.lower().replace('-', '_')}_data",
            "category": "telemetry",
            "data": value
        })

    def _handle_contacts(self, value):
        """Handle CONTACTS packet - detailed information about all detected objects."""
        if value and isinstance(value, list):
            logger.info(f"👀 Contacts detected: {len(value)} objects")

            # Store contact information for analysis
            self.vessel_data['contacts'] = value

            # Track hull and shield status for all vessels
            vessel_statuses = {}
            for contact in value:
                vessel_id = contact.get('ID')
                if vessel_id:
                    vessel_statuses[vessel_id] = {
                        'name': contact.get('Name', 'Unknown'),
                        'faction': contact.get('Faction', 'Unknown'),
                        'type': contact.get('BaseType', 'Unknown'),
                        'hull': contact.get('Integrity', 1) * 100,  # Convert to percentage
                        'shields': contact.get('Shields', 0) * 100   # Convert to percentage
                    }

            self.vessel_data['all_vessel_status'] = vessel_statuses

            # Process each contact through smart filters
            significant_contacts = []
            missile_events = 0

            for contact in value:
                # Check for missile events
                missile_event = self.missile_tracker.process_missile_event(contact)
                if missile_event:
                    # Add priority and emit missile event
                    missile_event["priority"] = self.event_prioritizer.get_event_priority(missile_event)
                    self._emit_event(missile_event)
                    missile_events += 1

                # Check if contact update is significant
                if self.contact_filter.is_significant_contact(contact):
                    significant_contacts.append(contact)

            # Log filtering results
            if missile_events > 0:
                logger.info(f"🚀 Missile events: {missile_events}")

            if significant_contacts:
                logger.info(f"📡 Significant contacts: {len(significant_contacts)}/{len(value)}")

                # Log details of significant contacts
                for i, contact in enumerate(significant_contacts[:3]):
                    contact_type = contact.get('BaseType', 'Unknown')
                    name = contact.get('Name', 'Unnamed')
                    faction = contact.get('Faction', 'Unknown')
                    hull = contact.get('Integrity', 1) * 100
                    shields = contact.get('Shields', 0) * 100
                    logger.info(f"   {i+1}. {contact_type}: {name} ({faction}) - Hull: {hull:.0f}%, Shields: {shields:.0f}%")

                # Only emit contacts event if we have significant updates
                contact_event = {
                    "type": "contacts_data",
                    "category": "sensors",
                    "data": significant_contacts
                }
                contact_event["priority"] = self.event_prioritizer.get_event_priority(contact_event)
                self._emit_event(contact_event)
            else:
                logger.info("📡 No significant contact changes (filtered)")

            # Cleanup expired missiles periodically
            self.missile_tracker.cleanup_expired_missiles()

    def _handle_contact_remove(self, value):
        """Handle CONTACT-REMOVE packet - when objects are destroyed or leave sensor range."""
        if value:
            logger.info(f"❌ Contact removed: {value}")

            # Check if this was a missile impact
            missile_event = self.missile_tracker.process_missile_removal(value)
            if missile_event:
                missile_event["priority"] = self.event_prioritizer.get_event_priority(missile_event)
                self._emit_event(missile_event)
                logger.info(f"🚀💥 Missile impact detected: {value}")

            # Standard contact removal event
            contact_event = {
                "type": "contact_removed",
                "category": "sensors",
                "data": {"contact_id": value}
            }
            contact_event["priority"] = self.event_prioritizer.get_event_priority(contact_event)
            self._emit_event(contact_event)

    def _handle_object_remove(self, value):
        """Handle OBJECT-REMOVE packet - when game objects are destroyed."""
        if value:
            logger.info(f"💥 Object destroyed: {value}")
            self._emit_event({
                "type": "object_destroyed",
                "category": "combat",
                "data": {"object_id": value}
            })

    def _handle_target_science(self, value):
        """Handle TARGET-SCIENCE packet - science station targeting."""
        if value:
            logger.info(f"🔬 Science targeting: {value}")
            self.vessel_data['science_target'] = value
            self._emit_event({
                "type": "science_target_selected",
                "category": "science",
                "data": {"target_id": value}
            })

    def _handle_players(self, value):
        """Handle PLAYERS packet - crew and player information."""
        if value and isinstance(value, list):
            logger.info(f"👥 Players/Crew: {len(value)} active")

            # Extract crew roles and positions
            crew_info = []
            for player in value:
                roles = player.get('Roles', [])
                screens = player.get('Screens', 0)
                vessel_id = player.get('VesselID', 'Unknown')
                crew_info.append({
                    'roles': roles,
                    'screens': screens,
                    'vessel': vessel_id
                })

            self.vessel_data['crew'] = crew_info
            self._emit_event({
                "type": "crew_status",
                "category": "crew",
                "data": crew_info
            })

    def _handle_system_log(self, value):
        """Handle SYSTEM-LOG packet - game system messages and events."""
        if value and isinstance(value, list):
            logger.info(f"📋 System log: {len(value)} new entries")

            # Log important system messages
            for entry in value[-3:]:  # Show last 3 entries
                message = entry.get('Message', 'Unknown')
                msg_type = entry.get('Type', 'Info')
                logger.info(f"   [{msg_type}] {message}")

            self._emit_event({
                "type": "system_log",
                "category": "system",
                "data": value
            })

    def _handle_mission(self, value):
        """Handle MISSION packet - detailed mission status and progress."""
        if value and isinstance(value, dict):
            mission_name = value.get('Name', 'Unknown')
            elapsed_time = value.get('ElapsedTime', '0')
            timer = value.get('Timer', '0')
            logger.info(f"🎯 Mission status: {mission_name} | Elapsed: {elapsed_time}s | Timer: {timer}s")

            self._emit_event({
                "type": "mission_status",
                "category": "mission",
                "data": value
            })

    def _handle_engineering_systems(self, system_name: str, value):
        """Handle engineering system packets - ENHANCED for comprehensive tracking."""
        if value is not None:
            # Parse different engineering data formats
            if isinstance(value, dict):
                # Complex engineering data (power distribution, etc.)
                logger.info(f"🔧 {system_name} detailed status: {len(value)} parameters")

                # Extract and update specific system data
                if system_name == "POWER":
                    # Update power distribution
                    if 'TotalAvailable' in value:
                        self.vessel_data['power']['total_available'] = value['TotalAvailable']
                    if 'Engines' in value:
                        self.vessel_data['power']['engines'] = value['Engines']
                    if 'Weapons' in value:
                        self.vessel_data['power']['weapons'] = value['Weapons']
                    if 'Shields' in value:
                        self.vessel_data['power']['shields'] = value['Shields']
                    if 'Auxiliary' in value:
                        self.vessel_data['power']['auxiliary'] = value['Auxiliary']
                    if 'LifeSupport' in value:
                        self.vessel_data['power']['life_support'] = value['LifeSupport']

                elif system_name == "REACTOR":
                    # Update reactor status
                    if 'Output' in value:
                        self.vessel_data['engineering']['reactor_output'] = value['Output']
                    if 'Efficiency' in value:
                        self.vessel_data['engineering']['reactor_efficiency'] = value['Efficiency']
                    if 'Temperature' in value:
                        self.vessel_data['engineering']['reactor_temperature'] = value['Temperature']

                elif system_name == "COOLANT":
                    # Update coolant system
                    if 'Level' in value:
                        self.vessel_data['engineering']['coolant_level'] = value['Level']
                    if 'Pressure' in value:
                        self.vessel_data['engineering']['coolant_pressure'] = value['Pressure']

                elif system_name == "SYSTEMS":
                    # Update individual system status
                    for sys_name, sys_health in value.items():
                        sys_key = sys_name.lower().replace('-', '_')
                        if sys_key in self.vessel_data['systems']:
                            self.vessel_data['systems'][sys_key] = sys_health

                elif system_name == "DAMAGE-CONTROL":
                    # Update damage control teams
                    if 'Teams' in value:
                        self.vessel_data['engineering']['damage_control_teams'] = value['Teams']

                elif system_name == "REPAIR-TEAMS":
                    # Update repair teams
                    if isinstance(value, list):
                        self.vessel_data['engineering']['repair_teams'] = value
                    elif 'Teams' in value:
                        self.vessel_data['engineering']['repair_teams'] = value['Teams']

                # Log key parameters
                for key, val in list(value.items())[:3]:  # Log first 3 entries
                    logger.info(f"   {key}: {val}")
                if len(value) > 3:
                    logger.info(f"   ... and {len(value)-3} more parameters")
            else:
                # Simple status value (typically percentage)
                logger.info(f"🔧 {system_name} status: {value}")

                # Update specific system percentages
                if system_name == "ENGINES" and isinstance(value, (int, float)):
                    self.vessel_data['systems']['engines'] = value
                elif system_name == "WEAPONS" and isinstance(value, (int, float)):
                    self.vessel_data['systems']['weapons'] = value
                elif system_name == "SENSORS" and isinstance(value, (int, float)):
                    self.vessel_data['systems']['sensors'] = value
                elif system_name == "LIFE-SUPPORT" and isinstance(value, (int, float)):
                    self.vessel_data['systems']['life_support'] = value
                elif system_name == "WARP" and isinstance(value, (int, float)):
                    self.vessel_data['systems']['warp_drive'] = value
                elif system_name == "IMPULSE" and isinstance(value, (int, float)):
                    self.vessel_data['systems']['impulse_drive'] = value

            # Store raw engineering data
            self.vessel_data[f"eng_{system_name.lower()}"] = value

            # Determine event priority based on system criticality
            critical_systems = ["REACTOR", "LIFE-SUPPORT", "HULL", "POWER-GRID"]
            priority = "CRITICAL" if system_name in critical_systems and self._is_system_critical(system_name, value) else "HIGH" if system_name in critical_systems else "MEDIUM"

            # Check for significant changes
            old_value = self.vessel_data.get(f"eng_{system_name.lower()}_previous")
            is_significant_change = self._is_significant_engineering_change(system_name, old_value, value)

            if is_significant_change or old_value is None:
                self.vessel_data[f"eng_{system_name.lower()}_previous"] = value

                event = {
                    "type": f"engineering_{system_name.lower()}",
                    "category": "engineering",
                    "priority": priority,
                    "data": {
                        "system": system_name,
                        "status": value,
                        "change_detected": old_value is not None
                    }
                }

                self._emit_event(event)

    def _is_significant_engineering_change(self, system_name: str, old_value, new_value) -> bool:
        """Determine if engineering change is significant enough to log."""
        if old_value is None:
            return True  # First reading is always significant

        # For numeric values, check for meaningful changes
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            # Different thresholds for different systems
            if system_name in ["HULL", "SHIELDS"]:
                return abs(new_value - old_value) > 5.0  # 5% change
            elif system_name in ["POWER", "REACTOR"]:
                return abs(new_value - old_value) > 10.0  # 10% change
            else:
                return abs(new_value - old_value) > 1.0  # 1% change

        # For dict values, check for key changes
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            # Check if any key values changed significantly
            for key in set(old_value.keys()) | set(new_value.keys()):
                old_val = old_value.get(key, 0)
                new_val = new_value.get(key, 0)
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if abs(new_val - old_val) > 5.0:  # 5% threshold for subsystems
                        return True

        # For other types, any change is significant
        return old_value != new_value

    def _handle_vessel_components(self, value):
        """Handle VESSEL-COMPONENTS packet - individual ship component status."""
        if value:
            logger.info(f"⚙️ Vessel components: {len(value) if isinstance(value, (list, dict)) else 'data'}")

            # Store component data
            if isinstance(value, list):
                # List of components
                for component in value:
                    if isinstance(component, dict) and 'ID' in component:
                        comp_id = component['ID']
                        self.advanced_systems['components'][comp_id] = component
            elif isinstance(value, dict):
                # Dictionary of components
                self.advanced_systems['components'].update(value)

            self._emit_event({
                "type": "vessel_components",
                "category": "engineering",
                "data": value
            })

    def _handle_vessel_waypoints(self, value):
        """Handle VESSEL-WAYPOINTS packet - navigation waypoint data."""
        if value:
            logger.info(f"🗺️ Vessel waypoints: {len(value) if isinstance(value, (list, dict)) else 'data'}")

            # Store waypoint data
            self.vessel_data['waypoints'] = value

            self._emit_event({
                "type": "vessel_waypoints",
                "category": "navigation",
                "data": value
            })

    def _handle_vessel_occupants(self, value):
        """Handle VESSEL-OCCUPANTS packet - crew locations on ship."""
        if value:
            logger.info(f"👥 Vessel occupants: {len(value) if isinstance(value, (list, dict)) else 'data'}")

            # Store occupant data
            self.ship_internals['occupants'] = value

            self._emit_event({
                "type": "vessel_occupants",
                "category": "crew",
                "data": value
            })

    def _handle_game_metadata(self, cmd: str, value):
        """Handle game metadata packets (SETTINGS, SVRS, USER-INFO, ROLES, SCREENS)."""
        if value:
            logger.info(f"ℹ️ {cmd}: {str(value)[:200]}...")

            # Store metadata based on type
            metadata_key = cmd.lower().replace('-', '_')
            self.vessel_data[f'game_{metadata_key}'] = value

            # Special handling for important metadata
            if cmd == "ROLES":
                logger.info(f"   Station roles: {value}")
            elif cmd == "SCREENS":
                logger.info(f"   Available screens: {len(value) if isinstance(value, list) else value}")
            elif cmd == "SETTINGS":
                # Game settings might contain important configuration
                if isinstance(value, dict):
                    for key in ['WebSocketHeartbeat', 'GameSpeed', 'Difficulty']:
                        if key in value:
                            logger.info(f"   {key}: {value[key]}")

            self._emit_event({
                "type": f"game_{metadata_key}",
                "category": "metadata",
                "data": value
            })

    def _handle_control_packet(self, cmd: str, value):
        """Handle control packets (RESET, CLEAR, SESSION-CLEAR)."""
        logger.warning(f"🔄 Control packet: {cmd}")

        # Handle based on type
        if cmd == "RESET":
            # Game reset - might need to clear some state
            logger.warning("Game reset detected")
        elif cmd == "CLEAR":
            # Data clear request
            logger.warning("Data clear request")
        elif cmd == "SESSION-CLEAR":
            # Session cleared
            self.session_id = None
            logger.warning("Session cleared")

        self._emit_event({
            "type": f"control_{cmd.lower().replace('-', '_')}",
            "category": "control",
            "data": value,
            "priority": "HIGH"
        })

    def disconnect(self):
        """Disconnect WebSocket."""
        if self.ws:
            self.ws.close()
            self.connected = False
            logger.info("Disconnected")