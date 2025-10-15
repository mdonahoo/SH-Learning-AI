#!/usr/bin/env python3
"""
Working WebSocket Client for Starship Horizons
Based on discovered protocol with Cmd/Value format.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable, Optional

import websocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StarshipWebSocketClient:
    """Working WebSocket client using discovered protocol."""

    def __init__(self, host: str = None, port: int = None):
        if host is None:
            host = os.getenv("GAME_HOST", "localhost")
        if port is None:
            port = int(os.getenv("GAME_PORT_WS", "1865"))
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}"
        self.ws = None
        self.connected = False
        self.session_id = None

        # Callbacks
        self._callbacks = []

        # Ship state from WebSocket
        self.ship_data = {}
        self.last_update = {}

        # Packet tracking
        self.packet_types_received = set()
        self.packet_counts = {}

    def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            logger.info(f"Connecting to {self.ws_url}")

            def on_message(ws, message):
                self._handle_message(message)

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")

            def on_close(ws, close_code, close_msg):
                logger.info(f"WebSocket closed")
                self.connected = False

            def on_open(ws):
                logger.info("WebSocket connected")
                self.connected = True

                # Send proper identification
                self._identify()

                # Request initial data
                self._request_all_data()

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
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
                    return True
                time.sleep(0.5)

            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def _identify(self):
        """Send proper identification."""
        ident = {
            "cmd": "IDENTIFY",
            "data": {
                "Type": "MainViewer",
                "Version": "41",
                "IsMainViewer": True
            }
        }
        self._send_packet(ident)

    def _request_all_data(self):
        """Request all available data types."""
        # Based on the game's packet types
        requests = [
            "VESSEL",
            "VESSEL-ID",
            "CONTACTS",
            "STATUS",
            "DAMAGE",
            "SYSTEMS",
            "VARIABLES",
            "MISSION"
        ]

        for req in requests:
            self._send_packet({"cmd": "GET", "type": req})
            time.sleep(0.1)

    def _send_packet(self, packet: Dict):
        """Send packet to server."""
        try:
            if self.ws and self.connected:
                msg = json.dumps(packet)
                self.ws.send(msg)
                logger.debug(f"Sent: {packet.get('cmd', 'UNKNOWN')}")
        except Exception as e:
            logger.error(f"Send failed: {e}")

    def _handle_message(self, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Extract command and value
            cmd = data.get("Cmd") or data.get("cmd")
            value = data.get("Value") or data.get("value") or data.get("data")

            if not cmd:
                return

            # Track packet types
            self.packet_types_received.add(cmd)
            self.packet_counts[cmd] = self.packet_counts.get(cmd, 0) + 1

            # Log new packet types
            if self.packet_counts[cmd] == 1:
                logger.info(f"New packet type: {cmd}")

            # Process based on command type
            if cmd == "SESSION":
                self._handle_session(value)
            elif cmd == "VESSEL":
                self._handle_vessel(value)
            elif cmd == "DAMAGE":
                self._handle_damage(value)
            elif cmd == "ALERT":
                self._handle_alert(value)
            elif cmd == "CONTACTS":
                self._handle_contacts(value)
            elif cmd == "SYSTEM-LOG":
                self._handle_system_log(value)
            elif cmd == "VARIABLES":
                self._handle_variables(value)
            elif cmd == "STATUS":
                self._handle_status(value)
            elif cmd == "PING":
                # Respond to ping
                self._send_packet({"cmd": "PONG"})
            else:
                # Log unknown packets for discovery
                if value:
                    logger.debug(f"Unhandled packet {cmd}: {str(value)[:100]}")

        except Exception as e:
            logger.debug(f"Message handling error: {e}")

    def _handle_session(self, data):
        """Handle session data."""
        if data:
            self.session_id = data.get("ID")
            mode = data.get("Mode")
            state = data.get("State")

            self._emit_event({
                "type": "session_update",
                "category": "session",
                "data": {
                    "session_id": self.session_id,
                    "mode": mode,
                    "state": state
                }
            })

    def _handle_vessel(self, data):
        """Handle vessel telemetry."""
        if not data:
            return

        # Check for changes
        changes = []

        # Common vessel fields based on game
        fields_to_check = [
            ("Shields", "shields"),
            ("Hull", "hull"),
            ("Energy", "energy"),
            ("Speed", "speed"),
            ("Heading", "heading"),
            ("Alert", "alert"),
            ("Power", "power")
        ]

        for field, name in fields_to_check:
            if field in data:
                old_value = self.ship_data.get(name)
                new_value = data[field]

                if old_value != new_value:
                    self.ship_data[name] = new_value
                    changes.append((name, old_value, new_value))

        # Emit change events
        for name, old_val, new_val in changes:
            self._emit_event({
                "type": f"{name}_changed",
                "category": "telemetry",
                "data": {
                    "system": name,
                    "old_value": old_val,
                    "new_value": new_val,
                    "timestamp": datetime.now().isoformat()
                }
            })

        # Store full vessel data
        self.ship_data.update(data)

    def _handle_damage(self, data):
        """Handle damage reports."""
        self._emit_event({
            "type": "damage_report",
            "category": "damage",
            "data": data
        })

    def _handle_alert(self, data):
        """Handle alert changes."""
        self._emit_event({
            "type": "alert_changed",
            "category": "alert",
            "data": data
        })

    def _handle_contacts(self, data):
        """Handle sensor contacts."""
        if data:
            num_contacts = len(data) if isinstance(data, list) else 1
            self._emit_event({
                "type": "contacts_update",
                "category": "sensors",
                "data": {
                    "count": num_contacts,
                    "contacts": data
                }
            })

    def _handle_system_log(self, data):
        """Handle system log messages."""
        self._emit_event({
            "type": "system_log",
            "category": "log",
            "data": data
        })

    def _handle_variables(self, data):
        """Handle game variables (might contain ship data)."""
        if data:
            # Variables might contain shield%, hull%, etc.
            self.ship_data["variables"] = data

            # Check for specific variables
            if isinstance(data, dict):
                for key, value in data.items():
                    if "shield" in key.lower():
                        self._emit_event({
                            "type": "shield_variable",
                            "category": "telemetry",
                            "data": {"key": key, "value": value}
                        })
                    elif "hull" in key.lower():
                        self._emit_event({
                            "type": "hull_variable",
                            "category": "telemetry",
                            "data": {"key": key, "value": value}
                        })

    def _handle_status(self, data):
        """Handle status updates."""
        if data:
            self._emit_event({
                "type": "status_update",
                "category": "status",
                "data": data
            })

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add event callback."""
        self._callbacks.append(callback)

    def _emit_event(self, event: Dict[str, Any]):
        """Emit event to callbacks."""
        # Log significant events
        if "changed" in event["type"] or event["type"] in ["damage_report", "alert_changed"]:
            logger.info(f"🚀 {event['type']}: {event.get('data', {})}")

        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def request_update(self):
        """Request fresh data from server."""
        requests = ["VESSEL", "STATUS", "DAMAGE", "CONTACTS"]
        for req in requests:
            self._send_packet({"cmd": "GET", "type": req})

    def get_summary(self) -> Dict:
        """Get connection summary."""
        return {
            "connected": self.connected,
            "session_id": self.session_id,
            "packet_types": list(self.packet_types_received),
            "packet_counts": self.packet_counts,
            "ship_data": self.ship_data
        }

    def disconnect(self):
        """Disconnect WebSocket."""
        if self.ws:
            self.ws.close()
            self.connected = False