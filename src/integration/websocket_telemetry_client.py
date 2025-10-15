#!/usr/bin/env python3
"""
WebSocket Telemetry Client for Starship Horizons
Captures real-time ship system data via WebSocket connection.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import websocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShipTelemetryClient:
    """WebSocket client for real-time ship telemetry."""

    def __init__(self, host: str = None, port: int = None):
        """
        Initialize telemetry client.

        Args:
            host: Game server host (defaults to env var GAME_HOST)
            port: WebSocket port (defaults to env var GAME_PORT_WS)
        """
        if host is None:
            host = os.getenv("GAME_HOST", "localhost")
        if port is None:
            port = int(os.getenv("GAME_PORT_WS", "1865"))
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}"

        self.ws = None
        self.connected = False
        self._callbacks = []

        # Ship state tracking
        self.ship_state = {
            "alert_level": "green",
            "shields": {"percent": 100, "status": "online"},
            "hull": {"percent": 100, "breaches": 0},
            "engines": {"power": 100, "speed": 0, "status": "online"},
            "weapons": {"status": "standby", "charged": 100},
            "power": {"total": 100, "distribution": {}},
            "damage": [],
            "systems": {}
        }

        self._last_state = {}
        self._packet_counts = {}

    def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            logger.info(f"Connecting to WebSocket at {self.ws_url}")

            def on_message(ws, message):
                self._handle_message(message)

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                self.connected = False

            def on_close(ws, close_code, close_msg):
                logger.info(f"WebSocket closed: {close_code} - {close_msg}")
                self.connected = False

            def on_open(ws):
                logger.info("WebSocket connected successfully")
                self.connected = True
                self._send_identification()
                self._emit_event({
                    "type": "telemetry_connected",
                    "category": "system",
                    "data": {"url": self.ws_url}
                })

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

            logger.warning("WebSocket connection timeout")
            return False

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def _send_identification(self):
        """Send identification packet to server."""
        try:
            # Mimic game client identification
            ident = {
                "cmd": "IDENTIFY",
                "data": {
                    "Type": "Observer",
                    "Version": "41",
                    "Guid": f"recorder_{datetime.now().timestamp()}",
                    "IsMainViewer": False
                }
            }
            self.ws.send(json.dumps(ident))
            logger.debug("Sent identification packet")
        except Exception as e:
            logger.error(f"Failed to send identification: {e}")

    def _handle_message(self, message):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message

            # Get packet type
            cmd = data.get("cmd") or data.get("type")
            if not cmd:
                return

            # Track packet
            self._packet_counts[cmd] = self._packet_counts.get(cmd, 0) + 1

            # Process based on type
            if cmd == "VESSEL":
                self._process_vessel_data(data.get("data", {}))
            elif cmd == "ALERT":
                self._process_alert(data.get("data", {}))
            elif cmd == "DAMAGE":
                self._process_damage(data.get("data", {}))
            elif cmd == "SYSTEM-LOG":
                self._process_system_log(data.get("data", {}))
            elif cmd == "CONTACTS":
                self._process_contacts(data.get("data", {}))
            elif cmd in ["SHIELDS", "HULL", "ENGINES", "WEAPONS", "POWER"]:
                self._process_system_status(cmd, data.get("data", {}))

        except Exception as e:
            logger.debug(f"Message handling error: {e}")

    def _process_vessel_data(self, vessel_data: Dict):
        """Process vessel telemetry data."""
        if not vessel_data:
            return

        changes = []

        # Check shields
        if "Shields" in vessel_data:
            new_shields = vessel_data["Shields"]
            if isinstance(new_shields, dict):
                shield_pct = new_shields.get("Percent", new_shields.get("Value", 100))
            else:
                shield_pct = new_shields

            if shield_pct != self.ship_state["shields"]["percent"]:
                old_value = self.ship_state["shields"]["percent"]
                self.ship_state["shields"]["percent"] = shield_pct
                changes.append(("shields", old_value, shield_pct))

        # Check hull
        if "Hull" in vessel_data:
            hull_pct = vessel_data["Hull"]
            if hull_pct != self.ship_state["hull"]["percent"]:
                old_value = self.ship_state["hull"]["percent"]
                self.ship_state["hull"]["percent"] = hull_pct
                changes.append(("hull", old_value, hull_pct))

        # Check engines
        if "Engines" in vessel_data or "Speed" in vessel_data:
            speed = vessel_data.get("Speed", 0)
            if speed != self.ship_state["engines"]["speed"]:
                old_value = self.ship_state["engines"]["speed"]
                self.ship_state["engines"]["speed"] = speed
                changes.append(("speed", old_value, speed))

        # Check power
        if "Power" in vessel_data:
            power = vessel_data.get("Power", {})
            if isinstance(power, dict):
                total_power = power.get("Total", 100)
            else:
                total_power = power

            if total_power != self.ship_state["power"]["total"]:
                old_value = self.ship_state["power"]["total"]
                self.ship_state["power"]["total"] = total_power
                changes.append(("power", old_value, total_power))

        # Emit change events
        for system, old_val, new_val in changes:
            self._emit_event({
                "type": f"{system}_changed",
                "category": "telemetry",
                "data": {
                    "system": system,
                    "old_value": old_val,
                    "new_value": new_val,
                    "timestamp": datetime.now().isoformat()
                }
            })

    def _process_alert(self, alert_data: Dict):
        """Process alert level changes."""
        alert_level = alert_data.get("Level") or alert_data.get("Status")

        if alert_level and alert_level != self.ship_state["alert_level"]:
            old_level = self.ship_state["alert_level"]
            self.ship_state["alert_level"] = alert_level

            self._emit_event({
                "type": "alert_level_changed",
                "category": "alert",
                "data": {
                    "old_level": old_level,
                    "new_level": alert_level,
                    "message": alert_data.get("Message", ""),
                    "timestamp": datetime.now().isoformat()
                }
            })

    def _process_damage(self, damage_data: Dict):
        """Process damage reports."""
        self._emit_event({
            "type": "damage_report",
            "category": "damage",
            "data": {
                "system": damage_data.get("System"),
                "severity": damage_data.get("Severity"),
                "description": damage_data.get("Description"),
                "timestamp": datetime.now().isoformat()
            }
        })

        # Track damage
        self.ship_state["damage"].append(damage_data)

    def _process_system_log(self, log_data: Dict):
        """Process system log entries."""
        self._emit_event({
            "type": "system_log",
            "category": "log",
            "data": {
                "message": log_data.get("Message"),
                "level": log_data.get("Level"),
                "source": log_data.get("Source"),
                "timestamp": datetime.now().isoformat()
            }
        })

    def _process_contacts(self, contacts_data: Dict):
        """Process sensor contacts."""
        num_contacts = len(contacts_data) if isinstance(contacts_data, list) else 1

        self._emit_event({
            "type": "sensor_contacts",
            "category": "sensors",
            "data": {
                "count": num_contacts,
                "contacts": contacts_data if isinstance(contacts_data, list) else [contacts_data],
                "timestamp": datetime.now().isoformat()
            }
        })

    def _process_system_status(self, system: str, status_data: Dict):
        """Process individual system status updates."""
        self.ship_state["systems"][system.lower()] = status_data

        self._emit_event({
            "type": f"{system.lower()}_status",
            "category": "systems",
            "data": {
                "system": system.lower(),
                "status": status_data,
                "timestamp": datetime.now().isoformat()
            }
        })

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add event callback."""
        self._callbacks.append(callback)

    def _emit_event(self, event: Dict[str, Any]):
        """Emit event to callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_ship_state(self) -> Dict[str, Any]:
        """Get current ship state."""
        return self.ship_state.copy()

    def get_packet_stats(self) -> Dict[str, int]:
        """Get packet statistics."""
        return self._packet_counts.copy()

    def disconnect(self):
        """Disconnect WebSocket."""
        if self.ws:
            self.ws.close()
            self.connected = False
            logger.info("WebSocket disconnected")