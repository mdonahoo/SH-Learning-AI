#!/usr/bin/env python3
"""
Starship Horizons API Client
Connects to the game and captures events in real-time.
"""

import json
import logging
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import requests
import websocket
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StarshipHorizonsClient:
    """Client for connecting to Starship Horizons game."""

    def __init__(self, host: str = None):
        """
        Initialize the Starship Horizons client.

        Args:
            host: Game server URL (defaults to env var GAME_HOST)
        """
        # Get host from environment or use provided host
        if host is None:
            game_host = os.getenv("GAME_HOST", "localhost")
            game_port = os.getenv("GAME_PORT_API", "1864")
            host = f"http://{game_host}:{game_port}"

        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/api"

        # Parse host for WebSocket connection
        parsed = urlparse(self.host)
        ws_protocol = "ws" if parsed.scheme == "http" else "wss"
        self.ws_url = f"{ws_protocol}://{parsed.netloc}/ws"

        self.session = requests.Session()
        self.ws = None
        self._polling = False
        self._poll_thread = None
        self._event_callbacks = []
        self._last_state = {}

    def test_connection(self) -> bool:
        """Test connection to the game server."""
        try:
            response = self.session.get(f"{self.api_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Connected to Starship Horizons - State: {data.get('State')}, Mode: {data.get('Mode')}")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to game: {e}")
        return False

    def get_game_status(self) -> Optional[Dict[str, Any]]:
        """Get current game status."""
        try:
            response = self.session.get(f"{self.api_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get game status: {e}")
        return None

    def get_ship_data(self) -> Optional[Dict[str, Any]]:
        """Get current ship data."""
        try:
            # Try different possible endpoints
            endpoints = [
                f"{self.api_url}/ship",
                f"{self.api_url}/ships",
                f"{self.api_url}/player/ship",
                f"{self.host}/ship",
                f"{self.host}/data/ship"
            ]

            for endpoint in endpoints:
                try:
                    response = self.session.get(endpoint, timeout=2)
                    if response.status_code == 200:
                        return response.json()
                except:
                    continue

        except Exception as e:
            logger.debug(f"No ship data available: {e}")
        return None

    def get_mission_data(self) -> Optional[Dict[str, Any]]:
        """Get current mission data."""
        try:
            response = self.session.get(f"{self.api_url}/mission", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"No mission data available: {e}")
        return None

    def add_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback for game events.

        Args:
            callback: Function to call when an event occurs
        """
        self._event_callbacks.append(callback)

    def start_polling(self, interval: float = 1.0):
        """
        Start polling for game events.

        Args:
            interval: Polling interval in seconds
        """
        if self._polling:
            logger.warning("Already polling")
            return

        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._polling_worker,
            args=(interval,),
            daemon=True
        )
        self._poll_thread.start()
        logger.info(f"Started polling game every {interval} seconds")

    def stop_polling(self):
        """Stop polling for game events."""
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        logger.info("Stopped polling")

    def _polling_worker(self, interval: float):
        """Worker thread for polling game state."""
        while self._polling:
            try:
                # Get current game state
                current_state = self.get_game_status() or {}

                # Check for state changes
                if current_state != self._last_state:
                    self._process_state_change(self._last_state, current_state)
                    self._last_state = current_state.copy()

                # Poll other endpoints
                self._poll_game_data()

            except Exception as e:
                logger.error(f"Polling error: {e}")

            time.sleep(interval)

    def _poll_game_data(self):
        """Poll various game data endpoints."""
        # Try to get ship data
        ship_data = self.get_ship_data()
        if ship_data:
            self._emit_event({
                "type": "ship_update",
                "category": "telemetry",
                "timestamp": datetime.now().isoformat(),
                "data": ship_data
            })

        # Try to get mission data
        mission_data = self.get_mission_data()
        if mission_data:
            self._emit_event({
                "type": "mission_update",
                "category": "mission",
                "timestamp": datetime.now().isoformat(),
                "data": mission_data
            })

    def _process_state_change(self, old_state: Dict, new_state: Dict):
        """Process state changes and emit events."""
        if not old_state:
            # Initial state
            self._emit_event({
                "type": "game_connected",
                "category": "system",
                "timestamp": datetime.now().isoformat(),
                "data": new_state
            })
            return

        # Check for mode changes
        if old_state.get("Mode") != new_state.get("Mode"):
            self._emit_event({
                "type": "mode_change",
                "category": "game_state",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "old_mode": old_state.get("Mode"),
                    "new_mode": new_state.get("Mode")
                }
            })

        # Check for state changes
        if old_state.get("State") != new_state.get("State"):
            self._emit_event({
                "type": "state_change",
                "category": "game_state",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "old_state": old_state.get("State"),
                    "new_state": new_state.get("State")
                }
            })

            # Special events for specific states
            if new_state.get("State") == "Running":
                self._emit_event({
                    "type": "mission_start",
                    "category": "mission",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"mission": new_state.get("Mission", "Unknown")}
                })
            elif new_state.get("State") == "Idle" and old_state.get("State") == "Running":
                self._emit_event({
                    "type": "mission_complete",
                    "category": "mission",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"mission": old_state.get("Mission", "Unknown")}
                })

        # Check for mission changes
        if old_state.get("Mission") != new_state.get("Mission"):
            self._emit_event({
                "type": "mission_change",
                "category": "mission",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "old_mission": old_state.get("Mission"),
                    "new_mission": new_state.get("Mission")
                }
            })

    def _emit_event(self, event: Dict[str, Any]):
        """Emit event to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def connect_websocket(self):
        """Connect via WebSocket for real-time events."""
        try:
            logger.info(f"Attempting WebSocket connection to {self.ws_url}")

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._emit_event({
                        "type": "websocket_message",
                        "category": "realtime",
                        "timestamp": datetime.now().isoformat(),
                        "data": data
                    })
                except Exception as e:
                    logger.error(f"WebSocket message error: {e}")

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")

            def on_open(ws):
                logger.info("WebSocket connection established")
                self._emit_event({
                    "type": "websocket_connected",
                    "category": "system",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"url": self.ws_url}
                })

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )

            # Run WebSocket in a thread
            ws_thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            ws_thread.start()

        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")

    def disconnect(self):
        """Disconnect from the game."""
        self.stop_polling()
        if self.ws:
            self.ws.close()
        logger.info("Disconnected from Starship Horizons")