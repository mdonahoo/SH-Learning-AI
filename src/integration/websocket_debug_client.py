#!/usr/bin/env python3
"""
WebSocket Debug Client for Starship Horizons
Enhanced debugging and packet request capabilities.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
import websocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class WebSocketDebugClient:
    """Debug WebSocket client to understand packet flow."""

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
        self.message_log = []
        self.raw_log = []

    def connect_and_debug(self):
        """Connect with full debugging."""
        try:
            logger.info(f"Connecting to {self.ws_url}")

            def on_message(ws, message):
                timestamp = datetime.now().isoformat()
                logger.info(f"[RECEIVED] {message[:200]}")
                self.message_log.append({"time": timestamp, "type": "received", "data": message})
                self.raw_log.append(message)

            def on_error(ws, error):
                logger.error(f"[ERROR] {error}")
                self.connected = False

            def on_close(ws, close_code, close_msg):
                logger.info(f"[CLOSED] Code: {close_code}, Msg: {close_msg}")
                self.connected = False

            def on_open(ws):
                logger.info("[CONNECTED] WebSocket opened")
                self.connected = True

                # Try different identification formats
                self.send_identification_packets(ws)

                # Request data
                self.request_telemetry_data(ws)

            # Enable debug trace
            websocket.enableTrace(True)

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
                header={
                    "User-Agent": "Mozilla/5.0 Starship Horizons Client",
                    "Origin": f"http://{self.host}:1864"
                }
            )

            # Run in thread
            ws_thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            ws_thread.start()

            # Monitor for 30 seconds
            logger.info("Monitoring for 30 seconds...")
            for i in range(30):
                if i % 5 == 0:
                    logger.info(f"  {30-i} seconds remaining... Messages received: {len(self.message_log)}")

                    # Try sending requests periodically
                    if self.connected and i > 0:
                        self.send_data_requests()

                time.sleep(1)

            # Close and report
            if self.ws:
                self.ws.close()

            self.print_summary()

        except Exception as e:
            logger.error(f"Connection failed: {e}")

    def send_identification_packets(self, ws):
        """Send various identification packet formats."""
        identifications = [
            # Format 1: Simple identification
            {
                "cmd": "IDENTIFY",
                "Type": "Client",
                "Version": "41"
            },
            # Format 2: Full identification like game
            {
                "cmd": "IDENTIFY",
                "data": {
                    "Type": "MainViewer",
                    "Version": "41",
                    "Guid": f"debug_{datetime.now().timestamp()}",
                    "IsMainViewer": True,
                    "UserInfo": {"Name": "Debug"}
                }
            },
            # Format 3: Role-based
            {
                "cmd": "ROLE",
                "Role": "Helm",
                "Station": "Bridge"
            }
        ]

        for ident in identifications:
            try:
                msg = json.dumps(ident)
                ws.send(msg)
                logger.info(f"[SENT] {msg}")
                self.message_log.append({"time": datetime.now().isoformat(), "type": "sent", "data": msg})
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Send failed: {e}")

    def request_telemetry_data(self, ws):
        """Request various telemetry data types."""
        requests = [
            {"cmd": "GET", "type": "VESSEL"},
            {"cmd": "GET", "type": "STATUS"},
            {"cmd": "GET", "type": "SHIELDS"},
            {"cmd": "GET", "type": "SYSTEMS"},
            {"cmd": "REQUEST", "data": "VESSEL"},
            {"cmd": "SUBSCRIBE", "type": "ALL"},
            {"cmd": "SUBSCRIBE", "type": "VESSEL"},
            {"cmd": "BATCH-REQUEST", "types": ["VESSEL", "STATUS", "DAMAGE"]},
            "VESSEL",  # Try plain text
            "STATUS"
        ]

        for req in requests:
            try:
                if isinstance(req, str):
                    msg = req
                else:
                    msg = json.dumps(req)

                ws.send(msg)
                logger.info(f"[SENT REQUEST] {msg}")
                self.message_log.append({"time": datetime.now().isoformat(), "type": "sent", "data": msg})
                time.sleep(0.2)
            except Exception as e:
                logger.debug(f"Request failed: {e}")

    def send_data_requests(self):
        """Send periodic data requests."""
        if not self.ws or not self.connected:
            return

        periodic_requests = [
            {"cmd": "PING"},
            {"cmd": "STATUS"},
            {"cmd": "UPDATE"}
        ]

        for req in periodic_requests:
            try:
                msg = json.dumps(req)
                self.ws.send(msg)
                logger.debug(f"[PERIODIC] {msg}")
            except:
                pass

    def print_summary(self):
        """Print summary of captured data."""
        logger.info("\n" + "="*60)
        logger.info("WEBSOCKET DEBUG SUMMARY")
        logger.info("="*60)

        logger.info(f"\nTotal messages: {len(self.message_log)}")

        sent_count = sum(1 for m in self.message_log if m["type"] == "sent")
        received_count = sum(1 for m in self.message_log if m["type"] == "received")

        logger.info(f"  Sent: {sent_count}")
        logger.info(f"  Received: {received_count}")

        if self.raw_log:
            logger.info("\nReceived message samples:")
            for i, msg in enumerate(self.raw_log[:5]):
                logger.info(f"  [{i+1}] {msg[:100]}...")
        else:
            logger.info("\nNo messages received from server")
            logger.info("Possible issues:")
            logger.info("  - Wrong port (try 1864, 1866, 1867)")
            logger.info("  - Need to be in-game as a player")
            logger.info("  - Different protocol (not WebSocket)")
            logger.info("  - Need authentication token")


def test_all_ports():
    """Test multiple ports to find the right one."""
    ports = [1864, 1865, 1866, 1867]

    for port in ports:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing port {port}")
        logger.info("="*60)

        client = WebSocketDebugClient(port=port)
        client.connect_and_debug()

        if client.raw_log:
            logger.info(f"✅ Port {port} received data!")
            return port

    logger.info("\n❌ No ports received data")
    return None


if __name__ == "__main__":
    # Test single port
    client = WebSocketDebugClient(port=1865)
    client.connect_and_debug()

    # Uncomment to test all ports
    # test_all_ports()