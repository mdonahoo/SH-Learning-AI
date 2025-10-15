#!/usr/bin/env python3
"""
Test script to connect as Tactical console and capture vessel data
"""

import json
import asyncio
import websockets
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TacticalWebSocket:
    def __init__(self, host="66.68.47.235", port=1865):
        self.host = host
        self.port = port
        self.ws = None
        self.packets_received = {}
        self.vessel_data = {}
        self.vessel_values = None

    async def connect(self):
        """Connect to WebSocket and identify as Tactical"""
        uri = f"ws://{self.host}:{self.port}/"
        logger.info(f"Connecting to {uri}")

        self.ws = await websockets.connect(uri)
        logger.info("Connected!")

        # Identify as Tactical console
        identify_data = {
            "ScreenName": "tactical",
            "Location": "Bridge",
            "ServerID": "",
            "Guid": "test-tactical-123",
            "IsMainViewer": True,
            "UserInfo": {
                "Name": "Tactical Test",
                "CallSign": "TACTICAL"
            }
        }
        await self.send("IDENTIFY", json.dumps(identify_data))
        logger.info("Sent IDENTIFY as tactical")

        # Register for tactical-specific packets
        tactical_packets = [
            "VESSEL", "VESSEL-ID", "VESSEL-VALUES",
            "SHIELDS", "HULL", "INTEGRITY", "DAMAGE",
            "WEAPONS", "WEAPON-GROUPS", "TARGET-TACTICAL",
            "CONTACTS", "ALERT", "STATUS", "BC"
        ]

        for packet in tactical_packets:
            await self.send("ACCEPT-PACKET", packet)
            logger.debug(f"Registered for {packet}")

    async def send(self, cmd, value=""):
        """Send command to server"""
        msg = {"Cmd": cmd, "Value": value}
        await self.ws.send(json.dumps(msg))

    async def receive_messages(self):
        """Receive and log messages"""
        try:
            while True:
                msg = await self.ws.recv()
                data = json.loads(msg)

                cmd = data.get("Cmd", "")
                value = data.get("Value", "")

                # Track packet types
                if cmd not in self.packets_received:
                    self.packets_received[cmd] = 0
                    logger.info(f"🆕 NEW PACKET TYPE: {cmd}")
                    if value:
                        logger.info(f"   Sample: {str(value)[:500]}...")

                self.packets_received[cmd] += 1

                # Log important packets
                if cmd == "VESSEL":
                    logger.info(f"🚢 VESSEL DATA RECEIVED!")
                    logger.info(f"   Keys: {list(value.keys()) if isinstance(value, dict) else 'Not a dict'}")

                    # Look for hull/shield data
                    if isinstance(value, dict):
                        if 'Integrity' in value:
                            logger.info(f"   ✅ Integrity: {value['Integrity']}")
                        if 'MaxIntegrity' in value:
                            logger.info(f"   ✅ MaxIntegrity: {value['MaxIntegrity']}")
                        if 'Shields' in value:
                            logger.info(f"   ✅ Shields: {value['Shields']}")
                        if 'Hull' in value:
                            logger.info(f"   ✅ Hull: {value['Hull']}")

                    self.vessel_data = value

                elif cmd == "VESSEL-VALUES":
                    logger.info(f"📊 VESSEL-VALUES: {value}")
                    self.vessel_values = value

                elif cmd == "VESSEL-ID":
                    logger.info(f"🆔 Vessel ID: {value}")
                    # Request vessel data after getting ID
                    await self.send("GET", "VESSEL")
                    await self.send("GET", "VESSEL-VALUES")

                elif cmd in ["INTEGRITY", "HULL", "SHIELDS"]:
                    logger.info(f"💚 {cmd}: {value}")

                elif cmd == "DAMAGE":
                    logger.info(f"💥 DAMAGE: {value}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error: {e}")

    async def periodic_requests(self):
        """Periodically request vessel data"""
        await asyncio.sleep(2)  # Wait for initial connection

        while True:
            await asyncio.sleep(5)

            # Request vessel data
            logger.info("📡 Requesting vessel data...")
            await self.send("GET", "VESSEL")
            await self.send("GET", "VESSEL-VALUES")
            await self.send("GET", "SHIELDS")
            await self.send("GET", "HULL")
            await self.send("GET", "INTEGRITY")

    async def run(self):
        """Run the test"""
        await self.connect()

        # Run receive and request tasks concurrently
        await asyncio.gather(
            self.receive_messages(),
            self.periodic_requests()
        )

async def main():
    tactical = TacticalWebSocket()

    try:
        await asyncio.wait_for(tactical.run(), timeout=30)
    except asyncio.TimeoutError:
        logger.info("\n" + "="*50)
        logger.info("TEST COMPLETE - Packets received:")
        for packet, count in sorted(tactical.packets_received.items()):
            logger.info(f"  {packet}: {count}")

        if tactical.vessel_data:
            logger.info("\n🚢 Final VESSEL data keys:")
            logger.info(f"  {list(tactical.vessel_data.keys())}")

        if tactical.vessel_values:
            logger.info("\n📊 Final VESSEL-VALUES:")
            logger.info(f"  {tactical.vessel_values}")

    finally:
        if tactical.ws:
            await tactical.ws.close()

if __name__ == "__main__":
    asyncio.run(main())