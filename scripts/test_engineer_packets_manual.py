#!/usr/bin/env python3
"""
Test script to connect as Engineer console and capture hull/systems data
"""

import json
import asyncio
import websockets
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EngineerWebSocket:
    def __init__(self, host="66.68.47.235", port=1865):
        self.host = host
        self.port = port
        self.ws = None
        self.packets_received = {}

    async def connect(self):
        """Connect to WebSocket and identify as Engineer"""
        uri = f"ws://{self.host}:{self.port}/"
        logger.info(f"Connecting to {uri}")

        self.ws = await websockets.connect(uri)
        logger.info("Connected!")

        # Identify as Engineer
        await self.send("IDENTIFY", "Engineer")
        logger.info("Sent IDENTIFY as Engineer")

        # Register for Engineer-specific packets
        engineer_packets = [
            "ROLES", "CONTACTS", "DECKS", "SHIELDS",
            "WEAPONS", "WEAPON-GROUPS", "CARGO-REPAIR",
            "DAMAGE", "DAMAGE-TEAMS", "DAMAGE-PRIORITY",
            "INTEGRITY", "HULL", "SYSTEMS", "COMPONENTS",
            "POWER", "ENERGY", "REACTOR"
        ]

        for packet in engineer_packets:
            await self.send("ACCEPT", packet)
            logger.info(f"Registered for {packet}")

    async def send(self, cmd, value=""):
        """Send command to server"""
        msg = {"cmd": cmd, "value": value}
        await self.ws.send(json.dumps(msg))

    async def receive_messages(self):
        """Receive and log messages"""
        try:
            while True:
                msg = await self.ws.recv()
                data = json.loads(msg)

                cmd = data.get("cmd", "")
                value = data.get("value", "")

                # Track packet types
                if cmd not in self.packets_received:
                    self.packets_received[cmd] = 0
                    logger.info(f"🆕 NEW PACKET TYPE: {cmd}")
                    if value:
                        logger.info(f"   Sample: {str(value)[:200]}...")

                self.packets_received[cmd] += 1

                # Log important packets
                if cmd in ["INTEGRITY", "HULL", "DECKS", "DAMAGE", "SYSTEMS", "COMPONENTS"]:
                    logger.info(f"📊 {cmd}: {value}")
                elif cmd == "SHIELDS":
                    logger.info(f"🛡️ SHIELDS: {value}")
                elif "DAMAGE" in cmd:
                    logger.info(f"💥 {cmd}: {value}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error: {e}")

    async def request_data(self):
        """Periodically request data"""
        while True:
            await asyncio.sleep(5)

            # Try various GET commands
            for cmd in ["INTEGRITY", "HULL", "DECKS", "SYSTEMS", "DAMAGE"]:
                await self.send("GET", cmd)
                logger.info(f"Requested {cmd}")

    async def run(self):
        """Run the test"""
        await self.connect()

        # Run receive and request tasks concurrently
        await asyncio.gather(
            self.receive_messages(),
            self.request_data()
        )

async def main():
    engineer = EngineerWebSocket()

    try:
        await asyncio.wait_for(engineer.run(), timeout=30)
    except asyncio.TimeoutError:
        logger.info("\n" + "="*50)
        logger.info("TEST COMPLETE - Packets received:")
        for packet, count in sorted(engineer.packets_received.items()):
            logger.info(f"  {packet}: {count}")
    finally:
        if engineer.ws:
            await engineer.ws.close()

if __name__ == "__main__":
    asyncio.run(main())