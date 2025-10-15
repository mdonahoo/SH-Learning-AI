#!/usr/bin/env python3
"""
Test identifying as specific console role to get vessel assignment
"""

import json
import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_console_role(console_name):
    uri = "ws://66.68.47.235:1865/"

    async with websockets.connect(uri) as ws:
        logger.info(f"Connected as {console_name}!")

        # Identify as specific console
        identify_data = {
            "ScreenName": console_name.lower(),
            "Location": "Bridge",
            "ServerID": "",
            "Guid": f"test-{console_name}-456",
            "IsMainViewer": True,  # Set as main viewer
            "UserInfo": {
                "Name": f"{console_name} Test",
                "CallSign": console_name.upper()
            }
        }

        await ws.send(json.dumps({
            "Cmd": "IDENTIFY",
            "Value": json.dumps(identify_data)
        }))

        # Accept packets
        packet_types = [
            "VESSEL", "VESSEL-ID", "VESSEL-VALUES", "VESSEL-NO",
            "SHIELDS", "HULL", "INTEGRITY", "DAMAGE",
            "CONTACTS", "ROLES", "PLAYERS", "BC"
        ]

        for ptype in packet_types:
            await ws.send(json.dumps({"Cmd": "ACCEPT-PACKET", "Value": ptype}))

        # Also try requesting vessel assignment
        await ws.send(json.dumps({"Cmd": "REQUEST-VESSEL", "Value": ""}))

        vessel_info = {"id": None, "no": None}
        packets_received = {}

        # Listen for 15 seconds
        timeout = asyncio.create_task(asyncio.sleep(15))

        while not timeout.done():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                data = json.loads(msg)
                cmd = data.get("Cmd", "")
                value = data.get("Value", "")

                if cmd not in packets_received:
                    packets_received[cmd] = 0
                    logger.info(f"📦 New packet: {cmd}")

                packets_received[cmd] += 1

                # Track vessel assignment
                if cmd == "VESSEL-ID":
                    vessel_info["id"] = value
                    logger.info(f"✅ Got VESSEL-ID: {value}")
                    # Request vessel data immediately
                    await ws.send(json.dumps({"Cmd": "GET", "Value": "VESSEL"}))

                elif cmd == "VESSEL-NO":
                    vessel_info["no"] = value
                    logger.info(f"✅ Got VESSEL-NO: {value}")

                elif cmd == "VESSEL":
                    logger.info(f"🎉 VESSEL packet received!")
                    if isinstance(value, dict):
                        if "Integrity" in value or "Shields" in value:
                            logger.info(f"   Has hull/shield data!")

                elif cmd == "PLAYERS":
                    # Check if we're in the players list
                    if isinstance(value, list):
                        for player in value:
                            if console_name.upper() in str(player):
                                logger.info(f"👤 Found us in players: VesselID={player.get('VesselID')}")

            except asyncio.TimeoutError:
                continue

        # Summary
        logger.info(f"\nSummary for {console_name}:")
        logger.info(f"  Vessel ID: {vessel_info['id']}")
        logger.info(f"  Vessel No: {vessel_info['no']}")
        logger.info(f"  Packets: {', '.join(packets_received.keys())}")

# Test different console roles
async def main():
    consoles = ["tactical", "helm", "science", "engineer", "captain"]

    for console in consoles:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {console} console...")
        await test_console_role(console)

asyncio.run(main())