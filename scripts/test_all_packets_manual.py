#!/usr/bin/env python3
"""
Test to capture ALL packets and find hull/shield data
"""

import json
import asyncio
import websockets
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_all_packets():
    uri = "ws://66.68.47.235:1865/"
    packets_seen = {}

    async with websockets.connect(uri) as ws:
        logger.info("Connected! Listening for ALL packets...")

        # Identify
        await ws.send(json.dumps({
            "Cmd": "IDENTIFY",
            "Value": json.dumps({
                "ScreenName": "test",
                "Location": "Bridge",
                "ServerID": "",
                "Guid": "test-all-123",
                "IsMainViewer": False,
                "UserInfo": {"Name": "Test All", "CallSign": "TESTALL"}
            })
        }))

        # Accept many packet types
        packet_types = [
            "VESSEL", "VESSEL-ID", "VESSEL-VALUES", "VESSEL-COMPONENTS",
            "SHIELDS", "HULL", "INTEGRITY", "DAMAGE", "STATUS",
            "ALERT", "BC", "CONTACTS", "SYSTEMS", "POWER", "ENERGY",
            "VARIABLES", "MISSION", "PLAYER-VESSELS"
        ]

        for ptype in packet_types:
            await ws.send(json.dumps({"Cmd": "ACCEPT-PACKET", "Value": ptype}))

        vessel_id = None
        start_time = datetime.now()

        # Listen for 30 seconds
        timeout = asyncio.create_task(asyncio.sleep(30))

        while not timeout.done():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                data = json.loads(msg)
                cmd = data.get("Cmd", "")
                value = data.get("Value", "")

                # Track packet types
                if cmd not in packets_seen:
                    packets_seen[cmd] = {"count": 0, "has_hull_data": False, "sample": None}
                    logger.info(f"🆕 NEW PACKET: {cmd}")

                packets_seen[cmd]["count"] += 1

                # Check for hull/shield related data in ANY packet
                if value and isinstance(value, dict):
                    hull_keys = ["Integrity", "MaxIntegrity", "Hull", "HullIntegrity",
                                 "Shields", "ShieldLevel", "Shield"]
                    found_keys = [k for k in hull_keys if k in value]

                    if found_keys:
                        packets_seen[cmd]["has_hull_data"] = True
                        packets_seen[cmd]["sample"] = {k: value[k] for k in found_keys}
                        logger.info(f"✅ {cmd} contains hull/shield data: {found_keys}")
                        logger.info(f"   Values: {packets_seen[cmd]['sample']}")

                # Special handling for specific packets
                if cmd == "VESSEL-ID":
                    vessel_id = value
                    logger.info(f"Got VESSEL-ID: {vessel_id}")
                    # Try multiple GET commands
                    for get_cmd in ["VESSEL", "STATUS", "SYSTEMS", "HULL", "SHIELDS"]:
                        await ws.send(json.dumps({"Cmd": "GET", "Value": get_cmd}))
                    logger.info("Sent multiple GET requests")

                elif cmd == "VARIABLES" and value:
                    # Check if variables contain hull/shield data
                    for var_name, var_value in value.items():
                        if any(keyword in var_name.lower() for keyword in ["hull", "shield", "integrity"]):
                            logger.info(f"📊 Variable {var_name} = {var_value}")

                elif cmd == "BC":  # Batch command - might contain multiple updates
                    logger.info(f"Batch command received: {str(value)[:200]}...")

            except asyncio.TimeoutError:
                continue

        # Summary
        logger.info("\n" + "="*50)
        logger.info(f"TEST COMPLETE - Ran for {(datetime.now() - start_time).seconds} seconds")
        logger.info("\nPackets received summary:")

        hull_packets = []
        for cmd, info in sorted(packets_seen.items()):
            marker = "✅" if info["has_hull_data"] else "  "
            logger.info(f"{marker} {cmd}: {info['count']} times")
            if info["has_hull_data"]:
                hull_packets.append(cmd)

        if hull_packets:
            logger.info(f"\n🎯 Packets with hull/shield data: {', '.join(hull_packets)}")
        else:
            logger.info("\n❌ No packets contained hull/shield data")
            logger.info("Try moving the ship, changing speed, or taking damage")

asyncio.run(test_all_packets())