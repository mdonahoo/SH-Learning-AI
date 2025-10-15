#!/usr/bin/env python3
"""
Test connecting as tactical console to get vessel data
"""

import json
import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tactical():
    uri = "ws://66.68.47.235:1865/"

    async with websockets.connect(uri) as ws:
        logger.info("Connected as tactical console!")

        # Identify as tactical
        identify_data = {
            "ScreenName": "tactical",
            "Location": "Bridge",
            "ServerID": "",
            "Guid": "test-tactical-789",
            "IsMainViewer": True,
            "UserInfo": {
                "Name": "Tactical Test",
                "CallSign": "TACTICAL"
            }
        }

        await ws.send(json.dumps({
            "Cmd": "IDENTIFY",
            "Value": json.dumps(identify_data)
        }))

        # Accept packets
        packet_types = [
            "VESSEL", "VESSEL-ID", "VESSEL-VALUES",
            "SHIELDS", "HULL", "INTEGRITY", "DAMAGE",
            "BATCH", "BC", "ALERT", "STATUS"
        ]

        for ptype in packet_types:
            await ws.send(json.dumps({"Cmd": "ACCEPT-PACKET", "Value": ptype}))

        vessel_id = None
        vessel_received = False
        batch_data = []

        # Listen for 20 seconds
        logger.info("Listening for packets... (20 seconds)")
        logger.info("Try moving the ship or changing systems to trigger updates!")

        timeout = asyncio.create_task(asyncio.sleep(20))

        while not timeout.done():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                data = json.loads(msg)
                cmd = data.get("Cmd", "")
                value = data.get("Value", "")

                if cmd == "VESSEL-ID":
                    vessel_id = value
                    logger.info(f"✅ Got VESSEL-ID: {vessel_id}")
                    # Request vessel data
                    await ws.send(json.dumps({"Cmd": "GET", "Value": "VESSEL"}))
                    await ws.send(json.dumps({"Cmd": "GET", "Value": "STATUS"}))

                elif cmd == "VESSEL":
                    vessel_received = True
                    logger.info(f"🎉 VESSEL packet received!")
                    if isinstance(value, dict):
                        # Check for hull/shield data
                        hull_keys = ["Integrity", "MaxIntegrity", "HealIntegrity",
                                    "Shields", "Shield", "ShieldLevel"]
                        found = {k: value[k] for k in hull_keys if k in value}
                        if found:
                            logger.info(f"✅ Hull/Shield data found:")
                            for k, v in found.items():
                                logger.info(f"   {k}: {v}")
                        else:
                            logger.info(f"   Available keys: {list(value.keys())[:10]}...")

                elif cmd == "BATCH" or cmd == "BC":
                    batch_data.append(value)
                    # Parse batch contents - BATCH contains direct data like {'VESSEL': {...}}
                    if isinstance(value, dict):
                        if 'VESSEL' in value:
                            vessel_data = value['VESSEL']
                            vessel_received = True
                            logger.info(f"🎉 VESSEL found in BATCH!")
                            if isinstance(vessel_data, dict):
                                hull_keys = ["Integrity", "MaxIntegrity", "HealIntegrity",
                                            "Shields", "Shield", "ShieldLevel"]
                                found = {k: vessel_data[k] for k in hull_keys if k in vessel_data}
                                if found:
                                    logger.info(f"✅ Hull/Shield data found in BATCH:")
                                    for k, v in found.items():
                                        logger.info(f"   {k}: {v}")
                                else:
                                    # Log available keys to find hull/shield fields
                                    logger.info(f"   Available keys: {list(vessel_data.keys())[:15]}...")
                    elif isinstance(value, list):
                        # Handle list format
                        for item in value:
                            if isinstance(item, dict) and 'VESSEL' in item:
                                vessel_received = True
                                logger.info(f"🎉 VESSEL found in BATCH list!")

                elif cmd == "STATUS":
                    logger.info(f"📊 STATUS: {value}")

                elif cmd == "PING":
                    # Respond to ping
                    await ws.send(json.dumps({"Cmd": "PONG", "Value": ""}))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error: {e}")
                break

        # Summary
        logger.info("\n" + "="*50)
        logger.info("TEST COMPLETE")
        logger.info(f"  Vessel ID: {vessel_id}")
        logger.info(f"  VESSEL packet received: {vessel_received}")
        logger.info(f"  BATCH packets received: {len(batch_data)}")

        if not vessel_received:
            logger.info("\n❌ No VESSEL packet received")
            logger.info("The ship needs to change state (move, take damage, change systems)")

asyncio.run(test_tactical())