#!/usr/bin/env python3
"""
Simple test to check for VESSEL packet
"""

import json
import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vessel():
    uri = "ws://66.68.47.235:1865/"

    async with websockets.connect(uri) as ws:
        logger.info("Connected!")

        # Identify
        await ws.send(json.dumps({
            "Cmd": "IDENTIFY",
            "Value": json.dumps({
                "ScreenName": "test",
                "Location": "Bridge",
                "ServerID": "",
                "Guid": "test-123",
                "IsMainViewer": False,
                "UserInfo": {"Name": "Test", "CallSign": "TEST"}
            })
        }))

        # Accept VESSEL packet
        await ws.send(json.dumps({"Cmd": "ACCEPT-PACKET", "Value": "VESSEL"}))
        await ws.send(json.dumps({"Cmd": "ACCEPT-PACKET", "Value": "VESSEL-ID"}))
        await ws.send(json.dumps({"Cmd": "ACCEPT-PACKET", "Value": "VESSEL-VALUES"}))

        vessel_id = None
        vessel_data_received = False

        # Listen for 20 seconds
        timeout = asyncio.create_task(asyncio.sleep(20))

        while not timeout.done():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                data = json.loads(msg)
                cmd = data.get("Cmd", "")
                value = data.get("Value", "")

                if cmd == "VESSEL-ID":
                    vessel_id = value
                    logger.info(f"Got VESSEL-ID: {vessel_id}")
                    # Request VESSEL data
                    await ws.send(json.dumps({"Cmd": "GET", "Value": "VESSEL"}))
                    logger.info("Sent GET VESSEL")

                elif cmd == "VESSEL":
                    vessel_data_received = True
                    logger.info(f"🎉 VESSEL packet received!")
                    if isinstance(value, dict):
                        logger.info(f"Keys: {list(value.keys())[:10]}...")
                        if 'Integrity' in value:
                            logger.info(f"✅ Has Integrity: {value['Integrity']}")
                        if 'MaxIntegrity' in value:
                            logger.info(f"✅ Has MaxIntegrity: {value['MaxIntegrity']}")
                        if 'Shields' in value:
                            logger.info(f"✅ Has Shields: {value['Shields']}")

                elif cmd == "VESSEL-VALUES":
                    logger.info(f"Got VESSEL-VALUES: {value}")

            except asyncio.TimeoutError:
                continue

        if not vessel_data_received:
            logger.warning("❌ No VESSEL packet received in 20 seconds")
            logger.info("Try moving the ship or taking damage to trigger VESSEL updates")

asyncio.run(test_vessel())