#!/usr/bin/env python3
"""Quick telemetry capture test."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.integration.browser_mimic_websocket import BrowserMimicWebSocket

def main():
    host = os.getenv('GAME_HOST', '192.168.68.56')
    port = int(os.getenv('GAME_PORT_WS', '1865'))

    client = BrowserMimicWebSocket(host=host, port=port)

    if not client.connect(screen_name='captain', is_main_viewer=True):
        print('Failed to connect')
        sys.exit(1)

    print('Connected! Capturing for 8 seconds...')
    time.sleep(5)
    client.request_vessel_data()
    time.sleep(3)

    print()
    print('='*60)
    print('TELEMETRY CAPTURE RESULTS')
    print('='*60)
    print(f'Total packet types: {len(client.packet_counts)}')
    print(f'Total packets: {sum(client.packet_counts.values())}')

    print()
    print('Packet types by count:')
    for pkt, count in sorted(client.packet_counts.items(), key=lambda x: -x[1]):
        print(f'  {pkt:30} : {count:4}')

    print()
    print('='*60)
    print('VESSEL DATA')
    print('='*60)
    vd = client.vessel_data
    for k, v in vd.items():
        if v is not None and v != {} and v != []:
            val_str = str(v)[:60] if not isinstance(v, (int, float, bool)) else str(v)
            print(f'  {k:25} : {val_str}')

    print()
    print('='*60)
    print('COMBAT DATA')
    print('='*60)
    cd = client.combat_enhanced
    for k, v in cd.items():
        if v is not None and v != {} and v != []:
            val_str = str(v)[:60] if not isinstance(v, (int, float, bool)) else str(v)
            print(f'  {k:25} : {val_str}')

    # Show last_packets to see what specific data we captured
    print()
    print('='*60)
    print('CONTACTS SAMPLE')
    print('='*60)
    contacts = client.last_packets.get('CONTACTS', [])
    if isinstance(contacts, list) and len(contacts) > 0:
        print(f'Total contacts: {len(contacts)}')
        # Show first few contacts
        for c in contacts[:3]:
            if isinstance(c, dict):
                print(f"  - {c.get('Name', 'Unknown')}: {c.get('ClassType', '')} at range {c.get('Range', 'N/A')}")
    else:
        print('No contacts data')

    client.disconnect()
    print()
    print('Disconnected.')

if __name__ == '__main__':
    main()
