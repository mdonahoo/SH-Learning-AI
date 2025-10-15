#!/usr/bin/env python3
"""
List all available audio input devices.

Usage:
    python scripts/list_audio_devices.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyaudio
except ImportError:
    print("ERROR: PyAudio not installed")
    print("Install with: pip install pyaudio")
    sys.exit(1)


def list_audio_devices():
    """List all available audio input devices."""
    audio = pyaudio.PyAudio()

    print("\n" + "="*70)
    print("Available Audio Input Devices for Starship Horizons")
    print("="*70)

    input_devices = []

    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))

    if not input_devices:
        print("\n⚠ No input devices found!")
        print("Check your microphone connections and permissions.")
    else:
        for i, info in input_devices:
            print(f"\n{'='*70}")
            print(f"Device Index: {i}")
            print(f"Name: {info['name']}")
            print(f"Max Input Channels: {info['maxInputChannels']}")
            print(f"Default Sample Rate: {int(info['defaultSampleRate'])} Hz")
            host_api = audio.get_host_api_info_by_index(info['hostApi'])
            print(f"Host API: {host_api['name']}")

    print("\n" + "="*70)
    print("Configuration:")
    print("="*70)
    print("\nSet AUDIO_INPUT_DEVICE in .env to use a specific device.")
    print("Example: AUDIO_INPUT_DEVICE=0")
    print("")

    audio.terminate()


if __name__ == "__main__":
    list_audio_devices()
