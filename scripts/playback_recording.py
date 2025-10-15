#!/usr/bin/env python3
"""
Playback a mission recording with synchronized audio and game events.
"""

import argparse
import json
import logging
import sys
import time
import wave
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyaudio
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class RecordingPlayback:
    """Playback recorded mission with audio and events."""

    def __init__(self, recording_path: str):
        """
        Initialize playback.

        Args:
            recording_path: Path to recording directory
        """
        self.recording_path = Path(recording_path)
        self.events = []
        self.transcripts = []
        self.audio_segments = []

    def load_recording(self):
        """Load recording data."""
        # Load events
        events_file = self.recording_path / "game_events.json"
        if events_file.exists():
            with open(events_file) as f:
                data = json.load(f)
                self.events = data.get('events', [])
                logger.info(f"✓ Loaded {len(self.events)} game events")

        # Load transcripts
        transcripts_file = self.recording_path / "transcripts.json"
        if transcripts_file.exists():
            with open(transcripts_file) as f:
                data = json.load(f)
                self.transcripts = data.get('transcripts', [])
                logger.info(f"✓ Loaded {len(self.transcripts)} transcripts")

        # Load audio segments
        audio_files = sorted(self.recording_path.glob("*.wav"))
        for audio_file in audio_files:
            self.audio_segments.append(audio_file)
        logger.info(f"✓ Found {len(self.audio_segments)} audio segments")

    def play_audio_segment(self, audio_path: Path):
        """
        Play audio segment.

        Args:
            audio_path: Path to WAV file
        """
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                p = pyaudio.PyAudio()

                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )

                # Play audio
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)

                stream.stop_stream()
                stream.close()
                p.terminate()
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")

    def playback(self, speed: float = 1.0, show_events: bool = True, show_audio: bool = True):
        """
        Playback recording.

        Args:
            speed: Playback speed multiplier
            show_events: Show game events
            show_audio: Play audio segments
        """
        logger.info("\n" + "="*70)
        logger.info("MISSION PLAYBACK")
        logger.info("="*70)

        if not self.events and not self.audio_segments:
            logger.error("No events or audio to playback!")
            return

        # Combine events and transcripts into timeline
        timeline = []

        for event in self.events:
            timeline.append({
                'type': 'event',
                'timestamp': datetime.fromisoformat(event['timestamp']),
                'data': event
            })

        for i, transcript in enumerate(self.transcripts):
            audio_file = self.audio_segments[i] if i < len(self.audio_segments) else None
            timeline.append({
                'type': 'transcript',
                'timestamp': datetime.fromisoformat(transcript['timestamp']),
                'data': transcript,
                'audio_file': audio_file
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])

        if not timeline:
            logger.error("Timeline is empty!")
            return

        # Start playback
        start_time = timeline[0]['timestamp']
        playback_start = time.time()

        for item in timeline:
            # Calculate delay
            real_elapsed = item['timestamp'] - start_time
            delay_seconds = real_elapsed.total_seconds() / speed

            # Wait until it's time to show this item
            while (time.time() - playback_start) < delay_seconds:
                time.sleep(0.01)

            # Display item
            if item['type'] == 'event' and show_events:
                event = item['data']
                timestamp_str = item['timestamp'].strftime("%H:%M:%S.%f")[:-3]
                logger.info(f"[{timestamp_str}] EVENT: {event.get('event_type', 'unknown')} - {event.get('category', '')}")

            elif item['type'] == 'transcript' and show_audio:
                transcript = item['data']
                timestamp_str = item['timestamp'].strftime("%H:%M:%S.%f")[:-3]
                speaker = transcript.get('speaker', 'unknown')
                text = transcript.get('text', '')
                confidence = transcript.get('confidence', 0.0)

                logger.info(f"\n[{timestamp_str}] 🎤 [{speaker}] ({confidence:.0%}): {text}")

                # Play audio if available
                if item.get('audio_file'):
                    self.play_audio_segment(item['audio_file'])

        logger.info("\n" + "="*70)
        logger.info("PLAYBACK COMPLETE")
        logger.info("="*70)


def main():
    """Main playback script."""
    parser = argparse.ArgumentParser(description='Playback mission recording')
    parser.add_argument('recording', help='Path to recording directory')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (default: 1.0)')
    parser.add_argument('--no-events', action='store_true', help='Hide game events')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio playback')

    args = parser.parse_args()

    playback = RecordingPlayback(args.recording)
    playback.load_recording()
    playback.playback(
        speed=args.speed,
        show_events=not args.no_events,
        show_audio=not args.no_audio
    )


if __name__ == "__main__":
    main()
