#!/usr/bin/env python3
"""
Test script for sub-segment diarization fix.

Tests the new multi-speaker segment splitting on the problematic recording.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pydub import AudioSegment

from src.audio.batch_diarizer import BatchSpeakerDiarizer, is_batch_diarizer_available

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_audio(audio_path: str) -> tuple:
    """Load audio file and return samples and sample rate."""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples, audio.frame_rate


def main():
    """Run sub-segment diarization test."""
    recording_path = Path("data/recordings/recording_20260120_214001.wav")

    if not recording_path.exists():
        logger.error(f"Recording not found: {recording_path}")
        return 1

    if not is_batch_diarizer_available():
        logger.error("Batch diarizer dependencies not available")
        return 1

    logger.info(f"Loading audio: {recording_path}")
    samples, sample_rate = load_audio(str(recording_path))
    logger.info(f"Audio loaded: {len(samples)/sample_rate:.1f}s at {sample_rate}Hz")

    # Create mock segments matching the original problematic transcription
    # The first segment is the problematic one: 2.8s to 33.05s with multiple speakers
    mock_segments = [
        {
            "start": 2.8,
            "end": 33.05,
            "text": "All stations, report readiness. Helm ready captain all maneuvering thrusters online warp drive standing by Tactical systems green across the board shields at 100%, weapons and systems armed and ready Long range sensors... Engineering reports all systems nominal power distribution optimal reactor running at 98% efficiency Operations Online Captain",
            "confidence": 0.31
        },
        {
            "start": 33.05,
            "end": 36.55,
            "text": "Captain, all channels clear. Communication protocols active.",
            "confidence": 0.27
        },
        {
            "start": 71.02,
            "end": 83.62,
            "text": "Recon stations to high, captain.",
            "confidence": 0.27
        },
        {
            "start": 84.64,
            "end": 98.41,
            "text": "Transmitting now. Space dock acknowledges we are cleared for undocked and laid in,",
            "confidence": 0.27
        },
        {
            "start": 98.51,
            "end": 100.81,
            "text": "captain. Ready to engage on your command.",
            "confidence": 0.27
        },
        {
            "start": 104.72,
            "end": 107.72,
            "text": "Stable energy flow balance. We're good to go, captain!",
            "confidence": 0.27
        },
        {
            "start": 108.72,
            "end": 111.12,
            "text": "Aye sir. Engaging warp dive",
            "confidence": 0.27
        },
        {
            "start": 111.12,
            "end": 114.54,
            "text": "Now ETA to first waypoint one minute",
            "confidence": 0.27
        },
        {
            "start": 149.1,
            "end": 151.12,
            "text": "reducing to impulse on your command",
            "confidence": 0.30
        }
    ]

    logger.info(f"Input segments: {len(mock_segments)}")
    for i, seg in enumerate(mock_segments):
        duration = seg['end'] - seg['start']
        logger.info(f"  [{i}] {seg['start']:.1f}s-{seg['end']:.1f}s ({duration:.1f}s): {seg['text'][:50]}...")

    # Initialize diarizer with debug-friendly settings
    logger.info("\nInitializing BatchSpeakerDiarizer...")
    diarizer = BatchSpeakerDiarizer(
        similarity_threshold=0.80,
        min_speakers=1,
        max_speakers=8
    )

    # Run diarization
    logger.info("\nRunning diarization with sub-segment splitting...")
    segments_out, result = diarizer.diarize_complete(samples, mock_segments, sample_rate)

    # Report results
    logger.info(f"\nOutput segments: {len(segments_out)}")
    logger.info(f"Total speakers detected: {result.total_speakers}")
    logger.info(f"Methodology: {result.methodology_note}")

    logger.info("\nSegment assignments:")
    for i, seg in enumerate(segments_out):
        speaker = seg.get('speaker_id', 'unknown')
        confidence = seg.get('speaker_confidence', 0)
        is_split = seg.get('_is_split', False)
        split_marker = " [SPLIT]" if is_split else ""
        text = seg.get('text', '')[:50]
        logger.info(
            f"  [{i}] {seg['start']:.1f}s-{seg['end']:.1f}s -> {speaker} "
            f"(conf={confidence:.2f}){split_marker}: {text}..."
        )

    # Check if the first long segment was split
    first_seg_duration = mock_segments[0]['end'] - mock_segments[0]['start']
    logger.info(f"\n--- Validation ---")
    logger.info(f"Original first segment duration: {first_seg_duration:.1f}s")

    if len(segments_out) > len(mock_segments):
        logger.info(f"SUCCESS: Segments were split ({len(mock_segments)} -> {len(segments_out)})")
    else:
        logger.warning(f"No segment splitting occurred")

    # Count unique speakers for the first segment's time range
    first_seg_speakers = set()
    for seg in segments_out:
        if seg['start'] < 33.05:  # Within first original segment
            first_seg_speakers.add(seg.get('speaker_id'))

    if len(first_seg_speakers) > 1:
        logger.info(f"SUCCESS: Multiple speakers detected in first segment: {first_seg_speakers}")
    else:
        logger.warning(f"Only one speaker detected in first segment: {first_seg_speakers}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
