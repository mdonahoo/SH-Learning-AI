#!/usr/bin/env python3
"""
Test script for real-time audio transcription.

This script tests the audio capture, speaker diarization, and transcription
pipeline in isolation to verify everything is working correctly.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import AudioCaptureManager
from src.audio.whisper_transcriber import WhisperTranscriber
from src.audio.speaker_diarization import SpeakerDiarizer, EngagementAnalyzer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_audio_capture_only(duration: int = 10):
    """Test audio capture without transcription."""
    logger.info("=" * 70)
    logger.info("Testing Audio Capture Only")
    logger.info("=" * 70)

    segment_count = 0

    def on_segment(audio_data, start_time, end_time):
        nonlocal segment_count
        segment_count += 1
        logger.info(f"Segment {segment_count}: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")

    capture = AudioCaptureManager(enable_vad=True)
    capture.set_segment_callback(on_segment)

    logger.info(f"Starting capture for {duration} seconds...")
    logger.info("Speak into your microphone to test VAD segmentation")

    capture.start_capture()
    time.sleep(duration)
    capture.stop_capture()

    logger.info(f"✓ Capture complete - {segment_count} speech segments detected")


def test_speaker_diarization(duration: int = 15):
    """Test speaker identification."""
    logger.info("=" * 70)
    logger.info("Testing Speaker Diarization")
    logger.info("=" * 70)

    diarizer = SpeakerDiarizer()
    speakers_detected = set()

    def on_segment(audio_data, start_time, end_time):
        speaker_id, confidence = diarizer.identify_speaker(audio_data)
        speakers_detected.add(speaker_id)
        logger.info(
            f"Segment: {start_time:.2f}s - {end_time:.2f}s | "
            f"Speaker: {speaker_id} (confidence: {confidence:.2f})"
        )

    capture = AudioCaptureManager(enable_vad=True)
    capture.set_segment_callback(on_segment)

    logger.info(f"Starting capture for {duration} seconds...")
    logger.info("Have multiple people speak to test speaker identification")

    capture.start_capture()
    time.sleep(duration)
    capture.stop_capture()

    logger.info(f"✓ Diarization complete - {len(speakers_detected)} unique speakers detected")
    logger.info(f"Speakers: {', '.join(sorted(speakers_detected))}")


def test_full_transcription(duration: int = 30):
    """Test complete audio transcription pipeline."""
    logger.info("=" * 70)
    logger.info("Testing Full Audio Transcription Pipeline")
    logger.info("=" * 70)

    # Initialize components
    logger.info("Initializing components...")
    transcriber = WhisperTranscriber()
    diarizer = SpeakerDiarizer()
    engagement = EngagementAnalyzer()

    transcripts = []

    def on_segment(audio_data, start_time, end_time):
        # Identify speaker
        speaker_id, confidence = diarizer.identify_speaker(audio_data)

        # Queue for transcription
        transcriber.queue_audio(
            audio_data,
            start_time,
            speaker_id=speaker_id
        )

        logger.info(
            f"Queued segment: {start_time:.2f}s | "
            f"Speaker: {speaker_id} (confidence: {confidence:.2f})"
        )

    capture = AudioCaptureManager(enable_vad=True)
    capture.set_segment_callback(on_segment)

    logger.info(f"Starting {duration} second recording with transcription...")
    logger.info("Speak clearly into your microphone")

    # Start transcription workers
    transcriber.start_workers()

    # Start capture
    capture.start_capture()

    # Monitor for results
    start_time = time.time()
    while time.time() - start_time < duration:
        # Check for transcription results
        results = transcriber.get_results()
        for result in results:
            transcripts.append(result)
            logger.info(
                f"📝 [{result['speaker_id']}] ({result['confidence']:.2f}): "
                f"{result['text']}"
            )
        time.sleep(0.5)

    # Stop capture
    capture.stop_capture()

    # Get any remaining results
    time.sleep(2)
    results = transcriber.get_results()
    transcripts.extend(results)
    for result in results:
        logger.info(
            f"📝 [{result['speaker_id']}] ({result['confidence']:.2f}): "
            f"{result['text']}"
        )

    # Stop transcription
    transcriber.stop_workers()

    # Calculate engagement metrics
    if transcripts:
        segments = [
            {
                'speaker_id': t['speaker_id'],
                'start_time': t['timestamp'],
                'end_time': t['timestamp'] + t['duration'],
                'text': t['text']
            }
            for t in transcripts
        ]
        metrics = engagement.calculate_engagement_metrics(segments)

        logger.info("=" * 70)
        logger.info("Engagement Metrics:")
        logger.info("=" * 70)
        logger.info(f"Total utterances: {metrics['total_utterances']}")
        logger.info(f"Unique speakers: {metrics['unique_speakers']}")
        logger.info(f"Turn-taking rate: {metrics['turn_taking_rate']:.2f} turns/min")
        logger.info(f"Average response time: {metrics['avg_response_time']:.2f}s")

        logger.info("\nSpeaker participation:")
        for speaker_id, stats in metrics['speaker_stats'].items():
            logger.info(
                f"  {speaker_id}: {stats['utterances']} utterances, "
                f"{stats['total_time']:.1f}s speaking time"
            )

    logger.info("=" * 70)
    logger.info(f"✓ Test complete - {len(transcripts)} transcripts generated")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test real-time audio transcription')
    parser.add_argument(
        '--mode',
        choices=['capture', 'diarization', 'full'],
        default='full',
        help='Test mode (default: full)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Test duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )

    args = parser.parse_args()

    # Check if audio is enabled
    if os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() != 'true':
        logger.warning("ENABLE_AUDIO_CAPTURE is not set to 'true' in .env")
        logger.warning("Audio capture may not work correctly")

    # List devices if requested
    if args.list_devices:
        from src.audio.capture import list_audio_devices
        list_audio_devices()
        return

    try:
        if args.mode == 'capture':
            test_audio_capture_only(args.duration)
        elif args.mode == 'diarization':
            test_speaker_diarization(args.duration)
        elif args.mode == 'full':
            test_full_transcription(args.duration)

        logger.info("\n✓ All tests completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
