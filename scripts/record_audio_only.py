#!/usr/bin/env python3
"""
Record bridge crew audio without game telemetry connection.

This script captures and transcribes audio from the bridge crew
independently of the game server connection.
"""

# Suppress noisy warnings from pyannote/pytorch before any imports
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*loss_func.*")
warnings.filterwarnings("ignore", message=".*task-dependent loss.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.*")

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import AudioCaptureManager
from src.audio.whisper_transcriber import WhisperTranscriber
from src.audio.speaker_diarization import SpeakerDiarizer, EngagementAnalyzer, SpeakerSegment

# Conditionally import neural diarization
try:
    from src.audio.neural_diarization import NeuralSpeakerDiarizer, PYANNOTE_AVAILABLE
except ImportError:
    PYANNOTE_AVAILABLE = False
    NeuralSpeakerDiarizer = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioOnlyRecorder:
    """
    Standalone audio recorder with transcription.

    Records bridge crew audio, performs speaker diarization,
    and generates transcripts without game server connection.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        session_name: Optional[str] = None,
        use_neural: Optional[bool] = None
    ):
        """
        Initialize audio recorder.

        Args:
            output_dir: Directory for output files
            session_name: Name for this recording session
            use_neural: Use neural diarization (default from USE_NEURAL_DIARIZATION env)
        """
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"audio_session_{timestamp}"
        self.session_id = f"{self.session_name}_{timestamp}"

        # Setup output directory
        base_dir = output_dir or os.getenv('RECORDING_PATH', './data/recordings')
        self.output_dir = Path(base_dir) / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine diarization mode
        if use_neural is None:
            use_neural = os.getenv('USE_NEURAL_DIARIZATION', 'false').lower() == 'true'
        self.use_neural = use_neural and PYANNOTE_AVAILABLE

        if use_neural and not PYANNOTE_AVAILABLE:
            logger.warning(
                "Neural diarization requested but pyannote.audio not available. "
                "Falling back to simple diarization."
            )

        # Initialize components
        self.capture: Optional[AudioCaptureManager] = None
        self.transcriber: Optional[WhisperTranscriber] = None
        self.diarizer = None  # Will be SpeakerDiarizer or NeuralSpeakerDiarizer
        self.engagement: Optional[EngagementAnalyzer] = None

        # Recording state
        self.is_recording = False
        self.start_time: Optional[float] = None
        self.transcripts: list = []

        logger.info(f"AudioOnlyRecorder initialized - Session: {self.session_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Diarization mode: {'neural' if self.use_neural else 'simple'}")

    def start_recording(self) -> str:
        """
        Start audio recording and transcription.

        Returns:
            Session ID
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return self.session_id

        logger.info("Initializing audio components...")

        # Initialize capture
        self.capture = AudioCaptureManager(enable_vad=True)

        # Initialize transcription
        self.transcriber = WhisperTranscriber()
        self.transcriber.start_workers()

        # Initialize speaker diarization
        if self.use_neural:
            logger.info("Using neural speaker diarization (pyannote.audio)")
            self.diarizer = NeuralSpeakerDiarizer()
        else:
            logger.info("Using simple speaker diarization")
            self.diarizer = SpeakerDiarizer()
        self.engagement = EngagementAnalyzer()

        # Set up audio segment callback
        self.capture.set_segment_callback(self._on_audio_segment)

        # Start capture
        if not self.capture.start_capture():
            raise RuntimeError("Failed to start audio capture")

        self.is_recording = True
        self.start_time = time.time()

        logger.info("Audio recording started")
        return self.session_id

    def _on_audio_segment(
        self,
        audio_data,
        start_time: float,
        end_time: float
    ):
        """Handle detected audio segment."""
        # Identify speaker
        speaker_id, confidence = self.diarizer.identify_speaker(audio_data)

        # Queue for transcription
        self.transcriber.queue_audio(
            audio_data,
            start_time,
            speaker_id=speaker_id
        )

        logger.debug(
            f"Segment queued: {start_time:.2f}s - {end_time:.2f}s | "
            f"Speaker: {speaker_id} ({confidence:.2f})"
        )

    def get_transcription_results(self) -> list:
        """
        Get pending transcription results.

        Returns:
            List of transcription results
        """
        if not self.transcriber:
            return []

        results = self.transcriber.get_results()

        # Update engagement analyzer with each result
        for result in results:
            self._update_engagement(result)

        self.transcripts.extend(results)
        return results

    def _update_engagement(self, transcript: dict):
        """Update engagement analyzer with a transcript result."""
        if not self.engagement:
            return

        speaker_id = transcript.get('speaker_id', 'Unknown')
        timestamp = transcript.get('timestamp', 0)
        duration = transcript.get('duration', 0)

        # Create a SpeakerSegment for the engagement analyzer
        segment = SpeakerSegment(
            speaker_id=speaker_id,
            start_time=timestamp,
            end_time=timestamp + duration,
            audio_data=np.array([], dtype=np.float32),
            confidence=transcript.get('confidence', 0),
            text=transcript.get('text', '')
        )

        self.engagement.update_speaker_stats(speaker_id, segment)

    def get_live_stats(self) -> dict:
        """
        Get live recording statistics.

        Returns:
            Statistics dictionary
        """
        if not self.is_recording:
            return {'status': 'not_recording'}

        duration = time.time() - self.start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        stats = {
            'status': 'recording',
            'session_id': self.session_id,
            'duration': f"{minutes:02d}:{seconds:02d}",
            'duration_seconds': duration,
            'transcripts_count': len(self.transcripts),
        }

        # Add engagement metrics (uses internal state from _update_engagement calls)
        if self.transcripts and self.engagement:
            stats['engagement_metrics'] = self.engagement.calculate_engagement_scores()

        return stats

    def stop_recording(self) -> dict:
        """
        Stop recording and save results.

        Returns:
            Session summary
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return {}

        logger.info("Stopping audio recording...")

        # Stop capture
        if self.capture:
            self.capture.stop_capture()

        # Wait for pending transcriptions
        time.sleep(2)

        # Get final results
        if self.transcriber:
            final_results = self.transcriber.get_results()
            # Update engagement analyzer with final results
            for result in final_results:
                self._update_engagement(result)
            self.transcripts.extend(final_results)
            self.transcriber.stop_workers()

        self.is_recording = False
        duration = time.time() - self.start_time

        # Save transcripts
        self._save_transcripts()

        # Generate summary
        summary = {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'duration_seconds': duration,
            'duration': f"{int(duration // 60):02d}:{int(duration % 60):02d}",
            'total_transcripts': len(self.transcripts),
            'output_dir': str(self.output_dir),
        }

        # Add engagement summary (uses internal state from _update_engagement calls)
        if self.transcripts and self.engagement:
            summary['engagement'] = self.engagement.calculate_engagement_scores()

        # Save summary
        summary_file = self.output_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Session summary saved to: {summary_file}")

        return summary

    def _save_transcripts(self):
        """Save transcripts to file."""
        if not self.transcripts:
            logger.info("No transcripts to save")
            return

        # Save as JSON
        json_file = self.output_dir / "transcripts.json"
        with open(json_file, 'w') as f:
            json.dump(self.transcripts, f, indent=2, default=str)

        # Save as readable text
        text_file = self.output_dir / "transcripts.txt"
        with open(text_file, 'w') as f:
            f.write(f"Audio Recording Session: {self.session_id}\n")
            f.write(f"{'=' * 60}\n\n")

            for t in sorted(self.transcripts, key=lambda x: x.get('timestamp', 0)):
                timestamp = t.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                speaker = t.get('speaker_id', 'Unknown')
                confidence = t.get('confidence', 0)
                text = t.get('text', '')

                f.write(f"[{minutes:02d}:{seconds:02d}] {speaker} ({confidence:.2f}): {text}\n")

        logger.info(f"Transcripts saved to: {json_file}")
        logger.info(f"Readable transcript: {text_file}")


def display_live_stats(recorder: AudioOnlyRecorder):
    """Display live recording statistics."""
    stats = recorder.get_live_stats()

    if stats.get('status') == 'not_recording':
        return

    logger.info("=" * 60)
    logger.info("Live Recording Statistics")
    logger.info("=" * 60)
    logger.info(f"Session ID: {stats.get('session_id')}")
    logger.info(f"Duration: {stats.get('duration')}")
    logger.info(f"Transcripts: {stats.get('transcripts_count')}")

    # engagement is a dict of speaker_id -> scores
    engagement = stats.get('engagement_metrics', {})
    if engagement:
        logger.info(f"Unique Speakers: {len(engagement)}")
        logger.info("Speaker Activity:")
        for speaker_id, data in engagement.items():
            logger.info(
                f"  {speaker_id}: {data.get('utterance_count', 0)} utterances, "
                f"{data.get('speaking_time_seconds', 0):.1f}s, "
                f"engagement: {data.get('engagement_score', 0):.0f}%"
            )

    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Record bridge crew audio without game connection'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Recording duration in seconds (default: record until interrupted)'
    )
    parser.add_argument(
        '--session-name',
        help='Name for this recording session'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for recordings'
    )
    parser.add_argument(
        '--stats-interval',
        type=int,
        default=15,
        help='Live stats display interval in seconds (default: 15)'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--neural',
        action='store_true',
        default=None,
        help='Use neural speaker diarization (better accuracy, more CPU/GPU)'
    )
    parser.add_argument(
        '--no-neural',
        action='store_true',
        help='Use simple speaker diarization (faster, less accurate)'
    )

    args = parser.parse_args()

    # Determine neural diarization setting
    use_neural = None  # Use environment default
    if args.neural:
        use_neural = True
    elif args.no_neural:
        use_neural = False

    # List devices if requested
    if args.list_devices:
        from src.audio.capture import list_audio_devices
        list_audio_devices()
        return

    # Check audio configuration
    if os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() != 'true':
        logger.warning("ENABLE_AUDIO_CAPTURE is not set to 'true' in .env")
        logger.warning("Set ENABLE_AUDIO_CAPTURE=true to enable audio capture")

    # Create recorder
    recorder = AudioOnlyRecorder(
        output_dir=args.output_dir,
        session_name=args.session_name,
        use_neural=use_neural
    )

    try:
        # Start recording
        session_id = recorder.start_recording()

        logger.info("=" * 60)
        logger.info("Audio Recording Started")
        logger.info("=" * 60)
        logger.info(f"Session ID: {session_id}")
        logger.info("Speak into your microphone - audio will be transcribed")
        logger.info("")
        logger.info("Press Ctrl+C to stop recording")
        logger.info("=" * 60)

        # Monitor recording
        start_time = time.time()
        last_stats_time = start_time

        while True:
            current_time = time.time()

            # Display stats periodically
            if current_time - last_stats_time >= args.stats_interval:
                display_live_stats(recorder)
                last_stats_time = current_time

            # Check duration limit
            if args.duration and (current_time - start_time) >= args.duration:
                logger.info(f"\nRecording duration limit ({args.duration}s) reached")
                break

            # Check for new transcriptions
            results = recorder.get_transcription_results()
            for result in results:
                logger.info(
                    f"[{result.get('speaker_id', 'Unknown')}] "
                    f"({result.get('confidence', 0):.2f}): {result.get('text', '')}"
                )

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("\n\nRecording interrupted by user")

    except Exception as e:
        logger.error(f"Recording error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Stop recording
        summary = recorder.stop_recording()

        if summary:
            logger.info("=" * 60)
            logger.info("Recording Complete")
            logger.info("=" * 60)
            logger.info(f"Session ID: {summary.get('session_id')}")
            logger.info(f"Duration: {summary.get('duration')}")
            logger.info(f"Total Transcripts: {summary.get('total_transcripts')}")
            logger.info(f"Output Directory: {summary.get('output_dir')}")

            engagement = summary.get('engagement', {})
            if engagement:
                logger.info(f"Unique Speakers: {len(engagement)}")

            logger.info("=" * 60)


if __name__ == "__main__":
    main()
