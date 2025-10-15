#!/usr/bin/env python3
"""
Game Recorder for Starship Horizons
Integrates the game client with our metrics system.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from src.metrics.event_recorder import EventRecorder
from src.metrics.audio_transcript import AudioTranscriptService
from src.metrics.mission_summarizer import MissionSummarizer
from src.integration.starship_horizons_client import StarshipHorizonsClient

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameRecorder:
    """Records Starship Horizons game sessions using the metrics system."""

    def __init__(self, game_host: str = None):
        """
        Initialize the game recorder.

        Args:
            game_host: Hostname or URL of the Starship Horizons game (defaults to env var GAME_HOST)
        """
        # Get host from environment or use provided host
        if game_host is None:
            host = os.getenv("GAME_HOST", "localhost")
            port = os.getenv("GAME_PORT_API", "1864")
            game_host = f"http://{host}:{port}"
        else:
            # If game_host doesn't have a scheme, add it
            if not game_host.startswith(('http://', 'https://')):
                port = os.getenv("GAME_PORT_API", "1864")
                game_host = f"http://{game_host}:{port}"

        self.game_host = game_host
        self.client = StarshipHorizonsClient(game_host)

        # Metrics components
        self.event_recorder = None
        self.audio_service = None
        self.mission_id = None
        self.is_recording = False

        # Register event handler
        self.client.add_event_callback(self._handle_game_event)

    def start_recording(self, mission_name: str = None) -> str:
        """
        Start recording a game session.

        Args:
            mission_name: Optional mission name

        Returns:
            Mission ID for this recording session
        """
        if self.is_recording:
            logger.warning("Already recording")
            return self.mission_id

        # Generate mission ID
        self.mission_id = f"GAME_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get current game status
        status = self.client.get_game_status()
        if status:
            if not mission_name:
                mission_name = status.get("Mission") or "Unknown Mission"
            logger.info(f"Game Status - State: {status.get('State')}, Mode: {status.get('Mode')}, Mission: {mission_name}")

        # Initialize event recorder
        self.event_recorder = EventRecorder(
            mission_id=self.mission_id,
            mission_name=mission_name or "Starship Horizons Session",
            bridge_crew=["Captain", "Helm", "Tactical", "Science", "Engineering", "Communications"]
        )

        # Initialize audio service with auto-transcription
        enable_audio = os.getenv('ENABLE_AUDIO_CAPTURE', 'false').lower() == 'true'
        self.audio_service = AudioTranscriptService(
            mission_id=self.mission_id,
            sample_rate=16000,
            channels=1,
            auto_transcribe=enable_audio
        )

        # Set storage path for audio recordings
        export_dir = Path("game_recordings") / self.mission_id
        export_dir.mkdir(parents=True, exist_ok=True)
        self.audio_service.set_storage_path(str(export_dir))

        # Start audio capture if enabled
        if enable_audio:
            logger.info("Starting audio capture...")
            if self.audio_service.start_audio_capture():
                logger.info("✓ Audio capture enabled for mission recording")

                # Start real-time transcription
                self.audio_service.start_realtime_transcription()
            else:
                logger.warning("Audio capture failed to start - continuing without audio")

        # Start client polling
        self.client.start_polling(interval=0.5)  # Poll every 500ms

        # Try WebSocket connection
        self.client.connect_websocket()

        self.is_recording = True
        logger.info(f"Started recording mission: {self.mission_id} - {mission_name}")

        # Record initial event
        self.event_recorder.record_event(
            event_type="recording_start",
            category="system",
            data={
                "mission_id": self.mission_id,
                "mission_name": mission_name,
                "game_host": self.game_host,
                "timestamp": datetime.now().isoformat()
            }
        )

        return self.mission_id

    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and generate summary.

        Returns:
            Recording summary with statistics
        """
        if not self.is_recording:
            logger.warning("Not currently recording")
            return {}

        # Stop client
        self.client.stop_polling()
        self.client.disconnect()

        # Stop audio capture
        if self.audio_service:
            self.audio_service.stop_audio_capture()
            self.audio_service.stop_realtime_transcription()
            logger.info("✓ Audio capture stopped")

        # Record final event
        self.event_recorder.record_event(
            event_type="recording_stop",
            category="system",
            data={
                "timestamp": datetime.now().isoformat()
            }
        )

        # Get statistics
        logger.info("Getting statistics...")
        stats = self.event_recorder.get_statistics()
        logger.info("Statistics obtained")

        # Export data
        export_dir = Path("game_recordings") / self.mission_id
        export_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Export directory created: {export_dir}")

        # Export events
        events_file = export_dir / "game_events.json"
        logger.info(f"Exporting events to {events_file}...")
        self.event_recorder.export_to_json(events_file)
        logger.info("Events exported")

        # Export transcripts if any
        if self.audio_service.get_all_transcripts():
            transcript_file = export_dir / "transcripts.json"
            self.audio_service.export_transcript(transcript_file)

        # Generate LLM-powered mission report if enabled
        llm_enabled = os.getenv('ENABLE_LLM_REPORTS', 'true').lower() == 'true'
        if llm_enabled:
            try:
                logger.info("Generating LLM-powered mission report...")
                summarizer = self.generate_summary()
                if summarizer:
                    report_style = os.getenv('LLM_REPORT_STYLE', 'entertaining')
                    report_file = export_dir / "mission_report_llm.md"
                    summarizer.generate_llm_report(style=report_style, output_file=report_file)
                    logger.info(f"✓ LLM report generated: {report_file}")
            except Exception as e:
                logger.warning(f"Failed to generate LLM report: {e}")

        self.is_recording = False

        logger.info(f"Stopped recording. Events saved to: {export_dir}")
        logger.info(f"Recording stats: {stats}")

        return {
            "mission_id": self.mission_id,
            "export_path": str(export_dir),
            "statistics": stats
        }

    def _handle_game_event(self, event: Dict[str, Any]):
        """
        Handle incoming game events.

        Args:
            event: Game event data
        """
        if not self.is_recording or not self.event_recorder:
            return

        try:
            # Map game events to our event types
            event_type = event.get("type", "unknown")
            category = event.get("category", "game")
            data = event.get("data", {})

            # Log the event
            logger.info(f"Game Event: {event_type} - {category}")

            # Special handling for different event types
            if event_type == "mission_start":
                self.event_recorder.record_event(
                    event_type="mission_start",
                    category="mission",
                    data=data
                )
            elif event_type == "mission_complete":
                self.event_recorder.record_event(
                    event_type="mission_complete",
                    category="mission",
                    data=data
                )
            elif event_type == "state_change":
                # Handle state changes
                new_state = data.get("new_state")
                if new_state == "Alert":
                    self.event_recorder.record_alert(
                        alert_level="yellow",
                        system="game",
                        message=f"Game state changed to {new_state}",
                        triggered_by="state_change"
                    )
                else:
                    self.event_recorder.record_event(
                        event_type="game_state_change",
                        category="system",
                        data=data
                    )
            elif event_type == "ship_update":
                # Record ship telemetry
                self._process_ship_update(data)
            elif event_type == "websocket_message":
                # Process WebSocket messages
                self._process_websocket_message(data)
            else:
                # Record generic event
                self.event_recorder.record_event(
                    event_type=event_type,
                    category=category,
                    data=data
                )

        except Exception as e:
            logger.error(f"Error handling game event: {e}")

    def _process_ship_update(self, ship_data: Dict[str, Any]):
        """Process ship update data."""
        # Extract relevant ship information
        if "shields" in ship_data:
            self.event_recorder.record_event(
                event_type="ship_status",
                category="telemetry",
                data={
                    "shields": ship_data.get("shields"),
                    "hull": ship_data.get("hull"),
                    "power": ship_data.get("power"),
                    "speed": ship_data.get("speed")
                }
            )

        # Check for alerts
        if ship_data.get("alert_status"):
            self.event_recorder.record_alert(
                alert_level=ship_data.get("alert_status"),
                system="ship",
                message="Ship alert status changed",
                triggered_by="ship_update"
            )

    def _process_websocket_message(self, message_data: Dict[str, Any]):
        """Process WebSocket messages."""
        # Parse different message types
        msg_type = message_data.get("type") or message_data.get("messageType")

        if msg_type == "chat" or msg_type == "communication":
            # Record as crew communication
            self.event_recorder.record_communication(
                speaker=message_data.get("sender", "Unknown"),
                message=message_data.get("message", ""),
                confidence=1.0
            )
        elif msg_type == "combat":
            # Record combat event
            self.event_recorder.record_event(
                event_type="combat_action",
                category="combat",
                data=message_data
            )
        elif msg_type == "damage":
            # Record damage event
            self.event_recorder.record_event(
                event_type="damage_report",
                category="damage",
                data=message_data
            )

    def generate_summary(self) -> Optional[MissionSummarizer]:
        """
        Generate a mission summary from recorded data.

        Returns:
            MissionSummarizer with loaded data
        """
        if not self.event_recorder:
            logger.error("No recording data available")
            return None

        summarizer = MissionSummarizer(
            mission_id=self.mission_id,
            mission_name=self.event_recorder.mission_name
        )

        # Load events
        summarizer.load_events(self.event_recorder.events)

        # Load transcripts if any
        if self.audio_service:
            summarizer.load_transcripts(self.audio_service.get_all_transcripts())

        return summarizer

    def get_live_stats(self) -> Dict[str, Any]:
        """Get live recording statistics."""
        if not self.is_recording or not self.event_recorder:
            return {"status": "not_recording"}

        stats = self.event_recorder.get_statistics()
        status = self.client.get_game_status() or {}

        live_stats = {
            "status": "recording",
            "mission_id": self.mission_id,
            "game_state": status.get("State"),
            "game_mode": status.get("Mode"),
            "events_recorded": stats["total_events"],
            "recording_duration": stats["duration"],
            "events_per_minute": stats["events_per_minute"]
        }

        # Add audio stats if available
        if self.audio_service:
            transcripts = self.audio_service.get_all_transcripts()
            live_stats.update({
                "transcripts_count": len(transcripts),
                "audio_duration": self.audio_service.get_total_duration(),
                "conversation_summary": self.audio_service.get_conversation_summary()
            })

            # Add engagement metrics if available
            engagement = self.audio_service.get_engagement_summary()
            if engagement:
                live_stats["engagement_metrics"] = engagement

        return live_stats

    def get_combined_timeline(self) -> Dict[str, Any]:
        """
        Get combined timeline of game events and audio transcripts.

        Returns:
            Dictionary with merged timeline of events and audio
        """
        if not self.event_recorder:
            return {"error": "No recording data available"}

        timeline = []

        # Add all game events
        for event in self.event_recorder.events:
            timeline.append({
                "timestamp": event["timestamp"],
                "type": "event",
                "event_type": event.get("event_type"),
                "category": event.get("category"),
                "data": event.get("data")
            })

        # Add all transcripts
        if self.audio_service:
            for transcript in self.audio_service.get_all_transcripts():
                timeline.append({
                    "timestamp": transcript["timestamp"],
                    "type": "transcript",
                    "speaker": transcript["speaker"],
                    "text": transcript["text"],
                    "confidence": transcript["confidence"]
                })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return {
            "mission_id": self.mission_id,
            "total_items": len(timeline),
            "timeline": timeline
        }