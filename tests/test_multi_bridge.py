#!/usr/bin/env python3
"""
Tests for multi-bridge support functionality.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.event_recorder import EventRecorder
from src.metrics.audio_transcript import AudioTranscriptService
from src.metrics.mission_summarizer import MissionSummarizer


class TestEventRecorderBridgeId:
    """Test EventRecorder bridge_id support."""

    def test_bridge_id_stored(self):
        """Test bridge_id is stored correctly."""
        recorder = EventRecorder(
            mission_id="TEST_001",
            mission_name="Test Mission",
            bridge_id="Bridge-Alpha"
        )
        assert recorder.bridge_id == "Bridge-Alpha"

    def test_bridge_id_none_by_default(self):
        """Test bridge_id is None when not provided."""
        recorder = EventRecorder(
            mission_id="TEST_002",
            mission_name="Test Mission"
        )
        assert recorder.bridge_id is None

    def test_bridge_id_in_json_export(self):
        """Test bridge_id is included in JSON export."""
        recorder = EventRecorder(
            mission_id="TEST_003",
            mission_name="Test Mission",
            bridge_id="Bridge-Beta"
        )

        # Record a test event
        recorder.record_event("test_event", "test", {"data": "value"})

        # Export to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = Path(f.name)

        try:
            recorder.export_to_json(filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert data.get('bridge_id') == "Bridge-Beta"
            assert data.get('mission_id') == "TEST_003"
        finally:
            filepath.unlink()

    def test_bridge_id_none_in_json_export(self):
        """Test bridge_id is null in JSON when not set."""
        recorder = EventRecorder(
            mission_id="TEST_004",
            mission_name="Test Mission"
        )

        recorder.record_event("test_event", "test", {"data": "value"})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = Path(f.name)

        try:
            recorder.export_to_json(filepath)

            with open(filepath) as f:
                data = json.load(f)

            assert data.get('bridge_id') is None
        finally:
            filepath.unlink()


class TestAudioTranscriptServiceBridgeId:
    """Test AudioTranscriptService bridge_id support."""

    def test_bridge_id_stored(self):
        """Test bridge_id is stored correctly."""
        with patch.dict(os.environ, {'ENABLE_AUDIO_CAPTURE': 'false'}):
            service = AudioTranscriptService(
                mission_id="TEST_001",
                bridge_id="Bridge-Charlie"
            )
            assert service.bridge_id == "Bridge-Charlie"

    def test_bridge_id_none_by_default(self):
        """Test bridge_id is None when not provided."""
        with patch.dict(os.environ, {'ENABLE_AUDIO_CAPTURE': 'false'}):
            service = AudioTranscriptService(mission_id="TEST_002")
            assert service.bridge_id is None

    def test_bridge_id_in_transcript_export(self):
        """Test bridge_id is included in transcript export."""
        with patch.dict(os.environ, {'ENABLE_AUDIO_CAPTURE': 'false'}):
            service = AudioTranscriptService(
                mission_id="TEST_003",
                bridge_id="Bridge-Alpha"
            )

            # Manually add a transcript
            service.transcripts.append({
                "timestamp": datetime.now(),
                "speaker": "Captain",
                "text": "Test transcript",
                "confidence": 0.95
            })

            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                filepath = Path(f.name)

            try:
                service.export_transcript(filepath)

                with open(filepath) as f:
                    data = json.load(f)

                assert data.get('bridge_id') == "Bridge-Alpha"
                assert data.get('mission_id') == "TEST_003"
            finally:
                filepath.unlink()


class TestMissionSummarizerBridgeId:
    """Test MissionSummarizer bridge_id support."""

    def test_bridge_id_stored(self):
        """Test bridge_id is stored correctly."""
        summarizer = MissionSummarizer(
            mission_id="TEST_001",
            mission_name="Test Mission",
            bridge_id="Bridge-Beta"
        )
        assert summarizer.bridge_id == "Bridge-Beta"

    def test_bridge_id_none_by_default(self):
        """Test bridge_id is None when not provided."""
        summarizer = MissionSummarizer(
            mission_id="TEST_002",
            mission_name="Test Mission"
        )
        assert summarizer.bridge_id is None

    def test_bridge_id_in_report_export(self):
        """Test bridge_id is included in exported reports."""
        summarizer = MissionSummarizer(
            mission_id="TEST_003",
            mission_name="Test Mission",
            bridge_id="Bridge-Charlie"
        )

        # Add test event
        summarizer.load_events([{
            "event_id": "E001",
            "timestamp": datetime.now(),
            "event_type": "test",
            "category": "test",
            "data": {}
        }])

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = Path(f.name)

        try:
            summarizer.export_report(filepath, format="json")

            with open(filepath) as f:
                data = json.load(f)

            assert data.get('bridge_id') == "Bridge-Charlie"
        finally:
            filepath.unlink()


class TestGameRecorderBridgeId:
    """Test GameRecorder bridge_id support."""

    @patch('src.integration.game_recorder.StarshipHorizonsClient')
    def test_bridge_id_from_parameter(self, mock_client):
        """Test bridge_id can be set via constructor parameter."""
        mock_client.return_value.test_connection.return_value = True

        from src.integration.game_recorder import GameRecorder

        with patch.dict(os.environ, {'BRIDGE_ID': ''}, clear=False):
            recorder = GameRecorder(bridge_id="Test-Bridge")
            assert recorder.bridge_id == "Test-Bridge"

    @patch('src.integration.game_recorder.StarshipHorizonsClient')
    def test_bridge_id_from_environment(self, mock_client):
        """Test bridge_id is read from BRIDGE_ID environment variable."""
        mock_client.return_value.test_connection.return_value = True

        from src.integration.game_recorder import GameRecorder

        with patch.dict(os.environ, {'BRIDGE_ID': 'Env-Bridge'}):
            recorder = GameRecorder()
            assert recorder.bridge_id == "Env-Bridge"

    @patch('src.integration.game_recorder.StarshipHorizonsClient')
    def test_bridge_id_none_when_not_set(self, mock_client):
        """Test bridge_id is None when not configured."""
        mock_client.return_value.test_connection.return_value = True

        from src.integration.game_recorder import GameRecorder

        with patch.dict(os.environ, {'BRIDGE_ID': ''}, clear=False):
            recorder = GameRecorder(bridge_id=None)
            assert recorder.bridge_id is None

    @patch('src.integration.game_recorder.StarshipHorizonsClient')
    def test_mission_id_format_with_bridge(self, mock_client):
        """Test mission ID includes bridge prefix when set."""
        mock_client.return_value.test_connection.return_value = True
        mock_client.return_value.get_game_status.return_value = {}

        from src.integration.game_recorder import GameRecorder

        with patch.dict(os.environ, {'BRIDGE_ID': '', 'ENABLE_AUDIO_CAPTURE': 'false'}):
            recorder = GameRecorder(bridge_id="TestBridge")
            mission_id = recorder.start_recording("Test Mission")

            assert mission_id.startswith("TestBridge_GAME_")
            assert recorder.mission_id == mission_id

            # Clean up
            recorder.is_recording = False

    @patch('src.integration.game_recorder.StarshipHorizonsClient')
    def test_mission_id_format_without_bridge(self, mock_client):
        """Test mission ID format unchanged without bridge_id."""
        mock_client.return_value.test_connection.return_value = True
        mock_client.return_value.get_game_status.return_value = {}

        from src.integration.game_recorder import GameRecorder

        with patch.dict(os.environ, {'BRIDGE_ID': '', 'ENABLE_AUDIO_CAPTURE': 'false'}):
            recorder = GameRecorder(bridge_id=None)
            mission_id = recorder.start_recording("Test Mission")

            assert mission_id.startswith("GAME_")
            assert not mission_id.startswith("None_")

            # Clean up
            recorder.is_recording = False


class TestMissionIdFormats:
    """Test various mission ID format scenarios."""

    def test_bridge_id_with_special_characters(self):
        """Test bridge_id with spaces and special characters."""
        recorder = EventRecorder(
            mission_id="Training-Room-1_GAME_20260114_120000",
            mission_name="Test",
            bridge_id="Training-Room-1"
        )
        assert recorder.bridge_id == "Training-Room-1"

    def test_bridge_id_with_underscores(self):
        """Test bridge_id with underscores."""
        recorder = EventRecorder(
            mission_id="Bridge_Alpha_GAME_20260114_120000",
            mission_name="Test",
            bridge_id="Bridge_Alpha"
        )
        assert recorder.bridge_id == "Bridge_Alpha"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
