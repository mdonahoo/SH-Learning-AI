#!/usr/bin/env python3
import pytest
from datetime import datetime
from typing import Dict, Any, List
import json
from pathlib import Path
import tempfile


class TestEventRecorder:
    """Test suite for the event recording system."""

    def test_event_recorder_initialization(self):
        """Test that EventRecorder can be initialized with mission metadata."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(
            mission_id="TEST_MISSION_001",
            mission_name="Test Mission Alpha",
            bridge_crew=["Captain", "Helm", "Tactical", "Science", "Engineering"]
        )

        assert recorder.mission_id == "TEST_MISSION_001"
        assert recorder.mission_name == "Test Mission Alpha"
        assert len(recorder.bridge_crew) == 5
        assert recorder.start_time is not None
        assert len(recorder.events) == 0

    def test_record_game_event(self):
        """Test recording game events with timestamps and metadata."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(mission_id="TEST_001")

        event = recorder.record_event(
            event_type="ship_action",
            category="navigation",
            data={
                "action": "set_course",
                "heading": 270,
                "speed": "warp_5",
                "initiated_by": "Helm"
            }
        )

        assert event["event_id"] is not None
        assert event["timestamp"] is not None
        assert event["event_type"] == "ship_action"
        assert event["category"] == "navigation"
        assert event["data"]["speed"] == "warp_5"
        assert len(recorder.events) == 1

    def test_record_crew_communication(self):
        """Test recording crew communications with speaker identification."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(mission_id="TEST_001")

        comm_event = recorder.record_communication(
            speaker="Captain",
            message="Set course for the Neutral Zone, warp factor 5",
            audio_file="audio_001.wav",
            confidence=0.95
        )

        assert comm_event["event_type"] == "communication"
        assert comm_event["data"]["speaker"] == "Captain"
        assert comm_event["data"]["message"] == "Set course for the Neutral Zone, warp factor 5"
        assert comm_event["data"]["audio_file"] == "audio_001.wav"
        assert comm_event["data"]["confidence"] == 0.95

    def test_record_system_alert(self):
        """Test recording system alerts and warnings."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(mission_id="TEST_001")

        alert = recorder.record_alert(
            alert_level="yellow",
            system="shields",
            message="Shield strength at 45%",
            triggered_by="damage"
        )

        assert alert["event_type"] == "alert"
        assert alert["category"] == "system_alert"
        assert alert["data"]["alert_level"] == "yellow"
        assert alert["data"]["system"] == "shields"

    def test_event_filtering_by_time(self):
        """Test filtering events by time range."""
        from src.metrics.event_recorder import EventRecorder
        import time

        recorder = EventRecorder(mission_id="TEST_001")

        # Record events at different times
        event1 = recorder.record_event("test", "cat1", {"data": 1})
        time.sleep(0.1)
        timestamp_middle = datetime.now()
        time.sleep(0.1)
        event2 = recorder.record_event("test", "cat2", {"data": 2})

        # Filter events
        events_after = recorder.get_events_after(timestamp_middle)
        events_before = recorder.get_events_before(timestamp_middle)

        assert len(events_after) == 1
        assert len(events_before) == 1
        assert events_after[0]["data"]["data"] == 2
        assert events_before[0]["data"]["data"] == 1

    def test_event_filtering_by_type(self):
        """Test filtering events by type and category."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(mission_id="TEST_001")

        recorder.record_event("navigation", "movement", {"speed": 5})
        recorder.record_event("combat", "weapons", {"target": "enemy1"})
        recorder.record_event("navigation", "course", {"heading": 180})
        recorder.record_communication("Captain", "Fire photon torpedoes")

        nav_events = recorder.get_events_by_type("navigation")
        combat_events = recorder.get_events_by_category("weapons")
        comm_events = recorder.get_events_by_type("communication")

        assert len(nav_events) == 2
        assert len(combat_events) == 1
        assert len(comm_events) == 1

    def test_export_events_to_json(self):
        """Test exporting recorded events to JSON format."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(
            mission_id="TEST_001",
            mission_name="Export Test"
        )

        # Record various events
        recorder.record_event("test1", "cat1", {"value": 1})
        recorder.record_communication("Helm", "Engaging warp drive")
        recorder.record_alert("red", "hull", "Hull breach deck 7")

        # Export to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "mission_events.json"
            recorder.export_to_json(filepath)

            # Verify file exists and contains data
            assert filepath.exists()

            with open(filepath, 'r') as f:
                data = json.load(f)

            assert data["mission_id"] == "TEST_001"
            assert data["mission_name"] == "Export Test"
            assert len(data["events"]) == 3
            assert "start_time" in data
            assert "end_time" in data

    def test_event_statistics(self):
        """Test generating statistics from recorded events."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(mission_id="TEST_001")

        # Record various events
        for _ in range(5):
            recorder.record_event("navigation", "course", {})
        for _ in range(3):
            recorder.record_event("combat", "weapons", {})
        for _ in range(7):
            recorder.record_communication("Various", "Message")

        stats = recorder.get_statistics()

        assert stats["total_events"] == 15
        assert stats["event_types"]["navigation"] == 5
        assert stats["event_types"]["combat"] == 3
        assert stats["event_types"]["communication"] == 7
        assert "duration" in stats
        assert "events_per_minute" in stats


class TestEventRecorderIntegration:
    """Integration tests for event recorder with other systems."""

    def test_concurrent_event_recording(self):
        """Test thread-safe concurrent event recording."""
        from src.metrics.event_recorder import EventRecorder
        import threading

        recorder = EventRecorder(mission_id="CONCURRENT_TEST")

        def record_events(event_type, count):
            for i in range(count):
                recorder.record_event(event_type, "test", {"index": i})

        threads = [
            threading.Thread(target=record_events, args=("type1", 10)),
            threading.Thread(target=record_events, args=("type2", 10)),
            threading.Thread(target=record_events, args=("type3", 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(recorder.events) == 30

    def test_memory_efficient_large_missions(self):
        """Test handling large number of events efficiently."""
        from src.metrics.event_recorder import EventRecorder

        recorder = EventRecorder(mission_id="LARGE_MISSION")

        # Record 1000 events
        for i in range(1000):
            recorder.record_event(
                event_type=f"type_{i % 10}",
                category=f"cat_{i % 5}",
                data={"index": i, "value": f"data_{i}"}
            )

        # Should handle large datasets efficiently
        assert len(recorder.events) == 1000

        # Test filtering performance
        filtered = recorder.get_events_by_type("type_0")
        assert len(filtered) == 100