#!/usr/bin/env python3
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json


class TestMissionSummarizer:
    """Test suite for mission summary generation."""

    def test_mission_summarizer_initialization(self):
        """Test initializing the mission summarizer."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(
            mission_id="SUMMARY_TEST_001",
            mission_name="Test Mission Beta",
            llm_model="llama3.2"
        )

        assert summarizer.mission_id == "SUMMARY_TEST_001"
        assert summarizer.mission_name == "Test Mission Beta"
        assert summarizer.llm_model == "llama3.2"

    def test_load_mission_data(self):
        """Test loading event and transcript data for a mission."""
        from src.metrics.mission_summarizer import MissionSummarizer
        from src.metrics.event_recorder import EventRecorder
        from src.metrics.audio_transcript import AudioTranscriptService

        summarizer = MissionSummarizer(mission_id="LOAD_TEST")

        # Create mock data
        event_recorder = EventRecorder(mission_id="LOAD_TEST")
        event_recorder.record_event("navigation", "course", {"heading": 180})
        event_recorder.record_communication("Captain", "Engage")

        transcript_service = AudioTranscriptService(mission_id="LOAD_TEST")
        transcript_service.add_transcript(
            timestamp=datetime.now(),
            speaker="Captain",
            text="Red alert, all hands to battle stations",
            confidence=0.95
        )

        # Load data into summarizer
        summarizer.load_events(event_recorder.events)
        summarizer.load_transcripts(transcript_service.get_all_transcripts())

        assert len(summarizer.events) == 2
        assert len(summarizer.transcripts) == 1

    def test_timeline_generation(self):
        """Test generating a mission timeline from events."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(mission_id="TIMELINE_TEST")

        # Add events at different times
        base_time = datetime.now()
        events = [
            {"timestamp": base_time, "type": "mission_start", "data": {"status": "launched"}},
            {"timestamp": base_time + timedelta(minutes=5), "type": "encounter", "data": {"contact": "unknown vessel"}},
            {"timestamp": base_time + timedelta(minutes=10), "type": "combat", "data": {"action": "shields raised"}},
            {"timestamp": base_time + timedelta(minutes=15), "type": "resolution", "data": {"outcome": "peaceful"}},
        ]

        summarizer.load_events(events)
        timeline = summarizer.generate_timeline()

        assert len(timeline) == 4
        assert timeline[0]["type"] == "mission_start"
        assert timeline[-1]["type"] == "resolution"
        assert timeline[1]["elapsed_time"] == "00:05:00"

    def test_key_moment_identification(self):
        """Test identifying key moments in a mission."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(mission_id="KEY_MOMENT_TEST")

        # Load events with varying importance
        events = [
            {"timestamp": datetime.now(), "type": "routine", "category": "navigation", "data": {"speed": 5}},
            {"timestamp": datetime.now(), "type": "alert", "category": "combat", "data": {"alert_level": "red"}},
            {"timestamp": datetime.now(), "type": "critical", "category": "damage", "data": {"hull_breach": True}},
            {"timestamp": datetime.now(), "type": "routine", "category": "scan", "data": {"result": "clear"}},
        ]

        summarizer.load_events(events)
        key_moments = summarizer.identify_key_moments()

        assert len(key_moments) == 2  # Alert and critical events
        assert key_moments[0]["type"] == "alert"
        assert key_moments[1]["type"] == "critical"

    def test_crew_performance_analysis(self):
        """Test analyzing crew performance from events and transcripts."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(mission_id="PERFORMANCE_TEST")

        # Load crew actions
        events = [
            {"timestamp": datetime.now(), "type": "crew_action", "data": {"crew": "Helm", "action": "evasive_maneuver", "success": True}},
            {"timestamp": datetime.now(), "type": "crew_action", "data": {"crew": "Tactical", "action": "fire_weapons", "success": True}},
            {"timestamp": datetime.now(), "type": "crew_action", "data": {"crew": "Engineering", "action": "repair_shields", "success": False}},
            {"timestamp": datetime.now(), "type": "crew_action", "data": {"crew": "Helm", "action": "set_course", "success": True}},
        ]

        transcripts = [
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "Good work, Helm"},
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "Engineering, we need those shields"},
        ]

        summarizer.load_events(events)
        summarizer.load_transcripts(transcripts)

        performance = summarizer.analyze_crew_performance()

        assert "Helm" in performance
        assert performance["Helm"]["success_rate"] == 1.0
        assert performance["Helm"]["action_count"] == 2
        assert performance["Engineering"]["success_rate"] == 0.0
        assert performance["Captain"]["communication_count"] == 2

    def test_narrative_summary_generation(self):
        """Test generating narrative mission summary."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(
            mission_id="NARRATIVE_TEST",
            mission_name="Diplomatic Escort"
        )

        # Load comprehensive mission data
        events = [
            {"timestamp": datetime.now(), "type": "mission_start", "data": {"objective": "Escort ambassador to peace talks"}},
            {"timestamp": datetime.now() + timedelta(minutes=30), "type": "encounter", "data": {"contact": "3 hostile vessels"}},
            {"timestamp": datetime.now() + timedelta(minutes=35), "type": "combat", "data": {"action": "defensive posture"}},
            {"timestamp": datetime.now() + timedelta(minutes=40), "type": "resolution", "data": {"outcome": "hostiles retreated"}},
            {"timestamp": datetime.now() + timedelta(hours=2), "type": "mission_complete", "data": {"status": "successful"}},
        ]

        transcripts = [
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "We are escorting the ambassador to the peace summit"},
            {"timestamp": datetime.now() + timedelta(minutes=30), "speaker": "Tactical", "text": "Three vessels approaching, they're charging weapons"},
            {"timestamp": datetime.now() + timedelta(minutes=35), "speaker": "Captain", "text": "Shields up, defensive pattern alpha"},
            {"timestamp": datetime.now() + timedelta(hours=2), "speaker": "Captain", "text": "Mission accomplished, the ambassador is safe"},
        ]

        summarizer.load_events(events)
        summarizer.load_transcripts(transcripts)

        summary = summarizer.generate_narrative_summary()

        assert summary is not None
        assert "mission_name" in summary
        assert "duration" in summary
        assert "narrative" in summary
        assert "key_events" in summary
        assert "crew_performance" in summary
        assert "outcome" in summary
        assert len(summary["narrative"]) > 100  # Should be substantial text

    def test_tactical_analysis(self):
        """Test generating tactical analysis from combat events."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(mission_id="TACTICAL_TEST")

        combat_events = [
            {"timestamp": datetime.now(), "type": "combat_start", "data": {"enemies": 2}},
            {"timestamp": datetime.now(), "type": "weapon_fired", "data": {"weapon": "phaser", "target": "enemy_1", "hit": True}},
            {"timestamp": datetime.now(), "type": "damage_taken", "data": {"system": "shields", "severity": "minor"}},
            {"timestamp": datetime.now(), "type": "weapon_fired", "data": {"weapon": "torpedo", "target": "enemy_2", "hit": True}},
            {"timestamp": datetime.now(), "type": "combat_end", "data": {"result": "victory"}},
        ]

        summarizer.load_events(combat_events)
        tactical = summarizer.analyze_tactical_performance()

        assert tactical["total_engagements"] == 1
        assert tactical["hit_rate"] == 1.0
        assert tactical["weapons_used"] == ["phaser", "torpedo"]
        assert tactical["damage_taken"]["shields"] == "minor"
        assert tactical["combat_effectiveness"] == "high"

    def test_learning_objectives_assessment(self):
        """Test assessing completion of learning objectives."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(mission_id="LEARNING_TEST")

        # Define learning objectives
        objectives = [
            {"id": "OBJ_001", "description": "Successfully navigate to destination", "category": "navigation"},
            {"id": "OBJ_002", "description": "Communicate effectively with crew", "category": "communication"},
            {"id": "OBJ_003", "description": "Manage combat situation", "category": "combat"},
        ]

        # Load mission data
        events = [
            {"timestamp": datetime.now(), "type": "navigation", "data": {"action": "course_set", "success": True}},
            {"timestamp": datetime.now(), "type": "arrival", "data": {"destination": "target_system"}},
            {"timestamp": datetime.now(), "type": "combat", "data": {"engagement": True, "outcome": "victory"}},
        ]

        transcripts = [
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "Set course for the target system"},
            {"timestamp": datetime.now(), "speaker": "Helm", "text": "Course laid in"},
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "All stations report"},
        ]

        summarizer.load_events(events)
        summarizer.load_transcripts(transcripts)
        assessment = summarizer.assess_learning_objectives(objectives)

        assert assessment["OBJ_001"]["completed"] == True
        assert assessment["OBJ_002"]["completed"] == True
        assert assessment["OBJ_003"]["completed"] == True
        assert assessment["completion_rate"] == 1.0

    def test_export_comprehensive_report(self):
        """Test exporting comprehensive mission report."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(
            mission_id="EXPORT_TEST",
            mission_name="Training Mission Alpha"
        )

        # Load sample data
        events = [
            {"timestamp": datetime.now(), "type": "mission_start", "data": {}},
            {"timestamp": datetime.now() + timedelta(hours=1), "type": "mission_complete", "data": {}},
        ]

        transcripts = [
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "Begin training mission"},
        ]

        summarizer.load_events(events)
        summarizer.load_transcripts(transcripts)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export as JSON
            json_path = Path(tmpdir) / "mission_report.json"
            summarizer.export_report(json_path, format="json")
            assert json_path.exists()

            # Export as Markdown
            md_path = Path(tmpdir) / "mission_report.md"
            summarizer.export_report(md_path, format="markdown")
            assert md_path.exists()

            # Export as HTML
            html_path = Path(tmpdir) / "mission_report.html"
            summarizer.export_report(html_path, format="html")
            assert html_path.exists()

    def test_comparative_analysis(self):
        """Test comparing multiple mission performances."""
        from src.metrics.mission_summarizer import MissionSummarizer

        # Create multiple mission summaries
        missions = []
        for i in range(3):
            summarizer = MissionSummarizer(mission_id=f"MISSION_{i}")
            summarizer.load_events([
                {"timestamp": datetime.now(), "type": "mission_complete", "data": {"score": 80 + i*5}}
            ])
            missions.append(summarizer)

        # Compare missions
        comparison = MissionSummarizer.compare_missions(missions)

        assert len(comparison["missions"]) == 3
        assert comparison["best_score"] == 90
        assert comparison["average_score"] == 85


class TestMissionSummarizerIntegration:
    """Integration tests for mission summarizer with LLM."""

    def test_llm_summary_generation(self):
        """Test generating summary using LLM."""
        from src.metrics.mission_summarizer import MissionSummarizer

        summarizer = MissionSummarizer(
            mission_id="LLM_TEST",
            llm_model="llama3.2"
        )

        # Load rich mission data
        events = [
            {"timestamp": datetime.now(), "type": "mission_start", "data": {"briefing": "Investigate distress signal"}},
            {"timestamp": datetime.now(), "type": "discovery", "data": {"finding": "Damaged cargo vessel"}},
            {"timestamp": datetime.now(), "type": "rescue", "data": {"survivors": 15}},
        ]

        transcripts = [
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "We're receiving a distress signal"},
            {"timestamp": datetime.now(), "speaker": "Science", "text": "Scanning for life signs"},
            {"timestamp": datetime.now(), "speaker": "Captain", "text": "Beam the survivors aboard"},
        ]

        summarizer.load_events(events)
        summarizer.load_transcripts(transcripts)

        # Generate LLM-powered summary
        llm_summary = summarizer.generate_llm_summary(
            include_recommendations=True,
            include_learning_points=True
        )

        assert llm_summary is not None
        assert "summary" in llm_summary
        assert "recommendations" in llm_summary
        assert "learning_points" in llm_summary
        assert len(llm_summary["summary"]) > 200

    def test_end_to_end_mission_processing(self):
        """Test complete end-to-end mission recording and summary."""
        from src.metrics.event_recorder import EventRecorder
        from src.metrics.audio_transcript import AudioTranscriptService
        from src.metrics.mission_summarizer import MissionSummarizer

        mission_id = "E2E_TEST"

        # 1. Record events during mission
        recorder = EventRecorder(mission_id=mission_id)
        recorder.record_event("mission_start", "initialization", {"status": "ready"})
        recorder.record_communication("Captain", "Begin mission")
        recorder.record_event("navigation", "course", {"destination": "Alpha Centauri"})
        recorder.record_alert("yellow", "sensors", "Unknown contact detected")
        recorder.record_event("mission_complete", "success", {"objectives_met": True})

        # 2. Process audio transcripts
        transcript_service = AudioTranscriptService(mission_id=mission_id)
        transcript_service.add_transcript(datetime.now(), "Captain", "Begin mission", 0.95)
        transcript_service.add_transcript(datetime.now(), "Helm", "Course set", 0.92)
        transcript_service.add_transcript(datetime.now(), "Science", "Contact identified as friendly", 0.88)

        # 3. Generate comprehensive summary
        summarizer = MissionSummarizer(mission_id=mission_id)
        summarizer.load_events(recorder.events)
        summarizer.load_transcripts(transcript_service.get_all_transcripts())

        # Generate all reports
        timeline = summarizer.generate_timeline()
        narrative = summarizer.generate_narrative_summary()
        performance = summarizer.analyze_crew_performance()

        # Verify complete processing
        assert len(timeline) == 5
        assert narrative is not None
        assert "Captain" in performance
        assert performance["Captain"]["communication_count"] == 1