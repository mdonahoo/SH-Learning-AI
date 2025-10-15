#!/usr/bin/env python3
"""
Demonstration of the comprehensive learning metrics and assessment system.
Shows how to record a complete mission with events, audio, and generate summaries.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.event_recorder import EventRecorder
from src.metrics.audio_transcript import AudioTranscriptService
from src.metrics.mission_summarizer import MissionSummarizer
import numpy as np


def simulate_mission():
    """Simulate a complete mission with events and audio."""
    print("=" * 60)
    print("STARSHIP HORIZONS LEARNING AI")
    print("Mission Recording & Assessment Demo")
    print("=" * 60)

    # Initialize components
    mission_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mission_name = "Training Mission: Neutral Zone Patrol"

    print(f"\nMission ID: {mission_id}")
    print(f"Mission Name: {mission_name}")

    # 1. Initialize Event Recorder
    print("\n[1] Initializing Event Recorder...")
    event_recorder = EventRecorder(
        mission_id=mission_id,
        mission_name=mission_name,
        bridge_crew=["Captain", "Helm", "Tactical", "Science", "Engineering"]
    )

    # 2. Initialize Audio Transcript Service
    print("[2] Initializing Audio Transcript Service...")
    audio_service = AudioTranscriptService(
        mission_id=mission_id,
        sample_rate=16000,
        channels=1,
        buffer_duration=10
    )

    # 3. Start Recording
    print("\n[3] Starting Mission Recording...")
    audio_service.start_recording()

    # Mission Start
    print("\n--- MISSION START ---")
    event_recorder.record_event(
        event_type="mission_start",
        category="initialization",
        data={
            "objective": "Patrol the Neutral Zone and investigate anomalous readings",
            "ship": "USS Explorer",
            "crew_complement": 5
        }
    )

    # Simulate audio recording
    audio_chunk = np.random.normal(0, 0.1, 16000).astype(np.float32)
    audio_service.add_audio_chunk(audio_chunk)

    # Captain's Opening Orders
    event_recorder.record_communication(
        speaker="Captain",
        message="All stations report readiness for patrol",
        confidence=0.95
    )
    audio_service.add_transcript(
        timestamp=datetime.now(),
        speaker="Captain",
        text="All stations report readiness for patrol",
        confidence=0.95
    )

    time.sleep(0.5)

    # Crew Reports
    crew_reports = [
        ("Helm", "Navigation systems online, course laid in"),
        ("Tactical", "Weapons and shields at standby"),
        ("Science", "Sensors calibrated and scanning"),
        ("Engineering", "Warp core stable at 98% efficiency")
    ]

    for crew_member, report in crew_reports:
        print(f"  {crew_member}: {report}")
        event_recorder.record_communication(
            speaker=crew_member,
            message=report,
            confidence=0.92
        )
        audio_service.add_transcript(
            timestamp=datetime.now(),
            speaker=crew_member,
            text=report,
            confidence=0.92
        )
        time.sleep(0.2)

    # Navigation Event
    print("\n--- ENTERING NEUTRAL ZONE ---")
    event_recorder.record_event(
        event_type="navigation",
        category="movement",
        data={
            "action": "enter_neutral_zone",
            "speed": "warp_5",
            "heading": 270
        }
    )

    # Simulate more audio
    audio_chunk = np.random.normal(0, 0.1, 16000).astype(np.float32)
    audio_service.add_audio_chunk(audio_chunk)

    # Sensor Alert
    print("\n--- SENSOR ALERT ---")
    event_recorder.record_alert(
        alert_level="yellow",
        system="sensors",
        message="Anomalous energy signature detected",
        triggered_by="automatic_scan"
    )

    event_recorder.record_communication(
        speaker="Science",
        message="Captain, detecting unusual energy patterns bearing 045 mark 12",
        confidence=0.89
    )

    # Combat Encounter
    print("\n--- COMBAT ENCOUNTER ---")
    event_recorder.record_event(
        event_type="combat_start",
        category="combat",
        data={
            "enemies": 2,
            "threat_level": "moderate"
        }
    )

    event_recorder.record_alert(
        alert_level="red",
        system="tactical",
        message="Hostile vessels detected",
        triggered_by="proximity_alert"
    )

    # Combat Actions
    combat_actions = [
        ("weapon_fired", {"weapon": "phaser", "target": "enemy_1", "hit": True}),
        ("damage_taken", {"system": "shields", "severity": "minor", "shield_strength": 85}),
        ("weapon_fired", {"weapon": "torpedo", "target": "enemy_1", "hit": True}),
        ("evasive_maneuver", {"pattern": "delta", "success": True}),
        ("weapon_fired", {"weapon": "phaser", "target": "enemy_2", "hit": False}),
        ("weapon_fired", {"weapon": "phaser", "target": "enemy_2", "hit": True}),
    ]

    for action_type, action_data in combat_actions:
        print(f"  Combat Action: {action_type}")
        event_recorder.record_event(
            event_type=action_type,
            category="combat",
            data=action_data
        )
        time.sleep(0.1)

    # Combat Resolution
    print("\n--- COMBAT RESOLVED ---")
    event_recorder.record_event(
        event_type="combat_end",
        category="combat",
        data={
            "result": "victory",
            "enemies_defeated": 2,
            "damage_sustained": "minimal"
        }
    )

    # Mission Completion
    print("\n--- MISSION COMPLETE ---")
    event_recorder.record_event(
        event_type="mission_complete",
        category="completion",
        data={
            "status": "successful",
            "objectives_met": True,
            "score": 92,
            "duration": "45 minutes"
        }
    )

    event_recorder.record_communication(
        speaker="Captain",
        message="Well done everyone. Mission accomplished. Set course for starbase",
        confidence=0.94
    )

    # Stop Recording
    print("\n[4] Stopping Recording...")
    recording_duration = audio_service.stop_recording()
    print(f"  Total recording duration: {recording_duration:.2f} seconds")

    # Generate Statistics
    print("\n[5] Generating Event Statistics...")
    stats = event_recorder.get_statistics()
    print(f"  Total events recorded: {stats['total_events']}")
    print(f"  Event types: {stats['event_types']}")
    print(f"  Events per minute: {stats['events_per_minute']}")

    # Export Data
    print("\n[6] Exporting Mission Data...")
    export_dir = Path("mission_exports") / mission_id
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export events
    events_file = export_dir / "events.json"
    event_recorder.export_to_json(events_file)
    print(f"  Events exported to: {events_file}")

    # Export transcripts
    transcript_file = export_dir / "transcripts.json"
    audio_service.export_transcript(transcript_file)
    print(f"  Transcripts exported to: {transcript_file}")

    # Generate Mission Summary
    print("\n[7] Generating Mission Summary...")
    summarizer = MissionSummarizer(
        mission_id=mission_id,
        mission_name=mission_name,
        llm_model="llama3.2"
    )

    # Load data into summarizer
    summarizer.load_events(event_recorder.events)
    summarizer.load_transcripts(audio_service.get_all_transcripts())

    # Generate various analyses
    print("\n  A. Timeline Generation")
    timeline = summarizer.generate_timeline()
    print(f"     {len(timeline)} timeline entries created")

    print("\n  B. Key Moment Identification")
    key_moments = summarizer.identify_key_moments()
    print(f"     {len(key_moments)} key moments identified")
    for moment in key_moments[:3]:
        print(f"       - {moment.get('event_type', moment.get('type'))}: {moment.get('category')}")

    print("\n  C. Crew Performance Analysis")
    crew_performance = summarizer.analyze_crew_performance()
    for crew_member, performance in crew_performance.items():
        if performance['communication_count'] > 0:
            print(f"     {crew_member}: {performance['communication_count']} communications")

    print("\n  D. Tactical Analysis")
    tactical = summarizer.analyze_tactical_performance()
    print(f"     Weapons fired: {tactical['weapons_fired']}")
    print(f"     Hit rate: {tactical['hit_rate']:.1%}")
    print(f"     Combat effectiveness: {tactical['combat_effectiveness']}")

    print("\n  E. Narrative Summary")
    narrative = summarizer.generate_narrative_summary()
    print(f"     Mission outcome: {narrative['outcome']}")
    print(f"     Duration: {narrative['duration']}")

    # Export comprehensive reports
    print("\n[8] Exporting Comprehensive Reports...")

    # JSON Report
    json_report = export_dir / "mission_report.json"
    summarizer.export_report(json_report, format="json")
    print(f"  JSON report: {json_report}")

    # Markdown Report
    md_report = export_dir / "mission_report.md"
    summarizer.export_report(md_report, format="markdown")
    print(f"  Markdown report: {md_report}")

    # HTML Report
    html_report = export_dir / "mission_report.html"
    summarizer.export_report(html_report, format="html")
    print(f"  HTML report: {html_report}")

    # Learning Objectives Assessment
    print("\n[9] Assessing Learning Objectives...")
    learning_objectives = [
        {"id": "OBJ_001", "description": "Navigate to patrol area", "category": "navigation"},
        {"id": "OBJ_002", "description": "Maintain crew communication", "category": "communication"},
        {"id": "OBJ_003", "description": "Handle combat situation", "category": "combat"},
    ]

    assessment = summarizer.assess_learning_objectives(learning_objectives)
    print(f"  Learning objectives completion: {assessment['completion_rate']:.0%}")
    for obj_id, result in assessment.items():
        if obj_id != "completion_rate":
            status = "✓" if result["completed"] else "✗"
            print(f"    {status} {obj_id}: {result['description']}")

    # Generate LLM Summary (mock)
    print("\n[10] Generating AI-Powered Summary...")
    llm_summary = summarizer.generate_llm_summary(
        include_recommendations=True,
        include_learning_points=True
    )
    print("  Summary generated with recommendations and learning points")

    print("\n" + "=" * 60)
    print("MISSION RECORDING DEMO COMPLETE")
    print("=" * 60)
    print(f"\nAll mission data saved to: {export_dir.absolute()}")
    print("\nThe system successfully:")
    print("  ✓ Recorded all game events in real-time")
    print("  ✓ Captured and transcribed bridge audio")
    print("  ✓ Generated comprehensive mission timeline")
    print("  ✓ Identified key tactical moments")
    print("  ✓ Analyzed crew performance metrics")
    print("  ✓ Assessed learning objective completion")
    print("  ✓ Created multiple report formats")
    print("  ✓ Produced AI-powered mission summary")

    return mission_id, export_dir


if __name__ == "__main__":
    try:
        mission_id, export_path = simulate_mission()
        print(f"\n✓ Demo completed successfully!")
        print(f"  View the mission reports in: {export_path}")
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()