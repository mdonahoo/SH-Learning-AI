"""
Tests for the Telemetry-Audio Correlator Module.

Tests correlation between game telemetry events and audio segments
for enhanced speaker role identification.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.integration.telemetry_audio_correlator import (
    TelemetryAudioCorrelator,
    CorrelationEvidence,
    RoleConfidenceUpdate,
    BridgeRole,
    CATEGORY_ROLE_MAP,
)


class TestTelemetryAudioCorrelator:
    """Test suite for TelemetryAudioCorrelator."""

    @pytest.fixture
    def correlator(self):
        """Create a test correlator instance."""
        return TelemetryAudioCorrelator(
            correlation_window_ms=500,
            min_confidence_boost=0.1,
            max_confidence_boost=0.3
        )

    @pytest.fixture
    def sample_events(self):
        """Create sample telemetry events."""
        return [
            {
                "event_id": "E001",
                "event_type": "throttle_change",
                "category": "helm",
                "timestamp": 12.3,
                "data": {"throttle": 0.5}
            },
            {
                "event_id": "E002",
                "event_type": "course_change",
                "category": "navigation",
                "timestamp": 45.1,
                "data": {"heading": 180}
            },
            {
                "event_id": "E003",
                "event_type": "weapons_fire",
                "category": "tactical",
                "timestamp": 60.0,
                "data": {"weapon": "phaser"}
            },
            {
                "event_id": "E004",
                "event_type": "scan_complete",
                "category": "science",
                "timestamp": 75.5,
                "data": {"target": "asteroid"}
            },
            {
                "event_id": "E005",
                "event_type": "power_reroute",
                "category": "engineering",
                "timestamp": 90.0,
                "data": {"from": "shields", "to": "weapons"}
            },
        ]

    @pytest.fixture
    def sample_transcripts(self):
        """Create sample audio transcripts with speaker IDs."""
        return [
            {
                "speaker_id": "Speaker_1",
                "text": "Setting course for the asteroid field",
                "start_time": 12.1,
                "end_time": 14.5,
                "confidence": 0.85
            },
            {
                "speaker_id": "Speaker_1",
                "text": "Adjusting heading to 180 mark 5",
                "start_time": 45.0,
                "end_time": 47.2,
                "confidence": 0.90
            },
            {
                "speaker_id": "Speaker_2",
                "text": "Firing phasers on the enemy vessel",
                "start_time": 59.8,
                "end_time": 62.0,
                "confidence": 0.88
            },
            {
                "speaker_id": "Speaker_3",
                "text": "Scan complete, no life signs detected",
                "start_time": 75.3,
                "end_time": 77.5,
                "confidence": 0.82
            },
            {
                "speaker_id": "Speaker_4",
                "text": "Rerouting power to weapons",
                "start_time": 89.8,
                "end_time": 91.5,
                "confidence": 0.79
            },
        ]

    def test_initialization(self, correlator):
        """Test correlator initializes with correct configuration."""
        assert correlator.correlation_window_ms == 500
        assert correlator.min_confidence_boost == 0.1
        assert correlator.max_confidence_boost == 0.3
        assert correlator.events == []
        assert correlator.transcripts == []

    def test_load_data(self, correlator, sample_events, sample_transcripts):
        """Test loading events and transcripts."""
        correlator.load_data(sample_events, sample_transcripts)

        assert len(correlator.events) == 5
        assert len(correlator.transcripts) == 5

    def test_correlation_within_window(self, correlator, sample_events, sample_transcripts):
        """Test that events within the time window are correlated."""
        correlator.load_data(sample_events, sample_transcripts)
        correlations = correlator.correlate_all()

        # Should find correlations for events that match with nearby transcripts
        assert len(correlations) > 0

        # Check that helm event E001 correlates with Speaker_1
        helm_correlations = [c for c in correlations if c.event_id == "E001"]
        assert len(helm_correlations) > 0
        assert helm_correlations[0].speaker_id == "Speaker_1"
        assert helm_correlations[0].expected_role == "Helm/Navigation"

    def test_no_correlation_outside_window(self, correlator):
        """Test that events outside the time window are not correlated."""
        # Event at t=0, transcript at t=10 (10 seconds apart, way outside 500ms window)
        events = [{"event_id": "E001", "event_type": "test", "category": "helm", "timestamp": 0.0}]
        transcripts = [{"speaker_id": "Speaker_1", "text": "test", "start_time": 10.0}]

        correlator.load_data(events, transcripts)
        correlations = correlator.correlate_all()

        assert len(correlations) == 0

    def test_multiple_speakers_in_window(self, correlator):
        """Test handling of multiple speakers near the same event."""
        events = [{"event_id": "E001", "event_type": "test", "category": "helm", "timestamp": 10.0}]
        transcripts = [
            {"speaker_id": "Speaker_1", "text": "first speaker", "start_time": 9.8},
            {"speaker_id": "Speaker_2", "text": "second speaker", "start_time": 10.2},
        ]

        correlator.load_data(events, transcripts)
        correlations = correlator.correlate_all()

        # Both speakers should have correlations
        speaker_ids = set(c.speaker_id for c in correlations)
        assert "Speaker_1" in speaker_ids
        assert "Speaker_2" in speaker_ids

    def test_confidence_boost_calculation(self, correlator, sample_events, sample_transcripts):
        """Test that confidence boost is calculated correctly."""
        correlator.load_data(sample_events, sample_transcripts)
        correlator.correlate_all()

        existing_roles = {
            "Speaker_1": {"role": "Helm/Navigation", "confidence": 0.72},
            "Speaker_2": {"role": "Tactical/Weapons", "confidence": 0.68},
            "Speaker_3": {"role": "Science/Sensors", "confidence": 0.75},
            "Speaker_4": {"role": "Engineering/Systems", "confidence": 0.65},
        }

        updates = correlator.update_role_confidences(existing_roles)

        # Check that Speaker_1 (Helm) got a boost
        assert "Speaker_1" in updates
        update = updates["Speaker_1"]
        assert update.boosted_confidence > update.base_confidence
        assert update.boosted_confidence <= 1.0
        assert update.evidence_count > 0

    def test_evidence_trail_generation(self, correlator, sample_events, sample_transcripts):
        """Test that evidence trails are generated correctly."""
        correlator.load_data(sample_events, sample_transcripts)
        correlator.correlate_all()

        existing_roles = {
            "Speaker_1": {"role": "Helm/Navigation", "confidence": 0.72},
        }

        updates = correlator.update_role_confidences(existing_roles)

        if "Speaker_1" in updates:
            update = updates["Speaker_1"]
            assert isinstance(update.evidence_trail, list)

            if update.evidence_trail:
                evidence = update.evidence_trail[0]
                assert isinstance(evidence, CorrelationEvidence)
                assert evidence.speaker_id == "Speaker_1"
                assert evidence.event_id is not None
                assert evidence.time_delta_ms >= 0

    def test_unknown_event_category(self, correlator):
        """Test handling of events with unknown categories."""
        events = [
            {"event_id": "E001", "event_type": "unknown_type", "category": "unknown_category", "timestamp": 10.0}
        ]
        transcripts = [{"speaker_id": "Speaker_1", "text": "test", "start_time": 10.0}]

        correlator.load_data(events, transcripts)
        correlations = correlator.correlate_all()

        # Unknown categories should not produce correlations
        assert len(correlations) == 0

    def test_empty_data_handling(self, correlator):
        """Test handling of empty events or transcripts."""
        # Empty events
        correlator.load_data([], [{"speaker_id": "S1", "text": "test", "start_time": 0}])
        correlations = correlator.correlate_all()
        assert len(correlations) == 0

        # Empty transcripts
        correlator.load_data([{"event_id": "E1", "category": "helm", "timestamp": 0}], [])
        correlations = correlator.correlate_all()
        assert len(correlations) == 0

        # Both empty
        correlator.load_data([], [])
        correlations = correlator.correlate_all()
        assert len(correlations) == 0

    def test_confidence_not_exceed_one(self, correlator):
        """Test that boosted confidence never exceeds 1.0."""
        # Create many correlating events to maximize boost
        events = [
            {"event_id": f"E{i}", "event_type": "throttle", "category": "helm", "timestamp": float(i)}
            for i in range(20)
        ]
        transcripts = [
            {"speaker_id": "Speaker_1", "text": "helm command", "start_time": float(i)}
            for i in range(20)
        ]

        correlator.load_data(events, transcripts)
        correlator.correlate_all()

        existing_roles = {
            "Speaker_1": {"role": "Helm/Navigation", "confidence": 0.95}
        }

        updates = correlator.update_role_confidences(existing_roles)

        assert "Speaker_1" in updates
        assert updates["Speaker_1"].boosted_confidence <= 1.0

    def test_realtime_correlation(self, correlator):
        """Test real-time single event correlation."""
        event = {
            "event_id": "RT001",
            "event_type": "throttle_change",
            "category": "helm",
            "relative_time": 10.0
        }
        recent_segments = [
            {"speaker_id": "Speaker_1", "text": "engaging", "start_time": 9.9},
            {"speaker_id": "Speaker_2", "text": "confirmed", "start_time": 10.5},
        ]

        evidence = correlator.correlate_realtime(event, recent_segments)

        assert evidence is not None
        assert evidence.speaker_id in ["Speaker_1", "Speaker_2"]
        assert evidence.expected_role == "Helm/Navigation"

    def test_category_role_mapping(self):
        """Test that category-to-role mapping is correct."""
        assert CATEGORY_ROLE_MAP["helm"] == BridgeRole.HELM
        assert CATEGORY_ROLE_MAP["tactical"] == BridgeRole.TACTICAL
        assert CATEGORY_ROLE_MAP["science"] == BridgeRole.SCIENCE
        assert CATEGORY_ROLE_MAP["engineering"] == BridgeRole.ENGINEERING
        assert CATEGORY_ROLE_MAP["operations"] == BridgeRole.OPERATIONS

    def test_correlation_summary(self, correlator, sample_events, sample_transcripts):
        """Test the correlation summary generation."""
        correlator.load_data(sample_events, sample_transcripts)
        correlator.correlate_all()

        summary = correlator.get_correlation_summary()

        assert "total_correlations" in summary
        assert "speakers_with_evidence" in summary
        assert "role_distribution" in summary
        assert "event_types" in summary
        assert "average_time_delta_ms" in summary

    def test_integrate_with_role_inference(self, correlator, sample_events, sample_transcripts):
        """Test integration with RoleInferenceEngine results."""
        correlator.load_data(sample_events, sample_transcripts)

        role_results = {
            'role_table': '| Speaker | Role |',
            'methodology': 'Keyword analysis',
            'speaker_roles': {
                'Speaker_1': {
                    'role': 'Helm/Navigation',
                    'confidence': 0.72,
                    'utterance_count': 10,
                    'key_indicators': ['course', 'heading']
                },
                'Speaker_2': {
                    'role': 'Tactical/Weapons',
                    'confidence': 0.68,
                    'utterance_count': 8,
                    'key_indicators': ['fire', 'target']
                }
            }
        }

        enhanced = correlator.integrate_with_role_inference(role_results)

        assert 'speaker_roles' in enhanced
        assert 'telemetry_correlation' in enhanced

        # Check enhanced speaker roles
        speaker_1 = enhanced['speaker_roles'].get('Speaker_1', {})
        if speaker_1.get('evidence_count', 0) > 0:
            assert 'voice_confidence' in speaker_1
            assert 'telemetry_confidence' in speaker_1
            assert speaker_1['confidence'] >= speaker_1['voice_confidence']

    def test_speaker_evidence_retrieval(self, correlator, sample_events, sample_transcripts):
        """Test retrieving evidence for a specific speaker."""
        correlator.load_data(sample_events, sample_transcripts)
        correlator.correlate_all()

        evidence = correlator.get_speaker_evidence("Speaker_1")
        assert isinstance(evidence, list)

        # Check unknown speaker
        no_evidence = correlator.get_speaker_evidence("Unknown_Speaker")
        assert len(no_evidence) == 0

    def test_clear(self, correlator, sample_events, sample_transcripts):
        """Test clearing all data."""
        correlator.load_data(sample_events, sample_transcripts)
        correlator.correlate_all()

        assert len(correlator._correlations) > 0

        correlator.clear()

        assert len(correlator.events) == 0
        assert len(correlator.transcripts) == 0
        assert len(correlator._correlations) == 0

    def test_datetime_timestamp_handling(self, correlator):
        """Test handling of datetime timestamps."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        events = [
            {
                "event_id": "E001",
                "event_type": "throttle_change",
                "category": "helm",
                "timestamp": base_time + timedelta(seconds=10)
            }
        ]
        transcripts = [
            {
                "speaker_id": "Speaker_1",
                "text": "engaging throttle",
                "start_time": 10.0
            }
        ]

        correlator.load_data(events, transcripts, mission_start=base_time)
        correlations = correlator.correlate_all()

        # Should handle datetime conversion properly
        assert len(correlator.events) == 1
        assert 'relative_time' in correlator.events[0]

    def test_roles_match_function(self, correlator):
        """Test the role matching logic."""
        assert correlator._roles_match("Helm/Navigation", "Helm/Navigation") is True
        assert correlator._roles_match("Helm/Navigation", "helm") is True
        assert correlator._roles_match("Tactical/Weapons", "Tactical/Weapons") is True
        assert correlator._roles_match("Tactical", "Tactical/Weapons") is True
        assert correlator._roles_match("Science/Sensors", "Science/Sensors") is True

        # Non-matching roles
        assert correlator._roles_match("Helm/Navigation", "Tactical/Weapons") is False
        assert correlator._roles_match("Science", "Engineering") is False


class TestCorrelationEvidence:
    """Test suite for CorrelationEvidence dataclass."""

    def test_evidence_creation(self):
        """Test creating a CorrelationEvidence instance."""
        evidence = CorrelationEvidence(
            event_id="E001",
            event_type="throttle_change",
            event_category="helm",
            speaker_id="Speaker_1",
            expected_role="Helm/Navigation",
            time_delta_ms=100.0,
            confidence_contribution=0.2,
            event_timestamp=10.0,
            segment_timestamp=10.1,
            segment_text="setting course"
        )

        assert evidence.event_id == "E001"
        assert evidence.speaker_id == "Speaker_1"
        assert evidence.time_delta_ms == 100.0
        assert evidence.confidence_contribution == 0.2


class TestRoleConfidenceUpdate:
    """Test suite for RoleConfidenceUpdate dataclass."""

    def test_update_creation(self):
        """Test creating a RoleConfidenceUpdate instance."""
        update = RoleConfidenceUpdate(
            speaker_id="Speaker_1",
            role="Helm/Navigation",
            base_confidence=0.72,
            boosted_confidence=0.91,
            evidence_count=8,
            evidence_trail=[],
            telemetry_boost=0.19,
            methodology_note="Test note"
        )

        assert update.speaker_id == "Speaker_1"
        assert update.base_confidence == 0.72
        assert update.boosted_confidence == 0.91
        assert update.evidence_count == 8
        assert update.telemetry_boost == 0.19
