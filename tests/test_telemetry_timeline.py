"""Tests for telemetry timeline builder module."""

import pytest
from src.metrics.telemetry_timeline import TelemetryTimelineBuilder


class TestTelemetryTimelineBuilder:
    """Test suite for TelemetryTimelineBuilder."""

    @pytest.fixture
    def sample_events(self):
        """Create sample telemetry events for testing."""
        return [
            {
                'event_type': 'alert_change',
                'category': 'tactical',
                'relative_time': 10.0,
                'data': {'level': 'Yellow Alert'},
            },
            {
                'event_type': 'heading_change',
                'category': 'helm',
                'relative_time': 30.0,
                'data': {'heading': '045'},
            },
            {
                'event_type': 'science_scan',
                'category': 'science',
                'relative_time': 45.0,
                'data': {'target_id': 11919},
            },
            {
                'event_type': 'scan_complete',
                'category': 'science',
                'relative_time': 60.0,
                'data': {'target': 'Asteroid Field'},
            },
            {
                'event_type': 'contact_detected',
                'category': 'science',
                'relative_time': 90.0,
                'data': {'name': 'USS Enterprise'},
            },
            {
                'event_type': 'cargo_operation',
                'category': 'operations',
                'relative_time': 120.0,
                'data': {'operation_type': 'CARGO'},
            },
            {
                'event_type': 'weapons_fire',
                'category': 'tactical',
                'relative_time': 150.0,
                'data': {'target': 'Enemy Vessel'},
            },
            {
                'event_type': 'shield_change',
                'category': 'tactical',
                'relative_time': 155.0,
                'data': {'level': '80'},
            },
            {
                'event_type': 'mission_update',
                'category': 'system_alert',
                'relative_time': 200.0,
                'data': {
                    'Name': 'Patrol Mission',
                    'Grade': 0.85,
                    'Objectives': {
                        'obj1': {'Description': 'Scan asteroid', 'Complete': True},
                        'obj2': {'Description': 'Deliver cargo', 'Complete': False},
                    },
                },
            },
            {
                'event_type': 'docking_complete',
                'category': 'operations',
                'relative_time': 250.0,
                'data': {'station': 'Starbase Alpha'},
            },
        ]

    @pytest.fixture
    def builder(self, sample_events):
        """Create a TelemetryTimelineBuilder with sample events."""
        return TelemetryTimelineBuilder(sample_events)

    def test_initialization(self, builder, sample_events):
        """Test builder initializes with correct event count."""
        assert len(builder.events) == len(sample_events)

    def test_initialization_empty(self):
        """Test builder handles empty events."""
        builder = TelemetryTimelineBuilder([])
        timeline = builder.build_timeline()
        assert timeline['total_events'] == 0
        assert timeline['phases'] == []
        assert timeline['key_events'] == []

    def test_initialization_none(self):
        """Test builder handles None events."""
        builder = TelemetryTimelineBuilder(None)
        assert len(builder.events) == 0

    def test_build_timeline(self, builder):
        """Test building complete timeline."""
        timeline = builder.build_timeline()

        assert timeline['total_events'] == 10
        assert timeline['duration_seconds'] > 0
        assert len(timeline['key_events']) > 0
        assert len(timeline['phases']) > 0
        assert 'event_summary' in timeline

    def test_key_events_extraction(self, builder):
        """Test that key events are extracted correctly."""
        timeline = builder.build_timeline()
        key_events = timeline['key_events']

        # All our sample events are significant types (including science_scan, cargo_operation)
        assert len(key_events) >= 7

        # Check event descriptions
        descriptions = [e['description'] for e in key_events]
        assert any('Alert' in d for d in descriptions)
        assert any('Weapons' in d or 'fired' in d for d in descriptions)
        assert any('Docked' in d or 'Starbase' in d for d in descriptions)
        assert any('scan' in d.lower() for d in descriptions)
        assert any('cargo' in d.lower() or 'CARGO' in d for d in descriptions)

    def test_key_events_have_formatted_time(self, builder):
        """Test that key events have formatted timestamps."""
        timeline = builder.build_timeline()
        for event in timeline['key_events']:
            assert 'time_formatted' in event
            assert ':' in event['time_formatted']

    def test_mission_update_description(self, builder):
        """Test mission update event description."""
        timeline = builder.build_timeline()
        key_events = timeline['key_events']

        mission_events = [e for e in key_events if 'Mission' in e.get('description', '')]
        assert len(mission_events) >= 1
        assert 'Objectives' in mission_events[0]['description']

    def test_event_summary(self, builder):
        """Test event summary statistics."""
        timeline = builder.build_timeline()
        summary = timeline['event_summary']

        assert summary['total_events'] == 10
        assert summary['unique_categories'] > 0
        assert summary['unique_event_types'] > 0
        assert 'category_distribution' in summary
        assert 'top_event_types' in summary

    def test_phases_cover_timeline(self, builder):
        """Test that phases cover the mission timeline."""
        timeline = builder.build_timeline()
        phases = timeline['phases']

        if phases:
            assert phases[0]['start_time'] == 0
            assert phases[-1]['end_time'] >= timeline['duration_seconds'] * 0.9

    def test_narrative_context(self, builder):
        """Test narrative context generation."""
        context = builder.build_narrative_context()

        assert isinstance(context, str)
        assert len(context) > 0
        assert 'Game Telemetry Events' in context
        assert 'Key Game Events' in context

    def test_narrative_context_empty(self):
        """Test narrative context with no events."""
        builder = TelemetryTimelineBuilder([])
        context = builder.build_narrative_context()
        assert context == ""

    def test_telemetry_summary(self, builder):
        """Test telemetry summary for output JSON."""
        summary = builder.build_telemetry_summary()

        assert summary['total_events'] == 10
        assert summary['duration_seconds'] > 0
        assert 'duration_formatted' in summary
        assert 'phases' in summary
        assert 'key_events' in summary
        assert 'event_summary' in summary

    def test_format_time(self, builder):
        """Test time formatting."""
        assert builder._format_time(0) == "00:00"
        assert builder._format_time(65) == "01:05"
        assert builder._format_time(3661) == "61:01"
        assert builder._format_time(-1) == "00:00"

    def test_describe_alert_event(self, builder):
        """Test alert event description."""
        event = {
            'event_type': 'alert_change',
            'category': 'tactical',
            'data': {'level': 'Red Alert'},
        }
        desc = builder._describe_event(event)
        assert 'Red Alert' in desc

    def test_describe_weapons_event(self, builder):
        """Test weapons event description."""
        event = {
            'event_type': 'weapons_fire',
            'category': 'tactical',
            'data': {'target': 'Klingon Bird of Prey'},
        }
        desc = builder._describe_event(event)
        assert 'Klingon Bird of Prey' in desc

    def test_describe_message_event(self, builder):
        """Test event with message data."""
        event = {
            'event_type': 'channel_message',
            'category': 'communications',
            'data': {'message': 'All hands, prepare for departure'},
        }
        desc = builder._describe_event(event)
        assert 'All hands' in desc


class TestTelemetryTimelineEdgeCases:
    """Test edge cases for TelemetryTimelineBuilder."""

    def test_single_event(self):
        """Test with a single event."""
        events = [{
            'event_type': 'alert_change',
            'category': 'tactical',
            'relative_time': 5.0,
            'data': {'level': 'Yellow'},
        }]
        builder = TelemetryTimelineBuilder(events)
        timeline = builder.build_timeline()
        assert timeline['total_events'] == 1
        assert len(timeline['key_events']) == 1

    def test_string_timestamps(self):
        """Test with ISO string timestamps."""
        events = [{
            'event_type': 'alert_change',
            'category': 'tactical',
            'timestamp': '2026-01-28T15:00:00',
            'data': {'level': 'Red'},
        }]
        builder = TelemetryTimelineBuilder(events)
        timeline = builder.build_timeline()
        assert timeline['total_events'] == 1

    def test_events_without_data(self):
        """Test events that have no data field."""
        events = [{
            'event_type': 'unknown_event',
            'category': 'unknown',
            'relative_time': 10.0,
        }]
        builder = TelemetryTimelineBuilder(events)
        timeline = builder.build_timeline()
        assert timeline['total_events'] == 1


class TestNewEventTypes:
    """Test science_scan and cargo_operation event types."""

    def test_describe_science_scan_with_target_id(self):
        """Test science_scan event description with target ID."""
        builder = TelemetryTimelineBuilder([])
        event = {
            'event_type': 'science_scan',
            'category': 'science',
            'data': {'target_id': 11919},
        }
        desc = builder._describe_event(event)
        assert desc is not None
        assert '11919' in desc

    def test_describe_science_scan_with_name(self):
        """Test science_scan event description with target name."""
        builder = TelemetryTimelineBuilder([])
        event = {
            'event_type': 'science_scan',
            'category': 'science',
            'data': {'target_name': 'Comm Station Alpha'},
        }
        desc = builder._describe_event(event)
        assert 'Comm Station Alpha' in desc

    def test_describe_science_scan_no_data(self):
        """Test science_scan event description with no target."""
        builder = TelemetryTimelineBuilder([])
        event = {
            'event_type': 'science_scan',
            'category': 'science',
            'data': {},
        }
        desc = builder._describe_event(event)
        assert desc is not None
        assert 'scan' in desc.lower()

    def test_describe_cargo_operation(self):
        """Test cargo_operation event description."""
        builder = TelemetryTimelineBuilder([])
        event = {
            'event_type': 'cargo_operation',
            'category': 'operations',
            'data': {'operation_type': 'CARGO'},
        }
        desc = builder._describe_event(event)
        assert desc is not None
        assert 'CARGO' in desc

    def test_describe_cargo_operation_with_item(self):
        """Test cargo_operation event description with item."""
        builder = TelemetryTimelineBuilder([])
        event = {
            'event_type': 'cargo_operation',
            'category': 'operations',
            'data': {'operation_type': 'CARGO', 'item': 'Medical Supplies'},
        }
        desc = builder._describe_event(event)
        assert 'Medical Supplies' in desc

    def test_science_scan_in_significant_types(self):
        """Test that science_scan is in SIGNIFICANT_EVENT_TYPES."""
        from src.metrics.telemetry_timeline import SIGNIFICANT_EVENT_TYPES
        assert 'science_scan' in SIGNIFICANT_EVENT_TYPES

    def test_cargo_operation_in_significant_types(self):
        """Test that cargo_operation is in SIGNIFICANT_EVENT_TYPES."""
        from src.metrics.telemetry_timeline import SIGNIFICANT_EVENT_TYPES
        assert 'cargo_operation' in SIGNIFICANT_EVENT_TYPES

    def test_science_scan_in_phase_indicators(self):
        """Test that science_scan is in science_ops phase indicators."""
        from src.metrics.telemetry_timeline import PHASE_INDICATORS
        assert 'science_scan' in PHASE_INDICATORS['science_ops']

    def test_cargo_operation_in_phase_indicators(self):
        """Test that cargo_operation is in logistics phase indicators."""
        from src.metrics.telemetry_timeline import PHASE_INDICATORS
        assert 'cargo_operation' in PHASE_INDICATORS['logistics']

    def test_science_scan_extracted_as_key_event(self):
        """Test that science_scan events are extracted as key events."""
        events = [
            {
                'event_type': 'science_scan',
                'category': 'science',
                'relative_time': 25.0,
                'data': {'target_id': 11919},
            },
            {
                'event_type': 'science_scan',
                'category': 'science',
                'relative_time': 27.0,
                'data': {'target_id': 3},
            },
        ]
        builder = TelemetryTimelineBuilder(events)
        timeline = builder.build_timeline()
        key_events = timeline['key_events']
        assert len(key_events) == 2
        assert all(e['event_type'] == 'science_scan' for e in key_events)

    def test_cargo_operation_extracted_as_key_event(self):
        """Test that cargo_operation events are extracted as key events."""
        events = [{
            'event_type': 'cargo_operation',
            'category': 'operations',
            'relative_time': 5.0,
            'data': {'operation_type': 'CARGO'},
        }]
        builder = TelemetryTimelineBuilder(events)
        timeline = builder.build_timeline()
        key_events = timeline['key_events']
        assert len(key_events) == 1
        assert key_events[0]['event_type'] == 'cargo_operation'


class TestBuildStoryEvents:
    """Test build_story_events deduplication method."""

    def test_groups_consecutive_same_type(self):
        """Test that consecutive same-type events are grouped."""
        events = [
            {'event_type': 'science_scan', 'category': 'science', 'relative_time': 25.0, 'data': {'target_id': 1}},
            {'event_type': 'science_scan', 'category': 'science', 'relative_time': 27.0, 'data': {'target_id': 2}},
            {'event_type': 'science_scan', 'category': 'science', 'relative_time': 30.0, 'data': {'target_id': 3}},
            {'event_type': 'alert_change', 'category': 'tactical', 'relative_time': 60.0, 'data': {'level': 'Red'}},
        ]
        builder = TelemetryTimelineBuilder(events)
        story_events = builder.build_story_events()

        # 3 science scans should be grouped into 1, plus 1 alert = 2 total
        assert len(story_events) == 2
        # First entry should mention count
        assert story_events[0]['count'] == 3
        assert 'science scan' in story_events[0]['description'].lower()

    def test_single_events_not_grouped(self):
        """Test that single events are not grouped."""
        events = [
            {'event_type': 'alert_change', 'category': 'tactical', 'relative_time': 10.0, 'data': {'level': 'Yellow'}},
            {'event_type': 'weapons_fire', 'category': 'tactical', 'relative_time': 60.0, 'data': {'target': 'Ship'}},
        ]
        builder = TelemetryTimelineBuilder(events)
        story_events = builder.build_story_events()

        assert len(story_events) == 2
        assert story_events[0]['count'] == 1
        assert story_events[1]['count'] == 1

    def test_non_consecutive_same_type_not_grouped(self):
        """Test that non-consecutive same-type events are separate."""
        events = [
            {'event_type': 'science_scan', 'category': 'science', 'relative_time': 25.0, 'data': {'target_id': 1}},
            {'event_type': 'alert_change', 'category': 'tactical', 'relative_time': 40.0, 'data': {'level': 'Red'}},
            {'event_type': 'science_scan', 'category': 'science', 'relative_time': 90.0, 'data': {'target_id': 2}},
        ]
        builder = TelemetryTimelineBuilder(events)
        story_events = builder.build_story_events()

        # Should be 3 entries since the science scans are not consecutive
        assert len(story_events) == 3

    def test_empty_events(self):
        """Test build_story_events with no events."""
        builder = TelemetryTimelineBuilder([])
        story_events = builder.build_story_events()
        assert story_events == []

    def test_max_events_limit(self):
        """Test that max_events limits output."""
        events = [
            {'event_type': f'type_{i}', 'category': 'test', 'relative_time': float(i * 10), 'data': {'message': f'Event {i}'}}
            for i in range(30)
        ]
        builder = TelemetryTimelineBuilder(events)
        story_events = builder.build_story_events(max_events=5)
        assert len(story_events) <= 5
