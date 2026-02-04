"""
Telemetry timeline builder for mission phase analysis.

Extracts key game events from telemetry data and organizes them into
a phase-by-phase mission timeline that can be used in LLM narratives,
scorecards, and learning evaluations.
"""

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Minimum events to consider a phase meaningful
MIN_PHASE_EVENTS = int(os.getenv('MIN_PHASE_EVENTS', '2'))

# Event categories that indicate mission phases
PHASE_INDICATORS: Dict[str, List[str]] = {
    'departure': [
        'throttle_change', 'heading_change', 'course_set',
        'impulse_change', 'warp_engage',
    ],
    'transit': [
        'heading_change', 'course_set', 'warp_engage',
        'warp_disengage', 'speed_change',
    ],
    'engagement': [
        'weapons_fire', 'shield_change', 'alert_change',
        'target_lock', 'torpedo_fire', 'phaser_fire',
        'damage_report', 'hull_breach',
    ],
    'science_ops': [
        'scan_initiated', 'scan_complete', 'contact_detected',
        'anomaly_detected', 'probe_launched', 'science_scan',
    ],
    'logistics': [
        'cargo_transfer', 'docking_initiated', 'docking_complete',
        'transporter_use', 'shuttle_launch', 'shuttle_return',
        'marines_deployed', 'credits_changed', 'cargo_operation',
    ],
}

# Event types that represent significant game state changes
SIGNIFICANT_EVENT_TYPES = {
    'alert_change', 'weapons_fire', 'shield_change', 'damage_report',
    'hull_breach', 'mission_update', 'contact_detected', 'scan_complete',
    'docking_complete', 'cargo_transfer', 'warp_engage', 'warp_disengage',
    'torpedo_fire', 'phaser_fire', 'target_lock',
    'science_scan', 'cargo_operation',
}

# Categories mapped to human-readable names
CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    'helm': 'Navigation',
    'navigation': 'Navigation',
    'tactical': 'Tactical/Combat',
    'combat': 'Tactical/Combat',
    'science': 'Science/Sensors',
    'sensors': 'Science/Sensors',
    'engineering': 'Engineering',
    'operations': 'Operations',
    'communications': 'Communications',
    'system_alert': 'System Alert',
    'critical': 'Critical Alert',
}


class TelemetryTimelineBuilder:
    """
    Builds a structured mission timeline from telemetry events.

    Extracts key events, identifies mission phases, and produces
    summaries suitable for LLM prompts and analysis output.
    """

    def __init__(self, events: List[Dict[str, Any]]):
        """
        Initialize timeline builder.

        Args:
            events: List of telemetry event dictionaries
        """
        self.events = events or []
        self._sorted_events: Optional[List[Dict[str, Any]]] = None
        logger.info(f"TelemetryTimelineBuilder initialized with {len(self.events)} events")

    def _get_sorted_events(self) -> List[Dict[str, Any]]:
        """Get events sorted by timestamp."""
        if self._sorted_events is None:
            self._sorted_events = sorted(
                self.events,
                key=lambda e: self._get_event_time(e)
            )
        return self._sorted_events

    def _get_event_time(self, event: Dict[str, Any]) -> float:
        """Extract numeric timestamp from event."""
        ts = event.get('relative_time', event.get('timestamp', 0))
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                return dt.timestamp()
            except ValueError:
                return 0.0
        if isinstance(ts, datetime):
            return ts.timestamp()
        return 0.0

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        if seconds < 0:
            return "00:00"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def build_timeline(self) -> Dict[str, Any]:
        """
        Build complete mission timeline.

        Returns:
            Dictionary with timeline phases, key events, and summary
        """
        if not self.events:
            return {
                'phases': [],
                'key_events': [],
                'event_summary': {},
                'duration_seconds': 0,
                'total_events': 0,
            }

        sorted_events = self._get_sorted_events()
        duration = self._calculate_duration(sorted_events)
        key_events = self._extract_key_events(sorted_events)
        phases = self._identify_phases(sorted_events, duration)
        event_summary = self._build_event_summary(sorted_events)

        return {
            'phases': phases,
            'key_events': key_events,
            'event_summary': event_summary,
            'duration_seconds': duration,
            'total_events': len(sorted_events),
        }

    def _calculate_duration(self, sorted_events: List[Dict[str, Any]]) -> float:
        """Calculate mission duration from events."""
        if not sorted_events:
            return 0.0

        first_time = self._get_event_time(sorted_events[0])
        last_time = self._get_event_time(sorted_events[-1])

        # If timestamps are relative (start near 0), use directly
        if first_time < 86400:  # Less than a day in seconds
            return last_time - first_time
        # Absolute timestamps — compute delta
        return last_time - first_time

    def _extract_key_events(
        self,
        sorted_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract significant game events for narrative use.

        Args:
            sorted_events: Events sorted by time

        Returns:
            List of key event dictionaries
        """
        key_events = []
        first_time = self._get_event_time(sorted_events[0]) if sorted_events else 0

        for event in sorted_events:
            event_type = event.get('event_type', 'unknown')
            category = event.get('category', '').lower()
            data = event.get('data', {})

            # Include significant events and events with meaningful data
            is_significant = event_type in SIGNIFICANT_EVENT_TYPES
            has_message = bool(data.get('message') or data.get('Message'))
            has_alert = 'alert' in event_type.lower() or 'alert' in category
            has_mission_data = event_type == 'mission_update'

            if is_significant or has_message or has_alert or has_mission_data:
                event_time = self._get_event_time(event)
                relative_time = event_time - first_time if first_time < 86400 else event_time

                description = self._describe_event(event)
                if description:
                    key_events.append({
                        'time': relative_time,
                        'time_formatted': self._format_time(relative_time),
                        'event_type': event_type,
                        'category': CATEGORY_DISPLAY_NAMES.get(category, category.title()),
                        'description': description,
                        'data': self._sanitize_event_data(data),
                    })

        return key_events[:50]  # Cap at 50 key events

    def _describe_event(self, event: Dict[str, Any]) -> Optional[str]:
        """Generate human-readable description of an event."""
        event_type = event.get('event_type', 'unknown')
        data = event.get('data', {})

        # Mission updates
        if event_type == 'mission_update':
            grade = data.get('Grade')
            name = data.get('Name') or data.get('MissionName', '')
            objectives = data.get('Objectives', {})
            completed = sum(
                1 for obj in objectives.values()
                if isinstance(obj, dict) and obj.get('Complete')
            )
            total = len(objectives)
            parts = []
            if name:
                parts.append(f"Mission: {name}")
            if total > 0:
                parts.append(f"Objectives: {completed}/{total} complete")
            if grade is not None:
                parts.append(f"Grade: {grade}")
            return ' | '.join(parts) if parts else None

        # Alert changes
        if 'alert' in event_type.lower():
            level = data.get('level') or data.get('Level') or data.get('alert_level', '')
            return f"Alert level changed to {level}" if level else "Alert status changed"

        # Combat events
        if event_type in ('weapons_fire', 'torpedo_fire', 'phaser_fire'):
            target = data.get('target', data.get('Target', 'unknown target'))
            return f"Weapons fired at {target}"

        if event_type == 'shield_change':
            level = data.get('level', data.get('Level', ''))
            return f"Shields at {level}%" if level else "Shield status changed"

        if event_type == 'damage_report':
            system = data.get('system', data.get('System', 'unknown'))
            severity = data.get('severity', data.get('Severity', ''))
            return f"Damage to {system}" + (f" ({severity})" if severity else "")

        # Navigation
        if event_type in ('warp_engage', 'warp_disengage'):
            action = 'engaged' if 'engage' in event_type else 'disengaged'
            return f"Warp drive {action}"

        if event_type == 'heading_change':
            heading = data.get('heading', data.get('Heading', ''))
            return f"Course changed to heading {heading}" if heading else "Course adjusted"

        # Science
        if event_type == 'contact_detected':
            contact = data.get('name', data.get('Name', 'unknown'))
            return f"Contact detected: {contact}"

        if event_type == 'scan_complete':
            target = data.get('target', data.get('Target', 'unknown'))
            return f"Scan complete: {target}"

        if event_type == 'science_scan':
            target_id = data.get('target_id', data.get('TargetID', ''))
            target_name = data.get('target_name', data.get('TargetName', ''))
            if target_name:
                return f"Science scan: {target_name}"
            return f"Science scan on target {target_id}" if target_id else "Science scan initiated"

        # Logistics
        if event_type == 'docking_complete':
            station = data.get('station', data.get('Station', 'unknown'))
            return f"Docked at {station}"

        if event_type == 'cargo_transfer':
            item = data.get('item', data.get('Item', 'cargo'))
            qty = data.get('quantity', data.get('Quantity', ''))
            return f"Transferred {qty} {item}" if qty else f"Cargo transfer: {item}"

        if event_type == 'cargo_operation':
            op_type = data.get('operation_type', data.get('OperationType', 'cargo'))
            item = data.get('item', data.get('Item', ''))
            if item:
                return f"Cargo operation ({op_type}): {item}"
            return f"Cargo operation: {op_type}"

        # Channel messages (game system messages)
        message = data.get('message') or data.get('Message') or data.get('Text', '')
        if message:
            return f"Game message: {str(message)[:120]}"

        return None

    def _sanitize_event_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove large/nested data, keep useful summary fields."""
        sanitized = {}
        skip_keys = {'Objectives', 'objectives', 'raw', 'Raw', 'packets'}
        for key, value in data.items():
            if key in skip_keys:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, dict) and len(value) <= 5:
                sanitized[key] = value
        return sanitized

    def _identify_phases(
        self,
        sorted_events: List[Dict[str, Any]],
        duration: float
    ) -> List[Dict[str, Any]]:
        """
        Identify mission phases from event patterns.

        Divides the mission into time windows and classifies each
        based on the dominant event types.

        Args:
            sorted_events: Events sorted by time
            duration: Total mission duration in seconds

        Returns:
            List of phase dictionaries
        """
        if duration <= 0 or not sorted_events:
            return []

        # Divide mission into windows (~2 minutes each, min 3 windows)
        num_windows = max(3, int(duration / 120))
        window_size = duration / num_windows

        first_time = self._get_event_time(sorted_events[0])
        phases = []

        for i in range(num_windows):
            window_start = i * window_size
            window_end = (i + 1) * window_size

            # Collect events in this window
            window_events = [
                e for e in sorted_events
                if window_start <= (self._get_event_time(e) - first_time) < window_end
            ]

            if not window_events:
                continue

            # Classify window by dominant activity
            category_counts: Dict[str, int] = defaultdict(int)
            for e in window_events:
                cat = e.get('category', '').lower()
                category_counts[cat] += 1
                # Also count event types for phase detection
                etype = e.get('event_type', '')
                for phase_name, indicators in PHASE_INDICATORS.items():
                    if etype in indicators:
                        category_counts[f'_phase_{phase_name}'] += 1

            # Determine phase type
            phase_type = self._classify_phase(category_counts)
            display_name = CATEGORY_DISPLAY_NAMES.get(
                max(
                    (k for k in category_counts if not k.startswith('_phase_')),
                    key=lambda k: category_counts[k],
                    default='unknown'
                ),
                phase_type.replace('_', ' ').title()
            )

            phases.append({
                'phase_number': i + 1,
                'start_time': window_start,
                'end_time': window_end,
                'start_formatted': self._format_time(window_start),
                'end_formatted': self._format_time(window_end),
                'phase_type': phase_type,
                'display_name': display_name,
                'event_count': len(window_events),
                'categories': {
                    k: v for k, v in category_counts.items()
                    if not k.startswith('_phase_')
                },
            })

        return phases

    def _classify_phase(self, category_counts: Dict[str, int]) -> str:
        """Classify a time window into a mission phase type."""
        phase_scores: Dict[str, int] = defaultdict(int)
        for key, count in category_counts.items():
            if key.startswith('_phase_'):
                phase_name = key.replace('_phase_', '')
                phase_scores[phase_name] += count

        if not phase_scores:
            # Fall back to raw categories
            tactical_count = sum(
                category_counts.get(c, 0)
                for c in ['tactical', 'combat', 'weapons', 'defensive']
            )
            science_count = sum(
                category_counts.get(c, 0)
                for c in ['science', 'sensors', 'scan']
            )
            nav_count = sum(
                category_counts.get(c, 0)
                for c in ['helm', 'navigation', 'course']
            )
            ops_count = sum(
                category_counts.get(c, 0)
                for c in ['operations', 'cargo', 'docking']
            )

            if tactical_count > science_count and tactical_count > nav_count:
                return 'engagement'
            elif science_count > nav_count:
                return 'science_ops'
            elif ops_count > nav_count:
                return 'logistics'
            elif nav_count > 0:
                return 'transit'
            return 'general'

        return max(phase_scores, key=lambda k: phase_scores[k])

    def _build_event_summary(
        self,
        sorted_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build summary statistics about telemetry events.

        Args:
            sorted_events: Events sorted by time

        Returns:
            Event summary dictionary
        """
        category_counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)

        for event in sorted_events:
            category = event.get('category', 'unknown').lower()
            event_type = event.get('event_type', 'unknown')
            category_counts[category] += 1
            type_counts[event_type] += 1

        # Top event types
        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Category distribution with display names
        category_dist = {
            CATEGORY_DISPLAY_NAMES.get(cat, cat.title()): count
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        }

        return {
            'total_events': len(sorted_events),
            'category_distribution': category_dist,
            'top_event_types': dict(top_types),
            'unique_categories': len(category_counts),
            'unique_event_types': len(type_counts),
        }

    def build_narrative_context(self) -> str:
        """
        Build a text summary of the timeline for LLM narrative prompts.

        Returns:
            Formatted text suitable for inclusion in LLM prompts
        """
        timeline = self.build_timeline()

        if not timeline['key_events']:
            return ""

        sections = []
        sections.append("## Game Telemetry Events (ACTUAL IN-GAME DATA)")
        sections.append(
            f"Total telemetry events recorded: {timeline['total_events']} "
            f"over {self._format_time(timeline['duration_seconds'])}"
        )
        sections.append("")

        # Event category summary
        event_summary = timeline['event_summary']
        if event_summary.get('category_distribution'):
            sections.append("### Activity by Category")
            for cat, count in event_summary['category_distribution'].items():
                sections.append(f"- {cat}: {count} events")
            sections.append("")

        # Mission phases
        if timeline['phases']:
            sections.append("### Mission Phases (from telemetry)")
            for phase in timeline['phases']:
                sections.append(
                    f"- **{phase['start_formatted']}-{phase['end_formatted']}**: "
                    f"{phase['display_name']} ({phase['event_count']} events)"
                )
            sections.append("")

        # Key events (limit to most important for prompt size)
        key_events = timeline['key_events']
        if key_events:
            sections.append("### Key Game Events (USE THESE to ground the narrative)")
            for event in key_events[:20]:
                sections.append(
                    f"- [{event['time_formatted']}] {event['description']}"
                )
            if len(key_events) > 20:
                sections.append(f"  ... and {len(key_events) - 20} more events")
            sections.append("")

        return '\n'.join(sections)

    def build_story_events(self, max_events: int = 25) -> List[Dict[str, Any]]:
        """
        Build deduplicated story events by grouping consecutive similar events.

        Consecutive events of the same type are collapsed into a single entry
        (e.g., "12 science scans over 2 minutes" instead of 12 separate entries).
        This keeps the story prompt concise.

        Args:
            max_events: Maximum number of grouped events to return

        Returns:
            List of grouped event dictionaries with description and time range
        """
        timeline = self.build_timeline()
        key_events = timeline.get('key_events', [])

        if not key_events:
            return []

        grouped: List[Dict[str, Any]] = []
        current_group: Optional[Dict[str, Any]] = None

        for event in key_events:
            event_type = event.get('event_type', 'unknown')

            if (current_group and
                    current_group['event_type'] == event_type):
                # Same type as current group — extend it
                current_group['count'] += 1
                current_group['end_time'] = event['time']
                current_group['end_formatted'] = event['time_formatted']
                # Keep latest description as representative
                current_group['last_description'] = event['description']
            else:
                # Different type — flush current group and start new one
                if current_group:
                    grouped.append(self._format_group(current_group))
                current_group = {
                    'event_type': event_type,
                    'category': event.get('category', ''),
                    'count': 1,
                    'start_time': event['time'],
                    'end_time': event['time'],
                    'start_formatted': event['time_formatted'],
                    'end_formatted': event['time_formatted'],
                    'first_description': event['description'],
                    'last_description': event['description'],
                }

        # Flush last group
        if current_group:
            grouped.append(self._format_group(current_group))

        return grouped[:max_events]

    def _format_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a group of consecutive events into a single story event.

        Args:
            group: Group dictionary with count, times, and descriptions

        Returns:
            Formatted event dictionary
        """
        count = group['count']
        event_type = group['event_type']

        if count == 1:
            description = group['first_description']
            time_str = group['start_formatted']
        else:
            # Build a summary description
            type_label = event_type.replace('_', ' ')
            duration_secs = group['end_time'] - group['start_time']
            if duration_secs > 60:
                duration_str = f"{duration_secs / 60:.0f} minutes"
            else:
                duration_str = f"{duration_secs:.0f} seconds"
            description = (
                f"{count} {type_label} events over {duration_str} "
                f"(e.g., {group['first_description']})"
            )
            time_str = f"{group['start_formatted']}-{group['end_formatted']}"

        return {
            'time_formatted': time_str,
            'event_type': event_type,
            'category': group['category'],
            'description': description,
            'count': count,
        }

    def build_telemetry_summary(self) -> Dict[str, Any]:
        """
        Build a telemetry summary for inclusion in analysis output JSON.

        Returns:
            Summary dictionary for the output JSON
        """
        timeline = self.build_timeline()

        return {
            'total_events': timeline['total_events'],
            'duration_seconds': timeline['duration_seconds'],
            'duration_formatted': self._format_time(timeline['duration_seconds']),
            'phases': timeline['phases'],
            'key_events': timeline['key_events'],
            'event_summary': timeline['event_summary'],
        }
