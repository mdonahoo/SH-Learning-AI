#!/usr/bin/env python3
"""
Event Recorder Module for Starship Horizons Learning AI
Records all game events, crew actions, and system states during missions.
"""

import json
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


class EventRecorder:
    """Records and manages game events during a mission."""

    def __init__(self, mission_id: str, mission_name: str = "", bridge_crew: List[str] = None):
        """
        Initialize the event recorder for a mission.

        Args:
            mission_id: Unique identifier for the mission
            mission_name: Human-readable mission name
            bridge_crew: List of crew positions
        """
        self.mission_id = mission_id
        self.mission_name = mission_name
        self.bridge_crew = bridge_crew or []
        self.start_time = datetime.now()
        self.end_time = None
        self.events = []
        self._lock = threading.RLock()  # Reentrant lock to avoid deadlock
        self._event_counter = 0

    def record_event(self, event_type: str, category: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a game event.

        Args:
            event_type: Type of event (e.g., 'ship_action', 'system_status')
            category: Category of event (e.g., 'navigation', 'combat')
            data: Event-specific data

        Returns:
            The recorded event with metadata
        """
        with self._lock:
            self._event_counter += 1
            event = {
                "event_id": f"{self.mission_id}_E{self._event_counter:05d}",
                "timestamp": datetime.now(),
                "event_type": event_type,
                "category": category,
                "data": data
            }
            self.events.append(event)
            return event

    def record_communication(self, speaker: str, message: str,
                            audio_file: str = None, confidence: float = None) -> Dict[str, Any]:
        """
        Record crew communication.

        Args:
            speaker: Who is speaking
            message: What was said
            audio_file: Path to audio recording
            confidence: Speech recognition confidence

        Returns:
            The recorded communication event
        """
        comm_data = {
            "speaker": speaker,
            "message": message
        }

        if audio_file:
            comm_data["audio_file"] = audio_file
        if confidence is not None:
            comm_data["confidence"] = confidence

        return self.record_event(
            event_type="communication",
            category="crew_communication",
            data=comm_data
        )

    def record_ship_state(self, ship_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record comprehensive ship state snapshot.

        Args:
            ship_data: Complete ship telemetry data

        Returns:
            The recorded ship state event
        """
        return self.record_event(
            event_type="ship_state_snapshot",
            category="telemetry",
            data=ship_data
        )

    def record_damage(self, damage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record damage event with detailed tracking.

        Args:
            damage_data: Damage information including amount, type, affected systems

        Returns:
            The recorded damage event
        """
        return self.record_event(
            event_type="damage_taken",
            category="combat",
            data=damage_data
        )

    def record_system_change(self, system: str, old_value: Any, new_value: Any,
                           is_critical: bool = False) -> Dict[str, Any]:
        """
        Record system status change.

        Args:
            system: System that changed
            old_value: Previous value
            new_value: New value
            is_critical: Whether this is a critical change

        Returns:
            The recorded system change event
        """
        return self.record_event(
            event_type="system_change",
            category="engineering" if not is_critical else "critical",
            data={
                "system": system,
                "old_value": old_value,
                "new_value": new_value,
                "is_critical": is_critical
            }
        )

    def record_shield_status(self, shield_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record shield status including quadrant data.

        Args:
            shield_data: Shield information including quadrants

        Returns:
            The recorded shield event
        """
        return self.record_event(
            event_type="shield_status",
            category="defensive",
            data=shield_data
        )

    def record_alert(self, alert_level: str, system: str, message: str,
                    triggered_by: str = None) -> Dict[str, Any]:
        """
        Record system alerts and warnings.

        Args:
            alert_level: Alert severity (e.g., 'yellow', 'red')
            system: System generating the alert
            message: Alert message
            triggered_by: What triggered the alert

        Returns:
            The recorded alert event
        """
        alert_data = {
            "alert_level": alert_level,
            "system": system,
            "message": message
        }

        if triggered_by:
            alert_data["triggered_by"] = triggered_by

        return self.record_event(
            event_type="alert",
            category="system_alert",
            data=alert_data
        )

    def get_events_after(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """Get all events after a specific timestamp."""
        with self._lock:
            return [e for e in self.events if e["timestamp"] > timestamp]

    def get_events_before(self, timestamp: datetime) -> List[Dict[str, Any]]:
        """Get all events before a specific timestamp."""
        with self._lock:
            return [e for e in self.events if e["timestamp"] < timestamp]

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all events of a specific type."""
        with self._lock:
            return [e for e in self.events if e["event_type"] == event_type]

    def get_events_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all events in a specific category."""
        with self._lock:
            return [e for e in self.events if e["category"] == category]

    def export_to_json(self, filepath: Path, ship_status: Dict[str, Any] = None) -> None:
        """
        Export all events to a JSON file with comprehensive ship status.

        Args:
            filepath: Path where to save the JSON file
            ship_status: Final ship status summary to include
        """
        with self._lock:
            self.end_time = datetime.now()

            # Get statistics
            stats = self.get_statistics()

            export_data = {
                "mission_id": self.mission_id,
                "mission_name": self.mission_name,
                "bridge_crew": self.bridge_crew,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration": str(self.end_time - self.start_time),
                "summary": stats,
                "final_ship_status": ship_status,
                "events": []
            }

            # Convert events with datetime to serializable format
            for event in self.events:
                event_copy = event.copy()
                event_copy["timestamp"] = event_copy["timestamp"].isoformat()
                export_data["events"].append(event_copy)

            # Write to file
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about recorded events.

        Returns:
            Dictionary containing event statistics
        """
        with self._lock:
            if not self.events:
                return {
                    "total_events": 0,
                    "event_types": {},
                    "categories": {},
                    "duration": "0:00:00",
                    "events_per_minute": 0
                }

            # Count by type
            type_counts = defaultdict(int)
            category_counts = defaultdict(int)

            for event in self.events:
                type_counts[event["event_type"]] += 1
                category_counts[event["category"]] += 1

            # Calculate duration
            current_time = self.end_time or datetime.now()
            duration = current_time - self.start_time
            duration_minutes = duration.total_seconds() / 60

            # Events per minute
            events_per_minute = len(self.events) / duration_minutes if duration_minutes > 0 else 0

            # Track damage and critical events
            damage_total = 0
            critical_events = 0
            system_changes = 0
            shield_events = 0

            for event in self.events:
                if event["event_type"] == "damage_taken":
                    damage_total += event["data"].get("amount", 0)
                if event["category"] == "critical":
                    critical_events += 1
                if event["event_type"] == "system_change":
                    system_changes += 1
                if event["event_type"] == "shield_status":
                    shield_events += 1

            return {
                "total_events": len(self.events),
                "event_types": dict(type_counts),
                "categories": dict(category_counts),
                "duration": str(duration),
                "events_per_minute": round(events_per_minute, 2),
                "total_damage": damage_total,
                "critical_events": critical_events,
                "system_changes": system_changes,
                "shield_events": shield_events
            }

    def clear_events(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self.events.clear()
            self._event_counter = 0

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self.events)

    def __repr__(self) -> str:
        """String representation of the recorder."""
        return f"EventRecorder(mission_id='{self.mission_id}', events={len(self.events)})"