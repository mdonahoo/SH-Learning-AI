#!/usr/bin/env python3
"""
Smart Filtering Components for Starship Horizons Telemetry
Reduces noise while preserving critical AI training data.
"""

import time
from typing import Dict, Any, Optional


class MissileTracker:
    """Smart filter for missile tracking events to reduce noise."""

    def __init__(self):
        self.active_missiles = {}  # missile_id -> {first_seen, last_position, launch_event_sent}
        self.impact_timeout = 5.0  # seconds before considering missile as missed/expired

    def process_missile_event(self, contact_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process missile contact and return significant events only.

        Returns:
            Event dict if significant (launch/impact), None if filtered noise
        """
        current_time = time.time()

        # Filter for missile-type contacts
        if not self._is_missile(contact_data):
            return None

        missile_id = contact_data.get("ID")
        if not missile_id:
            return None

        # Track missile state
        if missile_id not in self.active_missiles:
            # New missile detected - this is a launch event
            self.active_missiles[missile_id] = {
                "first_seen": current_time,
                "last_position": contact_data.get("Position", {}),
                "launch_event_sent": True,
                "target": contact_data.get("Target"),
                "faction": contact_data.get("Faction", "Unknown")
            }

            return {
                "type": "missile_launch",
                "category": "combat",
                "priority": "HIGH",
                "data": {
                    "missile_id": missile_id,
                    "target": contact_data.get("Target"),
                    "faction": contact_data.get("Faction"),
                    "position": contact_data.get("Position"),
                    "class": contact_data.get("Class", "Missile")
                }
            }
        else:
            # Update existing missile
            missile_info = self.active_missiles[missile_id]
            missile_info["last_position"] = contact_data.get("Position", {})
            missile_info["last_seen"] = current_time

            # Don't emit tracking noise - missiles move constantly
            return None

    def process_missile_removal(self, missile_id: int) -> Optional[Dict[str, Any]]:
        """Handle missile removal - could be impact or miss."""
        if missile_id in self.active_missiles:
            missile_info = self.active_missiles.pop(missile_id)

            return {
                "type": "missile_impact",
                "category": "combat",
                "priority": "HIGH",
                "data": {
                    "missile_id": missile_id,
                    "target": missile_info.get("target"),
                    "flight_time": time.time() - missile_info["first_seen"],
                    "final_position": missile_info["last_position"]
                }
            }
        return None

    def cleanup_expired_missiles(self):
        """Remove missiles that haven't been seen recently."""
        current_time = time.time()
        expired = []

        for missile_id, info in self.active_missiles.items():
            if current_time - info.get("last_seen", info["first_seen"]) > self.impact_timeout:
                expired.append(missile_id)

        for missile_id in expired:
            self.active_missiles.pop(missile_id, None)

    def _is_missile(self, contact_data: Dict[str, Any]) -> bool:
        """Identify if contact is a missile."""
        sub_type = contact_data.get("SubType", "").lower()
        base_type = contact_data.get("BaseType", "").lower()
        class_name = contact_data.get("Class", "").lower()

        return any([
            "missile" in sub_type,
            "torpedo" in sub_type,
            "projectile" in base_type,
            "missile" in class_name,
            "torpedo" in class_name
        ])


class ContactSignificanceFilter:
    """Filter contact events based on significance to training."""

    def __init__(self):
        self.known_contacts = {}  # contact_id -> last_significant_data
        self.significance_threshold = 0.1  # Change threshold for position/status

    def is_significant_contact(self, contact_data: Dict[str, Any]) -> bool:
        """Determine if contact update is significant enough to record."""
        contact_id = contact_data.get("ID")
        if not contact_id:
            return True  # New contacts are always significant

        # Always significant events
        significant_fields = ["IFF", "Identified", "Scanned", "Faction", "Class", "SubType"]
        for field in significant_fields:
            if field in contact_data:
                old_value = self.known_contacts.get(contact_id, {}).get(field)
                if old_value != contact_data[field]:
                    self._update_known_contact(contact_id, contact_data)
                    return True

        # Check for significant position changes (for ships, not debris)
        if self._is_active_vessel(contact_data):
            if self._has_significant_movement(contact_id, contact_data):
                self._update_known_contact(contact_id, contact_data)
                return True

        # Check for combat-related changes
        if self._has_combat_significance(contact_id, contact_data):
            self._update_known_contact(contact_id, contact_data)
            return True

        # Filter out noise (minor position updates of debris, etc.)
        return False

    def _update_known_contact(self, contact_id: int, contact_data: Dict[str, Any]):
        """Update our record of known contact data."""
        self.known_contacts[contact_id] = contact_data.copy()

    def _is_active_vessel(self, contact_data: Dict[str, Any]) -> bool:
        """Check if contact is an active vessel (not debris/asteroid)."""
        base_type = contact_data.get("BaseType", "").lower()
        sub_type = contact_data.get("SubType", "").lower()

        return base_type in ["vessel", "ship"] and "debris" not in sub_type

    def _has_significant_movement(self, contact_id: int, contact_data: Dict[str, Any]) -> bool:
        """Check if contact has moved significantly."""
        if contact_id not in self.known_contacts:
            return True

        old_pos = self.known_contacts[contact_id].get("Position", {})
        new_pos = contact_data.get("Position", {})

        if not old_pos or not new_pos:
            return True

        # Calculate distance moved
        dx = new_pos.get("X", 0) - old_pos.get("X", 0)
        dy = new_pos.get("Y", 0) - old_pos.get("Y", 0)
        dz = new_pos.get("Z", 0) - old_pos.get("Z", 0)

        distance = (dx*dx + dy*dy + dz*dz) ** 0.5

        # Threshold based on contact size/significance
        radius = contact_data.get("Radius", 100)
        threshold = max(radius * 0.5, 1000)  # Move at least half ship-length or 1km

        return distance > threshold

    def _has_combat_significance(self, contact_id: int, contact_data: Dict[str, Any]) -> bool:
        """Check for combat-related significance."""
        if contact_id not in self.known_contacts:
            return True

        old_data = self.known_contacts[contact_id]

        # Check shield/hull changes
        shield_change = abs(contact_data.get("Shields", 0) - old_data.get("Shields", 0))
        hull_change = abs(contact_data.get("Integrity", 100) - old_data.get("Integrity", 100))

        return shield_change > 0.05 or hull_change > 5.0


class EventPrioritizer:
    """Assign priority levels to events for AI training importance."""

    PRIORITY_CRITICAL = "CRITICAL"
    PRIORITY_HIGH = "HIGH"
    PRIORITY_MEDIUM = "MEDIUM"
    PRIORITY_LOW = "LOW"

    def get_event_priority(self, event: Dict[str, Any]) -> str:
        """Assign priority level to event."""
        event_type = event.get("type", "")
        category = event.get("category", "")
        data = event.get("data", {})

        # Critical events - immediate threat/mission critical
        if any([
            event_type in ["alert", "hull_breach", "reactor_failure", "missile_launch"],
            category == "critical",
            isinstance(data, dict) and data.get("alert_level") in [4, 5, "red"],
            "critical" in event_type.lower(),
            "emergency" in event_type.lower()
        ]):
            return self.PRIORITY_CRITICAL

        # High priority - combat, damage, major systems
        if any([
            event_type in ["damage", "weapon_fired", "missile_impact", "contact_hostile"],
            category in ["combat", "damage", "weapons"],
            "combat" in event_type.lower(),
            "weapon" in event_type.lower(),
            "damage" in event_type.lower()
        ]):
            return self.PRIORITY_HIGH

        # Medium priority - navigation, crew actions, systems
        if any([
            event_type in ["navigation", "crew_action", "system_status", "engineering"],
            category in ["navigation", "engineering", "crew"],
            "system" in event_type.lower(),
            "engineering" in event_type.lower()
        ]):
            return self.PRIORITY_MEDIUM

        # Low priority - routine updates, minor events
        return self.PRIORITY_LOW