#!/usr/bin/env python3
"""
Station-specific packet handlers for comprehensive telemetry capture.
Handles Operations, Game Master, and other bridge station data.
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)


class StationHandlers:
    """Collection of handlers for all bridge station packets."""

    # Operations Station Handlers
    @staticmethod
    def handle_hail(value) -> Dict[str, Any]:
        """Handle HAIL packet - outgoing hail attempts."""
        if value:
            logger.info(f"📡 Hail attempt: {value}")
            return {
                "type": "hail_attempt",
                "category": "operations",
                "priority": "MEDIUM",
                "data": value
            }
        return None

    @staticmethod
    def handle_hail_response(value) -> Dict[str, Any]:
        """Handle HAIL-RESPONSE packet - responses to hails."""
        if value:
            logger.info(f"📞 Hail response received: {value}")
            return {
                "type": "hail_response",
                "category": "operations",
                "priority": "MEDIUM",
                "data": value
            }
        return None

    @staticmethod
    def handle_communications(cmd: str, value) -> Dict[str, Any]:
        """Handle COMM packets - general communications."""
        if value:
            logger.info(f"💬 Communications ({cmd}): {value}")
            return {
                "type": "communication",
                "category": "operations",
                "priority": "MEDIUM",
                "data": {"message_type": cmd, "content": value}
            }
        return None

    @staticmethod
    def handle_transporter(value) -> Dict[str, Any]:
        """Handle TRANSPORTER packet - transporter operations."""
        if value:
            logger.info(f"🔄 Transporter operation: {value}")
            return {
                "type": "transporter_operation",
                "category": "operations",
                "priority": "HIGH",
                "data": value
            }
        return None

    @staticmethod
    def handle_docking(value) -> Dict[str, Any]:
        """Handle DOCKING packet - docking operations."""
        if value:
            logger.info(f"🚪 Docking operation: {value}")
            return {
                "type": "docking_operation",
                "category": "operations",
                "priority": "HIGH",
                "data": value
            }
        return None

    @staticmethod
    def handle_cargo(cmd: str, value) -> Dict[str, Any]:
        """Handle CARGO packets - cargo bay and transfers."""
        if value:
            logger.info(f"📦 Cargo ({cmd}): {value}")

            # Determine specific cargo event type
            if cmd == "CARGO-TRANSFER":
                event_type = "cargo_transfer"
                priority = "HIGH"
            elif cmd == "CARGO-BAY":
                event_type = "cargo_bay_status"
                priority = "LOW"
            else:
                event_type = "cargo_update"
                priority = "MEDIUM"

            return {
                "type": event_type,
                "category": "operations",
                "priority": priority,
                "data": value
            }
        return None

    @staticmethod
    def handle_shuttles(cmd: str, value) -> Dict[str, Any]:
        """Handle SHUTTLE packets - shuttle operations."""
        if value:
            logger.info(f"🚀 Shuttle ({cmd}): {value}")

            # Determine specific shuttle event
            if cmd == "SHUTTLE-LAUNCH":
                event_type = "shuttle_launch"
                priority = "HIGH"
            elif cmd == "SHUTTLE-DOCK":
                event_type = "shuttle_docking"
                priority = "HIGH"
            else:
                event_type = "shuttle_status"
                priority = "MEDIUM"

            return {
                "type": event_type,
                "category": "operations",
                "priority": priority,
                "data": value
            }
        return None

    @staticmethod
    def handle_fighters(cmd: str, value) -> Dict[str, Any]:
        """Handle FIGHTER packets - fighter squadron operations."""
        if value:
            logger.info(f"✈️ Fighter ops ({cmd}): {value}")

            # Determine specific fighter event
            if cmd == "FIGHTER-LAUNCH":
                event_type = "fighter_launch"
                priority = "HIGH"
            elif cmd == "FIGHTER-BAY":
                event_type = "fighter_bay_status"
                priority = "MEDIUM"
            else:
                event_type = "hangar_operation"
                priority = "MEDIUM"

            return {
                "type": event_type,
                "category": "tactical",
                "priority": priority,
                "data": value
            }
        return None

    # Helm Station Handlers
    @staticmethod
    def handle_helm_data(cmd: str, value) -> Dict[str, Any]:
        """Handle HELM packets - navigation and control."""
        if value is not None:
            logger.info(f"🎮 Helm ({cmd}): {value}")

            # Determine specific helm event
            if cmd == "THROTTLE":
                event_type = "throttle_change"
                priority = "MEDIUM"
            elif cmd == "COURSE":
                event_type = "course_change"
                priority = "HIGH"
            elif cmd == "AUTOPILOT":
                event_type = "autopilot_status"
                priority = "MEDIUM"
            else:
                event_type = "helm_control"
                priority = "MEDIUM"

            return {
                "type": event_type,
                "category": "helm",
                "priority": priority,
                "data": {"control": cmd, "value": value}
            }
        return None

    # Science Station Handlers
    @staticmethod
    def handle_science_operations(cmd: str, value) -> Dict[str, Any]:
        """Handle SCAN/PROBE packets - science operations."""
        if value:
            logger.info(f"🔬 Science ({cmd}): {value}")

            # Determine specific science event
            if cmd == "SCAN":
                event_type = "scan_initiated"
                priority = "MEDIUM"
            elif cmd == "SCAN-RESULT":
                event_type = "scan_result"
                priority = "HIGH"
            elif cmd == "PROBE":
                event_type = "probe_launch"
                priority = "HIGH"
            else:
                event_type = "science_operation"
                priority = "MEDIUM"

            return {
                "type": event_type,
                "category": "science",
                "priority": priority,
                "data": value
            }
        return None

    # Game Master Handlers
    @staticmethod
    def handle_gm_events(cmd: str, value) -> Dict[str, Any]:
        """Handle GM packets - Game Master events and controls."""
        if value:
            logger.info(f"🎯 GM Event ({cmd}): {value}")

            # Determine specific GM event
            if cmd == "GM-EVENT":
                event_type = "gm_event"
                priority = "CRITICAL"
            elif cmd == "SCENARIO":
                event_type = "scenario_change"
                priority = "HIGH"
            elif cmd == "NPC":
                event_type = "npc_action"
                priority = "MEDIUM"
            elif cmd == "SPAWN":
                event_type = "entity_spawn"
                priority = "HIGH"
            elif cmd == "OBJECTIVES":
                event_type = "objective_update"
                priority = "HIGH"
            else:
                event_type = "gm_control"
                priority = "MEDIUM"

            return {
                "type": event_type,
                "category": "game_master",
                "priority": priority,
                "data": value
            }
        return None

    # Crew Coordination Handlers
    @staticmethod
    def handle_crew_coordination(cmd: str, value) -> Dict[str, Any]:
        """Handle CREW packets - crew actions and coordination."""
        if value:
            logger.info(f"👥 Crew ({cmd}): {value}")

            # Determine specific crew event
            if cmd == "CREW-ACTION":
                event_type = "crew_action"
                priority = "HIGH"
            elif cmd == "STATION-REPORT":
                event_type = "station_report"
                priority = "MEDIUM"
            elif cmd == "CAPTAIN-ORDER":
                event_type = "captain_order"
                priority = "CRITICAL"
            else:
                event_type = "crew_update"
                priority = "LOW"

            return {
                "type": event_type,
                "category": "crew",
                "priority": priority,
                "data": value
            }
        return None


class CrewPerformanceTracker:
    """Track crew performance metrics across stations."""

    def __init__(self):
        self.station_metrics = {
            "operations": {
                "actions": 0,
                "response_times": [],
                "hails_attempted": 0,
                "hails_successful": 0,
                "cargo_transfers": 0,
                "shuttle_operations": 0
            },
            "helm": {
                "actions": 0,
                "course_changes": 0,
                "speed_changes": 0,
                "evasive_maneuvers": 0,
                "autopilot_usage": 0
            },
            "tactical": {
                "actions": 0,
                "weapons_fired": 0,
                "hits": 0,
                "misses": 0,
                "shields_adjusted": 0,
                "fighters_launched": 0
            },
            "science": {
                "actions": 0,
                "scans_initiated": 0,
                "scans_completed": 0,
                "probes_launched": 0,
                "contacts_identified": 0
            },
            "engineering": {
                "actions": 0,
                "power_adjustments": 0,
                "repairs_initiated": 0,
                "systems_optimized": 0,
                "emergencies_handled": 0
            }
        }

        self.coordination_events = []
        self.response_latencies = {}
        self.last_alert_time = None
        self.last_order_time = None

    def track_action(self, station: str, action_type: str, timestamp: float):
        """Track a crew action."""
        if station in self.station_metrics:
            self.station_metrics[station]["actions"] += 1

            # Calculate response time if this follows an alert/order
            if self.last_alert_time:
                response_time = timestamp - self.last_alert_time
                self.station_metrics[station]["response_times"].append(response_time)

    def track_coordination(self, stations: list, event_type: str, success: bool):
        """Track coordination between multiple stations."""
        self.coordination_events.append({
            "stations": stations,
            "event_type": event_type,
            "success": success,
            "timestamp": time.time()
        })

    def get_efficiency_score(self, station: str) -> float:
        """Calculate efficiency score for a station."""
        if station not in self.station_metrics:
            return 0.0

        metrics = self.station_metrics[station]

        # Calculate based on actions and response times
        if metrics["actions"] == 0:
            return 0.0

        # Average response time (lower is better)
        avg_response = sum(metrics["response_times"]) / len(metrics["response_times"]) if metrics["response_times"] else 10.0

        # Score calculation (simplified)
        efficiency = min(1.0, 10.0 / avg_response)  # Normalize to 0-1

        return efficiency

    def get_crew_coordination_score(self) -> float:
        """Calculate overall crew coordination score."""
        successful = sum(1 for e in self.coordination_events if e["success"])
        total = len(self.coordination_events)

        if total == 0:
            return 0.5  # Neutral score if no coordination events

        return successful / total