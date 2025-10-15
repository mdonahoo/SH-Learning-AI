#!/usr/bin/env python3
"""
Enhanced Starship Horizons Client with Meaningful Event Detection
Tracks actual gameplay changes rather than repetitive updates.
"""

import json
import logging
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Set
import requests
from copy import deepcopy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGameClient:
    """Enhanced client that detects meaningful game changes."""

    def __init__(self, host: str = None):
        """Initialize enhanced game client."""
        # Get host from environment or use provided host
        if host is None:
            game_host = os.getenv("GAME_HOST", "localhost")
            game_port = os.getenv("GAME_PORT_API", "1864")
            host = f"http://{game_host}:{game_port}"

        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/api"
        self.session = requests.Session()

        self._polling = False
        self._poll_thread = None
        self._event_callbacks = []

        # State tracking for change detection
        self._last_status = {}
        self._last_mission = {}
        self._last_objectives = {}
        self._mission_start_time = None
        self._objectives_completed = set()
        self._last_elapsed_time = 0

        # Track significant values
        self._last_grade = 0
        self._events_detected = []

    def test_connection(self) -> bool:
        """Test connection to game."""
        try:
            response = self.session.get(f"{self.api_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Connected - State: {data.get('State')}, Mission: {data.get('Mission', 'None')}")
                return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
        return False

    def add_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for game events."""
        self._event_callbacks.append(callback)

    def start_monitoring(self, interval: float = 0.5):
        """Start monitoring for meaningful changes."""
        if self._polling:
            return

        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._poll_thread.start()
        logger.info(f"Started monitoring (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop monitoring."""
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        logger.info("Stopped monitoring")

    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._polling:
            try:
                # Check status
                status = self._get_status()
                if status:
                    self._check_status_changes(status)

                # Check mission
                mission = self._get_mission()
                if mission:
                    self._check_mission_changes(mission)

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(interval)

    def _get_status(self) -> Optional[Dict]:
        """Get game status."""
        try:
            resp = self.session.get(f"{self.api_url}/status", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def _get_mission(self) -> Optional[Dict]:
        """Get mission data."""
        try:
            resp = self.session.get(f"{self.api_url}/mission", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def _check_status_changes(self, status: Dict):
        """Check for meaningful status changes."""
        if not self._last_status:
            # First status
            self._emit_event({
                "type": "session_start",
                "category": "session",
                "data": {
                    "state": status.get("State"),
                    "mode": status.get("Mode"),
                    "mission": status.get("Mission")
                }
            })
        else:
            # State changes
            if status.get("State") != self._last_status.get("State"):
                old_state = self._last_status.get("State")
                new_state = status.get("State")

                if new_state == "Running" and old_state != "Running":
                    self._emit_event({
                        "type": "mission_started",
                        "category": "mission",
                        "data": {
                            "mission": status.get("Mission"),
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    self._mission_start_time = datetime.now()

                elif old_state == "Running" and new_state != "Running":
                    self._emit_event({
                        "type": "mission_ended",
                        "category": "mission",
                        "data": {
                            "mission": self._last_status.get("Mission"),
                            "final_state": new_state,
                            "duration": str(datetime.now() - self._mission_start_time) if self._mission_start_time else None
                        }
                    })

            # Mission changes
            if status.get("Mission") != self._last_status.get("Mission"):
                self._emit_event({
                    "type": "mission_changed",
                    "category": "mission",
                    "data": {
                        "from": self._last_status.get("Mission"),
                        "to": status.get("Mission")
                    }
                })

        self._last_status = status.copy()

    def _check_mission_changes(self, mission: Dict):
        """Check for meaningful mission changes."""
        if not self._last_mission:
            # First mission data
            if mission.get("State") == "Running":
                self._emit_event({
                    "type": "mission_info",
                    "category": "mission",
                    "data": {
                        "name": mission.get("Name"),
                        "date": mission.get("Date"),
                        "total_objectives": mission.get("TotalObjectives"),
                        "total_waypoints": mission.get("TotalWaypoints")
                    }
                })
        else:
            # Check objective changes
            self._check_objective_changes(mission)

            # Check grade changes
            if mission.get("Grade") != self._last_mission.get("Grade"):
                self._emit_event({
                    "type": "grade_changed",
                    "category": "performance",
                    "data": {
                        "old_grade": self._last_mission.get("Grade", 0),
                        "new_grade": mission.get("Grade", 0),
                        "percentage": f"{mission.get('Grade', 0) * 100:.1f}%"
                    }
                })

            # Check completion
            if mission.get("Complete") and not self._last_mission.get("Complete"):
                self._emit_event({
                    "type": "mission_completed",
                    "category": "mission",
                    "data": {
                        "success": mission.get("Success"),
                        "final_grade": mission.get("Grade"),
                        "elapsed_time": mission.get("ElapsedTime")
                    }
                })

            # Check timer milestones (every 60 seconds)
            try:
                elapsed = float(mission.get("ElapsedTime", 0))
                last_elapsed = float(self._last_mission.get("ElapsedTime", 0))

                if int(elapsed / 60) > int(last_elapsed / 60):
                    minutes = int(elapsed / 60)
                    self._emit_event({
                        "type": "time_milestone",
                        "category": "progress",
                        "data": {
                            "elapsed_minutes": minutes,
                            "elapsed_time": f"{minutes} minutes"
                        }
                    })
            except:
                pass

        self._last_mission = mission.copy()

    def _check_objective_changes(self, mission: Dict):
        """Check for objective completion or progress."""
        objectives = mission.get("Objectives", {})

        for obj_name, obj_data in objectives.items():
            last_obj = self._last_objectives.get(obj_name, {})

            # New objective discovered
            if not last_obj and obj_data.get("Visible"):
                self._emit_event({
                    "type": "objective_discovered",
                    "category": "objectives",
                    "data": {
                        "name": obj_name,
                        "description": obj_data.get("Description"),
                        "type": obj_data.get("Type"),
                        "rank": obj_data.get("Rank"),
                        "required": obj_data.get("Required", False)
                    }
                })

            # Objective progress
            if obj_data.get("CurrentCount") != last_obj.get("CurrentCount"):
                self._emit_event({
                    "type": "objective_progress",
                    "category": "objectives",
                    "data": {
                        "name": obj_name,
                        "previous": last_obj.get("CurrentCount", 0),
                        "current": obj_data.get("CurrentCount", 0),
                        "target": obj_data.get("Count", 0),
                        "percentage": f"{(obj_data.get('CurrentCount', 0) / obj_data.get('Count', 1)) * 100:.0f}%" if obj_data.get('Count') else "0%"
                    }
                })

            # Objective completed
            if obj_data.get("Complete") and not last_obj.get("Complete"):
                self._emit_event({
                    "type": "objective_completed",
                    "category": "objectives",
                    "data": {
                        "name": obj_name,
                        "rank": obj_data.get("Rank"),
                        "required": obj_data.get("Required", False)
                    }
                })
                self._objectives_completed.add(obj_name)

            # Objective failed (was visible but now not complete at mission end)
            if mission.get("Complete") and obj_data.get("Visible") and not obj_data.get("Complete") and obj_data.get("Required"):
                if obj_name not in self._objectives_completed:
                    self._emit_event({
                        "type": "objective_failed",
                        "category": "objectives",
                        "data": {
                            "name": obj_name,
                            "rank": obj_data.get("Rank"),
                            "progress": f"{obj_data.get('CurrentCount', 0)}/{obj_data.get('Count', 0)}"
                        }
                    })

        self._last_objectives = deepcopy(objectives)

    def _emit_event(self, event: Dict[str, Any]):
        """Emit event to callbacks."""
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()

        # Track event
        self._events_detected.append(event["type"])

        # Log meaningful events
        logger.info(f"🎮 {event['type']}: {event.get('data', {})}")

        # Call callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_summary(self) -> Dict:
        """Get summary of detected events."""
        from collections import Counter
        event_counts = Counter(self._events_detected)

        return {
            "total_events": len(self._events_detected),
            "event_types": dict(event_counts),
            "objectives_completed": len(self._objectives_completed),
            "current_grade": self._last_mission.get("Grade", 0) * 100 if self._last_mission else 0
        }