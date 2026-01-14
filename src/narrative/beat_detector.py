"""
Dramatic beat detection from telemetry events.

Transforms raw telemetry data into story-significant dramatic beats
that can be used to construct a narrative arc.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BeatType(Enum):
    """Types of dramatic beats in a story."""

    # Act structure beats
    COLD_OPEN_HOOK = "cold_open"
    INCITING_INCIDENT = "inciting"
    COMPLICATION = "complication"
    ESCALATION = "escalation"
    CRISIS_POINT = "crisis"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    TRAGIC_RESOLUTION = "tragic_resolution"

    # Character beats
    CHARACTER_MOMENT = "character"
    CREW_TENSION = "crew_tension"
    HUMOR = "humor"
    SACRIFICE = "sacrifice"
    COMPETENCE = "competence"

    # Technical/tactical beats
    TECHNOBABBLE = "tech"
    DISCOVERY = "discovery"
    TACTICAL_DECISION = "tactical"
    DIPLOMATIC_BEAT = "diplomatic"

    # Transition beats
    QUIET_MOMENT = "quiet"
    TIME_PASSAGE = "time_passage"
    LOCATION_CHANGE = "location"


@dataclass
class DramaticBeat:
    """A single story-significant moment."""

    timestamp: datetime
    beat_type: BeatType
    tension_delta: float  # How much this changes tension (-1.0 to 1.0)
    description: str

    # Source data
    telemetry_events: List[Dict[str, Any]] = field(default_factory=list)
    transcript_lines: List[Dict[str, Any]] = field(default_factory=list)

    # Dramatic context
    stakes: str = ""
    emotion: str = ""
    pov_character: Optional[str] = None

    # Computed
    tension_level: float = 0.0  # Set by TensionAnalyzer

    def __post_init__(self) -> None:
        """Ensure timestamp is datetime."""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


# Event type to beat mapping with tension impacts
EVENT_BEAT_MAPPING: Dict[str, Dict[str, Any]] = {
    # Alert changes
    "alert_green_to_yellow": {
        "beat_type": BeatType.INCITING_INCIDENT,
        "tension_delta": 0.15,
        "emotion": "concern",
        "description": "Alert status elevated to yellow",
    },
    "alert_yellow_to_red": {
        "beat_type": BeatType.ESCALATION,
        "tension_delta": 0.25,
        "emotion": "urgency",
        "description": "Red alert declared",
    },
    "alert_to_green": {
        "beat_type": BeatType.RESOLUTION,
        "tension_delta": -0.20,
        "emotion": "relief",
        "description": "Alert status returned to normal",
    },
    # Combat events
    "first_weapon_fire": {
        "beat_type": BeatType.ESCALATION,
        "tension_delta": 0.20,
        "emotion": "determination",
        "description": "Weapons engaged",
    },
    "weapon_fire": {
        "beat_type": BeatType.TACTICAL_DECISION,
        "tension_delta": 0.05,
        "emotion": "focus",
        "description": "Weapons fire continues",
    },
    "shields_critical": {
        "beat_type": BeatType.CRISIS_POINT,
        "tension_delta": 0.30,
        "emotion": "fear",
        "description": "Shields failing",
    },
    "hull_breach": {
        "beat_type": BeatType.CRISIS_POINT,
        "tension_delta": 0.35,
        "emotion": "panic",
        "description": "Hull breach detected",
    },
    "hull_critical": {
        "beat_type": BeatType.CRISIS_POINT,
        "tension_delta": 0.40,
        "emotion": "desperation",
        "description": "Hull integrity critical",
    },
    # Mission events
    "mission_start": {
        "beat_type": BeatType.COLD_OPEN_HOOK,
        "tension_delta": 0.10,
        "emotion": "anticipation",
        "description": "Mission begins",
    },
    "objective_discovered": {
        "beat_type": BeatType.DISCOVERY,
        "tension_delta": 0.05,
        "emotion": "curiosity",
        "description": "New objective revealed",
    },
    "objective_complete": {
        "beat_type": BeatType.RESOLUTION,
        "tension_delta": -0.10,
        "emotion": "satisfaction",
        "description": "Objective completed",
    },
    "objective_failed": {
        "beat_type": BeatType.COMPLICATION,
        "tension_delta": 0.15,
        "emotion": "frustration",
        "description": "Objective failed",
    },
    "mission_complete": {
        "beat_type": BeatType.RESOLUTION,
        "tension_delta": -0.50,
        "emotion": "triumph",
        "description": "Mission successful",
    },
    "mission_failed": {
        "beat_type": BeatType.TRAGIC_RESOLUTION,
        "tension_delta": 0.0,
        "emotion": "defeat",
        "description": "Mission failed",
    },
    # Contact events
    "hostile_contact": {
        "beat_type": BeatType.INCITING_INCIDENT,
        "tension_delta": 0.20,
        "emotion": "alarm",
        "description": "Hostile vessel detected",
    },
    "multiple_hostiles": {
        "beat_type": BeatType.COMPLICATION,
        "tension_delta": 0.25,
        "emotion": "dread",
        "description": "Multiple hostile contacts",
    },
    "contact_identified": {
        "beat_type": BeatType.DISCOVERY,
        "tension_delta": 0.05,
        "emotion": "awareness",
        "description": "Contact identified",
    },
    # Crew events
    "station_manned": {
        "beat_type": BeatType.CHARACTER_MOMENT,
        "tension_delta": -0.05,
        "emotion": "readiness",
        "description": "Crew member takes station",
    },
    "station_abandoned": {
        "beat_type": BeatType.COMPLICATION,
        "tension_delta": 0.10,
        "emotion": "concern",
        "description": "Station unmanned",
    },
    # Communication events
    "hail_sent": {
        "beat_type": BeatType.DIPLOMATIC_BEAT,
        "tension_delta": 0.0,
        "emotion": "hope",
        "description": "Hailing frequencies open",
    },
    "hail_success": {
        "beat_type": BeatType.RESOLUTION,
        "tension_delta": -0.10,
        "emotion": "relief",
        "description": "Communication established",
    },
    "hail_failed": {
        "beat_type": BeatType.COMPLICATION,
        "tension_delta": 0.05,
        "emotion": "frustration",
        "description": "No response to hails",
    },
    # Navigation events
    "waypoint_arrival": {
        "beat_type": BeatType.LOCATION_CHANGE,
        "tension_delta": 0.0,
        "emotion": "anticipation",
        "description": "Arrived at waypoint",
    },
    "evasive_maneuver": {
        "beat_type": BeatType.TACTICAL_DECISION,
        "tension_delta": 0.10,
        "emotion": "urgency",
        "description": "Evasive action",
    },
    # Science events
    "scan_complete": {
        "beat_type": BeatType.DISCOVERY,
        "tension_delta": 0.0,
        "emotion": "curiosity",
        "description": "Scan complete",
    },
    "probe_launched": {
        "beat_type": BeatType.DISCOVERY,
        "tension_delta": 0.0,
        "emotion": "anticipation",
        "description": "Probe deployed",
    },
    # Engineering events
    "power_critical": {
        "beat_type": BeatType.CRISIS_POINT,
        "tension_delta": 0.25,
        "emotion": "alarm",
        "description": "Power systems critical",
    },
    "repair_complete": {
        "beat_type": BeatType.RESOLUTION,
        "tension_delta": -0.10,
        "emotion": "relief",
        "description": "Repairs complete",
    },
    "system_failure": {
        "beat_type": BeatType.COMPLICATION,
        "tension_delta": 0.15,
        "emotion": "concern",
        "description": "System failure",
    },
}


class BeatDetector:
    """
    Detects dramatic beats from telemetry events.

    Analyzes mission telemetry to identify story-significant moments
    and classify them into dramatic beat types.
    """

    def __init__(self) -> None:
        """Initialize the beat detector."""
        self.beats: List[DramaticBeat] = []
        self._alert_history: List[int] = []
        self._weapon_fired: bool = False
        self._shields_warned: bool = False
        self._hull_warned: bool = False

    def detect_beats(
        self,
        events: List[Dict[str, Any]],
        transcripts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DramaticBeat]:
        """
        Detect dramatic beats from mission events.

        Args:
            events: List of telemetry events with timestamps
            transcripts: Optional list of transcript entries

        Returns:
            List of detected dramatic beats in chronological order
        """
        self.beats = []
        self._reset_state()

        transcripts = transcripts or []

        for event in events:
            beat = self._process_event(event, transcripts)
            if beat:
                self.beats.append(beat)

        # Post-process to merge nearby beats and add context
        self._merge_nearby_beats()
        self._attach_transcripts(transcripts)

        logger.info(f"Detected {len(self.beats)} dramatic beats")
        return self.beats

    def _reset_state(self) -> None:
        """Reset internal state for new detection run."""
        self._alert_history = []
        self._weapon_fired = False
        self._shields_warned = False
        self._hull_warned = False
        self._mission_state = None
        self._objective_states = {}
        self._mission_started = False

    def _process_event(
        self,
        event: Dict[str, Any],
        transcripts: List[Dict[str, Any]],
    ) -> Optional[DramaticBeat]:
        """
        Process a single event and return a beat if significant.

        Args:
            event: Telemetry event
            transcripts: All transcripts for context

        Returns:
            DramaticBeat if event is significant, None otherwise
        """
        event_type = event.get("type", "").lower()
        category = event.get("category", "").lower()
        data = event.get("data", {})
        timestamp = event.get("timestamp", datetime.now().isoformat())

        # Parse timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()

        # Check for alert changes
        if event_type == "alert" or category == "alert":
            return self._process_alert_event(event, timestamp, data)

        # Check for combat events
        if event_type in ("weapon_fire", "weapons") or category == "tactical":
            return self._process_combat_event(event, timestamp, data)

        # Check for damage events
        if event_type == "damage" or "hull" in event_type or "shield" in event_type:
            return self._process_damage_event(event, timestamp, data)

        # Check for mission events
        if event_type == "mission" or category == "mission":
            return self._process_mission_event(event, timestamp, data)

        # Check for contact events
        if event_type == "contact" or category == "contacts":
            return self._process_contact_event(event, timestamp, data)

        # Check for crew events
        if event_type == "player" or event_type == "crew":
            return self._process_crew_event(event, timestamp, data)

        # Check for communication events
        if event_type == "hail" or category == "operations":
            return self._process_comm_event(event, timestamp, data)

        # Check for navigation events
        if event_type in ("waypoint", "course", "navigation") or category == "helm":
            return self._process_nav_event(event, timestamp, data)

        return None

    def _process_alert_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process alert level changes."""
        alert_level = data.get("level", data.get("alert", 0))

        if isinstance(alert_level, str):
            level_map = {"green": 2, "yellow": 3, "red": 4}
            alert_level = level_map.get(alert_level.lower(), 2)

        if not self._alert_history:
            self._alert_history.append(alert_level)
            # First alert - if it's already elevated, that's significant
            if alert_level == 3:
                mapping = EVENT_BEAT_MAPPING["alert_green_to_yellow"]
                return DramaticBeat(
                    timestamp=timestamp,
                    beat_type=mapping["beat_type"],
                    tension_delta=mapping["tension_delta"],
                    description=mapping["description"],
                    emotion=mapping["emotion"],
                    telemetry_events=[event],
                    stakes=self._infer_stakes(mapping["beat_type"]),
                )
            elif alert_level >= 4:
                mapping = EVENT_BEAT_MAPPING["alert_yellow_to_red"]
                return DramaticBeat(
                    timestamp=timestamp,
                    beat_type=mapping["beat_type"],
                    tension_delta=mapping["tension_delta"],
                    description=mapping["description"],
                    emotion=mapping["emotion"],
                    telemetry_events=[event],
                    stakes=self._infer_stakes(mapping["beat_type"]),
                )
            return None

        prev_level = self._alert_history[-1]
        self._alert_history.append(alert_level)

        # Detect transitions
        if prev_level == 2 and alert_level == 3:
            mapping = EVENT_BEAT_MAPPING["alert_green_to_yellow"]
        elif prev_level == 3 and alert_level == 4:
            mapping = EVENT_BEAT_MAPPING["alert_yellow_to_red"]
        elif prev_level < 4 and alert_level >= 4:
            # Jump to red from any lower level
            mapping = EVENT_BEAT_MAPPING["alert_yellow_to_red"]
        elif alert_level == 2 and prev_level > 2:
            mapping = EVENT_BEAT_MAPPING["alert_to_green"]
        else:
            return None

        return DramaticBeat(
            timestamp=timestamp,
            beat_type=mapping["beat_type"],
            tension_delta=mapping["tension_delta"],
            description=mapping["description"],
            emotion=mapping["emotion"],
            telemetry_events=[event],
            stakes=self._infer_stakes(mapping["beat_type"]),
        )

    def _process_combat_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process combat-related events."""
        if not self._weapon_fired:
            self._weapon_fired = True
            mapping = EVENT_BEAT_MAPPING["first_weapon_fire"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=mapping["description"],
                emotion=mapping["emotion"],
                telemetry_events=[event],
                stakes="Ship and crew survival",
            )

        # Subsequent weapon fire - less dramatic, batch them
        return None

    def _process_damage_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process damage-related events."""
        shields = data.get("shields", data.get("shield_percent", 100))
        hull = data.get("hull", data.get("hull_percent", 100))

        # Shield critical (below 25%)
        if shields < 25 and not self._shields_warned:
            self._shields_warned = True
            mapping = EVENT_BEAT_MAPPING["shields_critical"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=f"Shields at {shields}%",
                emotion=mapping["emotion"],
                telemetry_events=[event],
                stakes="Imminent hull damage",
            )

        # Hull critical (below 25%)
        if hull < 25 and not self._hull_warned:
            self._hull_warned = True
            mapping = EVENT_BEAT_MAPPING["hull_critical"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=f"Hull integrity at {hull}%",
                emotion=mapping["emotion"],
                telemetry_events=[event],
                stakes="Ship destruction imminent",
            )

        # Hull breach
        if data.get("breach", False) or data.get("breaches", 0) > 0:
            mapping = EVENT_BEAT_MAPPING["hull_breach"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=mapping["description"],
                emotion=mapping["emotion"],
                telemetry_events=[event],
                stakes="Crew lives at risk",
            )

        return None

    def _process_mission_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process mission-related events."""
        # Handle both lowercase and capitalized keys
        state = data.get("State", data.get("state", "")).lower()
        event_subtype = data.get("event", "").lower()
        event_type = event.get("event_type", "").lower()

        # Check for mission_update events with state tracking
        if event_type == "mission_update":
            return self._process_mission_update(event, timestamp, data)

        if state == "running" or event_subtype == "start":
            mapping = EVENT_BEAT_MAPPING["mission_start"]
        elif state == "complete" or event_subtype == "complete":
            mapping = EVENT_BEAT_MAPPING["mission_complete"]
        elif state == "failed" or event_subtype == "failed":
            mapping = EVENT_BEAT_MAPPING["mission_failed"]
        elif "objective" in event_subtype:
            if "complete" in event_subtype:
                mapping = EVENT_BEAT_MAPPING["objective_complete"]
            elif "failed" in event_subtype:
                mapping = EVENT_BEAT_MAPPING["objective_failed"]
            else:
                mapping = EVENT_BEAT_MAPPING["objective_discovered"]
        else:
            return None

        return DramaticBeat(
            timestamp=timestamp,
            beat_type=mapping["beat_type"],
            tension_delta=mapping["tension_delta"],
            description=data.get("description", mapping["description"]),
            emotion=mapping["emotion"],
            telemetry_events=[event],
            stakes=self._infer_stakes(mapping["beat_type"]),
        )

    def _process_mission_update(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process mission_update events to detect state changes."""
        # Initialize tracking on first call
        if not hasattr(self, "_mission_state"):
            self._mission_state = None
            self._objective_states = {}
            self._mission_started = False

        current_state = data.get("State", "").lower()
        mission_name = data.get("Name", "Unknown")
        objectives = data.get("Objectives", {})

        # Detect mission start (first Running state)
        if current_state == "running" and not self._mission_started:
            self._mission_started = True
            self._mission_state = current_state
            mapping = EVENT_BEAT_MAPPING["mission_start"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=f"Mission '{mission_name}' begins",
                emotion=mapping["emotion"],
                telemetry_events=[event],
                stakes="Mission success",
            )

        # Detect mission complete
        if data.get("Complete", False) and self._mission_state != "complete":
            self._mission_state = "complete"
            if data.get("Success", False):
                mapping = EVENT_BEAT_MAPPING["mission_complete"]
                return DramaticBeat(
                    timestamp=timestamp,
                    beat_type=mapping["beat_type"],
                    tension_delta=mapping["tension_delta"],
                    description=f"Mission '{mission_name}' successful",
                    emotion=mapping["emotion"],
                    telemetry_events=[event],
                    stakes="Victory achieved",
                )
            else:
                mapping = EVENT_BEAT_MAPPING["mission_failed"]
                return DramaticBeat(
                    timestamp=timestamp,
                    beat_type=mapping["beat_type"],
                    tension_delta=mapping["tension_delta"],
                    description=f"Mission '{mission_name}' failed",
                    emotion=mapping["emotion"],
                    telemetry_events=[event],
                    stakes="Mission lost",
                )

        # Detect objective changes
        for obj_name, obj_data in objectives.items():
            prev_state = self._objective_states.get(obj_name, {})
            prev_complete = prev_state.get("Complete", False)
            prev_count = prev_state.get("CurrentCount", 0)

            curr_complete = obj_data.get("Complete", False)
            curr_count = obj_data.get("CurrentCount", 0)
            total_count = obj_data.get("Count", 1)

            # Store current state
            self._objective_states[obj_name] = {
                "Complete": curr_complete,
                "CurrentCount": curr_count,
            }

            # Objective just completed
            if curr_complete and not prev_complete:
                mapping = EVENT_BEAT_MAPPING["objective_complete"]
                return DramaticBeat(
                    timestamp=timestamp,
                    beat_type=mapping["beat_type"],
                    tension_delta=mapping["tension_delta"],
                    description=f"Objective complete: {obj_name}",
                    emotion=mapping["emotion"],
                    telemetry_events=[event],
                    stakes="Progress made",
                )

            # Significant progress on objective (every 25%)
            if total_count > 0 and prev_count != curr_count:
                prev_pct = (prev_count / total_count) * 100
                curr_pct = (curr_count / total_count) * 100

                # Check if crossed a 25% threshold
                for threshold in [25, 50, 75]:
                    if prev_pct < threshold <= curr_pct:
                        return DramaticBeat(
                            timestamp=timestamp,
                            beat_type=BeatType.DISCOVERY,
                            tension_delta=0.05,
                            description=f"{obj_name}: {curr_count}/{total_count} ({curr_pct:.0f}%)",
                            emotion="progress",
                            telemetry_events=[event],
                            stakes="Making progress",
                        )

        return None

    def _process_contact_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process contact detection events."""
        faction = data.get("faction", "").lower()
        is_hostile = faction in ("hostile", "enemy", "daichi", "pirate")

        if is_hostile:
            # Check if multiple hostiles
            contact_count = data.get("count", 1)
            if contact_count > 1:
                mapping = EVENT_BEAT_MAPPING["multiple_hostiles"]
                description = f"{contact_count} hostile vessels detected"
            else:
                mapping = EVENT_BEAT_MAPPING["hostile_contact"]
                name = data.get("name", "Unknown vessel")
                description = f"Hostile contact: {name}"

            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=description,
                emotion=mapping["emotion"],
                telemetry_events=[event],
                stakes="Potential combat engagement",
            )

        return None

    def _process_crew_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process crew/player events."""
        action = data.get("action", "").lower()
        station = data.get("station", "unknown")

        if action == "join" or action == "manned":
            mapping = EVENT_BEAT_MAPPING["station_manned"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=f"{station} station manned",
                emotion=mapping["emotion"],
                telemetry_events=[event],
            )

        return None

    def _process_comm_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process communication events."""
        action = data.get("action", "").lower()
        success = data.get("success", False)

        if action == "hail":
            if success:
                mapping = EVENT_BEAT_MAPPING["hail_success"]
            else:
                mapping = EVENT_BEAT_MAPPING["hail_failed"]

            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=mapping["description"],
                emotion=mapping["emotion"],
                telemetry_events=[event],
            )

        return None

    def _process_nav_event(
        self,
        event: Dict[str, Any],
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> Optional[DramaticBeat]:
        """Process navigation events."""
        event_subtype = data.get("event", "").lower()

        if "waypoint" in event_subtype or "arrival" in event_subtype:
            location = data.get("location", data.get("name", "destination"))
            mapping = EVENT_BEAT_MAPPING["waypoint_arrival"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=f"Arrived at {location}",
                emotion=mapping["emotion"],
                telemetry_events=[event],
            )

        if "evasive" in event_subtype:
            mapping = EVENT_BEAT_MAPPING["evasive_maneuver"]
            return DramaticBeat(
                timestamp=timestamp,
                beat_type=mapping["beat_type"],
                tension_delta=mapping["tension_delta"],
                description=mapping["description"],
                emotion=mapping["emotion"],
                telemetry_events=[event],
            )

        return None

    def _merge_nearby_beats(self, window_seconds: float = 5.0) -> None:
        """
        Merge beats that occur very close together.

        Args:
            window_seconds: Time window for merging
        """
        if len(self.beats) < 2:
            return

        merged: List[DramaticBeat] = []
        current_group: List[DramaticBeat] = [self.beats[0]]

        for beat in self.beats[1:]:
            time_diff = (beat.timestamp - current_group[-1].timestamp).total_seconds()

            if time_diff <= window_seconds:
                current_group.append(beat)
            else:
                # Merge group and start new one
                merged.append(self._merge_beat_group(current_group))
                current_group = [beat]

        # Don't forget the last group
        merged.append(self._merge_beat_group(current_group))

        self.beats = merged

    def _merge_beat_group(self, group: List[DramaticBeat]) -> DramaticBeat:
        """Merge a group of beats into a single beat."""
        if len(group) == 1:
            return group[0]

        # Take the most dramatic beat type
        priority = [
            BeatType.CRISIS_POINT,
            BeatType.CLIMAX,
            BeatType.TRAGIC_RESOLUTION,
            BeatType.ESCALATION,
            BeatType.INCITING_INCIDENT,
            BeatType.COMPLICATION,
        ]

        best_beat = group[0]
        for beat in group[1:]:
            if beat.beat_type in priority:
                if best_beat.beat_type not in priority:
                    best_beat = beat
                elif priority.index(beat.beat_type) < priority.index(best_beat.beat_type):
                    best_beat = beat

        # Combine all telemetry and transcripts
        all_events = []
        all_transcripts = []
        total_tension = 0.0

        for beat in group:
            all_events.extend(beat.telemetry_events)
            all_transcripts.extend(beat.transcript_lines)
            total_tension += beat.tension_delta

        return DramaticBeat(
            timestamp=group[0].timestamp,
            beat_type=best_beat.beat_type,
            tension_delta=total_tension,
            description=best_beat.description,
            emotion=best_beat.emotion,
            stakes=best_beat.stakes,
            telemetry_events=all_events,
            transcript_lines=all_transcripts,
        )

    def _attach_transcripts(
        self,
        transcripts: List[Dict[str, Any]],
        window_seconds: float = 30.0,
    ) -> None:
        """
        Attach relevant transcripts to each beat.

        Args:
            transcripts: All transcript entries
            window_seconds: Time window for associating transcripts
        """
        for beat in self.beats:
            beat_time = beat.timestamp
            relevant = []

            for t in transcripts:
                try:
                    t_time = datetime.fromisoformat(
                        t.get("timestamp", "").replace("Z", "+00:00")
                    )
                    diff = abs((t_time - beat_time).total_seconds())

                    if diff <= window_seconds:
                        relevant.append(t)
                except (ValueError, TypeError):
                    continue

            beat.transcript_lines = relevant

    def _infer_stakes(self, beat_type: BeatType) -> str:
        """Infer narrative stakes from beat type."""
        stakes_map = {
            BeatType.CRISIS_POINT: "Survival of ship and crew",
            BeatType.CLIMAX: "Mission success or failure",
            BeatType.ESCALATION: "Situation deteriorating",
            BeatType.INCITING_INCIDENT: "New threat emerging",
            BeatType.COMPLICATION: "Plans disrupted",
            BeatType.RESOLUTION: "Crisis averted",
            BeatType.TRAGIC_RESOLUTION: "Loss and failure",
        }
        return stakes_map.get(beat_type, "")
