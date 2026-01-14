"""
Tension curve analysis and act structure detection.

Maps dramatic beats to a tension curve and identifies natural
act breaks for Star Trek-style episodic structure.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

from src.narrative.beat_detector import BeatType, DramaticBeat

logger = logging.getLogger(__name__)


class ActType(Enum):
    """Episode act types."""

    COLD_OPEN = "cold_open"
    ACT_ONE = "act_one"
    ACT_TWO = "act_two"
    ACT_THREE = "act_three"
    ACT_FOUR = "act_four"
    EPILOGUE = "epilogue"


@dataclass
class TensionPoint:
    """A point on the tension curve."""

    timestamp: datetime
    tension: float  # 0.0 to 1.0
    beat: Optional[DramaticBeat] = None


@dataclass
class ActBreak:
    """Marks a transition between acts."""

    timestamp: datetime
    from_act: ActType
    to_act: ActType
    trigger_beat: Optional[DramaticBeat] = None
    description: str = ""


@dataclass
class TensionCurve:
    """Complete tension analysis for a mission."""

    points: List[TensionPoint] = field(default_factory=list)
    act_breaks: List[ActBreak] = field(default_factory=list)

    # Key moments
    inciting_incident: Optional[DramaticBeat] = None
    crisis_point: Optional[DramaticBeat] = None
    climax: Optional[DramaticBeat] = None
    resolution: Optional[DramaticBeat] = None

    # Statistics
    peak_tension: float = 0.0
    peak_timestamp: Optional[datetime] = None
    average_tension: float = 0.0
    tension_volatility: float = 0.0

    def get_act_beats(self, act: ActType) -> List[DramaticBeat]:
        """Get all beats belonging to a specific act."""
        act_range = self._get_act_time_range(act)
        if not act_range:
            return []

        start, end = act_range
        return [
            p.beat for p in self.points
            if p.beat and start <= p.timestamp <= end
        ]

    def _get_act_time_range(
        self,
        act: ActType,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Get the time range for an act."""
        if not self.points:
            return None

        act_starts = {ActType.COLD_OPEN: self.points[0].timestamp}

        for ab in self.act_breaks:
            act_starts[ab.to_act] = ab.timestamp

        if act not in act_starts:
            return None

        start = act_starts[act]

        # Find end (start of next act or end of mission)
        acts_order = list(ActType)
        act_idx = acts_order.index(act)

        end = self.points[-1].timestamp
        for next_act in acts_order[act_idx + 1:]:
            if next_act in act_starts:
                end = act_starts[next_act]
                break

        return (start, end)


class TensionAnalyzer:
    """
    Analyzes dramatic beats to construct tension curves and act structure.

    Uses classic dramatic structure principles adapted for Star Trek
    episodic format.
    """

    # Tension thresholds for act transitions
    COLD_OPEN_MAX_TENSION = 0.3
    ACT_ONE_TRIGGER_TENSION = 0.25
    ACT_TWO_TRIGGER_TENSION = 0.45
    ACT_THREE_TRIGGER_TENSION = 0.65
    ACT_FOUR_TRIGGER_TENSION = 0.80
    RESOLUTION_TRIGGER_TENSION = 0.40

    # Minimum duration percentages for each act
    MIN_COLD_OPEN_PERCENT = 0.05
    MIN_ACT_PERCENT = 0.15

    def __init__(self) -> None:
        """Initialize the tension analyzer."""
        self.base_tension = 0.1  # Starting tension level

    def analyze(self, beats: List[DramaticBeat]) -> TensionCurve:
        """
        Analyze beats to produce tension curve and act structure.

        Args:
            beats: List of dramatic beats in chronological order

        Returns:
            Complete tension curve analysis
        """
        if not beats:
            logger.warning("No beats provided for tension analysis")
            return TensionCurve()

        curve = TensionCurve()

        # Calculate tension at each beat
        curve.points = self._calculate_tension_points(beats)

        # Identify key story moments
        self._identify_key_moments(curve, beats)

        # Determine act breaks
        curve.act_breaks = self._determine_act_breaks(curve, beats)

        # Calculate statistics
        self._calculate_statistics(curve)

        logger.info(
            f"Tension analysis complete: peak={curve.peak_tension:.2f}, "
            f"acts={len(curve.act_breaks) + 1}"
        )

        return curve

    def _calculate_tension_points(
        self,
        beats: List[DramaticBeat],
    ) -> List[TensionPoint]:
        """
        Calculate cumulative tension at each beat.

        Args:
            beats: Dramatic beats in order

        Returns:
            List of tension points
        """
        points = []
        current_tension = self.base_tension

        for beat in beats:
            # Apply tension delta with decay
            current_tension = self._apply_tension_change(
                current_tension,
                beat.tension_delta,
            )

            # Update beat's tension level
            beat.tension_level = current_tension

            points.append(TensionPoint(
                timestamp=beat.timestamp,
                tension=current_tension,
                beat=beat,
            ))

        return points

    def _apply_tension_change(
        self,
        current: float,
        delta: float,
        min_tension: float = 0.05,
        max_tension: float = 1.0,
    ) -> float:
        """
        Apply a tension change with realistic dynamics.

        Tension rises quickly but falls slowly (psychological realism).

        Args:
            current: Current tension level
            delta: Change to apply
            min_tension: Minimum tension floor
            max_tension: Maximum tension ceiling

        Returns:
            New tension level
        """
        if delta > 0:
            # Tension rises quickly
            new_tension = current + delta
        else:
            # Tension falls more slowly (half the rate)
            new_tension = current + (delta * 0.5)

        return max(min_tension, min(max_tension, new_tension))

    def _identify_key_moments(
        self,
        curve: TensionCurve,
        beats: List[DramaticBeat],
    ) -> None:
        """
        Identify the key dramatic moments in the story.

        Args:
            curve: Tension curve to populate
            beats: All dramatic beats
        """
        # Find inciting incident (first significant tension rise)
        for beat in beats:
            if beat.beat_type in (
                BeatType.INCITING_INCIDENT,
                BeatType.HOSTILE_CONTACT if hasattr(BeatType, 'HOSTILE_CONTACT') else None,
            ):
                curve.inciting_incident = beat
                break
            if beat.tension_delta >= 0.15:
                curve.inciting_incident = beat
                break

        # Find crisis point (highest tension before resolution)
        crisis_candidates = [
            b for b in beats
            if b.beat_type == BeatType.CRISIS_POINT
        ]
        if crisis_candidates:
            curve.crisis_point = max(crisis_candidates, key=lambda b: b.tension_level)
        elif curve.points:
            # Use highest tension point
            max_point = max(curve.points, key=lambda p: p.tension)
            curve.crisis_point = max_point.beat

        # Find climax (decisive action near peak tension)
        climax_candidates = [
            b for b in beats
            if b.beat_type == BeatType.CLIMAX
        ]
        if climax_candidates:
            curve.climax = climax_candidates[-1]  # Last climax moment
        elif curve.crisis_point:
            # Climax is often just after crisis
            crisis_idx = beats.index(curve.crisis_point) if curve.crisis_point in beats else -1
            if crisis_idx >= 0 and crisis_idx < len(beats) - 1:
                curve.climax = beats[crisis_idx + 1]

        # Find resolution
        resolution_candidates = [
            b for b in beats
            if b.beat_type in (BeatType.RESOLUTION, BeatType.TRAGIC_RESOLUTION)
        ]
        if resolution_candidates:
            curve.resolution = resolution_candidates[-1]

    def _determine_act_breaks(
        self,
        curve: TensionCurve,
        beats: List[DramaticBeat],
    ) -> List[ActBreak]:
        """
        Determine where act breaks should occur.

        Uses a combination of:
        - Tension thresholds
        - Key story moments
        - Time distribution

        Args:
            curve: Tension curve with key moments identified
            beats: All dramatic beats

        Returns:
            List of act breaks
        """
        if not beats:
            return []

        act_breaks = []
        mission_duration = (beats[-1].timestamp - beats[0].timestamp).total_seconds()

        # Find Cold Open end (hook moment or time-based)
        cold_open_end = self._find_cold_open_end(beats, mission_duration)
        if cold_open_end:
            act_breaks.append(ActBreak(
                timestamp=cold_open_end.timestamp,
                from_act=ActType.COLD_OPEN,
                to_act=ActType.ACT_ONE,
                trigger_beat=cold_open_end,
                description="Title card - mission truly begins",
            ))

        # Act One ends at inciting incident
        if curve.inciting_incident:
            act_breaks.append(ActBreak(
                timestamp=curve.inciting_incident.timestamp,
                from_act=ActType.ACT_ONE,
                to_act=ActType.ACT_TWO,
                trigger_beat=curve.inciting_incident,
                description="The problem emerges",
            ))

        # Act Two ends when tension crosses threshold or at midpoint complication
        act_two_end = self._find_act_two_end(curve, beats, mission_duration)
        if act_two_end:
            act_breaks.append(ActBreak(
                timestamp=act_two_end.timestamp,
                from_act=ActType.ACT_TWO,
                to_act=ActType.ACT_THREE,
                trigger_beat=act_two_end,
                description="Complications escalate",
            ))

        # Act Three ends at crisis point
        if curve.crisis_point:
            act_breaks.append(ActBreak(
                timestamp=curve.crisis_point.timestamp,
                from_act=ActType.ACT_THREE,
                to_act=ActType.ACT_FOUR,
                trigger_beat=curve.crisis_point,
                description="The darkest moment",
            ))

        # Sort by timestamp and remove duplicates
        act_breaks.sort(key=lambda ab: ab.timestamp)
        act_breaks = self._deduplicate_act_breaks(act_breaks)

        return act_breaks

    def _find_cold_open_end(
        self,
        beats: List[DramaticBeat],
        mission_duration: float,
    ) -> Optional[DramaticBeat]:
        """Find the beat that ends the cold open."""
        min_cold_open = mission_duration * self.MIN_COLD_OPEN_PERCENT
        max_cold_open = mission_duration * 0.15  # Max 15% for cold open

        start_time = beats[0].timestamp

        for beat in beats:
            elapsed = (beat.timestamp - start_time).total_seconds()

            # Must be past minimum duration
            if elapsed < min_cold_open:
                continue

            # Must be before maximum duration
            if elapsed > max_cold_open:
                # Force break at max duration
                return beat

            # Look for a hook moment
            if beat.beat_type in (
                BeatType.COLD_OPEN_HOOK,
                BeatType.INCITING_INCIDENT,
                BeatType.DISCOVERY,
            ):
                return beat

            # Or first tension spike
            if beat.tension_delta >= 0.15:
                return beat

        # Default to first beat after minimum duration
        for beat in beats:
            if (beat.timestamp - start_time).total_seconds() >= min_cold_open:
                return beat

        return beats[0] if beats else None

    def _find_act_two_end(
        self,
        curve: TensionCurve,
        beats: List[DramaticBeat],
        mission_duration: float,
    ) -> Optional[DramaticBeat]:
        """Find the beat that ends Act Two."""
        # Look for midpoint complication or tension threshold crossing
        midpoint_time = beats[0].timestamp + timedelta(seconds=mission_duration * 0.5)

        # Find beat closest to midpoint with significant tension
        candidates = []

        for beat in beats:
            if beat.tension_level >= self.ACT_TWO_TRIGGER_TENSION:
                candidates.append(beat)

            # Also consider beats near midpoint
            time_to_mid = abs((beat.timestamp - midpoint_time).total_seconds())
            if time_to_mid < mission_duration * 0.1:  # Within 10% of midpoint
                candidates.append(beat)

        if candidates:
            # Prefer escalation or complication beats
            for c in candidates:
                if c.beat_type in (BeatType.ESCALATION, BeatType.COMPLICATION):
                    return c
            return candidates[0]

        return None

    def _deduplicate_act_breaks(
        self,
        breaks: List[ActBreak],
    ) -> List[ActBreak]:
        """Remove duplicate or too-close act breaks."""
        if len(breaks) <= 1:
            return breaks

        deduped = [breaks[0]]

        for ab in breaks[1:]:
            # Skip if same act transition
            if ab.from_act == deduped[-1].from_act:
                continue

            # Skip if too close (within 30 seconds)
            time_diff = (ab.timestamp - deduped[-1].timestamp).total_seconds()
            if time_diff < 30:
                continue

            deduped.append(ab)

        return deduped

    def _calculate_statistics(self, curve: TensionCurve) -> None:
        """Calculate summary statistics for the tension curve."""
        if not curve.points:
            return

        tensions = [p.tension for p in curve.points]

        # Peak tension
        curve.peak_tension = max(tensions)
        peak_point = max(curve.points, key=lambda p: p.tension)
        curve.peak_timestamp = peak_point.timestamp

        # Average tension
        curve.average_tension = sum(tensions) / len(tensions)

        # Volatility (standard deviation of tension changes)
        if len(tensions) > 1:
            changes = [
                abs(tensions[i] - tensions[i - 1])
                for i in range(1, len(tensions))
            ]
            curve.tension_volatility = sum(changes) / len(changes)


def format_tension_curve_ascii(curve: TensionCurve, width: int = 60) -> str:
    """
    Create an ASCII visualization of the tension curve.

    Args:
        curve: Tension curve to visualize
        width: Character width of the visualization

    Returns:
        ASCII art string
    """
    if not curve.points:
        return "No tension data available"

    height = 10
    output = []

    # Create the graph
    tensions = [p.tension for p in curve.points]
    num_points = len(tensions)

    # Sample points to fit width
    if num_points > width:
        step = num_points / width
        sampled = [tensions[int(i * step)] for i in range(width)]
    else:
        sampled = tensions

    # Build rows from top to bottom
    for row in range(height, -1, -1):
        threshold = row / height
        line = ""

        for t in sampled:
            if t >= threshold:
                if t >= 0.8:
                    line += "█"
                elif t >= 0.6:
                    line += "▓"
                elif t >= 0.4:
                    line += "▒"
                else:
                    line += "░"
            else:
                line += " "

        # Add axis label
        if row == height:
            label = "1.0│"
        elif row == height // 2:
            label = "0.5│"
        elif row == 0:
            label = "0.0│"
        else:
            label = "   │"

        output.append(f"{label}{line}")

    # Add bottom axis
    output.append("   └" + "─" * len(sampled))
    output.append("    COLD  ACT1  ACT2  ACT3  ACT4")

    # Add key moments markers
    stats = [
        f"Peak: {curve.peak_tension:.2f}",
        f"Avg: {curve.average_tension:.2f}",
        f"Acts: {len(curve.act_breaks) + 1}",
    ]
    output.append("    " + " | ".join(stats))

    return "\n".join(output)
