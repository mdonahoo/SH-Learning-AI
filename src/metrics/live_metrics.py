"""
Live metrics computation for real-time dashboard updates.

Wraps existing analyzers (sentiment, communication quality, seven habits)
to produce lightweight aggregate metrics from streaming transcript segments.
All computations are regex-based and run in <5ms on typical segment counts.
"""

import logging
import math
import re
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class LiveMetricsComputer:
    """
    Computes team-level metrics from accumulated transcript segments.

    Designed for real-time use during streaming transcription.
    Each sub-computation is isolated so a failure in one metric
    does not prevent the others from updating.

    Attributes:
        _sentiment_analyzer: Lazy-loaded BridgeSentimentAnalyzer instance.
        _habits_analyzer_cls: SevenHabitsAnalyzer class reference.
        _effective_patterns: Compiled effective communication patterns.
        _improvement_patterns: Compiled improvement communication patterns.
    """

    # CES trend window in seconds
    _CES_TREND_WINDOW = 60.0
    # CES trend stability threshold in points
    _CES_TREND_THRESHOLD = 3.0

    def __init__(self) -> None:
        """Initialize with lazy-loaded analyzer references."""
        self._sentiment_analyzer: Any = None
        self._habits_analyzer_cls: Any = None
        self._effective_patterns: List[Dict[str, Any]] = []
        self._improvement_patterns: List[Dict[str, Any]] = []
        self._analyze_teamstepps: Any = None
        self._analyze_nasa4d: Any = None
        self._analyze_bloom: Any = None
        self._initialized = False
        self._ces_history: List[Tuple[float, float]] = []

    def _ensure_initialized(self) -> None:
        """Lazy-load analyzer classes on first use."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from src.metrics.sentiment_analyzer import BridgeSentimentAnalyzer
            self._sentiment_analyzer = BridgeSentimentAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to load BridgeSentimentAnalyzer: {e}")

        try:
            from src.metrics.seven_habits import SevenHabitsAnalyzer
            self._habits_analyzer_cls = SevenHabitsAnalyzer
        except Exception as e:
            logger.warning(f"Failed to load SevenHabitsAnalyzer: {e}")

        try:
            from src.llm.scientific_frameworks import (
                analyze_teamstepps,
                analyze_nasa_4d,
                analyze_bloom_levels,
            )
            self._analyze_teamstepps = analyze_teamstepps
            self._analyze_nasa4d = analyze_nasa_4d
            self._analyze_bloom = analyze_bloom_levels
        except Exception as e:
            logger.warning(f"Failed to load scientific frameworks: {e}")

        try:
            from src.metrics.communication_quality import (
                EFFECTIVE_PATTERNS,
                IMPROVEMENT_PATTERNS,
            )
            self._effective_patterns = [
                {
                    'name': p.name,
                    'description': p.description,
                    'compiled': [re.compile(r) for r in p.patterns],
                }
                for p in EFFECTIVE_PATTERNS
            ]
            self._improvement_patterns = [
                {
                    'name': p.name,
                    'description': p.description,
                    'compiled': [re.compile(r) for r in p.patterns],
                }
                for p in IMPROVEMENT_PATTERNS
            ]
        except Exception as e:
            logger.warning(f"Failed to load communication patterns: {e}")

    def compute(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run all metrics on accumulated segments.

        Each metric is computed independently with its own error handling.

        Args:
            segments: List of transcript segment dicts with at least
                'text', 'start', and 'end' keys.

        Returns:
            Dict with keys: stress, communication, speech, habits.
            Each key maps to a metric dict or default values on failure.
        """
        self._ensure_initialized()

        result: Dict[str, Any] = {}

        try:
            result['stress'] = self._compute_stress(segments)
        except Exception as e:
            logger.warning(f"Stress metric failed: {e}")
            result['stress'] = {
                'avg': 0.0, 'peak': 0.0,
                'label': 'unknown', 'distribution': {}
            }

        try:
            result['communication'] = self._compute_communication(segments)
        except Exception as e:
            logger.warning(f"Communication metric failed: {e}")
            result['communication'] = {
                'effective_count': 0, 'improvement_count': 0,
                'effective_pct': 0.0, 'recent_patterns': []
            }

        try:
            result['speech'] = self._compute_speech(segments)
        except Exception as e:
            logger.warning(f"Speech metric failed: {e}")
            result['speech'] = {
                'avg_wps': 0.0, 'total_words': 0,
                'total_duration_s': 0.0
            }

        try:
            result['habits'] = self._compute_habits(segments)
        except Exception as e:
            logger.warning(f"Habits metric failed: {e}")
            result['habits'] = []

        try:
            result['teamstepps'] = self._compute_teamstepps(segments)
        except Exception as e:
            logger.warning(f"TeamSTEPPS metric failed: {e}")
            result['teamstepps'] = []

        try:
            result['nasa4d'] = self._compute_nasa4d(segments)
        except Exception as e:
            logger.warning(f"NASA 4-D metric failed: {e}")
            result['nasa4d'] = []

        try:
            result['bloom'] = self._compute_bloom(segments)
        except Exception as e:
            logger.warning(f"Bloom's Taxonomy metric failed: {e}")
            result['bloom'] = {'levels': [], 'avg_level': 1.0, 'total': 0}

        try:
            result['crew_effectiveness'] = self._compute_crew_effectiveness(
                result['stress'], result['communication'],
                result['habits'], result['speech'],
            )
        except Exception as e:
            logger.warning(f"Crew effectiveness metric failed: {e}")
            result['crew_effectiveness'] = {
                'score': 0.0, 'label': 'Struggling',
                'trend': '→', 'components': {},
            }

        return result

    def _compute_stress(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute stress metrics using BridgeSentimentAnalyzer.

        Args:
            segments: Transcript segments.

        Returns:
            Dict with avg, peak, label, and distribution.
        """
        if not self._sentiment_analyzer or not segments:
            return {
                'avg': 0.0, 'peak': 0.0,
                'label': 'calm', 'distribution': {}
            }

        result = self._sentiment_analyzer.analyze_segments(segments)
        summary = result.get('summary', {})

        avg = summary.get('average_stress', 0.0)
        if avg < 0.3:
            label = 'calm'
        elif avg < 0.6:
            label = 'tense'
        else:
            label = 'critical'

        return {
            'avg': round(avg, 3),
            'peak': round(summary.get('peak_stress_level', 0.0), 3),
            'label': label,
            'distribution': summary.get('stress_distribution', {}),
        }

    def _compute_communication(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute communication quality by matching patterns against text.

        Runs EFFECTIVE_PATTERNS and IMPROVEMENT_PATTERNS regex directly
        on segment text to avoid instantiating the full analyzer.

        Args:
            segments: Transcript segments.

        Returns:
            Dict with effective_count, improvement_count, effective_pct,
            and recent_patterns list.
        """
        if not segments:
            return {
                'effective_count': 0, 'improvement_count': 0,
                'effective_pct': 0.0, 'recent_patterns': []
            }

        effective_count = 0
        improvement_count = 0
        recent_patterns: List[Dict[str, str]] = []

        for seg in segments:
            text = seg.get('text', '')
            if not text:
                continue

            for pat_group in self._effective_patterns:
                for compiled in pat_group['compiled']:
                    if compiled.search(text):
                        effective_count += 1
                        recent_patterns.append({
                            'category': 'effective',
                            'name': pat_group['name'],
                            'text': text[:80],
                        })
                        break

            for pat_group in self._improvement_patterns:
                for compiled in pat_group['compiled']:
                    if compiled.search(text):
                        improvement_count += 1
                        recent_patterns.append({
                            'category': 'improvement',
                            'name': pat_group['name'],
                            'text': text[:80],
                        })
                        break

        total = effective_count + improvement_count
        effective_pct = (
            round(effective_count / total * 100, 1) if total > 0 else 0.0
        )

        return {
            'effective_count': effective_count,
            'improvement_count': improvement_count,
            'effective_pct': effective_pct,
            'recent_patterns': recent_patterns[-5:],
        }

    def _compute_speech(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute speech energy metrics (words per second).

        Args:
            segments: Transcript segments.

        Returns:
            Dict with avg_wps, total_words, and total_duration_s.
        """
        if not segments:
            return {
                'avg_wps': 0.0, 'total_words': 0,
                'total_duration_s': 0.0
            }

        total_words = 0
        total_duration = 0.0

        for seg in segments:
            text = seg.get('text', '')
            words = len(text.split()) if text else 0
            total_words += words

            start = seg.get('start', seg.get('start_time', 0.0))
            end = seg.get('end', seg.get('end_time', 0.0))
            duration = max(0.0, end - start)
            total_duration += duration

        avg_wps = (
            round(total_words / total_duration, 2)
            if total_duration > 0 else 0.0
        )

        return {
            'avg_wps': avg_wps,
            'total_words': total_words,
            'total_duration_s': round(total_duration, 1),
        }

    def _compute_habits(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute Seven Habits scores from segments.

        Args:
            segments: Transcript segments.

        Returns:
            List of dicts with habit_num, name, score, and count.
        """
        if not self._habits_analyzer_cls or not segments:
            return []

        # SevenHabitsAnalyzer expects transcripts with 'text' key
        analyzer = self._habits_analyzer_cls(segments)
        assessments = analyzer.analyze_all_habits()

        habits = []
        for habit_enum, assessment in sorted(
            assessments.items(), key=lambda x: x[0].value
        ):
            habits.append({
                'habit_num': habit_enum.value,
                'name': assessment.youth_friendly_name,
                'score': assessment.score,
                'count': assessment.count,
            })

        return habits

    @staticmethod
    def _speech_bell_curve(wps: float) -> float:
        """
        Map words-per-second to a 0-100 engagement score via bell curve.

        0 WPS = 0 (silence), 2.5 WPS = 100 (ideal), 5+ WPS = 50 (frantic).
        Uses a Gaussian centered at 2.5 with sigma tuned so that
        5 WPS maps to ~50.

        Args:
            wps: Words per second value.

        Returns:
            Score from 0 to 100.
        """
        if wps <= 0:
            return 0.0
        # Gaussian: peak at 2.5, sigma chosen so f(5)=~50
        # exp(-((5-2.5)^2)/(2*sigma^2)) = 0.5 => sigma ~= 2.126
        sigma = 2.126
        center = 2.5
        score = 100.0 * math.exp(-((wps - center) ** 2) / (2 * sigma ** 2))
        return round(max(0.0, min(100.0, score)), 1)

    def _compute_crew_effectiveness(
        self,
        stress: Dict[str, Any],
        communication: Dict[str, Any],
        habits: List[Dict[str, Any]],
        speech: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute composite Crew Effectiveness Score (CES) from sub-metrics.

        Weighted composite 0-100:
        - Stress (inverted): 25%
        - Communication quality: 25%
        - 7 Habits aggregate: 30%
        - Speech engagement (bell curve): 20%

        Args:
            stress: Stress metric dict with 'avg' key (0-1).
            communication: Communication dict with 'effective_pct' (0-100).
            habits: List of habit dicts with 'score' (1-5 each).
            speech: Speech dict with 'avg_wps'.

        Returns:
            Dict with score, label, trend, and components breakdown.
        """
        # Component: Stress (inverted) — 0-1 avg mapped to 0-100
        stress_score = round((1.0 - (stress.get('avg', 0.0))) * 100, 1)

        # Component: Communication quality — already 0-100
        comm_score = round(communication.get('effective_pct', 0.0), 1)

        # Component: 7 Habits aggregate — sum of scores (7-35) -> 0-100
        if habits:
            habit_sum = sum(h.get('score', 1) for h in habits)
            habits_score = round((habit_sum - 7) / 28 * 100, 1)
            habits_score = max(0.0, min(100.0, habits_score))
        else:
            habits_score = 0.0

        # Component: Speech engagement — bell curve on WPS
        speech_score = self._speech_bell_curve(speech.get('avg_wps', 0.0))

        # Weighted composite
        score = round(
            stress_score * 0.25
            + comm_score * 0.25
            + habits_score * 0.30
            + speech_score * 0.20,
            1,
        )
        score = max(0.0, min(100.0, score))

        # Label
        if score >= 75:
            label = 'Excellent'
        elif score >= 50:
            label = 'Effective'
        elif score >= 25:
            label = 'Developing'
        else:
            label = 'Struggling'

        # Trend — compare to score from ~60s ago
        now = time.time()
        self._ces_history.append((now, score))
        # Prune entries older than 2x the trend window
        cutoff = now - self._CES_TREND_WINDOW * 2
        self._ces_history = [
            (t, s) for t, s in self._ces_history if t >= cutoff
        ]

        trend = '→'
        if len(self._ces_history) >= 2:
            # Find the entry closest to TREND_WINDOW seconds ago
            target_time = now - self._CES_TREND_WINDOW
            past_entry = min(
                self._ces_history[:-1],
                key=lambda e: abs(e[0] - target_time),
            )
            # Only use it if it's reasonably close to the target window
            if abs(past_entry[0] - target_time) < self._CES_TREND_WINDOW:
                diff = score - past_entry[1]
                if diff > self._CES_TREND_THRESHOLD:
                    trend = '↑'
                elif diff < -self._CES_TREND_THRESHOLD:
                    trend = '↓'

        return {
            'score': score,
            'label': label,
            'trend': trend,
            'components': {
                'stress_score': stress_score,
                'comm_score': comm_score,
                'habits_score': habits_score,
                'speech_score': speech_score,
            },
        }

    # Short display names for TeamSTEPPS domains
    _TEAMSTEPPS_NAMES: Dict[str, str] = {
        'team_structure': 'Structure',
        'leadership': 'Leadership',
        'situation_monitoring': 'Monitoring',
        'mutual_support': 'Support',
        'communication': 'Communication',
    }

    # Short display names for NASA 4-D dimensions
    _NASA4D_NAMES: Dict[str, str] = {
        'cultivating': 'Cultivating',
        'visioning': 'Visioning',
        'directing': 'Directing',
        'including': 'Including',
    }

    def _compute_teamstepps(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute TeamSTEPPS domain scores from segments.

        Args:
            segments: Transcript segments.

        Returns:
            List of dicts with name, key, score, and count per domain.
        """
        if not self._analyze_teamstepps or not segments:
            return []

        raw = self._analyze_teamstepps(segments)
        result = []
        for domain, data in sorted(raw.items(), key=lambda x: x[0].value):
            key = domain.value
            result.append({
                'name': self._TEAMSTEPPS_NAMES.get(key, key),
                'key': key,
                'score': data.get('score', 1),
                'count': data.get('count', 0),
            })
        return result

    def _compute_nasa4d(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute NASA 4-D dimension scores from segments.

        Args:
            segments: Transcript segments.

        Returns:
            List of dicts with name, key, score, and count per dimension.
        """
        if not self._analyze_nasa4d or not segments:
            return []

        raw = self._analyze_nasa4d(segments)
        result = []
        for dim, data in sorted(raw.items(), key=lambda x: x[0].value):
            key = dim.value
            result.append({
                'name': self._NASA4D_NAMES.get(key, key),
                'key': key,
                'score': data.get('score', 1),
                'count': data.get('total_count', 0),
            })
        return result

    def _compute_bloom(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute Bloom's Taxonomy cognitive level distribution.

        Args:
            segments: Transcript segments.

        Returns:
            Dict with levels list, avg_level, and total classified count.
        """
        if not self._analyze_bloom or not segments:
            return {'levels': [], 'avg_level': 1.0, 'total': 0}

        raw = self._analyze_bloom(segments)
        levels_data = raw.get('levels', {})
        levels = []
        for level_enum, data in sorted(
            levels_data.items(), key=lambda x: x[0].value
        ):
            levels.append({
                'name': level_enum.name.capitalize(),
                'level': level_enum.value,
                'count': data.get('count', 0),
                'pct': data.get('percentage', 0.0),
            })

        return {
            'levels': levels,
            'avg_level': raw.get('average_cognitive_level', 1.0),
            'total': raw.get('total_classified', 0),
        }
