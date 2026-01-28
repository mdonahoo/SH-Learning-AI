"""
Bridge crew stress and sentiment analyzer.

Five-signal hybrid scoring per transcript segment, using speech rate,
domain-specific stress lexicon, Whisper confidence variance, prosodic
text markers, and utterance timing dynamics. No external NLP packages
required — uses only numpy and regex.
"""

import logging
import re
from typing import Dict, List, Any, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_SPEECH_RATE = 0.25
WEIGHT_LEXICON = 0.30
WEIGHT_CONFIDENCE = 0.15
WEIGHT_PROSODIC = 0.15
WEIGHT_DYNAMICS = 0.15

# ---------------------------------------------------------------------------
# Bridge-specific stress lexicon
# ---------------------------------------------------------------------------
STRESS_TERMS: Dict[str, float] = {
    # Critical (0.8-1.0)
    "red alert": 0.95,
    "shields failing": 0.90,
    "hull breach": 0.95,
    "we're hit": 0.85,
    "enemy": 0.70,
    "damage report": 0.75,
    "evasive": 0.80,
    "brace": 0.90,
    "abandon": 0.95,
    "abandon ship": 0.98,
    "mayday": 0.95,
    "critical": 0.80,
    "overload": 0.75,
    "explosion": 0.85,
    "casualties": 0.90,
    "life support": 0.80,
    # High (0.6-0.8)
    "fire": 0.70,
    "incoming": 0.75,
    "torpedo": 0.70,
    "losing power": 0.70,
    "warning": 0.65,
    "danger": 0.70,
    "alert": 0.60,
    "shields down": 0.75,
    "weapons": 0.60,
    "hostiles": 0.70,
    "under attack": 0.80,
    "taking damage": 0.75,
    # Moderate (0.3-0.6)
    "contact": 0.40,
    "unknown": 0.45,
    "scan": 0.30,
    "unidentified": 0.45,
    "intercept": 0.50,
    "approach": 0.35,
    "anomaly": 0.40,
    "interference": 0.40,
    "fluctuation": 0.35,
}

CALM_TERMS: Dict[str, float] = {
    "acknowledged": -0.30,
    "aye": -0.25,
    "aye aye": -0.30,
    "steady": -0.30,
    "all clear": -0.50,
    "on course": -0.30,
    "nominal": -0.40,
    "standing by": -0.30,
    "secure": -0.35,
    "stable": -0.30,
    "green": -0.25,
    "all systems go": -0.40,
    "roger": -0.25,
    "copy": -0.20,
    "understood": -0.25,
    "affirmative": -0.25,
    "steady as she goes": -0.40,
    "no contacts": -0.35,
    "clear": -0.30,
}

# Baseline words-per-second for normal speech
NEUTRAL_BASELINE_WPS = 2.5

# Emotion classification thresholds
EMOTION_MAP = [
    (0.80, "critical"),
    (0.60, "urgent"),
    (0.40, "tense"),
    (0.20, "focused"),
    (0.00, "calm"),
]


class BridgeSentimentAnalyzer:
    """
    Hybrid stress/sentiment analyzer for bridge crew communications.

    Combines five signals — speech rate, lexicon matches, confidence
    variance, prosodic markers, and utterance dynamics — into a
    composite stress score per segment.

    Attributes:
        stress_terms: Domain-specific high-stress keywords and scores.
        calm_terms: Domain-specific calming keywords and scores.
        neutral_baseline_wps: Baseline words-per-second for normal speech.
    """

    def __init__(
        self,
        stress_terms: Optional[Dict[str, float]] = None,
        calm_terms: Optional[Dict[str, float]] = None,
        neutral_baseline_wps: float = NEUTRAL_BASELINE_WPS,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            stress_terms: Override stress keyword dict (keyword -> 0-1 score).
            calm_terms: Override calm keyword dict (keyword -> negative score).
            neutral_baseline_wps: Words-per-second considered normal pace.
        """
        self.stress_terms = stress_terms or STRESS_TERMS
        self.calm_terms = calm_terms or CALM_TERMS
        self.neutral_baseline_wps = neutral_baseline_wps

    def analyze_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze all segments, returning per-segment scores and a summary.

        Each segment dict should have at minimum:
            - text (str): Transcribed text.
            - start (float) or start_time (float): Segment start in seconds.
            - end (float) or end_time (float): Segment end in seconds.

        Optional segment keys:
            - confidence (float): Whisper transcription confidence.
            - words (list[dict]): Word-level timestamps with 'start', 'end', 'probability'.
            - speaker_id (str): Speaker identifier.

        Args:
            segments: List of transcription segment dicts.

        Returns:
            Dict with 'segments' (list of per-segment sentiment dicts)
            and 'summary' (aggregate statistics).
        """
        if not segments:
            return {
                'segments': [],
                'summary': self._empty_summary(),
            }

        # Compute global average confidence for relative scoring
        confidences = [
            s.get('confidence', 0.0)
            for s in segments
            if s.get('confidence') is not None
        ]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        scored_segments: List[Dict[str, Any]] = []

        for i, seg in enumerate(segments):
            prev_seg = segments[i - 1] if i > 0 else None
            next_seg = segments[i + 1] if i < len(segments) - 1 else None

            scored = self._score_segment(seg, prev_seg, next_seg, avg_confidence)
            scored_segments.append(scored)

        summary = self._build_summary(scored_segments, segments)

        return {
            'segments': scored_segments,
            'summary': summary,
        }

    # ------------------------------------------------------------------
    # Individual signal scoring (all return 0-1 float)
    # ------------------------------------------------------------------

    def _score_speech_rate(self, seg: Dict[str, Any]) -> float:
        """
        Score stress from speech rate. Fast talking indicates urgency.

        Uses word-level timestamps if available, otherwise estimates
        from text word count and segment duration.

        Args:
            seg: Segment dict.

        Returns:
            Stress score 0-1. Above 3.5 wps maps to high stress.
        """
        words = seg.get('words', [])
        text = seg.get('text', '')
        start = seg.get('start', seg.get('start_time', 0))
        end = seg.get('end', seg.get('end_time', 0))
        duration = end - start

        if duration <= 0:
            return 0.0

        if words and len(words) > 0:
            word_count = len(words)
        else:
            word_count = len(text.split())

        wps = word_count / duration

        # Map wps to 0-1 stress:
        # <= 1.5 wps → 0.0 (very slow, calm)
        # 2.5 wps → 0.3 (normal)
        # 3.5 wps → 0.7 (fast, stressed)
        # >= 5.0 wps → 1.0 (very fast, panic)
        if wps <= 1.5:
            return 0.0
        elif wps <= self.neutral_baseline_wps:
            return 0.3 * (wps - 1.5) / (self.neutral_baseline_wps - 1.5)
        elif wps <= 5.0:
            return 0.3 + 0.7 * (wps - self.neutral_baseline_wps) / (5.0 - self.neutral_baseline_wps)
        else:
            return 1.0

    def _score_lexicon(self, text: str) -> float:
        """
        Score stress from keyword matches in text.

        Scans for stress and calm terms, weighted by their lexicon scores.
        Result is clamped to 0-1.

        Args:
            text: Segment text.

        Returns:
            Stress score 0-1.
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.0
        matches = 0

        # Check stress terms (longer phrases first for greedy matching)
        for term, weight in sorted(
            self.stress_terms.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if term in text_lower:
                score += weight
                matches += 1

        # Check calm terms
        for term, weight in sorted(
            self.calm_terms.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if term in text_lower:
                score += weight  # weight is negative
                matches += 1

        if matches == 0:
            return 0.0

        # Average and clamp to 0-1
        avg = score / matches
        return float(np.clip(avg, 0.0, 1.0))

    def _score_confidence(self, confidence: float, avg_confidence: float) -> float:
        """
        Score stress from transcription confidence.

        Low confidence relative to average may indicate shouting,
        overlapping speech, or environmental noise — all correlated
        with stressful situations.

        Args:
            confidence: Segment confidence score.
            avg_confidence: Global average confidence.

        Returns:
            Stress score 0-1.
        """
        if avg_confidence <= 0:
            return 0.0

        # How far below average is this segment's confidence?
        # Large negative deviation = high stress
        deviation = avg_confidence - confidence

        if deviation <= 0:
            return 0.0
        elif deviation >= 0.3:
            return 1.0
        else:
            return deviation / 0.3

    def _score_prosodic(self, text: str) -> float:
        """
        Score stress from text prosodic markers.

        Detects exclamation marks, ALL CAPS words, short rapid
        utterances, and repeated punctuation.

        Args:
            text: Segment text.

        Returns:
            Stress score 0-1.
        """
        if not text or not text.strip():
            return 0.0

        signals = []

        # Exclamation marks
        excl_count = text.count('!')
        if excl_count >= 3:
            signals.append(1.0)
        elif excl_count >= 1:
            signals.append(0.5 + 0.25 * min(excl_count, 2))

        # ALL CAPS words (exclude single-letter words like "I")
        words = text.split()
        caps_words = [
            w for w in words
            if len(w) > 1 and w.isupper() and w.isalpha()
        ]
        caps_ratio = len(caps_words) / max(len(words), 1)
        if caps_ratio > 0.5:
            signals.append(0.9)
        elif caps_ratio > 0.2:
            signals.append(0.6)
        elif caps_ratio > 0:
            signals.append(0.3)

        # Very short utterance (1-3 words) — can indicate urgency
        if 1 <= len(words) <= 3:
            signals.append(0.4)

        # Question marks don't necessarily indicate stress but repeated ones do
        if text.count('?') >= 2:
            signals.append(0.3)

        if not signals:
            return 0.0

        return float(np.clip(np.mean(signals), 0.0, 1.0))

    def _score_dynamics(
        self,
        seg: Dict[str, Any],
        prev_seg: Optional[Dict[str, Any]],
        next_seg: Optional[Dict[str, Any]],
    ) -> float:
        """
        Score stress from utterance timing dynamics.

        Rapid-fire exchanges (short gaps between utterances) indicate
        tension or urgency on the bridge.

        Args:
            seg: Current segment.
            prev_seg: Previous segment (or None).
            next_seg: Next segment (or None).

        Returns:
            Stress score 0-1.
        """
        gaps = []

        seg_start = seg.get('start', seg.get('start_time', 0))
        seg_end = seg.get('end', seg.get('end_time', 0))

        if prev_seg:
            prev_end = prev_seg.get('end', prev_seg.get('end_time', 0))
            gap_before = seg_start - prev_end
            if gap_before >= 0:
                gaps.append(gap_before)

        if next_seg:
            next_start = next_seg.get('start', next_seg.get('start_time', 0))
            gap_after = next_start - seg_end
            if gap_after >= 0:
                gaps.append(gap_after)

        if not gaps:
            return 0.0

        avg_gap = float(np.mean(gaps))

        # Map gap to stress:
        # < 0.3s → 1.0 (rapid fire)
        # 0.5s → 0.7
        # 1.0s → 0.4
        # 2.0s → 0.1
        # > 3.0s → 0.0 (relaxed pacing)
        if avg_gap < 0.3:
            return 1.0
        elif avg_gap < 3.0:
            # Linear decay from 1.0 to 0.0 over 0.3-3.0s
            return 1.0 - (avg_gap - 0.3) / 2.7
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Segment-level composite scoring
    # ------------------------------------------------------------------

    def _score_segment(
        self,
        seg: Dict[str, Any],
        prev_seg: Optional[Dict[str, Any]],
        next_seg: Optional[Dict[str, Any]],
        avg_confidence: float,
    ) -> Dict[str, Any]:
        """
        Compute composite stress score for one segment.

        Args:
            seg: Current segment dict.
            prev_seg: Previous segment or None.
            next_seg: Next segment or None.
            avg_confidence: Global average confidence.

        Returns:
            Sentiment dict with stress_level, label, emotion, etc.
        """
        text = seg.get('text', '')
        confidence = seg.get('confidence', avg_confidence)

        rate_score = self._score_speech_rate(seg)
        lexicon_score = self._score_lexicon(text)
        conf_score = self._score_confidence(confidence, avg_confidence)
        prosodic_score = self._score_prosodic(text)
        dynamics_score = self._score_dynamics(seg, prev_seg, next_seg)

        # Compute speech rate for reporting
        start = seg.get('start', seg.get('start_time', 0))
        end = seg.get('end', seg.get('end_time', 0))
        duration = end - start
        words_list = seg.get('words', [])
        if words_list:
            wps = len(words_list) / duration if duration > 0 else 0.0
        else:
            wps = len(text.split()) / duration if duration > 0 else 0.0

        # Weighted composite
        stress_level = (
            WEIGHT_SPEECH_RATE * rate_score
            + WEIGHT_LEXICON * lexicon_score
            + WEIGHT_CONFIDENCE * conf_score
            + WEIGHT_PROSODIC * prosodic_score
            + WEIGHT_DYNAMICS * dynamics_score
        )
        stress_level = float(np.clip(stress_level, 0.0, 1.0))

        stress_label = self._classify_stress(stress_level)
        emotion = self._classify_emotion(stress_level, text)
        sentiment, sentiment_score = self._classify_sentiment(stress_level, text)

        return {
            'stress_level': round(stress_level, 4),
            'stress_label': stress_label,
            'emotion': emotion,
            'sentiment': sentiment,
            'sentiment_score': round(sentiment_score, 4),
            'signals': {
                'speech_rate_wps': round(wps, 2),
                'lexicon_score': round(lexicon_score, 4),
                'confidence_signal': round(conf_score, 4),
                'prosodic_score': round(prosodic_score, 4),
                'dynamics_score': round(dynamics_score, 4),
            },
        }

    def _classify_stress(self, stress: float) -> str:
        """
        Map stress score to label.

        Args:
            stress: 0-1 stress score.

        Returns:
            'low', 'moderate', or 'high'.
        """
        if stress >= 0.6:
            return "high"
        elif stress >= 0.3:
            return "moderate"
        else:
            return "low"

    def _classify_emotion(self, stress: float, text: str) -> str:
        """
        Map stress level and text to an emotion label.

        Args:
            stress: 0-1 stress score.
            text: Segment text for context.

        Returns:
            One of: calm, focused, tense, urgent, critical.
        """
        for threshold, label in EMOTION_MAP:
            if stress >= threshold:
                return label
        return "calm"

    def _classify_sentiment(self, stress: float, text: str) -> tuple:
        """
        Derive sentiment polarity from stress and text.

        Args:
            stress: 0-1 stress score.
            text: Segment text.

        Returns:
            Tuple of (sentiment_label, sentiment_score).
            sentiment_score ranges from -1 (negative) to +1 (positive).
        """
        # High stress = negative sentiment, low stress = positive
        # Map 0-1 stress to +1 to -1 sentiment
        sentiment_score = 1.0 - 2.0 * stress

        # Check for positive indicators in calm speech
        text_lower = text.lower() if text else ''
        positive_words = ['good', 'great', 'excellent', 'well done', 'perfect', 'nice']
        has_positive = any(pw in text_lower for pw in positive_words)

        if has_positive and stress < 0.3:
            sentiment_score = min(1.0, sentiment_score + 0.3)

        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment, sentiment_score

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        scored_segments: List[Dict[str, Any]],
        raw_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build aggregate summary from scored segments.

        Args:
            scored_segments: Per-segment sentiment results.
            raw_segments: Original segment dicts (for timing info).

        Returns:
            Summary dict with averages, distributions, per-speaker stats,
            and a downsampled stress timeline.
        """
        if not scored_segments:
            return self._empty_summary()

        stress_values = [s['stress_level'] for s in scored_segments]
        avg_stress = float(np.mean(stress_values))

        # Stress distribution
        labels = [s['stress_label'] for s in scored_segments]
        total = len(labels)
        distribution = {
            'low': round(100 * labels.count('low') / total),
            'moderate': round(100 * labels.count('moderate') / total),
            'high': round(100 * labels.count('high') / total),
        }

        # Peak stress time
        peak_idx = int(np.argmax(stress_values))
        peak_seg = raw_segments[peak_idx]
        peak_time = peak_seg.get('start', peak_seg.get('start_time', 0))

        # Per-speaker stress stats
        speaker_stress = self._compute_speaker_stress(scored_segments, raw_segments)

        # Downsampled stress timeline (one point per ~10 seconds)
        timeline = self._build_timeline(scored_segments, raw_segments)

        return {
            'average_stress': round(avg_stress, 4),
            'peak_stress_time': round(peak_time, 2),
            'peak_stress_level': round(float(max(stress_values)), 4),
            'stress_distribution': distribution,
            'speaker_stress': speaker_stress,
            'stress_timeline': timeline,
        }

    def _compute_speaker_stress(
        self,
        scored_segments: List[Dict[str, Any]],
        raw_segments: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute per-speaker stress statistics.

        Args:
            scored_segments: Per-segment sentiment results.
            raw_segments: Original segment dicts.

        Returns:
            Dict keyed by speaker_id with avg_stress, peak, dominant_emotion.
        """
        speaker_data: Dict[str, List[Dict[str, Any]]] = {}

        for scored, raw in zip(scored_segments, raw_segments):
            speaker = raw.get('speaker_id', raw.get('speaker', 'unknown'))
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append(scored)

        result = {}
        for speaker, scores in speaker_data.items():
            stress_vals = [s['stress_level'] for s in scores]
            emotions = [s['emotion'] for s in scores]

            # Most common emotion
            emotion_counts: Dict[str, int] = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)

            result[speaker] = {
                'avg_stress': round(float(np.mean(stress_vals)), 4),
                'peak': round(float(np.max(stress_vals)), 4),
                'dominant_emotion': dominant_emotion,
                'segment_count': len(scores),
            }

        return result

    def _build_timeline(
        self,
        scored_segments: List[Dict[str, Any]],
        raw_segments: List[Dict[str, Any]],
        bucket_seconds: float = 10.0,
    ) -> List[Dict[str, float]]:
        """
        Build downsampled stress timeline for frontend chart.

        Groups segments into time buckets and averages their stress.

        Args:
            scored_segments: Per-segment sentiment results.
            raw_segments: Original segment dicts.
            bucket_seconds: Time bucket size in seconds.

        Returns:
            List of {time, stress} dicts.
        """
        if not raw_segments:
            return []

        # Determine time range
        starts = [
            s.get('start', s.get('start_time', 0)) for s in raw_segments
        ]
        ends = [
            s.get('end', s.get('end_time', 0)) for s in raw_segments
        ]
        min_time = min(starts)
        max_time = max(ends)

        if max_time <= min_time:
            return [{'time': min_time, 'stress': scored_segments[0]['stress_level']}]

        # Create buckets
        timeline = []
        t = min_time
        while t < max_time:
            bucket_end = t + bucket_seconds
            # Average stress of segments overlapping this bucket
            bucket_scores = []
            for scored, raw in zip(scored_segments, raw_segments):
                seg_start = raw.get('start', raw.get('start_time', 0))
                seg_end = raw.get('end', raw.get('end_time', 0))
                # Check for overlap
                if seg_start < bucket_end and seg_end > t:
                    bucket_scores.append(scored['stress_level'])

            if bucket_scores:
                avg = float(np.mean(bucket_scores))
            else:
                # Interpolate from nearest neighbor if no segments in bucket
                avg = 0.0

            timeline.append({
                'time': round(t, 2),
                'stress': round(avg, 4),
            })
            t += bucket_seconds

        return timeline

    def _empty_summary(self) -> Dict[str, Any]:
        """Return a zeroed summary for empty input."""
        return {
            'average_stress': 0.0,
            'peak_stress_time': 0.0,
            'peak_stress_level': 0.0,
            'stress_distribution': {'low': 100, 'moderate': 0, 'high': 0},
            'speaker_stress': {},
            'stress_timeline': [],
        }
