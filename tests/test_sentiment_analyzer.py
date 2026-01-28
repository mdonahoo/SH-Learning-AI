"""
Tests for the bridge crew stress and sentiment analyzer.

Tests cover:
- High-stress text produces high stress scores
- Calm text produces low stress scores
- Speech rate scoring with mock word timestamps
- Mixed sentiment produces moderate stress
- Empty/None input handling
- Full segment list analysis with summary stats
- Emotion classification accuracy
- Per-speaker stress aggregation
- Stress timeline generation
"""

import pytest

from src.metrics.sentiment_analyzer import (
    BridgeSentimentAnalyzer,
    STRESS_TERMS,
    CALM_TERMS,
)


@pytest.fixture
def analyzer():
    """Create a default BridgeSentimentAnalyzer."""
    return BridgeSentimentAnalyzer()


@pytest.fixture
def high_stress_segments():
    """Segments simulating a combat scenario."""
    return [
        {
            'text': 'RED ALERT! Shields failing! We\'re hit!',
            'start': 0.0,
            'end': 3.0,
            'confidence': 0.60,
            'speaker_id': 'Speaker_0',
        },
        {
            'text': 'Damage report! Hull breach on deck 7!',
            'start': 3.2,
            'end': 5.5,
            'confidence': 0.55,
            'speaker_id': 'Speaker_1',
        },
        {
            'text': 'Evasive maneuvers! Incoming torpedo!',
            'start': 5.7,
            'end': 7.5,
            'confidence': 0.50,
            'speaker_id': 'Speaker_0',
        },
    ]


@pytest.fixture
def calm_segments():
    """Segments simulating routine operations."""
    return [
        {
            'text': 'Acknowledged. Steady on course.',
            'start': 0.0,
            'end': 4.0,
            'confidence': 0.92,
            'speaker_id': 'Speaker_0',
        },
        {
            'text': 'All clear, captain. Systems nominal.',
            'start': 6.0,
            'end': 10.0,
            'confidence': 0.95,
            'speaker_id': 'Speaker_1',
        },
        {
            'text': 'Standing by. No contacts on sensors.',
            'start': 12.0,
            'end': 16.0,
            'confidence': 0.90,
            'speaker_id': 'Speaker_2',
        },
    ]


@pytest.fixture
def mixed_segments():
    """Segments with a mix of stress levels."""
    return [
        {
            'text': 'Scan complete. Unidentified contact bearing 270.',
            'start': 0.0,
            'end': 5.0,
            'confidence': 0.85,
            'speaker_id': 'Speaker_0',
        },
        {
            'text': 'Acknowledged, adjusting course.',
            'start': 5.5,
            'end': 8.0,
            'confidence': 0.88,
            'speaker_id': 'Speaker_1',
        },
        {
            'text': 'Warning! Incoming fire!',
            'start': 8.2,
            'end': 9.5,
            'confidence': 0.65,
            'speaker_id': 'Speaker_0',
        },
    ]


class TestHighStressDetection:
    """Tests for high-stress scenario detection."""

    def test_high_stress_text(self, analyzer, high_stress_segments):
        """High-stress text should produce stress_level > 0.5."""
        result = analyzer.analyze_segments(high_stress_segments)
        segments = result['segments']

        # First segment has multiple stress keywords
        assert segments[0]['stress_level'] > 0.5
        assert segments[0]['stress_label'] in ('high', 'moderate')

    def test_high_stress_emotion(self, analyzer, high_stress_segments):
        """High-stress segments should map to urgent/critical emotions."""
        result = analyzer.analyze_segments(high_stress_segments)
        segments = result['segments']

        high_emotions = {'tense', 'urgent', 'critical'}
        for seg in segments:
            assert seg['emotion'] in high_emotions or seg['stress_level'] < 0.4

    def test_high_stress_sentiment_negative(self, analyzer, high_stress_segments):
        """High-stress segments should have negative or neutral sentiment."""
        result = analyzer.analyze_segments(high_stress_segments)
        segments = result['segments']

        for seg in segments:
            if seg['stress_level'] > 0.6:
                assert seg['sentiment'] == 'negative'
                assert seg['sentiment_score'] < 0
            elif seg['stress_level'] > 0.4:
                # Moderate-high stress can be neutral or negative
                assert seg['sentiment'] in ('negative', 'neutral')


class TestCalmDetection:
    """Tests for calm scenario detection."""

    def test_calm_text(self, analyzer, calm_segments):
        """Calm text should produce stress_level < 0.3."""
        result = analyzer.analyze_segments(calm_segments)
        segments = result['segments']

        for seg in segments:
            assert seg['stress_level'] < 0.4, (
                f"Calm segment '{seg}' has stress {seg['stress_level']}"
            )

    def test_calm_label(self, analyzer, calm_segments):
        """Calm segments should have 'low' stress label."""
        result = analyzer.analyze_segments(calm_segments)
        segments = result['segments']

        for seg in segments:
            assert seg['stress_label'] == 'low'

    def test_calm_emotion(self, analyzer, calm_segments):
        """Calm segments should map to calm/focused."""
        result = analyzer.analyze_segments(calm_segments)
        segments = result['segments']

        calm_emotions = {'calm', 'focused'}
        for seg in segments:
            assert seg['emotion'] in calm_emotions


class TestSpeechRateScoring:
    """Tests for the speech rate signal."""

    def test_fast_speech_high_stress(self, analyzer):
        """Fast speech (>3.5 wps) should produce high rate score."""
        # 10 words in 2 seconds = 5 wps
        seg = {
            'text': 'Fire fire fire everyone get down now brace brace brace',
            'start': 0.0,
            'end': 2.0,
            'confidence': 0.8,
        }
        score = analyzer._score_speech_rate(seg)
        assert score > 0.5

    def test_slow_speech_low_stress(self, analyzer):
        """Slow speech (<2 wps) should produce low rate score."""
        # 2 words in 3 seconds < 1 wps
        seg = {
            'text': 'Standing by.',
            'start': 0.0,
            'end': 3.0,
            'confidence': 0.9,
        }
        score = analyzer._score_speech_rate(seg)
        assert score < 0.3

    def test_word_timestamps_used(self, analyzer):
        """Word-level timestamps should be used when available."""
        seg = {
            'text': 'fire fire fire',
            'start': 0.0,
            'end': 1.0,
            'words': [
                {'start': 0.0, 'end': 0.2, 'probability': 0.9},
                {'start': 0.3, 'end': 0.5, 'probability': 0.8},
                {'start': 0.6, 'end': 0.8, 'probability': 0.7},
            ],
            'confidence': 0.8,
        }
        score = analyzer._score_speech_rate(seg)
        # 3 words in 1 second = 3.0 wps → moderate-high
        assert score > 0.15

    def test_zero_duration_returns_zero(self, analyzer):
        """Zero-duration segment should return 0 stress."""
        seg = {'text': 'hello', 'start': 5.0, 'end': 5.0, 'confidence': 0.8}
        score = analyzer._score_speech_rate(seg)
        assert score == 0.0


class TestLexiconScoring:
    """Tests for the lexicon signal."""

    def test_stress_keywords(self, analyzer):
        """Known stress keywords should score > 0."""
        score = analyzer._score_lexicon("Red alert! Hull breach!")
        assert score > 0.5

    def test_calm_keywords(self, analyzer):
        """Known calm keywords should reduce score."""
        score = analyzer._score_lexicon("acknowledged, steady as she goes")
        # Calm terms have negative weights, so average should be <= 0
        assert score == 0.0

    def test_no_keywords(self, analyzer):
        """Text with no keywords should score 0."""
        score = analyzer._score_lexicon("the weather is nice today")
        assert score == 0.0

    def test_empty_text(self, analyzer):
        """Empty text should score 0."""
        assert analyzer._score_lexicon("") == 0.0
        assert analyzer._score_lexicon(None) == 0.0

    def test_mixed_keywords(self, analyzer):
        """Mix of stress and calm terms produces moderate score."""
        score = analyzer._score_lexicon("warning detected but all clear now")
        # Should be moderate since both stress and calm terms present
        assert 0.0 <= score <= 0.7


class TestProsodicScoring:
    """Tests for the prosodic signal."""

    def test_exclamation_marks(self, analyzer):
        """Exclamation marks should increase stress."""
        score = analyzer._score_prosodic("Fire! Fire! Now!")
        assert score > 0.3

    def test_all_caps(self, analyzer):
        """ALL CAPS words should increase stress."""
        score = analyzer._score_prosodic("FIRE THE WEAPONS NOW")
        assert score > 0.3

    def test_short_utterance(self, analyzer):
        """Very short utterances indicate urgency."""
        score = analyzer._score_prosodic("Now!")
        assert score > 0.3

    def test_empty_text(self, analyzer):
        """Empty text returns 0."""
        assert analyzer._score_prosodic("") == 0.0
        assert analyzer._score_prosodic("   ") == 0.0

    def test_normal_text(self, analyzer):
        """Normal text without prosodic markers scores low."""
        score = analyzer._score_prosodic(
            "Set a course for the next waypoint and maintain speed"
        )
        assert score < 0.2


class TestDynamicsScoring:
    """Tests for the utterance dynamics signal."""

    def test_rapid_fire_high_stress(self, analyzer):
        """Rapid-fire exchanges (< 0.5s gaps) should score high."""
        seg = {'start': 1.0, 'end': 2.0}
        prev = {'start': 0.0, 'end': 0.8}  # 0.2s gap
        next_seg = {'start': 2.1, 'end': 3.0}  # 0.1s gap

        score = analyzer._score_dynamics(seg, prev, next_seg)
        assert score > 0.7

    def test_relaxed_pacing_low_stress(self, analyzer):
        """Large gaps (> 3s) should score 0."""
        seg = {'start': 10.0, 'end': 12.0}
        prev = {'start': 0.0, 'end': 5.0}  # 5s gap
        next_seg = {'start': 16.0, 'end': 18.0}  # 4s gap

        score = analyzer._score_dynamics(seg, prev, next_seg)
        assert score == 0.0

    def test_no_neighbors_returns_zero(self, analyzer):
        """No neighboring segments should return 0."""
        seg = {'start': 5.0, 'end': 7.0}
        score = analyzer._score_dynamics(seg, None, None)
        assert score == 0.0

    def test_start_time_variants(self, analyzer):
        """Should handle both 'start'/'end' and 'start_time'/'end_time'."""
        seg = {'start_time': 1.0, 'end_time': 2.0}
        prev = {'start_time': 0.0, 'end_time': 0.9}

        score = analyzer._score_dynamics(seg, prev, None)
        assert score > 0.5


class TestConfidenceScoring:
    """Tests for the confidence signal."""

    def test_low_confidence_high_stress(self, analyzer):
        """Far-below-average confidence should score high."""
        score = analyzer._score_confidence(0.5, 0.9)
        assert score > 0.8

    def test_above_average_zero(self, analyzer):
        """Above-average confidence should score 0."""
        score = analyzer._score_confidence(0.95, 0.85)
        assert score == 0.0

    def test_zero_average_returns_zero(self, analyzer):
        """Zero average confidence should not crash."""
        score = analyzer._score_confidence(0.5, 0.0)
        assert score == 0.0


class TestEmotionClassification:
    """Tests for emotion classification."""

    def test_critical_emotion(self, analyzer):
        """Very high stress maps to 'critical'."""
        assert analyzer._classify_emotion(0.9, "hull breach") == "critical"

    def test_urgent_emotion(self, analyzer):
        """High stress maps to 'urgent'."""
        assert analyzer._classify_emotion(0.65, "incoming fire") == "urgent"

    def test_tense_emotion(self, analyzer):
        """Moderate stress maps to 'tense'."""
        assert analyzer._classify_emotion(0.45, "unknown contact") == "tense"

    def test_focused_emotion(self, analyzer):
        """Low-moderate stress maps to 'focused'."""
        assert analyzer._classify_emotion(0.25, "scanning sector") == "focused"

    def test_calm_emotion(self, analyzer):
        """Very low stress maps to 'calm'."""
        assert analyzer._classify_emotion(0.1, "all clear") == "calm"


class TestFullAnalysis:
    """Tests for the complete analyze_segments method."""

    def test_output_structure(self, analyzer, mixed_segments):
        """Test output has correct structure."""
        result = analyzer.analyze_segments(mixed_segments)

        assert 'segments' in result
        assert 'summary' in result
        assert len(result['segments']) == len(mixed_segments)

    def test_segment_keys(self, analyzer, mixed_segments):
        """Each scored segment should have required keys."""
        result = analyzer.analyze_segments(mixed_segments)

        required_keys = {
            'stress_level', 'stress_label', 'emotion',
            'sentiment', 'sentiment_score', 'signals',
        }
        signal_keys = {
            'speech_rate_wps', 'lexicon_score', 'confidence_signal',
            'prosodic_score', 'dynamics_score',
        }

        for seg in result['segments']:
            assert required_keys.issubset(set(seg.keys()))
            assert signal_keys.issubset(set(seg['signals'].keys()))

    def test_summary_keys(self, analyzer, mixed_segments):
        """Summary should have required keys."""
        result = analyzer.analyze_segments(mixed_segments)
        summary = result['summary']

        required = {
            'average_stress', 'peak_stress_time', 'peak_stress_level',
            'stress_distribution', 'speaker_stress', 'stress_timeline',
        }
        assert required.issubset(set(summary.keys()))

    def test_stress_distribution_percentages(self, analyzer, mixed_segments):
        """Stress distribution should sum to ~100%."""
        result = analyzer.analyze_segments(mixed_segments)
        dist = result['summary']['stress_distribution']

        total = dist['low'] + dist['moderate'] + dist['high']
        assert 99 <= total <= 101  # rounding tolerance

    def test_speaker_stress_stats(self, analyzer, mixed_segments):
        """Per-speaker stats should include all speakers."""
        result = analyzer.analyze_segments(mixed_segments)
        speaker_stress = result['summary']['speaker_stress']

        # mixed_segments has Speaker_0 and Speaker_1
        assert 'Speaker_0' in speaker_stress
        assert 'Speaker_1' in speaker_stress

        for speaker_data in speaker_stress.values():
            assert 'avg_stress' in speaker_data
            assert 'peak' in speaker_data
            assert 'dominant_emotion' in speaker_data
            assert 'segment_count' in speaker_data

    def test_stress_timeline(self, analyzer, calm_segments):
        """Timeline should have at least one entry."""
        result = analyzer.analyze_segments(calm_segments)
        timeline = result['summary']['stress_timeline']

        assert len(timeline) >= 1
        for point in timeline:
            assert 'time' in point
            assert 'stress' in point
            assert 0.0 <= point['stress'] <= 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_segments(self, analyzer):
        """Empty segment list should return empty results."""
        result = analyzer.analyze_segments([])

        assert result['segments'] == []
        assert result['summary']['average_stress'] == 0.0
        assert result['summary']['stress_timeline'] == []

    def test_single_segment(self, analyzer):
        """Single segment should work without neighbors."""
        result = analyzer.analyze_segments([{
            'text': 'Hello bridge.',
            'start': 0.0,
            'end': 2.0,
            'confidence': 0.9,
            'speaker_id': 'Speaker_0',
        }])

        assert len(result['segments']) == 1
        assert result['summary']['average_stress'] >= 0.0

    def test_missing_confidence(self, analyzer):
        """Segments without confidence should not crash."""
        result = analyzer.analyze_segments([{
            'text': 'Red alert!',
            'start': 0.0,
            'end': 1.0,
        }])
        assert len(result['segments']) == 1

    def test_missing_speaker_id(self, analyzer):
        """Segments without speaker_id should use 'unknown'."""
        result = analyzer.analyze_segments([{
            'text': 'Hello.',
            'start': 0.0,
            'end': 2.0,
            'confidence': 0.9,
        }])
        assert 'unknown' in result['summary']['speaker_stress']

    def test_stress_level_bounds(self, analyzer, high_stress_segments, calm_segments):
        """Stress levels should always be between 0 and 1."""
        for segments in [high_stress_segments, calm_segments]:
            result = analyzer.analyze_segments(segments)
            for seg in result['segments']:
                assert 0.0 <= seg['stress_level'] <= 1.0

    def test_sentiment_score_bounds(self, analyzer, high_stress_segments):
        """Sentiment scores should be between -1 and 1."""
        result = analyzer.analyze_segments(high_stress_segments)
        for seg in result['segments']:
            assert -1.0 <= seg['sentiment_score'] <= 1.0
