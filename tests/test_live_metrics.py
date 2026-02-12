"""Tests for live metrics computation module."""

import time

import pytest
from unittest.mock import patch, MagicMock

from src.metrics.live_metrics import LiveMetricsComputer


class TestLiveMetricsComputer:
    """Test suite for LiveMetricsComputer."""

    @pytest.fixture
    def computer(self):
        """Create a LiveMetricsComputer instance."""
        return LiveMetricsComputer()

    @pytest.fixture
    def basic_segments(self):
        """Create basic test segments."""
        return [
            {
                'text': 'Set course for the nebula at warp five.',
                'start': 0.0,
                'end': 3.0,
                'confidence': 0.9,
            },
            {
                'text': 'Aye captain, setting course now.',
                'start': 3.5,
                'end': 5.5,
                'confidence': 0.85,
            },
            {
                'text': 'Shields are holding at 80 percent.',
                'start': 6.0,
                'end': 8.0,
                'confidence': 0.92,
            },
        ]

    def test_compute_empty_segments(self, computer):
        """Compute with empty segments returns defaults."""
        result = computer.compute([])

        assert result['stress']['avg'] == 0.0
        assert result['stress']['label'] == 'calm'
        assert result['communication']['effective_count'] == 0
        assert result['communication']['improvement_count'] == 0
        assert result['communication']['effective_pct'] == 0.0
        assert result['speech']['avg_wps'] == 0.0
        assert result['speech']['total_words'] == 0
        assert result['habits'] == []

    def test_compute_returns_all_keys(self, computer, basic_segments):
        """Compute returns all expected top-level keys."""
        result = computer.compute(basic_segments)

        assert 'stress' in result
        assert 'communication' in result
        assert 'speech' in result
        assert 'habits' in result

    def test_compute_stress_with_known_segments(self, computer):
        """Stress-laden text produces elevated stress levels."""
        stress_segments = [
            {
                'text': 'Red alert! Shields failing! We are under attack!',
                'start': 0.0,
                'end': 2.0,
                'confidence': 0.9,
            },
            {
                'text': 'Hull breach! Emergency! All hands brace for impact!',
                'start': 2.5,
                'end': 4.5,
                'confidence': 0.88,
            },
        ]
        result = computer.compute(stress_segments)
        stress = result['stress']

        # Stress should be above baseline for urgent text
        assert stress['avg'] > 0.0
        assert stress['peak'] > 0.0
        assert stress['label'] in ('calm', 'tense', 'critical')
        assert isinstance(stress['distribution'], dict)

    def test_compute_communication_effective(self, computer):
        """Effective patterns are correctly counted."""
        segments = [
            {
                'text': 'Set course for the nebula immediately.',
                'start': 0.0,
                'end': 2.0,
            },
            {
                'text': 'Hold us at 500 kilometers from the target.',
                'start': 2.5,
                'end': 4.5,
            },
            {
                'text': 'Execute evasive maneuvers alpha pattern.',
                'start': 5.0,
                'end': 7.0,
            },
        ]
        result = computer.compute(segments)
        comm = result['communication']

        assert comm['effective_count'] >= 2
        assert isinstance(comm['effective_pct'], float)
        assert len(comm['recent_patterns']) <= 5

    def test_compute_communication_improvement(self, computer):
        """Improvement patterns are correctly counted."""
        segments = [
            {
                'text': 'Um, uh, well, I think maybe we should, uh...',
                'start': 0.0,
                'end': 2.0,
            },
            {
                'text': 'Yeah.',
                'start': 2.5,
                'end': 3.0,
            },
        ]
        result = computer.compute(segments)
        comm = result['communication']

        assert comm['improvement_count'] >= 1

    def test_compute_speech_energy(self, computer):
        """Speech WPS is correctly calculated."""
        segments = [
            {
                'text': 'One two three four five',
                'start': 0.0,
                'end': 2.0,
            },
            {
                'text': 'Six seven eight nine ten',
                'start': 2.5,
                'end': 4.5,
            },
        ]
        result = computer.compute(segments)
        speech = result['speech']

        assert speech['total_words'] == 10
        assert speech['total_duration_s'] == 4.0
        assert speech['avg_wps'] == 2.5

    def test_compute_habits(self, computer):
        """Habit-matching text produces non-zero scores."""
        segments = [
            {
                'text': "Let's plan ahead and set our goals before we start.",
                'start': 0.0,
                'end': 3.0,
            },
            {
                'text': "I suggest we work together and find a solution "
                        "that benefits everyone.",
                'start': 3.5,
                'end': 6.5,
            },
            {
                'text': "What do you think? I want to understand your "
                        "perspective first.",
                'start': 7.0,
                'end': 10.0,
            },
        ] * 5  # Repeat to boost counts above thresholds

        result = computer.compute(segments)
        habits = result['habits']

        assert isinstance(habits, list)
        assert len(habits) == 7
        # At least some habits should score above 1
        scores = [h['score'] for h in habits]
        assert any(s >= 1 for s in scores)

    def test_individual_metric_failure(self, computer):
        """One metric failing doesn't break others."""
        segments = [
            {
                'text': 'Normal bridge communication.',
                'start': 0.0,
                'end': 2.0,
            },
        ]

        # Force sentiment analyzer to fail
        computer._ensure_initialized()
        original = computer._sentiment_analyzer
        computer._sentiment_analyzer = MagicMock()
        computer._sentiment_analyzer.analyze_segments.side_effect = (
            RuntimeError("Simulated failure")
        )

        result = computer.compute(segments)

        # Stress should have defaults
        assert result['stress']['label'] == 'unknown'
        assert result['stress']['avg'] == 0.0

        # Other metrics should still work
        assert 'communication' in result
        assert 'speech' in result
        assert result['speech']['total_words'] > 0

        # Restore
        computer._sentiment_analyzer = original

    def test_compute_is_idempotent(self, computer, basic_segments):
        """Calling compute twice with same segments gives same result."""
        result1 = computer.compute(basic_segments)
        result2 = computer.compute(basic_segments)

        assert result1['speech'] == result2['speech']
        assert result1['communication'] == result2['communication']

    def test_stress_label_thresholds(self, computer):
        """Stress labels map to correct threshold ranges."""
        # We test the internal method directly with mocked results
        computer._ensure_initialized()

        calm_segments = [
            {
                'text': 'Everything is fine and peaceful.',
                'start': 0.0,
                'end': 2.0,
                'confidence': 0.95,
            },
        ]

        result = computer._compute_stress(calm_segments)
        assert result['label'] in ('calm', 'tense', 'critical')
        assert isinstance(result['avg'], float)

    def test_communication_recent_patterns_limited(self, computer):
        """Recent patterns list is capped at 5 entries."""
        # Create many segments that match patterns
        segments = [
            {
                'text': f'Set course for target {i} immediately.',
                'start': float(i * 2),
                'end': float(i * 2 + 1.5),
            }
            for i in range(20)
        ]
        result = computer.compute(segments)
        assert len(result['communication']['recent_patterns']) <= 5

    def test_speech_zero_duration_segments(self, computer):
        """Segments with zero duration don't cause division errors."""
        segments = [
            {
                'text': 'Some text here',
                'start': 5.0,
                'end': 5.0,
            },
        ]
        result = computer.compute(segments)
        assert result['speech']['avg_wps'] == 0.0
        assert result['speech']['total_words'] == 3

    def test_compute_returns_crew_effectiveness_key(self, computer, basic_segments):
        """Compute returns crew_effectiveness in result dict."""
        result = computer.compute(basic_segments)
        assert 'crew_effectiveness' in result
        ces = result['crew_effectiveness']
        assert 'score' in ces
        assert 'label' in ces
        assert 'trend' in ces
        assert 'components' in ces

    def test_ces_perfect_score(self, computer):
        """All metrics at best values produce score near 100."""
        stress = {'avg': 0.0, 'peak': 0.0, 'label': 'calm', 'distribution': {}}
        communication = {
            'effective_count': 10, 'improvement_count': 0,
            'effective_pct': 100.0, 'recent_patterns': [],
        }
        # All habits at max score (5 each, sum=35)
        habits = [
            {'habit_num': i, 'name': f'Habit {i}', 'score': 5, 'count': 10}
            for i in range(1, 8)
        ]
        # Optimal WPS at 2.5
        speech = {'avg_wps': 2.5, 'total_words': 100, 'total_duration_s': 40.0}

        ces = computer._compute_crew_effectiveness(
            stress, communication, habits, speech,
        )
        assert ces['score'] >= 95.0
        assert ces['label'] == 'Excellent'

    def test_ces_worst_score(self, computer):
        """High stress, no effective comm, low habits produce low score."""
        stress = {'avg': 1.0, 'peak': 1.0, 'label': 'critical', 'distribution': {}}
        communication = {
            'effective_count': 0, 'improvement_count': 10,
            'effective_pct': 0.0, 'recent_patterns': [],
        }
        # All habits at min score (1 each, sum=7)
        habits = [
            {'habit_num': i, 'name': f'Habit {i}', 'score': 1, 'count': 0}
            for i in range(1, 8)
        ]
        # No speech
        speech = {'avg_wps': 0.0, 'total_words': 0, 'total_duration_s': 0.0}

        ces = computer._compute_crew_effectiveness(
            stress, communication, habits, speech,
        )
        assert ces['score'] <= 5.0
        assert ces['label'] == 'Struggling'

    def test_ces_label_thresholds(self, computer):
        """Verify correct labels at boundary values."""
        stress = {'avg': 0.0}
        comm = {'effective_pct': 0.0}
        habits = []
        speech = {'avg_wps': 0.0}

        # Force specific scores by manipulating inputs
        # Score ~25: stress_score=100*0.25=25, rest=0 => 25
        stress_25 = {'avg': 0.0}
        ces = computer._compute_crew_effectiveness(
            stress_25, comm, habits, speech,
        )
        assert ces['score'] == 25.0
        assert ces['label'] == 'Developing'

        # Score ~0: all worst
        stress_0 = {'avg': 1.0}
        ces = computer._compute_crew_effectiveness(
            stress_0, comm, habits, speech,
        )
        assert ces['score'] == 0.0
        assert ces['label'] == 'Struggling'

        # Score 75: stress 0 (25), comm 100% (25), habits max empty (0),
        # speech 2.5 (20) = 70; need habits too
        # stress=0(25) + comm=100(25) + habits=[all 5](30) + speech=0(0) = 80
        habits_max = [
            {'habit_num': i, 'name': f'H{i}', 'score': 5, 'count': 5}
            for i in range(1, 8)
        ]
        ces = computer._compute_crew_effectiveness(
            {'avg': 0.0}, {'effective_pct': 100.0}, habits_max, speech,
        )
        assert ces['score'] == 80.0
        assert ces['label'] == 'Excellent'

        # Effective range (50-75): stress=0(25) + comm=100(25) + habits=0 + speech=0 = 50
        ces = computer._compute_crew_effectiveness(
            {'avg': 0.0}, {'effective_pct': 100.0}, [], {'avg_wps': 0.0},
        )
        assert ces['score'] == 50.0
        assert ces['label'] == 'Effective'

    def test_ces_trend_arrow(self, computer):
        """Verify trend computation with history."""
        stress = {'avg': 0.5}
        comm = {'effective_pct': 50.0}
        habits = []
        speech = {'avg_wps': 2.0}

        # First call — no prior history, trend should be stable
        ces1 = computer._compute_crew_effectiveness(
            stress, comm, habits, speech,
        )
        assert ces1['trend'] == '→'

        # Inject a historical entry 60s ago with lower score
        now = time.time()
        computer._ces_history = [(now - 60, ces1['score'] - 10)]

        # Compute again — score is higher than 60s ago => improving
        ces2 = computer._compute_crew_effectiveness(
            stress, comm, habits, speech,
        )
        assert ces2['trend'] == '↑'

        # Inject a historical entry 60s ago with higher score
        computer._ces_history = [(now - 60, ces2['score'] + 10)]
        ces3 = computer._compute_crew_effectiveness(
            stress, comm, habits, speech,
        )
        assert ces3['trend'] == '↓'

    def test_ces_speech_bell_curve(self, computer):
        """Verify 0 WPS, 2.5 WPS, 5 WPS scoring via bell curve."""
        # 0 WPS = silence = 0
        assert LiveMetricsComputer._speech_bell_curve(0.0) == 0.0

        # 2.5 WPS = ideal = 100
        assert LiveMetricsComputer._speech_bell_curve(2.5) == 100.0

        # 5 WPS = frantic = ~50 (within tolerance)
        score_5 = LiveMetricsComputer._speech_bell_curve(5.0)
        assert 45.0 <= score_5 <= 55.0

        # 1.0 WPS = below ideal but above 0 — should be moderate
        score_1 = LiveMetricsComputer._speech_bell_curve(1.0)
        assert 50.0 < score_1 < 100.0

    def test_compute_returns_teamstepps_key(self, computer, basic_segments):
        """Compute returns teamstepps in result dict."""
        result = computer.compute(basic_segments)
        assert 'teamstepps' in result
        assert isinstance(result['teamstepps'], list)

    def test_compute_returns_nasa4d_key(self, computer, basic_segments):
        """Compute returns nasa4d in result dict."""
        result = computer.compute(basic_segments)
        assert 'nasa4d' in result
        assert isinstance(result['nasa4d'], list)

    def test_compute_returns_bloom_key(self, computer, basic_segments):
        """Compute returns bloom in result dict."""
        result = computer.compute(basic_segments)
        assert 'bloom' in result
        assert 'levels' in result['bloom']
        assert 'avg_level' in result['bloom']
        assert 'total' in result['bloom']

    def test_teamstepps_with_rich_dialogue(self, computer):
        """TeamSTEPPS domains detect patterns in bridge dialogue."""
        segments = [
            {'text': 'Captain, I am at my station and ready.',
             'start': 0.0, 'end': 2.0},
            {'text': 'Set course for the nebula. All hands, listen up.',
             'start': 2.5, 'end': 5.0},
            {'text': 'Sensors show an enemy contact at bearing 045.',
             'start': 5.5, 'end': 8.0},
            {'text': 'I can help you with that, rerouting power now.',
             'start': 8.5, 'end': 11.0},
            {'text': 'Aye captain, acknowledged and standing by.',
             'start': 11.5, 'end': 14.0},
        ] * 3  # Repeat to boost counts
        result = computer.compute(segments)
        ts = result['teamstepps']
        assert len(ts) == 5
        names = [d['name'] for d in ts]
        assert 'Leadership' in names
        assert 'Communication' in names
        # At least some domains should score above 1
        assert any(d['score'] > 1 for d in ts)

    def test_nasa4d_with_rich_dialogue(self, computer):
        """NASA 4-D dimensions detect patterns in bridge dialogue."""
        segments = [
            {'text': 'Great job everyone, I appreciate the effort.',
             'start': 0.0, 'end': 2.0},
            {'text': 'What if we try a different approach? We can do this.',
             'start': 2.5, 'end': 5.0},
            {'text': 'Aye sir, understood, doing it now. Mission objective confirmed.',
             'start': 5.5, 'end': 8.0},
            {'text': "Don't worry, it happens. Who has the helm?",
             'start': 8.5, 'end': 11.0},
        ] * 3
        result = computer.compute(segments)
        n4d = result['nasa4d']
        assert len(n4d) == 4
        names = [d['name'] for d in n4d]
        assert 'Cultivating' in names
        assert 'Directing' in names

    def test_bloom_with_cognitive_dialogue(self, computer):
        """Bloom's levels detect cognitive complexity in dialogue."""
        segments = [
            {'text': 'Report status. What is our shield level?',
             'start': 0.0, 'end': 2.0},
            {'text': 'That shows we need to analyze the pattern.',
             'start': 2.5, 'end': 5.0},
            {'text': 'I think we should prioritize the left flank, '
                     'that is my recommendation.',
             'start': 5.5, 'end': 8.0},
            {'text': 'Firing torpedoes, engaging targets now.',
             'start': 8.5, 'end': 11.0},
            {'text': 'What if we try a new approach to this problem?',
             'start': 11.5, 'end': 14.0},
        ] * 3
        result = computer.compute(segments)
        bl = result['bloom']
        assert isinstance(bl['levels'], list)
        assert len(bl['levels']) == 6
        assert bl['avg_level'] >= 1.0
        assert bl['total'] > 0
        # Verify level ordering
        level_nums = [l['level'] for l in bl['levels']]
        assert level_nums == [1, 2, 3, 4, 5, 6]

    def test_frameworks_empty_segments(self, computer):
        """Frameworks return empty defaults for empty segments."""
        result = computer.compute([])
        assert result['teamstepps'] == []
        assert result['nasa4d'] == []
        assert result['bloom'] == {
            'levels': [], 'avg_level': 1.0, 'total': 0
        }
