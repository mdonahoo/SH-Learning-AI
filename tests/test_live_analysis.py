"""Tests for live GM analysis module."""

import json
import time

import pytest
from unittest.mock import MagicMock, patch

from src.llm.live_analysis import LiveGMAnalyzer, GM_SYSTEM_PROMPT


class TestLiveGMAnalyzer:
    """Test suite for LiveGMAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer with short interval for testing."""
        return LiveGMAnalyzer(min_interval=1, max_tokens=200, window_size=5)

    @pytest.fixture
    def sample_segments(self):
        """Create sample transcript segments."""
        return [
            {'text': 'How do I raise the shields?', 'start': 10.0, 'end': 12.0},
            {'text': 'I dont know where the button is.', 'start': 13.0, 'end': 15.0},
            {'text': 'Captain, what should we do?', 'start': 16.0, 'end': 18.0},
            {'text': 'Set course for the nebula.', 'start': 20.0, 'end': 22.0},
            {'text': 'Aye captain, course set.', 'start': 23.0, 'end': 25.0},
        ]

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics dict."""
        return {
            'stress': {'avg': 0.25, 'label': 'calm'},
            'communication': {'effective_count': 3, 'improvement_count': 1},
            'speech': {'avg_wps': 2.5},
        }

    def test_should_analyze_respects_interval(self, analyzer):
        """Should_analyze returns True initially, False after setting time."""
        assert analyzer.should_analyze() is True

        # Simulate a recent analysis
        analyzer._last_analysis_time = time.monotonic()
        assert analyzer.should_analyze() is False

    def test_should_analyze_after_cooldown(self, analyzer):
        """Should_analyze returns True after interval expires."""
        analyzer._last_analysis_time = time.monotonic() - 2.0
        assert analyzer.should_analyze() is True

    def test_analyze_returns_none_without_client(self, analyzer):
        """Analyze returns None when LLM client is unavailable."""
        analyzer._client_available = False
        result = analyzer.analyze([{'text': 'hello', 'start': 0, 'end': 1}])
        assert result is None

    def test_analyze_returns_none_for_empty_segments(self, analyzer):
        """Analyze returns None for empty segment list."""
        analyzer._client_available = True
        analyzer._client = MagicMock()
        result = analyzer.analyze([])
        assert result is None

    def test_analyze_returns_none_when_throttled(self, analyzer):
        """Analyze returns None when cooldown hasn't expired."""
        analyzer._client_available = True
        analyzer._client = MagicMock()
        analyzer._last_analysis_time = time.monotonic()

        result = analyzer.analyze(
            [{'text': 'test', 'start': 0, 'end': 1}]
        )
        assert result is None

    @patch('src.llm.live_analysis.LiveGMAnalyzer._ensure_client')
    def test_analyze_calls_llm_with_correct_params(
        self, mock_ensure, analyzer, sample_segments, sample_metrics
    ):
        """Analyze builds prompt and calls LLM correctly."""
        mock_ensure.return_value = True
        analyzer._client_available = True

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            'needs_help': True,
            'urgency': 'medium',
            'insights': ['Crew confused about shields'],
            'suggestion': 'Show them the shield controls',
        })
        mock_response.tokens_per_second = 45.0
        mock_response.model = 'qwen2.5:14b'

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        analyzer._client = mock_client

        result = analyzer.analyze(sample_segments, sample_metrics)

        assert result is not None
        assert result['needs_help'] is True
        assert result['urgency'] == 'medium'
        assert len(result['insights']) == 1
        assert 'shields' in result['insights'][0].lower()
        assert result['generation_time'] >= 0
        assert result['tokens_per_second'] == 45.0

        # Verify LLM was called with system prompt
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs['system'] == GM_SYSTEM_PROMPT
        assert call_kwargs.kwargs['temperature'] == 0.3

    @patch('src.llm.live_analysis.LiveGMAnalyzer._ensure_client')
    def test_analyze_no_help_needed(
        self, mock_ensure, analyzer, sample_segments
    ):
        """Analyze handles 'no help needed' response."""
        mock_ensure.return_value = True
        analyzer._client_available = True

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            'needs_help': False,
            'insights': [],
            'suggestion': '',
        })
        mock_response.tokens_per_second = 50.0
        mock_response.model = 'qwen2.5:14b'

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        analyzer._client = mock_client

        result = analyzer.analyze(sample_segments)
        assert result is not None
        assert result['needs_help'] is False
        assert result['insights'] == []

    def test_parse_response_valid_json(self, analyzer):
        """Parse correctly handles valid JSON response."""
        text = json.dumps({
            'needs_help': True,
            'urgency': 'high',
            'insights': ['Crew stuck', 'Need objective help'],
            'suggestion': 'Announce next objective',
        })
        result = analyzer._parse_response(text)
        assert result['needs_help'] is True
        assert result['urgency'] == 'high'
        assert len(result['insights']) == 2

    def test_parse_response_strips_markdown_fences(self, analyzer):
        """Parse handles JSON wrapped in markdown code fences."""
        text = '```json\n{"needs_help": false, "insights": [], "suggestion": ""}\n```'
        result = analyzer._parse_response(text)
        assert result['needs_help'] is False

    def test_parse_response_fallback_on_invalid_json(self, analyzer):
        """Parse falls back gracefully on malformed JSON."""
        text = 'The crew seems to be struggling with navigation.'
        result = analyzer._parse_response(text)
        assert result['needs_help'] is True
        assert result['urgency'] == 'low'
        assert len(result['insights']) == 1
        assert 'navigation' in result['insights'][0].lower()

    def test_window_size_limits_segments(self, analyzer):
        """Only the last N segments are included in the prompt."""
        analyzer._client_available = True
        analyzer._window_size = 3

        mock_response = MagicMock()
        mock_response.text = '{"needs_help": false, "insights": [], "suggestion": ""}'
        mock_response.tokens_per_second = 40.0
        mock_response.model = 'test'

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        analyzer._client = mock_client

        # Create 10 segments
        segments = [
            {'text': f'Segment {i}', 'start': float(i), 'end': float(i + 1)}
            for i in range(10)
        ]

        analyzer.analyze(segments)

        # Check the prompt only contains the last 3 segments
        call_args = mock_client.generate.call_args
        prompt = call_args.kwargs['prompt']
        assert 'Segment 7' in prompt
        assert 'Segment 8' in prompt
        assert 'Segment 9' in prompt
        assert 'Segment 0' not in prompt

    def test_analysis_count_increments(self, analyzer):
        """Analysis count increments on successful analysis."""
        analyzer._client_available = True

        mock_response = MagicMock()
        mock_response.text = '{"needs_help": false, "insights": [], "suggestion": ""}'
        mock_response.tokens_per_second = 40.0
        mock_response.model = 'test'

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        analyzer._client = mock_client

        assert analyzer.analysis_count == 0

        analyzer.analyze([{'text': 'test', 'start': 0, 'end': 1}])
        assert analyzer.analysis_count == 1

        # Reset cooldown for second call
        analyzer._last_analysis_time = 0
        analyzer.analyze([{'text': 'test2', 'start': 2, 'end': 3}])
        assert analyzer.analysis_count == 2

    def test_last_result_property(self, analyzer):
        """Last result is stored after analysis."""
        assert analyzer.last_result is None

        analyzer._client_available = True
        mock_response = MagicMock()
        mock_response.text = '{"needs_help": true, "urgency": "low", "insights": ["test"], "suggestion": "help"}'
        mock_response.tokens_per_second = 30.0
        mock_response.model = 'test'

        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        analyzer._client = mock_client

        analyzer.analyze([{'text': 'help me', 'start': 0, 'end': 1}])
        assert analyzer.last_result is not None
        assert analyzer.last_result['needs_help'] is True

    @patch('src.llm.live_analysis.LiveGMAnalyzer._ensure_client')
    def test_analyze_handles_llm_exception(
        self, mock_ensure, analyzer
    ):
        """Analyze returns None when LLM raises an exception."""
        mock_ensure.return_value = True
        analyzer._client_available = True

        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("LLM timeout")
        analyzer._client = mock_client

        result = analyzer.analyze(
            [{'text': 'test', 'start': 0, 'end': 1}]
        )
        assert result is None
