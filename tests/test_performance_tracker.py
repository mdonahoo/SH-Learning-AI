"""Tests for performance tracker module."""

import time

import pytest

from src.metrics.performance_tracker import DependencyCall, PerformanceTracker


class TestDependencyCall:
    """Test suite for DependencyCall dataclass."""

    def test_basic_creation(self):
        """Test creating a DependencyCall with required fields."""
        call = DependencyCall(
            name='test_call',
            dependency_type='LLM',
            start_time=100.0,
            end_time=105.0,
            duration_seconds=5.0,
            success=True,
        )
        assert call.name == 'test_call'
        assert call.dependency_type == 'LLM'
        assert call.duration_seconds == 5.0
        assert call.success is True
        assert call.error is None
        assert call.metadata == {}

    def test_creation_with_error(self):
        """Test creating a DependencyCall with error info."""
        call = DependencyCall(
            name='failed_call',
            dependency_type='ML_MODEL',
            start_time=100.0,
            end_time=102.0,
            duration_seconds=2.0,
            success=False,
            error='Connection refused',
        )
        assert call.success is False
        assert call.error == 'Connection refused'

    def test_creation_with_metadata(self):
        """Test creating a DependencyCall with metadata."""
        meta = {'model': 'qwen2.5:14b', 'prompt_tokens': 500}
        call = DependencyCall(
            name='llm_call',
            dependency_type='LLM',
            start_time=100.0,
            end_time=110.0,
            duration_seconds=10.0,
            success=True,
            metadata=meta,
        )
        assert call.metadata['model'] == 'qwen2.5:14b'
        assert call.metadata['prompt_tokens'] == 500


class TestPerformanceTrackerTrackDependency:
    """Test suite for track_dependency context manager."""

    def test_successful_call(self):
        """Test tracking a successful dependency call."""
        tracker = PerformanceTracker()
        with tracker.track_dependency('test_op', 'LLM') as meta:
            meta['model'] = 'test-model'

        assert len(tracker.calls) == 1
        call = tracker.calls[0]
        assert call.name == 'test_op'
        assert call.dependency_type == 'LLM'
        assert call.success is True
        assert call.error is None
        assert call.metadata['model'] == 'test-model'
        assert call.duration_seconds >= 0

    def test_failed_call(self):
        """Test tracking a dependency call that raises an exception."""
        tracker = PerformanceTracker()
        with pytest.raises(ValueError, match='test error'):
            with tracker.track_dependency('bad_op', 'ML_MODEL') as meta:
                meta['stage'] = 'init'
                raise ValueError('test error')

        assert len(tracker.calls) == 1
        call = tracker.calls[0]
        assert call.success is False
        assert call.error == 'test error'
        assert call.metadata['stage'] == 'init'

    def test_timing_is_recorded(self):
        """Test that start/end times and duration are recorded."""
        tracker = PerformanceTracker()
        with tracker.track_dependency('timed_op', 'LLM'):
            time.sleep(0.05)

        call = tracker.calls[0]
        assert call.start_time > 0
        assert call.end_time >= call.start_time
        assert call.duration_seconds >= 0.04  # Allow small timing variance

    def test_initial_metadata(self):
        """Test passing initial metadata to track_dependency."""
        tracker = PerformanceTracker()
        with tracker.track_dependency('op', 'LLM', metadata={'init': True}) as meta:
            meta['extra'] = 'value'

        call = tracker.calls[0]
        assert call.metadata['init'] is True
        assert call.metadata['extra'] == 'value'

    def test_initial_metadata_not_mutated(self):
        """Test that original metadata dict is not mutated."""
        tracker = PerformanceTracker()
        original = {'key': 'original'}
        with tracker.track_dependency('op', 'LLM', metadata=original) as meta:
            meta['added'] = 'new'

        assert 'added' not in original
        assert tracker.calls[0].metadata['added'] == 'new'


class TestPerformanceTrackerRecordLLMMetrics:
    """Test suite for record_llm_metrics static method."""

    def test_standard_ollama_response(self):
        """Test extracting metrics from a typical Ollama response."""
        meta: dict = {}
        ollama_response = {
            'prompt_eval_count': 500,
            'eval_count': 200,
            'total_duration': 10_000_000_000,  # 10s in nanoseconds
            'load_duration': 1_000_000_000,     # 1s
            'eval_duration': 5_000_000_000,     # 5s
            'prompt_eval_duration': 2_000_000_000,  # 2s
        }

        PerformanceTracker.record_llm_metrics(meta, ollama_response, model='qwen2.5:14b')

        assert meta['model'] == 'qwen2.5:14b'
        assert meta['prompt_tokens'] == 500
        assert meta['completion_tokens'] == 200
        assert meta['total_tokens'] == 700
        assert meta['load_duration_seconds'] == 1.0
        assert meta['eval_duration_seconds'] == 5.0
        assert meta['prompt_eval_duration_seconds'] == 2.0
        assert meta['tokens_per_second'] == 40.0  # 200 / 5

    def test_zero_eval_duration(self):
        """Test with zero eval_duration returns 0 tokens_per_second."""
        meta: dict = {}
        ollama_response = {
            'prompt_eval_count': 100,
            'eval_count': 50,
            'total_duration': 1_000_000_000,
            'eval_duration': 0,
        }

        PerformanceTracker.record_llm_metrics(meta, ollama_response)

        assert meta['tokens_per_second'] == 0.0
        assert meta['total_tokens'] == 150

    def test_missing_fields(self):
        """Test with empty/partial Ollama response."""
        meta: dict = {}
        PerformanceTracker.record_llm_metrics(meta, {})

        assert meta['prompt_tokens'] == 0
        assert meta['completion_tokens'] == 0
        assert meta['total_tokens'] == 0
        assert meta['tokens_per_second'] == 0.0
        assert meta['load_duration_seconds'] == 0.0

    def test_no_model(self):
        """Test without providing model name."""
        meta: dict = {}
        PerformanceTracker.record_llm_metrics(meta, {'eval_count': 10})

        assert 'model' not in meta
        assert meta['completion_tokens'] == 10


class TestPerformanceTrackerGetSummary:
    """Test suite for get_summary aggregation."""

    def test_empty_tracker(self):
        """Test summary from a tracker with no calls."""
        tracker = PerformanceTracker()
        summary = tracker.get_summary()

        assert summary['total_duration_seconds'] >= 0
        assert summary['dependencies'] == []
        assert summary['summary']['total_llm_tokens'] == 0
        assert summary['summary']['total_llm_duration_seconds'] == 0
        assert summary['summary']['total_ml_duration_seconds'] == 0
        assert summary['summary']['dependency_count'] == 0
        assert summary['summary']['failed_dependencies'] == 0

    def test_multiple_dependencies(self):
        """Test summary aggregation across multiple calls."""
        tracker = PerformanceTracker()

        # Simulate ML call
        with tracker.track_dependency('whisper', 'ML_MODEL') as meta:
            meta['model'] = 'large-v3'

        # Simulate LLM call with token metrics
        with tracker.track_dependency('narrative', 'LLM') as meta:
            meta['total_tokens'] = 1000

        # Simulate another LLM call
        with tracker.track_dependency('story', 'LLM') as meta:
            meta['total_tokens'] = 2000

        summary = tracker.get_summary()

        assert summary['summary']['dependency_count'] == 3
        assert summary['summary']['total_llm_tokens'] == 3000
        assert summary['summary']['failed_dependencies'] == 0
        assert len(summary['dependencies']) == 3
        assert summary['dependencies'][0]['name'] == 'whisper'
        assert summary['dependencies'][0]['type'] == 'ML_MODEL'

    def test_failed_dependency_counted(self):
        """Test that failed dependencies are counted in summary."""
        tracker = PerformanceTracker()

        with tracker.track_dependency('good_call', 'LLM'):
            pass

        try:
            with tracker.track_dependency('bad_call', 'LLM'):
                raise RuntimeError('boom')
        except RuntimeError:
            pass

        summary = tracker.get_summary()

        assert summary['summary']['dependency_count'] == 2
        assert summary['summary']['failed_dependencies'] == 1
        assert summary['dependencies'][1]['error'] == 'boom'

    def test_summary_is_json_serializable(self):
        """Test that summary can be serialized to JSON."""
        import json
        tracker = PerformanceTracker()

        with tracker.track_dependency('test', 'LLM') as meta:
            meta['count'] = 42

        summary = tracker.get_summary()
        # Should not raise
        serialized = json.dumps(summary)
        assert '"test"' in serialized
