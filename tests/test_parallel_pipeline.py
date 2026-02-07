"""Tests for parallel analysis pipeline."""

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.hardware.parallel_pipeline import (
    ParallelAnalysisPipeline,
    ParallelProgressTracker,
    MetricStepConfig,
)


class TestParallelProgressTracker:
    """Tests for thread-safe progress tracking."""

    def test_initial_progress_is_zero(self):
        """Test that tracker starts at 0."""
        tracker = ParallelProgressTracker()
        assert tracker.current == 0

    def test_monotonic_increase(self):
        """Test that progress only increases, never decreases."""
        tracker = ParallelProgressTracker()
        tracker.update("step1", "First", 50)
        assert tracker.current == 50

        # Attempt to decrease — should be ignored
        tracker.update("step2", "Second", 30)
        assert tracker.current == 50

        # Higher value — should be accepted
        tracker.update("step3", "Third", 75)
        assert tracker.current == 75

    def test_callback_called_on_increase(self):
        """Test that callback is called when progress increases."""
        callback = MagicMock()
        tracker = ParallelProgressTracker(callback=callback)

        tracker.update("step1", "Testing", 50)
        callback.assert_called_once_with("step1", "Testing", 50)

    def test_callback_not_called_on_decrease(self):
        """Test that callback is NOT called when progress would decrease."""
        callback = MagicMock()
        tracker = ParallelProgressTracker(callback=callback)

        tracker.update("step1", "First", 50)
        callback.reset_mock()

        tracker.update("step2", "Second", 30)
        callback.assert_not_called()

    def test_callback_error_handled(self):
        """Test that callback errors don't crash the tracker."""
        callback = MagicMock(side_effect=RuntimeError("callback broke"))
        tracker = ParallelProgressTracker(callback=callback)

        # Should not raise
        tracker.update("step1", "Testing", 50)
        assert tracker.current == 50

    def test_set_floor(self):
        """Test setting a progress floor without callback."""
        callback = MagicMock()
        tracker = ParallelProgressTracker(callback=callback)

        tracker.set_floor(60)
        assert tracker.current == 60
        callback.assert_not_called()

        # Update below floor is ignored
        tracker.update("step1", "Below floor", 40)
        assert tracker.current == 60
        callback.assert_not_called()

    def test_thread_safety(self):
        """Test that concurrent updates maintain monotonicity."""
        tracker = ParallelProgressTracker()
        errors = []

        def update_thread(thread_id: int):
            try:
                for i in range(100):
                    # Each thread tries to set progress in its range
                    value = thread_id * 10 + (i % 10)
                    tracker.update(f"t{thread_id}", f"Thread {thread_id}", value)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_thread, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final value should be the max sent (9 * 10 + 9 = 99)
        assert tracker.current == 99


class TestMetricStepConfig:
    """Tests for MetricStepConfig dataclass."""

    def test_default_values(self):
        """Test default field values."""
        step = MetricStepConfig(
            name="test",
            weight=1.0,
            func=lambda: None,
            result_key="test_result",
        )
        assert step.args == ()
        assert step.kwargs == {}
        assert step.available is True


class TestParallelAnalysisPipeline:
    """Tests for the parallel analysis pipeline."""

    def test_initialization_defaults(self):
        """Test pipeline initialization with default env vars."""
        with patch.dict('os.environ', {}, clear=False):
            pipeline = ParallelAnalysisPipeline()
            assert pipeline._parallel_cpu_enabled is True
            assert pipeline._parallel_llm_enabled is True
            assert pipeline._cpu_workers >= 1

    def test_initialization_disabled(self):
        """Test pipeline with parallel disabled via env vars."""
        with patch.dict('os.environ', {
            'ENABLE_PARALLEL_CPU': 'false',
            'ENABLE_PARALLEL_LLM': 'false',
        }):
            pipeline = ParallelAnalysisPipeline()
            assert pipeline._parallel_cpu_enabled is False
            assert pipeline._parallel_llm_enabled is False

    def test_custom_worker_count(self):
        """Test custom CPU worker count from env var."""
        with patch.dict('os.environ', {'PARALLEL_CPU_WORKERS': '3'}):
            pipeline = ParallelAnalysisPipeline()
            assert pipeline._cpu_workers == 3

    def test_run_metrics_parallel_empty(self):
        """Test parallel metrics with no steps."""
        pipeline = ParallelAnalysisPipeline()
        results = pipeline.run_metrics_parallel([])
        assert results == {}

    def test_run_metrics_parallel_all_unavailable(self):
        """Test parallel metrics when all steps are unavailable."""
        pipeline = ParallelAnalysisPipeline()
        steps = [
            MetricStepConfig(
                name="test",
                weight=1.0,
                func=lambda: "result",
                result_key="test",
                available=False,
            ),
        ]
        results = pipeline.run_metrics_parallel(steps)
        assert results == {}

    def test_run_metrics_parallel_success(self):
        """Test parallel metrics with successful steps."""
        pipeline = ParallelAnalysisPipeline()

        def step_a():
            time.sleep(0.01)
            return {"quality": "high"}

        def step_b():
            time.sleep(0.01)
            return {"confidence": 0.95}

        def step_c():
            time.sleep(0.01)
            return {"habits_score": 4.5}

        steps = [
            MetricStepConfig(name="quality", weight=7, func=step_a, result_key="comm_quality"),
            MetricStepConfig(name="confidence", weight=5, func=step_b, result_key="conf_dist"),
            MetricStepConfig(name="habits", weight=9, func=step_c, result_key="seven_habits"),
        ]

        results = pipeline.run_metrics_parallel(steps)

        assert results['comm_quality'] == {"quality": "high"}
        assert results['conf_dist'] == {"confidence": 0.95}
        assert results['seven_habits'] == {"habits_score": 4.5}
        assert '_step_timings' in results
        assert 'quality' in results['_step_timings']
        assert 'confidence' in results['_step_timings']
        assert 'habits' in results['_step_timings']

    def test_run_metrics_parallel_isolation(self):
        """Test that one step failing doesn't affect others."""
        pipeline = ParallelAnalysisPipeline()

        def good_step():
            return {"status": "ok"}

        def bad_step():
            raise ValueError("Something broke")

        steps = [
            MetricStepConfig(name="good", weight=5, func=good_step, result_key="good_result"),
            MetricStepConfig(name="bad", weight=5, func=bad_step, result_key="bad_result"),
        ]

        results = pipeline.run_metrics_parallel(steps)

        assert results['good_result'] == {"status": "ok"}
        assert results['bad_result'] is None

    def test_run_metrics_parallel_progress(self):
        """Test that progress updates are emitted during parallel execution."""
        progress_updates = []

        def track_progress(step_id, label, pct):
            progress_updates.append((step_id, label, pct))

        pipeline = ParallelAnalysisPipeline(progress_callback=track_progress)

        steps = [
            MetricStepConfig(name="step1", weight=5, func=lambda: "r1", result_key="r1"),
            MetricStepConfig(name="step2", weight=5, func=lambda: "r2", result_key="r2"),
        ]

        pipeline.run_metrics_parallel(steps, base_progress=60, end_progress=95)

        assert len(progress_updates) > 0
        # Progress should be monotonically increasing
        pcts = [p[2] for p in progress_updates]
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1], f"Progress decreased: {pcts}"

    def test_run_metrics_sequential_fallback(self):
        """Test sequential fallback when parallel is disabled."""
        with patch.dict('os.environ', {'ENABLE_PARALLEL_CPU': 'false'}):
            pipeline = ParallelAnalysisPipeline()

        def step_func():
            return {"result": True}

        steps = [
            MetricStepConfig(name="only_step", weight=5, func=step_func, result_key="out"),
        ]

        results = pipeline.run_metrics_parallel(steps)
        assert results['out'] == {"result": True}

    def test_run_metrics_sequential_single_step(self):
        """Test that a single step runs sequentially even with parallel enabled."""
        pipeline = ParallelAnalysisPipeline()

        call_count = 0

        def step_func():
            nonlocal call_count
            call_count += 1
            return "done"

        steps = [
            MetricStepConfig(name="single", weight=1, func=step_func, result_key="out"),
        ]

        results = pipeline.run_metrics_parallel(steps)
        assert results['out'] == "done"
        assert call_count == 1

    def test_run_llm_parallel_success(self):
        """Test parallel LLM execution."""
        pipeline = ParallelAnalysisPipeline()

        def narrative_func():
            time.sleep(0.01)
            return {"text": "Team analysis..."}

        def story_func():
            time.sleep(0.01)
            return {"text": "Once upon a time..."}

        steps = [
            MetricStepConfig(name="narrative", weight=5, func=narrative_func, result_key="narrative"),
            MetricStepConfig(name="story", weight=5, func=story_func, result_key="story"),
        ]

        results = pipeline.run_llm_parallel(steps)

        assert results['narrative'] == {"text": "Team analysis..."}
        assert results['story'] == {"text": "Once upon a time..."}

    def test_run_llm_parallel_one_fails(self):
        """Test LLM parallel when one step fails."""
        pipeline = ParallelAnalysisPipeline()

        def good_func():
            return {"text": "Success"}

        def bad_func():
            raise ConnectionError("Ollama down")

        steps = [
            MetricStepConfig(name="good_llm", weight=5, func=good_func, result_key="good"),
            MetricStepConfig(name="bad_llm", weight=5, func=bad_func, result_key="bad"),
        ]

        results = pipeline.run_llm_parallel(steps)

        assert results['good'] == {"text": "Success"}
        assert results['bad'] is None

    def test_run_llm_sequential_fallback(self):
        """Test sequential LLM fallback when parallel disabled."""
        with patch.dict('os.environ', {'ENABLE_PARALLEL_LLM': 'false'}):
            pipeline = ParallelAnalysisPipeline()

        def llm_func():
            return {"text": "Generated"}

        steps = [
            MetricStepConfig(name="narrative", weight=5, func=llm_func, result_key="narrative"),
            MetricStepConfig(name="story", weight=5, func=llm_func, result_key="story"),
        ]

        results = pipeline.run_llm_parallel(steps)
        assert results['narrative'] == {"text": "Generated"}
        assert results['story'] == {"text": "Generated"}

    def test_run_llm_empty_steps(self):
        """Test LLM parallel with no steps."""
        pipeline = ParallelAnalysisPipeline()
        results = pipeline.run_llm_parallel([])
        assert results == {}

    def test_parallel_faster_than_sequential(self):
        """Test that parallel execution is faster than sequential for slow steps."""
        pipeline = ParallelAnalysisPipeline()

        def slow_step():
            time.sleep(0.05)
            return "done"

        steps = [
            MetricStepConfig(name=f"step{i}", weight=1, func=slow_step, result_key=f"r{i}")
            for i in range(4)
        ]

        start = time.time()
        results = pipeline.run_metrics_parallel(steps)
        parallel_time = time.time() - start

        # All results should be present
        for i in range(4):
            assert results[f'r{i}'] == "done"

        # Parallel should be significantly faster than 4 * 0.05 = 0.2s
        # Allow some overhead but should be well under sequential time
        assert parallel_time < 0.18, f"Parallel took {parallel_time:.3f}s, expected < 0.18s"

    def test_step_with_args_and_kwargs(self):
        """Test metric step receives correct args and kwargs."""
        pipeline = ParallelAnalysisPipeline()

        def step_with_args(a, b, extra=None):
            return {"sum": a + b, "extra": extra}

        steps = [
            MetricStepConfig(
                name="args_test",
                weight=1,
                func=step_with_args,
                args=(3, 7),
                kwargs={"extra": "bonus"},
                result_key="result",
            ),
        ]

        results = pipeline.run_metrics_parallel(steps)
        assert results['result'] == {"sum": 10, "extra": "bonus"}

    def test_training_recommendations_after_metrics(self):
        """Test that training recommendations can access metric results.

        In the real pipeline, training recommendations depend on results
        from quality, confidence, habits, and learning steps. This test
        verifies the pattern where parallel metrics complete first,
        then training runs with those results.
        """
        pipeline = ParallelAnalysisPipeline()

        # Phase 1: Run independent metrics in parallel
        metric_steps = [
            MetricStepConfig(name="quality", weight=5, func=lambda: {"score": 85}, result_key="quality"),
            MetricStepConfig(name="confidence", weight=5, func=lambda: {"avg": 0.9}, result_key="confidence"),
        ]
        metric_results = pipeline.run_metrics_parallel(metric_steps)

        # Phase 2: Training recommendations uses metric results
        def training_func(quality, confidence):
            return {
                "recommendations": f"Score={quality['score']}, Conf={confidence['avg']}"
            }

        training_steps = [
            MetricStepConfig(
                name="training",
                weight=5,
                func=training_func,
                args=(metric_results['quality'], metric_results['confidence']),
                result_key="training",
            ),
        ]
        training_results = pipeline.run_metrics_parallel(training_steps)

        assert "Score=85" in training_results['training']['recommendations']
        assert "Conf=0.9" in training_results['training']['recommendations']
