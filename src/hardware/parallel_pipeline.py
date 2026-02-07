"""
Parallel analysis pipeline for GPU-aware execution.

Orchestrates the analysis in three phases:
- Phase A: GPU/ML steps (transcription, diarization)
- Phase B: CPU metrics in parallel (quality, scorecards, confidence, etc.)
- Phase C: LLM calls in parallel (narrative + story)

Uses ThreadPoolExecutor for parallelism since the workloads are either
I/O-bound (LLM HTTP calls) or release the GIL (PyTorch, numpy).
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ParallelProgressTracker:
    """
    Thread-safe progress tracker with monotonically increasing values.

    Ensures progress only moves forward, even when parallel steps
    complete out of order.
    """

    def __init__(
        self,
        callback: Optional[Callable[[str, str, int], None]] = None
    ) -> None:
        """
        Initialize progress tracker.

        Args:
            callback: Progress callback function(step_id, label, progress_pct)
        """
        self._lock = threading.Lock()
        self._current_progress: int = 0
        self._callback = callback

    @property
    def current(self) -> int:
        """Get current progress percentage."""
        with self._lock:
            return self._current_progress

    def update(self, step_id: str, label: str, progress: int) -> None:
        """
        Update progress if the new value exceeds the current value.

        Args:
            step_id: Identifier for the current step
            label: Human-readable label for the step
            progress: New progress percentage (0-100)
        """
        with self._lock:
            if progress > self._current_progress:
                self._current_progress = progress
                if self._callback:
                    try:
                        self._callback(step_id, label, progress)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

    def set_floor(self, progress: int) -> None:
        """
        Set a minimum progress floor without emitting a callback.

        Args:
            progress: Minimum progress value
        """
        with self._lock:
            if progress > self._current_progress:
                self._current_progress = progress


@dataclass
class MetricStepConfig:
    """Configuration for a single metric analysis step."""

    name: str
    weight: float
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result_key: str = ""
    available: bool = True


class ParallelAnalysisPipeline:
    """
    Orchestrates analysis steps with configurable parallelism.

    Phases:
    - Phase A (0-60%): GPU/ML steps — sequential or multi-GPU parallel
    - Phase B (60-95%): CPU metric analysis — ThreadPoolExecutor
    - Phase C (95-100%): LLM generation — ThreadPoolExecutor
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, int], None]] = None,
    ) -> None:
        """
        Initialize parallel pipeline.

        Args:
            progress_callback: Optional callback(step_id, label, progress_pct)
        """
        self._tracker = ParallelProgressTracker(callback=progress_callback)

        # Read configuration from environment
        self._parallel_cpu_enabled = (
            os.getenv('ENABLE_PARALLEL_CPU', 'true').lower() == 'true'
        )
        self._parallel_llm_enabled = (
            os.getenv('ENABLE_PARALLEL_LLM', 'true').lower() == 'true'
        )

        # Worker count for CPU metrics
        env_workers = int(os.getenv('PARALLEL_CPU_WORKERS', '0'))
        if env_workers > 0:
            self._cpu_workers = env_workers
        else:
            cpu_count = os.cpu_count() or 4
            self._cpu_workers = min(6, max(1, cpu_count - 1))

        logger.info(
            f"ParallelAnalysisPipeline: cpu_parallel={self._parallel_cpu_enabled}, "
            f"llm_parallel={self._parallel_llm_enabled}, "
            f"cpu_workers={self._cpu_workers}"
        )

    @property
    def tracker(self) -> ParallelProgressTracker:
        """Get the progress tracker."""
        return self._tracker

    def run_metrics_parallel(
        self,
        steps: List[MetricStepConfig],
        base_progress: int = 60,
        end_progress: int = 95,
    ) -> Dict[str, Any]:
        """
        Run CPU metric analysis steps in parallel.

        Each step is wrapped in try/except so one failure doesn't
        affect others. Failed steps return None.

        Args:
            steps: List of MetricStepConfig describing each analysis step
            base_progress: Starting progress percentage for this phase
            end_progress: Ending progress percentage for this phase

        Returns:
            Dict mapping result_key to analysis result (or None on failure)
        """
        # Filter to only available steps
        active_steps = [s for s in steps if s.available]

        if not active_steps:
            return {}

        if not self._parallel_cpu_enabled or len(active_steps) <= 1:
            return self._run_metrics_sequential(
                active_steps, base_progress, end_progress
            )

        results: Dict[str, Any] = {}
        total_weight = sum(s.weight for s in active_steps)
        completed_weight = 0.0
        step_timings: Dict[str, float] = {}

        self._tracker.update(
            "metrics_parallel",
            f"Analyzing metrics (0/{len(active_steps)} complete)",
            base_progress,
        )

        with ThreadPoolExecutor(
            max_workers=self._cpu_workers,
            thread_name_prefix="metric"
        ) as executor:
            # Submit all steps
            future_to_step: Dict[Future, MetricStepConfig] = {}
            for step in active_steps:
                future = executor.submit(
                    self._run_single_metric, step
                )
                future_to_step[future] = step

            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                completed_count += 1
                completed_weight += step.weight

                try:
                    result, elapsed = future.result()
                    results[step.result_key] = result
                    step_timings[step.name] = elapsed

                    if result is not None:
                        logger.info(
                            f"Metric step '{step.name}' completed in {elapsed:.2f}s"
                        )
                    else:
                        logger.warning(
                            f"Metric step '{step.name}' returned None ({elapsed:.2f}s)"
                        )

                except Exception as e:
                    logger.error(
                        f"Metric step '{step.name}' failed: {e}",
                        exc_info=True,
                    )
                    results[step.result_key] = None
                    step_timings[step.name] = 0.0

                # Update progress
                progress = base_progress + int(
                    (end_progress - base_progress)
                    * (completed_weight / total_weight)
                )
                self._tracker.update(
                    "metrics_parallel",
                    f"Analyzing metrics ({completed_count}/{len(active_steps)} complete)",
                    progress,
                )

        results['_step_timings'] = step_timings
        return results

    def _run_single_metric(
        self, step: MetricStepConfig
    ) -> Tuple[Any, float]:
        """
        Run a single metric step with timing and error isolation.

        Args:
            step: Metric step configuration

        Returns:
            Tuple of (result, elapsed_seconds)
        """
        start = time.time()
        try:
            result = step.func(*step.args, **step.kwargs)
            return result, time.time() - start
        except Exception as e:
            logger.error(f"Metric '{step.name}' error: {e}", exc_info=True)
            return None, time.time() - start

    def _run_metrics_sequential(
        self,
        steps: List[MetricStepConfig],
        base_progress: int,
        end_progress: int,
    ) -> Dict[str, Any]:
        """
        Run metric steps sequentially (fallback mode).

        Args:
            steps: List of MetricStepConfig
            base_progress: Starting progress percentage
            end_progress: Ending progress percentage

        Returns:
            Dict mapping result_key to analysis result
        """
        results: Dict[str, Any] = {}
        step_timings: Dict[str, float] = {}
        total_weight = sum(s.weight for s in steps)
        completed_weight = 0.0

        for step in steps:
            self._tracker.update(
                step.name,
                f"Analyzing {step.name}...",
                base_progress + int(
                    (end_progress - base_progress)
                    * (completed_weight / total_weight)
                )
                if total_weight > 0 else base_progress,
            )

            start = time.time()
            try:
                results[step.result_key] = step.func(
                    *step.args, **step.kwargs
                )
            except Exception as e:
                logger.error(f"Metric '{step.name}' error: {e}", exc_info=True)
                results[step.result_key] = None

            elapsed = time.time() - start
            step_timings[step.name] = elapsed
            completed_weight += step.weight

        results['_step_timings'] = step_timings
        return results

    def run_llm_parallel(
        self,
        steps: List[MetricStepConfig],
        base_progress: int = 95,
        end_progress: int = 100,
    ) -> Dict[str, Any]:
        """
        Run LLM generation steps in parallel.

        These are I/O-bound HTTP calls to Ollama, so threading
        eliminates scheduling overhead between the two calls.

        Args:
            steps: List of MetricStepConfig for LLM generation steps
            base_progress: Starting progress percentage
            end_progress: Ending progress percentage

        Returns:
            Dict mapping result_key to LLM result (or None on failure)
        """
        active_steps = [s for s in steps if s.available]

        if not active_steps:
            return {}

        if not self._parallel_llm_enabled or len(active_steps) <= 1:
            return self._run_metrics_sequential(
                active_steps, base_progress, end_progress
            )

        results: Dict[str, Any] = {}
        step_timings: Dict[str, float] = {}

        self._tracker.update(
            "llm_parallel",
            "Generating narrative and story...",
            base_progress,
        )

        with ThreadPoolExecutor(
            max_workers=len(active_steps),
            thread_name_prefix="llm"
        ) as executor:
            future_to_step: Dict[Future, MetricStepConfig] = {}
            for step in active_steps:
                future = executor.submit(
                    self._run_single_metric, step
                )
                future_to_step[future] = step

            completed_count = 0
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                completed_count += 1

                try:
                    result, elapsed = future.result()
                    results[step.result_key] = result
                    step_timings[step.name] = elapsed

                    if result is not None:
                        logger.info(
                            f"LLM step '{step.name}' completed in {elapsed:.2f}s"
                        )
                    else:
                        logger.info(
                            f"LLM step '{step.name}' returned None ({elapsed:.2f}s)"
                        )

                except Exception as e:
                    logger.error(
                        f"LLM step '{step.name}' failed: {e}",
                        exc_info=True,
                    )
                    results[step.result_key] = None
                    step_timings[step.name] = 0.0

                progress = base_progress + int(
                    (end_progress - base_progress)
                    * (completed_count / len(active_steps))
                )
                self._tracker.update(
                    "llm_parallel",
                    f"Generating narrative and story ({completed_count}/{len(active_steps)})...",
                    progress,
                )

        results['_step_timings'] = step_timings
        return results
