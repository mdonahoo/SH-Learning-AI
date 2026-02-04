"""
Performance telemetry for tracking external dependency calls.

Provides structured timing and token metrics for all external calls
(LLM inference, ML model execution, etc.) during the audio analysis pipeline.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DependencyCall:
    """
    Record of a single external dependency call.

    Attributes:
        name: Human-readable identifier (e.g. 'whisper_transcription')
        dependency_type: Category string (e.g. 'LLM', 'ML_MODEL')
        start_time: Epoch timestamp when call began
        end_time: Epoch timestamp when call finished
        duration_seconds: Wall-clock duration
        success: Whether the call completed without error
        error: Error message if call failed
        metadata: Caller-populated dict with extra context
    """

    name: str
    dependency_type: str
    start_time: float
    end_time: float
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Tracks external dependency calls with timing and token metrics.

    Usage::

        tracker = PerformanceTracker()
        with tracker.track_dependency('ollama_narrative', 'LLM') as meta:
            result = call_ollama(prompt)
            tracker.record_llm_metrics(meta, result, model='qwen2.5:14b')

        summary = tracker.get_summary()

    Attributes:
        calls: Ordered list of recorded DependencyCall instances
    """

    def __init__(self) -> None:
        """Initialize with empty call list and start timestamp."""
        self.calls: List[DependencyCall] = []
        self._tracker_start: float = time.time()

    @contextmanager
    def track_dependency(
        self,
        name: str,
        dependency_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager that records timing and success/failure of a dependency call.

        Yields a mutable dict that the caller can populate with extra metadata
        (e.g. token counts, model name). The dict contents are stored in the
        resulting DependencyCall.metadata.

        Args:
            name: Identifier for this call (e.g. 'whisper_transcription')
            dependency_type: Category (e.g. 'LLM', 'ML_MODEL')
            metadata: Optional initial metadata to include

        Yields:
            Mutable dict for the caller to populate with extra context
        """
        call_meta: Dict[str, Any] = dict(metadata) if metadata else {}
        start = time.time()
        success = True
        error_msg: Optional[str] = None

        try:
            yield call_meta
        except Exception as exc:
            success = False
            error_msg = str(exc)
            raise
        finally:
            end = time.time()
            duration = round(end - start, 4)

            call = DependencyCall(
                name=name,
                dependency_type=dependency_type,
                start_time=start,
                end_time=end,
                duration_seconds=duration,
                success=success,
                error=error_msg,
                metadata=call_meta,
            )
            self.calls.append(call)
            logger.debug(
                f"Dependency '{name}' ({dependency_type}): "
                f"{duration:.2f}s success={success}"
            )

    @staticmethod
    def record_llm_metrics(
        call_metadata: Dict[str, Any],
        ollama_response: Dict[str, Any],
        model: Optional[str] = None
    ) -> None:
        """
        Extract Ollama token metrics into call_metadata.

        Ollama returns durations in nanoseconds. This method converts them to
        seconds and computes tokens_per_second where possible.

        Args:
            call_metadata: Mutable dict (from track_dependency yield) to populate
            ollama_response: Raw JSON dict from Ollama /api/generate
            model: Model name to record
        """
        if model:
            call_metadata['model'] = model

        prompt_tokens = ollama_response.get('prompt_eval_count', 0)
        completion_tokens = ollama_response.get('eval_count', 0)
        call_metadata['prompt_tokens'] = prompt_tokens
        call_metadata['completion_tokens'] = completion_tokens
        call_metadata['total_tokens'] = prompt_tokens + completion_tokens

        # Ollama durations are in nanoseconds
        total_duration_ns = ollama_response.get('total_duration', 0)
        load_duration_ns = ollama_response.get('load_duration', 0)
        eval_duration_ns = ollama_response.get('eval_duration', 0)
        prompt_eval_duration_ns = ollama_response.get('prompt_eval_duration', 0)

        call_metadata['total_duration_ns'] = total_duration_ns
        call_metadata['load_duration_seconds'] = round(load_duration_ns / 1e9, 4)
        call_metadata['eval_duration_seconds'] = round(eval_duration_ns / 1e9, 4)
        call_metadata['prompt_eval_duration_seconds'] = round(
            prompt_eval_duration_ns / 1e9, 4
        )

        # Compute tokens per second from eval_duration
        if eval_duration_ns > 0 and completion_tokens > 0:
            call_metadata['tokens_per_second'] = round(
                completion_tokens / (eval_duration_ns / 1e9), 2
            )
        else:
            call_metadata['tokens_per_second'] = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """
        Build a JSON-serializable summary of all tracked dependency calls.

        Returns:
            Dict with keys:
                - total_duration_seconds: Wall-clock time from tracker creation
                - dependencies: List of per-call dicts
                - summary: Aggregated token counts, durations, and counts
        """
        total_duration = round(time.time() - self._tracker_start, 2)

        dependencies: List[Dict[str, Any]] = []
        for call in self.calls:
            entry: Dict[str, Any] = {
                'name': call.name,
                'type': call.dependency_type,
                'duration_seconds': call.duration_seconds,
                'success': call.success,
            }
            if call.error:
                entry['error'] = call.error
            if call.metadata:
                entry['metadata'] = call.metadata
            dependencies.append(entry)

        # Aggregate summary
        total_llm_tokens = 0
        total_llm_duration = 0.0
        total_ml_duration = 0.0
        failed_count = 0

        for call in self.calls:
            if not call.success:
                failed_count += 1
            if call.dependency_type == 'LLM':
                total_llm_tokens += call.metadata.get('total_tokens', 0)
                total_llm_duration += call.duration_seconds
            elif call.dependency_type == 'ML_MODEL':
                total_ml_duration += call.duration_seconds

        summary = {
            'total_llm_tokens': total_llm_tokens,
            'total_llm_duration_seconds': round(total_llm_duration, 2),
            'total_ml_duration_seconds': round(total_ml_duration, 2),
            'dependency_count': len(self.calls),
            'failed_dependencies': failed_count,
        }

        return {
            'total_duration_seconds': total_duration,
            'dependencies': dependencies,
            'summary': summary,
        }
