"""
Hardware detection and parallel analysis pipeline.

Provides GPU-aware execution strategies for the audio analysis pipeline.
"""

from src.hardware.detector import HardwareDetector, HardwareProfile, GPUInfo
from src.hardware.parallel_pipeline import (
    ParallelAnalysisPipeline,
    ParallelProgressTracker,
    MetricStepConfig,
)

__all__ = [
    'HardwareDetector',
    'HardwareProfile',
    'GPUInfo',
    'ParallelAnalysisPipeline',
    'ParallelProgressTracker',
    'MetricStepConfig',
]
