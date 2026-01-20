"""
LLM integration for mission analysis and report generation.

Includes hallucination prevention utilities for reducing
fabricated content in LLM-generated debriefs and stories.
"""

from src.llm.ollama_client import OllamaClient

# Import hallucination prevention if available
try:
    from src.llm.hallucination_prevention import (
        ConstrainedContextBuilder,
        OutputValidator,
        ValidationIssue,
        clean_hallucinations,
        ANTI_HALLUCINATION_PARAMS,
        STORY_PARAMS,
    )
    __all__ = [
        'OllamaClient',
        'ConstrainedContextBuilder',
        'OutputValidator',
        'ValidationIssue',
        'clean_hallucinations',
        'ANTI_HALLUCINATION_PARAMS',
        'STORY_PARAMS',
    ]
except ImportError:
    __all__ = ['OllamaClient']
