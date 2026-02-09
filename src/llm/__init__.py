"""
LLM integration for mission analysis and report generation.

Provides a unified LLM client (OpenAI-compatible protocol) and
higher-level wrappers for mission analysis, narrative generation,
and hallucination prevention.
"""

from src.llm.llm_client import LLMClient, LLMResponse, get_default_client
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
        'LLMClient',
        'LLMResponse',
        'get_default_client',
        'OllamaClient',
        'ConstrainedContextBuilder',
        'OutputValidator',
        'ValidationIssue',
        'clean_hallucinations',
        'ANTI_HALLUCINATION_PARAMS',
        'STORY_PARAMS',
    ]
except ImportError:
    __all__ = ['LLMClient', 'LLMResponse', 'get_default_client', 'OllamaClient']
