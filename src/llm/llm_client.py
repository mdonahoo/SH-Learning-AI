"""
Unified LLM client using the OpenAI-compatible API.

Provides a single abstraction for communicating with any LLM backend
that implements the OpenAI /v1/chat/completions protocol (Ollama, vLLM,
OpenAI, etc.). Replaces scattered raw HTTP calls with a typed, tested
interface supporting both sync and async generation.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def _resolve_base_url() -> str:
    """
    Resolve the LLM base URL from environment variables.

    Falls back through: LLM_BASE_URL -> OLLAMA_HOST + "/v1" -> default.

    Returns:
        Base URL string ending with /v1
    """
    explicit = os.getenv('LLM_BASE_URL')
    if explicit:
        return explicit.rstrip('/')

    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434').rstrip('/')
    return f"{ollama_host}/v1"


def _resolve_model() -> str:
    """
    Resolve the LLM model name from environment variables.

    Returns:
        Model name string
    """
    return os.getenv('LLM_MODEL') or os.getenv('OLLAMA_MODEL', 'llama3.2')


def _resolve_timeout() -> int:
    """
    Resolve the LLM timeout from environment variables.

    Returns:
        Timeout in seconds
    """
    llm_timeout = os.getenv('LLM_TIMEOUT')
    if llm_timeout:
        return int(llm_timeout)
    return int(os.getenv('OLLAMA_TIMEOUT', '120'))


def _resolve_api_key() -> str:
    """
    Resolve the API key from environment variables.

    Ollama ignores the key but the OpenAI SDK requires a non-empty value.

    Returns:
        API key string
    """
    return os.getenv('LLM_API_KEY', 'ollama')


def _resolve_max_retries() -> int:
    """
    Resolve the max retries count from environment variables.

    The OpenAI SDK handles exponential backoff automatically for
    retryable errors (429, 500, 502, 503, 504).

    Returns:
        Max retries count
    """
    return int(os.getenv('LLM_MAX_RETRIES', '2'))


@dataclass
class LLMResponse:
    """
    Structured response from an LLM generation call.

    Attributes:
        text: Generated text content
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        model: Model name that generated the response
        tokens_per_second: Generation speed (completion tokens / wall-clock time)
    """

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    tokens_per_second: float = 0.0


class LLMClient:
    """
    Unified LLM client using the OpenAI-compatible chat completions API.

    Supports both synchronous and asynchronous generation, including
    streaming. Works with any backend that implements the OpenAI protocol
    (Ollama, vLLM, OpenAI, etc.).

    Attributes:
        base_url: API base URL (e.g. http://localhost:11434/v1)
        model: Default model name
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        api_key: Optional[str] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize LLM client.

        Args:
            base_url: API base URL (falls back to LLM_BASE_URL / OLLAMA_HOST env vars)
            model: Model name (falls back to LLM_MODEL / OLLAMA_MODEL env vars)
            timeout: Request timeout in seconds (falls back to LLM_TIMEOUT / OLLAMA_TIMEOUT)
            api_key: API key (falls back to LLM_API_KEY env var, default 'ollama')
            max_retries: Max retries for transient errors (falls back to LLM_MAX_RETRIES, default 2)
        """
        self.base_url = base_url or _resolve_base_url()
        self.model = model or _resolve_model()
        self.timeout = timeout or _resolve_timeout()
        self._api_key = api_key or _resolve_api_key()
        self.max_retries = max_retries if max_retries is not None else _resolve_max_retries()

        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

        logger.info(
            f"LLMClient initialized: base_url={self.base_url}, "
            f"model={self.model}, timeout={self.timeout}s, "
            f"max_retries={self.max_retries}"
        )

    def _get_sync_client(self) -> OpenAI:
        """Get or create synchronous OpenAI client (lazy init)."""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                base_url=self.base_url,
                api_key=self._api_key,
                timeout=float(self.timeout),
                max_retries=self.max_retries,
            )
        return self._sync_client

    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create asynchronous OpenAI client (lazy init)."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self._api_key,
                timeout=float(self.timeout),
                max_retries=self.max_retries,
            )
        return self._async_client

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build chat messages list from prompt and optional system message.

        Args:
            prompt: User prompt text
            system: Optional system prompt text

        Returns:
            List of message dicts for the chat completions API
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_extra_body(
        self,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build extra_body dict for non-standard parameters.

        Both Ollama and vLLM accept these via extra_body.

        Args:
            top_k: Top-k sampling limit
            repeat_penalty: Repetition penalty

        Returns:
            Dict of extra parameters, or None if empty
        """
        extra: Dict[str, Any] = {}
        if top_k is not None:
            extra["top_k"] = top_k
        if repeat_penalty is not None:
            extra["repeat_penalty"] = repeat_penalty
        if extra:
            return extra
        return None

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Optional[LLMResponse]:
        """
        Generate text synchronously.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit (via extra_body)
            repeat_penalty: Repetition penalty (via extra_body)
            stop: Stop sequences

        Returns:
            LLMResponse with generated text and metrics, or None on failure
        """
        try:
            client = self._get_sync_client()
            messages = self._build_messages(prompt, system)
            extra_body = self._build_extra_body(top_k, repeat_penalty)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop is not None:
                kwargs["stop"] = stop
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            logger.info(f"LLM generate: model={self.model}, temp={temperature}")

            start_time = time.monotonic()
            response = client.chat.completions.create(**kwargs)
            elapsed = time.monotonic() - start_time

            text = response.choices[0].message.content or ""
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            tps = (
                round(completion_tokens / elapsed, 2)
                if elapsed > 0 and completion_tokens > 0
                else 0.0
            )

            logger.info(f"LLM generated {len(text)} chars in {elapsed:.1f}s ({tps} tok/s)")

            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=response.model or self.model,
                tokens_per_second=tps,
            )

        except Exception as e:
            logger.error(f"LLM generate failed: {e}")
            return None

    async def agenerate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Optional[LLMResponse]:
        """
        Generate text asynchronously.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit (via extra_body)
            repeat_penalty: Repetition penalty (via extra_body)
            stop: Stop sequences

        Returns:
            LLMResponse with generated text and metrics, or None on failure
        """
        try:
            client = self._get_async_client()
            messages = self._build_messages(prompt, system)
            extra_body = self._build_extra_body(top_k, repeat_penalty)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop is not None:
                kwargs["stop"] = stop
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            logger.info(f"LLM agenerate: model={self.model}, temp={temperature}")

            start_time = time.monotonic()
            response = await client.chat.completions.create(**kwargs)
            elapsed = time.monotonic() - start_time

            text = response.choices[0].message.content or ""
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            tps = (
                round(completion_tokens / elapsed, 2)
                if elapsed > 0 and completion_tokens > 0
                else 0.0
            )

            logger.info(f"LLM agenerated {len(text)} chars in {elapsed:.1f}s ({tps} tok/s)")

            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=response.model or self.model,
                tokens_per_second=tps,
            )

        except Exception as e:
            logger.error(f"LLM agenerate failed: {e}")
            return None

    def generate_streaming(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Generate text with synchronous streaming.

        Yields text chunks as they arrive from the server.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            repeat_penalty: Repetition penalty
            stop: Stop sequences

        Yields:
            Text chunks as strings
        """
        try:
            client = self._get_sync_client()
            messages = self._build_messages(prompt, system)
            extra_body = self._build_extra_body(top_k, repeat_penalty)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop is not None:
                kwargs["stop"] = stop
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            logger.info(f"LLM streaming: model={self.model}, temp={temperature}")

            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")

    async def agenerate_streaming(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text with asynchronous streaming.

        Yields text chunks as they arrive from the server.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling limit
            repeat_penalty: Repetition penalty
            stop: Stop sequences

        Yields:
            Text chunks as strings
        """
        try:
            client = self._get_async_client()
            messages = self._build_messages(prompt, system)
            extra_body = self._build_extra_body(top_k, repeat_penalty)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop is not None:
                kwargs["stop"] = stop
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            logger.info(f"LLM async streaming: model={self.model}, temp={temperature}")

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM async streaming failed: {e}")

    def check_available(self) -> bool:
        """
        Check if the LLM backend is accessible.

        Returns:
            True if backend responds to a model list request
        """
        try:
            client = self._get_sync_client()
            client.models.list()
            return True
        except Exception as e:
            logger.debug(f"LLM backend not available: {e}")
            return False

    async def acheck_available(self) -> bool:
        """
        Check if the LLM backend is accessible (async).

        Returns:
            True if backend responds to a model list request
        """
        try:
            client = self._get_async_client()
            await client.models.list()
            return True
        except Exception as e:
            logger.debug(f"LLM backend not available: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available models on the backend.

        Returns:
            List of model ID strings
        """
        try:
            client = self._get_sync_client()
            response = client.models.list()
            return [m.id for m in response.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def alist_models(self) -> List[str]:
        """
        List available models on the backend (async).

        Returns:
            List of model ID strings
        """
        try:
            client = self._get_async_client()
            response = await client.models.list()
            return [m.id for m in response.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def close(self) -> None:
        """Close synchronous client and release resources."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close asynchronous client and release resources."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None


_default_client: Optional[LLMClient] = None


def get_default_client(**overrides: Any) -> LLMClient:
    """
    Get or create a shared LLMClient instance.

    On first call, creates a client using env var defaults.
    Subsequent calls return the same instance. Pass overrides
    to force a new instance (e.g., different timeout).

    Args:
        **overrides: Keyword arguments forwarded to LLMClient().
            When provided, a one-off (non-cached) client is returned.

    Returns:
        Shared or one-off LLMClient instance
    """
    global _default_client
    if overrides:
        return LLMClient(**overrides)
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
