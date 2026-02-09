"""Tests for the unified LLM client."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from src.llm.llm_client import (
    LLMClient,
    LLMResponse,
    _resolve_base_url,
    _resolve_model,
    _resolve_timeout,
    _resolve_api_key,
    _resolve_max_retries,
    get_default_client,
)
import src.llm.llm_client as llm_client_module


# ============================================================================
# TestLLMResponse
# ============================================================================


class TestLLMResponse:
    """Test suite for LLMResponse dataclass."""

    def test_default_fields(self):
        """Test default values for all fields."""
        resp = LLMResponse()
        assert resp.text == ""
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0
        assert resp.model == ""
        assert resp.tokens_per_second == 0.0

    def test_custom_fields(self):
        """Test LLMResponse with custom values."""
        resp = LLMResponse(
            text="Hello world",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="llama3.2",
            tokens_per_second=12.5,
        )
        assert resp.text == "Hello world"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.total_tokens == 15
        assert resp.model == "llama3.2"
        assert resp.tokens_per_second == 12.5


# ============================================================================
# TestLLMClientInit — env var resolution and fallback chain
# ============================================================================


class TestLLMClientInit:
    """Test suite for LLMClient initialization and env var resolution."""

    def test_explicit_params(self):
        """Test that explicit params override env vars."""
        client = LLMClient(
            base_url="http://custom:8000/v1",
            model="custom-model",
            timeout=60,
            api_key="custom-key",
        )
        assert client.base_url == "http://custom:8000/v1"
        assert client.model == "custom-model"
        assert client.timeout == 60

    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://llm-host:9000/v1',
        'LLM_MODEL': 'my-model',
        'LLM_TIMEOUT': '300',
        'LLM_API_KEY': 'my-key',
    }, clear=False)
    def test_llm_env_vars(self):
        """Test LLM_* env vars are used when no explicit params."""
        assert _resolve_base_url() == "http://llm-host:9000/v1"
        assert _resolve_model() == "my-model"
        assert _resolve_timeout() == 300
        assert _resolve_api_key() == "my-key"

    @patch.dict(os.environ, {
        'OLLAMA_HOST': 'http://ollama-host:11434',
        'OLLAMA_MODEL': 'qwen2.5:14b',
        'OLLAMA_TIMEOUT': '600',
    }, clear=False)
    def test_ollama_fallback_env_vars(self):
        """Test OLLAMA_* env vars are used as fallback."""
        # Remove LLM_* vars if present
        for key in ['LLM_BASE_URL', 'LLM_MODEL', 'LLM_TIMEOUT', 'LLM_API_KEY']:
            os.environ.pop(key, None)

        assert _resolve_base_url() == "http://ollama-host:11434/v1"
        assert _resolve_model() == "qwen2.5:14b"
        assert _resolve_timeout() == 600
        assert _resolve_api_key() == "ollama"

    @patch.dict(os.environ, {}, clear=True)
    def test_default_values(self):
        """Test defaults when no env vars are set."""
        assert _resolve_base_url() == "http://localhost:11434/v1"
        assert _resolve_model() == "llama3.2"
        assert _resolve_timeout() == 120
        assert _resolve_api_key() == "ollama"

    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://example.com/v1/',
    }, clear=False)
    def test_trailing_slash_stripped(self):
        """Test that trailing slashes are stripped from base_url."""
        for key in ['OLLAMA_HOST']:
            os.environ.pop(key, None)
        assert _resolve_base_url() == "http://example.com/v1"

    def test_lazy_init_clients(self):
        """Test that sync and async clients are not created until needed."""
        client = LLMClient(base_url="http://test/v1", model="test")
        assert client._sync_client is None
        assert client._async_client is None


# ============================================================================
# TestLLMClientGenerate — synchronous generation
# ============================================================================


class TestLLMClientGenerate:
    """Test suite for LLMClient.generate()."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return LLMClient(
            base_url="http://test:11434/v1",
            model="test-model",
            timeout=30,
        )

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_success(self, mock_openai_cls, client):
        """Test successful synchronous generation."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "test-model"

        mock_openai.chat.completions.create.return_value = mock_response

        result = client.generate("Test prompt")

        assert result is not None
        assert result.text == "Test response"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.model == "test-model"

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_with_system_prompt(self, mock_openai_cls, client):
        """Test generation with system prompt."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=3)
        mock_response.model = "test-model"
        mock_openai.chat.completions.create.return_value = mock_response

        client.generate("User prompt", system="System prompt")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        messages = call_kwargs['messages']
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System prompt"}
        assert messages[1] == {"role": "user", "content": "User prompt"}

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_with_extra_body_params(self, mock_openai_cls, client):
        """Test that top_k and repeat_penalty pass via extra_body."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=3)
        mock_response.model = "test-model"
        mock_openai.chat.completions.create.return_value = mock_response

        client.generate("Prompt", top_k=40, repeat_penalty=1.1)

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs['extra_body'] == {"top_k": 40, "repeat_penalty": 1.1}

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_with_stop_sequences(self, mock_openai_cls, client):
        """Test that stop sequences are passed correctly."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=3)
        mock_response.model = "test-model"
        mock_openai.chat.completions.create.return_value = mock_response

        client.generate("Prompt", stop=["\n", "."])

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs['stop'] == ["\n", "."]

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_failure_returns_none(self, mock_openai_cls, client):
        """Test that exceptions return None."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai
        mock_openai.chat.completions.create.side_effect = Exception("Connection refused")

        result = client.generate("Test prompt")
        assert result is None

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_no_extra_body_when_empty(self, mock_openai_cls, client):
        """Test that extra_body is not set when top_k and repeat_penalty are None."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=3)
        mock_response.model = "test-model"
        mock_openai.chat.completions.create.return_value = mock_response

        client.generate("Prompt")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert 'extra_body' not in call_kwargs


# ============================================================================
# TestLLMClientAsync — async generation
# ============================================================================


class TestLLMClientAsync:
    """Test suite for LLMClient async methods."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return LLMClient(
            base_url="http://test:11434/v1",
            model="test-model",
            timeout=30,
        )

    @pytest.mark.asyncio
    @patch('src.llm.llm_client.AsyncOpenAI')
    async def test_agenerate_success(self, mock_async_cls, client):
        """Test successful async generation."""
        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async

        mock_choice = Mock()
        mock_choice.message.content = "Async response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=8, completion_tokens=4)
        mock_response.model = "test-model"

        mock_async.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.agenerate("Test prompt")

        assert result is not None
        assert result.text == "Async response"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4

    @pytest.mark.asyncio
    @patch('src.llm.llm_client.AsyncOpenAI')
    async def test_agenerate_failure_returns_none(self, mock_async_cls, client):
        """Test that async exceptions return None."""
        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async
        mock_async.chat.completions.create = AsyncMock(
            side_effect=Exception("Timeout")
        )

        result = await client.agenerate("Test prompt")
        assert result is None


# ============================================================================
# TestLLMClientAvailability — check_available, list_models
# ============================================================================


class TestLLMClientAvailability:
    """Test suite for availability checks and model listing."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return LLMClient(
            base_url="http://test:11434/v1",
            model="test-model",
            timeout=5,
        )

    @patch('src.llm.llm_client.OpenAI')
    def test_check_available_success(self, mock_openai_cls, client):
        """Test check_available returns True when backend responds."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai
        mock_openai.models.list.return_value = Mock()

        assert client.check_available() is True

    @patch('src.llm.llm_client.OpenAI')
    def test_check_available_failure(self, mock_openai_cls, client):
        """Test check_available returns False on connection error."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai
        mock_openai.models.list.side_effect = Exception("Connection refused")

        assert client.check_available() is False

    @patch('src.llm.llm_client.OpenAI')
    def test_list_models_success(self, mock_openai_cls, client):
        """Test list_models returns model IDs."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_model1 = Mock()
        mock_model1.id = "llama3.2"
        mock_model2 = Mock()
        mock_model2.id = "qwen2.5:14b"
        mock_openai.models.list.return_value = Mock(data=[mock_model1, mock_model2])

        models = client.list_models()
        assert models == ["llama3.2", "qwen2.5:14b"]

    @patch('src.llm.llm_client.OpenAI')
    def test_list_models_failure(self, mock_openai_cls, client):
        """Test list_models returns empty list on error."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai
        mock_openai.models.list.side_effect = Exception("Error")

        models = client.list_models()
        assert models == []

    @pytest.mark.asyncio
    @patch('src.llm.llm_client.AsyncOpenAI')
    async def test_acheck_available_success(self, mock_async_cls, client):
        """Test async check_available returns True."""
        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async
        mock_async.models.list = AsyncMock(return_value=Mock())

        assert await client.acheck_available() is True

    @pytest.mark.asyncio
    @patch('src.llm.llm_client.AsyncOpenAI')
    async def test_acheck_available_failure(self, mock_async_cls, client):
        """Test async check_available returns False on error."""
        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async
        mock_async.models.list = AsyncMock(side_effect=Exception("Timeout"))

        assert await client.acheck_available() is False

    @pytest.mark.asyncio
    @patch('src.llm.llm_client.AsyncOpenAI')
    async def test_alist_models_success(self, mock_async_cls, client):
        """Test async list_models returns model IDs."""
        mock_async = MagicMock()
        mock_async_cls.return_value = mock_async

        mock_model = Mock()
        mock_model.id = "llama3.2"
        mock_async.models.list = AsyncMock(return_value=Mock(data=[mock_model]))

        models = await client.alist_models()
        assert models == ["llama3.2"]


# ============================================================================
# TestLLMClientStreaming — streaming generation
# ============================================================================


class TestLLMClientStreaming:
    """Test suite for streaming generation."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return LLMClient(
            base_url="http://test:11434/v1",
            model="test-model",
            timeout=30,
        )

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_streaming(self, mock_openai_cls, client):
        """Test synchronous streaming yields chunks."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        # Create mock stream chunks
        chunk1 = Mock()
        chunk1.choices = [Mock()]
        chunk1.choices[0].delta.content = "Hello "
        chunk2 = Mock()
        chunk2.choices = [Mock()]
        chunk2.choices[0].delta.content = "world"
        chunk3 = Mock()
        chunk3.choices = []

        mock_openai.chat.completions.create.return_value = [chunk1, chunk2, chunk3]

        chunks = list(client.generate_streaming("Test prompt"))
        assert chunks == ["Hello ", "world"]

    @patch('src.llm.llm_client.OpenAI')
    def test_generate_streaming_failure(self, mock_openai_cls, client):
        """Test streaming handles errors gracefully."""
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai
        mock_openai.chat.completions.create.side_effect = Exception("Error")

        chunks = list(client.generate_streaming("Test prompt"))
        assert chunks == []


# ============================================================================
# TestLLMClientCleanup — close / aclose
# ============================================================================


class TestLLMClientCleanup:
    """Test suite for client cleanup."""

    def test_close_sync(self):
        """Test close() cleans up sync client."""
        client = LLMClient(base_url="http://test/v1", model="test")
        mock_sync = MagicMock()
        client._sync_client = mock_sync

        client.close()

        mock_sync.close.assert_called_once()
        assert client._sync_client is None

    @pytest.mark.asyncio
    async def test_aclose_async(self):
        """Test aclose() cleans up async client."""
        client = LLMClient(base_url="http://test/v1", model="test")
        mock_async = MagicMock()
        mock_async.close = AsyncMock()
        client._async_client = mock_async

        await client.aclose()

        mock_async.close.assert_awaited_once()
        assert client._async_client is None

    def test_close_noop_when_no_client(self):
        """Test close() does nothing when no client exists."""
        client = LLMClient(base_url="http://test/v1", model="test")
        client.close()  # Should not raise


# ============================================================================
# TestLLMClientRetries — max_retries parameter
# ============================================================================


class TestLLMClientRetries:
    """Test suite for max_retries support."""

    def test_default_max_retries(self):
        """Test default max_retries is 2."""
        client = LLMClient(base_url="http://test/v1", model="test")
        assert client.max_retries == 2

    def test_explicit_max_retries(self):
        """Test explicit max_retries overrides default."""
        client = LLMClient(
            base_url="http://test/v1", model="test", max_retries=5
        )
        assert client.max_retries == 5

    def test_max_retries_zero(self):
        """Test max_retries=0 disables retries."""
        client = LLMClient(
            base_url="http://test/v1", model="test", max_retries=0
        )
        assert client.max_retries == 0

    @patch.dict(os.environ, {'LLM_MAX_RETRIES': '4'}, clear=False)
    def test_env_var_max_retries(self):
        """Test LLM_MAX_RETRIES env var is respected."""
        assert _resolve_max_retries() == 4

    @patch.dict(os.environ, {}, clear=True)
    def test_default_resolve_max_retries(self):
        """Test _resolve_max_retries returns 2 when no env var set."""
        assert _resolve_max_retries() == 2

    @patch('src.llm.llm_client.OpenAI')
    def test_max_retries_passed_to_sync_client(self, mock_openai_cls):
        """Test max_retries is passed to OpenAI constructor."""
        client = LLMClient(
            base_url="http://test/v1", model="test", timeout=30, max_retries=3
        )
        client._get_sync_client()

        mock_openai_cls.assert_called_once_with(
            base_url="http://test/v1",
            api_key="ollama",
            timeout=30.0,
            max_retries=3,
        )

    @patch('src.llm.llm_client.AsyncOpenAI')
    def test_max_retries_passed_to_async_client(self, mock_async_cls):
        """Test max_retries is passed to AsyncOpenAI constructor."""
        client = LLMClient(
            base_url="http://test/v1", model="test", timeout=30, max_retries=3
        )
        client._get_async_client()

        mock_async_cls.assert_called_once_with(
            base_url="http://test/v1",
            api_key="ollama",
            timeout=30.0,
            max_retries=3,
        )


# ============================================================================
# TestGetDefaultClient — singleton factory
# ============================================================================


class TestGetDefaultClient:
    """Test suite for get_default_client() factory."""

    def setup_method(self):
        """Reset singleton before each test."""
        llm_client_module._default_client = None

    def teardown_method(self):
        """Reset singleton after each test."""
        llm_client_module._default_client = None

    def test_returns_same_instance(self):
        """Test that repeated calls return the same instance."""
        client1 = get_default_client()
        client2 = get_default_client()
        assert client1 is client2

    def test_overrides_bypass_cache(self):
        """Test that passing overrides returns a new instance."""
        default = get_default_client()
        custom = get_default_client(timeout=999)
        assert custom is not default
        assert custom.timeout == 999

    def test_overrides_do_not_replace_cached(self):
        """Test that override calls do not replace the cached singleton."""
        default = get_default_client()
        get_default_client(timeout=999)
        default2 = get_default_client()
        assert default2 is default
