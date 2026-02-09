"""
Tests for Ollama LLM client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.llm_client import LLMResponse
from src.llm.ollama_client import OllamaClient


class TestOllamaClient:
    """Test suite for OllamaClient."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return OllamaClient(
            host='http://localhost:11434',
            model='llama3.2',
            timeout=30
        )

    def test_initialization(self, client):
        """Test client initializes with correct values."""
        assert client.host == 'http://localhost:11434'
        assert client.model == 'llama3.2'
        assert client.timeout == 30

    def test_host_trailing_slash_removed(self):
        """Test that trailing slashes are removed from host."""
        client = OllamaClient(host='http://localhost:11434/')
        assert client.host == 'http://localhost:11434'

    def test_llm_client_created(self, client):
        """Test that internal LLMClient is created with correct base_url."""
        assert client._llm is not None
        assert client._llm.base_url == 'http://localhost:11434/v1'
        assert client._llm.model == 'llama3.2'

    def test_check_connection_success(self, client):
        """Test successful connection check delegates to LLMClient."""
        client._llm = MagicMock()
        client._llm.check_available.return_value = True
        assert client.check_connection() is True
        client._llm.check_available.assert_called_once()

    def test_check_connection_failure(self, client):
        """Test failed connection check via LLMClient."""
        client._llm = MagicMock()
        client._llm.check_available.return_value = False
        assert client.check_connection() is False

    def test_list_models(self, client):
        """Test listing available models."""
        client._llm = MagicMock()
        client._llm.list_models.return_value = ['llama3.2', 'mistral', 'codellama']

        models = client.list_models()
        assert models == ['llama3.2', 'mistral', 'codellama']

    def test_generate_success(self, client):
        """Test successful text generation."""
        client._llm = MagicMock()
        client._llm.generate.return_value = LLMResponse(
            text='This is a test response from the LLM.',
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            model='llama3.2',
        )

        result = client.generate("Test prompt")
        assert result == 'This is a test response from the LLM.'

    def test_generate_with_system_prompt(self, client):
        """Test generation with system prompt."""
        client._llm = MagicMock()
        client._llm.generate.return_value = LLMResponse(text='Response', model='llama3.2')

        client.generate("User prompt", system="System prompt")

        call_kwargs = client._llm.generate.call_args[1]
        assert call_kwargs['system'] == "System prompt"

    def test_generate_failure_returns_none(self, client):
        """Test generation failure returns None."""
        client._llm = MagicMock()
        client._llm.generate.return_value = None

        result = client.generate("Test prompt")
        assert result is None

    def test_generate_populates_metrics_out(self, client):
        """Test that metrics_out dict is populated from LLMResponse."""
        client._llm = MagicMock()
        client._llm.generate.return_value = LLMResponse(
            text='Response',
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model='llama3.2',
        )

        metrics: dict = {}
        client.generate("Test", metrics_out=metrics)

        assert metrics['model'] == 'llama3.2'
        assert metrics['prompt_eval_count'] == 100
        assert metrics['eval_count'] == 50

    @patch('src.llm.ollama_client.OllamaClient.generate')
    def test_generate_mission_summary(self, mock_generate, client):
        """Test mission summary generation."""
        mock_generate.return_value = "Mission summary text"

        mission_data = {
            'mission_id': 'TEST_001',
            'mission_name': 'Test Mission',
            'events': [],
            'transcripts': []
        }

        result = client.generate_mission_summary(mission_data, style='entertaining')

        assert result == "Mission summary text"
        mock_generate.assert_called_once()

    @patch('src.llm.ollama_client.OllamaClient.generate')
    def test_generate_crew_analysis(self, mock_generate, client):
        """Test crew analysis generation."""
        mock_generate.return_value = "Crew analysis text"

        transcripts = [
            {'speaker': 'speaker_1', 'text': 'Hello'},
            {'speaker': 'speaker_2', 'text': 'World'}
        ]
        events = []

        result = client.generate_crew_analysis(transcripts, events)

        assert result == "Crew analysis text"
        mock_generate.assert_called_once()

    @patch('src.llm.ollama_client.OllamaClient.generate_with_progress')
    def test_generate_full_report(self, mock_generate_wp, client):
        """Test full report generation."""
        mock_generate_wp.return_value = "# Full Mission Report\n\nReport content..."

        mission_data = {
            'mission_id': 'TEST_001',
            'mission_name': 'Test Mission',
            'events': [],
            'transcripts': []
        }

        result = client.generate_full_report(mission_data, style='professional')

        assert result.startswith("# Full Mission Report")
        mock_generate_wp.assert_called_once()

    def test_generate_with_temperature(self, client):
        """Test that temperature parameter is passed correctly."""
        client._llm = MagicMock()
        client._llm.generate.return_value = LLMResponse(text='Response', model='llama3.2')

        client.generate("Test", temperature=0.5)

        call_kwargs = client._llm.generate.call_args[1]
        assert call_kwargs['temperature'] == 0.5

    def test_generate_with_max_tokens(self, client):
        """Test that max_tokens parameter is passed correctly."""
        client._llm = MagicMock()
        client._llm.generate.return_value = LLMResponse(text='Response', model='llama3.2')

        client.generate("Test", max_tokens=2048)

        call_kwargs = client._llm.generate.call_args[1]
        assert call_kwargs['max_tokens'] == 2048

    def test_generate_streaming_success(self, client):
        """Test streaming generation collects all chunks."""
        client._llm = MagicMock()
        client._llm.generate_streaming.return_value = iter(["Hello ", "world", "!"])

        chunks_received = []
        result = client.generate_streaming(
            "Test prompt",
            callback=lambda c: chunks_received.append(c)
        )

        assert result == "Hello world!"
        assert chunks_received == ["Hello ", "world", "!"]

    def test_generate_streaming_failure(self, client):
        """Test streaming generation failure."""
        client._llm = MagicMock()
        client._llm.generate_streaming.side_effect = Exception("Error")

        result = client.generate_streaming("Test prompt")
        assert result is None
