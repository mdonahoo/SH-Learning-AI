"""
Tests for Ollama LLM client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
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

    @patch('src.llm.ollama_client.requests.get')
    def test_check_connection_success(self, mock_get, client):
        """Test successful connection check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert client.check_connection() is True
        mock_get.assert_called_once_with('http://localhost:11434/api/tags', timeout=5)

    @patch('src.llm.ollama_client.requests.get')
    def test_check_connection_failure(self, mock_get, client):
        """Test failed connection check."""
        mock_get.side_effect = Exception("Connection refused")

        assert client.check_connection() is False

    @patch('src.llm.ollama_client.requests.get')
    def test_list_models(self, mock_get, client):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama3.2'},
                {'name': 'mistral'},
                {'name': 'codellama'}
            ]
        }
        mock_get.return_value = mock_response

        models = client.list_models()
        assert models == ['llama3.2', 'mistral', 'codellama']

    @patch('src.llm.ollama_client.requests.post')
    def test_generate_success(self, mock_post, client):
        """Test successful text generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'This is a test response from the LLM.'
        }
        mock_post.return_value = mock_response

        result = client.generate("Test prompt")

        assert result == 'This is a test response from the LLM.'
        mock_post.assert_called_once()

    @patch('src.llm.ollama_client.requests.post')
    def test_generate_with_system_prompt(self, mock_post, client):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Response'}
        mock_post.return_value = mock_response

        client.generate("User prompt", system="System prompt")

        # Check that system prompt was included in payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['system'] == "System prompt"

    @patch('src.llm.ollama_client.requests.post')
    def test_generate_timeout(self, mock_post, client):
        """Test generation timeout handling."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        result = client.generate("Test prompt")
        assert result is None

    @patch('src.llm.ollama_client.requests.post')
    def test_generate_request_exception(self, mock_post, client):
        """Test generation request exception handling."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        result = client.generate("Test prompt")
        assert result is None

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

    @patch('src.llm.ollama_client.OllamaClient.generate')
    def test_generate_full_report(self, mock_generate, client):
        """Test full report generation."""
        mock_generate.return_value = "# Full Mission Report\n\nReport content..."

        mission_data = {
            'mission_id': 'TEST_001',
            'mission_name': 'Test Mission',
            'events': [],
            'transcripts': []
        }

        result = client.generate_full_report(mission_data, style='professional')

        assert result.startswith("# Full Mission Report")
        mock_generate.assert_called_once()

    def test_generate_with_temperature(self, client):
        """Test that temperature parameter is passed correctly."""
        with patch('src.llm.ollama_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': 'Response'}
            mock_post.return_value = mock_response

            client.generate("Test", temperature=0.5)

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert payload['options']['temperature'] == 0.5

    def test_generate_with_max_tokens(self, client):
        """Test that max_tokens parameter is passed correctly."""
        with patch('src.llm.ollama_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': 'Response'}
            mock_post.return_value = mock_response

            client.generate("Test", max_tokens=2048)

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert payload['options']['num_predict'] == 2048
