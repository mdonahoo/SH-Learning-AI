"""Tests for domain post-correction and LLM transcript cleanup."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.audio.domain_postcorrector import (
    DomainPostCorrector,
    TranscriptLLMCleaner,
    PHRASE_CORRECTIONS,
    TRANSCRIPT_CORRECTION_PROMPT,
)


# ============================================================================
# TestDomainPostCorrector — Layer 3 phrase-level corrections
# ============================================================================


class TestDomainPostCorrector:
    """Test suite for DomainPostCorrector."""

    @pytest.fixture
    def corrector(self):
        """Create test corrector instance."""
        return DomainPostCorrector()

    def test_phrase_correction_with_intolerance(self, corrector):
        """Test 'with intolerance' is corrected to 'within tolerance'."""
        segments = [
            {'text': 'Structural integrity with intolerance', 'start': 0, 'end': 3}
        ]
        result, stats = corrector.correct_segments(segments)
        assert result[0]['text'] == 'Structural integrity within tolerance'
        assert stats['corrections_count'] == 1

    def test_phrase_correction_nod_unkindly(self, corrector):
        """Test 'nod unkindly' is corrected to 'not unkindly'."""
        segments = [
            {'text': 'He said nod unkindly that we should proceed', 'start': 0, 'end': 5}
        ]
        result, stats = corrector.correct_segments(segments)
        assert 'not unkindly' in result[0]['text']
        assert 'nod unkindly' not in result[0]['text']

    def test_phrase_correction_shit_breeze(self, corrector):
        """Test 'shit breeze' is corrected to 'ship breathe'."""
        segments = [
            {'text': 'Let the shit breeze for a moment', 'start': 0, 'end': 4}
        ]
        result, stats = corrector.correct_segments(segments)
        assert 'ship breathe' in result[0]['text']
        assert 'shit breeze' not in result[0]['text']

    def test_context_dependent_directed(self, corrector):
        """Test 'contact directed' becomes 'contact detected'."""
        segments = [
            {'text': 'New contact directed at bearing 045', 'start': 0, 'end': 5}
        ]
        result, stats = corrector.correct_segments(segments)
        assert 'contact detected' in result[0]['text']
        assert stats['corrections_count'] == 1

    def test_no_false_positives_directed(self, corrector):
        """Test 'directed the crew' stays unchanged (no preceding context word)."""
        segments = [
            {'text': 'The captain directed the crew to stations', 'start': 0, 'end': 5}
        ]
        result, stats = corrector.correct_segments(segments)
        assert result[0]['text'] == 'The captain directed the crew to stations'
        assert stats['corrections_count'] == 0

    def test_preserves_correct_text(self, corrector):
        """Test that correct text passes through unmodified."""
        segments = [
            {'text': 'Shields up, red alert!', 'start': 0, 'end': 3},
            {'text': 'Aye captain, all stations report ready', 'start': 3, 'end': 7},
        ]
        result, stats = corrector.correct_segments(segments)
        assert result[0]['text'] == 'Shields up, red alert!'
        assert result[1]['text'] == 'Aye captain, all stations report ready'
        assert stats['corrections_count'] == 0

    def test_empty_segments(self, corrector):
        """Test handles empty segment list."""
        result, stats = corrector.correct_segments([])
        assert result == []
        assert stats['corrections_count'] == 0
        assert stats['corrected_segments'] == 0

    def test_stats_tracking(self, corrector):
        """Test correction stats are properly tracked."""
        segments = [
            {'text': 'with intolerance the hull is fine', 'start': 0, 'end': 3},
            {'text': 'All systems nominal', 'start': 3, 'end': 6},
            {'text': 'shit breeze is happening', 'start': 6, 'end': 9},
        ]
        result, stats = corrector.correct_segments(segments)
        assert stats['corrections_count'] == 2
        assert stats['corrected_segments'] == 2
        assert len(stats['corrections_log']) == 2

    def test_multiple_corrections_single_segment(self, corrector):
        """Test multiple errors in one segment are all corrected."""
        segments = [
            {'text': 'Signal directed with intolerance', 'start': 0, 'end': 5}
        ]
        result, stats = corrector.correct_segments(segments)
        assert 'signal detected' in result[0]['text'].lower()
        assert 'within tolerance' in result[0]['text']
        assert stats['corrections_count'] >= 2

    def test_case_insensitive_matching(self, corrector):
        """Test corrections work regardless of case."""
        segments = [
            {'text': 'WITH INTOLERANCE the readings are stable', 'start': 0, 'end': 3}
        ]
        result, stats = corrector.correct_segments(segments)
        assert 'within tolerance' in result[0]['text'].lower()

    def test_extra_corrections(self):
        """Test that extra corrections are appended."""
        extra = [
            (r'\bfoo bar\b', 'baz qux', True),
        ]
        corrector = DomainPostCorrector(extra_corrections=extra)
        segments = [
            {'text': 'Testing foo bar replacement', 'start': 0, 'end': 3}
        ]
        result, stats = corrector.correct_segments(segments)
        assert 'baz qux' in result[0]['text']

    def test_segment_without_text_key(self, corrector):
        """Test segments without text key are handled gracefully."""
        segments = [
            {'start': 0, 'end': 3},  # no text key
            {'text': '', 'start': 3, 'end': 6},  # empty text
            {'text': 'with intolerance', 'start': 6, 'end': 9},
        ]
        result, stats = corrector.correct_segments(segments)
        assert stats['corrections_count'] == 1
        assert result[2]['text'] == 'within tolerance'


# ============================================================================
# TestTranscriptLLMCleaner — Layer 5 LLM cleanup
# ============================================================================


class TestTranscriptLLMCleaner:
    """Test suite for TranscriptLLMCleaner."""

    @pytest.fixture
    def cleaner(self):
        """Create test cleaner instance."""
        return TranscriptLLMCleaner(
            host='http://localhost:11434',
            model='llama3.2',
            timeout=30
        )

    def test_initialization(self):
        """Test cleaner initializes with env var configuration."""
        with patch.dict(os.environ, {
            'OLLAMA_HOST': 'http://test:11434',
            'OLLAMA_MODEL': 'test-model',
            'OLLAMA_TIMEOUT': '60',
            'LLM_TRANSCRIPT_CLEANUP': 'true'
        }):
            cleaner = TranscriptLLMCleaner()
            assert cleaner.host == 'http://test:11434'
            assert cleaner.model == 'test-model'
            assert cleaner.timeout == 60
            assert cleaner.enabled is True

    def test_build_batch_prompt(self, cleaner):
        """Test prompt format is correct."""
        batch = [
            {'text': 'with intolerance the structural integrity is nominal', 'start': 0, 'end': 5},
            {'text': 'aye sir setting course', 'start': 5, 'end': 8},
            {'text': 'nod unkindly the captain said', 'start': 8, 'end': 12},
        ]
        prompt = cleaner._build_batch_prompt(batch)
        assert '1: with intolerance the structural integrity is nominal' in prompt
        assert '2: aye sir setting course' in prompt
        assert '3: nod unkindly the captain said' in prompt
        assert 'LINE_NUMBER|corrected text' in prompt

    def test_parse_corrections(self, cleaner):
        """Test parsing of LLM correction output format."""
        response = "1|within tolerance the structural integrity is nominal\n3|not unkindly the captain said"
        corrections = cleaner._parse_corrections(response)
        assert corrections[1] == 'within tolerance the structural integrity is nominal'
        assert corrections[3] == 'not unkindly the captain said'
        assert 2 not in corrections

    def test_parse_corrections_handles_noise(self, cleaner):
        """Test parser ignores non-correction lines."""
        response = """Here are the corrections:
1|fixed text here
Some explanation
2|another fix
invalid line
"""
        corrections = cleaner._parse_corrections(response)
        assert len(corrections) == 2
        assert corrections[1] == 'fixed text here'
        assert corrections[2] == 'another fix'

    def test_parse_corrections_handles_variants(self, cleaner):
        """Test parser handles format variants like '1.|text' and '1:|text'."""
        response = "1.|corrected one\n2:|corrected two"
        corrections = cleaner._parse_corrections(response)
        assert corrections[1] == 'corrected one'
        assert corrections[2] == 'corrected two'

    @patch('src.audio.domain_postcorrector.requests')
    def test_clean_segments_with_mock(self, mock_requests, cleaner):
        """Test full cleanup flow with mocked Ollama."""
        # Mock connection check
        mock_tags_response = Mock()
        mock_tags_response.status_code = 200

        # Mock generation response
        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {
            'response': '1|within tolerance the hull is stable'
        }
        mock_gen_response.raise_for_status = Mock()

        mock_requests.get.return_value = mock_tags_response
        mock_requests.post.return_value = mock_gen_response

        segments = [
            {'text': 'with intolerance the hull is stable', 'start': 0, 'end': 5},
            {'text': 'aye sir', 'start': 5, 'end': 7},
        ]

        result, stats = cleaner.clean_segments(segments)
        assert stats['batches_sent'] == 1
        assert stats['corrections_made'] == 1
        assert result[0]['text'] == 'within tolerance the hull is stable'
        assert result[1]['text'] == 'aye sir'  # unchanged

    def test_disabled_by_env(self):
        """Test LLM cleanup is skipped when disabled by env var."""
        with patch.dict(os.environ, {'LLM_TRANSCRIPT_CLEANUP': 'false'}):
            cleaner = TranscriptLLMCleaner()
            segments = [
                {'text': 'some text', 'start': 0, 'end': 3}
            ]
            result, stats = cleaner.clean_segments(segments)
            assert stats['corrections_made'] == 0
            assert 'skipped_reason' in stats
            assert result[0]['text'] == 'some text'

    @patch('src.audio.domain_postcorrector.requests')
    def test_ollama_unavailable(self, mock_requests, cleaner):
        """Test graceful degradation when Ollama is down."""
        mock_requests.get.side_effect = ConnectionError("Connection refused")

        segments = [
            {'text': 'some text', 'start': 0, 'end': 3}
        ]
        result, stats = cleaner.clean_segments(segments)
        assert stats['corrections_made'] == 0
        assert 'skipped_reason' in stats
        assert result[0]['text'] == 'some text'

    def test_empty_segments(self, cleaner):
        """Test handles empty segment list."""
        result, stats = cleaner.clean_segments([])
        assert result == []
        assert stats['batches_sent'] == 0
        assert stats['corrections_made'] == 0

    @patch('src.audio.domain_postcorrector.requests')
    def test_batch_size_splitting(self, mock_requests, cleaner):
        """Test that large segment lists are batched correctly."""
        # Mock connection check
        mock_tags_response = Mock()
        mock_tags_response.status_code = 200

        # Mock generation - return no corrections to simplify
        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {'response': ''}
        mock_gen_response.raise_for_status = Mock()

        mock_requests.get.return_value = mock_tags_response
        mock_requests.post.return_value = mock_gen_response

        # Create 25 segments, batch_size=10
        segments = [
            {'text': f'Segment number {i}', 'start': i, 'end': i + 1}
            for i in range(25)
        ]

        result, stats = cleaner.clean_segments(segments, batch_size=10)
        # Should have made 3 batches (10 + 10 + 5)
        assert mock_requests.post.call_count == 3

    @patch('src.audio.domain_postcorrector.requests')
    def test_llm_does_not_apply_identical_text(self, mock_requests, cleaner):
        """Test that corrections identical to original are not counted."""
        mock_tags_response = Mock()
        mock_tags_response.status_code = 200

        mock_gen_response = Mock()
        mock_gen_response.status_code = 200
        mock_gen_response.json.return_value = {
            'response': '1|aye sir'  # same as original
        }
        mock_gen_response.raise_for_status = Mock()

        mock_requests.get.return_value = mock_tags_response
        mock_requests.post.return_value = mock_gen_response

        segments = [
            {'text': 'aye sir', 'start': 0, 'end': 2}
        ]
        result, stats = cleaner.clean_segments(segments)
        assert stats['corrections_made'] == 0

    @patch('src.audio.domain_postcorrector.requests')
    def test_ollama_returns_error_status(self, mock_requests, cleaner):
        """Test handling when Ollama tags endpoint returns non-200."""
        mock_tags_response = Mock()
        mock_tags_response.status_code = 500

        mock_requests.get.return_value = mock_tags_response

        segments = [
            {'text': 'test text', 'start': 0, 'end': 3}
        ]
        result, stats = cleaner.clean_segments(segments)
        assert stats['corrections_made'] == 0
        assert 'skipped_reason' in stats
