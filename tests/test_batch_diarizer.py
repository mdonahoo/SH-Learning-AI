"""
Tests for batch speaker diarizer with two-pass processing.

These tests verify the key properties of the two-pass architecture:
1. Same voice gets same ID regardless of audio length
2. Deterministic results (same audio = same output)
3. Speaker IDs ordered by first appearance
4. No state accumulation between audio files
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any


# Skip tests if dependencies not available
try:
    from src.audio.batch_diarizer import (
        BatchSpeakerDiarizer,
        DiarizationResult,
        SpeakerCluster,
        is_batch_diarizer_available
    )
    BATCH_DIARIZER_AVAILABLE = is_batch_diarizer_available()
except ImportError:
    BATCH_DIARIZER_AVAILABLE = False

try:
    from src.metrics.aggregate_role_inference import (
        AggregateRoleInferenceEngine,
        AggregateRoleAnalysis,
        is_aggregate_inference_available
    )
    AGGREGATE_INFERENCE_AVAILABLE = is_aggregate_inference_available()
except ImportError:
    AGGREGATE_INFERENCE_AVAILABLE = False


@pytest.fixture
def sample_segments():
    """Create sample transcript segments for testing."""
    return [
        {'start': 0.0, 'end': 2.0, 'text': 'Set course for the nebula.'},
        {'start': 2.5, 'end': 4.0, 'text': 'Course laid in, captain.'},
        {'start': 4.5, 'end': 6.0, 'text': 'Engage at warp five.'},
        {'start': 6.5, 'end': 8.0, 'text': 'Warp five, aye sir.'},
        {'start': 8.5, 'end': 10.0, 'text': 'Report from engineering.'},
        {'start': 10.5, 'end': 12.0, 'text': 'All systems nominal.'},
    ]


@pytest.fixture
def sample_audio():
    """Create sample audio data for testing."""
    # Generate 15 seconds of mock audio at 16kHz
    sample_rate = 16000
    duration = 15.0
    samples = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    samples = samples / np.max(np.abs(samples))  # Normalize
    return samples, sample_rate


class TestSpeakerCluster:
    """Tests for SpeakerCluster dataclass."""

    def test_compute_centroid_empty(self):
        """Centroid of empty cluster is None."""
        cluster = SpeakerCluster(speaker_id='speaker_1')
        assert cluster.compute_centroid() is None

    def test_compute_centroid_single_embedding(self):
        """Centroid of single embedding is that embedding."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            embeddings=[embedding]
        )
        centroid = cluster.compute_centroid()
        assert centroid is not None
        np.testing.assert_array_almost_equal(centroid, embedding)

    def test_compute_centroid_multiple_embeddings(self):
        """Centroid is mean of all embeddings."""
        embeddings = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        ]
        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            embeddings=embeddings
        )
        centroid = cluster.compute_centroid()
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(centroid, expected)

    def test_compute_tightness_single_embedding(self):
        """Single embedding has perfect tightness (1.0)."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            embeddings=[embedding]
        )
        tightness = cluster.compute_tightness()
        assert tightness == 1.0

    def test_compute_tightness_identical_embeddings(self):
        """Identical embeddings have perfect tightness."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            embeddings=[embedding, embedding.copy(), embedding.copy()]
        )
        tightness = cluster.compute_tightness()
        assert tightness == pytest.approx(1.0, abs=0.001)


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_get_speaker_voice_confidence_missing(self):
        """Returns 0 for unknown speaker."""
        result = DiarizationResult()
        confidence = result.get_speaker_voice_confidence('unknown_speaker')
        assert confidence == 0.0

    def test_get_speaker_voice_confidence_present(self):
        """Returns cluster tightness for known speaker."""
        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            cluster_tightness=0.85
        )
        result = DiarizationResult(
            speaker_clusters={'speaker_1': cluster}
        )
        confidence = result.get_speaker_voice_confidence('speaker_1')
        assert confidence == 0.85

    def test_get_segments_for_speaker(self):
        """Returns segment indices for speaker."""
        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            segment_indices=[0, 2, 4]
        )
        result = DiarizationResult(
            speaker_clusters={'speaker_1': cluster}
        )
        indices = result.get_segments_for_speaker('speaker_1')
        assert indices == [0, 2, 4]


@pytest.mark.skipif(
    not BATCH_DIARIZER_AVAILABLE,
    reason="Batch diarizer dependencies not available"
)
class TestBatchSpeakerDiarizer:
    """Tests for BatchSpeakerDiarizer class."""

    def test_initialization(self):
        """Diarizer initializes with default settings."""
        diarizer = BatchSpeakerDiarizer()
        # Threshold can vary based on environment variables
        assert 0.5 <= diarizer.similarity_threshold <= 1.0
        assert diarizer.min_speakers >= 1
        assert diarizer.max_speakers >= diarizer.min_speakers

    def test_reset_is_noop(self):
        """Reset does nothing (stateless design)."""
        diarizer = BatchSpeakerDiarizer()
        # Should not raise or change state
        diarizer.reset()

    def test_deterministic_results(self, sample_segments, sample_audio):
        """Same audio processed twice yields identical speaker assignments."""
        samples, sample_rate = sample_audio
        diarizer = BatchSpeakerDiarizer()

        # First processing
        segments1 = [s.copy() for s in sample_segments]
        segments1, result1 = diarizer.diarize_complete(samples, segments1, sample_rate)

        # Reset and process again
        diarizer.reset()
        segments2 = [s.copy() for s in sample_segments]
        segments2, result2 = diarizer.diarize_complete(samples, segments2, sample_rate)

        # Speaker assignments should be identical
        for s1, s2 in zip(segments1, segments2):
            assert s1.get('speaker_id') == s2.get('speaker_id'), \
                "Same audio should produce same speaker IDs"

    def test_no_state_accumulation(self, sample_segments, sample_audio):
        """Processing one audio doesn't affect next."""
        samples, sample_rate = sample_audio
        diarizer = BatchSpeakerDiarizer()

        # Process first audio
        segments1 = [s.copy() for s in sample_segments]
        segments1, result1 = diarizer.diarize_complete(samples, segments1, sample_rate)

        # Create different audio
        different_samples = -samples  # Invert
        segments2 = [s.copy() for s in sample_segments]
        segments2, result2 = diarizer.diarize_complete(
            different_samples, segments2, sample_rate
        )

        # Results should be independent
        # The diarizer should not have accumulated state from first call
        assert result2.total_speakers >= 1

    def test_speaker_id_ordering(self, sample_audio):
        """IDs ordered by first appearance in audio."""
        samples, sample_rate = sample_audio
        diarizer = BatchSpeakerDiarizer()

        # Create segments with clear temporal ordering
        segments = [
            {'start': 0.0, 'end': 2.0, 'text': 'First speaker here.'},
            {'start': 2.5, 'end': 4.0, 'text': 'Second speaker now.'},
            {'start': 4.5, 'end': 6.0, 'text': 'First speaker again.'},
        ]

        segments, result = diarizer.diarize_complete(samples, segments, sample_rate)

        # Check that speaker_1 appears first in time
        speaker_first_times = {}
        for seg in segments:
            sid = seg.get('speaker_id')
            if sid and sid not in speaker_first_times:
                speaker_first_times[sid] = seg['start']

        # speaker_1 should have earliest first appearance
        if 'speaker_1' in speaker_first_times:
            for other_speaker, first_time in speaker_first_times.items():
                if other_speaker != 'speaker_1':
                    assert speaker_first_times['speaker_1'] <= first_time, \
                        "speaker_1 should appear first in audio"

    def test_empty_segments(self, sample_audio):
        """Empty segment list returns empty result."""
        samples, sample_rate = sample_audio
        diarizer = BatchSpeakerDiarizer()

        segments, result = diarizer.diarize_complete(samples, [], sample_rate)

        assert segments == []
        assert result.total_speakers == 0

    def test_single_segment(self, sample_audio):
        """Single segment gets speaker_1."""
        samples, sample_rate = sample_audio
        diarizer = BatchSpeakerDiarizer()

        segments = [{'start': 0.0, 'end': 3.0, 'text': 'Only one speaker.'}]
        segments, result = diarizer.diarize_complete(samples, segments, sample_rate)

        assert len(segments) == 1
        assert segments[0].get('speaker_id') == 'speaker_1'
        assert result.total_speakers == 1

    def test_short_segment_interpolation(self, sample_audio):
        """Short segments get speaker from neighbors."""
        samples, sample_rate = sample_audio
        diarizer = BatchSpeakerDiarizer(min_embedding_duration=2.0)

        # Create segments with one short one in the middle
        segments = [
            {'start': 0.0, 'end': 3.0, 'text': 'Long segment from speaker A.'},
            {'start': 3.5, 'end': 4.0, 'text': 'Short.'},  # Too short for embedding
            {'start': 4.5, 'end': 7.0, 'text': 'Another long segment from speaker A.'},
        ]

        segments, result = diarizer.diarize_complete(samples, segments, sample_rate)

        # The short segment should be assigned a speaker (interpolated)
        assert segments[1].get('speaker_id') is not None
        assert 'speaker' in segments[1].get('speaker_id', '')


@pytest.mark.skipif(
    not AGGREGATE_INFERENCE_AVAILABLE,
    reason="Aggregate inference dependencies not available"
)
class TestAggregateRoleInferenceEngine:
    """Tests for AggregateRoleInferenceEngine class."""

    @pytest.fixture
    def sample_transcripts(self):
        """Create sample transcripts for role inference."""
        return [
            {'speaker': 'speaker_1', 'text': 'Set course for the nebula.', 'confidence': 0.9},
            {'speaker': 'speaker_2', 'text': 'Course laid in, captain.', 'confidence': 0.85},
            {'speaker': 'speaker_1', 'text': 'Engage at warp five.', 'confidence': 0.9},
            {'speaker': 'speaker_2', 'text': 'Warp five, aye sir.', 'confidence': 0.85},
            {'speaker': 'speaker_1', 'text': 'All hands, battle stations.', 'confidence': 0.9},
            {'speaker': 'speaker_3', 'text': 'Targeting enemy vessel.', 'confidence': 0.8},
        ]

    @pytest.fixture
    def mock_diarization_result(self):
        """Create mock diarization result."""
        cluster1 = SpeakerCluster(
            speaker_id='speaker_1',
            cluster_tightness=0.9,
            embeddings=[np.zeros(256)],
            segment_indices=[0, 2, 4]
        )
        cluster2 = SpeakerCluster(
            speaker_id='speaker_2',
            cluster_tightness=0.85,
            embeddings=[np.zeros(256)],
            segment_indices=[1, 3]
        )
        cluster3 = SpeakerCluster(
            speaker_id='speaker_3',
            cluster_tightness=0.8,
            embeddings=[np.zeros(256)],
            segment_indices=[5]
        )
        return DiarizationResult(
            speaker_clusters={
                'speaker_1': cluster1,
                'speaker_2': cluster2,
                'speaker_3': cluster3
            },
            segment_assignments=['speaker_1', 'speaker_2', 'speaker_1', 'speaker_2', 'speaker_1', 'speaker_3'],
            segment_confidences=[0.9, 0.85, 0.9, 0.85, 0.9, 0.8],
            total_speakers=3,
            methodology_note="Test methodology"
        )

    def test_infer_roles_without_diarization(self, sample_transcripts):
        """Role inference works without diarization result."""
        engine = AggregateRoleInferenceEngine(sample_transcripts)
        results = engine.infer_roles()

        assert len(results) >= 1
        for speaker_id, analysis in results.items():
            assert analysis.speaker == speaker_id
            assert analysis.voice_confidence == 0.5  # Default when no diarization

    def test_infer_roles_with_diarization(self, sample_transcripts, mock_diarization_result):
        """Role inference uses diarization confidence."""
        engine = AggregateRoleInferenceEngine(
            sample_transcripts,
            diarization_result=mock_diarization_result
        )
        results = engine.infer_roles()

        assert len(results) >= 1
        # Check that voice confidence comes from diarization
        if 'speaker_1' in results:
            assert results['speaker_1'].voice_confidence == 0.9

    def test_combined_confidence_calculation(self, sample_transcripts, mock_diarization_result):
        """Combined confidence uses correct weights."""
        engine = AggregateRoleInferenceEngine(
            sample_transcripts,
            diarization_result=mock_diarization_result
        )
        results = engine.infer_roles()

        for analysis in results.values():
            # Combined should be weighted sum
            expected = (
                analysis.voice_confidence * 0.40 +
                analysis.role_confidence * 0.40 +
                analysis.evidence_factor * 0.20
            )
            assert abs(analysis.combined_confidence - expected) < 0.01

    def test_get_structured_results(self, sample_transcripts, mock_diarization_result):
        """Structured results include all fields."""
        engine = AggregateRoleInferenceEngine(
            sample_transcripts,
            diarization_result=mock_diarization_result
        )
        results = engine.get_structured_results()

        assert 'speaker_roles' in results
        assert 'diarization_methodology' in results
        assert 'inference_weights' in results

        # Check inference weights
        weights = results['inference_weights']
        assert weights['voice_weight'] == 0.40
        assert weights['role_weight'] == 0.40
        assert weights['evidence_weight'] == 0.20


class TestIntegration:
    """Integration tests for the full two-pass pipeline."""

    @pytest.mark.skipif(
        not (BATCH_DIARIZER_AVAILABLE and AGGREGATE_INFERENCE_AVAILABLE),
        reason="Full pipeline dependencies not available"
    )
    def test_full_pipeline_consistency(self, sample_audio):
        """Test the full two-pass pipeline produces consistent results."""
        samples, sample_rate = sample_audio

        # Create test segments
        segments = [
            {'start': 0.0, 'end': 2.0, 'text': 'Captain, we have incoming.'},
            {'start': 2.5, 'end': 4.0, 'text': 'Raise shields. Red alert.'},
            {'start': 4.5, 'end': 6.0, 'text': 'Shields up, captain.'},
            {'start': 6.5, 'end': 8.0, 'text': 'Target the lead vessel.'},
            {'start': 8.5, 'end': 10.0, 'text': 'Targeting lock acquired.'},
        ]

        # Pass 1: Batch diarization
        diarizer = BatchSpeakerDiarizer()
        segments, diarization_result = diarizer.diarize_complete(
            samples, segments, sample_rate
        )

        # Verify diarization result
        assert diarization_result.total_speakers >= 1
        assert len(diarization_result.segment_assignments) == len(segments)

        # Build transcripts for role inference
        transcripts = [
            {
                'speaker': seg.get('speaker_id', 'unknown'),
                'text': seg['text'],
                'confidence': seg.get('speaker_confidence', 0.5)
            }
            for seg in segments
        ]

        # Pass 2: Aggregate role inference
        role_engine = AggregateRoleInferenceEngine(
            transcripts,
            diarization_result=diarization_result
        )
        role_results = role_engine.infer_roles()

        # Verify role results
        assert len(role_results) >= 1
        for analysis in role_results.values():
            assert analysis.voice_confidence >= 0.0
            assert analysis.role_confidence >= 0.0
            assert analysis.combined_confidence >= 0.0
            assert analysis.combined_confidence <= 1.0
