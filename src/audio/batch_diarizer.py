"""
Batch speaker diarization with two-pass processing architecture.

This module implements a stateless batch clustering approach to ensure
consistent speaker IDs across different audio lengths. The key insight
is that speaker IDs should be deterministic and based on the complete
audio, not incrementally assigned as segments are processed.

Two-Pass Architecture:
1. Pass 1 - Speaker Clustering: Process ENTIRE audio before assigning
   ANY speaker IDs. Use stateless batch clustering so the same voice
   always gets the same ID.
2. Pass 2 - Role Inference: With stable speaker IDs, aggregate ALL
   utterances per speaker and assign roles based on complete evidence.

Key Benefits:
- Same voice gets same speaker ID regardless of audio length
- Deterministic results (same audio = same output)
- Speaker IDs ordered by first appearance in audio
- No state accumulation between audio files
"""

import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    VoiceEncoder = None

try:
    from scipy.spatial.distance import cosine, cdist
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SpeakerCluster:
    """
    Information about a speaker cluster.

    Attributes:
        speaker_id: Deterministic speaker identifier (e.g., 'speaker_1')
        embeddings: All voice embeddings assigned to this speaker
        segment_indices: Indices of segments belonging to this speaker
        first_appearance: Timestamp of first appearance in audio
        total_duration: Total speaking time for this speaker
        centroid: Mean embedding vector for this cluster
        cluster_tightness: How similar embeddings are within cluster (0-1)
    """
    speaker_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    segment_indices: List[int] = field(default_factory=list)
    first_appearance: float = float('inf')
    total_duration: float = 0.0
    centroid: Optional[np.ndarray] = None
    cluster_tightness: float = 0.0

    def compute_centroid(self) -> Optional[np.ndarray]:
        """Compute and cache the centroid of all embeddings."""
        if not self.embeddings:
            return None
        self.centroid = np.mean(self.embeddings, axis=0)
        return self.centroid

    def compute_tightness(self) -> float:
        """
        Compute cluster tightness (voice confidence).

        Returns average similarity of embeddings to centroid.
        Higher = more consistent voice = higher confidence.
        """
        if len(self.embeddings) < 2:
            self.cluster_tightness = 1.0  # Single embedding = perfect consistency
            return self.cluster_tightness

        if self.centroid is None:
            self.compute_centroid()

        if self.centroid is None:
            return 0.0

        # Calculate average similarity to centroid
        similarities = []
        for emb in self.embeddings:
            sim = 1.0 - cosine(emb, self.centroid)
            similarities.append(sim)

        self.cluster_tightness = float(np.mean(similarities))
        return self.cluster_tightness


@dataclass
class DiarizationResult:
    """
    Complete result of batch diarization.

    This is the output of Pass 1 (speaker clustering) and serves as
    input to Pass 2 (role inference).

    Attributes:
        speaker_clusters: Dict mapping speaker_id to SpeakerCluster
        segment_assignments: List of speaker_id for each segment
        segment_confidences: Per-segment confidence scores
        total_speakers: Number of unique speakers detected
        methodology_note: Description of how diarization was performed
    """
    speaker_clusters: Dict[str, SpeakerCluster] = field(default_factory=dict)
    segment_assignments: List[str] = field(default_factory=list)
    segment_confidences: List[float] = field(default_factory=list)
    total_speakers: int = 0
    methodology_note: str = ""

    def get_speaker_voice_confidence(self, speaker_id: str) -> float:
        """
        Get voice confidence for a speaker.

        This reflects how consistently the speaker's voice clusters,
        independent of role inference.
        """
        cluster = self.speaker_clusters.get(speaker_id)
        if cluster:
            return cluster.cluster_tightness
        return 0.0

    def get_segments_for_speaker(self, speaker_id: str) -> List[int]:
        """Get segment indices for a specific speaker."""
        cluster = self.speaker_clusters.get(speaker_id)
        if cluster:
            return cluster.segment_indices
        return []


class BatchSpeakerDiarizer:
    """
    Stateless batch speaker diarizer using two-pass processing.

    Unlike incremental diarization, this processes the ENTIRE audio
    before assigning any speaker IDs. This ensures:
    1. Same voice always gets same ID regardless of audio length
    2. Results are deterministic (same audio = same output)
    3. Speaker IDs ordered by first appearance in audio

    Usage:
        diarizer = BatchSpeakerDiarizer()
        result = diarizer.diarize_complete(audio_path, segments)
        # result.speaker_clusters contains per-speaker voice data
        # result.segment_assignments contains speaker_id per segment
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        min_embedding_duration: float = 1.0
    ):
        """
        Initialize batch speaker diarizer.

        Args:
            similarity_threshold: Minimum similarity (0-1) to cluster speakers
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
            min_embedding_duration: Minimum segment duration for embedding extraction
        """
        if not RESEMBLYZER_AVAILABLE:
            raise ImportError(
                "resemblyzer is not installed. "
                "Install with: pip install resemblyzer"
            )

        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is not installed. "
                "Install with: pip install scipy"
            )

        # Load encoder
        logger.info("Loading resemblyzer voice encoder for batch diarization...")
        self.encoder = VoiceEncoder()
        logger.info("Voice encoder loaded")

        # Configuration
        # Lower threshold = voices grouped together more aggressively = fewer speakers
        # 0.60 works well for distinct bridge crews (Captain, Helm, Tactical, Science, Engineering, Communications)
        self.similarity_threshold = similarity_threshold or float(
            os.getenv('SPEAKER_EMBEDDING_THRESHOLD', '0.60')
        )
        self.min_speakers = min_speakers or int(os.getenv('MIN_EXPECTED_SPEAKERS', '4'))
        # For chunked diarization, allow reasonable number of speakers per chunk
        # Cross-chunk merging will consolidate similar voices
        self.max_speakers = max_speakers or int(os.getenv('MAX_EXPECTED_SPEAKERS', '10'))
        self.expected_speakers = int(os.getenv('EXPECTED_BRIDGE_CREW', '6'))
        self.min_embedding_duration = min_embedding_duration

        # Sample rate expected by resemblyzer
        self.sample_rate = 16000

        # Chunk-based diarization for long audio (prevents speaker drift)
        # Audio longer than this will be processed in chunks
        self.chunk_threshold_seconds = float(
            os.getenv('DIARIZATION_CHUNK_THRESHOLD', '900')  # 15 minutes
        )
        self.chunk_duration_seconds = float(
            os.getenv('DIARIZATION_CHUNK_DURATION', '600')  # 10 minutes per chunk
        )
        self.chunk_overlap_seconds = float(
            os.getenv('DIARIZATION_CHUNK_OVERLAP', '30')  # 30 second overlap
        )

        # Sub-segment diarization for long segments with multiple speakers
        # Segments longer than this threshold will be split into sub-segments
        self.subsegment_threshold_seconds = float(
            os.getenv('SUBSEGMENT_THRESHOLD', '3.0')  # Segments > 3s get sub-segmented
        )
        self.subsegment_window_seconds = float(
            os.getenv('SUBSEGMENT_WINDOW', '1.5')  # Extract embedding every 1.5s (reduced for faster changes)
        )
        self.subsegment_hop_seconds = float(
            os.getenv('SUBSEGMENT_HOP', '0.5')  # Hop 0.5s for fine-grained speaker change detection
        )
        # Minimum similarity to consider same speaker within a segment
        self.subsegment_speaker_threshold = float(
            os.getenv('SUBSEGMENT_SPEAKER_THRESHOLD', '0.75')
        )
        # Energy-based pause detection for finding speaker boundaries
        self.pause_energy_threshold = float(
            os.getenv('PAUSE_ENERGY_THRESHOLD', '0.02')  # RMS below this = silence/pause
        )
        self.min_pause_duration_seconds = float(
            os.getenv('MIN_PAUSE_DURATION', '0.25')  # Minimum pause to consider as speaker boundary (increased)
        )
        # Minimum segment duration after splitting (prevents micro-segments)
        self.min_split_segment_duration = float(
            os.getenv('MIN_SPLIT_SEGMENT_DURATION', '1.0')  # At least 1 second per speaker
        )
        # Minimum segments for a speaker to be considered "major" (others get merged)
        self.min_speaker_segments = int(
            os.getenv('MIN_SPEAKER_SEGMENTS', '5')  # Speakers with < this get consolidated
        )

        logger.info(
            f"BatchSpeakerDiarizer initialized: threshold={self.similarity_threshold}, "
            f"speakers={self.min_speakers}-{self.max_speakers}, "
            f"chunk_threshold={self.chunk_threshold_seconds}s, "
            f"subsegment_threshold={self.subsegment_threshold_seconds}s"
        )

    def diarize_complete(
        self,
        audio_data: np.ndarray,
        segments: List[Dict[str, Any]],
        sample_rate: int = 16000
    ) -> Tuple[List[Dict[str, Any]], DiarizationResult]:
        """
        Perform complete two-pass diarization.

        This is the main entry point. It processes all segments at once
        to ensure consistent speaker IDs.

        For long audio (>10 minutes by default), uses chunk-based processing
        to prevent speaker drift that occurs when embeddings accumulate errors.

        Args:
            audio_data: Full audio samples (float32, normalized -1 to 1)
            segments: List of segments with 'start', 'end', 'text' keys
            sample_rate: Audio sample rate

        Returns:
            Tuple of (updated_segments, DiarizationResult)
            - segments: Original segments with 'speaker_id' and 'speaker_confidence' added
            - result: DiarizationResult with cluster info for role inference
        """
        if not segments:
            return segments, DiarizationResult()

        # Calculate audio duration
        audio_duration = len(audio_data) / sample_rate
        logger.info(f"Batch diarization: {len(segments)} segments, {audio_duration:.1f}s audio")

        # Optional audio preprocessing for better embedding quality
        preprocess_enabled = os.getenv('DIARIZATION_PREPROCESS', 'true').lower() == 'true'
        if preprocess_enabled:
            try:
                from src.audio.preprocessing import preprocess_audio, get_audio_stats

                # Log pre-processing stats
                stats_before = get_audio_stats(audio_data, sample_rate)
                logger.info(
                    f"Audio stats before preprocessing: "
                    f"peak={stats_before['peak']:.3f}, rms={stats_before['rms']:.3f}, "
                    f"SNR≈{stats_before['snr_estimate']:.1f}dB"
                )

                # Apply preprocessing
                audio_data = preprocess_audio(
                    audio_data, sample_rate,
                    normalize=True,
                    highpass=True,
                    noise_reduce=True,
                    noise_strength=float(os.getenv('NOISE_REDUCE_STRENGTH', '0.5'))
                )

                # Log post-processing stats
                stats_after = get_audio_stats(audio_data, sample_rate)
                logger.info(
                    f"Audio stats after preprocessing: "
                    f"peak={stats_after['peak']:.3f}, rms={stats_after['rms']:.3f}, "
                    f"SNR≈{stats_after['snr_estimate']:.1f}dB"
                )
            except ImportError as e:
                logger.debug(f"Preprocessing not available: {e}")
            except Exception as e:
                logger.warning(f"Preprocessing failed, using original audio: {e}")

        # Use chunked diarization for long audio to prevent speaker drift
        if audio_duration > self.chunk_threshold_seconds:
            logger.info(
                f"Audio exceeds {self.chunk_threshold_seconds}s threshold, "
                f"using chunk-based diarization to prevent speaker drift"
            )
            return self._diarize_chunked(audio_data, segments, sample_rate)

        # Standard processing for shorter audio
        # Pass 0: Pre-process segments to split multi-speaker segments
        segments = self._preprocess_segments_for_multi_speaker(
            segments, audio_data, sample_rate
        )

        # Pass 1: Extract all embeddings
        embeddings_data = self._extract_all_embeddings(audio_data, segments, sample_rate)
        logger.info(f"Extracted {len(embeddings_data)} embeddings")

        if len(embeddings_data) < 2:
            # Not enough embeddings for clustering
            return self._handle_insufficient_embeddings(segments, embeddings_data)

        # Pass 1b: Global clustering
        cluster_labels = self._cluster_globally(embeddings_data)

        # Pass 1c: Assign deterministic speaker IDs (by first appearance)
        speaker_clusters = self._assign_speaker_ids(embeddings_data, cluster_labels, segments)

        # Build result
        result = self._build_result(segments, embeddings_data, speaker_clusters)

        # Update segments with speaker info
        segments = self._update_segments(segments, embeddings_data, speaker_clusters)

        # Interpolate gaps (short segments without embeddings)
        segments = self._interpolate_gaps(segments, speaker_clusters)

        # Merge sentence fragments from orphan speakers
        # This catches cases where a sentence is split mid-utterance
        # Scale orphan threshold with recording length to avoid absorbing
        # real speakers who happen to have few segments in longer recordings
        total_segments = len(segments)
        scaled_orphan_threshold = max(2, min(5, total_segments // 50))
        segments = self._merge_sentence_fragments(
            segments,
            max_orphan_segments=scaled_orphan_threshold,
            max_time_gap=3.0  # Merge if within 3 seconds of major speaker
        )

        logger.info(f"Batch diarization complete: {result.total_speakers} speakers")
        return segments, result

    def _diarize_chunked(
        self,
        audio_data: np.ndarray,
        segments: List[Dict[str, Any]],
        sample_rate: int
    ) -> Tuple[List[Dict[str, Any]], DiarizationResult]:
        """
        Perform chunk-based diarization for long audio.

        This prevents speaker drift that occurs over long recordings by:
        1. Splitting audio into overlapping chunks (e.g., 5 min with 30s overlap)
        2. Diarizing each chunk independently with fresh state
        3. Merging speaker clusters across chunks using centroid similarity

        Args:
            audio_data: Full audio samples
            segments: All transcript segments
            sample_rate: Audio sample rate

        Returns:
            Tuple of (updated_segments, DiarizationResult)
        """
        audio_duration = len(audio_data) / sample_rate
        chunk_duration = self.chunk_duration_seconds
        chunk_overlap = self.chunk_overlap_seconds

        # Calculate chunk boundaries
        chunk_boundaries = []
        chunk_start = 0.0
        while chunk_start < audio_duration:
            chunk_end = min(chunk_start + chunk_duration, audio_duration)
            chunk_boundaries.append((chunk_start, chunk_end))
            chunk_start = chunk_end - chunk_overlap
            if chunk_start >= audio_duration - chunk_overlap:
                break

        logger.info(
            f"Chunked diarization: {len(chunk_boundaries)} chunks of ~{chunk_duration}s "
            f"with {chunk_overlap}s overlap"
        )

        # Process each chunk independently
        chunk_results = []  # List of (chunk_idx, chunk_centroids, chunk_segment_speakers)

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            # Get audio for this chunk
            start_sample = int(chunk_start * sample_rate)
            end_sample = int(chunk_end * sample_rate)
            chunk_audio = audio_data[start_sample:end_sample]

            # Get segments within this chunk
            chunk_segments = []
            segment_indices = []
            for seg_idx, seg in enumerate(segments):
                # Include segment if it overlaps with chunk
                if seg['start'] < chunk_end and seg['end'] > chunk_start:
                    # Adjust timestamps relative to chunk start
                    adjusted_seg = seg.copy()
                    adjusted_seg['_original_start'] = seg['start']
                    adjusted_seg['_original_end'] = seg['end']
                    adjusted_seg['start'] = max(0, seg['start'] - chunk_start)
                    adjusted_seg['end'] = min(chunk_end - chunk_start, seg['end'] - chunk_start)
                    chunk_segments.append(adjusted_seg)
                    segment_indices.append(seg_idx)

            if not chunk_segments:
                continue

            logger.info(f"Chunk {chunk_idx + 1}/{len(chunk_boundaries)}: {len(chunk_segments)} segments")

            # Pre-process segments to split multi-speaker segments
            # Track mapping from split segments back to original indices
            original_segment_count = len(chunk_segments)
            chunk_segments = self._preprocess_segments_for_multi_speaker(
                chunk_segments, chunk_audio, sample_rate
            )

            # Build mapping from split segment index to original segment index
            split_to_original = {}
            for split_idx, seg in enumerate(chunk_segments):
                # If segment was split, it has '_original_segment_idx' from the split
                # But that's relative to pre-split chunk_segments, which maps to segment_indices
                orig_idx = seg.get('_original_segment_idx')
                if orig_idx is not None and orig_idx < len(segment_indices):
                    split_to_original[split_idx] = segment_indices[orig_idx]
                else:
                    # Not split - use the segment's _idx which was set before preprocessing
                    seg_idx = seg.get('_idx')
                    if seg_idx is not None and seg_idx < len(segment_indices):
                        split_to_original[split_idx] = segment_indices[seg_idx]

            # Extract embeddings for this chunk
            embeddings_data = self._extract_all_embeddings(chunk_audio, chunk_segments, sample_rate)

            if len(embeddings_data) < 2:
                # Not enough for clustering, assign all to single speaker
                for seg in chunk_segments:
                    seg['speaker_id'] = 'speaker_1'
                    seg['speaker_confidence'] = 0.5
                chunk_results.append({
                    'chunk_idx': chunk_idx,
                    'centroids': {},
                    'segment_speakers': {i: 'speaker_1' for i in segment_indices}
                })
                continue

            # Cluster this chunk
            cluster_labels = self._cluster_globally(embeddings_data)
            speaker_clusters = self._assign_speaker_ids(embeddings_data, cluster_labels, chunk_segments)

            # Compute centroids for merging
            centroids = {}
            for speaker_id, cluster in speaker_clusters.items():
                cluster.compute_centroid()
                if cluster.centroid is not None:
                    centroids[speaker_id] = cluster.centroid

            # Map segment indices to speaker IDs using split_to_original mapping
            segment_speakers = {}
            for data_idx, (emb_idx, emb, start, end, duration) in enumerate(embeddings_data):
                label = cluster_labels[data_idx]
                # Find speaker ID for this label
                for speaker_id, cluster in speaker_clusters.items():
                    if emb_idx in cluster.segment_indices:
                        # Map back to original segment index
                        original_idx = split_to_original.get(emb_idx)
                        if original_idx is not None:
                            segment_speakers[original_idx] = speaker_id
                        break

            chunk_results.append({
                'chunk_idx': chunk_idx,
                'centroids': centroids,
                'segment_speakers': segment_speakers,
                'embeddings_data': embeddings_data,
                'segment_indices': segment_indices
            })

        # Merge speaker clusters across chunks
        global_speaker_map = self._merge_chunk_speakers(chunk_results)

        # Apply global speaker IDs to all segments
        for seg in segments:
            seg['speaker_id'] = None
            seg['speaker_confidence'] = 0.0

        for chunk_result in chunk_results:
            for seg_idx, local_speaker in chunk_result['segment_speakers'].items():
                chunk_key = (chunk_result['chunk_idx'], local_speaker)
                global_speaker = global_speaker_map.get(chunk_key, local_speaker)
                segments[seg_idx]['speaker_id'] = global_speaker
                segments[seg_idx]['speaker_confidence'] = 0.7  # Chunked confidence

        # Interpolate gaps
        segments = self._interpolate_gaps_simple(segments)

        # Consolidate tiny speakers to reduce over-segmentation
        segments = self._consolidate_tiny_speakers(
            segments, chunk_results, global_speaker_map,
            min_segments=self.min_speaker_segments  # Configurable via MIN_SPEAKER_SEGMENTS
        )

        # Merge sentence fragments from orphan speakers
        # This catches cases where a sentence is split mid-utterance
        # Scale orphan threshold with recording length to avoid absorbing
        # real speakers who happen to have few segments in longer recordings
        total_segments = len(segments)
        scaled_orphan_threshold = max(2, min(5, total_segments // 50))
        segments = self._merge_sentence_fragments(
            segments,
            max_orphan_segments=scaled_orphan_threshold,
            max_time_gap=3.0  # Merge if within 3 seconds of major speaker
        )

        # Build speaker clusters with voice confidence from merged centroids
        speaker_clusters = self._build_speaker_clusters_from_chunks(
            segments, chunk_results, global_speaker_map
        )

        # Build result with voice confidence data
        speaker_ids = set(s.get('speaker_id') for s in segments if s.get('speaker_id'))
        result = DiarizationResult(
            speaker_clusters={c.speaker_id: c for c in speaker_clusters.values()},
            segment_assignments=[s.get('speaker_id', 'unknown') for s in segments],
            segment_confidences=[s.get('speaker_confidence', 0.0) for s in segments],
            total_speakers=len(speaker_ids),
            methodology_note=(
                f"Chunk-based diarization ({len(chunk_boundaries)} chunks) "
                f"to prevent speaker drift over {audio_duration:.0f}s audio"
            )
        )

        logger.info(
            f"Chunked diarization complete: {result.total_speakers} speakers "
            f"from {len(chunk_boundaries)} chunks"
        )
        return segments, result

    def _merge_chunk_speakers(
        self,
        chunk_results: List[Dict]
    ) -> Dict[Tuple[int, str], str]:
        """
        Merge speaker clusters across chunks using centroid similarity.

        Handles edge cases:
        - Speaker appears in chunk 1 and 3 but not 2: Clustering finds them
        - Two chunk speakers match same previous speaker: They remain separate
          if their centroids are different (distinct voices)
        - Short speakers with unreliable embeddings: Filtered by min_embedding_duration

        Args:
            chunk_results: List of chunk results with centroids

        Returns:
            Mapping from (chunk_idx, local_speaker_id) to global_speaker_id
        """
        if not chunk_results:
            return {}

        # Collect all centroids with their chunk/speaker info and speaking time
        all_centroids = []  # (chunk_idx, local_speaker_id, centroid, speaking_time)
        for cr in chunk_results:
            for speaker_id, centroid in cr['centroids'].items():
                # Calculate speaking time for this speaker in this chunk
                speaking_time = 0.0
                for seg_idx, spk in cr.get('segment_speakers', {}).items():
                    if spk == speaker_id:
                        speaking_time += 1.0  # Approximate
                all_centroids.append((cr['chunk_idx'], speaker_id, centroid, speaking_time))

        if len(all_centroids) < 2:
            # Not enough to merge
            return {(c[0], c[1]): c[1] for c in all_centroids}

        # Filter out speakers with very little speaking time (unreliable embeddings)
        min_speaking_time = 3.0  # At least 3 segments
        reliable_centroids = [c for c in all_centroids if c[3] >= min_speaking_time]
        unreliable_centroids = [c for c in all_centroids if c[3] < min_speaking_time]

        if len(reliable_centroids) < 2:
            # Fall back to using all centroids
            reliable_centroids = all_centroids
            unreliable_centroids = []

        logger.info(
            f"Cross-chunk merging: {len(reliable_centroids)} reliable speakers, "
            f"{len(unreliable_centroids)} short speakers"
        )

        # Cluster centroids across chunks
        centroid_matrix = np.array([c[2] for c in reliable_centroids])
        distances = cdist(centroid_matrix, centroid_matrix, metric='cosine')
        condensed = distances[np.triu_indices(len(reliable_centroids), k=1)]

        # Log distance statistics for debugging
        if len(condensed) > 0:
            logger.info(
                f"Cross-chunk distances: min={condensed.min():.3f}, "
                f"max={condensed.max():.3f}, mean={condensed.mean():.3f}"
            )

        # Use CONSERVATIVE threshold for cross-chunk merging
        # Balance between over-merging (too few speakers) and over-segmentation (too many)
        # 0.12 is conservative but allows same-voice matching across chunks
        merge_threshold = 0.12
        linkage_matrix = linkage(condensed, method='complete')  # Complete linkage is stricter
        labels = fcluster(linkage_matrix, t=merge_threshold, criterion='distance')

        num_clusters = len(set(labels))
        logger.info(f"Cross-chunk clustering: {num_clusters} clusters at threshold {merge_threshold}")

        # Build global speaker map
        # Group by cluster label, then assign global IDs by first appearance
        cluster_to_speakers = defaultdict(list)
        for idx, (chunk_idx, speaker_id, centroid, speaking_time) in enumerate(reliable_centroids):
            cluster_label = labels[idx]
            cluster_to_speakers[cluster_label].append((chunk_idx, speaker_id))

        # Assign global IDs ordered by first chunk appearance
        global_map = {}
        global_id = 1
        for cluster_label in sorted(cluster_to_speakers.keys()):
            speakers = sorted(cluster_to_speakers[cluster_label], key=lambda x: x[0])
            global_speaker_id = f"speaker_{global_id}"
            for chunk_idx, local_speaker in speakers:
                global_map[(chunk_idx, local_speaker)] = global_speaker_id
            global_id += 1

        # Handle unreliable speakers: match to nearest reliable speaker
        for chunk_idx, speaker_id, centroid, speaking_time in unreliable_centroids:
            # Find most similar reliable centroid
            best_match = None
            best_similarity = 0.0
            for r_chunk_idx, r_speaker_id, r_centroid, _ in reliable_centroids:
                similarity = 1.0 - cosine(centroid, r_centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (r_chunk_idx, r_speaker_id)

            if best_match and best_similarity > 0.7:
                global_map[(chunk_idx, speaker_id)] = global_map.get(best_match, f"speaker_{global_id}")
            else:
                # New speaker
                global_map[(chunk_idx, speaker_id)] = f"speaker_{global_id}"
                global_id += 1

        logger.info(
            f"Merged {len(all_centroids)} chunk speakers into {len(set(global_map.values()))} global speakers"
        )
        return global_map

    def _interpolate_gaps_simple(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Simple gap interpolation for segments without speaker IDs.

        Assigns unassigned segments to the nearest assigned speaker.
        """
        # Find segments without speaker IDs
        for i, seg in enumerate(segments):
            if seg.get('speaker_id') is None:
                # Look for nearest assigned segment
                prev_speaker = None
                next_speaker = None

                # Search backwards
                for j in range(i - 1, -1, -1):
                    if segments[j].get('speaker_id'):
                        prev_speaker = segments[j]['speaker_id']
                        break

                # Search forwards
                for j in range(i + 1, len(segments)):
                    if segments[j].get('speaker_id'):
                        next_speaker = segments[j]['speaker_id']
                        break

                # Assign to nearest (prefer previous)
                seg['speaker_id'] = prev_speaker or next_speaker or 'speaker_1'
                seg['speaker_confidence'] = 0.4  # Low confidence for interpolated

        return segments

    def _consolidate_tiny_speakers(
        self,
        segments: List[Dict[str, Any]],
        chunk_results: List[Dict],
        global_speaker_map: Dict[Tuple[int, str], str],
        min_segments: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Merge tiny speakers (<min_segments utterances) into their closest major speaker.

        This reduces over-segmentation by consolidating speakers who have too few
        utterances to be reliably distinct.

        Args:
            segments: List of segments with speaker_id assigned
            chunk_results: Original chunk results with centroids
            global_speaker_map: Mapping from chunk speakers to global speakers
            min_segments: Minimum segments to be considered a "major" speaker

        Returns:
            Updated segments with tiny speakers merged
        """
        from collections import Counter

        # Count segments per speaker
        speaker_counts = Counter(s.get('speaker_id') for s in segments if s.get('speaker_id'))

        # Identify major vs tiny speakers
        major_speakers = {spk for spk, count in speaker_counts.items() if count >= min_segments}
        tiny_speakers = {spk for spk, count in speaker_counts.items() if count < min_segments}

        if not tiny_speakers or not major_speakers:
            return segments

        logger.info(
            f"Consolidating {len(tiny_speakers)} tiny speakers (<{min_segments} segments) "
            f"into {len(major_speakers)} major speakers"
        )

        # Build centroid map for global speakers
        global_centroids = {}
        for cr in chunk_results:
            for local_speaker, centroid in cr.get('centroids', {}).items():
                chunk_key = (cr['chunk_idx'], local_speaker)
                global_speaker = global_speaker_map.get(chunk_key)
                if global_speaker and global_speaker not in global_centroids:
                    global_centroids[global_speaker] = centroid

        # Find best match for each tiny speaker
        merge_map = {}
        for tiny_spk in tiny_speakers:
            if tiny_spk not in global_centroids:
                # No centroid available - merge into most common major speaker
                merge_map[tiny_spk] = max(major_speakers, key=lambda s: speaker_counts.get(s, 0))
                continue

            tiny_centroid = global_centroids[tiny_spk]
            best_match = None
            best_similarity = 0.0

            for major_spk in major_speakers:
                if major_spk in global_centroids:
                    major_centroid = global_centroids[major_spk]
                    similarity = 1.0 - cosine(tiny_centroid, major_centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = major_spk

            if best_match and best_similarity > 0.6:  # Reasonable similarity threshold
                merge_map[tiny_spk] = best_match
                logger.debug(f"Merging {tiny_spk} -> {best_match} (similarity={best_similarity:.2f})")
            else:
                # Keep as separate speaker if no good match
                pass

        # Apply merges
        merged_count = 0
        for seg in segments:
            old_speaker = seg.get('speaker_id')
            if old_speaker in merge_map:
                seg['speaker_id'] = merge_map[old_speaker]
                seg['speaker_confidence'] *= 0.8  # Reduce confidence for merged
                merged_count += 1

        if merged_count > 0:
            # Renumber speakers to be consecutive
            segments = self._renumber_speakers(segments)
            new_count = len(set(s.get('speaker_id') for s in segments if s.get('speaker_id')))
            logger.info(f"Consolidated to {new_count} speakers ({merged_count} segments reassigned)")

        return segments

    def _merge_sentence_fragments(
        self,
        segments: List[Dict[str, Any]],
        max_orphan_segments: int = 2,
        max_time_gap: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Merge orphan speakers whose segments appear to be sentence fragments.

        This handles cases where a sentence gets split mid-utterance and the
        first part is assigned to a different speaker than the rest.

        Detection criteria:
        1. Speaker has very few segments (≤ max_orphan_segments)
        2. Segment ends without terminal punctuation (sentence fragment)
        3. Next segment from different speaker continues within time gap
        4. OR segment is temporally isolated and surrounded by one major speaker

        Args:
            segments: List of segments with speaker_id assigned
            max_orphan_segments: Max segments for a speaker to be considered orphan
            max_time_gap: Max seconds between segments to consider continuation

        Returns:
            Updated segments with sentence fragments merged
        """
        from collections import Counter
        import re

        if not segments:
            return segments

        # Sort by start time (check 'start' first, fall back to 'start_time')
        sorted_segments = sorted(
            segments,
            key=lambda s: s.get('start', s.get('start_time', 0))
        )

        # Count segments per speaker
        speaker_counts = Counter(s.get('speaker_id') for s in sorted_segments if s.get('speaker_id'))

        # Identify orphan vs major speakers
        orphan_speakers = {spk for spk, count in speaker_counts.items() if count <= max_orphan_segments}
        major_speakers = {spk for spk, count in speaker_counts.items() if count > max_orphan_segments}

        if not orphan_speakers or not major_speakers:
            return segments

        logger.info(
            f"Checking {len(orphan_speakers)} orphan speakers for sentence fragment merging"
        )

        # Patterns that indicate a complete sentence ending
        sentence_end_pattern = re.compile(r'[.!?][\s]*$')
        # Patterns that indicate a fragment (ends mid-thought)
        fragment_indicators = re.compile(r'\b(and|or|but|the|a|an|to|at|in|on|for|with|is|are|was|were|that|this|what|how|who|where|when|why)\s*$', re.I)

        def is_sentence_fragment(text: str) -> bool:
            """Check if text appears to be an incomplete sentence."""
            if not text or not text.strip():
                return False
            text = text.strip()
            # Ends with terminal punctuation = complete
            if sentence_end_pattern.search(text):
                return False
            # Ends with common continuation words = fragment
            if fragment_indicators.search(text):
                return True
            # Short text without punctuation = likely fragment
            if len(text.split()) <= 5 and not sentence_end_pattern.search(text):
                return True
            return False

        def _seg_start(s: Dict) -> float:
            """Get segment start time, checking both key conventions."""
            return s.get('start', s.get('start_time', 0))

        def _seg_end(s: Dict) -> float:
            """Get segment end time, checking both key conventions."""
            return s.get('end', s.get('end_time', 0))

        def get_nearby_major_speaker(idx: int, segments: List[Dict], major_speakers: set) -> Optional[str]:
            """Find the nearest major speaker to this segment."""
            current_time = _seg_start(segments[idx])

            # Look at previous and next segments
            prev_speaker = None
            next_speaker = None
            prev_time_gap = float('inf')
            next_time_gap = float('inf')

            # Check previous segments
            for i in range(idx - 1, max(0, idx - 5) - 1, -1):
                spk = segments[i].get('speaker_id')
                if spk in major_speakers:
                    prev_speaker = spk
                    prev_time_gap = current_time - _seg_end(segments[i])
                    break

            # Check next segments
            for i in range(idx + 1, min(len(segments), idx + 5)):
                spk = segments[i].get('speaker_id')
                if spk in major_speakers:
                    next_speaker = spk
                    next_time_gap = _seg_start(segments[i]) - current_time
                    break

            # Return the closer one if within time gap
            if prev_time_gap <= max_time_gap and prev_time_gap <= next_time_gap:
                return prev_speaker
            elif next_time_gap <= max_time_gap:
                return next_speaker
            elif prev_speaker and prev_time_gap <= max_time_gap * 2:
                return prev_speaker
            elif next_speaker and next_time_gap <= max_time_gap * 2:
                return next_speaker

            return None

        # Find segments to merge
        merge_count = 0
        for idx, seg in enumerate(sorted_segments):
            speaker = seg.get('speaker_id')
            if speaker not in orphan_speakers:
                continue

            text = seg.get('text', '')

            # Check if this is a sentence fragment
            is_fragment = is_sentence_fragment(text)

            # Check temporal context
            nearby_major = get_nearby_major_speaker(idx, sorted_segments, major_speakers)

            # Merge conditions:
            # 1. It's a fragment and there's a nearby major speaker
            # 2. OR it's temporally very close to a major speaker (< 1.5s)
            should_merge = False
            merge_reason = ""

            if is_fragment and nearby_major:
                should_merge = True
                merge_reason = "sentence fragment"
            elif nearby_major:
                # Check if very close temporally
                current_time = _seg_start(seg)
                for i in range(max(0, idx - 1), min(len(sorted_segments), idx + 2)):
                    if i == idx:
                        continue
                    other = sorted_segments[i]
                    if other.get('speaker_id') == nearby_major:
                        time_gap = abs(_seg_start(other) - current_time)
                        if time_gap < 1.5:
                            should_merge = True
                            merge_reason = f"temporal proximity ({time_gap:.1f}s)"
                            break

            if should_merge and nearby_major:
                old_speaker = seg['speaker_id']
                seg['speaker_id'] = nearby_major
                seg['speaker_confidence'] = seg.get('speaker_confidence', 0.7) * 0.85
                seg['merge_reason'] = merge_reason
                merge_count += 1
                logger.debug(
                    f"Merged orphan segment '{text[:40]}...' from {old_speaker} -> {nearby_major} ({merge_reason})"
                )

        if merge_count > 0:
            # Renumber speakers to be consecutive
            sorted_segments = self._renumber_speakers(sorted_segments)
            new_count = len(set(s.get('speaker_id') for s in sorted_segments if s.get('speaker_id')))
            logger.info(
                f"Sentence fragment merging: {merge_count} segments reassigned, "
                f"now {new_count} speakers"
            )

        return sorted_segments

    def _build_speaker_clusters_from_chunks(
        self,
        segments: List[Dict[str, Any]],
        chunk_results: List[Dict],
        global_speaker_map: Dict[Tuple[int, str], str]
    ) -> Dict[str, SpeakerCluster]:
        """
        Build speaker clusters with voice confidence from merged chunk data.

        This aggregates embeddings across chunks for each global speaker
        and computes cluster tightness (voice confidence).

        Args:
            segments: Segments with global speaker IDs
            chunk_results: Chunk results with centroids and embeddings
            global_speaker_map: Mapping from chunk speakers to global speakers

        Returns:
            Dictionary mapping speaker_id to SpeakerCluster with voice confidence
        """
        from collections import defaultdict

        # Collect all centroids for each global speaker
        speaker_centroids: Dict[str, List[np.ndarray]] = defaultdict(list)

        for cr in chunk_results:
            for local_speaker, centroid in cr.get('centroids', {}).items():
                chunk_key = (cr['chunk_idx'], local_speaker)
                global_speaker = global_speaker_map.get(chunk_key)
                if global_speaker and centroid is not None:
                    speaker_centroids[global_speaker].append(centroid)

        # Build speaker clusters
        speaker_clusters = {}

        for speaker_id in set(s.get('speaker_id') for s in segments if s.get('speaker_id')):
            centroids = speaker_centroids.get(speaker_id, [])

            # Get segment info for this speaker
            speaker_segments = [s for s in segments if s.get('speaker_id') == speaker_id]
            segment_indices = [i for i, s in enumerate(segments) if s.get('speaker_id') == speaker_id]

            # Compute voice confidence from centroid consistency
            if len(centroids) >= 2:
                # Multiple chunk centroids - compute consistency
                centroid_matrix = np.array(centroids)
                mean_centroid = np.mean(centroid_matrix, axis=0)

                # Compute average similarity to mean centroid
                similarities = []
                for c in centroids:
                    sim = 1.0 - cosine(c, mean_centroid)
                    similarities.append(sim)
                cluster_tightness = float(np.mean(similarities))
            elif len(centroids) == 1:
                # Single centroid - high confidence
                cluster_tightness = 0.85
                mean_centroid = centroids[0]
            else:
                # No centroid data - moderate confidence
                cluster_tightness = 0.7
                mean_centroid = None

            # Calculate first appearance and total duration
            first_appearance = min(s.get('start', 0) for s in speaker_segments) if speaker_segments else 0.0
            total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in speaker_segments)

            cluster = SpeakerCluster(
                speaker_id=speaker_id,
                embeddings=centroids,  # Store centroids as embeddings
                segment_indices=segment_indices,
                first_appearance=first_appearance,
                total_duration=total_duration,
                centroid=mean_centroid,
                cluster_tightness=cluster_tightness
            )

            speaker_clusters[speaker_id] = cluster

            # Update segment confidence with cluster tightness
            for seg in speaker_segments:
                seg['speaker_confidence'] = cluster_tightness

        logger.info(
            f"Built {len(speaker_clusters)} speaker clusters with voice confidence "
            f"(avg tightness: {np.mean([c.cluster_tightness for c in speaker_clusters.values()]):.2f})"
        )

        return speaker_clusters

    def _renumber_speakers(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Renumber speakers to be consecutive (speaker_1, speaker_2, etc.)."""
        # Get unique speakers in order of first appearance
        speaker_order = []
        seen = set()
        for seg in segments:
            spk = seg.get('speaker_id')
            if spk and spk not in seen:
                speaker_order.append(spk)
                seen.add(spk)

        # Create renumbering map
        renumber_map = {old: f"speaker_{i+1}" for i, old in enumerate(speaker_order)}

        # Apply renumbering
        for seg in segments:
            old_speaker = seg.get('speaker_id')
            if old_speaker in renumber_map:
                seg['speaker_id'] = renumber_map[old_speaker]

        return segments

    def _extract_all_embeddings(
        self,
        audio_data: np.ndarray,
        segments: List[Dict[str, Any]],
        sample_rate: int
    ) -> List[Tuple[int, np.ndarray, float, float, float]]:
        """
        Extract embeddings for all segments that meet duration threshold.

        Returns:
            List of tuples: (segment_index, embedding, start_time, end_time, duration)
        """
        embeddings_data = []

        for idx, seg in enumerate(segments):
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            duration = seg['end'] - seg['start']

            if len(segment_audio) == 0:
                continue

            if duration >= self.min_embedding_duration:
                embedding = self._get_embedding(segment_audio, sample_rate)
                if embedding is not None:
                    embeddings_data.append((
                        idx, embedding, seg['start'], seg['end'], duration
                    ))

        return embeddings_data

    def _get_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio segment."""
        try:
            # Resample if needed
            if sample_rate != self.sample_rate:
                ratio = self.sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)

            # Preprocess for resemblyzer (expects float64)
            audio_data = audio_data.astype(np.float64)

            # Get embedding
            embedding = self.encoder.embed_utterance(audio_data)
            return embedding

        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None

    def _detect_pauses(
        self,
        audio_data: np.ndarray,
        seg_start: float,
        seg_end: float,
        sample_rate: int
    ) -> List[float]:
        """
        Detect pauses within a segment using energy analysis.

        Finds brief silences that likely indicate speaker boundaries.
        This is critical for rapid-fire bridge communications where
        speakers hand off quickly with minimal pauses.

        Args:
            audio_data: Full audio samples
            seg_start: Segment start time in seconds
            seg_end: Segment end time in seconds
            sample_rate: Audio sample rate

        Returns:
            List of pause midpoint timestamps (potential speaker boundaries)
        """
        start_sample = int(seg_start * sample_rate)
        end_sample = int(seg_end * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]

        if len(segment_audio) == 0:
            return []

        # Analyze energy in small frames (20ms)
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop

        pause_regions = []
        in_pause = False
        pause_start = None

        pos = 0
        while pos + frame_size <= len(segment_audio):
            frame = segment_audio[pos:pos + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))

            current_time = seg_start + (pos / sample_rate)

            if rms < self.pause_energy_threshold:
                if not in_pause:
                    in_pause = True
                    pause_start = current_time
            else:
                if in_pause:
                    pause_end = current_time
                    pause_duration = pause_end - pause_start

                    # Only consider pauses longer than minimum
                    if pause_duration >= self.min_pause_duration_seconds:
                        # Use pause midpoint as potential speaker boundary
                        pause_mid = (pause_start + pause_end) / 2
                        pause_regions.append(pause_mid)
                        logger.debug(
                            f"Pause detected: {pause_start:.2f}s - {pause_end:.2f}s "
                            f"(duration={pause_duration:.3f}s)"
                        )

                    in_pause = False
                    pause_start = None

            pos += hop_size

        return pause_regions

    def _extract_subsegment_embeddings(
        self,
        audio_data: np.ndarray,
        seg_start: float,
        seg_end: float,
        sample_rate: int,
        pause_points: Optional[List[float]] = None
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract multiple embeddings from a long segment.

        Uses a hybrid approach:
        1. If pause points are provided, extract embeddings for each region between pauses
        2. Fall back to sliding window for regions without detected pauses

        This handles segments that may contain multiple speakers by extracting
        embeddings at natural speech boundaries.

        Args:
            audio_data: Full audio samples
            seg_start: Segment start time in seconds
            seg_end: Segment end time in seconds
            sample_rate: Audio sample rate
            pause_points: Optional list of pause timestamps from _detect_pauses

        Returns:
            List of (subseg_start, subseg_end, embedding) tuples
        """
        subsegments = []

        start_sample = int(seg_start * sample_rate)
        end_sample = int(seg_end * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]

        # If we have pause points, use them to define regions
        if pause_points:
            # Create boundaries from pause points
            boundaries = [seg_start] + sorted(pause_points) + [seg_end]

            for i in range(len(boundaries) - 1):
                region_start = boundaries[i]
                region_end = boundaries[i + 1]
                region_duration = region_end - region_start

                # Skip very short regions
                if region_duration < self.min_embedding_duration:
                    continue

                # Extract audio for this region
                region_start_sample = int((region_start - seg_start) * sample_rate)
                region_end_sample = int((region_end - seg_start) * sample_rate)
                region_audio = segment_audio[region_start_sample:region_end_sample]

                if len(region_audio) < int(self.min_embedding_duration * sample_rate):
                    continue

                # For long regions, use sliding window; for short ones, single embedding
                if region_duration > self.subsegment_window_seconds * 1.5:
                    # Sliding window within this region
                    region_subsegments = self._sliding_window_embeddings(
                        region_audio, region_start, sample_rate
                    )
                    subsegments.extend(region_subsegments)
                else:
                    # Single embedding for this region
                    embedding = self._get_embedding(region_audio, sample_rate)
                    if embedding is not None:
                        subsegments.append((region_start, region_end, embedding))

            if subsegments:
                return subsegments

        # Fall back to pure sliding window if no pause points or no valid regions
        return self._sliding_window_embeddings(segment_audio, seg_start, sample_rate)

    def _sliding_window_embeddings(
        self,
        segment_audio: np.ndarray,
        seg_start: float,
        sample_rate: int
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract embeddings using sliding window approach.

        Args:
            segment_audio: Audio samples for this segment
            seg_start: Start time of this segment
            sample_rate: Audio sample rate

        Returns:
            List of (subseg_start, subseg_end, embedding) tuples
        """
        subsegments = []
        window_samples = int(self.subsegment_window_seconds * sample_rate)
        hop_samples = int(self.subsegment_hop_seconds * sample_rate)

        if len(segment_audio) < window_samples:
            # Segment too short for windowing, extract single embedding
            embedding = self._get_embedding(segment_audio, sample_rate)
            if embedding is not None:
                seg_end = seg_start + (len(segment_audio) / sample_rate)
                subsegments.append((seg_start, seg_end, embedding))
            return subsegments

        # Sliding window extraction
        pos = 0
        while pos + window_samples <= len(segment_audio):
            window_audio = segment_audio[pos:pos + window_samples]
            embedding = self._get_embedding(window_audio, sample_rate)

            if embedding is not None:
                subseg_start = seg_start + (pos / sample_rate)
                subseg_end = seg_start + ((pos + window_samples) / sample_rate)
                subsegments.append((subseg_start, subseg_end, embedding))

            pos += hop_samples

        # Handle remaining audio at the end if significant
        remaining = len(segment_audio) - pos
        if remaining >= self.min_embedding_duration * sample_rate:
            window_audio = segment_audio[pos:]
            embedding = self._get_embedding(window_audio, sample_rate)
            if embedding is not None:
                subseg_start = seg_start + (pos / sample_rate)
                subseg_end = seg_start + (len(segment_audio) / sample_rate)
                subsegments.append((subseg_start, subseg_end, embedding))

        return subsegments

    def _detect_speaker_changes(
        self,
        subsegments: List[Tuple[float, float, np.ndarray]]
    ) -> List[float]:
        """
        Detect speaker change points within a segment's sub-embeddings.

        Compares consecutive embeddings and marks significant drops in
        similarity as speaker change points.

        Args:
            subsegments: List of (start, end, embedding) tuples

        Returns:
            List of timestamps where speaker changes are detected
        """
        if len(subsegments) < 2:
            return []

        change_points = []

        for i in range(1, len(subsegments)):
            prev_emb = subsegments[i - 1][2]
            curr_emb = subsegments[i][2]

            # Calculate similarity between consecutive windows
            similarity = 1.0 - cosine(prev_emb, curr_emb)

            # If similarity drops below threshold, mark as speaker change
            if similarity < self.subsegment_speaker_threshold:
                # Speaker change at the boundary between windows
                change_time = subsegments[i][0]  # Start of new speaker
                change_points.append(change_time)
                logger.debug(
                    f"Speaker change detected at {change_time:.2f}s "
                    f"(similarity={similarity:.3f})"
                )

        return change_points

    def _snap_to_word_boundary(
        self,
        change_point: float,
        words: List[Dict[str, Any]],
        seg_start: float,
        seg_end: float
    ) -> float:
        """
        Snap a change point to the nearest word boundary.

        Uses word end times to find natural breaks between speakers.
        This prevents cutting words in half.

        Args:
            change_point: Original change point timestamp
            words: List of word dicts with 'start', 'end', 'word' keys
            seg_start: Segment start time
            seg_end: Segment end time

        Returns:
            Adjusted change point at a word boundary
        """
        if not words:
            return change_point

        # Find the word that contains or is closest to the change point
        best_boundary = change_point
        min_distance = float('inf')

        for word in words:
            word_end = word.get('end', 0)

            # Consider word end times as potential boundaries
            if seg_start < word_end < seg_end:
                distance = abs(word_end - change_point)
                if distance < min_distance:
                    min_distance = distance
                    best_boundary = word_end

        # Only snap if we found a nearby word boundary (within 0.5s)
        if min_distance < 0.5:
            logger.debug(
                f"Snapped change point {change_point:.2f}s -> {best_boundary:.2f}s "
                f"(distance={min_distance:.3f}s)"
            )
            return best_boundary

        return change_point

    def _split_segment_at_changes(
        self,
        segment: Dict[str, Any],
        change_points: List[float],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Split a segment at detected speaker change points.

        Uses word timestamps to:
        1. Snap change points to natural word boundaries
        2. Accurately assign words to each split segment

        Args:
            segment: Original segment dict with 'start', 'end', 'text', optionally 'words'
            change_points: List of timestamps where speakers change
            audio_data: Full audio samples
            sample_rate: Audio sample rate

        Returns:
            List of new segment dicts, one per detected speaker
        """
        if not change_points:
            return [segment]

        # Sort change points and filter to segment boundaries
        seg_start = segment['start']
        seg_end = segment['end']
        valid_changes = sorted([
            cp for cp in change_points
            if seg_start < cp < seg_end
        ])

        if not valid_changes:
            return [segment]

        # Get word timestamps if available
        words = segment.get('words', [])

        # Snap change points to word boundaries if we have word timestamps
        if words:
            snapped_changes = []
            for cp in valid_changes:
                snapped = self._snap_to_word_boundary(cp, words, seg_start, seg_end)
                snapped_changes.append(snapped)
            # Remove duplicates and re-sort
            valid_changes = sorted(set(snapped_changes))

        # Create boundaries: [seg_start, change1, change2, ..., seg_end]
        boundaries = [seg_start] + valid_changes + [seg_end]

        new_segments = []

        for i in range(len(boundaries) - 1):
            split_start = boundaries[i]
            split_end = boundaries[i + 1]

            # Skip very short segments
            if split_end - split_start < 0.3:
                continue

            # Create new segment
            new_seg = {
                'start': split_start,
                'end': split_end,
                '_original_segment_idx': segment.get('_idx'),
                '_is_split': True
            }

            # Split text based on word timestamps or time proportion
            if words:
                # Use word timestamps for accurate text splitting
                # A word belongs to this segment if its midpoint is within the segment
                split_words = []
                for w in words:
                    word_start = w.get('start', 0)
                    word_end = w.get('end', 0)
                    word_mid = (word_start + word_end) / 2

                    # Word belongs to this segment if its midpoint is within bounds
                    if split_start <= word_mid < split_end:
                        split_words.append(w)

                new_seg['text'] = ' '.join(w.get('word', '') for w in split_words).strip()
                new_seg['words'] = split_words

                # Calculate confidence from word probabilities
                if split_words:
                    probs = [w.get('probability', 0) for w in split_words if w.get('probability')]
                    new_seg['confidence'] = sum(probs) / len(probs) if probs else 0
                else:
                    new_seg['confidence'] = 0

                # Update segment boundaries to match actual word boundaries
                if split_words:
                    new_seg['start'] = split_words[0].get('start', split_start)
                    new_seg['end'] = split_words[-1].get('end', split_end)
            else:
                # Approximate text split by time proportion
                full_text = segment.get('text', '')
                full_duration = seg_end - seg_start
                if full_duration > 0:
                    start_ratio = (split_start - seg_start) / full_duration
                    end_ratio = (split_end - seg_start) / full_duration

                    # Split words approximately
                    text_words = full_text.split()
                    start_word_idx = int(len(text_words) * start_ratio)
                    end_word_idx = max(start_word_idx + 1, int(len(text_words) * end_ratio))
                    new_seg['text'] = ' '.join(text_words[start_word_idx:end_word_idx]).strip()
                else:
                    new_seg['text'] = full_text if i == 0 else ''

                new_seg['confidence'] = segment.get('confidence', 0)

            # Only add if there's actual content
            if new_seg.get('text') or (new_seg['end'] - new_seg['start']) >= 0.5:
                new_segments.append(new_seg)

        logger.info(
            f"Split segment [{seg_start:.1f}s-{seg_end:.1f}s] into "
            f"{len(new_segments)} parts at {len(valid_changes)} change points"
        )

        return new_segments

    def _preprocess_segments_for_multi_speaker(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Pre-process segments to split those containing multiple speakers.

        Long segments are analyzed for speaker changes and split accordingly.
        This ensures each segment passed to clustering contains only one speaker.

        Args:
            segments: Original transcription segments
            audio_data: Full audio samples
            sample_rate: Audio sample rate

        Returns:
            Processed segments, potentially more numerous than input
        """
        processed_segments = []

        for idx, seg in enumerate(segments):
            duration = seg['end'] - seg['start']
            seg['_idx'] = idx  # Track original index

            # Short segments: keep as-is
            if duration < self.subsegment_threshold_seconds:
                processed_segments.append(seg)
                continue

            # Long segment: check for multiple speakers
            logger.debug(
                f"Analyzing long segment [{seg['start']:.1f}s-{seg['end']:.1f}s] "
                f"for multiple speakers"
            )

            # Step 1: Detect pauses (energy-based) - these are strong speaker boundaries
            pause_points = self._detect_pauses(
                audio_data, seg['start'], seg['end'], sample_rate
            )

            # Step 2: Extract sub-segment embeddings (using pause points if available)
            subsegments = self._extract_subsegment_embeddings(
                audio_data, seg['start'], seg['end'], sample_rate, pause_points
            )

            if len(subsegments) < 2:
                # Not enough sub-segments for analysis
                processed_segments.append(seg)
                continue

            # Step 3: Detect speaker changes based on embedding similarity
            embedding_change_points = self._detect_speaker_changes(subsegments)

            # Step 4: Combine pause-based and embedding-based change points
            # Only use pause points that are confirmed by embedding changes (within 0.5s)
            confirmed_pauses = []
            for pause in pause_points:
                for emb_change in embedding_change_points:
                    if abs(pause - emb_change) < 0.5:
                        confirmed_pauses.append(pause)
                        break

            # Also include strong embedding changes not near pauses
            all_change_points = set(confirmed_pauses) | set(embedding_change_points)
            change_points = sorted(all_change_points)

            # Step 5: Filter change points to ensure minimum segment duration
            if change_points:
                filtered_changes = []
                boundaries = [seg['start']] + change_points + [seg['end']]

                for i in range(1, len(boundaries) - 1):
                    prev_boundary = filtered_changes[-1] if filtered_changes else seg['start']
                    next_boundary = boundaries[i + 1] if i + 1 < len(boundaries) else seg['end']
                    current = boundaries[i]

                    # Check if this change would create segments >= min duration
                    before_duration = current - prev_boundary
                    after_duration = next_boundary - current

                    if before_duration >= self.min_split_segment_duration and \
                       after_duration >= self.min_split_segment_duration:
                        filtered_changes.append(current)

                change_points = filtered_changes

            if not change_points:
                # No valid speaker changes after filtering
                processed_segments.append(seg)
                continue

            logger.debug(
                f"Found {len(pause_points)} pauses, {len(embedding_change_points)} embedding changes, "
                f"{len(change_points)} after filtering for min duration"
            )

            # Split segment at change points
            split_segments = self._split_segment_at_changes(
                seg, change_points, audio_data, sample_rate
            )
            processed_segments.extend(split_segments)

        original_count = len(segments)
        new_count = len(processed_segments)

        if new_count > original_count:
            logger.info(
                f"Multi-speaker preprocessing: {original_count} segments → "
                f"{new_count} segments ({new_count - original_count} splits)"
            )

        return processed_segments

    def _cluster_globally(
        self,
        embeddings_data: List[Tuple]
    ) -> np.ndarray:
        """
        Perform hierarchical clustering on all embeddings at once.

        This ensures clustering decisions use complete information,
        not incremental state.
        """
        n = len(embeddings_data)
        if n < 2:
            return np.array([1])

        # Extract embeddings into matrix
        embeddings_matrix = np.array([e[1] for e in embeddings_data])

        # Calculate pairwise cosine distances
        distances = cdist(embeddings_matrix, embeddings_matrix, metric='cosine')
        condensed = distances[np.triu_indices(n, k=1)]

        # Hierarchical clustering
        linkage_matrix = linkage(condensed, method='average')

        # Log distance statistics
        logger.info(
            f"Embedding distances: min={condensed.min():.3f}, "
            f"max={condensed.max():.3f}, mean={condensed.mean():.3f}"
        )

        # Cluster using distance threshold
        distance_threshold = 1.0 - self.similarity_threshold
        labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

        num_clusters = len(np.unique(labels))
        logger.info(f"Initial clustering: {num_clusters} clusters")

        # Enforce speaker limits
        if num_clusters > self.max_speakers:
            labels = fcluster(linkage_matrix, t=self.max_speakers, criterion='maxclust')
            logger.info(f"Reduced to {self.max_speakers} clusters (max limit)")
        elif num_clusters < self.min_speakers and n >= self.min_speakers:
            labels = fcluster(linkage_matrix, t=self.min_speakers, criterion='maxclust')
            logger.info(f"Forced to {self.min_speakers} clusters (min limit)")

        return labels

    def _assign_speaker_ids(
        self,
        embeddings_data: List[Tuple],
        cluster_labels: np.ndarray,
        segments: List[Dict[str, Any]]
    ) -> Dict[int, SpeakerCluster]:
        """
        Assign deterministic speaker IDs based on first appearance.

        Key insight: Speaker IDs are assigned by earliest appearance
        in the audio, not by cluster number. This ensures:
        - First voice detected = speaker_1
        - Second voice detected = speaker_2
        - Same physical voice always gets same ID
        """
        # Group embeddings by cluster
        cluster_data: Dict[int, List[Tuple[int, np.ndarray, float, float, float]]] = defaultdict(list)
        for i, (idx, emb, start, end, duration) in enumerate(embeddings_data):
            cluster = cluster_labels[i]
            cluster_data[cluster].append((idx, emb, start, end, duration))

        # Find first appearance time for each cluster
        cluster_first_appearance = {}
        for cluster, data in cluster_data.items():
            first_time = min(d[2] for d in data)  # start time
            cluster_first_appearance[cluster] = first_time

        # Sort clusters by first appearance
        sorted_clusters = sorted(
            cluster_data.keys(),
            key=lambda c: cluster_first_appearance[c]
        )

        # Create speaker clusters with deterministic IDs
        speaker_clusters: Dict[int, SpeakerCluster] = {}
        for speaker_num, cluster in enumerate(sorted_clusters, 1):
            speaker_id = f"speaker_{speaker_num}"
            data = cluster_data[cluster]

            cluster_obj = SpeakerCluster(
                speaker_id=speaker_id,
                embeddings=[d[1] for d in data],
                segment_indices=[d[0] for d in data],
                first_appearance=cluster_first_appearance[cluster],
                total_duration=sum(d[4] for d in data)
            )
            cluster_obj.compute_centroid()
            cluster_obj.compute_tightness()

            speaker_clusters[cluster] = cluster_obj

        logger.info(
            f"Speaker ID assignment: {len(speaker_clusters)} speakers, "
            f"ordered by first appearance"
        )

        return speaker_clusters

    def _build_result(
        self,
        segments: List[Dict[str, Any]],
        embeddings_data: List[Tuple],
        speaker_clusters: Dict[int, SpeakerCluster]
    ) -> DiarizationResult:
        """Build the complete diarization result."""
        # Create speaker_id -> cluster mapping
        speaker_cluster_map = {
            cluster.speaker_id: cluster
            for cluster in speaker_clusters.values()
        }

        # Build segment assignments
        segment_assignments = ['unknown'] * len(segments)
        segment_confidences = [0.0] * len(segments)

        for cluster in speaker_clusters.values():
            for idx in cluster.segment_indices:
                segment_assignments[idx] = cluster.speaker_id
                segment_confidences[idx] = cluster.cluster_tightness

        methodology = (
            f"Two-pass batch diarization: {len(embeddings_data)} embeddings clustered "
            f"into {len(speaker_clusters)} speakers using hierarchical clustering "
            f"(threshold={self.similarity_threshold:.2f}). Speaker IDs assigned by "
            f"first appearance in audio for consistency across clip lengths."
        )

        return DiarizationResult(
            speaker_clusters=speaker_cluster_map,
            segment_assignments=segment_assignments,
            segment_confidences=segment_confidences,
            total_speakers=len(speaker_clusters),
            methodology_note=methodology
        )

    def _update_segments(
        self,
        segments: List[Dict[str, Any]],
        embeddings_data: List[Tuple],
        speaker_clusters: Dict[int, SpeakerCluster]
    ) -> List[Dict[str, Any]]:
        """Update segment dictionaries with speaker info."""
        # Build index -> cluster mapping
        segment_to_cluster = {}
        for cluster_id, cluster in speaker_clusters.items():
            for idx in cluster.segment_indices:
                segment_to_cluster[idx] = cluster

        # Update segments
        for idx, seg in enumerate(segments):
            if idx in segment_to_cluster:
                cluster = segment_to_cluster[idx]
                seg['speaker_id'] = cluster.speaker_id
                seg['speaker_confidence'] = cluster.cluster_tightness
            # Segments without embeddings handled by interpolation

        return segments

    def _interpolate_gaps(
        self,
        segments: List[Dict[str, Any]],
        speaker_clusters: Dict[int, SpeakerCluster]
    ) -> List[Dict[str, Any]]:
        """
        Handle short segments that couldn't get embeddings.

        Uses temporal context to assign speaker IDs to gaps.
        """
        for idx, seg in enumerate(segments):
            if seg.get('speaker_id'):
                continue  # Already assigned

            # Look for neighboring speakers
            prev_speaker = None
            next_speaker = None

            # Search backward
            for i in range(idx - 1, -1, -1):
                if segments[i].get('speaker_id'):
                    prev_speaker = segments[i]['speaker_id']
                    break

            # Search forward
            for i in range(idx + 1, len(segments)):
                if segments[i].get('speaker_id'):
                    next_speaker = segments[i]['speaker_id']
                    break

            # Assign based on neighbors
            if prev_speaker and next_speaker:
                if prev_speaker == next_speaker:
                    seg['speaker_id'] = prev_speaker
                    seg['speaker_confidence'] = 0.7  # Interpolated
                else:
                    # Between two different speakers - use previous
                    seg['speaker_id'] = prev_speaker
                    seg['speaker_confidence'] = 0.5  # Uncertain
            elif prev_speaker:
                seg['speaker_id'] = prev_speaker
                seg['speaker_confidence'] = 0.6
            elif next_speaker:
                seg['speaker_id'] = next_speaker
                seg['speaker_confidence'] = 0.6
            else:
                # No neighbors found - shouldn't happen often
                seg['speaker_id'] = 'speaker_1'
                seg['speaker_confidence'] = 0.3

        return segments

    def _handle_insufficient_embeddings(
        self,
        segments: List[Dict[str, Any]],
        embeddings_data: List[Tuple]
    ) -> Tuple[List[Dict[str, Any]], DiarizationResult]:
        """Handle case with fewer than 2 embeddings."""
        if not embeddings_data:
            # No embeddings at all - assign all to speaker_1
            for seg in segments:
                seg['speaker_id'] = 'speaker_1'
                seg['speaker_confidence'] = 0.3

            cluster = SpeakerCluster(
                speaker_id='speaker_1',
                segment_indices=list(range(len(segments))),
                first_appearance=segments[0]['start'] if segments else 0.0,
                cluster_tightness=0.3
            )

            return segments, DiarizationResult(
                speaker_clusters={'speaker_1': cluster},
                segment_assignments=['speaker_1'] * len(segments),
                segment_confidences=[0.3] * len(segments),
                total_speakers=1,
                methodology_note="Insufficient embeddings for clustering; single speaker assumed."
            )

        # Single embedding - assign to speaker_1
        idx, emb, start, end, duration = embeddings_data[0]

        cluster = SpeakerCluster(
            speaker_id='speaker_1',
            embeddings=[emb],
            segment_indices=[idx],
            first_appearance=start,
            total_duration=duration,
            cluster_tightness=1.0
        )
        cluster.compute_centroid()

        # Assign all segments to speaker_1
        for seg in segments:
            seg['speaker_id'] = 'speaker_1'
            seg['speaker_confidence'] = 0.5

        segments[idx]['speaker_confidence'] = 1.0

        return segments, DiarizationResult(
            speaker_clusters={'speaker_1': cluster},
            segment_assignments=['speaker_1'] * len(segments),
            segment_confidences=[0.5] * len(segments),
            total_speakers=1,
            methodology_note="Single embedding found; single speaker assumed."
        )

    def reset(self):
        """
        Reset is a no-op for stateless batch diarizer.

        Unlike incremental diarizers, this class maintains no state
        between calls to diarize_complete().
        """
        pass  # Intentionally empty - stateless design


def is_batch_diarizer_available() -> bool:
    """Check if batch diarization dependencies are available."""
    return RESEMBLYZER_AVAILABLE and SCIPY_AVAILABLE
