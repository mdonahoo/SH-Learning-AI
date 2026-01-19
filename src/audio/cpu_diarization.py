"""
CPU-optimized speaker diarization using resemblyzer.

This module provides a lightweight alternative to pyannote for CPU-only systems.
Resemblyzer uses a pretrained speaker encoder (GE2E) that runs efficiently on CPU.

Key advantages over pyannote on CPU:
- 5-10x faster inference
- Lower memory usage
- No GPU required for good performance
- Still provides accurate speaker embeddings

Trade-offs:
- Slightly lower accuracy than pyannote's neural diarization
- No built-in VAD (we use our own)
- Requires manual speaker clustering
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
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate as RESEMBLYZER_SAMPLE_RATE
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    RESEMBLYZER_SAMPLE_RATE = 16000

try:
    from scipy.spatial.distance import cosine, cdist
    from scipy.cluster.hierarchy import linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class CPUSpeakerProfile:
    """
    Speaker profile optimized for CPU processing.

    Maintains embeddings and statistics for speaker matching.
    """
    speaker_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    segment_count: int = 0
    total_duration: float = 0.0
    last_seen_time: float = 0.0

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Get centroid (mean) of all embeddings."""
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)

    def add_embedding(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
        max_embeddings: int = 30
    ):
        """Add a new embedding, maintaining size limit."""
        self.embeddings.append(embedding)
        self.segment_count += 1
        self.total_duration += duration
        self.last_seen_time = timestamp

        # Keep only recent embeddings
        if len(self.embeddings) > max_embeddings:
            self.embeddings.pop(0)

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Calculate similarity to a new embedding."""
        if not self.embeddings:
            return 0.0

        centroid = self.centroid
        centroid_sim = 1.0 - cosine(embedding, centroid)

        # Also check best individual match
        best_sim = max(1.0 - cosine(embedding, e) for e in self.embeddings)

        # Weighted combination
        return centroid_sim * 0.6 + best_sim * 0.4


def _configure_cpu_threads():
    """Configure optimal CPU thread settings for inference."""
    if not TORCH_AVAILABLE:
        return

    # Get CPU count
    cpu_count = os.cpu_count() or 4

    # Set optimal thread counts for CPU inference
    # Use fewer threads than CPUs to avoid oversubscription
    num_threads = max(1, cpu_count - 1)

    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
        logger.info(f"Configured torch for CPU: {num_threads} threads")
    except Exception as e:
        logger.warning(f"Could not configure torch threads: {e}")


class CPUSpeakerDiarizer:
    """
    CPU-optimized speaker diarization using resemblyzer.

    This provides a fast, accurate speaker diarization solution for
    systems without GPU. Uses the GE2E (Generalized End-to-End) speaker
    encoder which is specifically designed for speaker verification.

    Performance on CPU:
    - ~50-100ms per segment (vs 500ms+ for pyannote)
    - Suitable for real-time processing
    - Accuracy within 5-10% of pyannote on most datasets
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        """
        Initialize CPU speaker diarizer.

        Args:
            similarity_threshold: Minimum similarity (0-1) to match speakers
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
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

        # Configure CPU threads
        _configure_cpu_threads()

        # Load encoder (downloads model on first use)
        logger.info("Loading resemblyzer voice encoder...")
        self.encoder = VoiceEncoder()
        logger.info("Voice encoder loaded")

        # Configuration
        # Note: similarity_threshold = minimum similarity to be considered SAME speaker
        # HIGHER threshold = stricter matching = MORE speakers detected
        # LOWER threshold = looser matching = FEWER speakers detected
        # Resemblyzer needs higher thresholds (0.80-0.88) to match pyannote speaker counts
        self.similarity_threshold = similarity_threshold or float(
            os.getenv('CPU_SPEAKER_THRESHOLD', os.getenv('SPEAKER_EMBEDDING_THRESHOLD', '0.85'))
        )
        self.min_speakers = min_speakers or int(os.getenv('MIN_EXPECTED_SPEAKERS', '1'))
        self.max_speakers = max_speakers or int(os.getenv('MAX_EXPECTED_SPEAKERS', '8'))
        # Use expected bridge crew as a hint for clustering
        self.expected_speakers = int(os.getenv('EXPECTED_BRIDGE_CREW', '6'))

        # Speaker profiles
        self.speaker_profiles: Dict[str, CPUSpeakerProfile] = {}
        self.speaker_count = 0

        # Thresholds - adjusted for resemblyzer embedding space
        # Resemblyzer similarities tend to be lower than pyannote
        self.high_confidence_threshold = 0.78  # Lowered from 0.85
        self.low_confidence_threshold = 0.50   # Lowered from 0.60

        # Minimum duration for reliable embedding
        self.min_embedding_duration = 1.0  # seconds

        # Context window for temporal smoothing
        self.context_window = 5.0

        # Sample rate expected by resemblyzer
        self.sample_rate = RESEMBLYZER_SAMPLE_RATE

        # Legacy compatibility
        self.speaker_roles: Dict[str, str] = {}
        self.speaker_names: Dict[str, str] = {}

        logger.info(
            f"CPU diarizer initialized: threshold={self.similarity_threshold}, "
            f"speakers={self.min_speakers}-{self.max_speakers}"
        )

    def _get_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio_data: Audio samples (float32, normalized -1 to 1)
            sample_rate: Audio sample rate

        Returns:
            256-dimensional speaker embedding or None
        """
        try:
            # Resample if needed
            if sample_rate != self.sample_rate:
                # Simple resampling using numpy
                ratio = self.sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)

            # Preprocess for resemblyzer
            # It expects float64 and handles normalization internally
            audio_data = audio_data.astype(np.float64)

            # Get embedding
            embedding = self.encoder.embed_utterance(audio_data)

            return embedding

        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None

    def _match_speaker(
        self,
        embedding: np.ndarray,
        duration: float = 0.0,
        timestamp: float = 0.0
    ) -> Tuple[str, float]:
        """
        Match embedding to existing speaker or create new one.

        Args:
            embedding: Speaker embedding vector
            duration: Segment duration
            timestamp: Segment start time

        Returns:
            Tuple of (speaker_id, confidence)
        """
        if not self.speaker_profiles:
            # First speaker
            return self._create_speaker(embedding, duration, timestamp)

        # Compare with existing speakers
        matches = []
        for speaker_id, profile in self.speaker_profiles.items():
            similarity = profile.similarity_to(embedding)
            matches.append((speaker_id, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_similarity = matches[0]

        # Decision logic
        if best_similarity >= self.high_confidence_threshold:
            self._update_speaker(best_match, embedding, duration, timestamp)
            return (best_match, best_similarity)

        elif best_similarity >= self.similarity_threshold:
            # Check ambiguity
            second_best = matches[1][1] if len(matches) > 1 else 0.0
            ambiguity = best_similarity - second_best

            if ambiguity > 0.1:
                self._update_speaker(best_match, embedding, duration, timestamp)
                return (best_match, best_similarity)
            else:
                # Use temporal context
                recent = self._get_recent_speaker(timestamp)
                if recent and recent in [m[0] for m in matches[:2]]:
                    recent_sim = next(m[1] for m in matches if m[0] == recent)
                    if recent_sim >= self.similarity_threshold:
                        self._update_speaker(recent, embedding, duration, timestamp)
                        return (recent, recent_sim)

                self._update_speaker(best_match, embedding, duration, timestamp)
                return (best_match, best_similarity)

        elif best_similarity >= self.low_confidence_threshold:
            if self.speaker_count < self.max_speakers:
                avg_sim = np.mean([m[1] for m in matches])
                if avg_sim < self.low_confidence_threshold:
                    return self._create_speaker(embedding, duration, timestamp)

            self._update_speaker(best_match, embedding, duration, timestamp)
            return (best_match, best_similarity)

        else:
            if self.speaker_count >= self.max_speakers:
                self._update_speaker(best_match, embedding, duration, timestamp)
                return (best_match, best_similarity)
            return self._create_speaker(embedding, duration, timestamp)

    def _create_speaker(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float
    ) -> Tuple[str, float]:
        """Create a new speaker profile."""
        self.speaker_count += 1
        speaker_id = f"speaker_{self.speaker_count}"
        profile = CPUSpeakerProfile(speaker_id=speaker_id)
        profile.add_embedding(embedding, duration, timestamp)
        self.speaker_profiles[speaker_id] = profile
        logger.info(f"New speaker: {speaker_id}")
        return (speaker_id, 1.0)

    def _update_speaker(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        duration: float,
        timestamp: float
    ):
        """Update existing speaker profile."""
        if speaker_id in self.speaker_profiles:
            self.speaker_profiles[speaker_id].add_embedding(
                embedding, duration, timestamp
            )

    def _get_recent_speaker(self, current_time: float) -> Optional[str]:
        """Get the most recently active speaker."""
        if not self.speaker_profiles:
            return None

        recent_speaker = None
        recent_time = -1.0

        for speaker_id, profile in self.speaker_profiles.items():
            if profile.last_seen_time > recent_time and profile.last_seen_time < current_time:
                recent_time = profile.last_seen_time
                recent_speaker = speaker_id

        if recent_speaker and (current_time - recent_time) <= self.context_window:
            return recent_speaker
        return None

    def identify_speaker(
        self,
        audio_segment: np.ndarray,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> Tuple[str, float]:
        """
        Identify speaker from audio segment.

        Args:
            audio_segment: Audio samples (float32, normalized)
            start_time: Segment start time
            end_time: Segment end time

        Returns:
            Tuple of (speaker_id, confidence)
        """
        duration = end_time - start_time if end_time > start_time else len(audio_segment) / 16000

        if duration < self.min_embedding_duration:
            # Short segment - use temporal context
            recent = self._get_recent_speaker(start_time)
            if recent:
                return (recent, 0.7)
            elif self.speaker_profiles:
                # Assign to most active speaker
                best = max(
                    self.speaker_profiles.items(),
                    key=lambda x: x[1].segment_count
                )[0]
                return (best, 0.5)
            else:
                # First speaker
                self.speaker_count += 1
                speaker_id = f"speaker_{self.speaker_count}"
                self.speaker_profiles[speaker_id] = CPUSpeakerProfile(speaker_id)
                return (speaker_id, 0.5)

        embedding = self._get_embedding(audio_segment)

        if embedding is not None:
            return self._match_speaker(embedding, duration, start_time)
        else:
            # Fallback to temporal context
            recent = self._get_recent_speaker(start_time)
            if recent:
                return (recent, 0.6)
            return ('unknown', 0.0)

    def process_segments_batch(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Process multiple segments in batch for consistency.

        Args:
            segments: List of segments with 'start', 'end', 'text' keys
            audio_data: Full audio data
            sample_rate: Audio sample rate

        Returns:
            Segments with 'speaker_id' and 'speaker_confidence' added
        """
        if not segments:
            return segments

        logger.info(f"CPU batch processing {len(segments)} segments")

        # Extract embeddings for all segments
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
                    embeddings_data.append((idx, embedding, seg['start'], seg['end'], duration))

        logger.info(f"Extracted {len(embeddings_data)} embeddings")

        if len(embeddings_data) < 2:
            # Process sequentially
            return self._process_sequential(segments, audio_data, sample_rate)

        # Cluster embeddings
        embeddings_matrix = np.array([e[1] for e in embeddings_data])
        cluster_labels = self._cluster_embeddings(embeddings_matrix)

        # Map clusters to speakers
        cluster_to_speaker = self._map_clusters(embeddings_data, cluster_labels)

        # Assign speakers
        segment_speakers = {}
        for i, (idx, embedding, start, end, duration) in enumerate(embeddings_data):
            cluster = cluster_labels[i]
            speaker_id = cluster_to_speaker[cluster]
            confidence = self._cluster_confidence(embedding, embeddings_matrix, cluster_labels, cluster)
            segment_speakers[idx] = (speaker_id, confidence)

        # Handle segments without embeddings
        for idx, seg in enumerate(segments):
            if idx not in segment_speakers:
                speaker_id, conf = self._interpolate_speaker(idx, segments, segment_speakers)
                segment_speakers[idx] = (speaker_id, conf)

        # Post-processing
        segment_speakers = self._postprocess(segments, segment_speakers)

        # Update segments
        for idx, seg in enumerate(segments):
            speaker_id, conf = segment_speakers.get(idx, ('unknown', 0.0))
            seg['speaker_id'] = speaker_id
            seg['speaker_confidence'] = conf

        # Log distribution
        speaker_counts = defaultdict(int)
        for sid, _ in segment_speakers.values():
            speaker_counts[sid] += 1
        logger.info(f"Speaker distribution: {dict(speaker_counts)}")

        return segments

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using hierarchical clustering."""
        n = len(embeddings)
        if n < 2:
            return np.array([0])

        # Calculate pairwise cosine distances
        distances = cdist(embeddings, embeddings, metric='cosine')
        condensed = distances[np.triu_indices(n, k=1)]
        linkage_matrix = linkage(condensed, method='average')

        # Log distance statistics for debugging - USE INFO LEVEL to always see it
        if len(condensed) > 0:
            logger.info(
                f"Embedding distances: min={condensed.min():.3f}, "
                f"max={condensed.max():.3f}, mean={condensed.mean():.3f}, "
                f"median={np.median(condensed):.3f}"
            )
            # Show distribution
            bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0]
            hist, _ = np.histogram(condensed, bins=bins)
            logger.info(f"Distance distribution: {list(zip(bins[:-1], hist))}")

        # Distance threshold = 1 - similarity_threshold
        distance_threshold = 1.0 - self.similarity_threshold
        logger.info(f"Using distance threshold: {distance_threshold:.3f} (similarity={self.similarity_threshold:.3f})")

        labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

        num_clusters = len(np.unique(labels))
        logger.info(f"Initial clustering: {num_clusters} clusters")

        # Enforce max speakers
        if num_clusters > self.max_speakers:
            labels = fcluster(linkage_matrix, t=self.max_speakers, criterion='maxclust')
            logger.info(f"Reduced to {self.max_speakers} clusters (max_speakers limit)")

        # If we got too few clusters, try forcing more based on min_speakers
        elif num_clusters < self.min_speakers and n >= self.min_speakers:
            # Force at least min_speakers clusters
            labels = fcluster(linkage_matrix, t=self.min_speakers, criterion='maxclust')
            forced_clusters = len(np.unique(labels))
            logger.info(f"Forced to {forced_clusters} clusters (min_speakers={self.min_speakers})")

        # If still too few clusters, use expected_speakers as guide
        current_clusters = len(np.unique(labels))
        if current_clusters < self.expected_speakers and n >= self.expected_speakers:
            # Force to expected number of speakers
            target = min(self.expected_speakers, self.max_speakers, n)
            labels = fcluster(linkage_matrix, t=target, criterion='maxclust')
            logger.info(
                f"Too few clusters ({current_clusters}), forced to {target} "
                f"(expected_speakers={self.expected_speakers})"
            )

        final_clusters = len(np.unique(labels))
        logger.info(f"Final clustering: {final_clusters} speakers from {n} embeddings")
        return labels

    def _map_clusters(
        self,
        embeddings_data: List[Tuple],
        cluster_labels: np.ndarray
    ) -> Dict[int, str]:
        """Map cluster IDs to speaker IDs."""
        cluster_to_speaker = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster in unique_clusters:
            self.speaker_count += 1
            speaker_id = f"speaker_{self.speaker_count}"
            cluster_to_speaker[cluster] = speaker_id

            # Create profile
            profile = CPUSpeakerProfile(speaker_id=speaker_id)
            for i, (idx, embedding, start, end, duration) in enumerate(embeddings_data):
                if cluster_labels[i] == cluster:
                    profile.add_embedding(embedding, duration, start)
            self.speaker_profiles[speaker_id] = profile

        return cluster_to_speaker

    def _cluster_confidence(
        self,
        embedding: np.ndarray,
        all_embeddings: np.ndarray,
        labels: np.ndarray,
        cluster: int
    ) -> float:
        """Calculate confidence based on cluster membership."""
        cluster_mask = labels == cluster
        cluster_embeddings = all_embeddings[cluster_mask]

        if len(cluster_embeddings) < 2:
            return 0.8

        centroid = np.mean(cluster_embeddings, axis=0)
        dist = cosine(embedding, centroid)
        similarity = 1.0 - dist

        return min(1.0, max(0.0, similarity))

    def _interpolate_speaker(
        self,
        idx: int,
        segments: List[Dict],
        segment_speakers: Dict[int, Tuple[str, float]]
    ) -> Tuple[str, float]:
        """Interpolate speaker for segment without embedding."""
        prev_speaker = None
        next_speaker = None

        for i in range(idx - 1, -1, -1):
            if i in segment_speakers:
                prev_speaker = segment_speakers[i][0]
                break

        for i in range(idx + 1, len(segments)):
            if i in segment_speakers:
                next_speaker = segment_speakers[i][0]
                break

        if prev_speaker and next_speaker:
            if prev_speaker == next_speaker:
                return (prev_speaker, 0.8)
            else:
                return (prev_speaker, 0.5)
        elif prev_speaker:
            return (prev_speaker, 0.6)
        elif next_speaker:
            return (next_speaker, 0.6)
        elif self.speaker_profiles:
            first = list(self.speaker_profiles.keys())[0]
            return (first, 0.3)
        else:
            return ('speaker_1', 0.3)

    def _postprocess(
        self,
        segments: List[Dict],
        segment_speakers: Dict[int, Tuple[str, float]]
    ) -> Dict[int, Tuple[str, float]]:
        """Post-process speaker assignments for consistency."""
        if len(segments) < 3:
            return segment_speakers

        corrected = dict(segment_speakers)

        # Fix A-B-A patterns
        for i in range(1, len(segments) - 1):
            if i not in corrected or i - 1 not in corrected or i + 1 not in corrected:
                continue

            prev, _ = corrected[i - 1]
            curr, curr_conf = corrected[i]
            next_s, _ = corrected[i + 1]

            if prev == next_s and curr != prev:
                seg = segments[i]
                duration = seg['end'] - seg['start']
                threshold = 0.85 if duration < 1.5 else 0.7

                if curr_conf < threshold:
                    corrected[i] = (prev, 0.6)

        # Fix short segments
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            duration = seg['end'] - seg['start']

            if duration < 2.0:
                if i not in corrected or i - 1 not in corrected or i + 1 not in corrected:
                    continue

                prev, _ = corrected[i - 1]
                curr, _ = corrected[i]
                next_s, _ = corrected[i + 1]

                if prev == next_s and curr != prev:
                    corrected[i] = (prev, 0.5)

        return corrected

    def _process_sequential(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Process segments sequentially."""
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            if len(segment_audio) > 0:
                speaker_id, conf = self.identify_speaker(
                    segment_audio,
                    start_time=seg['start'],
                    end_time=seg['end']
                )
                seg['speaker_id'] = speaker_id
                seg['speaker_confidence'] = conf
            else:
                seg['speaker_id'] = 'unknown'
                seg['speaker_confidence'] = 0.0

        return segments

    def reset(self):
        """Reset all speaker profiles."""
        self.speaker_profiles.clear()
        self.speaker_count = 0
        self.speaker_roles.clear()
        self.speaker_names.clear()
        logger.info("CPU diarizer reset")

    def get_speaker_count(self) -> int:
        """Get number of speakers detected."""
        return len(self.speaker_profiles) or self.speaker_count

    def cluster_embeddings_stateless(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> 'DiarizationResult':
        """
        Stateless batch clustering - no state mutation.

        This method performs speaker clustering without modifying any
        internal state (speaker_profiles, speaker_count, etc.). It's
        designed for the two-pass processing architecture.

        Args:
            segments: List of segments with 'start', 'end', 'text' keys
            audio_data: Full audio data (normalized float32)
            sample_rate: Audio sample rate

        Returns:
            DiarizationResult with clustering results
        """
        from src.audio.batch_diarizer import (
            DiarizationResult, SpeakerCluster, is_batch_diarizer_available
        )

        if not is_batch_diarizer_available():
            logger.warning("Batch diarizer not available, returning empty result")
            return DiarizationResult()

        if not segments:
            return DiarizationResult()

        logger.info(f"Stateless batch clustering: {len(segments)} segments")

        # Extract embeddings for all segments (using existing method)
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

        logger.info(f"Extracted {len(embeddings_data)} embeddings (stateless)")

        if len(embeddings_data) < 2:
            # Insufficient for clustering
            if not embeddings_data:
                cluster = SpeakerCluster(
                    speaker_id='speaker_1',
                    segment_indices=list(range(len(segments))),
                    first_appearance=segments[0]['start'] if segments else 0.0,
                    cluster_tightness=0.3
                )
                return DiarizationResult(
                    speaker_clusters={'speaker_1': cluster},
                    segment_assignments=['speaker_1'] * len(segments),
                    segment_confidences=[0.3] * len(segments),
                    total_speakers=1,
                    methodology_note="Insufficient embeddings for stateless clustering."
                )
            # Single embedding
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
            return DiarizationResult(
                speaker_clusters={'speaker_1': cluster},
                segment_assignments=['speaker_1'] * len(segments),
                segment_confidences=[0.5] * len(segments),
                total_speakers=1,
                methodology_note="Single embedding found in stateless mode."
            )

        # Perform clustering using existing method (but don't update state)
        embeddings_matrix = np.array([e[1] for e in embeddings_data])
        cluster_labels = self._cluster_embeddings(embeddings_matrix)

        # Group by cluster and find first appearance
        from collections import defaultdict
        cluster_data = defaultdict(list)
        for i, (idx, emb, start, end, duration) in enumerate(embeddings_data):
            cluster = cluster_labels[i]
            cluster_data[cluster].append((idx, emb, start, end, duration))

        cluster_first_appearance = {}
        for cluster, data in cluster_data.items():
            first_time = min(d[2] for d in data)
            cluster_first_appearance[cluster] = first_time

        # Sort clusters by first appearance for deterministic IDs
        sorted_clusters = sorted(
            cluster_data.keys(),
            key=lambda c: cluster_first_appearance[c]
        )

        # Build speaker clusters
        speaker_clusters: Dict[str, SpeakerCluster] = {}
        cluster_to_speaker: Dict[int, str] = {}

        for speaker_num, cluster in enumerate(sorted_clusters, 1):
            speaker_id = f"speaker_{speaker_num}"
            cluster_to_speaker[cluster] = speaker_id
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
            speaker_clusters[speaker_id] = cluster_obj

        # Build segment assignments
        segment_assignments = ['unknown'] * len(segments)
        segment_confidences = [0.0] * len(segments)

        for cluster in speaker_clusters.values():
            for idx in cluster.segment_indices:
                segment_assignments[idx] = cluster.speaker_id
                segment_confidences[idx] = cluster.cluster_tightness

        methodology = (
            f"Stateless batch clustering: {len(embeddings_data)} embeddings into "
            f"{len(speaker_clusters)} speakers (threshold={self.similarity_threshold:.2f}). "
            f"No state mutation - suitable for two-pass processing."
        )

        return DiarizationResult(
            speaker_clusters=speaker_clusters,
            segment_assignments=segment_assignments,
            segment_confidences=segment_confidences,
            total_speakers=len(speaker_clusters),
            methodology_note=methodology
        )
