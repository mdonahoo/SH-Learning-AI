"""
Neural speaker diarization using pyannote.audio.

Provides advanced speaker identification using deep learning embeddings
for improved accuracy over simple acoustic features.

Enhanced features:
- Batch processing with clustering for consistent speaker assignment
- Multiple embedding comparison for robust matching
- Consistency checking across segments
- Improved short segment handling with context
"""

# Suppress pyannote warnings before import (torchcodec, lightning checkpoint)
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*loss_func.*")
warnings.filterwarnings("ignore", message=".*task-dependent loss.*")

import numpy as np
import logging
import os
import tempfile
import wave
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv

try:
    from pyannote.audio import Pipeline, Model
    from pyannote.audio.core.inference import Inference
    from scipy.spatial.distance import cosine, cdist
    from scipy.cluster.hierarchy import linkage, fcluster
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """
    Enhanced speaker profile with multiple embeddings and statistics.

    Maintains a collection of embeddings for robust matching,
    along with confidence metrics and temporal information.
    """
    speaker_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    segment_count: int = 0
    total_duration: float = 0.0
    last_seen_time: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Get centroid (mean) of all embeddings."""
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)

    @property
    def average_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return np.mean(self.confidence_scores)

    def add_embedding(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
        confidence: float,
        max_embeddings: int = 20
    ):
        """Add a new embedding, maintaining size limit with FIFO."""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)
        self.segment_count += 1
        self.total_duration += duration
        self.last_seen_time = timestamp

        # Keep only the most recent embeddings
        if len(self.embeddings) > max_embeddings:
            self.embeddings.pop(0)
            self.confidence_scores.pop(0)

    def similarity_to(self, embedding: np.ndarray, method: str = 'centroid_plus_best') -> float:
        """
        Calculate similarity to a new embedding.

        Args:
            embedding: New embedding to compare
            method: Comparison method:
                - 'centroid': Compare to centroid only
                - 'best': Compare to most similar stored embedding
                - 'centroid_plus_best': Average of centroid and best match (default)

        Returns:
            Similarity score (0-1)
        """
        if not self.embeddings:
            return 0.0

        centroid = self.centroid
        centroid_sim = 1.0 - cosine(embedding, centroid)

        if method == 'centroid':
            return centroid_sim

        # Find best individual match
        best_sim = 0.0
        for stored_emb in self.embeddings:
            sim = 1.0 - cosine(embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim

        if method == 'best':
            return best_sim

        # Default: weighted combination (centroid + best) / 2
        # Gives robustness of centroid + precision of best match
        return (centroid_sim * 0.6 + best_sim * 0.4)


# Determine device for pyannote models
def _get_device():
    """Get the best available device for pyannote models."""
    if PYANNOTE_AVAILABLE:
        if torch.cuda.is_available():
            logger.info("CUDA available for pyannote, using GPU")
            return torch.device("cuda")
        else:
            logger.info("CUDA not available for pyannote, using CPU")
            return torch.device("cpu")
    return None


class NeuralSpeakerDiarizer:
    """
    Neural network-based speaker diarization using pyannote.audio.

    Enhanced features for improved accuracy:
    - Batch processing with hierarchical clustering
    - Multiple embedding comparison (centroid + best match)
    - Consistency checking and post-processing
    - Improved short segment handling with context
    - Adaptive threshold based on embedding distribution
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_auth_token: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialize neural speaker diarizer.

        Args:
            model_name: Hugging Face model name for diarization pipeline
            min_speakers: Minimum number of speakers expected
            max_speakers: Maximum number of speakers expected
            use_auth_token: Hugging Face authentication token (if needed)
            similarity_threshold: Minimum similarity (0-1) to match speakers
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not installed. "
                "Install with: pip install pyannote.audio"
            )

        self.model_name = model_name
        self.min_speakers = min_speakers or int(os.getenv('MIN_EXPECTED_SPEAKERS', '1'))
        self.max_speakers = max_speakers or int(os.getenv('MAX_EXPECTED_SPEAKERS', '8'))

        # Get auth token from environment if not provided
        self.use_auth_token = use_auth_token or os.getenv('HUGGINGFACE_TOKEN')

        # Enhanced speaker tracking with SpeakerProfile objects
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        self.speaker_count = 0

        # Base similarity threshold - can be adapted dynamically
        self.similarity_threshold = similarity_threshold or float(
            os.getenv('SPEAKER_EMBEDDING_THRESHOLD', '0.70')  # Lowered from 0.80 for better recall
        )

        # High confidence threshold for definite matches
        self.high_confidence_threshold = 0.85

        # Low confidence threshold - below this, consider creating new speaker
        self.low_confidence_threshold = 0.55

        # Speaker metadata (for compatibility with SpeakerDiarizer interface)
        self.speaker_roles: Dict[str, str] = {}
        self.speaker_names: Dict[str, str] = {}

        # Legacy compatibility
        self.speaker_embeddings: Dict[str, List[np.ndarray]] = {}

        # Minimum segment duration for reliable embedding (seconds)
        self.min_embedding_duration = 0.8  # Reduced from 1.0 for more coverage

        # Maximum embeddings to keep per speaker (for memory efficiency)
        self.max_embeddings_per_speaker = 20  # Increased for better profile

        # Context window for short segment handling (seconds)
        self.context_window = 2.0

        # Batch processing cache
        self._batch_embeddings: List[Tuple[int, np.ndarray, float, float]] = []  # (idx, emb, start, end)
        self._batch_mode = False

        # Load pipeline and embedding model lazily
        self.pipeline = None
        self.embedding_model = None

        logger.info(
            f"Neural diarizer initialized: model={model_name}, "
            f"expected_speakers={self.min_speakers}-{self.max_speakers}, "
            f"threshold={self.similarity_threshold}, "
            f"high_conf={self.high_confidence_threshold}, "
            f"low_conf={self.low_confidence_threshold}"
        )

    def _load_pipeline(self):
        """Lazy load the diarization pipeline."""
        if self.pipeline is None:
            try:
                logger.info(f"Loading pyannote pipeline: {self.model_name}")
                # Try new API (token) first, fall back to old API (use_auth_token)
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_name,
                        token=self.use_auth_token
                    )
                except TypeError:
                    # Fall back to old API
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_name,
                        use_auth_token=self.use_auth_token
                    )
                logger.info("✓ Pyannote pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load pyannote pipeline: {e}")
                raise

    def _load_embedding_model(self):
        """Lazy load speaker embedding model."""
        if self.embedding_model is None:
            try:
                logger.info("Loading speaker embedding model: pyannote/embedding")
                # Load the model first
                model = Model.from_pretrained(
                    "pyannote/embedding",
                    token=self.use_auth_token
                )
                # Create inference wrapper
                self.embedding_model = Inference(model)
                logger.info("✓ Speaker embedding model loaded")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None

    def _get_speaker_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segment.

        Args:
            audio_data: Audio samples (float32, normalized)
            sample_rate: Sample rate

        Returns:
            Speaker embedding vector or None if extraction fails
        """
        self._load_embedding_model()

        if self.embedding_model is None:
            return None

        try:
            # Method 1: Try direct waveform input (preferred, no disk I/O)
            try:
                # Convert audio to torch tensor format expected by pyannote
                # pyannote expects (channel, samples) format
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()

                # Create audio dict format expected by pyannote
                audio_dict = {
                    'waveform': audio_tensor,
                    'sample_rate': sample_rate
                }

                # Run inference on the whole audio segment
                with torch.no_grad():
                    embedding_output = self.embedding_model(audio_dict)

                    # Extract the actual data from SlidingWindowFeature
                    if hasattr(embedding_output, 'data'):
                        embedding = embedding_output.data
                    else:
                        embedding = embedding_output

                    # Convert to numpy if it's a torch tensor
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()

                    # Average over chunks and frames to get single embedding vector
                    while embedding.ndim > 1:
                        embedding = embedding.mean(axis=0)

                return embedding

            except (NameError, RuntimeError) as e:
                # Method 2: Fallback to temp WAV file (more compatible)
                logger.debug(f"Direct waveform failed ({e}), using temp file")

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    # Convert to int16 for WAV
                    audio_int16 = (audio_data * 32767).astype(np.int16)

                    # Write WAV file
                    with wave.open(tmp_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())

                    # Run inference on file
                    with torch.no_grad():
                        embedding_output = self.embedding_model(tmp_path)

                        if hasattr(embedding_output, 'data'):
                            embedding = embedding_output.data
                        else:
                            embedding = embedding_output

                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()

                        while embedding.ndim > 1:
                            embedding = embedding.mean(axis=0)

                    return embedding

                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            logger.debug(f"Embedding extraction error details: {e}", exc_info=True)
            return None

    def _match_speaker(
        self,
        embedding: np.ndarray,
        duration: float = 0.0,
        timestamp: float = 0.0
    ) -> Tuple[str, float]:
        """
        Match embedding to existing speaker or create new speaker.

        Uses enhanced matching with:
        - Multiple embedding comparison (centroid + best match)
        - Adaptive thresholds based on confidence
        - Temporal context for ambiguous cases

        Args:
            embedding: Speaker embedding vector
            duration: Segment duration in seconds
            timestamp: Segment start time in seconds

        Returns:
            Tuple of (speaker_id, similarity_score)
        """
        if not self.speaker_profiles:
            # First speaker
            self.speaker_count += 1
            speaker_id = f"speaker_{self.speaker_count}"
            profile = SpeakerProfile(speaker_id=speaker_id)
            profile.add_embedding(embedding, duration, timestamp, 1.0, self.max_embeddings_per_speaker)
            self.speaker_profiles[speaker_id] = profile
            # Legacy compatibility
            self.speaker_embeddings[speaker_id] = [embedding]
            logger.info(f"Created first speaker: {speaker_id}")
            return (speaker_id, 1.0)

        # Compare with all existing speakers using enhanced comparison
        matches: List[Tuple[str, float]] = []

        for speaker_id, profile in self.speaker_profiles.items():
            similarity = profile.similarity_to(embedding, method='centroid_plus_best')
            matches.append((speaker_id, similarity))

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_similarity = matches[0]

        # Get second best for ambiguity detection
        second_best_sim = matches[1][1] if len(matches) > 1 else 0.0

        # Calculate ambiguity - if top two are close, be more careful
        ambiguity = best_similarity - second_best_sim

        # Decision logic with adaptive thresholds
        if best_similarity >= self.high_confidence_threshold:
            # High confidence match - definitely this speaker
            self._update_speaker_profile(best_match, embedding, duration, timestamp, best_similarity)
            logger.debug(
                f"High confidence match: {best_match} "
                f"(sim={best_similarity:.3f}, ambiguity={ambiguity:.3f})"
            )
            return (best_match, best_similarity)

        elif best_similarity >= self.similarity_threshold:
            # Medium confidence - check ambiguity
            if ambiguity > 0.1:
                # Clear winner, accept the match
                self._update_speaker_profile(best_match, embedding, duration, timestamp, best_similarity)
                logger.debug(
                    f"Medium confidence match: {best_match} "
                    f"(sim={best_similarity:.3f}, ambiguity={ambiguity:.3f})"
                )
                return (best_match, best_similarity)
            else:
                # Ambiguous - use temporal context
                # Prefer speaker who was active recently
                recent_speaker = self._get_most_recent_speaker(timestamp)
                if recent_speaker and recent_speaker in [m[0] for m in matches[:2]]:
                    # Recent speaker is in top 2, prefer them
                    recent_sim = next(m[1] for m in matches if m[0] == recent_speaker)
                    if recent_sim >= self.similarity_threshold:
                        self._update_speaker_profile(recent_speaker, embedding, duration, timestamp, recent_sim)
                        logger.debug(
                            f"Temporal context match: {recent_speaker} "
                            f"(sim={recent_sim:.3f}, was recent)"
                        )
                        return (recent_speaker, recent_sim)

                # Fall back to best match
                self._update_speaker_profile(best_match, embedding, duration, timestamp, best_similarity)
                return (best_match, best_similarity)

        elif best_similarity >= self.low_confidence_threshold:
            # Low confidence - might be new speaker or poor audio quality
            # If we haven't hit max speakers, consider creating new one
            if self.speaker_count < self.max_speakers:
                # Check if this could be a new speaker by looking at variance
                # New speaker should be consistently different from all existing
                avg_similarity = np.mean([m[1] for m in matches])

                if avg_similarity < self.low_confidence_threshold:
                    # Low similarity to everyone - likely new speaker
                    return self._create_new_speaker(embedding, duration, timestamp)

            # Assign to best match but with low confidence
            self._update_speaker_profile(best_match, embedding, duration, timestamp, best_similarity)
            logger.debug(
                f"Low confidence match: {best_match} "
                f"(sim={best_similarity:.3f})"
            )
            return (best_match, best_similarity)

        else:
            # Very low similarity - likely new speaker
            if self.speaker_count >= self.max_speakers:
                # At capacity, force match to closest
                logger.warning(
                    f"Max speakers ({self.max_speakers}) reached. "
                    f"Forcing match to {best_match} (sim={best_similarity:.3f})"
                )
                self._update_speaker_profile(best_match, embedding, duration, timestamp, best_similarity)
                return (best_match, best_similarity)

            return self._create_new_speaker(embedding, duration, timestamp)

    def _create_new_speaker(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float
    ) -> Tuple[str, float]:
        """Create a new speaker profile."""
        self.speaker_count += 1
        speaker_id = f"speaker_{self.speaker_count}"
        profile = SpeakerProfile(speaker_id=speaker_id)
        profile.add_embedding(embedding, duration, timestamp, 1.0, self.max_embeddings_per_speaker)
        self.speaker_profiles[speaker_id] = profile
        # Legacy compatibility
        self.speaker_embeddings[speaker_id] = [embedding]
        logger.info(f"New speaker detected: {speaker_id}")
        return (speaker_id, 1.0)

    def _update_speaker_profile(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        duration: float,
        timestamp: float,
        confidence: float
    ):
        """Update existing speaker profile with new embedding."""
        if speaker_id in self.speaker_profiles:
            self.speaker_profiles[speaker_id].add_embedding(
                embedding, duration, timestamp, confidence, self.max_embeddings_per_speaker
            )
            # Legacy compatibility
            if speaker_id not in self.speaker_embeddings:
                self.speaker_embeddings[speaker_id] = []
            self.speaker_embeddings[speaker_id].append(embedding)
            if len(self.speaker_embeddings[speaker_id]) > self.max_embeddings_per_speaker:
                self.speaker_embeddings[speaker_id].pop(0)

    def _get_most_recent_speaker(self, current_time: float) -> Optional[str]:
        """Get the speaker who was most recently active."""
        if not self.speaker_profiles:
            return None

        recent_speaker = None
        recent_time = -1.0

        for speaker_id, profile in self.speaker_profiles.items():
            if profile.last_seen_time > recent_time and profile.last_seen_time < current_time:
                recent_time = profile.last_seen_time
                recent_speaker = speaker_id

        # Only return if they spoke within context window
        if recent_speaker and (current_time - recent_time) <= self.context_window:
            return recent_speaker
        return None

    def process_audio_file(
        self,
        audio_path: str,
        sample_rate: int = 16000
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Process an audio file and identify speaker segments.

        Args:
            audio_path: Path to audio file
            sample_rate: Audio sample rate

        Returns:
            Dictionary mapping speaker_id to list of (start_time, end_time) tuples
        """
        self._load_pipeline()

        try:
            # Run diarization
            diarization = self.pipeline(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )

            # Convert to our format
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))

            logger.info(f"Detected {len(speaker_segments)} speakers in audio file")
            return speaker_segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return {}

    def process_audio_segment(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
        end_time: float = 0.0,
        return_all_speakers: bool = False
    ) -> Tuple[str, float]:
        """
        Process a single audio segment and identify the speaker(s) using embeddings.

        Enhanced with:
        - Better short segment handling using temporal context
        - Improved embedding extraction with fallbacks
        - Confidence-aware speaker matching

        Args:
            audio_data: Audio samples (float32, normalized -1.0 to 1.0)
            sample_rate: Audio sample rate
            start_time: Segment start time (for temporal context)
            end_time: Segment end time (for duration calculation)
            return_all_speakers: If True, return list of (speaker, start, end) tuples

        Returns:
            Tuple of (speaker_id, confidence)
        """
        # Calculate segment duration
        segment_duration = len(audio_data) / sample_rate
        if end_time > start_time:
            segment_duration = end_time - start_time

        # Try to extract embedding
        embedding = None

        if segment_duration >= self.min_embedding_duration:
            embedding = self._get_speaker_embedding(audio_data, sample_rate)

        if embedding is not None:
            # Match embedding to existing speaker or create new one
            speaker_id, similarity = self._match_speaker(
                embedding,
                duration=segment_duration,
                timestamp=start_time
            )
            return (speaker_id, similarity)

        # Fallback for short segments or failed embedding extraction
        # Use temporal context to make a smart assignment
        return self._handle_short_segment(audio_data, sample_rate, start_time, segment_duration)

    def _handle_short_segment(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        timestamp: float,
        duration: float
    ) -> Tuple[str, float]:
        """
        Handle short segments that can't produce reliable embeddings.

        Uses temporal context and speaker profile statistics to make
        the best assignment decision.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            timestamp: Segment start time
            duration: Segment duration

        Returns:
            Tuple of (speaker_id, confidence)
        """
        if not self.speaker_profiles:
            # First speaker - create profile even for short segment
            self.speaker_count += 1
            speaker_id = f"speaker_{self.speaker_count}"
            profile = SpeakerProfile(speaker_id=speaker_id)
            self.speaker_profiles[speaker_id] = profile
            self.speaker_embeddings[speaker_id] = []
            logger.info(f"Created first speaker from short segment: {speaker_id}")
            return (speaker_id, 0.5)

        # Find most likely speaker using temporal context
        recent_speaker = self._get_most_recent_speaker(timestamp)

        if recent_speaker:
            # Assign to recent speaker with moderate confidence
            logger.debug(
                f"Short segment ({duration:.2f}s) assigned to recent speaker: {recent_speaker}"
            )
            return (recent_speaker, 0.6)

        # No recent speaker - assign to the speaker with most segments
        # (likely the most active/dominant speaker)
        best_speaker = max(
            self.speaker_profiles.items(),
            key=lambda x: x[1].segment_count
        )[0]

        logger.debug(
            f"Short segment ({duration:.2f}s) assigned to most active speaker: {best_speaker}"
        )
        return (best_speaker, 0.4)

    def identify_speaker(
        self,
        audio_segment: np.ndarray,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> Tuple[str, float]:
        """
        Identify speaker from audio segment (compatible with simple diarizer interface).

        Args:
            audio_segment: Audio samples (float32, normalized)
            start_time: Segment start time (optional, for context)
            end_time: Segment end time (optional, for duration)

        Returns:
            Tuple of (speaker_id, confidence)
        """
        return self.process_audio_segment(
            audio_segment,
            sample_rate=16000,
            start_time=start_time,
            end_time=end_time
        )

    def get_speaker_count(self) -> int:
        """Get the number of unique speakers identified."""
        return len(self.speaker_profiles) or self.speaker_count

    # ==================== Batch Processing Methods ====================

    def process_segments_batch(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Process multiple segments in batch mode for improved consistency.

        This method:
        1. Extracts embeddings for all segments
        2. Clusters embeddings to identify speakers
        3. Assigns speakers consistently across all segments
        4. Post-processes to fix obvious errors

        Args:
            segments: List of segment dicts with 'start', 'end', 'text' keys
            audio_data: Full audio data (float32, normalized)
            sample_rate: Audio sample rate

        Returns:
            Segments with 'speaker_id' and 'speaker_confidence' added
        """
        if not segments:
            return segments

        logger.info(f"Batch processing {len(segments)} segments")

        # Phase 1: Extract embeddings for all segments
        embeddings_data = []  # (segment_idx, embedding, start, end, duration)

        for idx, seg in enumerate(segments):
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            duration = seg['end'] - seg['start']

            if len(segment_audio) == 0:
                continue

            if duration >= self.min_embedding_duration:
                embedding = self._get_speaker_embedding(segment_audio, sample_rate)
                if embedding is not None:
                    embeddings_data.append((idx, embedding, seg['start'], seg['end'], duration))

        logger.info(f"Extracted {len(embeddings_data)} embeddings from {len(segments)} segments")

        if len(embeddings_data) < 2:
            # Not enough data for clustering, use sequential processing
            return self._process_segments_sequential(segments, audio_data, sample_rate)

        # Phase 2: Cluster embeddings
        embeddings_matrix = np.array([e[1] for e in embeddings_data])
        cluster_labels = self._cluster_embeddings(embeddings_matrix)

        # Phase 3: Map clusters to speaker IDs
        cluster_to_speaker = self._map_clusters_to_speakers(
            embeddings_data, cluster_labels
        )

        # Phase 4: Assign speakers to segments
        segment_speakers = {}  # idx -> (speaker_id, confidence)

        for i, (idx, embedding, start, end, duration) in enumerate(embeddings_data):
            cluster = cluster_labels[i]
            speaker_id = cluster_to_speaker[cluster]
            # Calculate confidence based on distance to cluster centroid
            confidence = self._calculate_cluster_confidence(
                embedding, embeddings_matrix, cluster_labels, cluster
            )
            segment_speakers[idx] = (speaker_id, confidence)

        # Phase 5: Handle segments without embeddings (short segments)
        for idx, seg in enumerate(segments):
            if idx not in segment_speakers:
                # Use temporal interpolation
                speaker_id, confidence = self._interpolate_speaker(
                    idx, segments, segment_speakers
                )
                segment_speakers[idx] = (speaker_id, confidence)

        # Phase 6: Post-process for consistency
        segment_speakers = self._postprocess_speaker_assignments(
            segments, segment_speakers
        )

        # Phase 7: Update segment dicts
        for idx, seg in enumerate(segments):
            speaker_id, confidence = segment_speakers.get(idx, ('unknown', 0.0))
            seg['speaker_id'] = speaker_id
            seg['speaker_confidence'] = confidence

        # Log speaker distribution
        speaker_counts = defaultdict(int)
        for speaker_id, _ in segment_speakers.values():
            speaker_counts[speaker_id] += 1
        logger.info(f"Speaker distribution: {dict(speaker_counts)}")

        return segments

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using hierarchical clustering.

        Args:
            embeddings: Matrix of embeddings (n_samples, n_features)

        Returns:
            Cluster labels for each embedding
        """
        n_samples = len(embeddings)

        if n_samples < 2:
            return np.array([0])

        # Calculate pairwise distances
        distances = cdist(embeddings, embeddings, metric='cosine')

        # Hierarchical clustering with average linkage
        # Convert to condensed form for linkage
        condensed_dist = distances[np.triu_indices(n_samples, k=1)]
        linkage_matrix = linkage(condensed_dist, method='average')

        # Determine number of clusters
        # Use distance threshold based on similarity threshold
        distance_threshold = 1.0 - self.similarity_threshold

        # Get cluster labels
        labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

        # Ensure we don't exceed max speakers
        unique_labels = np.unique(labels)
        if len(unique_labels) > self.max_speakers:
            # Re-cluster with fewer clusters
            labels = fcluster(linkage_matrix, t=self.max_speakers, criterion='maxclust')

        logger.info(f"Clustering produced {len(np.unique(labels))} clusters from {n_samples} embeddings")
        return labels

    def _map_clusters_to_speakers(
        self,
        embeddings_data: List[Tuple],
        cluster_labels: np.ndarray
    ) -> Dict[int, str]:
        """
        Map cluster IDs to speaker IDs, matching to existing profiles if possible.

        Args:
            embeddings_data: List of (idx, embedding, start, end, duration) tuples
            cluster_labels: Cluster label for each embedding

        Returns:
            Dict mapping cluster ID to speaker ID
        """
        cluster_to_speaker = {}
        unique_clusters = np.unique(cluster_labels)

        # Calculate centroid for each cluster
        cluster_centroids = {}
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_embeddings = np.array([
                embeddings_data[i][1] for i in range(len(embeddings_data)) if cluster_mask[i]
            ])
            cluster_centroids[cluster] = np.mean(cluster_embeddings, axis=0)

        # Try to match clusters to existing speaker profiles
        matched_clusters = set()
        matched_speakers = set()

        if self.speaker_profiles:
            # Calculate similarity between cluster centroids and speaker profiles
            for cluster, centroid in cluster_centroids.items():
                best_speaker = None
                best_similarity = 0.0

                for speaker_id, profile in self.speaker_profiles.items():
                    if speaker_id in matched_speakers:
                        continue
                    similarity = profile.similarity_to(centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_speaker = speaker_id

                if best_speaker and best_similarity >= self.similarity_threshold:
                    cluster_to_speaker[cluster] = best_speaker
                    matched_clusters.add(cluster)
                    matched_speakers.add(best_speaker)
                    logger.debug(
                        f"Cluster {cluster} matched to existing speaker {best_speaker} "
                        f"(similarity={best_similarity:.3f})"
                    )

        # Create new speakers for unmatched clusters
        for cluster in unique_clusters:
            if cluster not in cluster_to_speaker:
                self.speaker_count += 1
                speaker_id = f"speaker_{self.speaker_count}"
                cluster_to_speaker[cluster] = speaker_id

                # Create profile for new speaker
                profile = SpeakerProfile(speaker_id=speaker_id)
                # Add all embeddings from this cluster
                for i, (idx, embedding, start, end, duration) in enumerate(embeddings_data):
                    if cluster_labels[i] == cluster:
                        profile.add_embedding(
                            embedding, duration, start, 1.0, self.max_embeddings_per_speaker
                        )
                self.speaker_profiles[speaker_id] = profile
                self.speaker_embeddings[speaker_id] = list(profile.embeddings)

                logger.info(f"Created new speaker {speaker_id} for cluster {cluster}")

        return cluster_to_speaker

    def _calculate_cluster_confidence(
        self,
        embedding: np.ndarray,
        all_embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        cluster: int
    ) -> float:
        """
        Calculate confidence score based on embedding's position within cluster.

        Args:
            embedding: The embedding to evaluate
            all_embeddings: All embeddings in the batch
            cluster_labels: Cluster assignments
            cluster: The assigned cluster

        Returns:
            Confidence score (0-1)
        """
        # Get cluster members
        cluster_mask = cluster_labels == cluster
        cluster_embeddings = all_embeddings[cluster_mask]

        if len(cluster_embeddings) < 2:
            return 0.8  # Single-member cluster

        # Calculate distance to cluster centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        dist_to_centroid = cosine(embedding, centroid)
        similarity_to_centroid = 1.0 - dist_to_centroid

        # Calculate average distance to other clusters
        other_distances = []
        for other_cluster in np.unique(cluster_labels):
            if other_cluster != cluster:
                other_mask = cluster_labels == other_cluster
                other_centroid = np.mean(all_embeddings[other_mask], axis=0)
                other_distances.append(cosine(embedding, other_centroid))

        if other_distances:
            avg_other_dist = np.mean(other_distances)
            # Higher confidence if far from other clusters
            separation = avg_other_dist - dist_to_centroid
            confidence = min(1.0, similarity_to_centroid + (separation * 0.5))
        else:
            confidence = similarity_to_centroid

        return max(0.0, min(1.0, confidence))

    def _interpolate_speaker(
        self,
        idx: int,
        segments: List[Dict],
        segment_speakers: Dict[int, Tuple[str, float]]
    ) -> Tuple[str, float]:
        """
        Interpolate speaker for a segment without an embedding.

        Uses surrounding segments to make the best assignment.

        Args:
            idx: Segment index
            segments: All segments
            segment_speakers: Known speaker assignments

        Returns:
            Tuple of (speaker_id, confidence)
        """
        # Look for nearest segments with speaker assignments
        prev_speaker = None
        next_speaker = None

        # Search backwards
        for i in range(idx - 1, -1, -1):
            if i in segment_speakers:
                prev_speaker = segment_speakers[i][0]
                break

        # Search forwards
        for i in range(idx + 1, len(segments)):
            if i in segment_speakers:
                next_speaker = segment_speakers[i][0]
                break

        if prev_speaker and next_speaker:
            if prev_speaker == next_speaker:
                # Surrounded by same speaker - high confidence
                return (prev_speaker, 0.8)
            else:
                # Different speakers - use time proximity
                seg = segments[idx]
                seg_time = (seg['start'] + seg['end']) / 2

                if idx > 0 and idx - 1 in segment_speakers:
                    prev_time = segments[idx - 1]['end']
                else:
                    prev_time = 0

                if idx < len(segments) - 1 and idx + 1 in segment_speakers:
                    next_time = segments[idx + 1]['start']
                else:
                    next_time = float('inf')

                # Closer to previous or next?
                if abs(seg_time - prev_time) < abs(seg_time - next_time):
                    return (prev_speaker, 0.5)
                else:
                    return (next_speaker, 0.5)

        elif prev_speaker:
            return (prev_speaker, 0.6)
        elif next_speaker:
            return (next_speaker, 0.6)
        else:
            # No context - assign to first speaker
            if self.speaker_profiles:
                first_speaker = list(self.speaker_profiles.keys())[0]
                return (first_speaker, 0.3)
            else:
                return ('speaker_1', 0.3)

    def _postprocess_speaker_assignments(
        self,
        segments: List[Dict],
        segment_speakers: Dict[int, Tuple[str, float]]
    ) -> Dict[int, Tuple[str, float]]:
        """
        Post-process speaker assignments to fix obvious errors.

        Applies:
        - Smoothing: isolated single-segment speaker changes
        - Merging: very short segments assigned to different speaker than surroundings

        Args:
            segments: All segments
            segment_speakers: Current speaker assignments

        Returns:
            Corrected speaker assignments
        """
        if len(segments) < 3:
            return segment_speakers

        corrected = dict(segment_speakers)

        # Pass 1: Fix isolated speaker changes (A-B-A pattern)
        for i in range(1, len(segments) - 1):
            if i not in corrected or i - 1 not in corrected or i + 1 not in corrected:
                continue

            prev_speaker, _ = corrected[i - 1]
            curr_speaker, curr_conf = corrected[i]
            next_speaker, _ = corrected[i + 1]

            if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                # Isolated change - check confidence
                if curr_conf < 0.7:
                    # Low confidence isolated segment - reassign to surrounding speaker
                    corrected[i] = (prev_speaker, 0.6)
                    logger.debug(
                        f"Corrected isolated segment {i}: {curr_speaker} -> {prev_speaker}"
                    )

        # Pass 2: Check very short segments surrounded by same speaker
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            duration = seg['end'] - seg['start']

            if duration < 1.0:  # Very short segment
                if i not in corrected or i - 1 not in corrected or i + 1 not in corrected:
                    continue

                prev_speaker, _ = corrected[i - 1]
                curr_speaker, curr_conf = corrected[i]
                next_speaker, _ = corrected[i + 1]

                if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                    # Short segment between same speakers - likely wrong
                    corrected[i] = (prev_speaker, 0.5)
                    logger.debug(
                        f"Corrected short segment {i} ({duration:.2f}s): "
                        f"{curr_speaker} -> {prev_speaker}"
                    )

        return corrected

    def _process_segments_sequential(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Process segments sequentially (fallback when batch processing isn't possible).

        Args:
            segments: List of segment dicts
            audio_data: Full audio data
            sample_rate: Audio sample rate

        Returns:
            Segments with speaker assignments
        """
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            if len(segment_audio) > 0:
                speaker_id, confidence = self.identify_speaker(
                    segment_audio,
                    start_time=seg['start'],
                    end_time=seg['end']
                )
                seg['speaker_id'] = speaker_id
                seg['speaker_confidence'] = confidence
            else:
                seg['speaker_id'] = 'unknown'
                seg['speaker_confidence'] = 0.0

        return segments

    def reset(self):
        """Reset all speaker profiles and state."""
        self.speaker_profiles.clear()
        self.speaker_embeddings.clear()
        self.speaker_count = 0
        self.speaker_roles.clear()
        self.speaker_names.clear()
        self._batch_embeddings.clear()
        logger.info("Speaker diarizer reset")
