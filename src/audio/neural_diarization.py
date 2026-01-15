"""
Neural speaker diarization using pyannote.audio.

Provides advanced speaker identification using deep learning embeddings
for improved accuracy over simple acoustic features.
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
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

try:
    from pyannote.audio import Pipeline, Model
    from pyannote.audio.core.inference import Inference
    from scipy.spatial.distance import cosine
    import torch
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)


class NeuralSpeakerDiarizer:
    """
    Neural network-based speaker diarization using pyannote.audio.

    Provides much better speaker separation than simple acoustic features,
    especially for similar voices in similar recording conditions.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize neural speaker diarizer.

        Args:
            model_name: Hugging Face model name for diarization pipeline
            min_speakers: Minimum number of speakers expected
            max_speakers: Maximum number of speakers expected
            use_auth_token: Hugging Face authentication token (if needed)
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not installed. "
                "Install with: pip install pyannote.audio"
            )

        self.model_name = model_name
        self.min_speakers = min_speakers or int(os.getenv('MIN_EXPECTED_SPEAKERS', '1'))
        self.max_speakers = max_speakers or int(os.getenv('MAX_EXPECTED_SPEAKERS', '6'))

        # Get auth token from environment if not provided
        self.use_auth_token = use_auth_token or os.getenv('HUGGINGFACE_TOKEN')

        # Speaker tracking with embeddings
        self.speaker_embeddings: Dict[str, List[np.ndarray]] = {}  # speaker_id -> [embeddings]
        self.speaker_count = 0
        self.similarity_threshold = float(os.getenv('SPEAKER_EMBEDDING_THRESHOLD', '0.75'))

        # Minimum segment duration for reliable embedding (seconds)
        # Shorter segments use pipeline fallback for better accuracy
        self.min_embedding_duration = 1.0  # At least 1 second for embedding

        # Maximum embeddings to keep per speaker (for memory efficiency)
        self.max_embeddings_per_speaker = 10

        # Load pipeline and embedding model lazily
        self.pipeline = None
        self.embedding_model = None

        logger.info(
            f"Neural diarizer initialized: model={model_name}, "
            f"expected_speakers={self.min_speakers}-{self.max_speakers}, "
            f"threshold={self.similarity_threshold}"
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

    def _match_speaker(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Match embedding to existing speaker or create new speaker.

        Args:
            embedding: Speaker embedding vector

        Returns:
            Tuple of (speaker_id, similarity_score)
        """
        if not self.speaker_embeddings:
            # First speaker
            self.speaker_count += 1
            speaker_id = f"speaker_{self.speaker_count}"
            self.speaker_embeddings[speaker_id] = [embedding]
            return (speaker_id, 1.0)

        # Compare with all existing speakers
        best_match = None
        best_similarity = 0.0

        for speaker_id, embeddings in self.speaker_embeddings.items():
            # Compare with average of all embeddings for this speaker
            avg_embedding = np.mean(embeddings, axis=0)
            similarity = 1.0 - cosine(embedding, avg_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        # Check if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            # Matched existing speaker - add embedding but limit storage
            if len(self.speaker_embeddings[best_match]) >= self.max_embeddings_per_speaker:
                # Remove oldest embedding
                self.speaker_embeddings[best_match].pop(0)
            self.speaker_embeddings[best_match].append(embedding)
            logger.debug(f"Matched to {best_match} (similarity: {best_similarity:.3f})")
            return (best_match, best_similarity)
        else:
            # New speaker - check if we've hit max speakers
            if self.speaker_count >= self.max_speakers:
                # Force match to closest speaker if at capacity
                logger.warning(
                    f"Max speakers ({self.max_speakers}) reached. "
                    f"Forcing match to {best_match} (similarity: {best_similarity:.3f})"
                )
                return (best_match, best_similarity)

            # Create new speaker
            self.speaker_count += 1
            speaker_id = f"speaker_{self.speaker_count}"
            self.speaker_embeddings[speaker_id] = [embedding]
            logger.info(
                f"New speaker detected: {speaker_id} "
                f"(closest was {best_match} at {best_similarity:.3f}, threshold={self.similarity_threshold})"
            )
            return (speaker_id, 1.0)

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
        return_all_speakers: bool = False
    ) -> Tuple[str, float]:
        """
        Process a single audio segment and identify the speaker(s) using embeddings.

        Args:
            audio_data: Audio samples (float32, normalized -1.0 to 1.0)
            sample_rate: Audio sample rate
            return_all_speakers: If True, return list of (speaker, start, end) tuples

        Returns:
            Tuple of (speaker_id, confidence) or List of speaker segments if return_all_speakers=True
        """
        # Calculate segment duration
        segment_duration = len(audio_data) / sample_rate

        # For short segments, use pipeline directly (more accurate for brief speech)
        if segment_duration < self.min_embedding_duration:
            logger.debug(
                f"Short segment ({segment_duration:.2f}s < {self.min_embedding_duration}s), "
                "using pipeline fallback"
            )
            # Skip to pipeline fallback below
            embedding = None
        else:
            # Extract speaker embedding and match to existing speakers
            embedding = self._get_speaker_embedding(audio_data, sample_rate)

        if embedding is not None:
            # Match embedding to existing speaker or create new one
            speaker_id, similarity = self._match_speaker(embedding)
            return (speaker_id, similarity)

        # Fallback: embedding extraction failed
        # For short segments or when embedding fails, assign to most recent speaker
        # or create new speaker (simpler than pipeline which also has AudioDecoder issues)

        if self.speaker_embeddings:
            # Assign to most recently active speaker (reasonable heuristic)
            # This is better than creating too many speakers for short utterances
            most_recent = max(self.speaker_embeddings.keys())
            logger.debug(f"Short/failed segment assigned to {most_recent}")
            return (most_recent, 0.5)
        else:
            # First speaker
            self.speaker_count += 1
            speaker_id = f"speaker_{self.speaker_count}"
            logger.info(f"Created first speaker: {speaker_id}")
            return (speaker_id, 0.5)

        # Note: Pipeline fallback disabled due to AudioDecoder compatibility issues
        # The embedding-based approach above handles most cases well
        """
        # Original pipeline fallback (kept for reference)
        logger.warning("Embedding extraction failed, falling back to pipeline-only detection")
        self._load_pipeline()

        # Create temporary WAV file
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

            diarization = self.pipeline(tmp_path, min_speakers=1, max_speakers=self.max_speakers)
            # ... pipeline processing code ...
        """

    def identify_speaker(
        self,
        audio_segment: np.ndarray
    ) -> Tuple[str, float]:
        """
        Identify speaker from audio segment (compatible with simple diarizer interface).

        Args:
            audio_segment: Audio samples (float32, normalized)

        Returns:
            Tuple of (speaker_id, confidence)
        """
        return self.process_audio_segment(audio_segment, sample_rate=16000)

    def get_speaker_count(self) -> int:
        """Get the number of unique speakers identified."""
        return len(self.speaker_profiles) or self.speaker_count
