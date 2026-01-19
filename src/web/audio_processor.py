"""
Audio processing pipeline for web API.

Orchestrates transcription, speaker diarization, and quality analysis
using existing Starship Horizons audio modules.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Audio conversion imports
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not installed. Audio conversion unavailable.")

# Project imports
try:
    from src.audio.whisper_transcriber import WhisperTranscriber, WHISPER_AVAILABLE
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperTranscriber = None

try:
    from src.audio.speaker_diarization import (
        SpeakerDiarizer,
        EngagementAnalyzer,
        SpeakerSegment
    )
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    SpeakerDiarizer = None
    EngagementAnalyzer = None
    SpeakerSegment = None

try:
    from src.audio.neural_diarization import NeuralSpeakerDiarizer, PYANNOTE_AVAILABLE
    NEURAL_DIARIZATION_AVAILABLE = PYANNOTE_AVAILABLE
except (ImportError, TypeError, Exception) as e:
    # TypeError can occur with lightning/pyannote version conflicts
    NEURAL_DIARIZATION_AVAILABLE = False
    NeuralSpeakerDiarizer = None
    logger.warning(f"Neural diarization not available: {e}")

try:
    from src.audio.cpu_diarization import CPUSpeakerDiarizer, RESEMBLYZER_AVAILABLE
    CPU_DIARIZATION_AVAILABLE = RESEMBLYZER_AVAILABLE
except (ImportError, Exception) as e:
    CPU_DIARIZATION_AVAILABLE = False
    CPUSpeakerDiarizer = None
    logger.warning(f"CPU diarization not available: {e}")

# Two-pass batch diarization
try:
    from src.audio.batch_diarizer import (
        BatchSpeakerDiarizer,
        DiarizationResult,
        is_batch_diarizer_available
    )
    BATCH_DIARIZATION_AVAILABLE = is_batch_diarizer_available()
except (ImportError, Exception) as e:
    BATCH_DIARIZATION_AVAILABLE = False
    BatchSpeakerDiarizer = None
    DiarizationResult = None
    logger.warning(f"Batch diarization not available: {e}")

try:
    from src.metrics.communication_quality import CommunicationQualityAnalyzer
    QUALITY_ANALYZER_AVAILABLE = True
except ImportError:
    QUALITY_ANALYZER_AVAILABLE = False
    CommunicationQualityAnalyzer = None

# New detailed analysis imports
try:
    from src.metrics.role_inference import RoleInferenceEngine, EnhancedRoleInferenceEngine
    ROLE_INFERENCE_AVAILABLE = True
except ImportError:
    ROLE_INFERENCE_AVAILABLE = False
    RoleInferenceEngine = None
    EnhancedRoleInferenceEngine = None

# Aggregate role inference with diarization confidence
try:
    from src.metrics.aggregate_role_inference import (
        AggregateRoleInferenceEngine,
        is_aggregate_inference_available
    )
    AGGREGATE_ROLE_INFERENCE_AVAILABLE = is_aggregate_inference_available()
except (ImportError, Exception) as e:
    AGGREGATE_ROLE_INFERENCE_AVAILABLE = False
    AggregateRoleInferenceEngine = None
    logger.warning(f"Aggregate role inference not available: {e}")

try:
    from src.metrics.speaker_scorecard import SpeakerScorecardGenerator
    SCORECARD_AVAILABLE = True
except ImportError:
    SCORECARD_AVAILABLE = False
    SpeakerScorecardGenerator = None

try:
    from src.metrics.confidence_analyzer import ConfidenceAnalyzer
    CONFIDENCE_ANALYZER_AVAILABLE = True
except ImportError:
    CONFIDENCE_ANALYZER_AVAILABLE = False
    ConfidenceAnalyzer = None

try:
    from src.metrics.learning_evaluator import LearningEvaluator
    LEARNING_EVALUATOR_AVAILABLE = True
except ImportError:
    LEARNING_EVALUATOR_AVAILABLE = False
    LearningEvaluator = None

# Educational framework imports
try:
    from src.metrics.seven_habits import SevenHabitsAnalyzer
    SEVEN_HABITS_AVAILABLE = True
except ImportError:
    SEVEN_HABITS_AVAILABLE = False
    SevenHabitsAnalyzer = None

try:
    from src.metrics.training_recommendations import TrainingRecommendationEngine
    TRAINING_RECOMMENDATIONS_AVAILABLE = True
except ImportError:
    TRAINING_RECOMMENDATIONS_AVAILABLE = False
    TrainingRecommendationEngine = None

# Telemetry-Audio Correlator for enhanced role confidence
try:
    from src.integration.telemetry_audio_correlator import TelemetryAudioCorrelator
    TELEMETRY_CORRELATOR_AVAILABLE = True
except ImportError:
    TELEMETRY_CORRELATOR_AVAILABLE = False
    TelemetryAudioCorrelator = None

# Title generator and archive manager imports
try:
    from src.web.title_generator import TitleGenerator
    TITLE_GENERATOR_AVAILABLE = True
except ImportError:
    TITLE_GENERATOR_AVAILABLE = False
    TitleGenerator = None

try:
    from src.web.archive_manager import ArchiveManager
    ARCHIVE_MANAGER_AVAILABLE = True
except ImportError:
    ARCHIVE_MANAGER_AVAILABLE = False
    ArchiveManager = None

try:
    from src.web.narrative_summary import (
        NarrativeSummaryGenerator,
        generate_summary_sync,
        generate_story_sync
    )
    NARRATIVE_GENERATOR_AVAILABLE = True
except ImportError:
    NARRATIVE_GENERATOR_AVAILABLE = False
    NarrativeSummaryGenerator = None
    generate_summary_sync = None
    generate_story_sync = None


# Progress step definitions
ANALYSIS_STEPS = [
    {"id": "convert", "label": "Converting audio", "weight": 5},
    {"id": "transcribe", "label": "Transcribing audio", "weight": 25},
    {"id": "diarize", "label": "Identifying speakers", "weight": 12},
    {"id": "roles", "label": "Inferring roles", "weight": 8},
    {"id": "quality", "label": "Analyzing communication quality", "weight": 8},
    {"id": "scorecards", "label": "Generating scorecards", "weight": 10},
    {"id": "confidence", "label": "Analyzing confidence", "weight": 5},
    {"id": "learning", "label": "Evaluating learning metrics", "weight": 7},
    {"id": "habits", "label": "Analyzing 7 Habits", "weight": 10},
    {"id": "training", "label": "Generating training recommendations", "weight": 5},
    {"id": "narrative", "label": "Generating team analysis", "weight": 5},
    {"id": "story", "label": "Generating mission story", "weight": 5},
]


class AudioProcessor:
    """
    Unified audio processing pipeline for web API.

    Handles:
    - Audio format conversion (WebM/MP3/etc → WAV)
    - Whisper transcription with word timestamps
    - Speaker diarization and identification
    - Engagement and communication quality analysis
    """

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        preload_model: bool = False
    ):
        """
        Initialize audio processor.

        Args:
            whisper_model: Whisper model size (tiny/base/small/medium/large-v3)
            preload_model: Whether to load Whisper model immediately
        """
        self.whisper_model_size = whisper_model or os.getenv(
            'WHISPER_MODEL_SIZE', 'base'
        )
        self._transcriber: Optional[WhisperTranscriber] = None
        self._model_loaded = False

        # Configuration
        self.sample_rate = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
        self.temp_dir = os.getenv('AUDIO_TEMP_DIR', tempfile.gettempdir())

        # Chunking configuration for long audio files
        # Files longer than this will be split into chunks to prevent Whisper loops
        self.chunk_duration_seconds = int(os.getenv('AUDIO_CHUNK_DURATION', '300'))  # 5 minutes
        self.chunk_overlap_seconds = int(os.getenv('AUDIO_CHUNK_OVERLAP', '5'))  # 5 second overlap

        # Speaker diarization configuration
        self.use_neural_diarization = (
            os.getenv('USE_NEURAL_DIARIZATION', 'true').lower() == 'true'
            and NEURAL_DIARIZATION_AVAILABLE
        )
        # CPU-optimized mode: use resemblyzer instead of pyannote
        # Auto-detect: use CPU mode if no GPU available and resemblyzer is installed
        self.use_cpu_diarization = (
            os.getenv('USE_CPU_DIARIZATION', 'auto').lower() == 'true'
            or (
                os.getenv('USE_CPU_DIARIZATION', 'auto').lower() == 'auto'
                and CPU_DIARIZATION_AVAILABLE
                and not self._has_gpu()
            )
        )
        self.speaker_similarity_threshold = float(
            os.getenv('SPEAKER_SIMILARITY_THRESHOLD', '0.75')
        )
        self.speaker_embedding_threshold = float(
            os.getenv('SPEAKER_EMBEDDING_THRESHOLD', '0.75')
        )
        self.min_expected_speakers = int(os.getenv('MIN_EXPECTED_SPEAKERS', '1'))
        self.max_expected_speakers = int(os.getenv('MAX_EXPECTED_SPEAKERS', '6'))

        # Cache for diarizers (expensive to initialize)
        self._neural_diarizer: Optional[NeuralSpeakerDiarizer] = None
        self._cpu_diarizer: Optional[CPUSpeakerDiarizer] = None
        self._batch_diarizer: Optional[BatchSpeakerDiarizer] = None

        # Use two-pass batch diarization by default when available
        # This provides more consistent speaker IDs across different audio lengths
        self.use_batch_diarization = (
            os.getenv('USE_BATCH_DIARIZATION', 'true').lower() == 'true'
            and BATCH_DIARIZATION_AVAILABLE
        )

        # Log diarization mode
        if self.use_cpu_diarization and CPU_DIARIZATION_AVAILABLE:
            logger.info("Speaker diarization: CPU-optimized mode (resemblyzer)")
        elif self.use_neural_diarization and NEURAL_DIARIZATION_AVAILABLE:
            logger.info("Speaker diarization: Neural mode (pyannote)")
        else:
            logger.info("Speaker diarization: Simple mode (spectral features)")

        # Recordings and analyses storage
        self.save_recordings = os.getenv('SAVE_RECORDINGS', 'true').lower() == 'true'
        self.recordings_dir = Path(os.getenv('RECORDINGS_DIR', 'data/recordings'))
        self.analyses_dir = Path(os.getenv('ANALYSES_DIR', 'data/analyses'))
        if self.save_recordings:
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            self.analyses_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Recordings will be saved to: {self.recordings_dir}")
            logger.info(f"Analyses will be saved to: {self.analyses_dir}")

        # Analysis confidence filter
        # Segments below this threshold are excluded from pattern analysis
        # to avoid analyzing garbled/low-quality transcriptions
        self.analysis_confidence_threshold = float(
            os.getenv('ANALYSIS_CONFIDENCE_THRESHOLD', '0.10')
        )
        if self.analysis_confidence_threshold > 0:
            logger.info(
                f"Analysis confidence filter: {self.analysis_confidence_threshold:.0%} "
                "(segments below this excluded from pattern analysis)"
            )

        # Initialize archive manager and title generator
        self._archive_manager: Optional[ArchiveManager] = None
        self._title_generator: Optional[TitleGenerator] = None
        if ARCHIVE_MANAGER_AVAILABLE:
            self._archive_manager = ArchiveManager()
        if TITLE_GENERATOR_AVAILABLE:
            self._title_generator = TitleGenerator()

        logger.info(
            f"AudioProcessor initialized: model={self.whisper_model_size}, "
            f"sample_rate={self.sample_rate}"
        )

        if preload_model:
            self.load_model()

    def _has_gpu(self) -> bool:
        """Check if GPU (CUDA) is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def load_model(self) -> bool:
        """
        Load Whisper model.

        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True

        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available. Install faster-whisper.")
            return False

        try:
            self._transcriber = WhisperTranscriber(
                model_size=self.whisper_model_size
            )
            self._transcriber.load_model()
            self._model_loaded = True
            logger.info(f"Whisper model '{self.whisper_model_size}' loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    @property
    def is_model_loaded(self) -> bool:
        """Check if Whisper model is loaded."""
        return self._model_loaded

    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available, cannot get duration")
            return 0.0

        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # pydub uses milliseconds
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def chunk_audio_file(
        self,
        audio_path: str,
        chunk_duration: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Split a long audio file into smaller chunks for processing.

        Chunks help prevent Whisper from getting stuck in repetition loops
        on long audio files.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds (default: 5 min)
            overlap: Overlap between chunks in seconds (default: 5 sec)

        Returns:
            List of tuples: (chunk_path, start_time, end_time)
        """
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available, returning original file")
            return [(audio_path, 0.0, self.get_audio_duration(audio_path))]

        chunk_duration = chunk_duration or self.chunk_duration_seconds
        overlap = overlap or self.chunk_overlap_seconds

        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration_ms = len(audio)
            total_duration_sec = total_duration_ms / 1000.0

            # If file is shorter than chunk duration, no need to split
            if total_duration_sec <= chunk_duration:
                logger.info(
                    f"Audio {total_duration_sec:.1f}s <= {chunk_duration}s, "
                    "no chunking needed"
                )
                return [(audio_path, 0.0, total_duration_sec)]

            chunks = []
            chunk_duration_ms = chunk_duration * 1000
            overlap_ms = overlap * 1000
            step_ms = chunk_duration_ms - overlap_ms

            logger.info(
                f"Chunking {total_duration_sec:.1f}s audio into "
                f"{chunk_duration}s chunks with {overlap}s overlap"
            )

            chunk_idx = 0
            start_ms = 0

            while start_ms < total_duration_ms:
                end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)

                # Extract chunk
                chunk_audio = audio[start_ms:end_ms]

                # Save chunk to temp file
                chunk_path = os.path.join(
                    self.temp_dir,
                    f"chunk_{chunk_idx}_{os.path.basename(audio_path)}"
                )
                chunk_audio.export(chunk_path, format="wav")

                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                chunks.append((chunk_path, start_sec, end_sec))

                logger.debug(
                    f"Created chunk {chunk_idx}: {start_sec:.1f}s - {end_sec:.1f}s"
                )

                chunk_idx += 1
                start_ms += step_ms

            logger.info(f"Created {len(chunks)} audio chunks")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk audio: {e}")
            # Return original file as fallback
            return [(audio_path, 0.0, self.get_audio_duration(audio_path))]

    def cleanup_chunks(self, chunks: List[Tuple[str, float, float]]) -> None:
        """
        Clean up temporary chunk files.

        Args:
            chunks: List of (chunk_path, start_time, end_time) tuples
        """
        for chunk_path, _, _ in chunks:
            # Only delete if it's in the temp directory and looks like a chunk
            if (
                chunk_path.startswith(self.temp_dir) and
                os.path.basename(chunk_path).startswith("chunk_")
            ):
                try:
                    os.remove(chunk_path)
                    logger.debug(f"Cleaned up chunk: {chunk_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup chunk {chunk_path}: {e}")

    def save_recording(self, audio_path: str, original_filename: str = None) -> Optional[str]:
        """
        Save a recording to the recordings directory.

        Args:
            audio_path: Path to the audio file (typically WAV after conversion)
            original_filename: Original filename for reference

        Returns:
            Path to saved recording, or None if saving is disabled
        """
        if not self.save_recordings:
            return None

        try:
            from datetime import datetime
            import shutil

            # Generate timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = Path(audio_path).suffix or '.wav'
            saved_name = f"recording_{timestamp}{ext}"
            saved_path = self.recordings_dir / saved_name

            # Copy the file
            shutil.copy2(audio_path, saved_path)
            logger.info(f"Recording saved: {saved_path}")

            return str(saved_path)

        except Exception as e:
            logger.warning(f"Failed to save recording: {e}")
            return None

    def save_analysis(
        self,
        results: Dict[str, Any],
        recording_path: Optional[str] = None,
        suffix: str = "",
        register_archive: bool = True
    ) -> Optional[str]:
        """
        Save analysis results to JSON file with auto-generated title.

        Args:
            results: Analysis results dictionary
            recording_path: Path to associated recording (for linking)
            suffix: Optional suffix for filename (e.g., "_pre_llm")
            register_archive: Whether to register in archive index (False for intermediate saves)

        Returns:
            Path to saved analysis JSON, or None if saving is disabled
        """
        if not self.save_recordings:
            return None

        try:
            import json
            from datetime import datetime

            # Generate timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_name = f"analysis_{timestamp}{suffix}.json"
            saved_path = self.analyses_dir / saved_name

            # Generate title using LLM or fallback
            auto_title = None
            if self._title_generator:
                try:
                    from src.web.title_generator import generate_title_sync
                    full_text = results.get('full_text', '')
                    speakers = results.get('speakers', [])
                    duration = results.get('duration_seconds', 0)
                    auto_title = generate_title_sync(full_text, speakers, duration)
                    logger.info(f"Generated title: {auto_title}")
                except Exception as e:
                    logger.warning(f"Title generation failed: {e}")
                    # Fallback: use first sentence of transcript
                    full_text = results.get('full_text', '')
                    if full_text:
                        first_sentence = full_text.split('.')[0][:80]
                        auto_title = first_sentence if first_sentence else None

            # Add metadata
            analysis_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'recording_file': Path(recording_path).name if recording_path else None,
                    'duration_seconds': results.get('duration_seconds', 0),
                    'speaker_count': len(results.get('speakers', [])),
                    'segment_count': len(results.get('transcription', [])),
                    'auto_title': auto_title,
                },
                'results': results
            }

            # Save as JSON
            with open(saved_path, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)

            logger.info(f"Analysis saved: {saved_path}")

            # Register with archive manager (skip for intermediate saves like pre-LLM)
            if self._archive_manager and register_archive:
                try:
                    recording_filename = Path(recording_path).name if recording_path else None
                    self._archive_manager.add_analysis(
                        filename=saved_name,
                        recording_filename=recording_filename,
                        auto_title=auto_title,
                        duration_seconds=results.get('duration_seconds', 0),
                        speaker_count=len(results.get('speakers', [])),
                        segment_count=len(results.get('transcription', []))
                    )
                    logger.info(f"Analysis registered in archive index: {saved_name}")
                except Exception as e:
                    logger.warning(f"Failed to register analysis in archive: {e}")

            return str(saved_path)

        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
            return None

    def list_analyses(self) -> List[Dict[str, Any]]:
        """
        List all saved analyses.

        Returns:
            List of analysis metadata dictionaries
        """
        if not self.analyses_dir.exists():
            return []

        analyses = []
        for f in sorted(self.analyses_dir.glob("analysis_*.json"), reverse=True):
            try:
                import json
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    metadata = data.get('metadata', {})
                    analyses.append({
                        'filename': f.name,
                        'created_at': metadata.get('created_at'),
                        'duration_seconds': metadata.get('duration_seconds', 0),
                        'speaker_count': metadata.get('speaker_count', 0),
                        'segment_count': metadata.get('segment_count', 0),
                        'recording_file': metadata.get('recording_file'),
                        'size_bytes': f.stat().st_size,
                    })
            except Exception as e:
                logger.warning(f"Failed to read analysis {f}: {e}")
                continue

        return analyses[:100]  # Limit to 100 most recent

    def get_analysis(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific saved analysis.

        Args:
            filename: Analysis filename (e.g., 'analysis_20260116_120000.json')

        Returns:
            Analysis data or None if not found
        """
        # Security: only allow files from analyses directory
        if '..' in filename or '/' in filename:
            return None

        file_path = self.analyses_dir / filename
        if not file_path.exists():
            return None

        try:
            import json
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load analysis {filename}: {e}")
            return None

    def convert_to_wav(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert audio file to WAV format for Whisper.

        Args:
            input_path: Path to input audio file
            output_path: Optional output path (generates temp file if None)

        Returns:
            Path to converted WAV file

        Raises:
            RuntimeError: If conversion fails
        """
        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub not installed. Cannot convert audio.")

        input_ext = Path(input_path).suffix.lower()

        # If already WAV with correct format, return as-is
        if input_ext == '.wav':
            # Check if it needs resampling
            try:
                audio = AudioSegment.from_wav(input_path)
                if audio.frame_rate == self.sample_rate and audio.channels == 1:
                    return input_path
            except Exception:
                pass

        # Generate output path if needed
        if output_path is None:
            output_path = os.path.join(
                self.temp_dir,
                f"converted_{int(time.time())}.wav"
            )

        try:
            # Load audio based on format
            if input_ext in ['.webm', '.opus']:
                audio = AudioSegment.from_file(input_path, format='webm')
            elif input_ext == '.mp3':
                audio = AudioSegment.from_mp3(input_path)
            elif input_ext == '.m4a':
                audio = AudioSegment.from_file(input_path, format='m4a')
            elif input_ext == '.ogg':
                audio = AudioSegment.from_ogg(input_path)
            elif input_ext == '.flac':
                audio = AudioSegment.from_file(input_path, format='flac')
            elif input_ext == '.wav':
                audio = AudioSegment.from_wav(input_path)
            else:
                # Try generic loading
                audio = AudioSegment.from_file(input_path)

            # Convert to mono, 16kHz, 16-bit for Whisper
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_sample_width(2)  # 16-bit

            audio.export(output_path, format='wav')
            logger.info(f"Converted {input_path} to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")

    def _transcribe_single_chunk(
        self,
        audio_path: str,
        time_offset: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe audio file with segment-level details.

        Args:
            audio_path: Path to audio chunk
            time_offset: Time offset to add to segment timestamps

        Returns:
            Tuple of (segments_list, info_dict)
            - segments: List of {start, end, text, confidence, words}
            - info: {language, duration, language_probability}
        """
        if not self._model_loaded:
            self.load_model()

        if not self._transcriber:
            raise RuntimeError("Whisper transcriber not available")

        try:
            # Access the underlying model directly for segment info
            # Always use English and domain-specific prompt for better accuracy
            segments_gen, info = self._transcriber._model.transcribe(
                audio_path,
                language='en',  # Force English for Starship Horizons
                initial_prompt=self._transcriber.initial_prompt,
                vad_filter=True,
                word_timestamps=True,
                beam_size=5,  # Better accuracy with beam search
                best_of=5,    # Consider more candidates
                temperature=0.0,  # Deterministic output for consistency
                condition_on_previous_text=True,  # Better context awareness
                no_speech_threshold=0.6,  # Filter out non-speech
            )

            total_duration = info.duration
            segments = []

            for segment in segments_gen:
                # Filter hallucinations
                from src.audio.whisper_transcriber import is_hallucination
                if is_hallucination(segment.text):
                    continue

                seg_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'confidence': segment.avg_logprob,
                    'words': []
                }

                # Add word-level timestamps if available
                if segment.words:
                    seg_data['words'] = [
                        {
                            'word': w.word,
                            'start': w.start,
                            'end': w.end,
                            'probability': w.probability
                        }
                        for w in segment.words
                    ]

                segments.append(seg_data)

            info_dict = {
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            }

            return segments, info_dict

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")

    def transcribe_with_segments(
        self,
        audio_path: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe audio file with segment-level details.

        For long files (>10 minutes), uses chunked processing to prevent
        Whisper hallucination loops.

        Args:
            audio_path: Path to audio file (WAV preferred)
            progress_callback: Optional callback(step_id, label, progress_pct)

        Returns:
            Tuple of (segments_list, info_dict)
            - segments: List of {start, end, text, confidence, words}
            - info: {language, duration, language_probability}
        """
        if not self._model_loaded:
            self.load_model()

        if not self._transcriber:
            raise RuntimeError("Whisper transcriber not available")

        # Check audio duration to decide on chunked vs direct processing
        audio_duration = self.get_audio_duration(audio_path)
        logger.info(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")

        # Threshold: 10 minutes (600 seconds) - use chunked processing for longer files
        CHUNK_THRESHOLD_SECONDS = 600

        if audio_duration > CHUNK_THRESHOLD_SECONDS:
            # Use chunked processing for long files
            logger.info(
                f"Audio exceeds {CHUNK_THRESHOLD_SECONDS}s threshold, "
                "using chunked transcription to prevent hallucination loops"
            )
            return self._transcribe_chunked(audio_path, audio_duration, progress_callback)
        else:
            # Use direct processing for shorter files
            return self._transcribe_direct(audio_path, progress_callback)

    def _transcribe_chunked(
        self,
        audio_path: str,
        audio_duration: float,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe using chunked processing for long files.

        Args:
            audio_path: Path to audio file
            audio_duration: Duration in seconds
            progress_callback: Optional progress callback

        Returns:
            Tuple of (segments_list, info_dict)
        """
        # Wrapper for progress callback that adapts to chunked format
        def chunk_progress_wrapper(chunk_num: int, total_chunks: int, chunk_pct: int):
            if progress_callback:
                # Map chunk progress to 5-35% range
                overall_progress = 5 + int(chunk_pct * 0.30)
                progress_callback(
                    "transcribe",
                    f"Transcribing chunk {chunk_num}/{total_chunks}...",
                    overall_progress
                )

        # Call the chunked transcription method from whisper_transcriber
        result = self._transcriber.transcribe_file_chunked(
            audio_path,
            chunk_duration=300,  # 5 minute chunks
            min_chunk_duration=120,  # 2 minute minimum
            max_chunk_duration=420,  # 7 minute maximum
            progress_callback=chunk_progress_wrapper
        )

        # Convert chunked result to expected format
        segments = result.get('segments', [])

        # Filter hallucinations from combined result
        from src.audio.whisper_transcriber import is_hallucination
        filtered_segments = [
            seg for seg in segments
            if seg.get('text') and not is_hallucination(seg['text'])
        ]

        info_dict = {
            'language': result.get('language', 'en'),
            'language_probability': 1.0,  # Not available in chunked mode
            'duration': result.get('duration', audio_duration),
            'chunk_count': result.get('chunk_count', 1),
            'chunked': True
        }

        logger.info(
            f"Chunked transcription produced {len(filtered_segments)} segments "
            f"from {info_dict.get('chunk_count', 1)} chunks"
        )

        return filtered_segments, info_dict

    def _transcribe_direct(
        self,
        audio_path: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe using direct single-pass processing.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback

        Returns:
            Tuple of (segments_list, info_dict)
        """
        # Access the underlying model directly for segment info
        segments_gen, info = self._transcriber._model.transcribe(
            audio_path,
            language='en',  # Force English for Starship Horizons
            initial_prompt=self._transcriber.initial_prompt,
            vad_filter=True,
            word_timestamps=True,
            beam_size=5,  # Better accuracy with beam search
            best_of=5,    # Consider more candidates
            temperature=0.0,  # Deterministic output for consistency
            condition_on_previous_text=False,  # Prevent hallucination propagation
            no_speech_threshold=0.6,  # Filter out non-speech
            repetition_penalty=1.5,  # Penalize repeated tokens
            no_repeat_ngram_size=3,  # Prevent 3+ word phrases from repeating
        )

        total_duration = info.duration
        segments = []

        for segment in segments_gen:
            # Filter hallucinations
            from src.audio.whisper_transcriber import is_hallucination
            if is_hallucination(segment.text):
                continue

            seg_data = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'confidence': segment.avg_logprob,
                'words': []
            }

            # Add word-level timestamps if available
            if segment.words:
                seg_data['words'] = [
                    {
                        'word': w.word,
                        'start': w.start,
                        'end': w.end,
                        'probability': w.probability
                    }
                    for w in segment.words
                ]

            segments.append(seg_data)

            # Report granular progress based on how far through audio we are
            if progress_callback and total_duration > 0:
                segment_progress = min(segment.end / total_duration, 1.0)
                overall_progress = 5 + int(segment_progress * 30)
                progress_callback(
                    "transcribe",
                    f"Transcribing... {segment.end:.1f}s / {total_duration:.1f}s",
                    overall_progress
                )

        info_dict = {
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'chunked': False
        }

        return segments, info_dict

    def analyze_audio(
        self,
        audio_path: str,
        include_diarization: bool = True,
        include_quality: bool = True,
        include_detailed: bool = True,
        include_narrative: bool = True,
        include_story: bool = True,
        progress_callback: Optional[callable] = None,
        events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run full audio analysis pipeline.

        Args:
            audio_path: Path to audio file
            include_diarization: Whether to run speaker diarization
            include_quality: Whether to run communication quality analysis
            include_detailed: Whether to run detailed analysis (scorecards, learning, etc.)
            include_narrative: Whether to generate LLM team analysis narrative
            include_story: Whether to generate LLM mission story
            progress_callback: Optional callback function(step_id, step_label, progress_pct)
            events: Optional list of telemetry events for role confidence boosting

        Returns:
            Complete analysis results dictionary
        """
        start_time = time.time()
        logger.info(f"analyze_audio called with include_narrative={include_narrative}, include_story={include_story}")

        def update_progress(step_id: str, step_label: str, progress: int):
            """Helper to call progress callback if provided."""
            if progress_callback:
                try:
                    progress_callback(step_id, step_label, progress)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        results = {
            'transcription': [],
            'full_text': '',
            'duration_seconds': 0,
            'speakers': [],
            'communication_quality': None,
            'speaker_scorecards': [],
            'role_assignments': [],
            'confidence_distribution': None,
            'learning_evaluation': None,
            'seven_habits': None,
            'training_recommendations': None,
            'narrative_summary': None,
            'saved_recording_path': None,
            'processing_time_seconds': 0
        }

        # Track cumulative progress
        progress = 0

        # Step 1: Convert to WAV if needed
        update_progress("convert", "Converting audio format", progress)
        wav_path = audio_path
        converted = False
        input_ext = Path(audio_path).suffix.lower()
        if input_ext != '.wav':
            try:
                wav_path = self.convert_to_wav(audio_path)
                converted = True
            except Exception as e:
                logger.warning(f"Conversion failed, trying direct: {e}")

        # Save recording if enabled
        saved_path = self.save_recording(wav_path)
        results['saved_recording_path'] = saved_path
        progress = 5

        try:
            # Step 2: Transcription (with granular progress updates)
            update_progress("transcribe", "Transcribing audio with Whisper", progress)
            segments, info = self.transcribe_with_segments(
                wav_path,
                progress_callback=progress_callback  # Pass through for granular updates
            )
            results['duration_seconds'] = info.get('duration', 0)
            results['language'] = info.get('language', 'unknown')
            progress = 35

            # Step 3: Speaker diarization
            diarization_result = None
            if include_diarization and DIARIZATION_AVAILABLE and segments:
                update_progress("diarize", "Identifying speakers", progress)
                segments, diarization_result = self._add_speaker_info(wav_path, segments)
                results['speakers'] = self._calculate_speaker_stats(segments)

                # Add diarization methodology to results if available
                if diarization_result:
                    results['diarization_methodology'] = diarization_result.methodology_note
            progress = 50

            # Format transcription segments for response
            results['transcription'] = [
                {
                    'start_time': s['start'],
                    'end_time': s['end'],
                    'text': s['text'],
                    'confidence': min(1.0, max(0.0, (s.get('confidence', 0) + 1) / 2)),
                    'speaker_id': s.get('speaker_id'),
                    'speaker_role': s.get('speaker_role')
                }
                for s in segments
            ]
            results['full_text'] = ' '.join(s['text'] for s in segments)

            # Build transcripts list for analysis modules
            transcripts = self._build_transcripts_list(segments)

            # Filter transcripts for pattern analysis (exclude low-confidence segments)
            # This prevents garbled/unclear speech from being analyzed as meaningful patterns
            filtered_transcripts, filter_stats = self._filter_transcripts_for_analysis(
                transcripts
            )
            results['confidence_filter'] = filter_stats

            # Step 4: Role inference (uses ALL transcripts - needs full speech for detection)
            if include_detailed and ROLE_INFERENCE_AVAILABLE and transcripts:
                update_progress("roles", "Inferring crew roles", progress)
                results['role_assignments'] = self._analyze_roles(
                    transcripts,
                    diarization_result=diarization_result
                )

                # Step 4b: Telemetry-Audio Correlation (boosts role confidence)
                if events and TELEMETRY_CORRELATOR_AVAILABLE and TelemetryAudioCorrelator:
                    try:
                        correlator = TelemetryAudioCorrelator()
                        correlator.load_data(events, transcripts)
                        correlator.correlate_all()

                        # Build speaker_roles dict for correlation
                        speaker_roles = {}
                        for ra in results['role_assignments']:
                            speaker_roles[ra['speaker_id']] = {
                                'role': ra['role'],
                                'confidence': ra['confidence'],
                                'keyword_matches': ra.get('keyword_matches', 0)
                            }

                        # Update confidences with telemetry evidence
                        updates = correlator.update_role_confidences(speaker_roles)

                        # Merge enhanced data back into role_assignments
                        for ra in results['role_assignments']:
                            speaker_id = ra['speaker_id']
                            if speaker_id in updates:
                                update = updates[speaker_id]
                                ra['voice_confidence'] = update.base_confidence
                                ra['telemetry_confidence'] = update.telemetry_boost
                                ra['confidence'] = update.boosted_confidence
                                ra['evidence_count'] = update.evidence_count
                                ra['methodology'] = update.methodology_note

                        # Add correlation summary to results
                        results['telemetry_correlation'] = correlator.get_correlation_summary()
                        logger.info(
                            f"Telemetry correlation: {len(updates)} speakers enhanced "
                            f"with {results['telemetry_correlation']['total_correlations']} correlations"
                        )
                    except Exception as e:
                        logger.warning(f"Telemetry correlation failed: {e}")

            progress = 60

            # Step 5: Communication quality analysis (uses FILTERED transcripts)
            if include_quality and QUALITY_ANALYZER_AVAILABLE and filtered_transcripts:
                update_progress("quality", "Analyzing communication quality", progress)
                # Build filtered segments for quality analyzer
                filtered_segments = [
                    s for s in segments
                    if min(1.0, max(0.0, (s.get('confidence', 0) + 1) / 2)) >= self.analysis_confidence_threshold
                ] if self.analysis_confidence_threshold > 0 else segments
                results['communication_quality'] = self._analyze_quality(filtered_segments)
            progress = 70

            # Step 6: Speaker scorecards (uses FILTERED transcripts)
            if include_detailed and SCORECARD_AVAILABLE and filtered_transcripts:
                update_progress("scorecards", "Generating speaker scorecards", progress)
                results['speaker_scorecards'] = self._generate_scorecards(filtered_transcripts)
            progress = 85

            # Step 7: Confidence distribution (uses ALL transcripts - shows full distribution)
            if include_detailed and CONFIDENCE_ANALYZER_AVAILABLE and transcripts:
                update_progress("confidence", "Analyzing confidence distribution", progress)
                results['confidence_distribution'] = self._analyze_confidence(transcripts)
            progress = 90

            # Step 8: Learning evaluation (uses FILTERED transcripts)
            if include_detailed and LEARNING_EVALUATOR_AVAILABLE and filtered_transcripts:
                update_progress("learning", "Evaluating learning metrics", progress)
                results['learning_evaluation'] = self._evaluate_learning(filtered_transcripts)
            progress = 80

            # Step 9: 7 Habits analysis (uses FILTERED transcripts)
            if include_detailed and SEVEN_HABITS_AVAILABLE and filtered_transcripts:
                update_progress("habits", "Analyzing 7 Habits framework", progress)
                results['seven_habits'] = self._analyze_seven_habits(filtered_transcripts)
            progress = 90

            # Step 10: Training recommendations (uses FILTERED transcripts)
            if include_detailed and TRAINING_RECOMMENDATIONS_AVAILABLE and filtered_transcripts:
                update_progress("training", "Generating training recommendations", progress)
                # Pass analysis results wrapped in expected format for training engine
                comm_quality = results.get('communication_quality') or {}
                conf_dist = results.get('confidence_distribution') or {}
                analysis_context = {
                    'communication_quality': {
                        'statistics': {
                            'improvement_count': comm_quality.get('improvement_count', 0),
                            'total_utterances': comm_quality.get('effective_count', 0) + comm_quality.get('improvement_count', 0),
                        }
                    },
                    'confidence_analysis': {
                        'statistics': {
                            'average_confidence': conf_dist.get('average_confidence', 0),
                        }
                    },
                    'role_analysis': results.get('role_assignments') or [],
                }
                results['training_recommendations'] = self._generate_training_recommendations(
                    filtered_transcripts, analysis_context
                )
            progress = 95

            # Save analysis BEFORE LLM processing (for debugging/comparison)
            pre_llm_path = self.save_analysis(
                results,
                results.get('saved_recording_path'),
                suffix="_pre_llm",
                register_archive=False  # Don't clutter archive with intermediate saves
            )
            if pre_llm_path:
                results['pre_llm_analysis_path'] = pre_llm_path
                logger.info(f"Pre-LLM analysis saved: {pre_llm_path}")

            # Step 11: LLM Team Analysis Narrative
            if include_detailed and include_narrative and NARRATIVE_GENERATOR_AVAILABLE and transcripts:
                update_progress("narrative", "Generating team analysis (this may take 1-2 minutes)...", progress)
                try:
                    llm_start = time.time()
                    logger.info("Starting LLM team analysis generation...")

                    narrative_result = generate_summary_sync(results)

                    llm_duration = time.time() - llm_start
                    if narrative_result:
                        results['narrative_summary'] = narrative_result
                        results['narrative_summary']['generation_time'] = round(llm_duration, 1)
                        logger.info(f"Generated team analysis in {llm_duration:.1f}s")
                    else:
                        logger.info("Team analysis skipped (Ollama unavailable)")
                except Exception as e:
                    logger.warning(f"Team analysis generation failed: {e}")
            elif not include_narrative:
                logger.info("Team analysis skipped (disabled by user)")
            progress = 97

            # Step 12: LLM Story Narrative
            if include_detailed and include_story and NARRATIVE_GENERATOR_AVAILABLE and generate_story_sync and transcripts:
                update_progress("story", "Generating mission story (this may take 1-2 minutes)...", progress)
                try:
                    story_start = time.time()
                    logger.info("Starting LLM story generation...")

                    story_result = generate_story_sync(results)

                    story_duration = time.time() - story_start
                    if story_result:
                        results['story_narrative'] = story_result
                        results['story_narrative']['generation_time'] = round(story_duration, 1)
                        logger.info(f"Generated story narrative in {story_duration:.1f}s")
                    else:
                        logger.info("Story narrative skipped (Ollama unavailable)")
                except Exception as e:
                    logger.warning(f"Story generation failed: {e}")
            elif not include_story:
                logger.info("Story narrative skipped (disabled by user)")
            progress = 100

            update_progress("complete", "Analysis complete", progress)

        finally:
            # Clean up converted file
            if converted and wav_path != audio_path:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        results['processing_time_seconds'] = time.time() - start_time

        # Save analysis results
        saved_analysis_path = self.save_analysis(results, results.get('saved_recording_path'))
        results['saved_analysis_path'] = saved_analysis_path

        return results

    def _build_transcripts_list(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build transcripts list in format expected by analysis modules."""
        return [
            {
                'text': s['text'],
                'speaker': s.get('speaker_id', 'unknown'),
                'speaker_id': s.get('speaker_id', 'unknown'),
                'confidence': min(1.0, max(0.0, (s.get('confidence', 0) + 1) / 2)),
                'timestamp': s['start'],
                'start_time': s['start'],
                'end_time': s['end']
            }
            for s in segments
        ]

    def _filter_transcripts_for_analysis(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter transcripts by confidence threshold for pattern analysis.

        Low-confidence segments (garbled/unclear speech) are excluded from
        pattern analysis (Seven Habits, communication quality, etc.) to avoid
        analyzing transcription errors as if they were actual speech patterns.

        Args:
            transcripts: Full list of transcript segments

        Returns:
            Tuple of (filtered_transcripts, filter_stats)
        """
        threshold = self.analysis_confidence_threshold

        # If threshold is 0, no filtering
        if threshold <= 0:
            return transcripts, {
                'enabled': False,
                'threshold': 0,
                'total_segments': len(transcripts),
                'included_segments': len(transcripts),
                'excluded_segments': 0,
                'excluded_percentage': 0.0
            }

        # Filter by confidence
        filtered = [t for t in transcripts if t.get('confidence', 0) >= threshold]
        excluded = [t for t in transcripts if t.get('confidence', 0) < threshold]

        total = len(transcripts)
        included_count = len(filtered)
        excluded_count = len(excluded)
        excluded_pct = (excluded_count / total * 100) if total > 0 else 0

        # Log filtering summary
        if excluded_count > 0:
            logger.info(
                f"Confidence filter: {included_count}/{total} segments included "
                f"({excluded_count} excluded below {threshold:.0%} threshold)"
            )

            # Log some examples of excluded segments for debugging
            if excluded and logger.isEnabledFor(logging.DEBUG):
                examples = excluded[:3]
                for ex in examples:
                    logger.debug(
                        f"  Excluded (conf={ex.get('confidence', 0):.2f}): "
                        f"\"{ex.get('text', '')[:50]}...\""
                    )

        stats = {
            'enabled': True,
            'threshold': threshold,
            'total_segments': total,
            'included_segments': included_count,
            'excluded_segments': excluded_count,
            'excluded_percentage': round(excluded_pct, 1),
            'excluded_examples': [
                {
                    'text': t.get('text', '')[:100],
                    'confidence': round(t.get('confidence', 0), 3),
                    'timestamp': t.get('timestamp', 0)
                }
                for t in excluded[:5]  # Include up to 5 examples
            ] if excluded else []
        }

        return filtered, stats

    def _analyze_roles(
        self,
        transcripts: List[Dict[str, Any]],
        use_voice_patterns: bool = True,
        diarization_result: Optional['DiarizationResult'] = None
    ) -> List[Dict[str, Any]]:
        """
        Run role inference analysis.

        Args:
            transcripts: List of transcript dictionaries
            use_voice_patterns: Whether to use voice pattern analysis
                               in addition to keyword matching
            diarization_result: Optional result from batch diarization
                               for enhanced confidence calculation

        Returns:
            List of role assignment dictionaries
        """
        try:
            role_assignments = []

            # Priority: Use aggregate role inference with diarization confidence
            if (diarization_result and
                AGGREGATE_ROLE_INFERENCE_AVAILABLE and
                AggregateRoleInferenceEngine):
                logger.info(
                    "Using aggregate role inference with diarization confidence "
                    f"({diarization_result.total_speakers} speakers)"
                )

                engine = AggregateRoleInferenceEngine(
                    transcripts,
                    diarization_result=diarization_result
                )
                results = engine.infer_roles()

                for speaker_id, analysis in results.items():
                    role_name = (
                        analysis.inferred_role.value
                        if hasattr(analysis.inferred_role, 'value')
                        else str(analysis.inferred_role)
                    )
                    role_assignments.append({
                        'speaker_id': speaker_id,
                        'role': role_name,
                        'confidence': analysis.combined_confidence,
                        'voice_confidence': analysis.voice_confidence,
                        'role_confidence': analysis.role_confidence,
                        'evidence_factor': analysis.evidence_factor,
                        'keyword_matches': analysis.total_keyword_matches,
                        'key_indicators': analysis.key_indicators[:5] if analysis.key_indicators else [],
                        'example_utterances': analysis.example_utterances or [],
                        'methodology': analysis.methodology_notes
                    })

                logger.info(f"Aggregate role inference complete: {len(role_assignments)} speakers")
                return role_assignments

            # Fallback: Use enhanced engine with voice patterns
            if use_voice_patterns and EnhancedRoleInferenceEngine:
                engine = EnhancedRoleInferenceEngine(transcripts, use_voice_patterns=True)
                logger.info("Using enhanced role inference with voice patterns")
            else:
                engine = RoleInferenceEngine(transcripts)
                logger.info("Using keyword-only role inference")

            results = engine.analyze_all_speakers()

            for speaker_id, analysis in results.items():
                # analysis is a SpeakerRoleAnalysis dataclass
                role_name = (
                    analysis.inferred_role.value
                    if hasattr(analysis.inferred_role, 'value')
                    else str(analysis.inferred_role)
                )
                role_assignments.append({
                    'speaker_id': speaker_id,
                    'role': role_name,
                    'confidence': analysis.confidence,
                    'keyword_matches': analysis.total_keyword_matches,
                    'key_indicators': analysis.key_indicators[:5] if analysis.key_indicators else [],
                    'example_utterances': analysis.example_utterances or [],
                    'methodology': analysis.methodology_notes
                })

            # Add voice pattern summary if available
            if hasattr(engine, 'get_voice_pattern_summary'):
                voice_summary = engine.get_voice_pattern_summary()
                if voice_summary:
                    logger.debug(
                        f"Voice pattern summary: "
                        f"{len(voice_summary.get('speaker_patterns', {}))} speakers analyzed"
                    )

            return role_assignments

        except Exception as e:
            logger.error(f"Role inference failed: {e}")
            return []

    def _generate_scorecards(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate speaker scorecards with evidence."""
        try:
            generator = SpeakerScorecardGenerator(transcripts)

            # Use get_structured_results for evidence fields
            structured = generator.get_structured_results()
            result = []

            # speaker_scorecards is a dict keyed by speaker_id
            speaker_scorecards = structured.get('speaker_scorecards', {})

            for speaker_id, scorecard_data in speaker_scorecards.items():
                metrics = []
                # scores is a dict keyed by metric_name
                scores_dict = scorecard_data.get('scores', {})
                for metric_name, score_data in scores_dict.items():
                    metrics.append({
                        'name': metric_name.replace('_', ' ').title(),
                        'score': score_data.get('score', 1),
                        'evidence': score_data.get('evidence', ''),
                        'supporting_quotes': score_data.get('supporting_quotes', []),
                        # Evidence fields
                        'threshold_info': score_data.get('threshold_info', ''),
                        'pattern_breakdown': score_data.get('pattern_breakdown', {}),
                        'calculation_details': score_data.get('calculation_details', '')
                    })

                result.append({
                    'speaker_id': speaker_id,
                    'role': scorecard_data.get('role', 'Crew Member'),
                    'utterance_count': scorecard_data.get('utterance_count', 0),
                    'overall_score': scorecard_data.get('overall_score', 1),
                    'metrics': metrics,
                    'strengths': scorecard_data.get('strengths', []),
                    'areas_for_improvement': scorecard_data.get('development_areas', [])
                })

            return result

        except Exception as e:
            logger.error(f"Scorecard generation failed: {e}")
            return []

    def _analyze_confidence(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze confidence distribution."""
        try:
            analyzer = ConfidenceAnalyzer(transcripts)
            result = analyzer.analyze_distribution()

            # Transform distribution to buckets for frontend
            buckets = []
            for dist_item in result.get('distribution', []):
                buckets.append({
                    'label': dist_item.get('range', 'Unknown'),
                    'count': dist_item.get('count', 0),
                    'percentage': dist_item.get('percentage', 0)
                })

            # Get per-speaker averages
            speaker_averages = {}
            for speaker_id, stats in result.get('speaker_stats', {}).items():
                speaker_averages[speaker_id] = stats.get('average_confidence', 0)

            avg_conf = result.get('average_confidence', 0)

            # Calculate median from transcripts
            confidences = sorted([
                t.get('confidence', 0) if t.get('confidence', 0) <= 1 else t.get('confidence', 0) / 100
                for t in transcripts
            ])
            median_conf = confidences[len(confidences) // 2] if confidences else 0

            return {
                'total_utterances': result.get('total_utterances', 0),
                'average_confidence': avg_conf,
                'median_confidence': median_conf,
                'buckets': buckets,
                'speaker_averages': speaker_averages,
                'quality_assessment': result.get('quality_assessment', 'Unknown')
            }

        except Exception as e:
            logger.error(f"Confidence analysis failed: {e}")
            return {
                'total_utterances': 0,
                'average_confidence': 0,
                'median_confidence': 0,
                'buckets': [],
                'speaker_averages': {},
                'quality_assessment': 'Unknown'
            }

    def _evaluate_learning(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate learning metrics."""
        try:
            # LearningEvaluator needs events too, but we'll pass empty list for audio-only
            evaluator = LearningEvaluator(events=[], transcripts=transcripts)
            results = evaluator.evaluate_all()

            # Extract Kirkpatrick levels
            kirkpatrick = results.get('kirkpatrick', {})
            levels = []
            level_names = ['Reaction', 'Learning', 'Behavior', 'Results']
            level_keys = ['level_1_reaction', 'level_2_learning', 'level_3_behavior', 'level_4_results']

            for i, (name, key) in enumerate(zip(level_names, level_keys), 1):
                level_data = kirkpatrick.get(key, {})
                # Calculate a score based on available metrics (0-100 scale)
                score_pct = self._calculate_level_score(level_data, i)
                levels.append({
                    'level': i,
                    'name': name,
                    'score': score_pct / 100,  # Convert to 0-1 range for frontend
                    'interpretation': level_data.get('interpretation', f'Level {i} assessment')
                })

            # Extract Bloom's taxonomy
            blooms = results.get('blooms_taxonomy', {})
            blooms_level = blooms.get('highest_level', 'Remember')
            blooms_score = blooms.get('score', 50)

            # Extract NASA teamwork
            nasa = results.get('nasa_teamwork', {})
            nasa_score = nasa.get('overall_score', 50) / 100  # Convert to 0-1 range

            # Calculate overall engagement from participation
            mission = results.get('mission_specific', {})
            engagement = mission.get('communication_frequency', {}).get('score', 50) / 100

            # Calculate overall score
            overall = sum(l['score'] for l in levels) / 4 if levels else 50

            # Get top quotes from structured report
            structured = evaluator.generate_structured_report()
            top_communications = structured.get('top_communications', [])

            return {
                'kirkpatrick_levels': levels,
                'blooms_level': blooms_level,
                'blooms_score': blooms_score / 100,  # Convert to 0-1
                'nasa_tlx_score': nasa_score,
                'engagement_score': engagement,
                'overall_learning_score': overall / 100,  # Convert to 0-1
                'top_communications': top_communications,
                'speaker_statistics': structured.get('speaker_statistics', {})
            }

        except Exception as e:
            logger.error(f"Learning evaluation failed: {e}")
            return None

    def _calculate_level_score(self, level_data: Dict, level_num: int) -> float:
        """Calculate score for a Kirkpatrick level."""
        if level_num == 1:
            # Reaction: based on engagement and participation
            equity = level_data.get('participation_equity_score', 50)
            return min(100, max(0, equity))
        elif level_num == 2:
            # Learning: based on knowledge demonstrated
            return level_data.get('knowledge_score', 50)
        elif level_num == 3:
            # Behavior: based on response times
            return level_data.get('behavior_score', 50)
        elif level_num == 4:
            # Results: mission success
            return level_data.get('results_score', 50)
        return 50

    def _add_speaker_info(
        self,
        wav_path: str,
        segments: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional['DiarizationResult']]:
        """
        Add speaker identification to segments.

        Supports four modes (in priority order):
        1. Two-pass batch (default): Consistent IDs across audio lengths
        2. CPU-optimized (resemblyzer): Fast, good for CPU-only systems
        3. Neural (pyannote): Most accurate, benefits from GPU
        4. Simple (spectral): Fallback when others unavailable

        Returns:
            Tuple of (updated_segments, DiarizationResult or None)
            - segments: Original segments with 'speaker_id' and 'speaker_confidence'
            - diarization_result: Contains voice confidence data for role inference
        """
        diarization_result = None

        try:
            diarizer = None

            # Load audio once for all methods
            audio = AudioSegment.from_wav(wav_path)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # Normalize to -1 to 1

            # Priority 0: Two-pass batch diarization (most consistent across clip lengths)
            if self.use_batch_diarization and BATCH_DIARIZATION_AVAILABLE:
                logger.info("Using two-pass batch diarization (consistent speaker IDs)")

                # Initialize batch diarizer if needed
                if self._batch_diarizer is None:
                    self._batch_diarizer = BatchSpeakerDiarizer(
                        similarity_threshold=self.speaker_embedding_threshold,
                        min_speakers=self.min_expected_speakers,
                        max_speakers=self.max_expected_speakers
                    )

                # Perform two-pass diarization
                segments, diarization_result = self._batch_diarizer.diarize_complete(
                    samples, segments, audio.frame_rate
                )

            # Priority 1: CPU-optimized diarization (for CPU-only VMs)
            elif self.use_cpu_diarization and CPU_DIARIZATION_AVAILABLE:
                logger.info("Using CPU-optimized diarization (resemblyzer)")

                # Initialize CPU diarizer if needed
                if self._cpu_diarizer is None:
                    self._cpu_diarizer = CPUSpeakerDiarizer(
                        similarity_threshold=self.speaker_embedding_threshold,
                        min_speakers=self.min_expected_speakers,
                        max_speakers=self.max_expected_speakers
                    )

                segments = self._cpu_diarizer.process_segments_batch(
                    segments, samples, audio.frame_rate
                )

                # Get diarization result for role inference if possible
                if hasattr(self._cpu_diarizer, 'cluster_embeddings_stateless'):
                    try:
                        diarization_result = self._cpu_diarizer.cluster_embeddings_stateless(
                            segments, samples, audio.frame_rate
                        )
                    except Exception as e:
                        logger.warning(f"Stateless clustering failed: {e}")

                diarizer = self._cpu_diarizer

            # Priority 2: Neural diarization with full pipeline (most accurate)
            elif NEURAL_DIARIZATION_AVAILABLE and self.use_neural_diarization:
                logger.info("Using pyannote full pipeline diarization (most accurate)")

                # Initialize neural diarizer if needed
                if self._neural_diarizer is None:
                    self._neural_diarizer = NeuralSpeakerDiarizer(
                        similarity_threshold=self.speaker_embedding_threshold,
                        min_speakers=self.min_expected_speakers,
                        max_speakers=self.max_expected_speakers
                    )

                # Use full pipeline diarization and alignment
                segments = self._neural_diarizer.diarize_and_align(wav_path, segments)
                diarizer = self._neural_diarizer

            # Priority 3: Simple speaker diarization (fallback)
            else:
                logger.info("Using simple speaker diarization (spectral features)")

                diarizer = SpeakerDiarizer(similarity_threshold=self.speaker_similarity_threshold)

                for seg in segments:
                    start_sample = int(seg['start'] * self.sample_rate)
                    end_sample = int(seg['end'] * self.sample_rate)
                    segment_audio = samples[start_sample:end_sample]

                    if len(segment_audio) > 0:
                        speaker_id, confidence = diarizer.identify_speaker(segment_audio)
                        seg['speaker_id'] = speaker_id
                        seg['speaker_confidence'] = confidence

            # Add speaker roles if available from legacy diarizer
            speaker_roles = getattr(diarizer, 'speaker_roles', {}) if diarizer else {}
            for seg in segments:
                speaker_id = seg.get('speaker_id')
                if speaker_id:
                    seg['speaker_role'] = speaker_roles.get(speaker_id)

            # Track engagement metrics
            if DIARIZATION_AVAILABLE and EngagementAnalyzer:
                engagement = EngagementAnalyzer()
                for seg in segments:
                    if seg.get('speaker_id') and SpeakerSegment:
                        start_sample = int(seg['start'] * self.sample_rate)
                        end_sample = int(seg['end'] * self.sample_rate)
                        segment_audio = samples[start_sample:end_sample]

                        speaker_seg = SpeakerSegment(
                            speaker_id=seg['speaker_id'],
                            start_time=seg['start'],
                            end_time=seg['end'],
                            audio_data=segment_audio,
                            confidence=seg.get('speaker_confidence', 0.5),
                            text=seg.get('text', '')
                        )
                        engagement.update_speaker_stats(seg['speaker_id'], speaker_seg)

            # Count unique speakers
            unique_speakers = set(s.get('speaker_id') for s in segments if s.get('speaker_id'))
            logger.info(f"Speaker diarization complete: {len(unique_speakers)} speakers detected")

            # Log speaker distribution for debugging
            speaker_counts = {}
            for seg in segments:
                sid = seg.get('speaker_id', 'unknown')
                speaker_counts[sid] = speaker_counts.get(sid, 0) + 1
            logger.info(f"Speaker distribution: {speaker_counts}")

            return segments, diarization_result

        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}", exc_info=True)
            return segments, None

    def _calculate_speaker_stats(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate speaker statistics from segments."""
        from collections import defaultdict

        speaker_data = defaultdict(lambda: {
            'total_time': 0.0,
            'utterance_count': 0,
            'role': None
        })

        for seg in segments:
            speaker_id = seg.get('speaker_id')
            if speaker_id:
                duration = seg['end'] - seg['start']
                speaker_data[speaker_id]['total_time'] += duration
                speaker_data[speaker_id]['utterance_count'] += 1
                if seg.get('speaker_role'):
                    speaker_data[speaker_id]['role'] = seg['speaker_role']

        return [
            {
                'speaker_id': sid,
                'total_speaking_time': data['total_time'],
                'utterance_count': data['utterance_count'],
                'avg_utterance_duration': (
                    data['total_time'] / data['utterance_count']
                    if data['utterance_count'] > 0 else 0
                ),
                'role': data['role'],
                'engagement_score': None  # Could calculate from engagement analyzer
            }
            for sid, data in speaker_data.items()
        ]

    def _analyze_quality(
        self,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run communication quality analysis with evidence."""
        try:
            # Format for CommunicationQualityAnalyzer
            transcripts = [
                {
                    'text': s['text'],
                    'speaker': s.get('speaker_id', 'unknown'),
                    'speaker_id': s.get('speaker_id', 'unknown'),
                    'confidence': s.get('confidence', 0),
                    'timestamp': s['start']
                }
                for s in segments
            ]

            analyzer = CommunicationQualityAnalyzer(transcripts)

            # Use the new get_structured_results for evidence
            structured = analyzer.get_structured_results()
            stats = structured.get('statistics', {})

            effective_count = stats.get('effective_count', 0)
            improvement_count = stats.get('improvement_count', 0)
            total = effective_count + improvement_count

            # Build pattern summary with full examples and evidence
            patterns = []
            pattern_counts = structured.get('pattern_counts', {})

            # Process effective patterns
            for pattern_name, count in pattern_counts.get('effective', {}).items():
                # Get examples for this pattern
                examples = [
                    ex for ex in structured.get('effective_examples', [])
                    if ex.get('pattern') == pattern_name
                ][:5]

                # Build evidence details
                evidence_details = [
                    {
                        'text': ex.get('text', ''),
                        'speaker': ex.get('speaker', 'unknown'),
                        'timestamp': ex.get('timestamp', ''),
                        'matched_substring': ex.get('matched_substring', '')
                    }
                    for ex in examples
                ]

                patterns.append({
                    'pattern_name': pattern_name,
                    'category': 'effective',
                    'description': examples[0].get('assessment', '') if examples else '',
                    'count': count,
                    'examples': [ex.get('text', '') for ex in examples],
                    'evidence_details': evidence_details
                })

            # Process improvement patterns
            for pattern_name, count in pattern_counts.get('needs_improvement', {}).items():
                examples = [
                    ex for ex in structured.get('improvement_examples', [])
                    if ex.get('pattern') == pattern_name
                ][:5]

                evidence_details = [
                    {
                        'text': ex.get('text', ''),
                        'speaker': ex.get('speaker', 'unknown'),
                        'timestamp': ex.get('timestamp', ''),
                        'matched_substring': ex.get('matched_substring', '')
                    }
                    for ex in examples
                ]

                patterns.append({
                    'pattern_name': pattern_name,
                    'category': 'needs_improvement',
                    'description': examples[0].get('issue', '') if examples else '',
                    'count': count,
                    'examples': [ex.get('text', '') for ex in examples],
                    'evidence_details': evidence_details
                })

            return {
                'effective_count': effective_count,
                'improvement_count': improvement_count,
                'effective_percentage': (
                    (effective_count / total * 100) if total > 0 else 0
                ),
                'patterns': patterns,
                # New evidence fields
                'total_utterances_assessed': structured.get('total_utterances_assessed', 0),
                'calculation_summary': structured.get('calculation_summary', ''),
                'evidence_details': structured.get('evidence_details', [])[:20]
            }

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {
                'effective_count': 0,
                'improvement_count': 0,
                'effective_percentage': 0,
                'patterns': [],
                'total_utterances_assessed': 0,
                'calculation_summary': '',
                'evidence_details': []
            }

    def _analyze_seven_habits(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze transcripts using 7 Habits framework with evidence."""
        try:
            analyzer = SevenHabitsAnalyzer(transcripts)
            results = analyzer.get_structured_results()

            # Format habits for frontend with evidence fields
            habits = []
            for habit_name, habit_data in results.get('habits', {}).items():
                # Include full examples with speaker context
                formatted_examples = []
                for ex in habit_data.get('examples', [])[:5]:  # Up to 5 examples
                    formatted_examples.append({
                        'text': ex.get('text', ''),
                        'speaker': ex.get('speaker', 'unknown'),
                        'timestamp': ex.get('timestamp', '')
                    })

                habits.append({
                    'habit_number': habit_data.get('habit_number', 0),
                    'habit_name': habit_name.replace('_', ' ').title(),
                    'youth_friendly_name': habit_data.get('youth_name', ''),
                    'score': habit_data.get('score', 1),
                    'observation_count': habit_data.get('count', 0),
                    'interpretation': habit_data.get('interpretation', ''),
                    'development_tip': habit_data.get('development_tip', ''),
                    'examples': formatted_examples,
                    # New evidence fields
                    'pattern_breakdown': habit_data.get('pattern_breakdown', {}),
                    'speaker_contributions': habit_data.get('speaker_contributions', {}),
                    'gap_to_next_score': habit_data.get('gap_to_next_score', '')
                })

            # Sort by habit number
            habits.sort(key=lambda x: x['habit_number'])

            return {
                'overall_score': results.get('overall_effectiveness_score', 0),
                'habits': habits,
                'strengths': results.get('strengths', []),
                'growth_areas': results.get('growth_areas', []),
                # Include score thresholds for evidence
                'score_thresholds': results.get('score_thresholds', {})
            }

        except Exception as e:
            logger.error(f"7 Habits analysis failed: {e}")
            return None

    def _generate_training_recommendations(
        self,
        transcripts: List[Dict[str, Any]],
        analysis_context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive training recommendations."""
        try:
            engine = TrainingRecommendationEngine(
                transcripts,
                analysis_results=analysis_context or {}
            )
            results = engine.get_structured_results()

            # Format immediate actions
            immediate_actions = []
            for action in results.get('immediate_actions', []):
                immediate_actions.append({
                    'title': action.get('title', ''),
                    'description': action.get('description', ''),
                    'priority': action.get('priority', 'MEDIUM'),
                    'category': action.get('category', ''),
                    'frameworks': action.get('frameworks', []),
                    'scout_connection': action.get('scout_connection'),
                    'habit_connection': action.get('habit_connection'),
                    'success_criteria': action.get('success_criteria', '')
                })

            # Format drills
            drills = []
            for drill in results.get('drills', []):
                drills.append({
                    'name': drill.get('name', ''),
                    'purpose': drill.get('purpose', ''),
                    'duration': drill.get('duration', ''),
                    'participants': 'full team',
                    'steps': drill.get('steps', []),
                    'debrief_questions': drill.get('debrief_questions', []),
                    'frameworks_addressed': []
                })

            # Format discussion topics
            discussion_topics = []
            for topic in results.get('discussion_topics', []):
                discussion_topics.append({
                    'topic': topic.get('topic', ''),
                    'question': topic.get('question', ''),
                    'scout_connection': topic.get('scout_connection'),
                    'discussion_points': topic.get('discussion_points', [])
                })

            return {
                'immediate_actions': immediate_actions,
                'communication_improvements': [],  # Extracted from immediate_actions by category
                'leadership_development': [],
                'teamwork_enhancements': [],
                'drills': drills,
                'discussion_topics': discussion_topics,
                'framework_alignment': results.get('framework_alignment', {}),
                'total_recommendations': results.get('total_recommendations', 0)
            }

        except Exception as e:
            logger.error(f"Training recommendations failed: {e}")
            return {
                'immediate_actions': [],
                'communication_improvements': [],
                'leadership_development': [],
                'teamwork_enhancements': [],
                'drills': [],
                'discussion_topics': [],
                'framework_alignment': {},
                'total_recommendations': 0
            }

    def transcribe_only(self, audio_path: str) -> Dict[str, Any]:
        """
        Run transcription only (no diarization or quality analysis).

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription results
        """
        start_time = time.time()

        # Convert if needed
        wav_path = audio_path
        converted = False
        if Path(audio_path).suffix.lower() != '.wav':
            try:
                wav_path = self.convert_to_wav(audio_path)
                converted = True
            except Exception:
                pass

        try:
            segments, info = self.transcribe_with_segments(wav_path)

            return {
                'segments': [
                    {
                        'start_time': s['start'],
                        'end_time': s['end'],
                        'text': s['text'],
                        'confidence': min(1.0, max(0.0, (s.get('confidence', 0) + 1) / 2)),
                        'speaker_id': None,
                        'speaker_role': None
                    }
                    for s in segments
                ],
                'full_text': ' '.join(s['text'] for s in segments),
                'duration_seconds': info.get('duration', 0),
                'language': info.get('language', 'unknown'),
                'processing_time_seconds': time.time() - start_time
            }

        finally:
            if converted and wav_path != audio_path:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
