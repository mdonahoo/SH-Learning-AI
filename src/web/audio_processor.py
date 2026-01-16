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

try:
    from src.metrics.communication_quality import CommunicationQualityAnalyzer
    QUALITY_ANALYZER_AVAILABLE = True
except ImportError:
    QUALITY_ANALYZER_AVAILABLE = False
    CommunicationQualityAnalyzer = None

# New detailed analysis imports
try:
    from src.metrics.role_inference import RoleInferenceEngine
    ROLE_INFERENCE_AVAILABLE = True
except ImportError:
    ROLE_INFERENCE_AVAILABLE = False
    RoleInferenceEngine = None

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


# Progress step definitions
ANALYSIS_STEPS = [
    {"id": "convert", "label": "Converting audio", "weight": 5},
    {"id": "transcribe", "label": "Transcribing audio", "weight": 30},
    {"id": "diarize", "label": "Identifying speakers", "weight": 15},
    {"id": "roles", "label": "Inferring roles", "weight": 10},
    {"id": "quality", "label": "Analyzing communication quality", "weight": 10},
    {"id": "scorecards", "label": "Generating scorecards", "weight": 15},
    {"id": "confidence", "label": "Analyzing confidence", "weight": 5},
    {"id": "learning", "label": "Evaluating learning metrics", "weight": 10},
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

        logger.info(
            f"AudioProcessor initialized: model={self.whisper_model_size}, "
            f"sample_rate={self.sample_rate}"
        )

        if preload_model:
            self.load_model()

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

    def transcribe_with_segments(
        self,
        audio_path: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Transcribe audio file with segment-level details.

        Args:
            audio_path: Path to audio file (WAV preferred)

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
            segments_gen, info = self._transcriber._model.transcribe(
                audio_path,
                language=None if self._transcriber.language == 'auto' else self._transcriber.language,
                initial_prompt=self._transcriber.initial_prompt,
                vad_filter=True,
                word_timestamps=True
            )

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

    def analyze_audio(
        self,
        audio_path: str,
        include_diarization: bool = True,
        include_quality: bool = True,
        include_detailed: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run full audio analysis pipeline.

        Args:
            audio_path: Path to audio file
            include_diarization: Whether to run speaker diarization
            include_quality: Whether to run communication quality analysis
            include_detailed: Whether to run detailed analysis (scorecards, learning, etc.)
            progress_callback: Optional callback function(step_id, step_label, progress_pct)

        Returns:
            Complete analysis results dictionary
        """
        start_time = time.time()

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
        progress = 5

        try:
            # Step 2: Transcription
            update_progress("transcribe", "Transcribing audio with Whisper", progress)
            segments, info = self.transcribe_with_segments(wav_path)
            results['duration_seconds'] = info.get('duration', 0)
            results['language'] = info.get('language', 'unknown')
            progress = 35

            # Step 3: Speaker diarization
            if include_diarization and DIARIZATION_AVAILABLE and segments:
                update_progress("diarize", "Identifying speakers", progress)
                segments = self._add_speaker_info(wav_path, segments)
                results['speakers'] = self._calculate_speaker_stats(segments)
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

            # Step 4: Role inference
            if include_detailed and ROLE_INFERENCE_AVAILABLE and transcripts:
                update_progress("roles", "Inferring crew roles", progress)
                results['role_assignments'] = self._analyze_roles(transcripts)
            progress = 60

            # Step 5: Communication quality analysis
            if include_quality and QUALITY_ANALYZER_AVAILABLE and segments:
                update_progress("quality", "Analyzing communication quality", progress)
                results['communication_quality'] = self._analyze_quality(segments)
            progress = 70

            # Step 6: Speaker scorecards
            if include_detailed and SCORECARD_AVAILABLE and transcripts:
                update_progress("scorecards", "Generating speaker scorecards", progress)
                results['speaker_scorecards'] = self._generate_scorecards(transcripts)
            progress = 85

            # Step 7: Confidence distribution
            if include_detailed and CONFIDENCE_ANALYZER_AVAILABLE and transcripts:
                update_progress("confidence", "Analyzing confidence distribution", progress)
                results['confidence_distribution'] = self._analyze_confidence(transcripts)
            progress = 90

            # Step 8: Learning evaluation
            if include_detailed and LEARNING_EVALUATOR_AVAILABLE and transcripts:
                update_progress("learning", "Evaluating learning metrics", progress)
                results['learning_evaluation'] = self._evaluate_learning(transcripts)
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

    def _analyze_roles(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run role inference analysis."""
        try:
            engine = RoleInferenceEngine(transcripts)
            results = engine.analyze_all()

            role_assignments = []
            for speaker_id, analysis in results.get('speakers', {}).items():
                role_assignments.append({
                    'speaker_id': speaker_id,
                    'role': analysis.get('inferred_role', 'Crew Member'),
                    'confidence': analysis.get('confidence', 0),
                    'keyword_matches': analysis.get('total_keyword_matches', 0),
                    'key_indicators': analysis.get('key_indicators', [])[:5]
                })

            return role_assignments

        except Exception as e:
            logger.error(f"Role inference failed: {e}")
            return []

    def _generate_scorecards(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate speaker scorecards."""
        try:
            generator = SpeakerScorecardGenerator(transcripts)
            scorecards = generator.generate_all()

            result = []
            for scorecard in scorecards:
                scores = []
                for score in scorecard.scores:
                    scores.append({
                        'metric_name': score.metric_name,
                        'display_name': score.metric_name.replace('_', ' ').title(),
                        'score': score.score,
                        'evidence': score.evidence
                    })

                result.append({
                    'speaker_id': scorecard.speaker,
                    'inferred_role': scorecard.inferred_role,
                    'utterance_count': scorecard.utterance_count,
                    'overall_score': scorecard.overall_score,
                    'scores': scores,
                    'strengths': scorecard.strengths,
                    'development_areas': scorecard.development_areas
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
            distribution = analyzer.analyze_distribution()

            buckets = []
            for bucket_name, data in distribution.get('buckets', {}).items():
                buckets.append({
                    'label': data.get('label', bucket_name),
                    'range_name': bucket_name,
                    'count': data.get('count', 0),
                    'percentage': data.get('percentage', 0)
                })

            # Get per-speaker averages
            speaker_averages = {}
            for speaker_data in distribution.get('per_speaker', {}).values():
                speaker_id = speaker_data.get('speaker', 'unknown')
                speaker_averages[speaker_id] = speaker_data.get('average', 0)

            avg_conf = distribution.get('overall', {}).get('average', 0)
            quality = "Excellent" if avg_conf > 0.9 else \
                      "Good" if avg_conf > 0.8 else \
                      "Acceptable" if avg_conf > 0.7 else \
                      "Marginal" if avg_conf > 0.6 else "Poor"

            return {
                'total_utterances': distribution.get('overall', {}).get('count', 0),
                'average_confidence': avg_conf,
                'buckets': buckets,
                'speaker_averages': speaker_averages,
                'quality_assessment': quality
            }

        except Exception as e:
            logger.error(f"Confidence analysis failed: {e}")
            return None

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
                # Calculate a score based on available metrics
                score = self._calculate_level_score(level_data, i)
                levels.append({
                    'level': i,
                    'name': name,
                    'score': score,
                    'interpretation': level_data.get('interpretation', f'Level {i} assessment')
                })

            # Extract Bloom's taxonomy
            blooms = results.get('blooms_taxonomy', {})
            blooms_level = blooms.get('highest_level', 'Remember')
            blooms_score = blooms.get('score', 50)

            # Extract NASA teamwork
            nasa = results.get('nasa_teamwork', {})
            nasa_score = nasa.get('overall_score', 50)

            # Calculate overall score
            overall = sum(l['score'] for l in levels) / 4 if levels else 50

            return {
                'kirkpatrick_levels': levels,
                'blooms_level': blooms_level,
                'blooms_score': blooms_score,
                'nasa_teamwork_score': nasa_score,
                'overall_learning_score': overall
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
    ) -> List[Dict[str, Any]]:
        """Add speaker identification to segments."""
        try:
            # Load audio for diarization
            audio = AudioSegment.from_wav(wav_path)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # Normalize to -1 to 1

            diarizer = SpeakerDiarizer()
            engagement = EngagementAnalyzer()

            for seg in segments:
                start_sample = int(seg['start'] * self.sample_rate)
                end_sample = int(seg['end'] * self.sample_rate)
                segment_audio = samples[start_sample:end_sample]

                if len(segment_audio) > 0:
                    speaker_id, confidence = diarizer.identify_speaker(
                        segment_audio,
                        seg['start']
                    )
                    seg['speaker_id'] = speaker_id
                    seg['speaker_confidence'] = confidence
                    seg['speaker_role'] = diarizer.speaker_roles.get(speaker_id)

                    # Track engagement
                    speaker_seg = SpeakerSegment(
                        speaker_id=speaker_id,
                        start_time=seg['start'],
                        end_time=seg['end'],
                        audio_data=segment_audio,
                        confidence=confidence,
                        text=seg['text']
                    )
                    engagement.update_speaker_stats(speaker_id, speaker_seg)

            return segments

        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return segments

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
        """Run communication quality analysis."""
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
            results = analyzer.analyze_all()

            effective_count = len(results.get('effective', []))
            improvement_count = len(results.get('needs_improvement', []))
            total = effective_count + improvement_count

            # Build pattern summary
            patterns = []
            for pattern_name, assessments in results.get('effective_by_pattern', {}).items():
                patterns.append({
                    'pattern_name': pattern_name,
                    'category': 'effective',
                    'description': '',
                    'count': len(assessments),
                    'examples': [a.evidence[:100] for a in assessments[:3]]
                })
            for pattern_name, assessments in results.get('improvement_by_pattern', {}).items():
                patterns.append({
                    'pattern_name': pattern_name,
                    'category': 'needs_improvement',
                    'description': '',
                    'count': len(assessments),
                    'examples': [a.evidence[:100] for a in assessments[:3]]
                })

            return {
                'effective_count': effective_count,
                'improvement_count': improvement_count,
                'effective_percentage': (
                    (effective_count / total * 100) if total > 0 else 0
                ),
                'patterns': patterns
            }

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return None

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
