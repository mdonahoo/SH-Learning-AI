"""
Whisper Transcription Module for Starship Horizons Bridge Audio.

Handles speech-to-text transcription using Faster-Whisper (local AI model).
No cloud dependency - all processing happens locally.
"""

import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from dotenv import load_dotenv

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("faster-whisper not installed. Transcription unavailable.")

load_dotenv()

logger = logging.getLogger(__name__)

# Domain-specific vocabulary prompt for Starship Horizons and Empty Epsilon
# This biases Whisper toward recognizing game-specific terms
STARSHIP_HORIZONS_PROMPT = (
    # Starship Horizons specific
    "Starship Horizons bridge simulator, USS Donahoo, captain's log, stardate. "
    "Star systems: Paravera, Calavera, Faraday, Parrot Arrow, Starbase Delta. "

    # Bridge stations (both games)
    "Bridge stations: helm, tactical, science, engineering, operations, communications, "
    "weapons officer, relay officer, single pilot. "

    # Ship systems
    "Ship systems: warp drive, warp core, warp engines, impulse engines, impulse drive, "
    "shields, shield strength, shield frequency, phasers, phaser banks, phaser array, "
    "torpedoes, photon torpedoes, homing missiles, nukes, EMPs, mines, "
    "forward shields, aft shields, port shields, starboard shields, "
    "sensor array, long-range sensors, short-range sensors, scanners, "
    "transporter, tractor beam, combat maneuver, jump drive, reactor. "

    # Navigation terms
    "Navigation: waypoint, sector, bearing, mark, coordinates, heading, "
    "course plotted, course laid in, set course, engage, "
    "all stop, full stop, full impulse, half impulse, one quarter impulse, "
    "all back, reverse, maximum warp, warp factor, drop out of warp, de-warp, "
    "ETA, coming about, closing distance, intercept course, evasive maneuvers. "

    # Communications
    "Communications: comm station, hail, hailing frequencies, channel open, "
    "transmitting, receiving, subspace, distress signal, distress call, "
    "Space Dock, docking permission, undock, undocking, request permission. "

    # Alerts and conditions
    "Alerts: red alert, condition red, yellow alert, condition yellow, "
    "battle stations, all hands, shields up, shields down, "
    "general quarters, action stations, stand down. "

    # Combat terminology
    "Combat: fire, open fire, cease fire, fire at will, weapons free, "
    "attack pattern, attack pattern Delta, attack pattern Alpha, "
    "weapons hot, target locked, lock on, targeting, "
    "direct hit, hull damage, hull breach, shields failing, shields holding, "
    "torpedoes away, firing phasers, beam weapons, projectile weapons, "
    "missile lock, countermeasures, point defense. "

    # Commands and responses
    "Commands: engage, make it so, on screen, main viewer, acknowledged, "
    "aye sir, aye captain, aye aye, affirmative, negative, copy that, roger, "
    "helm responds, ready captain, standing by, awaiting orders. "

    # Status reports
    "Status: nominal, operational, online, offline, damaged, destroyed, "
    "critical, stable, maneuvering, in position, on station, "
    "power levels, energy reserves, coolant levels, heat levels, "
    "damage control, repair teams, damage report. "

    # Empty Epsilon specific
    "Empty Epsilon: Artemis, spaceship bridge simulator, "
    "nebula, asteroid field, black hole, wormhole, "
    "friendly, hostile, neutral, unknown contact, "
    "probe, supply drop, artifact, anomaly, "
    "dock, supply station, defense platform, sentry turret, "
    "player ship, CPU ship, scenario, mission objective. "

    # Crew coordination
    "Crew: speak from diaphragm, bridge crew, away team, "
    "triangulate, scan the planet, scan complete, scan results, "
    "balance power, power distribution, boost shields, boost weapons, "
    "route power, divert power, emergency power. "
)

# Alternative shorter prompt if the full one causes issues
STARSHIP_HORIZONS_PROMPT_SHORT = (
    "Starship bridge simulator. Stations: helm, tactical, science, engineering, ops. "
    "Terms: warp, impulse, shields, phasers, torpedoes, bearing, mark, heading. "
    "Commands: engage, fire, on screen, aye captain, red alert, shields up. "
    "Locations: Paravera, Calavera, Faraday, Starbase Delta."
)

# Known Whisper hallucinations to filter out
# These appear when Whisper processes silence, unclear audio, or background noise
WHISPER_HALLUCINATIONS = [
    # YouTube/video outro hallucinations
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "thanks for listening",
    "subscribe",
    "like and subscribe",
    "hit the like button",
    "hit the bell",
    "notification bell",
    "click here",
    "link in the description",
    "don't forget to",
    "see you next time",
    "see you in the next",
    "until next time",
    "bye bye",
    "goodbye",
    "take care",

    # Caption/subtitle service hallucinations
    "caption",
    "captions by",
    "captions provided",
    "captioning by",
    "captioning provided",
    "subtitles by",
    "subtitles provided",
    "transcribed by",
    "transcription by",
    "gettranscribed",
    "getcaptioned",
    "captionedthis",
    "rev.com",
    "amara.org",

    # Copyright/legal text hallucinations
    "copyright",
    "©",
    "all rights reserved",
    "rights reserved",
    "trademark",
    "™",
    "®",
    "licensed under",
    "creative commons",
    "new thinking allowed",
    "foundation",
    "productions",
    "entertainment",
    "studios",
    "broadcasting",
    "network",
    "channel",

    # URL and website patterns
    "www.",
    ".com",
    ".org",
    ".net",
    ".edu",
    ".gov",
    ".io",
    "http",
    "https",
    "://",
    "@gmail",
    "@yahoo",
    "@hotmail",

    # Music/sound effect markers
    "music",
    "[music]",
    "(music)",
    "[music playing]",
    "♪",
    "♫",
    "applause",
    "[applause]",
    "(applause)",
    "[laughter]",
    "(laughter)",
    "[laughing]",
    "[silence]",
    "[inaudible]",
    "[unintelligible]",
    "[background noise]",
    "[static]",
    "[beep]",
    "[bleep]",

    # Podcast/video intros and outros
    "welcome back",
    "hey everyone",
    "hi everyone",
    "hello everyone",
    "hello and welcome",
    "welcome to the show",
    "welcome to the podcast",
    "in this video",
    "in this episode",
    "in today's video",
    "in today's episode",
    "today we're going to",
    "today we will",
    "let's get started",
    "let's jump right in",
    "without further ado",
    "before we begin",
    "quick reminder",

    # Foreign language artifacts (common Whisper errors)
    "sous-titres",  # French subtitles
    "untertitel",   # German subtitles
    "sottotitoli",  # Italian subtitles
    "subtítulos",   # Spanish subtitles

    # Repetitive filler (often from silence)
    "um um um",
    "uh uh uh",
    "hmm hmm hmm",
    "the the the",
    "a a a a",
    "i i i i",

    # Sponsor/ad reads
    "this episode is sponsored",
    "this video is sponsored",
    "brought to you by",
    "special thanks to",
    "shout out to",
    "patreon",
    "ko-fi",
    "buy me a coffee",

    # Social media
    "follow me on",
    "follow us on",
    "twitter",
    "instagram",
    "facebook",
    "tiktok",
    "discord server",
    "join our discord",

    # Common misheard artifacts
    "you",  # Single word "you" with nothing else is often hallucination
]

# Patterns that indicate hallucination only if they're the ENTIRE transcription
# (not just contained within legitimate speech)
WHISPER_HALLUCINATIONS_EXACT = [
    "you",
    "the",
    "a",
    "i",
    "it",
    "so",
    "yeah",
    "okay",
    "right",
    "thank you",
    "thanks",
    "yes",
    "no",
    "hmm",
    "uh",
    "um",
]


def is_hallucination(text: str) -> bool:
    """
    Check if transcription text is a known Whisper hallucination.

    Args:
        text: Transcription text to check

    Returns:
        True if text appears to be a hallucination
    """
    if not text:
        return True

    text_lower = text.lower().strip()

    # Remove punctuation for checking
    text_clean = ''.join(c for c in text_lower if c.isalnum() or c.isspace()).strip()

    # Check for very short text that's likely noise
    if len(text_clean) < 3:
        return True

    # Check for exact match hallucinations (single words/short phrases)
    if text_clean in WHISPER_HALLUCINATIONS_EXACT:
        return True

    # Check for known hallucination patterns (substring match)
    for hallucination in WHISPER_HALLUCINATIONS:
        if hallucination in text_lower:
            return True

    # Check for repeated characters/words (common hallucination)
    words = text_clean.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True

    # Check for excessive repetition (e.g., "right right right right")
    if len(words) >= 4:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        max_count = max(word_counts.values())
        if max_count >= len(words) * 0.75:  # 75% same word
            return True

    return False


class WhisperTranscriber:
    """
    Local AI transcription using Faster-Whisper.

    Features:
    - Local model inference (no cloud dependency)
    - Real-time transcription via worker threads
    - Automatic language detection
    - Word-level timestamps
    - Memory-efficient processing
    - Configurable model size and precision

    Performance:
    - base model: ~7x realtime on CPU, ~70x on GPU
    - small model: ~4x realtime on CPU, ~40x on GPU
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cpu or cuda)
            compute_type: Compute precision (int8, float16, float32)
            language: Language code (en, es, fr, etc.) or 'auto'
            num_workers: Number of transcription worker threads
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper not installed. "
                "Run: pip install faster-whisper"
            )

        # Load configuration
        self.model_size = model_size or os.getenv('WHISPER_MODEL_SIZE', 'large-v3')
        self.device = device or os.getenv('WHISPER_DEVICE', 'cpu')
        self.compute_type = compute_type or os.getenv('WHISPER_COMPUTE_TYPE', 'int8')
        self.language = language or os.getenv('TRANSCRIBE_LANGUAGE', 'en')
        self.num_workers = num_workers or int(os.getenv('TRANSCRIPTION_WORKERS', '2'))

        # Initial prompt for domain-specific vocabulary
        self.initial_prompt = os.getenv(
            'WHISPER_INITIAL_PROMPT',
            STARSHIP_HORIZONS_PROMPT
        )

        # Model path
        model_path = Path(os.getenv('WHISPER_MODEL_PATH', './data/models/whisper/'))
        model_path.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path

        # Model instance (lazy loaded)
        self._model: Optional[WhisperModel] = None
        self._model_loaded = False
        self._model_lock = threading.Lock()

        # Transcription queue and workers
        self._transcription_queue = queue.Queue(
            maxsize=int(os.getenv('MAX_SEGMENT_QUEUE_SIZE', '100'))
        )
        self._worker_threads: List[threading.Thread] = []
        self._is_running = False

        # Results storage
        self._results_lock = threading.Lock()
        self._pending_results = []

        logger.info(
            f"WhisperTranscriber initialized: "
            f"model={self.model_size}, device={self.device}, "
            f"compute={self.compute_type}, language={self.language}, "
            f"workers={self.num_workers}, "
            f"prompt_length={len(self.initial_prompt)} chars"
        )

    def load_model(self) -> bool:
        """
        Load the Whisper model into memory.

        Returns:
            True if model loaded successfully
        """
        with self._model_lock:
            if self._model_loaded:
                logger.debug("Model already loaded")
                return True

            try:
                logger.info(f"Loading Whisper model: {self.model_size}")
                start_time = time.time()

                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.model_path)
                )

                load_time = time.time() - start_time
                self._model_loaded = True

                logger.info(f"✓ Whisper model loaded in {load_time:.2f}s")

                # Warm up model with dummy audio
                self._warmup_model()

                return True

            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                return False

    def _warmup_model(self):
        """Warm up model with dummy inference to avoid first-call latency."""
        try:
            logger.info("Warming up Whisper model...")
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
            segments, info = self._model.transcribe(dummy_audio)
            list(segments)  # Force transcription to complete
            logger.info("✓ Model warmed up")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def start_workers(self):
        """Start transcription worker threads."""
        if self._is_running:
            logger.warning("Transcription workers already running")
            return

        # Ensure model is loaded
        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load Whisper model")

        self._is_running = True

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._transcription_worker,
                args=(i,),
                daemon=True,
                name=f"WhisperWorker-{i}"
            )
            worker.start()
            self._worker_threads.append(worker)

        logger.info(f"Started {self.num_workers} transcription worker threads")

    def stop_workers(self):
        """Stop transcription worker threads."""
        if not self._is_running:
            return

        logger.info("Stopping transcription workers...")
        self._is_running = False

        # Send sentinel values to wake up all workers
        for _ in range(self.num_workers):
            try:
                self._transcription_queue.put(None, timeout=1)
            except queue.Full:
                pass

        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5)

        self._worker_threads.clear()
        logger.info("✓ Transcription workers stopped")

    def queue_audio(
        self,
        audio_data: np.ndarray,
        timestamp: float,
        speaker_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        """
        Queue audio for transcription.

        Args:
            audio_data: Audio samples (float32, normalized)
            timestamp: Timestamp of audio start
            speaker_id: Optional speaker identifier
            metadata: Optional metadata dictionary
        """
        try:
            self._transcription_queue.put_nowait({
                'audio': audio_data,
                'timestamp': timestamp,
                'speaker_id': speaker_id,
                'metadata': metadata or {}
            })
        except queue.Full:
            logger.warning("Transcription queue full, dropping audio segment")

    def _transcription_worker(self, worker_id: int):
        """
        Worker thread that processes transcription queue.

        Args:
            worker_id: Worker thread identifier
        """
        logger.info(f"Whisper worker {worker_id} started")

        while self._is_running:
            try:
                # Get item from queue (timeout to allow checking is_running)
                item = self._transcription_queue.get(timeout=1)

                # Check for sentinel
                if item is None:
                    break

                # Transcribe audio
                result = self._transcribe_segment(
                    item['audio'],
                    item['timestamp'],
                    item.get('speaker_id'),
                    item['metadata']
                )

                # Store result if valid
                if result:
                    with self._results_lock:
                        self._pending_results.append(result)

                # Mark task done
                self._transcription_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Whisper worker {worker_id} stopped")

    def _transcribe_segment(
        self,
        audio_data: np.ndarray,
        timestamp: float,
        speaker_id: Optional[str],
        metadata: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single audio segment.

        Args:
            audio_data: Audio samples (float32, normalized)
            timestamp: Timestamp of audio start
            speaker_id: Speaker identifier
            metadata: Metadata dictionary

        Returns:
            Transcription result dictionary or None
        """
        try:
            start_time = time.time()

            # Transcribe with Whisper
            segments, info = self._model.transcribe(
                audio_data,
                language=None if self.language == 'auto' else self.language,
                initial_prompt=self.initial_prompt,  # Domain vocabulary
                vad_filter=True,  # Use built-in VAD
                word_timestamps=True,
                condition_on_previous_text=False,  # Prevent hallucination propagation
                no_speech_threshold=0.6,  # Higher threshold to avoid hallucinations
                log_prob_threshold=-1.0,  # Filter low-confidence output
            )

            # Extract text and words
            transcription_text = []
            word_segments = []

            for segment in segments:
                transcription_text.append(segment.text)

                # Extract word-level timestamps
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_segments.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })

            full_text = ' '.join(transcription_text).strip()

            # Skip empty transcriptions
            if not full_text:
                return None

            # Filter out known Whisper hallucinations
            if is_hallucination(full_text):
                logger.debug(f"Filtered hallucination: {full_text[:50]}...")
                return None

            transcription_time = time.time() - start_time

            # Calculate average confidence
            avg_confidence = np.mean([
                w['probability'] for w in word_segments
            ]) if word_segments else 0.0

            # Check confidence threshold
            min_confidence = float(os.getenv('MIN_TRANSCRIPTION_CONFIDENCE', '0.5'))
            if avg_confidence < min_confidence:
                logger.debug(
                    f"Transcription confidence too low: {avg_confidence:.2f} < {min_confidence}"
                )
                return None

            result = {
                'timestamp': timestamp,
                'text': full_text,
                'confidence': float(avg_confidence),
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'transcription_time': transcription_time,
                'words': word_segments,
                'speaker_id': speaker_id,
                'metadata': metadata
            }

            logger.debug(
                f"Transcribed: '{full_text[:50]}...' "
                f"(confidence: {avg_confidence:.2f}, time: {transcription_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def get_results(self, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get pending transcription results.

        Args:
            max_results: Maximum number of results to return (None = all)

        Returns:
            List of transcription result dictionaries
        """
        with self._results_lock:
            if max_results:
                results = self._pending_results[:max_results]
                self._pending_results = self._pending_results[max_results:]
            else:
                results = self._pending_results.copy()
                self._pending_results.clear()

            return results

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file (for batch processing).

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription result dictionary
        """
        if not self._model_loaded:
            self.load_model()

        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=None if self.language == 'auto' else self.language,
                initial_prompt=self.initial_prompt,  # Domain vocabulary
                vad_filter=True,
                word_timestamps=True
            )

            full_text = ' '.join([segment.text for segment in segments])

            return {
                'text': full_text,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration
            }

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return {
                'text': '',
                'error': str(e)
            }

    def __enter__(self):
        """Context manager entry."""
        self.start_workers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers()
