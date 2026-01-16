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
    "Star systems: Paravera, Calavera, Caravera, Faraday, Parrot Arrow, Starbase Delta. "
    "Galaxy map, galaxy view, zoom in, zoom out. "

    # Alien races and factions
    "Alien races: Craylor, Craylord, Craylords, cralor, Kralien, Torgoth, Arvonian, Ximni, Skaraan. "
    "Human Navy, human navy headquarters, fleet admiral, fleet command, fleet disposition. "
    "Treaty: treaty, the treaty, Craylor treaty, human Craylor treaty, human Craylord treaty, "
    "held a treaty, treaty for years, treaty is in effect. "
    "Neutral zone, blue neutral border zone, border patrol, tensions rising. "

    # Ship names and designations - Hedge is critical (often misheard)
    "Player ship: Hedge, USS Hedge, the Hedge, of Hedge, crew of Hedge. "
    "Common phrases: captain and crew of Hedge, commanding officer of Hedge, "
    "greetings captain and crew of Hedge, to the commanding officer of Hedge. "
    "Enemy ships: Lutren, Lutrench, Belgore, MSF, Wickert, Phobos, core trailer, chord trailer. "
    "Ship designations: 215, 302, 307, 310, 22, 32, R2. "

    # Warp and navigation - critical distinctions
    "Warp terms: warp jammer, warp jamming, warp jammed, warp champed, my warp is jammed. "
    "War: war declared, war has been declared, war depicted. "

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

    # Navigation terms - warp is critical (not 'work')
    "Navigation: waypoint, sector, bearing, mark, coordinates, heading, "
    "waypoint one, waypoint two, waypoint three, waypoint four, waypoint five, "
    "course plotted, course laid in, set course, tap set course, engage, "
    "all stop, full stop, full impulse, half impulse, one quarter impulse, quarter impulse, "
    "all back, back us out, reverse, "
    "warp, go to warp, warp there, warp drive, maximum warp, warp factor, drop out of warp, de-warp, "
    "ready to warp, warping, we warp, warp one, warp two, best speed, recommend warp two, "
    "ETA, coming about, closing distance, intercept course, evasive maneuvers. "
    "Maneuvers: combat maneuver, flyby, fly past, bring us around, turn us around, hell turn. "

    # UI interaction terms
    "UI commands: tap, tap the system, tap a thing, scan it, select, click, swipe. "

    # Communications and docking
    "Communications: comm station, hail, hailing frequencies, channel open, "
    "transmitting, receiving, subspace, distress signal, distress call. "
    "Docking: Space Dock, docking permission, undock, undock us, undocking, "
    "request permission, back us out, meters clear, hundred meters. "

    # Alerts and conditions
    "Alerts: red alert, condition red, yellow alert, condition yellow, "
    "battle stations, all hands, shields up, shields down, "
    "general quarters, action stations, stand down. "

    # Combat terminology
    "Combat: fire, open fire, cease fire, fire at will, weapons free, "
    "attack pattern, attack pattern Delta, attack pattern Alpha, "
    "weapons hot, target locked, lock on, targeting, target them, retarget, "
    "direct hit, hull damage, hull breach, shields failing, shields holding, "
    "torpedoes away, firing phasers, beam weapons, projectile weapons, "
    "missile lock, countermeasures, point defense. "

    # Weapons and tubes - critical for accuracy
    "Tubes: left tube, right tube, load tubes, put in tubes, unload, "
    "install the right tube, launch the missile, launch left tube, launch right tube. "
    "Missiles: homing missiles, homings, nukes, EMPs, EMP, mines, torpedoes. "
    "Beam frequency: terahertz, 800 terahertz, set to 800 terahertz, set your laser to 800 terahertz, "
    "500 terahertz, 460 terahertz, 100 terahertz, 200 terahertz, frequency in terahertz. "
    "Calibrate: calibrate, recalibrate, calibrate the shields, calibrate to their shields. "
    "Beam ship: beam ship, we are a beam ship, we are primarily a beam ship, "
    "primarily a beam ship, we are a beam vessel, our beams. "

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

    # Crew coordination and scanning
    "Crew: speak from diaphragm, bridge crew, away team, "
    "triangulate, scan the planet, scan the planets, scan complete, scan results, "
    "balance power, power distribution, boost shields, boost weapons, "
    "route power, divert power, emergency power. "

    # Mission briefing and commands
    "Briefing: here's the briefing, greetings captain, greetings captain and crew, "
    "your mission, patrol the border, patrol the border area, "
    "relieved of command, relieved of fleet command duties, "
    "relay officer's directives, imminent intrusion, imminent Craylor intrusion. "

    # Station-specific commands
    "Science: scan, scanning, get a scan, scan the contacts, scan on, "
    "call sign, interpretation, type, ship type, weak against. "
    "Engineering: boost to beams, boosting our beam weapons, energy is down. "
    "Weapons: stand by to engage, alternate targets, focus on, switch to. "
    "Helm: undock, undock us, take us to, copy that, copy down. "

    # Objects and entities
    "Objects: drones, interact with drones, probes, asteroids, debris, "
    "planets, moons, stations, ships, contacts. "
)

# Alternative shorter prompt if the full one causes issues
STARSHIP_HORIZONS_PROMPT_SHORT = (
    "Starship bridge simulator. Stations: helm, tactical, science, engineering, ops. "
    "Terms: warp, impulse, shields, phasers, torpedoes, bearing, mark, heading. "
    "Commands: engage, fire, on screen, aye captain, red alert, shields up. "
    "Races: Craylor, Craylord, human Craylord treaty, Kralien, Torgoth. "
    "Ships: Hedge, Lutren, Belgore, MSF, warp jammer. "
    "Weapons: left tube, right tube, homing missiles, nukes, EMPs, terahertz. "
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

    Only filters obvious hallucinations - errs on the side of keeping speech.

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

    # Check for very short text that's likely noise (less than 2 chars)
    if len(text_clean) < 2:
        return True

    # Check for exact match hallucinations (single words/short phrases)
    # Only filter if the ENTIRE transcription matches
    if text_clean in WHISPER_HALLUCINATIONS_EXACT:
        return True

    # Check for known hallucination patterns - only if they're the START of the text
    # or if they make up most of the text (not just appearing somewhere in speech)
    for hallucination in WHISPER_HALLUCINATIONS:
        # Skip single-word patterns that could be in normal speech
        if len(hallucination.split()) == 1 and len(hallucination) < 6:
            continue
        # Only filter if hallucination is at the start or is >80% of the text
        if text_lower.startswith(hallucination):
            return True
        if len(hallucination) > len(text_clean) * 0.8:
            if hallucination in text_lower:
                return True

    # Check for repeated single word (e.g., "the the the the")
    words = text_clean.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return True

    # Check for excessive repetition (e.g., "right right right right right")
    if len(words) >= 5:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        max_count = max(word_counts.values())
        if max_count >= len(words) * 0.8:  # 80% same word
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
        # Default to 'large-v3' model for best accuracy (YouTube-quality transcription)
        self.model_size = model_size or os.getenv('WHISPER_MODEL_SIZE', 'large-v3')

        # Auto-detect GPU if not specified
        env_device = os.getenv('WHISPER_DEVICE', 'auto')
        if device:
            self.device = device
        elif env_device != 'auto':
            self.device = env_device
        else:
            self.device = self._detect_device()

        # Set compute type based on device
        env_compute = os.getenv('WHISPER_COMPUTE_TYPE', 'auto')
        if compute_type:
            self.compute_type = compute_type
        elif env_compute != 'auto':
            self.compute_type = env_compute
        else:
            self.compute_type = 'float16' if self.device == 'cuda' else 'int8'
        self.language = language or os.getenv('TRANSCRIBE_LANGUAGE', 'en')
        self.num_workers = num_workers or int(os.getenv('TRANSCRIPTION_WORKERS', '4'))

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
            maxsize=int(os.getenv('MAX_SEGMENT_QUEUE_SIZE', '1000'))
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

    def _detect_device(self) -> str:
        """
        Auto-detect the best available device (CUDA or CPU).

        Returns:
            'cuda' if GPU is available and working, 'cpu' otherwise
        """
        try:
            import torch
            if torch.cuda.is_available():
                # Test that CUDA actually works
                try:
                    torch.zeros(1).cuda()
                    logger.info("CUDA detected and working, using GPU")
                    return 'cuda'
                except Exception as e:
                    logger.warning(f"CUDA available but not working: {e}")
                    return 'cpu'
            else:
                logger.info("CUDA not available, using CPU")
                return 'cpu'
        except ImportError:
            logger.info("PyTorch not available for GPU detection, using CPU")
            return 'cpu'

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
        segments_processed = 0
        segments_failed = 0

        while self._is_running:
            item = None
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
                    segments_processed += 1
                else:
                    segments_failed += 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                segments_failed += 1
            finally:
                # ALWAYS mark task done to prevent queue from getting stuck
                if item is not None:
                    try:
                        self._transcription_queue.task_done()
                    except ValueError:
                        pass  # Already marked done

        logger.info(
            f"Whisper worker {worker_id} stopped - "
            f"processed: {segments_processed}, failed: {segments_failed}"
        )

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
            # Settings optimized for maximum accuracy with large-v3 model
            segments, info = self._model.transcribe(
                audio_data,
                language=None if self.language == 'auto' else self.language,
                initial_prompt=self.initial_prompt,  # Domain vocabulary
                vad_filter=False,  # Disable - we already do VAD upstream
                word_timestamps=True,
                condition_on_previous_text=False,  # Prevent hallucination propagation
                no_speech_threshold=0.6,  # Higher = more lenient speech detection
                log_prob_threshold=-1.0,  # Filter low-confidence output
                beam_size=5,  # Larger beam = better accuracy (default is 5)
                best_of=5,  # Consider more candidates for better accuracy
                temperature=0.0,  # Deterministic output for consistency
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
            min_confidence = float(os.getenv('MIN_TRANSCRIPTION_CONFIDENCE', '0.3'))
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
