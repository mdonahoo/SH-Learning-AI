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

    # CRITICAL: "warp" not "work" - this is a sci-fi bridge simulator
    "IMPORTANT: In this context, 'warp' is a navigation term, NOT 'work'. "
    "Warp terms: warp, warp drive, warp to, go to warp, warp factor, warping, "
    "warp jammer, warp jamming, warp jammed, my warp is jammed, warp has jammed, "
    "warp speed, maximum warp, engage warp, drop out of warp, out of warp, "
    "warp one, warp two, warp three, warp point, warp point three, waypoint. "
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
    "Beam frequency: terahertz (NOT hertz or earths), "
    "800 terahertz, set to 800 terahertz, set your laser to 800 terahertz, set lasers to 800 terahertz, "
    "500 terahertz, 460 terahertz, 100 terahertz, 200 terahertz, 300 terahertz, "
    "frequency in terahertz, beam frequency terahertz, shield frequency terahertz. "
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

    # Check for repeated 2-word phrases (e.g., "all right all right all right")
    # This catches common Whisper looping on short phrases
    if len(words) >= 4:
        for start in range(len(words) - 3):
            phrase = ' '.join(words[start:start + 2])
            if len(phrase) >= 5:  # Minimum 5 chars for 2-word phrase
                # Count occurrences of this 2-word phrase
                count = 0
                for i in range(0, len(words) - 1, 2):
                    if ' '.join(words[i:i + 2]) == phrase:
                        count += 1
                if count >= 3 and count * 2 >= len(words) * 0.75:
                    # 2-word phrase repeated 3+ times and makes up 75%+ of text
                    logger.debug(f"Detected 2-word phrase repetition: '{phrase}' x{count}")
                    return True

    # Check for repeated phrases (e.g., "We're going to buzz in close." repeated)
    # This catches Whisper's looping hallucination pattern
    if len(text_clean) > 50:
        # Look for repeated 3-6 word phrases
        for phrase_len in range(3, 7):
            if len(words) >= phrase_len * 3:  # Need at least 3 repetitions
                for start in range(len(words) - phrase_len * 2):
                    phrase = ' '.join(words[start:start + phrase_len])
                    if len(phrase) < 7:  # Skip very short phrases (was 10, now 7)
                        continue
                    # Count how many times this phrase appears
                    count = text_lower.count(phrase)
                    if count >= 3:
                        # Phrase repeated 3+ times is likely hallucination
                        logger.debug(f"Detected repeated phrase hallucination: '{phrase}' x{count}")
                        return True

    return False


def clean_repetitive_text(text: str) -> str:
    """
    Remove repetitive phrases from text that indicate Whisper looping.

    This cleans up text where Whisper got stuck repeating phrases like:
    "We're going to buzz in close. We're going to buzz in close. We're going to..."

    Args:
        text: Text to clean

    Returns:
        Cleaned text with repetitions removed
    """
    if not text or len(text) < 50:
        return text

    import re

    # Split into sentences for sentence-level deduplication
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Detect and remove repeated sentences (only if we have multiple sentences)
    cleaned = []
    prev_sentence = None
    repeat_count = 0

    for sentence in sentences:
        sentence_normalized = sentence.lower().strip()
        # Skip very short fragments
        if len(sentence_normalized) < 10:
            cleaned.append(sentence)
            prev_sentence = sentence_normalized
            continue

        if sentence_normalized == prev_sentence:
            repeat_count += 1
            # Allow up to 1 natural repeat, filter more
            if repeat_count <= 1:
                cleaned.append(sentence)
        else:
            cleaned.append(sentence)
            repeat_count = 0

        prev_sentence = sentence_normalized

    result = ' '.join(cleaned)

    # Detect repeated phrases by splitting on comma/period and deduplicating
    import re

    # Split on common delimiters (comma, period, semicolon)
    parts = re.split(r'([,;.!?])', result)

    # Remove consecutive duplicate parts
    deduped_parts = []
    prev_content = None
    repeat_count = 0
    skip_next_delimiter = False

    for part in parts:
        part_stripped = part.strip()
        part_normalized = part_stripped.lower()

        # Handle delimiters
        if len(part_stripped) <= 1 and part_stripped in ',;.!?':
            if skip_next_delimiter:
                skip_next_delimiter = False
                continue  # Skip delimiter after skipped content
            deduped_parts.append(part)
            continue

        # Skip empty parts
        if not part_stripped:
            continue

        # Check for repeat
        if part_normalized == prev_content:
            repeat_count += 1
            # Allow 1 natural repeat, skip more
            if repeat_count <= 1:
                deduped_parts.append(part)
            else:
                skip_next_delimiter = True  # Skip the comma after this
        else:
            deduped_parts.append(part)
            repeat_count = 0
            prev_content = part_normalized

    result = ''.join(deduped_parts)

    # Clean up any resulting issues
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'[,;]{2,}', ',', result)
    result = re.sub(r'\.\s*\.', '.', result)

    return result.strip()


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

        # Cross-segment repetition detection
        # Track recent transcription word sets to detect Whisper looping
        self._recent_texts: List[set] = []  # Stores word sets for overlap comparison
        self._max_recent_texts = 10  # Track last 10 segments
        self._repetition_threshold = 5  # 5+ similar segments = likely loop

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

                # Store result if valid and not a cross-segment repetition loop
                if result:
                    text = result.get('text', '').strip().lower()
                    # Remove punctuation for comparison
                    text_words = set(
                        ''.join(c for c in text if c.isalnum() or c.isspace()).split()
                    )
                    is_loop = False

                    with self._results_lock:
                        # Check for cross-segment repetition (Whisper looping)
                        if text_words and len(text) < 100:
                            # Count segments with high word overlap (80%+)
                            similar_count = 0
                            for recent_words in self._recent_texts:
                                if recent_words:
                                    # Calculate word overlap between segments
                                    overlap = len(text_words & recent_words)
                                    min_len = min(len(text_words), len(recent_words))
                                    # 80% overlap = similar segment (likely loop)
                                    if min_len > 0 and overlap / min_len >= 0.8:
                                        similar_count += 1

                            if similar_count >= self._repetition_threshold:
                                logger.warning(
                                    f"Detected cross-segment loop: '{text[:50]}' "
                                    f"similar to {similar_count} recent segments, skipping"
                                )
                                is_loop = True

                        if not is_loop:
                            self._pending_results.append(result)
                            # Track recent word sets for loop detection
                            self._recent_texts.append(text_words)
                            if len(self._recent_texts) > self._max_recent_texts:
                                self._recent_texts.pop(0)

                    if is_loop:
                        segments_failed += 1
                    else:
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
            # Anti-hallucination settings prevent looping on long/unclear audio
            segments, info = self._model.transcribe(
                audio_data,
                language=None if self.language == 'auto' else self.language,
                initial_prompt=self.initial_prompt,  # Domain vocabulary
                vad_filter=False,  # Disable - we already do VAD upstream
                word_timestamps=True,
                condition_on_previous_text=False,  # Critical: prevents loop propagation
                no_speech_threshold=0.6,  # Higher = more lenient speech detection
                log_prob_threshold=-1.0,  # Filter low-confidence output
                compression_ratio_threshold=2.4,  # Reject highly repetitive segments
                beam_size=5,  # Larger beam = better accuracy (default is 5)
                best_of=5,  # Consider more candidates for better accuracy
                temperature=0.0,  # Deterministic output for consistency
                repetition_penalty=1.5,  # Stronger penalty for repeated phrases
                no_repeat_ngram_size=3,  # Prevent 3+ word phrases from repeating
                hallucination_silence_threshold=0.5,  # Skip text during detected silence
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
                word_timestamps=True,
                # Anti-hallucination settings for long recordings
                condition_on_previous_text=False,  # Critical: prevents loop propagation
                repetition_penalty=1.5,  # Penalize repeated tokens
                no_repeat_ngram_size=3,  # Prevent 3+ word phrases from repeating
                compression_ratio_threshold=2.4,  # Reject highly repetitive segments
                log_prob_threshold=-1.0,  # Filter low-confidence output
                no_speech_threshold=0.6,
                temperature=0.0,  # Deterministic output
                hallucination_silence_threshold=0.5,  # Skip text during silence (prevents loops)
            )

            # Filter out repetitive/hallucinated segments
            filtered_segments = []
            for segment in segments:
                text = segment.text.strip()
                # Clean repetition within the segment first
                text = clean_repetitive_text(text)
                # Skip if segment looks like hallucination
                if text and not is_hallucination(text):
                    filtered_segments.append(text)
                else:
                    logger.debug(f"Filtered hallucinated segment: {text[:50]}...")

            full_text = ' '.join(filtered_segments)
            # Final pass to clean any remaining repetition in joined text
            full_text = clean_repetitive_text(full_text)

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

    def transcribe_file_chunked(
        self,
        audio_path: str,
        chunk_duration: int = 300,
        min_chunk_duration: int = 120,
        max_chunk_duration: int = 420,
        overlap_seconds: float = 2.0,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Transcribe long audio files by processing in chunks.

        This method prevents Whisper hallucination loops by:
        - Splitting audio into manageable chunks at silence boundaries
        - Processing each chunk independently with fresh model state
        - Using timeout protection per chunk

        Args:
            audio_path: Path to audio file
            chunk_duration: Target chunk size in seconds (default 5 minutes)
            min_chunk_duration: Minimum chunk size in seconds (default 2 minutes)
            max_chunk_duration: Maximum chunk size in seconds (default 7 minutes)
            overlap_seconds: Overlap between chunks to avoid cutting words
            progress_callback: Optional callback(chunk_num, total_chunks, progress_pct)

        Returns:
            Combined transcription result with all segments
        """
        if not self._model_loaded:
            self.load_model()

        try:
            from pydub import AudioSegment
            from pydub.silence import detect_silence
        except ImportError:
            logger.warning("pydub not available, falling back to non-chunked transcription")
            return self.transcribe_file(audio_path)

        try:
            # Load audio file
            logger.info(f"Loading audio file for chunked transcription: {audio_path}")
            audio = AudioSegment.from_file(audio_path)
            total_duration_ms = len(audio)
            total_duration_sec = total_duration_ms / 1000.0

            logger.info(f"Audio duration: {total_duration_sec:.1f}s ({total_duration_sec/60:.1f} minutes)")

            # If audio is shorter than 2x chunk duration, process normally
            if total_duration_sec < chunk_duration * 2:
                logger.info("Audio is short enough for single-pass transcription")
                return self.transcribe_file(audio_path)

            # Detect silence points for natural chunk boundaries
            # Silence detection: min 500ms of silence at -40dB
            silence_ranges = detect_silence(
                audio,
                min_silence_len=500,
                silence_thresh=-40
            )

            # Convert silence ranges to candidate split points (midpoints)
            split_candidates = []
            for start_ms, end_ms in silence_ranges:
                midpoint = (start_ms + end_ms) / 2
                split_candidates.append(midpoint)

            # Generate chunk boundaries
            chunk_boundaries = self._calculate_chunk_boundaries(
                total_duration_ms=total_duration_ms,
                split_candidates=split_candidates,
                target_chunk_ms=chunk_duration * 1000,
                min_chunk_ms=min_chunk_duration * 1000,
                max_chunk_ms=max_chunk_duration * 1000
            )

            logger.info(f"Splitting audio into {len(chunk_boundaries)} chunks")

            # Process each chunk
            all_segments = []
            chunk_results = []
            import tempfile
            import os as os_module

            for i, (start_ms, end_ms) in enumerate(chunk_boundaries):
                chunk_num = i + 1
                total_chunks = len(chunk_boundaries)

                # Add overlap at start (except first chunk)
                actual_start_ms = max(0, start_ms - overlap_seconds * 1000) if i > 0 else start_ms
                # Add overlap at end (except last chunk)
                actual_end_ms = min(total_duration_ms, end_ms + overlap_seconds * 1000) if i < len(chunk_boundaries) - 1 else end_ms

                chunk_audio = audio[actual_start_ms:actual_end_ms]
                chunk_duration_sec = len(chunk_audio) / 1000.0

                logger.info(
                    f"Processing chunk {chunk_num}/{total_chunks}: "
                    f"{start_ms/1000:.1f}s - {end_ms/1000:.1f}s "
                    f"(duration: {chunk_duration_sec:.1f}s)"
                )

                # Report progress
                if progress_callback:
                    progress_pct = int((chunk_num - 1) / total_chunks * 100)
                    try:
                        progress_callback(chunk_num, total_chunks, progress_pct)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

                # Export chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    chunk_path = tmp_file.name

                try:
                    # Export as WAV with correct format for Whisper
                    chunk_audio = chunk_audio.set_channels(1)
                    chunk_audio = chunk_audio.set_frame_rate(16000)
                    chunk_audio = chunk_audio.set_sample_width(2)
                    chunk_audio.export(chunk_path, format='wav')

                    # Transcribe chunk with timeout protection
                    chunk_result = self._transcribe_chunk_with_timeout(
                        chunk_path,
                        timeout_seconds=chunk_duration_sec * 2,  # 2x duration as timeout
                        chunk_num=chunk_num
                    )

                    if chunk_result and chunk_result.get('segments'):
                        # Adjust timestamps by chunk offset
                        time_offset = start_ms / 1000.0  # Convert to seconds

                        for seg in chunk_result['segments']:
                            # Adjust for overlap - skip segments that started in previous chunk
                            seg_adjusted_start = seg['start'] + time_offset - (overlap_seconds if i > 0 else 0)

                            # Skip if segment is before this chunk's actual start (in overlap region)
                            if i > 0 and seg['start'] < overlap_seconds:
                                continue

                            adjusted_seg = {
                                'start': seg_adjusted_start,
                                'end': seg['end'] + time_offset - (overlap_seconds if i > 0 else 0),
                                'text': seg['text'],
                                'confidence': seg.get('confidence', 0),
                                'words': []
                            }

                            # Adjust word timestamps if present
                            if seg.get('words'):
                                for word in seg['words']:
                                    adjusted_seg['words'].append({
                                        'word': word['word'],
                                        'start': word['start'] + time_offset - (overlap_seconds if i > 0 else 0),
                                        'end': word['end'] + time_offset - (overlap_seconds if i > 0 else 0),
                                        'probability': word.get('probability', 0)
                                    })

                            all_segments.append(adjusted_seg)

                        chunk_results.append({
                            'chunk_num': chunk_num,
                            'start_time': start_ms / 1000.0,
                            'end_time': end_ms / 1000.0,
                            'segment_count': len(chunk_result['segments']),
                            'text_length': len(chunk_result.get('text', ''))
                        })
                    else:
                        logger.warning(f"Chunk {chunk_num} produced no valid segments")

                finally:
                    # Clean up temp file
                    try:
                        os_module.remove(chunk_path)
                    except Exception:
                        pass

            # Final progress update
            if progress_callback:
                try:
                    progress_callback(len(chunk_boundaries), len(chunk_boundaries), 100)
                except Exception:
                    pass

            # Combine all text
            full_text = ' '.join(seg['text'] for seg in all_segments)

            # Final cleanup pass for any remaining repetition
            full_text = clean_repetitive_text(full_text)

            logger.info(
                f"Chunked transcription complete: "
                f"{len(all_segments)} segments from {len(chunk_boundaries)} chunks"
            )

            return {
                'text': full_text,
                'segments': all_segments,
                'duration': total_duration_sec,
                'chunk_count': len(chunk_boundaries),
                'chunk_results': chunk_results,
                'language': self.language
            }

        except Exception as e:
            logger.error(f"Chunked transcription failed: {e}", exc_info=True)
            # Fall back to regular transcription
            logger.info("Falling back to non-chunked transcription")
            return self.transcribe_file(audio_path)

    def _calculate_chunk_boundaries(
        self,
        total_duration_ms: int,
        split_candidates: List[float],
        target_chunk_ms: int,
        min_chunk_ms: int,
        max_chunk_ms: int
    ) -> List[tuple]:
        """
        Calculate chunk boundaries using silence points as natural breaks.

        Args:
            total_duration_ms: Total audio duration in milliseconds
            split_candidates: List of silence midpoint timestamps (ms)
            target_chunk_ms: Target chunk duration in ms
            min_chunk_ms: Minimum chunk duration in ms
            max_chunk_ms: Maximum chunk duration in ms

        Returns:
            List of (start_ms, end_ms) tuples for each chunk
        """
        boundaries = []
        current_start = 0

        while current_start < total_duration_ms:
            # Target end point
            target_end = current_start + target_chunk_ms

            # If we're near the end, just include the rest
            if target_end >= total_duration_ms - min_chunk_ms:
                boundaries.append((current_start, total_duration_ms))
                break

            # Find the best split point near target
            best_split = target_end  # Default if no silence found

            # Look for silence points within acceptable range
            min_acceptable = current_start + min_chunk_ms
            max_acceptable = current_start + max_chunk_ms

            candidates_in_range = [
                sp for sp in split_candidates
                if min_acceptable <= sp <= max_acceptable
            ]

            if candidates_in_range:
                # Find candidate closest to target
                best_split = min(
                    candidates_in_range,
                    key=lambda x: abs(x - target_end)
                )

            # Ensure we don't exceed total duration
            best_split = min(best_split, total_duration_ms)

            boundaries.append((current_start, best_split))
            current_start = best_split

        return boundaries

    def _transcribe_chunk_with_timeout(
        self,
        chunk_path: str,
        timeout_seconds: float,
        chunk_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single chunk with timeout protection.

        Args:
            chunk_path: Path to chunk audio file
            timeout_seconds: Maximum time allowed for transcription
            chunk_num: Chunk number for logging

        Returns:
            Transcription result or None if failed/timeout
        """
        import concurrent.futures

        def _do_transcribe():
            """Inner function to run transcription."""
            segments, info = self._model.transcribe(
                chunk_path,
                language=None if self.language == 'auto' else self.language,
                initial_prompt=self.initial_prompt,
                vad_filter=True,
                word_timestamps=True,
                # Critical anti-hallucination settings
                condition_on_previous_text=False,  # Fresh state per chunk
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                temperature=0.0,
                hallucination_silence_threshold=0.5,
            )

            # Collect and filter segments
            result_segments = []
            texts = []

            for segment in segments:
                text = segment.text.strip()

                # Clean repetition within segment
                text = clean_repetitive_text(text)

                # Skip hallucinations
                if not text or is_hallucination(text):
                    continue

                seg_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': text,
                    'confidence': getattr(segment, 'avg_logprob', 0),
                    'words': []
                }

                if hasattr(segment, 'words') and segment.words:
                    seg_data['words'] = [
                        {
                            'word': w.word,
                            'start': w.start,
                            'end': w.end,
                            'probability': w.probability
                        }
                        for w in segment.words
                    ]

                result_segments.append(seg_data)
                texts.append(text)

            return {
                'segments': result_segments,
                'text': ' '.join(texts),
                'duration': info.duration if hasattr(info, 'duration') else 0
            }

        # Run with timeout
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_transcribe)
                result = future.result(timeout=timeout_seconds)
                return result

        except concurrent.futures.TimeoutError:
            logger.warning(
                f"Chunk {chunk_num} transcription timed out after {timeout_seconds:.1f}s"
            )
            return None

        except Exception as e:
            logger.error(f"Chunk {chunk_num} transcription failed: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start_workers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers()
