"""
Audio system configuration and initialization.

Manages audio device setup, microphone access, and text-to-speech (TTS) engine
configuration for the audio capture and playback subsystem.
"""

try:
    import pyaudio
except ImportError:
    pyaudio = None
import speech_recognition as sr
import pyttsx3
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioConfig:
    def __init__(self):
        self.audio = None
        self.recognizer = None
        self.tts_engine = None
        self.microphone = None
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        if pyaudio:
            self.format = pyaudio.paInt16
        else:
            self.format = None  # Will set dynamically when pyaudio is imported

    def initialize_audio(self) -> bool:
        try:
            if pyaudio is None:
                logger.warning("PyAudio not available, using system PyAudio")
                import pyaudio as pa
                self.audio = pa.PyAudio()
                self.format = pa.paInt16
            else:
                self.audio = pyaudio.PyAudio()
            logger.info("PyAudio initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            return False

    def initialize_microphone(self) -> bool:
        try:
            self.recognizer = sr.Recognizer()

            # Try to use PulseAudio source if available
            available_mics = sr.Microphone.list_microphone_names()
            logger.info(f"Available microphones: {available_mics}")

            if available_mics:
                # Look for RDP source or use first available
                rdp_index = None
                for i, name in enumerate(available_mics):
                    if 'RDP' in name or 'rdp' in name:
                        rdp_index = i
                        break

                mic_index = rdp_index if rdp_index is not None else 0
                self.microphone = sr.Microphone(device_index=mic_index)
                logger.info(f"Using microphone: {available_mics[mic_index]}")
            else:
                # Try default microphone
                self.microphone = sr.Microphone()
                logger.info("Using default microphone")

            # Test microphone access
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Microphone initialized and calibrated")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            return False

    def initialize_tts(self) -> bool:
        try:
            self.tts_engine = pyttsx3.init()

            # Configure TTS settings
            self.tts_engine.setProperty('rate', 150)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume

            # List available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                logger.info(f"Available TTS voices: {len(voices)}")
                # Use first available voice
                self.tts_engine.setProperty('voice', voices[0].id)

            logger.info("TTS engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return False

    def get_audio_info(self) -> Dict[str, Any]:
        info = {
            "pyaudio_initialized": self.audio is not None,
            "microphone_initialized": self.microphone is not None,
            "tts_initialized": self.tts_engine is not None,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "channels": self.channels
        }

        if self.audio:
            info["default_input_device"] = self.audio.get_default_input_device_info()
            info["default_output_device"] = self.audio.get_default_output_device_info()
            info["device_count"] = self.audio.get_device_count()

        return info

    def cleanup(self):
        if self.audio:
            self.audio.terminate()
            logger.info("PyAudio terminated")

        if self.tts_engine:
            self.tts_engine.stop()
            logger.info("TTS engine stopped")