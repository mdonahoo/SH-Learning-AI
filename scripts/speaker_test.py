import pyttsx3
try:
    import pyaudio
except ImportError:
    pyaudio = None
import numpy as np
import wave
import io
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerTest:
    def __init__(self):
        self.tts_engine = None
        self.pyaudio = None

    def test_pyaudio_output(self) -> bool:
        try:
            logger.info("Testing PyAudio output devices...")
            if pyaudio is None:
                import pyaudio as pa
                self.pyaudio = pa.PyAudio()
            else:
                self.pyaudio = pyaudio.PyAudio()

            # List output devices
            output_devices = []
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    output_devices.append((i, info['name']))
                    logger.info(f"  Output device [{i}]: {info['name']}")

            if not output_devices:
                logger.warning("No output devices found")
                return False

            # Test tone generation on default output
            logger.info("Generating test tone...")
            duration = 1.0  # seconds
            frequency = 440.0  # Hz (A4 note)
            sample_rate = 44100
            samples = int(sample_rate * duration)

            # Generate sine wave
            t = np.linspace(0, duration, samples, False)
            wave_data = np.sin(frequency * t * 2 * np.pi)

            # Scale to 16-bit range
            wave_data = (wave_data * 32767).astype(np.int16)

            # Play the tone
            pa_format = pyaudio.paInt16 if pyaudio else pa.paInt16
            stream = self.pyaudio.open(
                format=pa_format,
                channels=1,
                rate=sample_rate,
                output=True
            )

            stream.write(wave_data.tobytes())
            stream.stop_stream()
            stream.close()

            logger.info("Test tone played successfully")
            return True

        except Exception as e:
            logger.error(f"PyAudio output test failed: {e}")
            return False
        finally:
            if self.pyaudio:
                self.pyaudio.terminate()

    def test_tts_initialization(self) -> bool:
        try:
            logger.info("Initializing TTS engine...")
            self.tts_engine = pyttsx3.init()

            # Get TTS properties
            voices = self.tts_engine.getProperty('voices')
            rate = self.tts_engine.getProperty('rate')
            volume = self.tts_engine.getProperty('volume')

            logger.info(f"TTS initialized successfully")
            logger.info(f"  Available voices: {len(voices) if voices else 0}")
            logger.info(f"  Speech rate: {rate}")
            logger.info(f"  Volume: {volume}")

            if voices:
                for i, voice in enumerate(voices[:3]):  # Show first 3 voices
                    logger.info(f"  Voice {i}: {voice.name}")

            return True

        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            return False

    def test_tts_speech(self, text: str = "Hello, this is a test of the text to speech system.") -> bool:
        if not self.tts_engine:
            logger.error("TTS engine not initialized")
            return False

        try:
            logger.info(f"Testing TTS speech: '{text}'")

            # Configure TTS settings
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)

            # Use first available voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)

            # Speak the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

            logger.info("TTS speech completed successfully")
            return True

        except Exception as e:
            logger.error(f"TTS speech test failed: {e}")
            return False

    def test_tts_to_file(self, text: str = "Test audio file", filename: str = "/tmp/test_audio.wav") -> bool:
        if not self.tts_engine:
            logger.error("TTS engine not initialized")
            return False

        try:
            logger.info(f"Testing TTS to file: '{filename}'")

            # Save to file
            self.tts_engine.save_to_file(text, filename)
            self.tts_engine.runAndWait()

            # Verify file exists and has content
            import os
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                logger.info(f"Audio file created: {file_size} bytes")
                return file_size > 0
            else:
                logger.error("Audio file not created")
                return False

        except Exception as e:
            logger.error(f"TTS to file test failed: {e}")
            return False

    def run_all_tests(self) -> dict:
        results = {
            "pyaudio_output": False,
            "tts_initialization": False,
            "tts_speech": False,
            "tts_to_file": False
        }

        # Test PyAudio output
        results["pyaudio_output"] = self.test_pyaudio_output()

        # Test TTS initialization
        results["tts_initialization"] = self.test_tts_initialization()

        if results["tts_initialization"]:
            # Test TTS speech
            results["tts_speech"] = self.test_tts_speech()

            # Test TTS to file
            results["tts_to_file"] = self.test_tts_to_file()

        # Cleanup
        if self.tts_engine:
            self.tts_engine.stop()

        return results


if __name__ == "__main__":
    tester = SpeakerTest()
    results = tester.run_all_tests()

    logger.info("\n=== Speaker Test Results ===")
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\nAll speaker tests passed!")
    else:
        logger.warning("\nSome speaker tests failed. Check audio configuration.")