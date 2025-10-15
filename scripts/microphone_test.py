import speech_recognition as sr
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrophoneTest:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None

    def list_microphones(self):
        logger.info("Listing available microphones...")
        mics = sr.Microphone.list_microphone_names()
        for i, mic_name in enumerate(mics):
            logger.info(f"  [{i}] {mic_name}")
        return mics

    def test_microphone_access(self, device_index: Optional[int] = None) -> bool:
        try:
            if device_index is not None:
                self.microphone = sr.Microphone(device_index=device_index)
            else:
                self.microphone = sr.Microphone()

            logger.info("Testing microphone access...")
            with self.microphone as source:
                logger.info("Microphone opened successfully")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Ambient noise calibration complete")
                return True

        except Exception as e:
            logger.error(f"Microphone access failed: {e}")
            return False

    def test_audio_capture(self, duration: int = 3) -> bool:
        if not self.microphone:
            logger.error("Microphone not initialized")
            return False

        try:
            logger.info(f"Recording {duration} seconds of audio...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=duration)
                logger.info(f"Audio captured: {len(audio.frame_data)} bytes")
                return True

        except sr.WaitTimeoutError:
            logger.error("No audio detected within timeout")
            return False
        except Exception as e:
            logger.error(f"Audio capture failed: {e}")
            return False

    def test_speech_recognition(self) -> bool:
        if not self.microphone:
            logger.error("Microphone not initialized")
            return False

        try:
            logger.info("Testing speech recognition...")
            logger.info("Please say something in the next 5 seconds...")

            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                logger.info("Processing speech...")

                # Try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"Speech recognized: '{text}'")
                    return True
                except sr.UnknownValueError:
                    logger.warning("Speech was not understood")
                    return False
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {e}")
                    return False

        except sr.WaitTimeoutError:
            logger.error("No speech detected within timeout")
            return False
        except Exception as e:
            logger.error(f"Speech recognition test failed: {e}")
            return False

    def run_all_tests(self) -> dict:
        results = {
            "microphones_available": False,
            "microphone_access": False,
            "audio_capture": False,
            "speech_recognition": False
        }

        # List microphones
        mics = self.list_microphones()
        results["microphones_available"] = len(mics) > 0

        if results["microphones_available"]:
            # Test microphone access
            # Try RDP source first if available
            device_index = None
            for i, mic_name in enumerate(mics):
                if 'RDP' in mic_name:
                    device_index = i
                    logger.info(f"Using RDP microphone at index {i}")
                    break

            results["microphone_access"] = self.test_microphone_access(device_index)

            if results["microphone_access"]:
                # Test audio capture
                results["audio_capture"] = self.test_audio_capture()

                # Test speech recognition
                results["speech_recognition"] = self.test_speech_recognition()

        return results


if __name__ == "__main__":
    tester = MicrophoneTest()
    results = tester.run_all_tests()

    logger.info("\n=== Microphone Test Results ===")
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\nAll microphone tests passed!")
    else:
        logger.warning("\nSome microphone tests failed. Check audio configuration.")