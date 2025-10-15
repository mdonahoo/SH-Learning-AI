#!/usr/bin/env python3
import logging
import sys
import time
from config import AudioConfig
from microphone_test import MicrophoneTest
from speaker_test import SpeakerTest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioTestSuite:
    def __init__(self):
        self.audio_config = AudioConfig()
        self.mic_tester = MicrophoneTest()
        self.speaker_tester = SpeakerTest()
        self.test_results = {}

    def run_configuration_tests(self):
        logger.info("\n" + "="*50)
        logger.info("AUDIO CONFIGURATION TESTS")
        logger.info("="*50)

        results = {}

        # Test PyAudio initialization
        logger.info("\n1. Testing PyAudio initialization...")
        results["pyaudio_init"] = self.audio_config.initialize_audio()

        # Test microphone initialization
        logger.info("\n2. Testing microphone initialization...")
        results["mic_init"] = self.audio_config.initialize_microphone()

        # Test TTS initialization
        logger.info("\n3. Testing TTS initialization...")
        results["tts_init"] = self.audio_config.initialize_tts()

        # Get audio info
        if results["pyaudio_init"]:
            logger.info("\n4. Audio system information:")
            info = self.audio_config.get_audio_info()
            for key, value in info.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for k, v in value.items():
                        logger.info(f"    {k}: {v}")
                else:
                    logger.info(f"  {key}: {value}")

        return results

    def run_microphone_tests(self):
        logger.info("\n" + "="*50)
        logger.info("MICROPHONE TESTS")
        logger.info("="*50)

        return self.mic_tester.run_all_tests()

    def run_speaker_tests(self):
        logger.info("\n" + "="*50)
        logger.info("SPEAKER TESTS")
        logger.info("="*50)

        return self.speaker_tester.run_all_tests()

    def run_integration_test(self):
        logger.info("\n" + "="*50)
        logger.info("INTEGRATION TEST: Speech-to-Text & Text-to-Speech")
        logger.info("="*50)

        try:
            # Initialize components
            if not self.audio_config.initialize_audio():
                logger.error("Failed to initialize audio")
                return False

            if not self.audio_config.initialize_microphone():
                logger.error("Failed to initialize microphone")
                return False

            if not self.audio_config.initialize_tts():
                logger.error("Failed to initialize TTS")
                return False

            # Test TTS output
            logger.info("\nTesting TTS output...")
            test_text = "Audio system initialized. Ready for voice commands."
            self.audio_config.tts_engine.say(test_text)
            self.audio_config.tts_engine.runAndWait()
            logger.info("TTS output successful")

            # Test voice input
            logger.info("\nTesting voice input...")
            logger.info("Please say something in the next 5 seconds...")

            with self.audio_config.microphone as source:
                self.audio_config.recognizer.adjust_for_ambient_noise(source, duration=1)
                try:
                    audio = self.audio_config.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                    logger.info("Audio captured, processing...")

                    # Try recognition
                    text = self.audio_config.recognizer.recognize_google(audio)
                    logger.info(f"Recognized: '{text}'")

                    # Echo back what was heard
                    response = f"I heard you say: {text}"
                    logger.info(f"TTS Response: {response}")
                    self.audio_config.tts_engine.say(response)
                    self.audio_config.tts_engine.runAndWait()

                    return True

                except Exception as e:
                    logger.warning(f"Voice recognition failed: {e}")
                    logger.info("Continuing with TTS-only test...")

                    # At least confirm TTS works
                    self.audio_config.tts_engine.say("Voice input not available, but text to speech is working.")
                    self.audio_config.tts_engine.runAndWait()
                    return False

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
        finally:
            self.audio_config.cleanup()

    def run_all_tests(self):
        logger.info("\n" + "#"*60)
        logger.info("# COMPREHENSIVE AUDIO SYSTEM TEST SUITE")
        logger.info("#"*60)

        all_results = {}

        # Run configuration tests
        config_results = self.run_configuration_tests()
        all_results["configuration"] = config_results

        # Run microphone tests
        mic_results = self.run_microphone_tests()
        all_results["microphone"] = mic_results

        # Run speaker tests
        speaker_results = self.run_speaker_tests()
        all_results["speaker"] = speaker_results

        # Run integration test if basic components work
        if (config_results.get("pyaudio_init") and
            config_results.get("mic_init") and
            config_results.get("tts_init")):

            logger.info("\nBasic components initialized. Running integration test...")
            time.sleep(2)  # Brief pause before integration test
            integration_result = self.run_integration_test()
            all_results["integration"] = integration_result
        else:
            logger.warning("\nSkipping integration test due to initialization failures")
            all_results["integration"] = False

        # Generate summary report
        self.generate_report(all_results)

        return all_results

    def generate_report(self, results):
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY REPORT")
        logger.info("="*60)

        total_tests = 0
        passed_tests = 0

        # Configuration tests
        logger.info("\nConfiguration Tests:")
        for test, result in results.get("configuration", {}).items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"  {test}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1

        # Microphone tests
        logger.info("\nMicrophone Tests:")
        for test, result in results.get("microphone", {}).items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"  {test}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1

        # Speaker tests
        logger.info("\nSpeaker Tests:")
        for test, result in results.get("speaker", {}).items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"  {test}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1

        # Integration test
        if "integration" in results:
            status = "✓ PASSED" if results["integration"] else "✗ FAILED"
            logger.info(f"\nIntegration Test: {status}")
            total_tests += 1
            if results["integration"]:
                passed_tests += 1

        # Overall summary
        logger.info("\n" + "-"*60)
        logger.info(f"OVERALL: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            logger.info("✓ All audio tests PASSED! System is ready.")
        elif passed_tests > 0:
            logger.warning("⚠ Some tests failed. Audio system partially functional.")
        else:
            logger.error("✗ All tests failed. Audio system not functional.")

        # Recommendations
        if passed_tests < total_tests:
            logger.info("\nRecommendations:")
            if not results.get("configuration", {}).get("pyaudio_init"):
                logger.info("  - Check PyAudio installation: pip install pyaudio")
            if not results.get("microphone", {}).get("microphones_available"):
                logger.info("  - Check microphone permissions and PulseAudio configuration")
            if not results.get("speaker", {}).get("tts_initialization"):
                logger.info("  - Check pyttsx3 installation: pip install pyttsx3")
            if not results.get("speaker", {}).get("pyaudio_output"):
                logger.info("  - Check audio output device configuration")


def main():
    test_suite = AudioTestSuite()

    try:
        results = test_suite.run_all_tests()

        # Return exit code based on results
        all_passed = all(
            all(v for v in results.get("configuration", {}).values()),
            all(v for v in results.get("microphone", {}).values()),
            all(v for v in results.get("speaker", {}).values()),
            results.get("integration", False)
        )

        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()