#!/usr/bin/env python3
"""
Minimal test script to verify WSL2 audio setup matches starship voice assistant configuration
Based on starship_voice_assistant.py lines 36-305
"""

import os
import sys
import subprocess
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ALSA error suppression (replicate lines 13-18 from starship_voice_assistant.py)
try:
    from alsa_error_handler import ALSASuppressor
except ImportError:
    class ALSASuppressor:
        def __enter__(self): return self
        def __exit__(self, *args): return False

class WSL2AudioTester:
    """Test WSL2 audio configuration for voice assistant compatibility"""

    def __init__(self):
        self.test_results = {}

    def setup_audio_environment(self):
        """Replicate audio environment setup from starship_voice_assistant.py lines 36-40"""
        logger.info("🔧 Setting up audio environment...")

        # Set audio backend for container environments (from line 38)
        os.environ['AUDIO_BACKEND'] = 'pulse'
        os.environ['SDL_AUDIODRIVER'] = 'pulse'  # Use PulseAudio for actual playback (line 39)

        logger.info(f"✅ AUDIO_BACKEND set to: {os.environ.get('AUDIO_BACKEND')}")
        logger.info(f"✅ SDL_AUDIODRIVER set to: {os.environ.get('SDL_AUDIODRIVER')}")

        # Check for WSLg PulseAudio environment
        pulse_server = os.environ.get('PULSE_SERVER')
        if not pulse_server:
            # Set the WSLg PulseAudio server path
            os.environ['PULSE_SERVER'] = 'unix:/mnt/wslg/runtime-dir/pulse/native'
            logger.info(f"🔧 Set PULSE_SERVER to: {os.environ['PULSE_SERVER']}")
        else:
            logger.info(f"✅ PULSE_SERVER already set to: {pulse_server}")

    def ensure_pulseaudio(self):
        """Replicate _ensure_pulseaudio method from starship_voice_assistant.py lines 287-305"""
        logger.info("🎵 Checking PulseAudio status...")

        try:
            # Check if PulseAudio is running (line 291)
            result = subprocess.run(['pactl', 'info'], capture_output=True, timeout=1)
            if result.returncode == 0:
                logger.info("✅ PulseAudio already running")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Could not check PulseAudio status: {e}")

        # Try to start PulseAudio (lines 299-303)
        try:
            logger.info("🔄 Attempting to start PulseAudio daemon...")
            subprocess.run(['pulseaudio', '-D', '--exit-idle-time=-1'], capture_output=True, timeout=2)
            time.sleep(1)  # Give it time to start
            logger.info("✅ Started PulseAudio daemon")
            return True
        except Exception as e:
            logger.error(f"❌ Could not start PulseAudio: {e}")
            return False

    def test_pulseaudio_info(self):
        """Test PulseAudio connection and info"""
        logger.info("🔍 Testing PulseAudio info...")

        try:
            result = subprocess.run(['pactl', 'info'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ PulseAudio info retrieved successfully")
                # Extract key info
                lines = result.stdout.split('\n')
                for line in lines[:10]:  # First 10 lines contain key info
                    if line.strip():
                        logger.info(f"   {line}")
                self.test_results['pulseaudio_info'] = True
                return True
            else:
                logger.error(f"❌ pactl info failed: {result.stderr}")
                self.test_results['pulseaudio_info'] = False
                return False
        except Exception as e:
            logger.error(f"❌ PulseAudio info test failed: {e}")
            self.test_results['pulseaudio_info'] = False
            return False

    def test_audio_devices(self):
        """Test audio device detection"""
        logger.info("🎤 Testing audio device detection...")

        try:
            # List audio sources (microphones)
            result = subprocess.run(['pactl', 'list', 'sources'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                sources = [line for line in result.stdout.split('\n') if 'Name:' in line]
                logger.info(f"✅ Found {len(sources)} audio sources:")
                for source in sources[:3]:  # Limit to first 3
                    logger.info(f"   {source.strip()}")
                self.test_results['audio_sources'] = len(sources)
                return len(sources) > 0
            else:
                logger.error(f"❌ Could not list audio sources: {result.stderr}")
                self.test_results['audio_sources'] = 0
                return False
        except Exception as e:
            logger.error(f"❌ Audio device test failed: {e}")
            self.test_results['audio_sources'] = 0
            return False

    def test_pyaudio_import(self):
        """Test PyAudio import and basic functionality"""
        logger.info("🐍 Testing PyAudio import...")

        try:
            with ALSASuppressor():
                import pyaudio

            # Test PyAudio initialization
            with ALSASuppressor():
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                logger.info(f"✅ PyAudio initialized successfully, found {device_count} devices")

                # List input devices
                input_devices = []
                for i in range(device_count):
                    try:
                        info = p.get_device_info_by_index(i)
                        if info['maxInputChannels'] > 0:
                            input_devices.append(f"Device {i}: {info['name']}")
                    except:
                        continue

                logger.info(f"✅ Found {len(input_devices)} input devices:")
                for device in input_devices[:3]:  # Limit to first 3
                    logger.info(f"   {device}")

                p.terminate()
                self.test_results['pyaudio'] = True
                return True

        except ImportError as e:
            logger.error(f"❌ PyAudio not available: {e}")
            self.test_results['pyaudio'] = False
            return False
        except Exception as e:
            logger.error(f"❌ PyAudio test failed: {e}")
            self.test_results['pyaudio'] = False
            return False

    def test_pygame_init(self):
        """Test pygame initialization (used in starship voice assistant)"""
        logger.info("🎮 Testing pygame initialization...")

        try:
            with ALSASuppressor():
                import pygame
                pygame.init()

            logger.info("✅ Pygame initialized successfully")
            self.test_results['pygame'] = True
            return True
        except ImportError as e:
            logger.error(f"❌ Pygame not available: {e}")
            self.test_results['pygame'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Pygame initialization failed: {e}")
            self.test_results['pygame'] = False
            return False

    def test_voice_assistant_core_import(self):
        """Test importing voice assistant core components"""
        logger.info("🗣️ Testing voice assistant core import...")

        try:
            # Add tests directory to path (from starship_voice_assistant.py line 46)
            sys.path.append('/workspaces/LocalAss/tests')

            with ALSASuppressor():
                from voice_assistant_core import VoiceAssistantCore

            logger.info("✅ VoiceAssistantCore imported successfully")
            self.test_results['voice_core'] = True
            return True
        except ImportError as e:
            logger.error(f"❌ Could not import VoiceAssistantCore: {e}")
            self.test_results['voice_core'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Voice assistant core test failed: {e}")
            self.test_results['voice_core'] = False
            return False

    def test_basic_audio_recording(self):
        """Test basic audio recording capability"""
        logger.info("🎙️ Testing basic audio recording...")

        try:
            with ALSASuppressor():
                import pyaudio

            # Test parameters from voice assistant
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100

            with ALSASuppressor():
                p = pyaudio.PyAudio()

                # Try to open a stream
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )

                logger.info("✅ Audio stream opened successfully")

                # Record a small sample
                logger.info("🎙️ Recording 1 second of audio...")
                data = stream.read(RATE)  # 1 second
                logger.info(f"✅ Recorded {len(data)} bytes of audio data")

                stream.stop_stream()
                stream.close()
                p.terminate()

                self.test_results['recording'] = True
                return True

        except Exception as e:
            logger.error(f"❌ Audio recording test failed: {e}")
            self.test_results['recording'] = False
            return False

    def run_all_tests(self):
        """Run all audio tests"""
        logger.info("🚀 Starting WSL2 Audio Setup Tests")
        logger.info("=" * 50)

        tests = [
            ("Environment Setup", self.setup_audio_environment),
            ("PulseAudio Availability", self.ensure_pulseaudio),
            ("PulseAudio Info", self.test_pulseaudio_info),
            ("Audio Device Detection", self.test_audio_devices),
            ("PyAudio Import", self.test_pyaudio_import),
            ("Pygame Initialization", self.test_pygame_init),
            ("Voice Assistant Core", self.test_voice_assistant_core_import),
            ("Basic Audio Recording", self.test_basic_audio_recording)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")

        logger.info("=" * 50)
        logger.info(f"🏁 Test Results: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All tests passed! WSL2 audio setup is working correctly.")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed. Check configuration above.")

        return self.test_results

def main():
    """Main test runner"""
    tester = WSL2AudioTester()
    results = tester.run_all_tests()

    print("\n" + "=" * 50)
    print("📊 SUMMARY OF TEST RESULTS")
    print("=" * 50)

    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:<20}: {status}")

    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())