#!/usr/bin/env python3
"""
Download and cache Whisper models for offline bridge audio transcription.

Usage:
    python scripts/download_whisper_models.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_size: str, device: str = "cpu", compute_type: str = "int8") -> bool:
    """
    Download and cache a Whisper model.

    Args:
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (cpu or cuda)
        compute_type: Compute precision (int8, float16, float32)

    Returns:
        True if successful
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        logger.error("Or: pip install -r requirements.txt")
        return False

    logger.info(f"\nDownloading Whisper model: {model_size}")
    logger.info(f"Device: {device}, Compute type: {compute_type}")

    model_path = Path(os.getenv("WHISPER_MODEL_PATH", "./data/models/whisper/"))
    model_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Initializing model (this will download if not cached)...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(model_path)
        )

        # Test model with dummy audio
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        segments, info = model.transcribe(dummy_audio)
        list(segments)  # Force transcription to complete

        logger.info(f"✓ Model {model_size} downloaded and verified")
        logger.info(f"✓ Model stored in: {model_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        return False


def main():
    """Main entry point."""
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    logger.info("="*60)
    logger.info("Starship Horizons - Whisper Model Downloader")
    logger.info("="*60)
    logger.info("\nThis will download the Whisper model for offline use.")
    logger.info("The model will be cached and reused for future sessions.")
    logger.info("")

    success = download_model(model_size, device, compute_type)

    if success:
        logger.info("\n" + "="*60)
        logger.info("✓ Ready for bridge audio transcription!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("  1. Test audio: python scripts/test_realtime_audio.py")
        logger.info("  2. Start recording: python scripts/record_mission_with_audio.py")
    else:
        logger.error("\n✗ Model download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
