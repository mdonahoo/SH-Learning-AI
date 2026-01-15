#!/usr/bin/env python3
"""
Ollama setup and management for local LLM inference.

Usage:
    python scripts/ollama_setup.py start      # Start Ollama server
    python scripts/ollama_setup.py pull       # Pull default model
    python scripts/ollama_setup.py status     # Check server status
    python scripts/ollama_setup.py list       # List available models
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Default models by size/capability
MODELS = {
    'small': 'llama3.2',           # ~2GB, fast
    'medium': 'qwen2.5:7b',        # ~4GB, balanced
    'large': 'qwen2.5:14b-instruct' # ~9GB, high quality
}


def is_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-x', 'ollama'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def start_server() -> bool:
    """Start Ollama server in background."""
    if is_ollama_running():
        logger.info("✓ Ollama server already running")
        return True

    logger.info("Starting Ollama server...")
    try:
        # Start in background
        subprocess.Popen(
            ['ollama', 'serve'],
            stdout=open('/tmp/ollama.log', 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

        # Wait for startup
        for i in range(10):
            time.sleep(1)
            if is_ollama_running():
                logger.info("✓ Ollama server started (log: /tmp/ollama.log)")
                return True
            logger.info(f"  Waiting for server... ({i+1}/10)")

        logger.error("✗ Ollama failed to start - check /tmp/ollama.log")
        return False

    except FileNotFoundError:
        logger.error("✗ Ollama not installed. Run: curl -fsSL https://ollama.com/install.sh | sh")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to start Ollama: {e}")
        return False


def stop_server() -> bool:
    """Stop Ollama server."""
    if not is_ollama_running():
        logger.info("Ollama server not running")
        return True

    try:
        subprocess.run(['pkill', '-x', 'ollama'], timeout=5)
        time.sleep(1)
        if not is_ollama_running():
            logger.info("✓ Ollama server stopped")
            return True
        logger.warning("Server may still be running")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to stop Ollama: {e}")
        return False


def pull_model(model: str = None) -> bool:
    """Pull a model from Ollama registry."""
    if not model:
        model = os.getenv('OLLAMA_MODEL', MODELS['small'])

    if not is_ollama_running():
        logger.info("Starting Ollama server first...")
        if not start_server():
            return False

    logger.info(f"Pulling model: {model}")
    logger.info("This may take several minutes depending on model size...")

    try:
        result = subprocess.run(
            ['ollama', 'pull', model],
            timeout=1800  # 30 minutes max
        )
        if result.returncode == 0:
            logger.info(f"✓ Model '{model}' ready")
            return True
        else:
            logger.error(f"✗ Failed to pull model '{model}'")
            return False
    except subprocess.TimeoutExpired:
        logger.error("✗ Download timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ Error pulling model: {e}")
        return False


def list_models() -> None:
    """List locally available models."""
    if not is_ollama_running():
        logger.warning("Ollama server not running - starting...")
        start_server()

    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            if result.stdout.strip():
                logger.info("Available models:")
                print(result.stdout)
            else:
                logger.info("No models installed. Run: python scripts/ollama_setup.py pull")
        else:
            logger.error("Failed to list models")
    except Exception as e:
        logger.error(f"Error listing models: {e}")


def show_status() -> None:
    """Show Ollama server status and configuration."""
    logger.info("=== Ollama Status ===")

    # Server status
    if is_ollama_running():
        logger.info("Server: ✓ Running")
    else:
        logger.info("Server: ✗ Not running")

    # Configuration
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    model = os.getenv('OLLAMA_MODEL', 'llama3.2')
    models_dir = os.getenv('OLLAMA_MODELS', '~/.ollama/models')

    logger.info(f"Host: {host}")
    logger.info(f"Model: {model}")
    logger.info(f"Models dir: {models_dir}")

    # Available models
    logger.info("\n=== Recommended Models ===")
    for size, name in MODELS.items():
        logger.info(f"  {size}: {name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ollama setup and management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ollama_setup.py start          # Start server
  python scripts/ollama_setup.py pull           # Pull model from .env
  python scripts/ollama_setup.py pull llama3.2  # Pull specific model
  python scripts/ollama_setup.py status         # Check status
  python scripts/ollama_setup.py list           # List installed models
  python scripts/ollama_setup.py stop           # Stop server
        """
    )

    parser.add_argument(
        'action',
        choices=['start', 'stop', 'pull', 'list', 'status'],
        help='Action to perform'
    )
    parser.add_argument(
        'model',
        nargs='?',
        help='Model name for pull action (optional)'
    )

    args = parser.parse_args()

    if args.action == 'start':
        success = start_server()
    elif args.action == 'stop':
        success = stop_server()
    elif args.action == 'pull':
        success = pull_model(args.model)
    elif args.action == 'list':
        list_models()
        success = True
    elif args.action == 'status':
        show_status()
        success = True
    else:
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
