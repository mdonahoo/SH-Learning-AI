#!/usr/bin/env python3
"""
Run the audio analysis web server.

Starts the FastAPI server for audio transcription and analysis.
Provides a web interface for recording and uploading audio files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Configure logging before importing app modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Start the web server."""
    parser = argparse.ArgumentParser(
        description='Start the audio analysis web server'
    )
    parser.add_argument(
        '--host',
        default=os.getenv('WEB_SERVER_HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('WEB_SERVER_PORT', '8000')),
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--preload-model',
        action='store_true',
        help='Preload Whisper model on startup'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )

    args = parser.parse_args()

    # Set preload environment variable
    if args.preload_model:
        os.environ['PRELOAD_WHISPER'] = 'true'

    try:
        import uvicorn

        logger.info(f"Starting web server at http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop")

        uvicorn.run(
            "src.web.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )

    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
