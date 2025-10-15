#!/usr/bin/env python3
"""
Script purpose and description.

This script [what it does]. It is used for [when/why to use it].

Usage:
    python script_name.py [options]

Examples:
    python script_name.py --host 192.168.1.100 --port 1865
    python script_name.py --config config.yaml --verbose
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
from dotenv import load_dotenv

# Local imports (after adding to path)
from src.integration import GameClient
from src.metrics import EventRecorder

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_requested = True


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Script description',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Connection arguments
    parser.add_argument(
        '--host',
        type=str,
        default=os.getenv('GAME_HOST', 'localhost'),
        help='Server hostname or IP address'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('GAME_PORT', '1864')),
        help='Server port number'
    )

    # Operation arguments
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Operation timeout in seconds'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Output file path'
    )

    # Optional flags
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making actual changes'
    )

    return parser.parse_args()


async def main_logic(args: argparse.Namespace) -> int:
    """
    Main script logic.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Set up verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Display configuration
    logger.info(f"Starting script with configuration:")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Timeout: {args.timeout}s")

    # Initialize components
    client = None
    recorder = None

    try:
        # Create client instance
        client = GameClient(host=args.host, port=args.port)
        recorder = EventRecorder()

        # Connect to server
        logger.info("Connecting to server...")
        if not await client.connect():
            logger.error("Failed to connect to server")
            return 1

        logger.info("Successfully connected")

        # Main operation loop
        start_time = datetime.now()
        operation_count = 0

        while not shutdown_requested:
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= args.timeout:
                logger.info(f"Timeout reached ({args.timeout}s)")
                break

            # Perform main operation
            try:
                # Your main logic here
                data = await client.fetch_data()
                if data:
                    recorder.record_event(data)
                    operation_count += 1

                # Small delay to avoid spinning
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Operation error: {e}")
                # Decide whether to continue or abort
                if args.verbose:
                    logger.exception("Full error details:")

        # Save results
        if not args.dry_run:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            recorder.save_to_file(output_path)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.info("Dry run - no files saved")

        # Summary
        logger.info(f"Script completed successfully")
        logger.info(f"  Operations: {operation_count}")
        logger.info(f"  Duration: {elapsed:.1f}s")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Full error details:")
        return 1

    finally:
        # Cleanup
        if client:
            logger.info("Disconnecting client...")
            await client.disconnect()
        if recorder:
            recorder.cleanup()
        logger.info("Cleanup completed")


def main():
    """Script entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse arguments
    args = parse_arguments()

    # Run async main logic
    try:
        exit_code = asyncio.run(main_logic(args))
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()