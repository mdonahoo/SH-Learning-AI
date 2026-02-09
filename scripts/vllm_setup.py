#!/usr/bin/env python3
"""
vLLM setup and management for local LLM inference.

Manages a vLLM server process for high-throughput GPU-accelerated inference
using quantized models. Designed to coexist with Ollama on separate ports.

Usage:
    python scripts/vllm_setup.py start      # Start vLLM server
    python scripts/vllm_setup.py stop       # Stop vLLM server
    python scripts/vllm_setup.py status     # Check server status
    python scripts/vllm_setup.py models     # List available models
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
DEFAULT_MODEL = 'Qwen/Qwen2.5-3B-Instruct-AWQ'
DEFAULT_PORT = 8100
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_MODEL_LEN = 16384
VLLM_LOG_PATH = '/tmp/vllm.log'
VLLM_PID_PATH = '/tmp/vllm.pid'


def _get_config() -> Dict[str, str]:
    """
    Read vLLM configuration from environment variables.

    Returns:
        Dictionary with model, port, gpu_memory_utilization, max_model_len,
        tensor_parallel_size, pipeline_parallel_size, data_parallel_size,
        quantization, enforce_eager
    """
    return {
        'model': os.getenv('VLLM_MODEL', DEFAULT_MODEL),
        'port': os.getenv('VLLM_PORT', str(DEFAULT_PORT)),
        'gpu_memory_utilization': os.getenv(
            'VLLM_GPU_MEMORY_UTILIZATION',
            str(DEFAULT_GPU_MEMORY_UTILIZATION)
        ),
        'max_model_len': os.getenv('VLLM_MAX_MODEL_LEN', str(DEFAULT_MAX_MODEL_LEN)),
        'tensor_parallel_size': os.getenv('VLLM_TENSOR_PARALLEL_SIZE', '1'),
        'pipeline_parallel_size': os.getenv('VLLM_PIPELINE_PARALLEL_SIZE', '1'),
        'data_parallel_size': os.getenv('VLLM_DATA_PARALLEL_SIZE', '1'),
        'quantization': os.getenv('VLLM_QUANTIZATION', 'awq'),
        'enforce_eager': os.getenv('VLLM_ENFORCE_EAGER', 'true'),
    }


def _get_vllm_pid() -> Optional[int]:
    """
    Read the vLLM server PID from the PID file.

    Returns:
        PID as integer, or None if no PID file or process not running
    """
    try:
        pid_path = Path(VLLM_PID_PATH)
        if not pid_path.exists():
            return None
        pid = int(pid_path.read_text().strip())
        # Check if process is actually running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        # PID file exists but process is gone — clean up stale file
        try:
            Path(VLLM_PID_PATH).unlink(missing_ok=True)
        except OSError:
            pass
        return None


def is_vllm_running() -> bool:
    """Check if vLLM server process is running."""
    return _get_vllm_pid() is not None


def _probe_endpoint(port: str) -> Tuple[bool, List[str]]:
    """
    Probe the vLLM OpenAI-compatible endpoint for available models.

    Args:
        port: Port number as string

    Returns:
        Tuple of (is_reachable, list_of_model_ids)
    """
    try:
        url = f"http://localhost:{port}/v1/models"
        req = urllib.request.Request(url, method='GET')
        req.add_header('Accept', 'application/json')

        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                models = [
                    m.get('id', '') for m in data.get('data', [])
                ]
                return True, models

        return False, []
    except (urllib.error.URLError, ConnectionRefusedError, OSError, json.JSONDecodeError):
        return False, []


def start_server() -> bool:
    """
    Start vLLM server in background.

    Launches ``vllm serve`` as a detached subprocess, writing its PID to
    /tmp/vllm.pid and logs to /tmp/vllm.log. The model is downloaded
    from HuggingFace on the first run (cached in ~/.cache/huggingface).

    Returns:
        True if server started successfully
    """
    if is_vllm_running():
        logger.info("✓ vLLM server already running")
        return True

    config = _get_config()
    model = config['model']
    port = config['port']
    gpu_mem = config['gpu_memory_utilization']
    max_len = config['max_model_len']
    quantization = config['quantization']
    enforce_eager = config['enforce_eager'].lower() == 'true'
    tp = config['tensor_parallel_size']
    pp = config['pipeline_parallel_size']
    dp = config['data_parallel_size']

    logger.info(f"Starting vLLM server...")
    logger.info(f"  Model: {model}")
    logger.info(f"  Port: {port}")
    logger.info(f"  GPU memory utilization: {gpu_mem}")
    logger.info(f"  Max model length: {max_len}")
    logger.info(f"  Quantization: {quantization}")
    if int(tp) > 1:
        logger.info(f"  Tensor parallel: {tp}")
    if int(pp) > 1:
        logger.info(f"  Pipeline parallel: {pp}")
    if int(dp) > 1:
        logger.info(f"  Data parallel: {dp}")
    logger.info(f"  Enforce eager: {enforce_eager}")
    logger.info(f"  Log: {VLLM_LOG_PATH}")

    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model,
        '--port', port,
        '--gpu-memory-utilization', gpu_mem,
        '--max-model-len', max_len,
        '--trust-remote-code',
    ]

    # Quantization (omit for FP16 when set to "none")
    if quantization.lower() != 'none':
        cmd.extend(['--quantization', quantization])

    # Enforce eager mode (skip torch.compile to save VRAM)
    if enforce_eager:
        cmd.append('--enforce-eager')

    # Multi-GPU parallelism flags (only when > 1)
    if int(tp) > 1:
        cmd.extend(['--tensor-parallel-size', tp])
    if int(pp) > 1:
        cmd.extend(['--pipeline-parallel-size', pp])
    if int(dp) > 1:
        cmd.extend(['--data-parallel-size', dp])

    try:
        log_file = open(VLLM_LOG_PATH, 'w')
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Write PID file
        Path(VLLM_PID_PATH).write_text(str(process.pid))

        logger.info(f"  PID: {process.pid}")
        logger.info("Waiting for server to initialize (model download may take several minutes)...")

        # Wait for the endpoint to become reachable
        for i in range(120):
            time.sleep(2)

            # Check if process died
            if process.poll() is not None:
                logger.error(f"✗ vLLM process exited with code {process.returncode}")
                logger.error(f"  Check {VLLM_LOG_PATH} for details")
                Path(VLLM_PID_PATH).unlink(missing_ok=True)
                return False

            reachable, _ = _probe_endpoint(port)
            if reachable:
                logger.info(f"✓ vLLM server started on port {port}")
                return True

            if i % 15 == 14:
                logger.info(f"  Still starting... ({(i + 1) * 2}s elapsed)")

        logger.error(f"✗ vLLM failed to start within 240s — check {VLLM_LOG_PATH}")
        return False

    except FileNotFoundError:
        logger.error("✗ vLLM not installed. Run: pip install vllm")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to start vLLM: {e}")
        return False


def stop_server() -> bool:
    """
    Stop the vLLM server process.

    Sends SIGTERM, waits briefly, then SIGKILL if needed.

    Returns:
        True if server stopped successfully
    """
    pid = _get_vllm_pid()
    if pid is None:
        logger.info("vLLM server not running")
        return True

    try:
        logger.info(f"Stopping vLLM server (PID {pid})...")
        os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown
        for _ in range(10):
            time.sleep(1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                Path(VLLM_PID_PATH).unlink(missing_ok=True)
                logger.info("✓ vLLM server stopped")
                return True

        # Force kill if still running
        logger.warning("Sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        Path(VLLM_PID_PATH).unlink(missing_ok=True)
        logger.info("✓ vLLM server killed")
        return True

    except ProcessLookupError:
        Path(VLLM_PID_PATH).unlink(missing_ok=True)
        logger.info("✓ vLLM server already stopped")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to stop vLLM: {e}")
        return False


def show_status() -> None:
    """Show vLLM server status and configuration."""
    config = _get_config()
    port = config['port']

    logger.info("=== vLLM Status ===")

    pid = _get_vllm_pid()
    if pid is not None:
        logger.info(f"Server: ✓ Running (PID {pid})")
    else:
        logger.info("Server: ✗ Not running")

    # Endpoint check
    reachable, models = _probe_endpoint(port)
    if reachable:
        logger.info(f"Endpoint: ✓ Reachable at http://localhost:{port}/v1")
        if models:
            logger.info(f"Serving: {', '.join(models)}")
    else:
        logger.info(f"Endpoint: ✗ Not reachable at http://localhost:{port}/v1")

    # Configuration
    logger.info(f"\n=== Configuration ===")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Port: {port}")
    logger.info(f"GPU memory utilization: {config['gpu_memory_utilization']}")
    logger.info(f"Max model length: {config['max_model_len']}")
    logger.info(f"Quantization: {config['quantization']}")
    logger.info(f"Enforce eager: {config['enforce_eager']}")
    logger.info(f"Log file: {VLLM_LOG_PATH}")
    logger.info(f"PID file: {VLLM_PID_PATH}")

    # Parallelism
    tp = config['tensor_parallel_size']
    pp = config['pipeline_parallel_size']
    dp = config['data_parallel_size']
    logger.info(f"\n=== Parallelism ===")
    logger.info(f"Tensor parallel: {tp}")
    logger.info(f"Pipeline parallel: {pp}")
    logger.info(f"Data parallel: {dp}")

    # GPU memory usage
    _show_gpu_memory()


def _show_gpu_memory() -> None:
    """Display current GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"\n=== GPU Memory ===")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                used_mb = (total_bytes - free_bytes) / (1024 * 1024)
                total_mb = total_bytes / (1024 * 1024)
                logger.info(
                    f"GPU {i} ({name}): {used_mb:.0f} MB / {total_mb:.0f} MB "
                    f"({used_mb / total_mb * 100:.1f}% used)"
                )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU memory query failed: {e}")


def list_models() -> None:
    """List models available at the vLLM endpoint."""
    config = _get_config()
    port = config['port']

    reachable, models = _probe_endpoint(port)
    if not reachable:
        logger.info(f"vLLM endpoint not reachable at http://localhost:{port}/v1")
        logger.info("Start the server with: python scripts/vllm_setup.py start")
        return

    if models:
        logger.info("Available models:")
        for model_id in models:
            logger.info(f"  - {model_id}")
    else:
        logger.info("No models currently served")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='vLLM setup and management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/vllm_setup.py start       # Start vLLM server
  python scripts/vllm_setup.py stop        # Stop vLLM server
  python scripts/vllm_setup.py status      # Check status & GPU usage
  python scripts/vllm_setup.py models      # List served models

Environment variables:
  VLLM_MODEL                    Model to serve (default: Qwen/Qwen2.5-3B-Instruct-AWQ)
  VLLM_PORT                     Port to listen on (default: 8100)
  VLLM_GPU_MEMORY_UTILIZATION   Fraction of GPU memory to use (default: 0.85)
  VLLM_MAX_MODEL_LEN            Max sequence length (default: 16384)
  VLLM_QUANTIZATION             Quantization method: awq, gptq, none (default: awq)
  VLLM_ENFORCE_EAGER            Skip torch.compile (default: true)
  VLLM_TENSOR_PARALLEL_SIZE     Split model across N GPUs (default: 1)
  VLLM_PIPELINE_PARALLEL_SIZE   Split layers across N GPUs (default: 1)
  VLLM_DATA_PARALLEL_SIZE       Run N model copies (default: 1)
  VLLM_AUTO_START               Auto-start with web server (default: false)

To use vLLM as the default LLM backend, set in .env:
  LLM_BASE_URL=http://localhost:8100/v1
  LLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
        """
    )

    parser.add_argument(
        'action',
        choices=['start', 'stop', 'status', 'models'],
        help='Action to perform'
    )

    args = parser.parse_args()

    if args.action == 'start':
        success = start_server()
    elif args.action == 'stop':
        success = stop_server()
    elif args.action == 'status':
        show_status()
        success = True
    elif args.action == 'models':
        list_models()
        success = True
    else:
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
