"""
Hardware capability detection for GPU-aware parallel analysis.

Detects available GPUs, CPU cores, system RAM, and Ollama availability
to inform the parallel analysis pipeline's execution strategy.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU device."""

    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory_mb / 1024.0

    @property
    def free_memory_gb(self) -> float:
        """Free memory in GB."""
        return self.free_memory_mb / 1024.0


@dataclass
class HardwareProfile:
    """
    Cached hardware capability profile.

    Attributes:
        gpu_count: Number of CUDA GPUs available
        gpus: Per-GPU information (name, memory)
        cpu_count: Number of logical CPU cores
        ram_total_mb: Total system RAM in MB
        ram_available_mb: Available system RAM in MB
        ollama_available: Whether Ollama server is reachable
        ollama_models: List of model names available on Ollama
        vllm_available: Whether vLLM server is reachable
        vllm_models: List of model IDs available on vLLM
        detected_at: Timestamp when profile was created
    """

    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu_count: int = 1
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    ollama_available: bool = False
    ollama_models: List[str] = field(default_factory=list)
    vllm_available: bool = False
    vllm_models: List[str] = field(default_factory=list)
    detected_at: float = 0.0

    @property
    def has_gpu(self) -> bool:
        """Whether any GPU is available."""
        return self.gpu_count > 0

    @property
    def has_multi_gpu(self) -> bool:
        """Whether multiple GPUs are available."""
        return self.gpu_count >= 2

    @property
    def ram_total_gb(self) -> float:
        """Total RAM in GB."""
        return self.ram_total_mb / 1024.0

    @property
    def ram_available_gb(self) -> float:
        """Available RAM in GB."""
        return self.ram_available_mb / 1024.0

    def summary(self) -> str:
        """Human-readable hardware summary."""
        parts = []

        if self.gpu_count > 0:
            gpu_names = [g.name for g in self.gpus]
            gpu_mem = sum(g.total_memory_mb for g in self.gpus)
            parts.append(
                f"GPU: {self.gpu_count}x ({', '.join(gpu_names)}, "
                f"{gpu_mem / 1024:.1f} GB VRAM total)"
            )
        else:
            parts.append("GPU: none")

        parts.append(f"CPU: {self.cpu_count} cores")
        parts.append(f"RAM: {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB available")

        if self.ollama_available:
            parts.append(f"Ollama: available ({len(self.ollama_models)} models)")
        else:
            parts.append("Ollama: not available")

        if self.vllm_available:
            parts.append(f"vLLM: available ({len(self.vllm_models)} models)")
        else:
            parts.append("vLLM: not available")

        return " | ".join(parts)


class HardwareDetector:
    """
    Centralized hardware capability detection.

    Detects GPUs, CPU cores, RAM, and Ollama availability.
    Results are cached and can be refreshed on demand.
    """

    def __init__(self) -> None:
        """Initialize hardware detector."""
        self._profile: Optional[HardwareProfile] = None
        self._ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self._vllm_port = os.getenv('VLLM_PORT', '8100')
        logger.debug("HardwareDetector initialized")

    @property
    def profile(self) -> HardwareProfile:
        """
        Get cached hardware profile, detecting on first access.

        Returns:
            HardwareProfile with current system capabilities
        """
        if self._profile is None:
            self._profile = self.detect()
        return self._profile

    def detect(self) -> HardwareProfile:
        """
        Run full hardware detection and cache the result.

        Returns:
            Fresh HardwareProfile
        """
        profile = HardwareProfile(detected_at=time.time())

        # Detect GPU
        gpu_count, gpus = self._detect_gpus()
        profile.gpu_count = gpu_count
        profile.gpus = gpus

        # Detect CPU
        profile.cpu_count = self._detect_cpu_count()

        # Detect RAM
        total_mb, available_mb = self._detect_ram()
        profile.ram_total_mb = total_mb
        profile.ram_available_mb = available_mb

        # Probe Ollama
        available, models = self._probe_ollama()
        profile.ollama_available = available
        profile.ollama_models = models

        # Probe vLLM
        vllm_available, vllm_models = self._probe_vllm()
        profile.vllm_available = vllm_available
        profile.vllm_models = vllm_models

        self._profile = profile

        # Log summary
        logger.info(f"Hardware profile: {profile.summary()}")

        return profile

    def refresh(self) -> HardwareProfile:
        """
        Force re-detection of hardware capabilities.

        Returns:
            Fresh HardwareProfile
        """
        self._profile = None
        return self.detect()

    def _detect_gpus(self) -> tuple:
        """
        Detect CUDA GPUs via PyTorch.

        Returns:
            Tuple of (gpu_count, list_of_GPUInfo)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.debug("CUDA not available")
                return 0, []

            gpu_count = torch.cuda.device_count()
            gpus: List[GPUInfo] = []

            for i in range(gpu_count):
                try:
                    name = torch.cuda.get_device_name(i)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                    gpus.append(GPUInfo(
                        index=i,
                        name=name,
                        total_memory_mb=total_bytes // (1024 * 1024),
                        free_memory_mb=free_bytes // (1024 * 1024),
                    ))
                except Exception as e:
                    logger.warning(f"Failed to query GPU {i}: {e}")
                    gpus.append(GPUInfo(
                        index=i,
                        name=f"GPU {i} (query failed)",
                        total_memory_mb=0,
                        free_memory_mb=0,
                    ))

            return gpu_count, gpus

        except ImportError:
            logger.debug("PyTorch not installed, no GPU detection")
            return 0, []
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return 0, []

    def _detect_cpu_count(self) -> int:
        """
        Detect number of logical CPU cores.

        Returns:
            Number of CPU cores (minimum 1)
        """
        count = os.cpu_count()
        return count if count and count > 0 else 1

    def _detect_ram(self) -> tuple:
        """
        Detect system RAM by parsing /proc/meminfo.

        Returns:
            Tuple of (total_mb, available_mb), both 0 if detection fails
        """
        try:
            total_kb = 0
            available_kb = 0

            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        total_kb = int(line.split()[1])
                    elif line.startswith('MemAvailable:'):
                        available_kb = int(line.split()[1])

            return total_kb // 1024, available_kb // 1024

        except FileNotFoundError:
            logger.debug("/proc/meminfo not found (non-Linux system)")
            return 0, 0
        except Exception as e:
            logger.warning(f"RAM detection failed: {e}")
            return 0, 0

    def _probe_ollama(self) -> tuple:
        """
        Check if Ollama server is reachable and list available models.

        Returns:
            Tuple of (is_available, list_of_model_names)
        """
        try:
            import urllib.request
            import json

            url = f"{self._ollama_host}/api/tags"
            req = urllib.request.Request(url, method='GET')
            req.add_header('Accept', 'application/json')

            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    models = [
                        m.get('name', '') for m in data.get('models', [])
                    ]
                    return True, models

            return False, []

        except Exception as e:
            logger.debug(f"Ollama probe failed: {e}")
            return False, []

    def _probe_vllm(self) -> tuple:
        """
        Check if vLLM server is reachable and list available models.

        Probes the OpenAI-compatible ``/v1/models`` endpoint.

        Returns:
            Tuple of (is_available, list_of_model_ids)
        """
        try:
            import urllib.request
            import json

            url = f"http://localhost:{self._vllm_port}/v1/models"
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

        except Exception as e:
            logger.debug(f"vLLM probe failed: {e}")
            return False, []
