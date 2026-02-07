"""Tests for hardware capability detection module."""

import time
from unittest.mock import patch, MagicMock, mock_open

import pytest

from src.hardware.detector import (
    HardwareDetector,
    HardwareProfile,
    GPUInfo,
)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_memory_conversions(self):
        """Test MB to GB conversion properties."""
        gpu = GPUInfo(index=0, name="Test GPU", total_memory_mb=8192, free_memory_mb=6144)
        assert gpu.total_memory_gb == 8.0
        assert gpu.free_memory_gb == 6.0

    def test_zero_memory(self):
        """Test zero memory values."""
        gpu = GPUInfo(index=0, name="Test GPU", total_memory_mb=0, free_memory_mb=0)
        assert gpu.total_memory_gb == 0.0
        assert gpu.free_memory_gb == 0.0


class TestHardwareProfile:
    """Tests for HardwareProfile dataclass."""

    def test_defaults(self):
        """Test default profile has no hardware."""
        profile = HardwareProfile()
        assert profile.gpu_count == 0
        assert profile.has_gpu is False
        assert profile.has_multi_gpu is False
        assert profile.cpu_count == 1
        assert profile.ram_total_mb == 0
        assert profile.ollama_available is False

    def test_has_gpu(self):
        """Test GPU detection properties."""
        profile = HardwareProfile(gpu_count=1, gpus=[
            GPUInfo(0, "RTX 4090", 24576, 20000)
        ])
        assert profile.has_gpu is True
        assert profile.has_multi_gpu is False

    def test_has_multi_gpu(self):
        """Test multi-GPU detection."""
        profile = HardwareProfile(gpu_count=2, gpus=[
            GPUInfo(0, "RTX 4090", 24576, 20000),
            GPUInfo(1, "RTX 4090", 24576, 20000),
        ])
        assert profile.has_gpu is True
        assert profile.has_multi_gpu is True

    def test_ram_conversion(self):
        """Test RAM MB to GB conversion."""
        profile = HardwareProfile(ram_total_mb=32768, ram_available_mb=16384)
        assert profile.ram_total_gb == 32.0
        assert profile.ram_available_gb == 16.0

    def test_summary_no_gpu(self):
        """Test summary string with no GPU."""
        profile = HardwareProfile(
            cpu_count=8,
            ram_total_mb=16384,
            ram_available_mb=8192,
        )
        summary = profile.summary()
        assert "GPU: none" in summary
        assert "CPU: 8 cores" in summary
        assert "16.0 GB total" in summary

    def test_summary_with_gpu(self):
        """Test summary string with GPU."""
        profile = HardwareProfile(
            gpu_count=1,
            gpus=[GPUInfo(0, "RTX 4090", 24576, 20000)],
            cpu_count=16,
            ram_total_mb=65536,
            ram_available_mb=32768,
            ollama_available=True,
            ollama_models=["llama3.2", "qwen2.5"],
        )
        summary = profile.summary()
        assert "GPU: 1x" in summary
        assert "RTX 4090" in summary
        assert "Ollama: available (2 models)" in summary


class TestHardwareDetector:
    """Tests for HardwareDetector."""

    def test_lazy_profile(self):
        """Test that profile is detected lazily on first access."""
        detector = HardwareDetector()
        assert detector._profile is None

        with patch.object(detector, 'detect') as mock_detect:
            mock_detect.return_value = HardwareProfile(cpu_count=4)
            profile = detector.profile
            mock_detect.assert_called_once()
            assert profile.cpu_count == 4

    def test_cached_profile(self):
        """Test that profile is cached after first detection."""
        detector = HardwareDetector()
        cached = HardwareProfile(cpu_count=8)
        detector._profile = cached

        with patch.object(detector, 'detect') as mock_detect:
            profile = detector.profile
            mock_detect.assert_not_called()
            assert profile.cpu_count == 8

    def test_refresh_clears_cache(self):
        """Test that refresh forces re-detection."""
        detector = HardwareDetector()
        detector._profile = HardwareProfile(cpu_count=4)

        with patch.object(detector, 'detect') as mock_detect:
            mock_detect.return_value = HardwareProfile(cpu_count=16)
            profile = detector.refresh()
            mock_detect.assert_called_once()
            assert profile.cpu_count == 16

    @patch('src.hardware.detector.HardwareDetector._probe_ollama')
    @patch('src.hardware.detector.HardwareDetector._detect_ram')
    @patch('src.hardware.detector.HardwareDetector._detect_cpu_count')
    @patch('src.hardware.detector.HardwareDetector._detect_gpus')
    def test_detect_assembles_profile(
        self, mock_gpus, mock_cpu, mock_ram, mock_ollama
    ):
        """Test that detect() assembles all hardware info."""
        mock_gpus.return_value = (1, [GPUInfo(0, "GTX 1080", 8192, 6000)])
        mock_cpu.return_value = 12
        mock_ram.return_value = (32768, 24000)
        mock_ollama.return_value = (True, ["llama3.2"])

        detector = HardwareDetector()
        profile = detector.detect()

        assert profile.gpu_count == 1
        assert len(profile.gpus) == 1
        assert profile.cpu_count == 12
        assert profile.ram_total_mb == 32768
        assert profile.ram_available_mb == 24000
        assert profile.ollama_available is True
        assert profile.ollama_models == ["llama3.2"]
        assert profile.detected_at > 0

    def test_detect_gpus_no_torch(self):
        """Test GPU detection when PyTorch is not installed."""
        detector = HardwareDetector()
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError("No torch")):
                count, gpus = detector._detect_gpus()
                assert count == 0
                assert gpus == []

    def test_detect_gpus_no_cuda(self):
        """Test GPU detection when CUDA is not available."""
        detector = HardwareDetector()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict('sys.modules', {'torch': mock_torch}):
            count, gpus = detector._detect_gpus()
            assert count == 0
            assert gpus == []

    def test_detect_cpu_count(self):
        """Test CPU count detection."""
        detector = HardwareDetector()
        with patch('os.cpu_count', return_value=16):
            assert detector._detect_cpu_count() == 16

    def test_detect_cpu_count_none(self):
        """Test CPU count when os.cpu_count returns None."""
        detector = HardwareDetector()
        with patch('os.cpu_count', return_value=None):
            assert detector._detect_cpu_count() == 1

    def test_detect_ram_linux(self):
        """Test RAM detection from /proc/meminfo."""
        meminfo = (
            "MemTotal:       32891552 kB\n"
            "MemFree:         1234567 kB\n"
            "MemAvailable:   24000000 kB\n"
            "Buffers:          500000 kB\n"
        )
        detector = HardwareDetector()
        with patch('builtins.open', mock_open(read_data=meminfo)):
            total, available = detector._detect_ram()
            assert total == 32891552 // 1024  # ~32 GB
            assert available == 24000000 // 1024  # ~23.4 GB

    def test_detect_ram_no_procfs(self):
        """Test RAM detection when /proc/meminfo doesn't exist."""
        detector = HardwareDetector()
        with patch('builtins.open', side_effect=FileNotFoundError):
            total, available = detector._detect_ram()
            assert total == 0
            assert available == 0

    def test_probe_ollama_available(self):
        """Test Ollama probe when server is available."""
        detector = HardwareDetector()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"models":[{"name":"llama3.2"},{"name":"qwen2.5"}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            available, models = detector._probe_ollama()
            assert available is True
            assert models == ["llama3.2", "qwen2.5"]

    def test_probe_ollama_unavailable(self):
        """Test Ollama probe when server is down."""
        detector = HardwareDetector()
        with patch('urllib.request.urlopen', side_effect=ConnectionRefusedError):
            available, models = detector._probe_ollama()
            assert available is False
            assert models == []

    def test_probe_ollama_timeout(self):
        """Test Ollama probe when server times out."""
        detector = HardwareDetector()
        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("timeout")):
            available, models = detector._probe_ollama()
            assert available is False
            assert models == []
