"""Tests for vLLM setup and management script."""

import json
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest

from scripts.vllm_setup import (
    _get_config,
    _get_vllm_pid,
    _probe_endpoint,
    is_vllm_running,
    start_server,
    stop_server,
    show_status,
    list_models,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_MODEL_LEN,
    VLLM_PID_PATH,
    VLLM_LOG_PATH,
)


class TestGetConfig:
    """Tests for _get_config()."""

    def test_defaults(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = _get_config()
            assert config['model'] == DEFAULT_MODEL
            assert config['port'] == str(DEFAULT_PORT)
            assert config['gpu_memory_utilization'] == str(DEFAULT_GPU_MEMORY_UTILIZATION)
            assert config['max_model_len'] == str(DEFAULT_MAX_MODEL_LEN)
            assert config['tensor_parallel_size'] == '1'
            assert config['pipeline_parallel_size'] == '1'
            assert config['data_parallel_size'] == '1'
            assert config['quantization'] == 'awq'
            assert config['enforce_eager'] == 'true'

    def test_env_overrides(self):
        """Test environment variable overrides."""
        env = {
            'VLLM_MODEL': 'custom/model',
            'VLLM_PORT': '9999',
            'VLLM_GPU_MEMORY_UTILIZATION': '0.8',
            'VLLM_MAX_MODEL_LEN': '16384',
        }
        with patch.dict(os.environ, env, clear=True):
            config = _get_config()
            assert config['model'] == 'custom/model'
            assert config['port'] == '9999'
            assert config['gpu_memory_utilization'] == '0.8'
            assert config['max_model_len'] == '16384'

    def test_multi_gpu_env_overrides(self):
        """Test multi-GPU environment variable overrides."""
        env = {
            'VLLM_TENSOR_PARALLEL_SIZE': '4',
            'VLLM_PIPELINE_PARALLEL_SIZE': '2',
            'VLLM_DATA_PARALLEL_SIZE': '3',
            'VLLM_QUANTIZATION': 'gptq',
            'VLLM_ENFORCE_EAGER': 'false',
        }
        with patch.dict(os.environ, env, clear=True):
            config = _get_config()
            assert config['tensor_parallel_size'] == '4'
            assert config['pipeline_parallel_size'] == '2'
            assert config['data_parallel_size'] == '3'
            assert config['quantization'] == 'gptq'
            assert config['enforce_eager'] == 'false'


class TestGetVllmPid:
    """Tests for _get_vllm_pid()."""

    def test_no_pid_file(self):
        """Test returns None when no PID file exists."""
        with patch.object(Path, 'exists', return_value=False):
            assert _get_vllm_pid() is None

    def test_valid_pid(self):
        """Test returns PID when process is running."""
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value='12345\n'), \
             patch('os.kill') as mock_kill:
            mock_kill.return_value = None  # os.kill(pid, 0) succeeds
            assert _get_vllm_pid() == 12345

    def test_stale_pid_file(self):
        """Test cleans up stale PID file when process not running."""
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value='99999\n'), \
             patch('os.kill', side_effect=ProcessLookupError), \
             patch.object(Path, 'unlink') as mock_unlink:
            assert _get_vllm_pid() is None
            mock_unlink.assert_called_once()

    def test_invalid_pid_content(self):
        """Test handles non-numeric PID file content."""
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value='not_a_number'), \
             patch.object(Path, 'unlink'):
            assert _get_vllm_pid() is None


class TestIsVllmRunning:
    """Tests for is_vllm_running()."""

    @patch('scripts.vllm_setup._get_vllm_pid')
    def test_running(self, mock_pid):
        """Test returns True when PID is valid."""
        mock_pid.return_value = 12345
        assert is_vllm_running() is True

    @patch('scripts.vllm_setup._get_vllm_pid')
    def test_not_running(self, mock_pid):
        """Test returns False when no PID."""
        mock_pid.return_value = None
        assert is_vllm_running() is False


class TestProbeEndpoint:
    """Tests for _probe_endpoint()."""

    def test_reachable_with_models(self):
        """Test probing a reachable endpoint with models."""
        response_data = {
            'data': [
                {'id': 'Qwen/Qwen2.5-7B-Instruct-AWQ'},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_resp):
            reachable, models = _probe_endpoint('8100')
            assert reachable is True
            assert models == ['Qwen/Qwen2.5-7B-Instruct-AWQ']

    def test_unreachable(self):
        """Test probing an unreachable endpoint."""
        with patch('urllib.request.urlopen', side_effect=ConnectionRefusedError):
            reachable, models = _probe_endpoint('8100')
            assert reachable is False
            assert models == []

    def test_timeout(self):
        """Test probing with timeout."""
        import urllib.error
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("timeout")):
            reachable, models = _probe_endpoint('8100')
            assert reachable is False
            assert models == []


class TestStartServer:
    """Tests for start_server()."""

    @patch('scripts.vllm_setup.is_vllm_running', return_value=True)
    def test_already_running(self, mock_running):
        """Test start when server is already running."""
        assert start_server() is True

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_start_success(self, mock_running, mock_sleep, mock_probe):
        """Test successful server start."""
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch('subprocess.Popen', return_value=mock_process), \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            assert start_server() is True

    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_start_process_dies(self, mock_running, mock_sleep):
        """Test start when process exits immediately."""
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = 1  # exited with error
        mock_process.returncode = 1

        with patch('subprocess.Popen', return_value=mock_process), \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'), \
             patch.object(Path, 'unlink'):
            assert start_server() is False

    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_start_vllm_not_installed(self, mock_running):
        """Test start when vllm is not installed."""
        with patch('subprocess.Popen', side_effect=FileNotFoundError):
            assert start_server() is False


class TestStartServerCommand:
    """Tests for command construction in start_server()."""

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_tensor_parallel_flag(self, mock_running, mock_sleep, mock_probe):
        """Test --tensor-parallel-size flag when TP > 1."""
        env = {'VLLM_TENSOR_PARALLEL_SIZE': '4'}
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            assert '--tensor-parallel-size' in cmd
            tp_idx = cmd.index('--tensor-parallel-size')
            assert cmd[tp_idx + 1] == '4'

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_no_parallel_flags_at_default(self, mock_running, mock_sleep, mock_probe):
        """Test no parallelism flags when sizes are 1."""
        env = {
            'VLLM_TENSOR_PARALLEL_SIZE': '1',
            'VLLM_PIPELINE_PARALLEL_SIZE': '1',
            'VLLM_DATA_PARALLEL_SIZE': '1',
        }
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            assert '--tensor-parallel-size' not in cmd
            assert '--pipeline-parallel-size' not in cmd
            assert '--data-parallel-size' not in cmd

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_quantization_none_omits_flag(self, mock_running, mock_sleep, mock_probe):
        """Test --quantization omitted when set to 'none'."""
        env = {'VLLM_QUANTIZATION': 'none'}
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            assert '--quantization' not in cmd

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_quantization_awq_includes_flag(self, mock_running, mock_sleep, mock_probe):
        """Test --quantization included when set to 'awq'."""
        env = {'VLLM_QUANTIZATION': 'awq'}
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            assert '--quantization' in cmd
            q_idx = cmd.index('--quantization')
            assert cmd[q_idx + 1] == 'awq'

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_enforce_eager_false_omits_flag(self, mock_running, mock_sleep, mock_probe):
        """Test --enforce-eager omitted when VLLM_ENFORCE_EAGER=false."""
        env = {'VLLM_ENFORCE_EAGER': 'false'}
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            assert '--enforce-eager' not in cmd

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_enforce_eager_true_includes_flag(self, mock_running, mock_sleep, mock_probe):
        """Test --enforce-eager included when VLLM_ENFORCE_EAGER=true."""
        env = {'VLLM_ENFORCE_EAGER': 'true'}
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            assert '--enforce-eager' in cmd

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model']))
    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup.is_vllm_running', return_value=False)
    def test_all_parallel_flags(self, mock_running, mock_sleep, mock_probe):
        """Test all parallelism flags set together."""
        env = {
            'VLLM_TENSOR_PARALLEL_SIZE': '2',
            'VLLM_PIPELINE_PARALLEL_SIZE': '3',
            'VLLM_DATA_PARALLEL_SIZE': '4',
        }
        mock_process = MagicMock()
        mock_process.pid = 42
        mock_process.poll.return_value = None

        with patch.dict(os.environ, env, clear=False), \
             patch('subprocess.Popen', return_value=mock_process) as mock_popen, \
             patch('builtins.open', mock_open()), \
             patch.object(Path, 'write_text'):
            start_server()
            cmd = mock_popen.call_args[0][0]
            tp_idx = cmd.index('--tensor-parallel-size')
            assert cmd[tp_idx + 1] == '2'
            pp_idx = cmd.index('--pipeline-parallel-size')
            assert cmd[pp_idx + 1] == '3'
            dp_idx = cmd.index('--data-parallel-size')
            assert cmd[dp_idx + 1] == '4'


class TestStopServer:
    """Tests for stop_server()."""

    @patch('scripts.vllm_setup._get_vllm_pid', return_value=None)
    def test_not_running(self, mock_pid):
        """Test stop when server is not running."""
        assert stop_server() is True

    @patch('scripts.vllm_setup.time.sleep')
    @patch('scripts.vllm_setup._get_vllm_pid', return_value=12345)
    def test_stop_graceful(self, mock_pid, mock_sleep):
        """Test graceful stop with SIGTERM."""
        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0 and len(kill_calls) > 1:
                raise ProcessLookupError

        with patch('os.kill', side_effect=mock_kill), \
             patch.object(Path, 'unlink'):
            assert stop_server() is True

    @patch('scripts.vllm_setup._get_vllm_pid', return_value=12345)
    def test_stop_already_dead(self, mock_pid):
        """Test stop when process already exited."""
        with patch('os.kill', side_effect=ProcessLookupError), \
             patch.object(Path, 'unlink'):
            assert stop_server() is True


class TestShowStatus:
    """Tests for show_status()."""

    @patch('scripts.vllm_setup._show_gpu_memory')
    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['test-model']))
    @patch('scripts.vllm_setup._get_vllm_pid', return_value=42)
    def test_status_running(self, mock_pid, mock_probe, mock_gpu, capsys):
        """Test status display when server is running."""
        show_status()
        # Should not raise

    @patch('scripts.vllm_setup._show_gpu_memory')
    @patch('scripts.vllm_setup._probe_endpoint', return_value=(False, []))
    @patch('scripts.vllm_setup._get_vllm_pid', return_value=None)
    def test_status_not_running(self, mock_pid, mock_probe, mock_gpu, capsys):
        """Test status display when server is not running."""
        show_status()
        # Should not raise


class TestListModels:
    """Tests for list_models()."""

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, ['model-a', 'model-b']))
    def test_list_available(self, mock_probe):
        """Test listing when models are available."""
        list_models()  # Should not raise

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(False, []))
    def test_list_not_reachable(self, mock_probe):
        """Test listing when endpoint is unreachable."""
        list_models()  # Should not raise

    @patch('scripts.vllm_setup._probe_endpoint', return_value=(True, []))
    def test_list_no_models(self, mock_probe):
        """Test listing when no models are served."""
        list_models()  # Should not raise


class TestHardwareDetectorVllm:
    """Tests for vLLM detection in HardwareDetector."""

    def test_profile_has_vllm_fields(self):
        """Test that HardwareProfile includes vLLM fields."""
        from src.hardware.detector import HardwareProfile
        profile = HardwareProfile()
        assert profile.vllm_available is False
        assert profile.vllm_models == []

    def test_summary_vllm_not_available(self):
        """Test summary string when vLLM is not available."""
        from src.hardware.detector import HardwareProfile
        profile = HardwareProfile()
        summary = profile.summary()
        assert "vLLM: not available" in summary

    def test_summary_vllm_available(self):
        """Test summary string when vLLM is available."""
        from src.hardware.detector import HardwareProfile
        profile = HardwareProfile(
            vllm_available=True,
            vllm_models=['Qwen/Qwen2.5-7B-Instruct-AWQ'],
        )
        summary = profile.summary()
        assert "vLLM: available (1 models)" in summary

    def test_probe_vllm_available(self):
        """Test vLLM probe when server is available."""
        from src.hardware.detector import HardwareDetector
        detector = HardwareDetector()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"data":[{"id":"Qwen/Qwen2.5-7B-Instruct-AWQ"}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            available, models = detector._probe_vllm()
            assert available is True
            assert models == ['Qwen/Qwen2.5-7B-Instruct-AWQ']

    def test_probe_vllm_unavailable(self):
        """Test vLLM probe when server is down."""
        from src.hardware.detector import HardwareDetector
        detector = HardwareDetector()

        with patch('urllib.request.urlopen', side_effect=ConnectionRefusedError):
            available, models = detector._probe_vllm()
            assert available is False
            assert models == []

    @patch('src.hardware.detector.HardwareDetector._probe_vllm')
    @patch('src.hardware.detector.HardwareDetector._probe_ollama')
    @patch('src.hardware.detector.HardwareDetector._detect_ram')
    @patch('src.hardware.detector.HardwareDetector._detect_cpu_count')
    @patch('src.hardware.detector.HardwareDetector._detect_gpus')
    def test_detect_includes_vllm(
        self, mock_gpus, mock_cpu, mock_ram, mock_ollama, mock_vllm
    ):
        """Test that detect() includes vLLM probe results."""
        from src.hardware.detector import HardwareDetector
        mock_gpus.return_value = (0, [])
        mock_cpu.return_value = 4
        mock_ram.return_value = (8192, 4096)
        mock_ollama.return_value = (False, [])
        mock_vllm.return_value = (True, ['test-model'])

        detector = HardwareDetector()
        profile = detector.detect()

        assert profile.vllm_available is True
        assert profile.vllm_models == ['test-model']
        mock_vllm.assert_called_once()
