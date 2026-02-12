"""
FastAPI web server for audio analysis.

Provides REST API endpoints for audio transcription and analysis.
"""

import asyncio
import json
import logging
import os
import re
import socket
import sys
import tempfile
import threading
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.web.models import (
    AnalysisResult,
    TranscriptionResult,
    HealthResponse,
    ErrorResponse,
    ServiceStatus,
    ServicesStatusResponse,
)
from src.web.audio_processor import AudioProcessor, ANALYSIS_STEPS
from src.web.archive_manager import ArchiveManager
from src.web.workspace_manager import WorkspaceManager

# Import telemetry integration (optional)
try:
    from src.web.telemetry import initialize_telemetry, track_event
except ImportError:
    initialize_telemetry = None
    track_event = None

# Import narrative generators for regeneration
try:
    from src.web.narrative_summary import NarrativeSummaryGenerator
    NARRATIVE_AVAILABLE = True
except ImportError:
    NARRATIVE_AVAILABLE = False
    logger.warning("Narrative summary module not available for regeneration")

# Import hallucination prevention
try:
    from src.llm.hallucination_prevention import clean_hallucinations
    HALLUCINATION_PREVENTION_AVAILABLE = True
except ImportError:
    HALLUCINATION_PREVENTION_AVAILABLE = False
    clean_hallucinations = None

# Streaming transcription support
try:
    from src.web.streaming_transcriber import (
        StreamingTranscriptionManager,
        STREAMING_ENABLED
    )
    STREAMING_TRANSCRIPTION_AVAILABLE = True
except ImportError:
    STREAMING_TRANSCRIPTION_AVAILABLE = False
    StreamingTranscriptionManager = None
    STREAMING_ENABLED = False

# Live metrics for real-time dashboard
try:
    from src.metrics.live_metrics import LiveMetricsComputer
    LIVE_METRICS_AVAILABLE = True
except ImportError:
    LIVE_METRICS_AVAILABLE = False
    LiveMetricsComputer = None

# Live GM analysis for real-time crew support
try:
    from src.llm.live_analysis import LiveGMAnalyzer
    LIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    LIVE_ANALYSIS_AVAILABLE = False
    LiveGMAnalyzer = None

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
WEB_HOST = os.getenv('WEB_SERVER_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('WEB_SERVER_PORT', '8000'))
MAX_UPLOAD_MB = int(os.getenv('WEB_MAX_UPLOAD_MB', '2048'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CORS_ORIGINS = os.getenv('WEB_CORS_ORIGINS', '*').split(',')
READ_ONLY_MODE = os.getenv('READ_ONLY_MODE', 'false').lower() == 'true'
# READ_ONLY_MODE implies DISABLE_AUDIO_FILES
DISABLE_AUDIO_FILES = READ_ONLY_MODE or os.getenv('DISABLE_AUDIO_FILES', 'false').lower() == 'true'

# Project root for .env file operations
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _is_valid_host(host: str) -> bool:
    """
    Validate a hostname or IPv4 address.

    Args:
        host: Hostname or IPv4 address string.

    Returns:
        True if the host is a valid IPv4 address or RFC 1123 hostname.
    """
    if not host or len(host) > 253:
        return False

    # Check IPv4
    ipv4_pattern = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
    m = ipv4_pattern.match(host)
    if m:
        return all(0 <= int(octet) <= 255 for octet in m.groups())

    # Check RFC 1123 hostname
    hostname_pattern = re.compile(
        r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?'
        r'(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
    )
    return bool(hostname_pattern.match(host))


def _update_env_file(updates: Dict[str, str]) -> None:
    """
    Update the .env file with the given key-value pairs.

    Reads the existing file, replaces matching KEY=value lines, appends
    missing keys, and writes atomically via a temp file + rename.

    Args:
        updates: Dictionary of environment variable names to new values.

    Raises:
        OSError: If the file cannot be written.
    """
    env_path = _PROJECT_ROOT / '.env'
    lines: List[str] = []
    found_keys: set = set()

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                # Check if this line sets one of our keys
                matched = False
                for key in updates:
                    if stripped.startswith(f'{key}=') or stripped.startswith(f'# {key}='):
                        # Only replace uncommented lines
                        if stripped.startswith(f'{key}='):
                            lines.append(f'{key}={updates[key]}\n')
                            found_keys.add(key)
                            matched = True
                            break
                if not matched:
                    lines.append(line if line.endswith('\n') else line + '\n')

    # Append any keys that were not found in the file
    for key, value in updates.items():
        if key not in found_keys:
            lines.append(f'{key}={value}\n')

    # Write atomically via temp file + rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(env_path.parent), suffix='.env.tmp'
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.writelines(lines)
        os.replace(tmp_path, str(env_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# Global processor instance (shared — ML models are expensive)
_processor: Optional[AudioProcessor] = None
_processor_lock = threading.Lock()

# Global workspace manager (routes file I/O per workspace)
_workspace_manager: Optional[WorkspaceManager] = None

# Concurrency limiter for analysis operations
MAX_CONCURRENT_ANALYSES = int(os.getenv('MAX_CONCURRENT_ANALYSES', '2'))
_analysis_semaphore: Optional[asyncio.Semaphore] = None


def get_analysis_semaphore() -> asyncio.Semaphore:
    """Get or create the analysis concurrency semaphore."""
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
    return _analysis_semaphore


def get_processor() -> AudioProcessor:
    """Get or create the audio processor instance (thread-safe)."""
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = AudioProcessor()
    return _processor


def get_workspace_manager() -> WorkspaceManager:
    """Get or create the workspace manager instance."""
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager()
    return _workspace_manager


# Global streaming transcription manager
_streaming_manager: Optional['StreamingTranscriptionManager'] = None


def get_streaming_manager() -> Optional['StreamingTranscriptionManager']:
    """Get or create the streaming transcription manager."""
    global _streaming_manager
    if not STREAMING_TRANSCRIPTION_AVAILABLE or not STREAMING_ENABLED:
        return None
    if _streaming_manager is None:
        _streaming_manager = StreamingTranscriptionManager()
    return _streaming_manager


async def get_workspace_id(request: Request) -> str:
    """
    Extract and validate workspace ID from request header.

    Args:
        request: The incoming HTTP request.

    Returns:
        Validated workspace ID string.

    Raises:
        HTTPException: If header is missing or ID format is invalid.
    """
    workspace_id = request.headers.get('X-Workspace-ID', '')
    if not workspace_id:
        raise HTTPException(400, "Missing X-Workspace-ID header")
    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(400, "Invalid workspace ID format")
    return workspace_id


def _resolve_analysis(
    filename: str,
    workspace_dirs: Dict,
    ws_mgr: WorkspaceManager
) -> Optional[Path]:
    """
    Find the directory containing a given analysis file.

    Checks the workspace first, then falls back to shared.

    Args:
        filename: Analysis JSON filename.
        workspace_dirs: Workspace directory dict from ensure_workspace().
        ws_mgr: WorkspaceManager instance.

    Returns:
        Path to the directory containing the file, or None if not found.
    """
    ws_path = workspace_dirs['analyses'] / filename
    if ws_path.exists():
        return workspace_dirs['analyses']
    shared_dir = ws_mgr.shared_analyses_dir
    if shared_dir.exists() and (shared_dir / filename).exists():
        return shared_dir
    return None


# Request model for metadata updates
class MetadataUpdate(BaseModel):
    """Request body for updating analysis metadata."""
    user_title: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    starred: Optional[bool] = None


def _save_telemetry_snapshot(session_id: str, session: dict) -> Optional[str]:
    """
    Save a periodic snapshot of telemetry data for an active session.

    Overwrites the same file each time so cumulative data is always on disk.

    Args:
        session_id: The telemetry session identifier.
        session: The session dict from _telemetry_sessions.

    Returns:
        Path to the snapshot file, or None on failure.
    """
    try:
        from datetime import datetime

        client = session.get('client')
        if not client:
            return None

        ws_mgr = get_workspace_manager()
        workspace_id = session.get('workspace_id', '')
        dirs = ws_mgr.ensure_workspace(workspace_id)
        telemetry_dir = dirs['telemetry']

        telemetry_file = telemetry_dir / f"telemetry_{session_id}.json"

        start_time = session['start_time']
        now = datetime.now()
        duration = (now - start_time).total_seconds()

        telemetry_data = {
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'snapshot_time': now.isoformat(),
            'duration_seconds': duration,
            'status': 'recording',
            'packet_counts': dict(client.packet_counts),
            'tracked_events': client.get_tracked_events(),
            'tracked_events_count': len(client.get_tracked_events()),
            'vessel_data': client.vessel_data,
            'mission_data': client.mission_data,
            'combat_data': client.combat_enhanced,
            'connection_health': client.connection_health,
            'last_packets': {
                k: v for k, v in client.last_packets.items()
                if k in [
                    'CONTACTS', 'MISSION', 'ALERT',
                    'DAMAGE', 'PLAYER-OBJECTIVES',
                ]
            },
        }

        with open(telemetry_file, 'w') as f:
            json.dump(telemetry_data, f, indent=2, default=str)

        session['last_snapshot_time'] = now.isoformat()
        logger.debug(f"Telemetry snapshot saved for session {session_id}")
        return str(telemetry_file)

    except Exception as e:
        logger.warning(f"Telemetry snapshot failed for session {session_id}: {e}")
        return None


async def _telemetry_snapshot_loop() -> None:
    """Background task: save a telemetry snapshot every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        for sid, session in list(_telemetry_sessions.items()):
            if session.get('status') == 'recording':
                # Run synchronous I/O in a thread so we don't block the loop
                await asyncio.to_thread(_save_telemetry_snapshot, sid, session)


async def _warmup_llm() -> None:
    """
    Warm up the Ollama/LLM model by sending a tiny generation request.

    This loads the model into GPU memory so the first real request
    doesn't suffer a cold-start delay. Runs in a background thread
    to avoid blocking the event loop.
    """
    try:
        from src.llm.llm_client import get_default_client
        llm = get_default_client()
        model_name = llm.model
        logger.info(f"Warming up LLM model: {model_name}...")

        def _do_warmup() -> None:
            response = llm.generate(
                prompt="Say OK.",
                system="Respond with just OK.",
                temperature=0.0,
                max_tokens=4,
            )
            if response and response.text:
                logger.info(
                    f"LLM model warmed up: {model_name} "
                    f"({response.tokens_per_second} tok/s)"
                )
            else:
                logger.warning("LLM warmup returned empty response")

        await asyncio.to_thread(_do_warmup)
    except Exception as e:
        logger.warning(f"LLM warmup failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for model loading."""
    logger.info("Starting audio analysis server...")
    processor = get_processor()

    # Preload model on startup (disable with PRELOAD_WHISPER=false)
    if os.getenv('PRELOAD_WHISPER', 'true').lower() == 'true':
        logger.info("Preloading Whisper model...")
        processor.load_model()

    # Warm up Ollama/LLM model (disable with PRELOAD_LLM=false)
    if os.getenv('PRELOAD_LLM', 'true').lower() == 'true':
        asyncio.create_task(_warmup_llm())

    # Start periodic telemetry snapshot task
    snapshot_task = asyncio.create_task(_telemetry_snapshot_loop())

    yield

    logger.info("Shutting down audio analysis server...")

    # Cancel snapshot task
    snapshot_task.cancel()
    try:
        await snapshot_task
    except asyncio.CancelledError:
        pass

    # Save final snapshots and disconnect active telemetry clients
    for sid, session in list(_telemetry_sessions.items()):
        if session.get('status') == 'recording':
            _save_telemetry_snapshot(sid, session)
            client = session.get('client')
            if client:
                try:
                    client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting telemetry client {sid}: {e}")
            session['status'] = 'stopped'


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Starship Horizons Audio Analyzer",
        description="Audio transcription and analysis API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Application Insights telemetry (no-ops if not configured)
    if initialize_telemetry is not None:
        initialize_telemetry(application)

    # Mount static files
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        application.mount(
            "/static",
            StaticFiles(directory=str(static_dir)),
            name="static"
        )

    return application


app = create_app()


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    static_dir = Path(__file__).parent.parent.parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    processor = get_processor()
    return HealthResponse(
        status="ok",
        whisper_loaded=processor.is_model_loaded,
        whisper_model=processor.whisper_model_size if processor.is_model_loaded else None
    )


@app.get("/api/config")
async def get_config():
    """
    Get client configuration settings.

    Returns configuration flags that affect UI behavior.
    """
    return {
        "audio_disabled": DISABLE_AUDIO_FILES,
        "read_only": READ_ONLY_MODE,
        "max_upload_mb": MAX_UPLOAD_MB,
        "appinsights_connection_string": os.getenv('APPINSIGHTS_CONNECTION_STRING', ''),
    }


@app.get("/api/services-status", response_model=ServicesStatusResponse)
async def get_services_status():
    """
    Check status of all external services (Whisper, Ollama, Diarization).

    Returns availability and status details for each service.
    """
    import httpx

    processor = get_processor()

    # Check Whisper status
    whisper_status = ServiceStatus(
        available=False,
        status="Not loaded",
        details=None
    )

    try:
        from src.audio.whisper_transcriber import WHISPER_AVAILABLE
        if WHISPER_AVAILABLE:
            if processor.is_model_loaded:
                whisper_status = ServiceStatus(
                    available=True,
                    status="Ready",
                    details=f"Model: {processor.whisper_model_size}"
                )
            else:
                whisper_status = ServiceStatus(
                    available=True,
                    status="Available (not loaded)",
                    details=f"Model: {processor.whisper_model_size} (will load on first use)"
                )
        else:
            whisper_status = ServiceStatus(
                available=False,
                status="Not installed",
                details="Install faster-whisper package"
            )
    except ImportError:
        whisper_status = ServiceStatus(
            available=False,
            status="Not installed",
            details="Install faster-whisper package"
        )

    # Check Ollama/LLM status
    ollama_status = ServiceStatus(
        available=False,
        status="Not connected",
        details=None
    )

    try:
        from src.llm.llm_client import LLMClient

        ollama_model = os.getenv('LLM_MODEL') or os.getenv('OLLAMA_MODEL', 'llama3.2')
        llm = LLMClient(timeout=10)

        model_list = await llm.alist_models()
        if model_list:
            # Get both full names and base names for matching
            full_names = model_list
            base_names = [name.split(':')[0] for name in full_names]
            ollama_model_base = ollama_model.split(':')[0]

            # Check if configured model exists (match full name or base name)
            model_found = (
                ollama_model in full_names or
                ollama_model_base in base_names or
                any(ollama_model in name for name in full_names)
            )

            if model_found:
                ollama_status = ServiceStatus(
                    available=True,
                    status="Ready",
                    details=f"Model: {ollama_model} ({len(model_list)} models available)"
                )
            else:
                ollama_status = ServiceStatus(
                    available=True,
                    status="Running (model not found)",
                    details=f"Model '{ollama_model}' not found. Available: {', '.join(full_names[:3])}"
                )
        else:
            ollama_status = ServiceStatus(
                available=False,
                status="Not running",
                details="No models returned from backend"
            )
    except Exception as e:
        ollama_status = ServiceStatus(
            available=False,
            status="Error",
            details=str(e)[:100]
        )

    # Check diarization status
    diarization_status = ServiceStatus(
        available=False,
        status="Not available",
        details=None
    )

    try:
        from src.web.audio_processor import (
            NEURAL_DIARIZATION_AVAILABLE,
            CPU_DIARIZATION_AVAILABLE,
            DIARIZATION_AVAILABLE
        )

        if processor.use_cpu_diarization and CPU_DIARIZATION_AVAILABLE:
            diarization_status = ServiceStatus(
                available=True,
                status="Ready (CPU mode)",
                details="Using resemblyzer for speaker identification"
            )
        elif processor.use_neural_diarization and NEURAL_DIARIZATION_AVAILABLE:
            diarization_status = ServiceStatus(
                available=True,
                status="Ready (Neural mode)",
                details="Using pyannote for speaker identification"
            )
        elif DIARIZATION_AVAILABLE:
            diarization_status = ServiceStatus(
                available=True,
                status="Ready (Simple mode)",
                details="Using spectral features for speaker identification"
            )
        else:
            diarization_status = ServiceStatus(
                available=False,
                status="Not installed",
                details="Install speaker diarization dependencies"
            )
    except ImportError:
        diarization_status = ServiceStatus(
            available=False,
            status="Not installed",
            details="Speaker diarization module not available"
        )

    # Check Horizons game server status
    horizons_status = ServiceStatus(
        available=False,
        status="Not configured",
        details=None
    )

    try:
        game_host = os.getenv('GAME_HOST', '')
        game_port_ws = os.getenv('GAME_PORT_WS', '1865')
        game_port_api = os.getenv('GAME_PORT_API', '1864')

        if not game_host:
            horizons_status = ServiceStatus(
                available=False,
                status="Not configured",
                details="Set GAME_HOST in .env file"
            )
        else:
            import socket

            # Try to connect to WebSocket port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            try:
                result = sock.connect_ex((game_host, int(game_port_ws)))
                if result == 0:
                    horizons_status = ServiceStatus(
                        available=True,
                        status="Connected",
                        details=f"{game_host}:{game_port_ws}"
                    )
                else:
                    # Try API port as fallback
                    result = sock.connect_ex((game_host, int(game_port_api)))
                    if result == 0:
                        horizons_status = ServiceStatus(
                            available=True,
                            status="API only",
                            details=f"{game_host}:{game_port_api} (WS port {game_port_ws} not responding)"
                        )
                    else:
                        horizons_status = ServiceStatus(
                            available=False,
                            status="Not reachable",
                            details=f"Cannot connect to {game_host}:{game_port_ws}"
                        )
            finally:
                sock.close()
    except Exception as e:
        horizons_status = ServiceStatus(
            available=False,
            status="Error",
            details=str(e)[:100]
        )

    return ServicesStatusResponse(
        whisper=whisper_status,
        ollama=ollama_status,
        diarization=diarization_status,
        horizons=horizons_status
    )


# ============================================================================
# Telemetry Recording Endpoints
# ============================================================================

# Global telemetry sessions storage
_telemetry_sessions: dict = {}


class TelemetryStartResponse(BaseModel):
    """Response for starting telemetry recording."""
    session_id: str
    status: str
    message: str


class TelemetryStopRequest(BaseModel):
    """Request to stop telemetry recording."""
    session_id: str


class TelemetryStopResponse(BaseModel):
    """Response for stopping telemetry recording."""
    session_id: str
    status: str
    events_captured: int
    duration_seconds: float
    telemetry_file: Optional[str] = None


class GameServerConfig(BaseModel):
    """Request body for game server configuration update."""
    game_host: str
    game_port_ws: int = Field(ge=1, le=65535)
    game_port_api: int = Field(ge=1, le=65535)
    game_port_https: int = Field(default=1866, ge=1, le=65535)
    game_port_wss: int = Field(default=1867, ge=1, le=65535)


class ConnectionTestRequest(BaseModel):
    """Request body for testing game server connectivity."""
    host: str
    port_ws: int = Field(ge=1, le=65535)
    port_api: int = Field(ge=1, le=65535)


class ConnectionTestResponse(BaseModel):
    """Response for game server connection test."""
    ws_reachable: bool
    api_reachable: bool
    ws_detail: str
    api_detail: str


@app.post("/api/telemetry/start", response_model=TelemetryStartResponse)
async def start_telemetry_recording(
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Start recording game telemetry from Starship Horizons.

    Returns a session_id that should be passed to the analyze endpoint.
    """
    import uuid
    from datetime import datetime

    # Check if Horizons is available
    game_host = os.getenv('GAME_HOST', '')
    if not game_host:
        raise HTTPException(status_code=503, detail="Game server not configured (GAME_HOST not set)")

    try:
        # Import telemetry client
        from src.integration.browser_mimic_websocket import BrowserMimicWebSocket

        session_id = str(uuid.uuid4())[:8]
        game_port = int(os.getenv('GAME_PORT_WS', '1865'))

        # Create telemetry client
        logger.info(f"Creating telemetry client for {game_host}:{game_port}")
        client = BrowserMimicWebSocket(host=game_host, port=game_port)

        # Connect to game server
        if not client.connect(screen_name='mainscreen', is_main_viewer=True, user_name='AI-Observer'):
            raise HTTPException(
                status_code=503,
                detail=f"WebSocket connection to {game_host}:{game_port} failed (5s timeout). "
                       f"TCP port is reachable but WebSocket handshake did not complete."
            )

        # Brief wait to confirm game is sending packets back
        import time
        time.sleep(1.0)
        packet_count = sum(client.packet_counts.values())
        if packet_count == 0:
            logger.warning(
                "Connected to game server but received 0 packets after 1s. "
                "The game may not have accepted the client."
            )

        # Start event tracking for role correlation
        client.start_event_tracking()

        # Store session with workspace ownership
        _telemetry_sessions[session_id] = {
            'workspace_id': workspace_id,
            'client': client,
            'start_time': datetime.now(),
            'events': [],
            'status': 'recording'
        }

        logger.info(
            f"Started telemetry recording session: {session_id} "
            f"({packet_count} packets received in first 1s)"
        )

        return TelemetryStartResponse(
            session_id=session_id,
            status="recording",
            message=(
                f"Telemetry recording started. Connected to {game_host}:{game_port} "
                f"as AI-Observer ({packet_count} packets received)"
            )
        )

    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Telemetry module not available: {e}")
    except Exception as e:
        logger.error(f"Failed to start telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start telemetry: {e}")


@app.post("/api/telemetry/stop", response_model=TelemetryStopResponse)
async def stop_telemetry_recording(
    request: TelemetryStopRequest,
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Stop recording game telemetry and save the data.

    Returns information about the captured telemetry.
    """
    from datetime import datetime

    session_id = request.session_id

    if session_id not in _telemetry_sessions:
        raise HTTPException(status_code=404, detail=f"Telemetry session not found: {session_id}")

    session = _telemetry_sessions[session_id]

    # Verify workspace ownership
    if session.get('workspace_id') != workspace_id:
        raise HTTPException(status_code=404, detail=f"Telemetry session not found: {session_id}")

    try:
        client = session['client']
        start_time = session['start_time']
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Get captured data from client
        events_captured = sum(client.packet_counts.values())

        # Get tracked events for role correlation
        tracked_events = client.get_tracked_events()

        # Save telemetry data to workspace directory
        ws_mgr = get_workspace_manager()
        dirs = ws_mgr.ensure_workspace(workspace_id)
        telemetry_dir = dirs['telemetry']

        telemetry_file = telemetry_dir / f"telemetry_{session_id}.json"

        telemetry_data = {
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'packet_counts': dict(client.packet_counts),
            'tracked_events': tracked_events,  # Timestamped events for role correlation
            'tracked_events_count': len(tracked_events),
            'vessel_data': client.vessel_data,
            'mission_data': client.mission_data,
            'combat_data': client.combat_enhanced,
            'last_packets': {k: v for k, v in client.last_packets.items()
                           if k in ['CONTACTS', 'MISSION', 'ALERT', 'DAMAGE', 'PLAYER-OBJECTIVES']}
        }

        logger.info(f"Telemetry session {session_id}: {len(tracked_events)} tracked events for role correlation")

        with open(telemetry_file, 'w') as f:
            json.dump(telemetry_data, f, indent=2, default=str)

        # Disconnect client
        client.disconnect()

        # Update session status
        session['status'] = 'stopped'
        session['telemetry_file'] = str(telemetry_file)
        session['events_captured'] = events_captured
        session['duration'] = duration

        logger.info(f"Stopped telemetry session {session_id}: {events_captured} events in {duration:.1f}s")

        return TelemetryStopResponse(
            session_id=session_id,
            status="stopped",
            events_captured=events_captured,
            duration_seconds=duration,
            telemetry_file=str(telemetry_file)
        )

    except Exception as e:
        logger.error(f"Failed to stop telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop telemetry: {e}")
    finally:
        # Clean up session after a delay (keep for potential analysis)
        # In production, you might want to clean up old sessions periodically
        pass


@app.get("/api/telemetry/status/{session_id}")
async def get_telemetry_status(
    session_id: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Get the status of a telemetry recording session.
    """
    if session_id not in _telemetry_sessions:
        raise HTTPException(status_code=404, detail=f"Telemetry session not found: {session_id}")

    session = _telemetry_sessions[session_id]

    # Verify workspace ownership
    if session.get('workspace_id') != workspace_id:
        raise HTTPException(status_code=404, detail=f"Telemetry session not found: {session_id}")

    client = session.get('client')

    response = {
        'session_id': session_id,
        'status': session['status'],
        'start_time': session['start_time'].isoformat(),
        'events_captured': sum(client.packet_counts.values()) if client else 0,
        'packet_types': len(client.packet_counts) if client else 0,
        'last_snapshot_time': session.get('last_snapshot_time'),
    }

    # Include connection health when client is available
    if client and hasattr(client, 'connection_health'):
        response['connection_health'] = client.connection_health

    # Include live game state for dashboard
    if client:
        try:
            ship = client.get_ship_status_summary()
            tracked = client.get_tracked_events()
            alert_level_names = {1: "Docked", 2: "Green", 3: "Yellow", 4: "Red", 5: "Red Alert"}
            alert_val = ship.get('alert_level', 2)
            response['game_state'] = {
                'hull': ship.get('hull', 100),
                'shields': ship.get('shields', 100),
                'alert_level': alert_val,
                'alert_name': alert_level_names.get(alert_val, 'Unknown'),
                'weapons_armed': ship.get('combat_readiness', {}).get('weapons_armed', False),
                'speed': ship.get('navigation', {}).get('speed', 0),
                'vessel_name': client.vessel_data.get('vessel_name'),
                'mission_name': client.mission_data.get('current_mission'),
                'objectives': client.mission_data.get('player_objectives', []),
                'gm_objectives': client.mission_data.get('gm_objectives', []),
                'recent_events': tracked[-10:] if tracked else [],
                'total_events': len(tracked),
            }
        except Exception as e:
            logger.warning(f"Failed to get game state for dashboard: {e}")

    return response


@app.get("/api/telemetry/{session_id}/download")
async def download_telemetry(
    session_id: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Download the telemetry JSON file for a completed session.

    Args:
        session_id: The telemetry session identifier.
        workspace_id: Workspace ID from request header.

    Returns:
        The telemetry JSON file as a download.
    """
    telemetry_file = None

    # Try to get file path from session data
    if session_id in _telemetry_sessions:
        session = _telemetry_sessions[session_id]
        if session.get('workspace_id') != workspace_id:
            raise HTTPException(status_code=404, detail=f"Telemetry session not found: {session_id}")
        telemetry_file = session.get('telemetry_file')

    # Fall back to scanning workspace telemetry directory
    if not telemetry_file:
        ws_mgr = get_workspace_manager()
        dirs = ws_mgr.ensure_workspace(workspace_id)
        candidate = dirs['telemetry'] / f"telemetry_{session_id}.json"
        if candidate.exists():
            telemetry_file = str(candidate)

    if not telemetry_file or not Path(telemetry_file).exists():
        raise HTTPException(status_code=404, detail=f"Telemetry file not found for session: {session_id}")

    return FileResponse(
        str(telemetry_file),
        media_type="application/json",
        filename=f"telemetry_{session_id}.json"
    )


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze"),
    include_diarization: bool = Query(
        True, description="Include speaker diarization"
    ),
    include_quality: bool = Query(
        True, description="Include communication quality analysis"
    ),
    include_detailed: bool = Query(
        True, description="Include detailed analysis (scorecards, learning metrics)"
    ),
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Analyze audio file with full pipeline.

    Performs:
    - Whisper transcription with word timestamps
    - Speaker diarization and identification
    - Role inference
    - Communication quality pattern analysis
    - Speaker scorecards (1-5 ratings)
    - Confidence distribution
    - Learning framework evaluation
    """
    # Check if audio uploads are disabled
    if DISABLE_AUDIO_FILES:
        raise HTTPException(
            status_code=403,
            detail="Audio file uploads are disabled on this server"
        )

    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_MB}MB"
        )

    # Validate file type
    allowed_types = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/webm', 'audio/ogg', 'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/m4a', 'audio/flac', 'audio/x-flac',
        'video/webm',  # Browser may send video/webm for audio recording
    }
    content_type = file.content_type or ''
    if content_type and content_type not in allowed_types:
        # Be lenient - if extension looks like audio, allow it
        ext = Path(file.filename or '').suffix.lower()
        audio_exts = {'.wav', '.webm', '.mp3', '.m4a', '.ogg', '.flac', '.opus'}
        if ext not in audio_exts:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}"
            )

    # Save to temp file
    suffix = Path(file.filename or 'audio.wav').suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ws_mgr = get_workspace_manager()
        dirs = ws_mgr.ensure_workspace(workspace_id)
        archive_mgr = ws_mgr.get_archive_manager(workspace_id)

        processor = get_processor()
        results = processor.analyze_audio(
            tmp_path,
            include_diarization=include_diarization,
            include_quality=include_quality,
            include_detailed=include_detailed,
            recordings_dir=str(dirs['recordings']),
            analyses_dir=str(dirs['analyses']),
            archive_manager=archive_mgr,
            telemetry_dir=str(dirs['telemetry'])
        )

        return AnalysisResult(
            transcription=results['transcription'],
            full_text=results['full_text'],
            duration_seconds=results['duration_seconds'],
            speakers=results['speakers'],
            communication_quality=results.get('communication_quality'),
            speaker_scorecards=results.get('speaker_scorecards', []),
            role_assignments=results.get('role_assignments', []),
            confidence_distribution=results.get('confidence_distribution'),
            learning_evaluation=results.get('learning_evaluation'),
            processing_time_seconds=results['processing_time_seconds']
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/analyze-stream")
async def analyze_audio_stream(
    file: UploadFile = File(..., description="Audio file to analyze"),
    include_narrative: bool = Query(True, description="Include LLM team analysis"),
    include_story: bool = Query(True, description="Include LLM mission story"),
    telemetry_session_id: Optional[str] = Query(None, description="Telemetry recording session ID"),
    streaming_session_id: Optional[str] = Query(None, description="Streaming transcription session ID"),
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Analyze audio file with streaming progress updates.

    Returns Server-Sent Events (SSE) stream with:
    - Progress updates: {"type": "progress", "step": "...", "label": "...", "progress": N}
    - Final result: {"type": "result", "data": {...}}
    - Error: {"type": "error", "message": "..."}

    Query Parameters:
    - include_narrative: Include LLM team analysis (default: True)
    - include_story: Include LLM mission story generation (default: True)
    - telemetry_session_id: If provided, include game telemetry from this session
    - streaming_session_id: If provided, use pre-computed streaming transcription
    """
    # Check if audio uploads are disabled
    if DISABLE_AUDIO_FILES:
        async def disabled_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Audio file uploads are disabled on this server'})}\n\n"
        return StreamingResponse(disabled_stream(), media_type="text/event-stream")

    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': f'File too large. Maximum size is {MAX_UPLOAD_MB}MB'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Save to temp file
    suffix = Path(file.filename or 'audio.wav').suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Queue for progress updates
    progress_queue: queue.Queue = queue.Queue()

    def progress_callback(step_id: str, step_label: str, progress: int):
        """Callback for progress updates."""
        progress_queue.put({
            'type': 'progress',
            'step': step_id,
            'label': step_label,
            'progress': progress
        })

    # Capture options for closure
    _include_narrative = include_narrative
    _include_story = include_story
    _telemetry_session_id = telemetry_session_id
    _streaming_session_id = streaming_session_id

    # Retrieve pre-computed streaming transcription if available
    _precomputed_segments = None
    _precomputed_info = None
    if _streaming_session_id:
        streaming_mgr = get_streaming_manager()
        if streaming_mgr:
            streaming_result = streaming_mgr.finalize_session(_streaming_session_id)
            if streaming_result and streaming_result.get('segments'):
                _precomputed_segments = streaming_result['segments']
                _precomputed_info = streaming_result.get('info')
                logger.info(
                    f"Using streaming transcription: {len(_precomputed_segments)} "
                    f"pre-computed segments from session {_streaming_session_id}"
                )

    logger.info(f"Analysis options: include_narrative={_include_narrative}, include_story={_include_story}, telemetry_session_id={_telemetry_session_id}, streaming_session_id={_streaming_session_id}")

    # Resolve workspace directories before spawning thread
    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    archive_mgr = ws_mgr.get_archive_manager(workspace_id)

    def run_analysis():
        """Run analysis in background thread."""
        try:
            processor = get_processor()
            results = processor.analyze_audio(
                tmp_path,
                include_diarization=True,
                include_quality=True,
                include_detailed=True,
                include_narrative=_include_narrative,
                include_story=_include_story,
                progress_callback=progress_callback,
                telemetry_session_id=_telemetry_session_id,
                recordings_dir=str(dirs['recordings']),
                analyses_dir=str(dirs['analyses']),
                archive_manager=archive_mgr,
                telemetry_dir=str(dirs['telemetry']),
                precomputed_segments=_precomputed_segments,
                precomputed_info=_precomputed_info
            )
            progress_queue.put({'type': 'result', 'data': results})
        except Exception as e:
            logger.error(f"Stream analysis failed: {e}", exc_info=True)
            progress_queue.put({'type': 'error', 'message': str(e)})
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Start analysis in background thread
    thread = threading.Thread(target=run_analysis)
    thread.start()

    async def event_stream():
        """Generate SSE events."""
        import time as _time

        # Send large initial padding to force proxy buffers to flush.
        # Azure reverse proxies, VS Code port forwarding, and cloud
        # load balancers buffer small responses; 16KB forces them to
        # start streaming immediately.
        yield ": " + " " * 16384 + "\n\n"

        # Track time since last yield to send periodic heartbeats.
        # Long-running steps (transcription, LLM cleanup, narrative)
        # can be silent for minutes; without heartbeats, Azure proxies
        # buffer all events and deliver them only at the end.
        last_yield = _time.monotonic()
        HEARTBEAT_INTERVAL = 15  # seconds

        while True:
            try:
                # Non-blocking check — never block the event loop
                try:
                    msg = progress_queue.get_nowait()
                except queue.Empty:
                    # Send heartbeat comment to keep proxy flushing
                    now = _time.monotonic()
                    if now - last_yield >= HEARTBEAT_INTERVAL:
                        yield ": heartbeat\n\n"
                        last_yield = now
                    await asyncio.sleep(0.1)
                    continue

                yield f"data: {json.dumps(msg)}\n\n"
                last_yield = _time.monotonic()

                # Stop on result or error
                if msg['type'] in ('result', 'error'):
                    break

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.websocket("/ws/transcribe-stream")
async def websocket_transcribe_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio transcription.

    The client sends binary audio frames during recording. The server
    accumulates chunks and periodically runs Whisper on new audio,
    sending transcript segments back as JSON.

    Protocol:
        Client -> Server:
            - Binary frames: raw audio chunk data
            - JSON text: {"type": "stop"} to end session
        Server -> Client:
            - JSON: {"type": "session_start", "session_id": "..."}
            - JSON: {"type": "transcript", "segments": [...]}
            - JSON: {"type": "info", "message": "..."}
            - JSON: {"type": "error", "message": "..."}
    """
    streaming_mgr = get_streaming_manager()
    if streaming_mgr is None:
        await websocket.accept()
        await websocket.send_json({
            'type': 'error',
            'message': 'Streaming transcription is not available'
        })
        await websocket.close()
        return

    await websocket.accept()

    # Extract workspace ID from query params or headers
    workspace_id = websocket.query_params.get(
        'workspace_id',
        websocket.headers.get('x-workspace-id', 'default')
    )

    session_id = streaming_mgr.create_session(workspace_id)
    await websocket.send_json({
        'type': 'session_start',
        'session_id': session_id
    })

    # Create live metrics computer if available
    metrics_computer = None
    if LIVE_METRICS_AVAILABLE and LiveMetricsComputer:
        try:
            metrics_computer = LiveMetricsComputer()
        except Exception as e:
            logger.warning(f"Failed to create LiveMetricsComputer: {e}")

    # Create live GM analyzer if available
    gm_analyzer = None
    if LIVE_ANALYSIS_AVAILABLE and LiveGMAnalyzer:
        try:
            gm_analyzer = LiveGMAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to create LiveGMAnalyzer: {e}")

    # Cache latest metrics for GM analysis prompt context
    latest_metrics = None
    # Background task handle for non-blocking GM analysis
    gm_analysis_task: Optional[asyncio.Task] = None

    logger.info(f"WebSocket streaming session started: {session_id}")

    # Track accumulated new audio duration for periodic transcription
    import time as _time
    last_process_time = _time.monotonic()

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: session {session_id}")
                break

            if message.get('type') == 'websocket.disconnect':
                break

            # Handle binary audio data
            if 'bytes' in message and message['bytes']:
                streaming_mgr.add_audio_chunk(session_id, message['bytes'])

                # Check if we should process new audio
                now = _time.monotonic()
                if now - last_process_time >= 10:
                    last_process_time = now

                    # Run transcription in thread executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    try:
                        processor = get_processor()
                        new_segments = await loop.run_in_executor(
                            None,
                            streaming_mgr.process_new_audio,
                            session_id,
                            processor
                        )
                        if new_segments:
                            await websocket.send_json({
                                'type': 'transcript',
                                'segments': new_segments
                            })
                            # Compute and send live metrics
                            if metrics_computer:
                                try:
                                    session = streaming_mgr.get_session(
                                        session_id
                                    )
                                    if session:
                                        all_segs = session.get_segments()
                                        metrics = await loop.run_in_executor(
                                            None,
                                            metrics_computer.compute,
                                            all_segs
                                        )
                                        latest_metrics = metrics
                                        await websocket.send_json({
                                            'type': 'live_metrics',
                                            'metrics': metrics
                                        })
                                except Exception as e:
                                    logger.warning(
                                        f"Live metrics error: {e}"
                                    )

                            # Run GM analysis as background task
                            # (non-blocking so it doesn't stall
                            # the audio receive loop)
                            if gm_analyzer:
                                task_running = (
                                    gm_analysis_task is not None
                                    and not gm_analysis_task.done()
                                )
                                if (
                                    not task_running
                                    and gm_analyzer.should_analyze()
                                ):
                                    session = streaming_mgr.get_session(
                                        session_id
                                    )
                                    if session:
                                        _segs = session.get_segments()
                                        _mets = latest_metrics

                                        async def _bg_gm_analysis(
                                            segs, mets
                                        ):
                                            try:
                                                _loop = asyncio.get_event_loop()
                                                res = await _loop.run_in_executor(
                                                    None,
                                                    gm_analyzer.analyze,
                                                    segs,
                                                    mets,
                                                )
                                                if res is not None:
                                                    await websocket.send_json({
                                                        'type': 'live_analysis',
                                                        'analysis': res,
                                                    })
                                            except Exception as exc:
                                                logger.warning(
                                                    f"Live GM analysis "
                                                    f"error: {exc}"
                                                )

                                        gm_analysis_task = (
                                            asyncio.create_task(
                                                _bg_gm_analysis(
                                                    _segs, _mets
                                                )
                                            )
                                        )
                    except Exception as e:
                        logger.warning(
                            f"Streaming transcription error: {e}"
                        )
                        await websocket.send_json({
                            'type': 'info',
                            'message': f'Transcription processing: {e}'
                        })

            # Handle text messages (control commands)
            elif 'text' in message and message['text']:
                try:
                    data = json.loads(message['text'])
                    if data.get('type') == 'stop':
                        logger.info(
                            f"Stop command received for session {session_id}"
                        )
                        # Do a final transcription pass (force=True
                        # to capture any trailing audio shorter than
                        # the minimum chunk threshold)
                        loop = asyncio.get_event_loop()
                        try:
                            processor = get_processor()

                            def _final_transcribe():
                                return streaming_mgr.process_new_audio(
                                    session_id, processor, force=True
                                )

                            new_segments = await loop.run_in_executor(
                                None, _final_transcribe
                            )
                            if new_segments:
                                await websocket.send_json({
                                    'type': 'transcript',
                                    'segments': new_segments
                                })
                            # Send final live metrics
                            if metrics_computer:
                                try:
                                    session = streaming_mgr.get_session(
                                        session_id
                                    )
                                    if session:
                                        all_segs = session.get_segments()
                                        metrics = await loop.run_in_executor(
                                            None,
                                            metrics_computer.compute,
                                            all_segs
                                        )
                                        await websocket.send_json({
                                            'type': 'live_metrics',
                                            'metrics': metrics
                                        })
                                except Exception as e:
                                    logger.warning(
                                        f"Final live metrics error: {e}"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"Final transcription pass error: {e}"
                            )

                        await websocket.send_json({
                            'type': 'session_end',
                            'session_id': session_id
                        })
                        break
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        logger.error(
            f"WebSocket error for session {session_id}: {e}",
            exc_info=True
        )
        try:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
        except Exception:
            pass

    # Cancel any in-flight GM analysis background task
    if gm_analysis_task is not None and not gm_analysis_task.done():
        gm_analysis_task.cancel()
        try:
            await gm_analysis_task
        except (asyncio.CancelledError, Exception):
            pass

    logger.info(f"WebSocket streaming session ended: {session_id}")


@app.get("/api/analysis-steps")
async def get_analysis_steps():
    """Get the list of analysis steps for progress tracking."""
    return {"steps": ANALYSIS_STEPS}


@app.get("/api/recordings/{filename}")
async def download_recording(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Download a saved recording."""
    # Check if audio downloads are disabled
    if DISABLE_AUDIO_FILES:
        raise HTTPException(
            status_code=403,
            detail="Audio file downloads are disabled on this server"
        )

    # Security: only allow files from recordings directory
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)

    # Check workspace first, then shared
    file_path = dirs['recordings'] / filename
    if not file_path.exists():
        shared_path = ws_mgr.shared_recordings_dir / filename
        if shared_path.exists():
            file_path = shared_path
        else:
            raise HTTPException(status_code=404, detail="Recording not found")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.get("/api/recordings")
async def list_recordings(
    workspace_id: str = Depends(get_workspace_id)
):
    """List all saved recordings."""
    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)

    seen_filenames = set()
    recordings = []

    # Workspace recordings first
    recordings_dir = dirs['recordings']
    if recordings_dir.exists():
        for f in sorted(recordings_dir.glob("*.wav"), reverse=True):
            seen_filenames.add(f.name)
            recordings.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "created": f.stat().st_mtime,
                "shared": False
            })

    # Shared (global) recordings
    shared_dir = ws_mgr.shared_recordings_dir
    if shared_dir.exists():
        for f in sorted(shared_dir.glob("*.wav"), reverse=True):
            if f.name not in seen_filenames:
                recordings.append({
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "created": f.stat().st_mtime,
                    "shared": True
                })

    # Sort combined list by creation time descending
    recordings.sort(key=lambda r: r["created"], reverse=True)

    return {"recordings": recordings[:50]}  # Limit to 50 most recent


@app.get("/api/analyses")
async def list_analyses(
    workspace_id: str = Depends(get_workspace_id)
):
    """List all saved analyses (workspace + shared)."""
    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    processor = get_processor()

    # Workspace analyses
    workspace_analyses = processor.list_analyses(analyses_dir=str(dirs['analyses']))
    for a in workspace_analyses:
        a['shared'] = False
    seen = {a['filename'] for a in workspace_analyses}

    # Shared (global) analyses
    shared_dir = ws_mgr.shared_analyses_dir
    if shared_dir.exists():
        shared_analyses = processor.list_analyses(analyses_dir=str(shared_dir))
        for a in shared_analyses:
            if a['filename'] not in seen:
                a['shared'] = True
                workspace_analyses.append(a)

    return {"analyses": workspace_analyses}


@app.get("/api/analyses/{filename}")
async def get_analysis(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Get a specific saved analysis (workspace first, then shared)."""
    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    processor = get_processor()

    # Check workspace first
    analysis = processor.get_analysis(filename, analyses_dir=str(dirs['analyses']))
    if analysis is not None:
        return analysis

    # Fall back to shared
    shared_dir = ws_mgr.shared_analyses_dir
    if shared_dir.exists():
        analysis = processor.get_analysis(filename, analyses_dir=str(shared_dir))
        if analysis is not None:
            return analysis

    raise HTTPException(status_code=404, detail="Analysis not found")


@app.delete("/api/analyses/{filename}")
async def delete_analysis(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Delete a saved analysis from workspace or shared analyses."""
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail="Server is in read-only mode. Deletion is disabled."
        )

    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    archive_mgr = ws_mgr.get_archive_manager(workspace_id)

    file_path = dirs['analyses'] / filename
    if not file_path.exists():
        # Check if it's in the shared analyses directory
        shared_path = ws_mgr.shared_analyses_dir / filename
        if shared_path.exists():
            file_path = shared_path
        else:
            raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        file_path.unlink()
        # Remove from workspace archive index
        archive_mgr.delete_analysis(filename)
        # Also remove from shared archive index if present
        try:
            shared_mgr = ws_mgr.get_shared_archive_manager()
            shared_mgr.delete_analysis(filename)
        except Exception:
            pass
        return {"status": "ok", "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyses/{filename}/regenerate-narrative")
async def regenerate_narrative(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Regenerate the AI team analysis narrative for an existing analysis."""
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail="Server is in read-only mode. Analysis regeneration is disabled."
        )

    if not NARRATIVE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Narrative generation not available. Ensure Ollama is running."
        )

    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    processor = get_processor()

    # Load existing analysis (workspace first, then shared)
    analyses_dir = _resolve_analysis(filename, dirs, ws_mgr)
    if analyses_dir is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis = processor.get_analysis(filename, analyses_dir=str(analyses_dir))
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        import time
        start_time = time.time()

        # Generate new narrative using async method
        logger.info(f"Regenerating narrative for {filename}")
        generator = NarrativeSummaryGenerator()
        # Pass the results object (handle nested structure)
        results_data = analysis.get('results', analysis)
        narrative_result = await generator.generate_summary(results_data)

        if narrative_result:
            generation_time = round(time.time() - start_time, 1)
            narrative_result['generation_time'] = generation_time

            # Update the analysis with new narrative (handle nested 'results' structure)
            if 'results' in analysis:
                analysis['results']['narrative_summary'] = narrative_result
            else:
                analysis['narrative_summary'] = narrative_result

            # Save updated analysis to workspace (even if source was shared)
            file_path = dirs['analyses'] / filename
            with open(file_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info(f"Narrative regenerated in {generation_time}s for {filename}")
            return {
                "status": "ok",
                "narrative_summary": narrative_result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate narrative. Check Ollama connection."
            )

    except Exception as e:
        logger.error(f"Error regenerating narrative: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyses/{filename}/regenerate-story")
async def regenerate_story(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Regenerate the AI mission story for an existing analysis."""
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail="Server is in read-only mode. Story regeneration is disabled."
        )

    if not NARRATIVE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Story generation not available. Ensure Ollama is running."
        )

    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    processor = get_processor()

    # Load existing analysis (workspace first, then shared)
    analyses_dir = _resolve_analysis(filename, dirs, ws_mgr)
    if analyses_dir is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis = processor.get_analysis(filename, analyses_dir=str(analyses_dir))
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        import time
        start_time = time.time()

        # Generate new story using async method
        logger.info(f"Regenerating story for {filename}")
        generator = NarrativeSummaryGenerator()
        # Pass the results object (handle nested structure)
        results_data = analysis.get('results', analysis)
        story_result = await generator.generate_story(results_data)

        if story_result:
            generation_time = round(time.time() - start_time, 1)
            story_result['generation_time'] = generation_time

            # Update the analysis with new story (handle nested 'results' structure)
            if 'results' in analysis:
                analysis['results']['story_narrative'] = story_result
            else:
                analysis['story_narrative'] = story_result

            # Save updated analysis
            file_path = dirs['analyses'] / filename
            with open(file_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)

            logger.info(f"Story regenerated in {generation_time}s for {filename}")
            return {
                "status": "ok",
                "story_narrative": story_result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate story. Check Ollama connection."
            )

    except Exception as e:
        logger.error(f"Error regenerating story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyses/{filename}/regenerate-narrative-stream")
async def regenerate_narrative_stream(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Regenerate narrative with streaming progress updates."""
    if READ_ONLY_MODE:
        async def readonly_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Server is in read-only mode. Regeneration is disabled.'})}\n\n"
        return StreamingResponse(readonly_stream(), media_type="text/event-stream")

    if not NARRATIVE_AVAILABLE:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Narrative generation not available'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    processor = get_processor()

    if '..' in filename or '/' in filename:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid filename'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Check workspace first, then shared
    resolved_dir = _resolve_analysis(filename, dirs, ws_mgr)
    if resolved_dir is None:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis not found'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    analysis = processor.get_analysis(filename, analyses_dir=str(resolved_dir))
    if analysis is None:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis not found'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate_stream():
        import time
        from src.llm.llm_client import LLMClient

        start_time = time.time()
        generator = NarrativeSummaryGenerator()

        # Get the results object (handle nested structure)
        results_data = analysis.get('results', analysis)

        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting narrative generation...'})}\n\n"

        try:
            # Check LLM availability
            if not await generator.check_llm_available():
                yield f"data: {json.dumps({'type': 'error', 'message': 'LLM not available'})}\n\n"
                return

            # Build prompt using results data
            prompt = generator._build_prompt(results_data)
            yield f"data: {json.dumps({'type': 'progress', 'chars': 0, 'message': 'Sending to LLM...'})}\n\n"

            # Stream from LLM backend
            narrative_parts = []
            chars_generated = 0

            async for chunk_text in generator._llm.agenerate_streaming(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2500,
            ):
                narrative_parts.append(chunk_text)
                chars_generated += len(chunk_text)
                yield f"data: {json.dumps({'type': 'progress', 'chars': chars_generated, 'message': f'Generating... {chars_generated} chars'})}\n\n"

            narrative = ''.join(narrative_parts).strip()
            generation_time = round(time.time() - start_time, 1)

            if narrative:
                narrative_result = {
                    'narrative': narrative,
                    'model': generator.ollama_model,
                    'generation_time': generation_time
                }

                # Update and save analysis (handle nested 'results' structure)
                if 'results' in analysis:
                    analysis['results']['narrative_summary'] = narrative_result
                else:
                    analysis['narrative_summary'] = narrative_result

                file_path = dirs['analyses'] / filename
                with open(file_path, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)

                yield f"data: {json.dumps({'type': 'complete', 'narrative_summary': narrative_result, 'chars': chars_generated})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Empty response from LLM'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming narrative error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/api/analyses/{filename}/regenerate-story-stream")
async def regenerate_story_stream(
    filename: str,
    workspace_id: str = Depends(get_workspace_id)
):
    """Regenerate story with streaming progress updates."""
    if READ_ONLY_MODE:
        async def readonly_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Server is in read-only mode. Regeneration is disabled.'})}\n\n"
        return StreamingResponse(readonly_stream(), media_type="text/event-stream")

    if not NARRATIVE_AVAILABLE:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Story generation not available'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    processor = get_processor()

    if '..' in filename or '/' in filename:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid filename'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Check workspace first, then shared
    resolved_dir = _resolve_analysis(filename, dirs, ws_mgr)
    if resolved_dir is None:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis not found'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    analysis = processor.get_analysis(filename, analyses_dir=str(resolved_dir))
    if analysis is None:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis not found'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def generate_stream():
        import time

        start_time = time.time()
        generator = NarrativeSummaryGenerator()

        # Get the results object (handle nested structure)
        results_data = analysis.get('results', analysis)

        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting story generation...'})}\n\n"

        try:
            if not await generator.check_llm_available():
                yield f"data: {json.dumps({'type': 'error', 'message': 'LLM not available'})}\n\n"
                return

            # Build story prompt using results data
            prompt = generator._build_story_prompt(results_data)
            yield f"data: {json.dumps({'type': 'progress', 'chars': 0, 'message': 'Sending to LLM...'})}\n\n"

            story_parts = []
            chars_generated = 0

            async for chunk_text in generator._llm.agenerate_streaming(
                prompt=prompt,
                temperature=0.5,
                max_tokens=2500,
                top_p=0.9,
                top_k=50,
                repeat_penalty=1.1,
            ):
                story_parts.append(chunk_text)
                chars_generated += len(chunk_text)
                yield f"data: {json.dumps({'type': 'progress', 'chars': chars_generated, 'message': f'Writing... {chars_generated} chars'})}\n\n"

            story = ''.join(story_parts).strip()
            generation_time = round(time.time() - start_time, 1)

            if story:
                # Validate output if hallucination prevention available
                validation_issues = []
                if HALLUCINATION_PREVENTION_AVAILABLE and clean_hallucinations:
                    try:
                        transcripts = results_data.get('transcription', [])
                        story, validation_issues = clean_hallucinations(
                            story,
                            transcripts,
                            results_data,
                            add_warning=True
                        )
                    except Exception as e:
                        logger.warning(f"Hallucination validation failed: {e}")

                story_result = {
                    'story': story,
                    'model': generator.ollama_model,
                    'generation_time': generation_time,
                    'validation_issues': len(validation_issues) if validation_issues else 0
                }

                # Update and save analysis (handle nested 'results' structure)
                if 'results' in analysis:
                    analysis['results']['story_narrative'] = story_result
                else:
                    analysis['story_narrative'] = story_result

                file_path = dirs['analyses'] / filename
                with open(file_path, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)

                yield f"data: {json.dumps({'type': 'complete', 'story_narrative': story_result, 'chars': chars_generated})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Empty response from LLM'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming story error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# ============================================================================
# Archive Index Routes
# ============================================================================

@app.get("/api/archive-index")
async def get_archive_index(
    starred_only: bool = Query(False, description="Only return starred analyses"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    search: Optional[str] = Query(None, description="Search in titles and notes"),
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Get the full archive index with metadata.

    Returns all analyses with their titles, tags, notes, and other metadata.
    Includes both workspace-specific and shared (global) analyses.
    Supports filtering by starred status, tags, and search query.
    """
    ws_mgr = get_workspace_manager()
    archive_mgr = ws_mgr.get_archive_manager(workspace_id)

    # Sync with filesystem in case files were added/removed externally
    archive_mgr.sync_with_filesystem()

    analyses = archive_mgr.list_analyses(
        starred_only=starred_only,
        tag_filter=tag,
        search_query=search
    )

    # Workspace analyses
    ws_list = []
    seen = set()
    for a in analyses:
        d = a.to_dict()
        d['shared'] = False
        ws_list.append(d)
        seen.add(a.filename)

    # Shared (global) analyses
    shared_mgr = ws_mgr.get_shared_archive_manager()
    shared_mgr.sync_with_filesystem()
    shared_analyses = shared_mgr.list_analyses(
        starred_only=starred_only,
        tag_filter=tag,
        search_query=search
    )
    for a in shared_analyses:
        if a.filename not in seen:
            d = a.to_dict()
            d['shared'] = True
            ws_list.append(d)

    # Merge tags from both
    all_tags = list(set(archive_mgr.get_all_tags() + shared_mgr.get_all_tags()))

    return {
        "analyses": ws_list,
        "summary": archive_mgr.get_index_summary(),
        "tags": all_tags
    }


@app.patch("/api/analyses/{filename}/metadata")
async def update_analysis_metadata(
    filename: str,
    update: MetadataUpdate,
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Update metadata for a specific analysis.

    Allows updating:
    - user_title: Custom title (overrides auto-generated)
    - tags: List of tags
    - notes: Free-form notes
    - starred: Favorite/star status

    Updates both:
    - archive_index.json (for quick listing)
    - The individual analysis JSON file (for persistence)
    """
    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    dirs = ws_mgr.ensure_workspace(workspace_id)
    archive_mgr = ws_mgr.get_archive_manager(workspace_id)

    # Check if analysis exists in workspace or shared
    is_shared = False
    metadata = archive_mgr.get_analysis(filename)
    if not metadata:
        archive_mgr.sync_with_filesystem()
        metadata = archive_mgr.get_analysis(filename)
    if not metadata:
        # Check shared archive
        shared_mgr = ws_mgr.get_shared_archive_manager()
        shared_meta = shared_mgr.get_analysis(filename)
        if not shared_meta:
            shared_mgr.sync_with_filesystem()
            shared_meta = shared_mgr.get_analysis(filename)
        if not shared_meta:
            raise HTTPException(status_code=404, detail="Analysis not found")
        # Register shared analysis in workspace archive for metadata tracking
        is_shared = True
        archive_mgr.add_analysis(
            filename=shared_meta.filename,
            recording_filename=shared_meta.recording_filename,
            auto_title=shared_meta.auto_title,
            duration_seconds=shared_meta.duration_seconds,
            speaker_count=shared_meta.speaker_count,
            segment_count=shared_meta.segment_count
        )

    # Update workspace archive index
    updated = archive_mgr.update_analysis(
        filename=filename,
        user_title=update.user_title,
        tags=update.tags,
        notes=update.notes,
        starred=update.starred
    )

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update metadata")

    # Also update the analysis JSON file itself (workspace copy only)
    try:
        file_path = dirs['analyses'] / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                analysis_data = json.load(f)

            # Update metadata in the analysis file
            if 'metadata' not in analysis_data:
                analysis_data['metadata'] = {}

            if update.user_title is not None:
                analysis_data['metadata']['user_title'] = update.user_title
            if update.tags is not None:
                analysis_data['metadata']['tags'] = update.tags
            if update.notes is not None:
                analysis_data['metadata']['notes'] = update.notes
            if update.starred is not None:
                analysis_data['metadata']['starred'] = update.starred

            # Save updated analysis file
            with open(file_path, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)

            logger.info(f"Updated analysis file metadata: {filename}")

    except Exception as e:
        logger.warning(f"Failed to update analysis file {filename}: {e}")
        # Don't fail the request - index is already updated

    return {"status": "ok", "metadata": updated.to_dict()}


@app.post("/api/archive-index/sync")
async def sync_archive_index(
    workspace_id: str = Depends(get_workspace_id)
):
    """
    Manually trigger archive index synchronization.

    Syncs the index with the filesystem, adding entries for new files
    and removing entries for deleted files.
    """
    ws_mgr = get_workspace_manager()
    archive_mgr = ws_mgr.get_archive_manager(workspace_id)
    result = archive_mgr.sync_with_filesystem()

    return {
        "status": "ok",
        "added": result['added'],
        "removed": result['removed'],
        "total": len(archive_mgr.list_analyses())
    }


# ============================================================================
# Archive Browsing Routes
# ============================================================================

ARCHIVE_BASE = Path(os.getenv('DATA_DIR', 'data')) / 'archive'
ARCHIVE_ID_PATTERN = re.compile(r'^\d{8}_\d{6}$')


def _validate_archive_id(archive_id: str) -> Path:
    """
    Validate archive ID and return its directory path.

    Args:
        archive_id: Archive batch identifier (YYYYMMDD_HHMMSS).

    Returns:
        Path to the archive directory.

    Raises:
        HTTPException: If archive ID is invalid or not found.
    """
    if not ARCHIVE_ID_PATTERN.match(archive_id):
        raise HTTPException(status_code=400, detail="Invalid archive ID format")
    archive_dir = ARCHIVE_BASE / archive_id
    if not archive_dir.is_dir():
        raise HTTPException(status_code=404, detail="Archive not found")
    return archive_dir


@app.get("/api/archives")
async def list_archives():
    """
    List all archive batches.

    Scans data/archive/ for directories containing a manifest.json.
    Returns a summary of each batch sorted newest-first.
    """
    if not ARCHIVE_BASE.is_dir():
        return {"archives": []}

    archives = []
    for entry in sorted(ARCHIVE_BASE.iterdir(), reverse=True):
        if not entry.is_dir() or not ARCHIVE_ID_PATTERN.match(entry.name):
            continue
        manifest_path = entry / 'manifest.json'
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            archives.append({
                'archive_id': manifest.get('archive_id', entry.name),
                'created_at': manifest.get('created_at'),
                'description': manifest.get('description', ''),
                'file_counts': manifest.get('file_counts', {}),
                'total_size_bytes': manifest.get('total_size_bytes', 0),
                'session_count': len(manifest.get('sessions', [])),
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read manifest for archive {entry.name}: {e}")

    return {"archives": archives}


@app.get("/api/archives/{archive_id}")
async def get_archive_manifest(archive_id: str):
    """
    Get the full manifest for an archive batch.

    Returns session groupings, file counts, and metadata.
    """
    archive_dir = _validate_archive_id(archive_id)
    manifest_path = archive_dir / 'manifest.json'

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        return manifest
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid manifest: {e}")


@app.get("/api/archives/{archive_id}/analyses")
async def list_archive_analyses(archive_id: str):
    """
    List all analyses in an archive batch.

    Returns the archive index with metadata for each analysis.
    """
    archive_dir = _validate_archive_id(archive_id)
    index_path = archive_dir / 'archive_index.json'

    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid archive index: {e}")

    # Fallback: list files directly
    analyses_dir = archive_dir / 'analyses'
    if not analyses_dir.is_dir():
        return {"analyses": {}}

    files = sorted(f.name for f in analyses_dir.glob('analysis_*.json'))
    return {"analyses": {f.replace('.json', ''): {"filename": f} for f in files}}


@app.get("/api/archives/{archive_id}/analyses/{filename}")
async def get_archive_analysis(archive_id: str, filename: str):
    """
    Get a specific analysis JSON from an archive batch.

    Returns the full analysis data including results and metadata.
    """
    archive_dir = _validate_archive_id(archive_id)

    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.endswith('.json'):
        filename = f"{filename}.json"

    file_path = archive_dir / 'analyses' / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    return FileResponse(
        str(file_path),
        media_type="application/json",
        filename=filename,
    )


@app.get("/api/archives/{archive_id}/recordings/{filename}")
async def get_archive_recording(archive_id: str, filename: str):
    """
    Download a recording WAV file from an archive batch.
    """
    archive_dir = _validate_archive_id(archive_id)

    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = archive_dir / 'recordings' / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


@app.get("/api/archives/{archive_id}/telemetry/{filename}")
async def get_archive_telemetry(archive_id: str, filename: str):
    """
    Download a telemetry JSON file from an archive batch.
    """
    archive_dir = _validate_archive_id(archive_id)

    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.endswith('.json'):
        filename = f"{filename}.json"

    file_path = archive_dir / 'telemetry' / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Telemetry file not found")

    return FileResponse(
        str(file_path),
        media_type="application/json",
        filename=filename,
    )


@app.post("/api/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Transcribe audio file (no diarization or quality analysis).

    Faster than full analysis when only transcription is needed.
    """
    # Check if audio uploads are disabled
    if DISABLE_AUDIO_FILES:
        raise HTTPException(
            status_code=403,
            detail="Audio file uploads are disabled on this server"
        )

    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_MB}MB"
        )

    # Save to temp file
    suffix = Path(file.filename or 'audio.wav').suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        processor = get_processor()
        results = processor.transcribe_only(tmp_path)

        return TranscriptionResult(
            segments=results['segments'],
            full_text=results['full_text'],
            duration_seconds=results['duration_seconds'],
            language=results['language'],
            processing_time_seconds=results['processing_time_seconds']
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/load-model")
async def load_model():
    """Manually trigger Whisper model loading."""
    try:
        processor = get_processor()
        success = processor.load_model()
        if success:
            return {"status": "ok", "message": "Model loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Admin Panel Routes
# ============================================================================

@app.get("/admin", response_class=FileResponse)
async def serve_admin():
    """Serve the admin panel HTML page."""
    static_dir = Path(__file__).parent.parent.parent / "static"
    admin_path = static_dir / "admin.html"
    if admin_path.exists():
        return FileResponse(str(admin_path))
    raise HTTPException(status_code=404, detail="Admin panel not found")


@app.get("/api/admin/stats")
async def admin_global_stats():
    """Get global statistics across all workspaces."""
    ws_mgr = get_workspace_manager()
    return ws_mgr.get_global_stats()


@app.get("/api/admin/workspaces")
async def admin_list_workspaces():
    """List all workspaces with summary statistics."""
    ws_mgr = get_workspace_manager()
    return {"workspaces": ws_mgr.list_workspaces()}


@app.get("/api/admin/workspaces/{workspace_id}")
async def admin_get_workspace(workspace_id: str):
    """Get detailed statistics for a single workspace."""
    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")
    stats = ws_mgr.get_workspace_stats(workspace_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return stats


@app.delete("/api/admin/workspaces/{workspace_id}")
async def admin_delete_workspace(workspace_id: str):
    """Delete an entire workspace and all its data."""
    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")
    deleted = ws_mgr.delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"status": "ok", "message": f"Workspace {workspace_id} deleted"}


@app.get("/api/admin/workspaces/{workspace_id}/archive-index")
async def admin_workspace_archive_index(workspace_id: str):
    """Get the archive index for a specific workspace."""
    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")

    ws_dir = ws_mgr.base_dir / workspace_id
    if not ws_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    archive_mgr = ws_mgr.get_archive_manager(workspace_id)
    archive_mgr.sync_with_filesystem()
    analyses = archive_mgr.list_analyses()
    return {
        "analyses": [a.to_dict() for a in analyses],
        "summary": archive_mgr.get_index_summary(),
    }


@app.get("/api/admin/workspaces/{workspace_id}/analyses/{filename}")
async def admin_get_analysis(workspace_id: str, filename: str):
    """Get a specific analysis JSON from a workspace."""
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")

    ws_dir = ws_mgr.base_dir / workspace_id
    if not ws_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    file_path = ws_dir / 'analyses' / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to read analysis: {e}")


@app.get("/api/admin/workspaces/{workspace_id}/analyses/{filename}/download")
async def admin_download_analysis(workspace_id: str, filename: str):
    """Download an analysis file from a workspace."""
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")

    file_path = ws_mgr.base_dir / workspace_id / 'analyses' / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    return FileResponse(
        str(file_path),
        media_type="application/json",
        filename=filename,
    )


@app.delete("/api/admin/workspaces/{workspace_id}/analyses/{filename}")
async def admin_delete_analysis(workspace_id: str, filename: str):
    """Delete an analysis file from a workspace (admin can also delete shared)."""
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")

    # Check workspace analyses first
    file_path = ws_mgr.base_dir / workspace_id / 'analyses' / filename
    if file_path.exists():
        try:
            file_path.unlink()
            archive_mgr = ws_mgr.get_archive_manager(workspace_id)
            archive_mgr.delete_analysis(filename)
            return {"status": "ok", "message": f"Deleted {filename}"}
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Admin privilege: also check and delete shared analyses
    shared_path = ws_mgr.shared_analyses_dir / filename
    if shared_path.exists():
        try:
            shared_path.unlink()
            shared_mgr = ws_mgr.get_shared_archive_manager()
            shared_mgr.delete_analysis(filename)
            return {"status": "ok", "message": f"Deleted shared analysis {filename}"}
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail="Analysis not found")


@app.get("/api/admin/workspaces/{workspace_id}/recordings/{filename}")
async def admin_get_recording(workspace_id: str, filename: str):
    """Serve a recording audio file from a workspace."""
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ws_mgr = get_workspace_manager()
    if not ws_mgr.validate_workspace_id(workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace ID format")

    file_path = ws_mgr.base_dir / workspace_id / 'recordings' / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


@app.get("/api/admin/shared")
async def admin_list_shared():
    """List all shared data files (analyses, recordings, telemetry)."""
    ws_mgr = get_workspace_manager()
    result: Dict[str, list] = {
        'analyses': [],
        'recordings': [],
        'telemetry': [],
    }

    for key, dir_path in [
        ('analyses', ws_mgr.shared_analyses_dir),
        ('recordings', ws_mgr.shared_recordings_dir),
        ('telemetry', ws_mgr.shared_telemetry_dir),
    ]:
        if dir_path.exists():
            for f in sorted(dir_path.iterdir()):
                if f.is_file():
                    try:
                        stat = f.stat()
                        result[key].append({
                            'filename': f.name,
                            'size_bytes': stat.st_size,
                            'modified': stat.st_mtime,
                        })
                    except OSError:
                        continue

    return result


# ============================================================================
# Game Server Configuration Endpoints
# ============================================================================

@app.get("/api/admin/game-config")
async def get_game_config():
    """
    Get current game server configuration.

    Returns the current GAME_HOST, GAME_PORT_WS, and GAME_PORT_API
    values from the environment, plus a read_only flag.
    """
    return {
        "game_host": os.environ.get('GAME_HOST', ''),
        "game_port_ws": int(os.environ.get('GAME_PORT_WS', '1865')),
        "game_port_api": int(os.environ.get('GAME_PORT_API', '1864')),
        "game_port_https": int(os.environ.get('GAME_PORT_HTTPS', '1866')),
        "game_port_wss": int(os.environ.get('GAME_PORT_WSS', '1867')),
        "read_only": READ_ONLY_MODE,
    }


@app.post("/api/admin/game-config")
async def update_game_config(config: GameServerConfig):
    """
    Update game server configuration.

    Validates input, updates os.environ for immediate effect, and
    persists the changes to the .env file for restart durability.
    Blocked by READ_ONLY_MODE.
    """
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail="Server is in read-only mode. Configuration changes are disabled."
        )

    # Validate hostname
    if not config.game_host.strip():
        raise HTTPException(status_code=400, detail="Game host cannot be empty")
    if not _is_valid_host(config.game_host.strip()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hostname or IP address: {config.game_host}"
        )

    host = config.game_host.strip()
    port_ws = str(config.game_port_ws)
    port_api = str(config.game_port_api)
    port_https = str(config.game_port_https)
    port_wss = str(config.game_port_wss)

    # Update os.environ for immediate effect
    os.environ['GAME_HOST'] = host
    os.environ['GAME_PORT_WS'] = port_ws
    os.environ['GAME_PORT_API'] = port_api
    os.environ['GAME_PORT_HTTPS'] = port_https
    os.environ['GAME_PORT_WSS'] = port_wss

    # Persist to .env file
    env_error = None
    try:
        _update_env_file({
            'GAME_HOST': host,
            'GAME_PORT_WS': port_ws,
            'GAME_PORT_API': port_api,
            'GAME_PORT_HTTPS': port_https,
            'GAME_PORT_WSS': port_wss,
        })
    except OSError as e:
        env_error = str(e)
        logger.error(f"Failed to write .env file: {e}")

    logger.info(f"Game server config updated: {host}:{port_ws} (API: {port_api})")

    result = {
        "status": "ok",
        "game_host": host,
        "game_port_ws": int(port_ws),
        "game_port_api": int(port_api),
        "game_port_https": int(port_https),
        "game_port_wss": int(port_wss),
    }

    if env_error:
        result["warning"] = (
            f"Configuration applied in memory but failed to save to .env file: {env_error}. "
            "Changes will be lost on restart."
        )

    return result


@app.post("/api/admin/test-connection", response_model=ConnectionTestResponse)
async def test_game_connection(request: ConnectionTestRequest):
    """
    Test connectivity to a game server.

    Performs TCP socket connect tests on both the WebSocket and API ports
    with a 3-second timeout. Accepts arbitrary values so the user can
    test before saving.
    """
    if not request.host.strip():
        raise HTTPException(status_code=400, detail="Host cannot be empty")
    if not _is_valid_host(request.host.strip()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hostname or IP address: {request.host}"
        )

    host = request.host.strip()
    timeout = 3.0

    def _test_port(port: int) -> tuple:
        """Test TCP connectivity to host:port."""
        try:
            socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        except socket.gaierror as e:
            return False, f"DNS resolution failed for {host}: {e}"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True, f"Connected to {host}:{port}"
        except ConnectionRefusedError:
            return False, f"Connection refused on {host}:{port}"
        except (TimeoutError, socket.timeout):
            return False, f"Connection timed out on {host}:{port} ({timeout:.0f}s)"
        except OSError as e:
            return False, f"Connection error on {host}:{port}: {e}"
        finally:
            sock.close()

    # Run both tests (blocking but fast with 3s timeout)
    ws_ok, ws_detail = await asyncio.get_event_loop().run_in_executor(
        None, _test_port, request.port_ws
    )
    api_ok, api_detail = await asyncio.get_event_loop().run_in_executor(
        None, _test_port, request.port_api
    )

    return ConnectionTestResponse(
        ws_reachable=ws_ok,
        api_reachable=api_ok,
        ws_detail=ws_detail,
        api_detail=api_detail,
    )


# ============================================================================
# GPU & LLM Configuration Endpoint
# ============================================================================

@app.get("/api/admin/gpu-llm-config")
async def get_gpu_llm_config():
    """
    Get current LLM backend configuration, vLLM status, and GPU info.

    Returns:
        JSON with llm, vllm, gpus, gpu_count, ram_total_mb, ram_available_mb.
    """
    def _collect() -> dict:
        """Collect hardware and LLM config (may block on probes)."""
        from src.hardware.detector import HardwareDetector
        from src.llm.llm_client import _resolve_base_url, _resolve_model, _resolve_timeout

        # Hardware detection
        detector = HardwareDetector()
        profile = detector.detect()

        # LLM config
        base_url = _resolve_base_url()
        model = _resolve_model()
        timeout = _resolve_timeout()

        vllm_port = os.getenv('VLLM_PORT', '8100')
        if os.getenv('LLM_BASE_URL'):
            backend = "vllm" if vllm_port in os.getenv('LLM_BASE_URL', '') else "custom"
        else:
            backend = "ollama"

        llm_section = {
            "base_url": base_url,
            "model": model,
            "timeout": timeout,
            "backend": backend,
        }

        # vLLM config
        try:
            sys.path.insert(0, str(_PROJECT_ROOT / 'scripts'))
            from vllm_setup import _get_config, is_vllm_running, _probe_endpoint
            cfg = _get_config()
            running = is_vllm_running()
            _, served = _probe_endpoint(cfg['port'])
            vllm_section = {
                "running": running,
                "port": cfg['port'],
                "model": cfg['model'],
                "quantization": cfg['quantization'],
                "enforce_eager": cfg['enforce_eager'],
                "gpu_memory_utilization": cfg['gpu_memory_utilization'],
                "max_model_len": cfg['max_model_len'],
                "tensor_parallel_size": cfg['tensor_parallel_size'],
                "pipeline_parallel_size": cfg['pipeline_parallel_size'],
                "data_parallel_size": cfg['data_parallel_size'],
                "auto_start": os.getenv('VLLM_AUTO_START', 'false').lower() == 'true',
                "served_models": served,
            }
        except Exception as e:
            logger.warning(f"Failed to read vLLM config: {e}")
            vllm_section = {"running": False, "error": str(e)}

        gpus = [
            {
                "index": g.index,
                "name": g.name,
                "total_memory_mb": g.total_memory_mb,
                "free_memory_mb": g.free_memory_mb,
            }
            for g in profile.gpus
        ]

        return {
            "llm": llm_section,
            "vllm": vllm_section,
            "gpus": gpus,
            "gpu_count": profile.gpu_count,
            "ram_total_mb": profile.ram_total_mb,
            "ram_available_mb": profile.ram_available_mb,
        }

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _collect)
        return result
    except Exception as e:
        logger.error(f"GPU/LLM config collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    """Handle file too large errors."""
    return JSONResponse(
        status_code=413,
        content={"error": "File too large", "detail": str(exc.detail)}
    )


@app.exception_handler(415)
async def unsupported_media_type(request, exc):
    """Handle unsupported file type errors."""
    return JSONResponse(
        status_code=415,
        content={"error": "Unsupported file type", "detail": str(exc.detail)}
    )
