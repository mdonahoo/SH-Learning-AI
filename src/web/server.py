"""
FastAPI web server for audio analysis.

Provides REST API endpoints for audio transcription and analysis.
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
WEB_HOST = os.getenv('WEB_SERVER_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('WEB_SERVER_PORT', '8000'))
MAX_UPLOAD_MB = int(os.getenv('WEB_MAX_UPLOAD_MB', '2048'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CORS_ORIGINS = os.getenv('WEB_CORS_ORIGINS', '*').split(',')

# Global processor and archive manager instances
_processor: Optional[AudioProcessor] = None
_archive_manager: Optional[ArchiveManager] = None


def get_processor() -> AudioProcessor:
    """Get or create the audio processor instance."""
    global _processor
    if _processor is None:
        _processor = AudioProcessor()
    return _processor


def get_archive_manager() -> ArchiveManager:
    """Get or create the archive manager instance."""
    global _archive_manager
    if _archive_manager is None:
        _archive_manager = ArchiveManager()
        # Sync with filesystem on first access
        _archive_manager.sync_with_filesystem()
    return _archive_manager


# Request model for metadata updates
class MetadataUpdate(BaseModel):
    """Request body for updating analysis metadata."""
    user_title: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    starred: Optional[bool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for model loading."""
    logger.info("Starting audio analysis server...")
    processor = get_processor()

    # Preload model on startup (disable with PRELOAD_WHISPER=false)
    if os.getenv('PRELOAD_WHISPER', 'true').lower() == 'true':
        logger.info("Preloading Whisper model...")
        processor.load_model()

    yield

    logger.info("Shutting down audio analysis server...")


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

    # Check Ollama status
    ollama_status = ServiceStatus(
        available=False,
        status="Not connected",
        details=None
    )

    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2')

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{ollama_host}/api/tags")
            if response.status_code == 200:
                tags_data = response.json()
                models = tags_data.get('models', [])
                # Get both full names and base names for matching
                full_names = [m.get('name', '') for m in models]
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
                        details=f"Model: {ollama_model} ({len(models)} models available)"
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
                    status="Error",
                    details=f"HTTP {response.status_code}"
                )
    except httpx.ConnectError:
        ollama_status = ServiceStatus(
            available=False,
            status="Not running",
            details=f"Cannot connect to {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}"
        )
    except httpx.TimeoutException:
        ollama_status = ServiceStatus(
            available=False,
            status="Timeout",
            details="Connection timed out"
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
    )
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
        processor = get_processor()
        results = processor.analyze_audio(
            tmp_path,
            include_diarization=include_diarization,
            include_quality=include_quality,
            include_detailed=include_detailed
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
    include_story: bool = Query(True, description="Include LLM mission story")
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
    """
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

    logger.info(f"Analysis options: include_narrative={_include_narrative}, include_story={_include_story}")

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
                progress_callback=progress_callback
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
        while True:
            try:
                # Check for updates (non-blocking with timeout)
                try:
                    msg = progress_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue

                yield f"data: {json.dumps(msg)}\n\n"

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
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/analysis-steps")
async def get_analysis_steps():
    """Get the list of analysis steps for progress tracking."""
    return {"steps": ANALYSIS_STEPS}


@app.get("/api/recordings/{filename}")
async def download_recording(filename: str):
    """Download a saved recording."""
    processor = get_processor()

    # Security: only allow files from recordings directory
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = processor.recordings_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.get("/api/recordings")
async def list_recordings():
    """List all saved recordings."""
    processor = get_processor()

    if not processor.recordings_dir.exists():
        return {"recordings": []}

    recordings = []
    for f in sorted(processor.recordings_dir.glob("*.wav"), reverse=True):
        recordings.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "created": f.stat().st_mtime
        })

    return {"recordings": recordings[:50]}  # Limit to 50 most recent


@app.get("/api/analyses")
async def list_analyses():
    """List all saved analyses."""
    processor = get_processor()
    analyses = processor.list_analyses()
    return {"analyses": analyses}


@app.get("/api/analyses/{filename}")
async def get_analysis(filename: str):
    """Get a specific saved analysis."""
    processor = get_processor()

    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    analysis = processor.get_analysis(filename)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return analysis


@app.delete("/api/analyses/{filename}")
async def delete_analysis(filename: str):
    """Delete a saved analysis."""
    processor = get_processor()
    archive_mgr = get_archive_manager()

    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = processor.analyses_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        file_path.unlink()
        # Also remove from archive index
        archive_mgr.delete_analysis(filename)
        return {"status": "ok", "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Archive Index Routes
# ============================================================================

@app.get("/api/archive-index")
async def get_archive_index(
    starred_only: bool = Query(False, description="Only return starred analyses"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    search: Optional[str] = Query(None, description="Search in titles and notes")
):
    """
    Get the full archive index with metadata.

    Returns all analyses with their titles, tags, notes, and other metadata.
    Supports filtering by starred status, tags, and search query.
    """
    archive_mgr = get_archive_manager()

    # Sync with filesystem in case files were added/removed externally
    archive_mgr.sync_with_filesystem()

    analyses = archive_mgr.list_analyses(
        starred_only=starred_only,
        tag_filter=tag,
        search_query=search
    )

    return {
        "analyses": [a.to_dict() for a in analyses],
        "summary": archive_mgr.get_index_summary(),
        "tags": archive_mgr.get_all_tags()
    }


@app.patch("/api/analyses/{filename}/metadata")
async def update_analysis_metadata(
    filename: str,
    update: MetadataUpdate
):
    """
    Update metadata for a specific analysis.

    Allows updating:
    - user_title: Custom title (overrides auto-generated)
    - tags: List of tags
    - notes: Free-form notes
    - starred: Favorite/star status
    """
    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    archive_mgr = get_archive_manager()

    # Check if analysis exists
    metadata = archive_mgr.get_analysis(filename)
    if not metadata:
        # Try to sync and check again
        archive_mgr.sync_with_filesystem()
        metadata = archive_mgr.get_analysis(filename)
        if not metadata:
            raise HTTPException(status_code=404, detail="Analysis not found")

    # Update metadata
    updated = archive_mgr.update_analysis(
        filename=filename,
        user_title=update.user_title,
        tags=update.tags,
        notes=update.notes,
        starred=update.starred
    )

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update metadata")

    return {"status": "ok", "metadata": updated.to_dict()}


@app.post("/api/archive-index/sync")
async def sync_archive_index():
    """
    Manually trigger archive index synchronization.

    Syncs the index with the filesystem, adding entries for new files
    and removing entries for deleted files.
    """
    archive_mgr = get_archive_manager()
    result = archive_mgr.sync_with_filesystem()

    return {
        "status": "ok",
        "added": result['added'],
        "removed": result['removed'],
        "total": len(archive_mgr.list_analyses())
    }


@app.post("/api/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Transcribe audio file (no diarization or quality analysis).

    Faster than full analysis when only transcription is needed.
    """
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
