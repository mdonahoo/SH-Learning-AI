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
from typing import Optional, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.web.models import (
    AnalysisResult,
    TranscriptionResult,
    HealthResponse,
    ErrorResponse,
)
from src.web.audio_processor import AudioProcessor, ANALYSIS_STEPS

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
WEB_HOST = os.getenv('WEB_SERVER_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('WEB_SERVER_PORT', '8000'))
MAX_UPLOAD_MB = int(os.getenv('WEB_MAX_UPLOAD_MB', '50'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CORS_ORIGINS = os.getenv('WEB_CORS_ORIGINS', '*').split(',')

# Global processor instance
_processor: Optional[AudioProcessor] = None


def get_processor() -> AudioProcessor:
    """Get or create the audio processor instance."""
    global _processor
    if _processor is None:
        _processor = AudioProcessor()
    return _processor


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
    file: UploadFile = File(..., description="Audio file to analyze")
):
    """
    Analyze audio file with streaming progress updates.

    Returns Server-Sent Events (SSE) stream with:
    - Progress updates: {"type": "progress", "step": "...", "label": "...", "progress": N}
    - Final result: {"type": "result", "data": {...}}
    - Error: {"type": "error", "message": "..."}
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

    def run_analysis():
        """Run analysis in background thread."""
        try:
            processor = get_processor()
            results = processor.analyze_audio(
                tmp_path,
                include_diarization=True,
                include_quality=True,
                include_detailed=True,
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

    # Security: only allow valid filenames
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = processor.analyses_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        file_path.unlink()
        return {"status": "ok", "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
