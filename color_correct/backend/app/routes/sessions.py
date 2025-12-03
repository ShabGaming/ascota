"""Session management routes."""

from fastapi import APIRouter, HTTPException, Query
from typing import List
import logging
from pathlib import Path

from app.services.models import CreateSessionRequest, CreateSessionResponse, JobStatusResponse
from app.services.session_store import get_session_store
from app.services.session_persistence import list_sessions as list_persisted_sessions
from app.services.ascota_storage import ensure_context_ascota_folder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new color correction session."""
    try:
        store = get_session_store()
        session_id = store.create_session(request.contexts, request.options)
        logger.info(f"Created session {session_id}")
        return CreateSessionResponse(session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/status", response_model=JobStatusResponse)
async def get_session_status(session_id: str, job_id: str = None):
    """Get status of a session or specific job."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if job_id:
        job = session.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_response()
    
    # Return general session status
    return JobStatusResponse(
        status="completed",
        progress=1.0,
        message="Session active"
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session from memory and disk.
    
    Returns success even if session doesn't exist (idempotent operation).
    """
    store = get_session_store()
    
    # Try to delete from memory (may not exist)
    store.delete_session(session_id)
    
    # Also try to delete from disk (may not exist)
    from app.services.session_persistence import delete_session as delete_persisted_session
    delete_persisted_session(session_id)
    
    # Always return success (idempotent - deleting non-existent session is OK)
    return {"message": "Session deleted"}


@router.get("")
async def list_sessions():
    """List all active sessions.
    
    Validates that persisted sessions actually exist on disk.
    """
    store = get_session_store()
    active_sessions = store.list_sessions()
    
    # Get persisted sessions and validate they exist
    persisted_sessions = list_persisted_sessions()
    
    # Validate active sessions - remove any that don't exist on disk
    validated_active = []
    for session_id in active_sessions:
        session = store.get_session(session_id)
        if session:
            # Check if session file exists
            from app.services.session_persistence import load_session
            if load_session(session_id):
                validated_active.append(session_id)
            else:
                # Session file doesn't exist, remove from memory
                store.delete_session(session_id)
    
    # Validate persisted sessions - only return those that actually exist
    validated_persisted = []
    for session_info in persisted_sessions:
        session_id = session_info.get("session_id")
        if session_id:
            from app.services.session_persistence import load_session
            if load_session(session_id):
                validated_persisted.append(session_info)
            else:
                # Session file doesn't exist, try to delete it
                from app.services.session_persistence import delete_session
                delete_session(session_id)
                logger.info(f"Removed invalid session from list: {session_id}")
    
    return {
        "active_sessions": validated_active,
        "persisted_sessions": validated_persisted
    }


@router.get("/check-context-status")
async def check_context_status(context_path: str = Query(..., description="Path to context directory")):
    """Check if a context directory has been color corrected.
    
    Args:
        context_path: Path to context directory
        
    Returns:
        Dictionary with is_color_corrected boolean
    """
    try:
        import json
        
        ascota_dir = ensure_context_ascota_folder(context_path)
        status_file = ascota_dir / "context_status.json"
        
        if not status_file.exists():
            return {"is_color_corrected": False}
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check latest_status for color_correct_status
            if "latest_status" in data:
                latest = data["latest_status"]
                is_corrected = latest.get("color_correct_status", False)
                return {"is_color_corrected": bool(is_corrected)}
            
            # Fallback: check status_history
            if "status_history" in data and len(data["status_history"]) > 0:
                last_entry = data["status_history"][-1]
                is_corrected = last_entry.get("color_correct_status", False)
                return {"is_color_corrected": bool(is_corrected)}
            
            return {"is_color_corrected": False}
            
        except Exception as e:
            logger.warning(f"Failed to read context_status.json for {context_path}: {e}")
            return {"is_color_corrected": False}
            
    except Exception as e:
        logger.error(f"Error checking context status for {context_path}: {e}")
        return {"is_color_corrected": False}


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    """Get session information including options."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    # Try to restore from disk if not in memory
    if not session:
        session = store.restore_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "contexts": session.contexts,
        "options": {
            "image_source": session.options.image_source.value,
            "overwrite": session.options.overwrite,
            "custom_k": session.options.custom_k,
            "sensitivity": session.options.sensitivity,
            "preview_resolution": session.options.preview_resolution,
        }
    }


@router.post("/{session_id}/restore")
async def restore_session(session_id: str):
    """Restore a session from disk."""
    store = get_session_store()
    session = store.restore_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found on disk")
    
    return {
        "session_id": session.session_id,
        "message": "Session restored"
    }

