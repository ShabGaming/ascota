"""Session management routes."""

from fastapi import APIRouter, HTTPException
from typing import List
import logging

from app.services.models import CreateSessionRequest, CreateSessionResponse, JobStatusResponse
from app.services.session_store import get_session_store

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
    """Delete a session."""
    store = get_session_store()
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


@router.get("")
async def list_sessions():
    """List all active sessions."""
    store = get_session_store()
    return {"sessions": store.list_sessions()}


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    """Get session information including options."""
    store = get_session_store()
    session = store.get_session(session_id)
    
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
        }
    }

