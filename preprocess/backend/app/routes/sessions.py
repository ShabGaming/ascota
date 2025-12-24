"""Session management routes."""

from fastapi import APIRouter, HTTPException
from typing import List
import logging

from app.services.models import CreateSessionRequest, CreateSessionResponse, ImageItem
from app.services.session_store import get_session_store
from typing import Dict

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new preprocess session."""
    try:
        store = get_session_store()
        session_id = store.create_session(request.contexts)
        logger.info(f"Created session {session_id}")
        return CreateSessionResponse(session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    store = get_session_store()
    deleted = store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


@router.get("")
async def list_sessions():
    """List all active sessions."""
    store = get_session_store()
    session_ids = store.list_sessions()
    return {"session_ids": session_ids}


@router.get("/{session_id}/images")
async def get_session_images(session_id: str):
    """Get all images in a session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Convert ImageItem objects to dictionaries
    images_dict = {
        img_id: img.dict()
        for img_id, img in session.images.items()
    }
    
    return {
        "images": images_dict,
        "total": len(images_dict)
    }

