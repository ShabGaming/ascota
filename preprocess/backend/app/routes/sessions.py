"""Session management routes."""

from fastapi import APIRouter, HTTPException, Query
from typing import List
import logging
from pathlib import Path

from app.services.models import CreateSessionRequest, CreateSessionResponse, ImageItem
from app.services.session_store import get_session_store
from app.services.metadata import ensure_context_ascota_folder
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


@router.get("/check-context-status")
async def check_context_status(context_path: str = Query(..., description="Path to context directory")):
    """Check if a context directory has been preprocessed.
    
    Args:
        context_path: Path to context directory
        
    Returns:
        Dictionary with is_preprocessed boolean
    """
    try:
        import json
        
        ascota_dir = ensure_context_ascota_folder(context_path)
        status_file = ascota_dir / "context_status.json"
        
        if not status_file.exists():
            return {"is_preprocessed": False}
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check latest_status for preprocess_status
            if "latest_status" in data:
                latest = data["latest_status"]
                is_preprocessed = latest.get("preprocess_status", False)
                return {"is_preprocessed": bool(is_preprocessed)}
            
            # Fallback: check status_history
            if "status_history" in data and len(data["status_history"]) > 0:
                last_entry = data["status_history"][-1]
                is_preprocessed = last_entry.get("preprocess_status", False)
                return {"is_preprocessed": bool(is_preprocessed)}
            
            return {"is_preprocessed": False}
            
        except Exception as e:
            logger.warning(f"Failed to read context_status.json for {context_path}: {e}")
            return {"is_preprocessed": False}
            
    except Exception as e:
        logger.error(f"Error checking context status for {context_path}: {e}")
        return {"is_preprocessed": False}

