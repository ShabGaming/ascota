"""Session persistence service for saving/loading sessions from disk."""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Session storage directory
SESSION_DIR = Path(tempfile.gettempdir()) / "color_correct_sessions"


def ensure_session_dir():
    """Ensure the session directory exists."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, session_data: Dict[str, Any]) -> bool:
    """Save a session to disk.
    
    Args:
        session_id: Session ID
        session_data: Session data dictionary (from session.to_dict() or similar)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_session_dir()
        
        session_file = SESSION_DIR / f"{session_id}.json"
        
        # Add metadata
        session_data["_metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Atomic write: write to temp file, then rename
        temp_file = session_file.with_suffix(".tmp")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        # Rename to final file
        temp_file.replace(session_file)
        
        logger.info(f"Saved session {session_id} to {session_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save session {session_id}: {e}", exc_info=True)
        return False


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session from disk.
    
    Args:
        session_id: Session ID
        
    Returns:
        Session data dictionary or None if not found
    """
    try:
        session_file = SESSION_DIR / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        logger.info(f"Loaded session {session_id} from {session_file}")
        return session_data
        
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}", exc_info=True)
        return None


def delete_session(session_id: str) -> bool:
    """Delete a session file from disk.
    
    Args:
        session_id: Session ID
        
    Returns:
        True if deleted, False if not found
    """
    try:
        session_file = SESSION_DIR / f"{session_id}.json"
        
        if not session_file.exists():
            return False
        
        session_file.unlink()
        logger.info(f"Deleted session {session_id} from {session_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
        return False


def list_sessions() -> List[Dict[str, Any]]:
    """List all saved sessions with metadata.
    
    Validates that session files actually exist and are readable.
    
    Returns:
        List of session metadata dictionaries
    """
    try:
        ensure_session_dir()
        
        sessions = []
        
        for session_file in SESSION_DIR.glob("*.json"):
            # Skip temp files
            if session_file.suffix == ".tmp":
                continue
                
            try:
                # Verify file exists and is readable
                if not session_file.exists():
                    logger.debug(f"Session file does not exist: {session_file}")
                    continue
                
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session_id = session_file.stem
                metadata = session_data.get("_metadata", {})
                
                # Extract basic info
                contexts = session_data.get("contexts", [])
                clusters = session_data.get("clusters", [])
                images = session_data.get("images", {})
                
                sessions.append({
                    "session_id": session_id,
                    "saved_at": metadata.get("saved_at", "Unknown"),
                    "context_count": len(contexts),
                    "cluster_count": len(clusters),
                    "image_count": len(images),
                    "contexts": contexts
                })
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in session file {session_file}: {e}")
                # Optionally delete corrupted files
                try:
                    session_file.unlink()
                    logger.info(f"Deleted corrupted session file: {session_file}")
                except:
                    pass
                continue
            except Exception as e:
                logger.warning(f"Failed to read session file {session_file}: {e}")
                continue
        
        # Sort by saved_at (newest first)
        sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        return []

