"""In-memory session store for preprocess sessions."""

import uuid
from typing import Dict, Optional, List
from datetime import datetime
import threading
import logging

from app.services.models import ImageItem
from app.services.scanner import scan_all_contexts

logger = logging.getLogger(__name__)


class Session:
    """Represents a preprocess session."""
    
    def __init__(self, session_id: str, contexts: List[str]):
        self.session_id = session_id
        self.contexts = contexts
        self.images: Dict[str, ImageItem] = {}
        self.stage1_results: Dict[str, Dict] = {}  # image_id -> card detection results
        self.stage2_results: Dict[str, Dict] = {}  # image_id -> mask results
        self.stage3_results: Dict[str, Dict] = {}  # image_id -> scale results
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def get_image(self, image_id: str) -> Optional[ImageItem]:
        """Get an image by ID."""
        return self.images.get(image_id)
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "contexts": self.contexts,
            "images": {img_id: img.dict() for img_id, img in self.images.items()},
            "stage1_results": self.stage1_results,
            "stage2_results": self.stage2_results,
            "stage3_results": self.stage3_results,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Session':
        """Create a session from a dictionary."""
        session = cls(data["session_id"], data["contexts"])
        session.images = {
            img_id: ImageItem(**img_data)
            for img_id, img_data in data.get("images", {}).items()
        }
        session.stage1_results = data.get("stage1_results", {})
        session.stage2_results = data.get("stage2_results", {})
        session.stage3_results = data.get("stage3_results", {})
        if "created_at" in data:
            session.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            session.updated_at = datetime.fromisoformat(data["updated_at"])
        return session


class SessionStore:
    """In-memory store for all sessions."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
    
    def create_session(self, contexts: List[str]) -> str:
        """Create a new session and scan images."""
        session_id = str(uuid.uuid4())
        with self._lock:
            session = Session(session_id, contexts)
            # Scan images from all contexts
            images = scan_all_contexts(contexts)
            session.images = images
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id} with {len(contexts)} contexts, {len(images)} images")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        with self._lock:
            return list(self._sessions.keys())


# Global session store instance
_store = SessionStore()


def get_session_store() -> SessionStore:
    """Get the global session store instance."""
    return _store

