"""In-memory session store for color correction sessions."""

import uuid
from typing import Dict, Optional, List, Any
from datetime import datetime
import threading
import logging
from functools import wraps
import time

from app.services.models import (
    SessionOptions, Cluster, ImageItem, CorrectionParams,
    JobStatus, JobStatusResponse
)
from app.services.session_persistence import save_session as persist_session, load_session as load_persisted_session

logger = logging.getLogger(__name__)

# Debounce delay for persistence (seconds)
PERSIST_DEBOUNCE = 2.0


class Job:
    """Represents a background job."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.progress = 0.0
        self.message: Optional[str] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_response(self) -> JobStatusResponse:
        """Convert to response model."""
        return JobStatusResponse(
            status=self.status,
            progress=self.progress,
            message=self.message,
            result=self.result,
            error=self.error
        )


class Session:
    """Represents a color correction session."""
    
    def __init__(self, session_id: str, contexts: List[str], options: SessionOptions):
        self.session_id = session_id
        self.contexts = contexts
        self.options = options
        self.clusters: List[Cluster] = []
        self.images: Dict[str, ImageItem] = {}
        self.jobs: Dict[str, Job] = {}
        self.previous_clusters: Optional[List[Cluster]] = None  # For undo
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def get_image(self, image_id: str) -> Optional[ImageItem]:
        """Get an image by ID."""
        return self.images.get(image_id)
    
    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get a cluster by ID."""
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None
    
    def move_image(self, image_id: str, target_cluster_id: str) -> bool:
        """Move an image from its current cluster to a target cluster."""
        # Remove from current cluster
        for cluster in self.clusters:
            if image_id in cluster.image_ids:
                cluster.image_ids.remove(image_id)
                break
        
        # Add to target cluster
        target_cluster = self.get_cluster(target_cluster_id)
        if target_cluster:
            target_cluster.image_ids.append(image_id)
            self.updated_at = datetime.now()
            return True
        return False
    
    def set_correction(self, cluster_id: str, params: CorrectionParams) -> bool:
        """Set correction parameters for a cluster."""
        cluster = self.get_cluster(cluster_id)
        if cluster:
            cluster.correction_params = params
            self.updated_at = datetime.now()
            return True
        return False
    
    def set_individual_correction(self, image_id: str, params: CorrectionParams) -> bool:
        """Set individual correction parameters for an image.
        
        Args:
            image_id: Image ID
            params: Correction parameters
            
        Returns:
            True if successful, False if image not found
        """
        image = self.get_image(image_id)
        if image:
            # Store individual correction in image metadata
            # We'll need to extend ImageItem model to support this
            # For now, we'll store it in a separate dict
            if not hasattr(self, '_individual_corrections'):
                self._individual_corrections = {}
            self._individual_corrections[image_id] = params
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_individual_correction(self, image_id: str) -> Optional[CorrectionParams]:
        """Get individual correction parameters for an image.
        
        Args:
            image_id: Image ID
            
        Returns:
            Correction parameters or None
        """
        if hasattr(self, '_individual_corrections'):
            return self._individual_corrections.get(image_id)
        return None
    
    def create_job(self, job_id: Optional[str] = None) -> Job:
        """Create a new job for this session."""
        if job_id is None:
            job_id = str(uuid.uuid4())
        job = Job(job_id)
        self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def save_clusters_for_undo(self):
        """Save current clusters state for undo operation."""
        # Deep copy clusters for undo
        import copy
        self.previous_clusters = copy.deepcopy(self.clusters)
    
    def undo_clusters(self) -> bool:
        """Restore previous clusters state.
        
        Returns:
            True if undo was successful, False if no previous state
        """
        if self.previous_clusters is None:
            return False
        
        self.clusters = self.previous_clusters
        self.previous_clusters = None
        self.updated_at = datetime.now()
        return True
    
    def delete_cluster(self, cluster_id: str) -> bool:
        """Delete a cluster from the session.
        
        Args:
            cluster_id: ID of the cluster to delete
            
        Returns:
            True if cluster was deleted, False if not found
        """
        for i, cluster in enumerate(self.clusters):
            if cluster.id == cluster_id:
                self.clusters.pop(i)
                self.updated_at = datetime.now()
                return True
        return False
    
    def create_cluster(self, image_id: Optional[str] = None) -> Cluster:
        """Create a new empty cluster.
        
        Args:
            image_id: Optional image ID to add to the new cluster
            
        Returns:
            The newly created Cluster
        """
        cluster_id = str(uuid.uuid4())
        image_ids = [image_id] if image_id else []
        
        cluster = Cluster(
            id=cluster_id,
            image_ids=image_ids,
            correction_params=None
        )
        
        self.clusters.append(cluster)
        self.updated_at = datetime.now()
        
        return cluster
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for persistence.
        
        Returns:
            Dictionary representation of the session
        """
        result = {
            "session_id": self.session_id,
            "contexts": self.contexts,
            "options": {
                "image_source": self.options.image_source.value,
                "overwrite": self.options.overwrite,
                "custom_k": self.options.custom_k,
                "sensitivity": self.options.sensitivity,
                "preview_resolution": self.options.preview_resolution,
            },
            "clusters": [c.dict() for c in self.clusters],
            "images": {img_id: img.dict() for img_id, img in self.images.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        
        # Include individual corrections if they exist
        if hasattr(self, '_individual_corrections'):
            result["individual_corrections"] = {
                img_id: params.dict()
                for img_id, params in self._individual_corrections.items()
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create a session from a dictionary.
        
        Args:
            data: Dictionary representation of the session
            
        Returns:
            Session instance
        """
        from app.services.models import ImageSourceMode
        
        session_id = data["session_id"]
        contexts = data["contexts"]
        
        # Reconstruct options
        opts_data = data["options"]
        options = SessionOptions(
            image_source=ImageSourceMode(opts_data["image_source"]),
            overwrite=opts_data["overwrite"],
            custom_k=opts_data.get("custom_k"),
            sensitivity=opts_data["sensitivity"],
            preview_resolution=opts_data.get("preview_resolution", 1500),
        )
        
        session = cls(session_id, contexts, options)
        
        # Restore clusters
        session.clusters = [Cluster(**c) for c in data.get("clusters", [])]
        
        # Restore images
        session.images = {
            img_id: ImageItem(**img_data)
            for img_id, img_data in data.get("images", {}).items()
        }
        
        # Restore individual corrections if they exist
        if "individual_corrections" in data:
            session._individual_corrections = {
                img_id: CorrectionParams(**params_data)
                for img_id, params_data in data["individual_corrections"].items()
            }
        
        # Restore timestamps
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
        self._persist_timers: Dict[str, Any] = {}  # For debouncing persistence
    
    def _schedule_persist(self, session_id: str):
        """Schedule a debounced persistence for a session."""
        import threading
        
        # Cancel existing timer if any
        if session_id in self._persist_timers:
            self._persist_timers[session_id].cancel()
        
        def persist():
            session = self.get_session(session_id)
            if session:
                persist_session(session_id, session.to_dict())
            if session_id in self._persist_timers:
                del self._persist_timers[session_id]
        
        timer = threading.Timer(PERSIST_DEBOUNCE, persist)
        timer.start()
        self._persist_timers[session_id] = timer
    
    def create_session(self, contexts: List[str], options: SessionOptions) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            session = Session(session_id, contexts, options)
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id} with {len(contexts)} contexts")
            # Persist immediately on creation
            persist_session(session_id, session.to_dict())
        return session_id
    
    def restore_session(self, session_id: str) -> Optional[Session]:
        """Restore a session from disk.
        
        Args:
            session_id: Session ID to restore
            
        Returns:
            Restored Session or None if not found
        """
        data = load_persisted_session(session_id)
        if not data:
            return None
        
        try:
            session = Session.from_dict(data)
            with self._lock:
                self._sessions[session_id] = session
            logger.info(f"Restored session {session_id} from disk")
            return session
        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}", exc_info=True)
            return None
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        from app.services.session_persistence import delete_session as delete_persisted_session
        
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                # Cancel any pending persistence
                if session_id in self._persist_timers:
                    self._persist_timers[session_id].cancel()
                    del self._persist_timers[session_id]
                # Delete from disk
                delete_persisted_session(session_id)
                logger.info(f"Deleted session {session_id}")
                return True
        return False
    
    def mark_session_updated(self, session_id: str):
        """Mark a session as updated and schedule persistence."""
        if session_id in self._sessions:
            self._sessions[session_id].updated_at = datetime.now()
            self._schedule_persist(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        with self._lock:
            return list(self._sessions.keys())


# Global session store instance
_store = SessionStore()


def get_session_store() -> SessionStore:
    """Get the global session store instance."""
    return _store

