"""In-memory session store for color correction sessions."""

import uuid
from typing import Dict, Optional, List, Any
from datetime import datetime
import threading
import logging

from app.services.models import (
    SessionOptions, Cluster, ImageItem, CorrectionParams,
    JobStatus, JobStatusResponse
)

logger = logging.getLogger(__name__)


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


class SessionStore:
    """In-memory store for all sessions."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
    
    def create_session(self, contexts: List[str], options: SessionOptions) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            session = Session(session_id, contexts, options)
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id} with {len(contexts)} contexts")
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

