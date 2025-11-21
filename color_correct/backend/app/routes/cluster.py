"""Clustering routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Optional
from datetime import datetime
import logging
import asyncio

from app.services.models import ClusterResponse, MoveImageRequest, SetCorrectionRequest, JobStatusResponse
from app.services.session_store import get_session_store
from app.services.scanner import scan_all_contexts
from app.services.clustering import cluster_images
from app.services.models import JobStatus
from app.services.correction import estimate_auto_correction

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_clustering_job(session_id: str, job_id: str, new_sensitivity: float = None):
    """Background task to run clustering."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        logger.error(f"Session {session_id} not found")
        return
    
    job = session.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return
    
    try:
        job.status = JobStatus.RUNNING
        job.progress = 0.1
        job.message = "Scanning contexts..."
        
        # Update sensitivity if provided
        if new_sensitivity is not None:
            session.options.sensitivity = new_sensitivity
        
        # Scan all contexts
        images = scan_all_contexts(session.contexts, session.options.image_source.value)
        
        if not images:
            job.status = JobStatus.FAILED
            job.error = "No images found in contexts"
            return
        
        job.progress = 0.4
        job.message = f"Found {len(images)} images, clustering..."
        
        # Run clustering with preview resolution
        preview_resolution = session.options.preview_resolution if hasattr(session.options, 'preview_resolution') else 1500
        clusters = cluster_images(
            images,
            k=session.options.custom_k,
            sensitivity=session.options.sensitivity,
            preview_resolution=preview_resolution
        )
        
        # Update session
        session.images = images
        session.clusters = clusters
        
        # Persist session
        store.mark_session_updated(session_id)
        
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.message = f"Clustering complete: {len(clusters)} clusters"
        job.result = {
            "clusters": [c.dict() for c in clusters],
            "total_images": len(images)
        }
        
        logger.info(f"Clustering job {job_id} completed: {len(clusters)} clusters, {len(images)} images")
        
    except Exception as e:
        logger.error(f"Clustering job {job_id} failed: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)


@router.post("/{session_id}/cluster")
async def cluster_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    sensitivity: Optional[float] = Query(None, description="Optional sensitivity value for reclustering")
):
    """Start clustering for a session.
    
    Args:
        session_id: Session ID
        sensitivity: Optional new sensitivity value for reclustering
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save current clusters for undo if reclustering
    if sensitivity is not None and session.clusters:
        session.save_clusters_for_undo()
    
    # Create a job
    job = session.create_job()
    
    # Start background task
    background_tasks.add_task(_run_clustering_job, session_id, job.job_id, sensitivity)
    
    return {"job_id": job.job_id, "message": "Clustering started"}


@router.post("/{session_id}/clusters/undo")
async def undo_clusters(session_id: str):
    """Undo the last reclustering operation."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = session.undo_clusters()
    
    if not success:
        raise HTTPException(status_code=400, detail="No previous cluster state to undo")
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Clusters restored"}


@router.get("/{session_id}/discovered-images")
async def get_discovered_images(session_id: str):
    """Get discovered images before clustering (for debug mode)."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Scan contexts to get images
    images = scan_all_contexts(session.contexts, session.options.image_source.value)
    
    # Convert to serializable format
    images_dict = {img_id: img.dict() for img_id, img in images.items()}
    
    return {
        "images": images_dict,
        "total": len(images),
        "image_source": session.options.image_source.value
    }


@router.post("/{session_id}/clusters")
async def create_cluster(
    session_id: str,
    image_id: Optional[str] = Query(None, description="Optional image ID to add to the new cluster")
):
    """Create a new empty cluster.
    
    Args:
        session_id: Session ID
        image_id: Optional image ID to add to the new cluster
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create new cluster
    cluster = session.create_cluster(image_id)
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"cluster_id": cluster.id, "message": "Cluster created"}


@router.delete("/{session_id}/clusters/{cluster_id}")
async def delete_cluster(session_id: str, cluster_id: str):
    """Delete a cluster from a session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cluster = session.get_cluster(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    # Only allow deletion of empty clusters
    if cluster.image_ids:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete cluster with images. Remove all images first."
        )
    
    success = session.delete_cluster(cluster_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Cluster deleted"}


@router.get("/{session_id}/clusters", response_model=ClusterResponse)
async def get_clusters(session_id: str):
    """Get clusters and images for a session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Convert images dict to serializable format
    images_dict = {img_id: img.dict() for img_id, img in session.images.items()}
    
    return ClusterResponse(
        clusters=[c.dict() for c in session.clusters],
        images=images_dict
    )


@router.patch("/{session_id}/clusters/move")
async def move_image(session_id: str, request: MoveImageRequest):
    """Move an image to a different cluster."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = session.move_image(request.image_id, request.target_cluster_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to move image")
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Image moved successfully"}


@router.post("/{session_id}/clusters/{cluster_id}/set-correction")
async def set_cluster_correction(
    session_id: str,
    cluster_id: str,
    request: SetCorrectionRequest
):
    """Set correction parameters for a cluster."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = session.set_correction(cluster_id, request.params)
    
    if not success:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Correction parameters set"}


@router.post("/{session_id}/clusters/{cluster_id}/auto-correct")
async def auto_correct_cluster(
    session_id: str,
    cluster_id: str,
    image_id: Optional[str] = Query(None, description="Optional image ID to use for white balance calculation")
):
    """Compute auto-correction (white balance) for a cluster.
    
    Args:
        session_id: Session ID
        cluster_id: Cluster ID
        image_id: Optional specific image ID to use for white balance calculation.
                  If not provided, uses the first image in the cluster.
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cluster = session.get_cluster(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    if not cluster.image_ids:
        raise HTTPException(status_code=400, detail="Cluster has no images")
    
    # Use specified image or first image in cluster
    target_image_id = image_id if image_id and image_id in cluster.image_ids else cluster.image_ids[0]
    image = session.get_image(target_image_id)
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Estimate correction from primary path (white balance only - RGB gains)
    params = estimate_auto_correction(image.primary_path)
    
    # Note: We don't apply to cluster here - just return the params
    # The frontend will merge these into the current correction params
    
    return {"message": "White balance calculated", "params": params.dict()}


@router.post("/{session_id}/clusters/{cluster_id}/reset")
async def reset_cluster_correction(session_id: str, cluster_id: str):
    """Reset correction parameters for a cluster to defaults."""
    from app.services.models import CorrectionParams
    
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cluster = session.get_cluster(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    # Reset to default parameters
    default_params = CorrectionParams()
    session.set_correction(cluster_id, default_params)
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Correction reset to defaults", "params": default_params.dict()}


@router.post("/{session_id}/images/{image_id}/set-individual-correction")
async def set_individual_correction(
    session_id: str,
    image_id: str,
    request: SetCorrectionRequest
):
    """Set individual correction parameters for an image."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    success = session.set_individual_correction(image_id, request.params)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to set individual correction")
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Individual correction parameters set"}


@router.post("/{session_id}/clusters/{cluster_id}/reset-individual")
async def reset_individual_corrections(session_id: str, cluster_id: str):
    """Reset individual correction parameters for all images in a cluster."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cluster = session.get_cluster(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    # Remove individual corrections for all images in cluster
    if hasattr(session, '_individual_corrections'):
        for image_id in cluster.image_ids:
            if image_id in session._individual_corrections:
                del session._individual_corrections[image_id]
        session.updated_at = datetime.now()
    
    # Persist session
    store.mark_session_updated(session_id)
    
    return {"message": "Individual corrections reset for cluster"}

