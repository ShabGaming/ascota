"""Export routes for rendering corrected images."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging

from app.services.session_store import get_session_store
from app.services.exporter import export_session_images
from app.services.models import ExportResponse, JobStatus, CorrectionParams
from app.services.ascota_storage import build_corrections_data, save_color_correct_json

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_export_job(session_id: str, job_id: str):
    """Background task to export images."""
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
        job.progress = 0.0
        job.message = "Starting export..."
        
        # Prepare clusters with params
        clusters_with_params = []
        for cluster in session.clusters:
            params = cluster.correction_params or CorrectionParams()
            clusters_with_params.append((cluster.image_ids, params))
        
        # Count total images
        total_images = sum(len(image_ids) for image_ids, _ in clusters_with_params)
        
        if total_images == 0:
            job.status = JobStatus.FAILED
            job.error = "No images to export"
            return
        
        # Progress callback for real-time updates
        def update_progress(current: int, total: int, message: str):
            if total > 0:
                job.progress = current / total
            job.message = message or f"Exporting {current}/{total} images..."
        
        # Run export with progress callback
        summary = export_session_images(
            session.images,
            clusters_with_params,
            session.options.image_source.value,
            session.options.overwrite,
            session,  # Pass session for individual corrections
            progress_callback=update_progress
        )
        
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.message = f"Export complete: {summary.total_files_written} files written"
        job.result = summary.dict()
        
        # Save corrections to .ascota folders (one per find)
        try:
            corrections_data_by_find = build_corrections_data(
                session.images,
                session.clusters,
                session
            )
            
            # Save to each find directory
            for find_path, corrections_data in corrections_data_by_find.items():
                save_color_correct_json(find_path, corrections_data)
            
            logger.info(f"Saved corrections to .ascota folders for {len(corrections_data_by_find)} finds")
        except Exception as e:
            logger.error(f"Failed to save corrections to .ascota folders: {e}", exc_info=True)
            # Don't fail the export if saving corrections fails
        
        # Append color correction status to context directories
        try:
            from app.services.ascota_storage import append_context_status
            
            # Get unique context paths from images
            context_paths = set()
            for image in session.images.values():
                if image.context_id:
                    context_paths.add(image.context_id)
            
            # Append status to each context directory
            for context_path in context_paths:
                append_context_status(context_path, {
                    "color_correct_status": True,
                    "export_summary": {
                        "total_images": summary.total_images,
                        "total_files_written": summary.total_files_written,
                        "new_files_count": summary.new_files_count,
                        "overwritten_count": summary.overwritten_count
                    }
                })
            
            logger.info(f"Appended color correction status to {len(context_paths)} context directories")
        except Exception as e:
            logger.error(f"Failed to append context status: {e}", exc_info=True)
            # Don't fail the export if saving context status fails
        
        logger.info(f"Export job {job_id} completed: {summary.total_files_written} files written")
        
    except Exception as e:
        logger.error(f"Export job {job_id} failed: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)


@router.post("/{session_id}/export", response_model=ExportResponse)
async def export_session(session_id: str, background_tasks: BackgroundTasks):
    """Export all corrected images for a session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.clusters:
        raise HTTPException(status_code=400, detail="No clusters to export")
    
    # Create export job
    job = session.create_job()
    
    # Start background task
    background_tasks.add_task(_run_export_job, session_id, job.job_id)
    
    return ExportResponse(job_id=job.job_id)

