"""Reference images routes for managing reference images in sessions."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import logging
from typing import List
import tempfile
from pydantic import BaseModel

from app.services.session_store import get_session_store

logger = logging.getLogger(__name__)
router = APIRouter()

# Reference images storage directory
REFERENCE_IMAGES_DIR = Path(tempfile.gettempdir()) / "color_correct_reference_images"

# Preset reference images directory (in backend/app directory)
PRESET_REFERENCES_DIR = Path(__file__).parent.parent / "preset_references"


def ensure_reference_dir(session_id: str) -> Path:
    """Ensure the reference images directory for a session exists."""
    session_dir = REFERENCE_IMAGES_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_reference_image_path(session_id: str, image_id: str) -> Path:
    """Get the path to a reference image file."""
    session_dir = ensure_reference_dir(session_id)
    # Find the image file by ID (stored in filename or metadata)
    for file in session_dir.glob("*"):
        if file.stem == image_id or file.name.startswith(f"{image_id}_"):
            return file
    raise FileNotFoundError(f"Reference image {image_id} not found")


@router.get("/{session_id}/reference-images")
async def get_reference_images(session_id: str):
    """Get all reference images for a session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_dir = ensure_reference_dir(session_id)
    
    # Get all image files
    images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
    
    for file in session_dir.iterdir():
        if file.is_file() and file.suffix in image_extensions:
            image_id = file.stem.split('_')[0] if '_' in file.stem else file.stem
            images.append({
                "id": image_id,
                "name": file.name,
                "url": f"/api/sessions/{session_id}/reference-images/{image_id}/file",
                "path": str(file)
            })
    
    return {"images": images}


@router.post("/{session_id}/reference-images")
async def upload_reference_images(
    session_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload reference images for a session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_dir = ensure_reference_dir(session_id)
    uploaded_images = []
    
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"Skipping non-image file: {file.filename}")
            continue
        
        # Generate unique ID for the image
        image_id = str(uuid.uuid4())
        
        # Save file with original extension
        file_ext = Path(file.filename).suffix if file.filename else '.jpg'
        saved_filename = f"{image_id}_{file.filename or 'image'}{file_ext}"
        file_path = session_dir / saved_filename
        
        try:
            # Save file
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            
            uploaded_images.append({
                "id": image_id,
                "name": file.filename or "image",
                "url": f"/api/sessions/{session_id}/reference-images/{image_id}/file",
                "path": str(file_path)
            })
            
            logger.info(f"Uploaded reference image {image_id} for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save reference image {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
    
    # Return all images (including newly uploaded ones)
    all_images = await get_reference_images(session_id)
    return all_images


@router.get("/{session_id}/reference-images/{image_id}/file")
async def get_reference_image_file(session_id: str, image_id: str):
    """Get a reference image file."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        file_path = get_reference_image_path(session_id, image_id)
        return FileResponse(
            file_path,
            media_type="image/jpeg"  # Default, could be improved to detect actual type
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Reference image not found")


@router.delete("/{session_id}/reference-images/{image_id}")
async def delete_reference_image(session_id: str, image_id: str):
    """Delete a reference image."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        file_path = get_reference_image_path(session_id, image_id)
        file_path.unlink()
        logger.info(f"Deleted reference image {image_id} for session {session_id}")
        return {"message": "Reference image deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Reference image not found")
    except Exception as e:
        logger.error(f"Failed to delete reference image {image_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")


async def list_preset_references():
    """List all available preset reference folders."""
    try:
        # Ensure directory exists
        PRESET_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get all subdirectories (each represents a preset)
        presets = []
        if PRESET_REFERENCES_DIR.exists():
            for item in PRESET_REFERENCES_DIR.iterdir():
                if item.is_dir():
                    # Count images in the directory
                    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
                    image_count = sum(1 for f in item.iterdir() if f.is_file() and f.suffix in image_extensions)
                    presets.append({
                        "name": item.name,
                        "path": str(item),
                        "image_count": image_count
                    })
        
        return {"presets": sorted(presets, key=lambda x: x["name"])}
    except Exception as e:
        logger.error(f"Failed to list preset references: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list presets: {str(e)}")


class LoadPresetRequest(BaseModel):
    preset_name: str


@router.post("/{session_id}/reference-images/load-preset")
async def load_preset_references(session_id: str, request: LoadPresetRequest):
    """Load all images from a preset reference folder into the session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get preset directory
        preset_dir = PRESET_REFERENCES_DIR / request.preset_name
        
        if not preset_dir.exists() or not preset_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Preset '{request.preset_name}' not found")
        
        # Get session directory
        session_dir = ensure_reference_dir(session_id)
        
        # Image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}
        
        # Copy all images from preset to session
        loaded_images = []
        for file in preset_dir.iterdir():
            if file.is_file() and file.suffix in image_extensions:
                # Generate unique ID for the image
                image_id = str(uuid.uuid4())
                
                # Copy file to session directory
                new_filename = f"{image_id}_{file.name}"
                new_file_path = session_dir / new_filename
                
                shutil.copy2(file, new_file_path)
                
                loaded_images.append({
                    "id": image_id,
                    "name": file.name,
                    "url": f"/api/sessions/{session_id}/reference-images/{image_id}/file",
                    "path": str(new_file_path)
                })
                
                logger.info(f"Loaded preset image {file.name} from {request.preset_name} to session {session_id}")
        
        # Return all images (including newly loaded ones)
        all_images = await get_reference_images(session_id)
        return {
            "message": f"Loaded {len(loaded_images)} images from preset '{request.preset_name}'",
            "loaded_count": len(loaded_images),
            "images": all_images["images"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load preset references: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load preset: {str(e)}")

