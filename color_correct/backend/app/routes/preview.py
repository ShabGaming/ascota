"""Preview routes for corrected images."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import logging
from PIL import Image
import numpy as np

from app.services.session_store import get_session_store
from app.services.correction import load_image_as_float, apply_correction_params
from app.services.models import CorrectionParams

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{session_id}/preview")
async def get_preview(
    session_id: str,
    image_id: str,
    cluster_id: str = None,
    max_size: int = 800,
    temperature: float = None,
    tint: float = None,
    exposure: float = None,
    contrast: float = None,
    saturation: float = None,
    red_gain: float = None,
    green_gain: float = None,
    blue_gain: float = None,
):
    """Get a preview of an image with corrections applied.
    
    Args:
        session_id: Session ID
        image_id: Image ID
        cluster_id: Optional cluster ID to apply cluster corrections
        max_size: Maximum dimension for preview (default 800px)
        temperature, tint, exposure, contrast, saturation, red_gain, green_gain, blue_gain:
            Optional override parameters for real-time preview
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Determine correction params
    params = CorrectionParams()
    
    # Start with cluster corrections if available
    if cluster_id:
        cluster = session.get_cluster(cluster_id)
        if cluster and cluster.correction_params:
            params = cluster.correction_params
    
    # Override with query parameters if provided (for real-time preview)
    if temperature is not None:
        params.temperature = temperature
    if tint is not None:
        params.tint = tint
    if exposure is not None:
        params.exposure = exposure
    if contrast is not None:
        params.contrast = contrast
    if saturation is not None:
        params.saturation = saturation
    if red_gain is not None:
        params.red_gain = red_gain
    if green_gain is not None:
        params.green_gain = green_gain
    if blue_gain is not None:
        params.blue_gain = blue_gain
    
    # Load image (prefer proxy for speed)
    image_path = image.proxy_1500 or image.proxy_3000 or image.proxy_450 or image.primary_path
    
    if not image_path:
        raise HTTPException(status_code=404, detail="No image path available")
    
    try:
        # Load as float
        img_float = load_image_as_float(image_path)
        
        if img_float is None:
            raise HTTPException(status_code=500, detail="Failed to load image")
        
        # Apply corrections
        corrected = apply_correction_params(img_float, params)
        
        # Convert to PIL Image
        img_uint8 = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='RGB')
        
        # Resize for preview
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Failed to generate preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

