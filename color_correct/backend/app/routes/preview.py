"""Preview routes for corrected images."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import os
import logging
from PIL import Image
import numpy as np
from pathlib import Path

from app.services.session_store import get_session_store
from app.services.correction import load_image_as_float, apply_correction_params
from app.services.models import CorrectionParams
from app.services.preview_cache import get_preview_source, save_to_cache
from app.services.clustering import convert_raw_to_jpg, is_raw_file

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
    individual_temperature: float = None,
    individual_tint: float = None,
    individual_exposure: float = None,
    individual_contrast: float = None,
    individual_saturation: float = None,
    individual_red_gain: float = None,
    individual_green_gain: float = None,
    individual_blue_gain: float = None,
    show_overall: bool = True,
    show_individual: bool = True,
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
    
    # Determine correction params - composite overall (cluster) and individual
    overall_params = CorrectionParams()
    individual_params = None
    
    # Start with cluster corrections if available (overall layer)
    if cluster_id:
        cluster = session.get_cluster(cluster_id)
        if cluster and cluster.correction_params:
            overall_params = cluster.correction_params
    
    # Get individual corrections for this image
    individual_params = session.get_individual_correction(image_id)
    
    # Override with query parameters if provided (for real-time preview)
    # Override overall params
    if temperature is not None:
        overall_params.temperature = temperature
    if tint is not None:
        overall_params.tint = tint
    if exposure is not None:
        overall_params.exposure = exposure
    if contrast is not None:
        overall_params.contrast = contrast
    if saturation is not None:
        overall_params.saturation = saturation
    if red_gain is not None:
        overall_params.red_gain = red_gain
    if green_gain is not None:
        overall_params.green_gain = green_gain
    if blue_gain is not None:
        overall_params.blue_gain = blue_gain
    
    # Override individual params if provided (for real-time preview of individual corrections)
    if individual_temperature is not None or individual_tint is not None or individual_exposure is not None or \
       individual_contrast is not None or individual_saturation is not None or \
       individual_red_gain is not None or individual_green_gain is not None or individual_blue_gain is not None:
        # Create or update individual params
        if individual_params is None:
            individual_params = CorrectionParams()
        
        if individual_temperature is not None:
            individual_params.temperature = individual_temperature
        if individual_tint is not None:
            individual_params.tint = individual_tint
        if individual_exposure is not None:
            individual_params.exposure = individual_exposure
        if individual_contrast is not None:
            individual_params.contrast = individual_contrast
        if individual_saturation is not None:
            individual_params.saturation = individual_saturation
        if individual_red_gain is not None:
            individual_params.red_gain = individual_red_gain
        if individual_green_gain is not None:
            individual_params.green_gain = individual_green_gain
        if individual_blue_gain is not None:
            individual_params.blue_gain = individual_blue_gain
    
    # Get preview resolution from session options
    preview_resolution = session.options.preview_resolution if hasattr(session.options, 'preview_resolution') else 1500
    
    # Get best preview source (cached JPG, existing proxy, or RAW)
    source_path, is_fast_source = get_preview_source(image, preview_resolution)
    
    if not source_path:
        raise HTTPException(status_code=404, detail="No image source available")
    
    try:
        # If using cached/proxy, load directly
        if is_fast_source:
            logger.debug(f"Using fast source for preview: {source_path}")
            img_float = load_image_as_float(source_path)
            if img_float is None:
                raise HTTPException(status_code=500, detail="Failed to load cached/proxy image")
        else:
            # Must convert RAW and cache it
            logger.debug(f"Converting RAW to JPG and caching: {source_path}")
            import tempfile
            import rawpy
            
            # Create temp file for conversion
            temp_jpg = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_jpg_path = temp_jpg.name
            temp_jpg.close()
            
            try:
                # Convert RAW to JPG
                if convert_raw_to_jpg(source_path, temp_jpg_path, max_width=preview_resolution):
                    # Load the converted JPG
                    img_float = load_image_as_float(temp_jpg_path)
                    if img_float is None:
                        raise HTTPException(status_code=500, detail="Failed to load converted image")
                    
                    # Cache it for future use
                    with open(temp_jpg_path, 'rb') as f:
                        jpg_data = f.read()
                    save_to_cache(image.raw_path, preview_resolution, jpg_data)
                else:
                    raise HTTPException(status_code=500, detail="Failed to convert RAW file")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_jpg_path)
                except:
                    pass
        
        # Apply corrections: overall first, then individual on top (respecting visibility)
        if show_overall:
            corrected = apply_correction_params(img_float, overall_params)
        else:
            corrected = img_float.copy()
        
        if show_individual and individual_params:
            corrected = apply_correction_params(corrected, individual_params)
        
        # Convert to PIL Image
        img_uint8 = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='RGB')
        
        # Resize for preview if needed
        if max_size < 1500:
            pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
        
    except ImportError:
        raise HTTPException(status_code=500, detail="rawpy not available - cannot process RAW files")
    except Exception as e:
        logger.error(f"Failed to generate preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

