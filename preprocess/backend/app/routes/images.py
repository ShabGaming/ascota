"""Image serving routes."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import logging

from app.services.session_store import get_session_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{session_id}/image/{image_id}")
async def serve_image(
    session_id: str,
    image_id: str,
    size: str = Query("3000", description="Image size: 3000, 1500, or 450")
):
    """Serve an image file from a session.
    
    Args:
        session_id: Session ID
        image_id: Image ID
        size: Image size variant (3000, 1500, or 450)
        
    Returns:
        Image file response
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Determine which image path to use based on size
    image_path = None
    if size == "3000" and image.proxy_3000:
        image_path = image.proxy_3000
    elif size == "1500" and image.proxy_1500:
        image_path = image.proxy_1500
    elif size == "450" and image.proxy_450:
        image_path = image.proxy_450
    elif image.proxy_3000:
        # Fallback to 3000 if requested size not available
        image_path = image.proxy_3000
    else:
        # Final fallback to primary path
        image_path = image.primary_path
    
    if not image_path:
        raise HTTPException(status_code=404, detail="Image file not found")
    
    path = Path(image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image file does not exist: {image_path}")
    
    # Determine media type
    media_type = "image/jpeg"
    if path.suffix.lower() in ['.png', '.PNG']:
        media_type = "image/png"
    
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name
    )


@router.get("/{session_id}/card_crop/{image_id}")
async def serve_card_crop(
    session_id: str,
    image_id: str
):
    """Serve the cropped card image for 8-hybrid card editing.
    
    Args:
        session_id: Session ID
        image_id: Image ID
        
    Returns:
        Card crop image file response
    """
    from app.services.scale_calc import crop_card_from_image
    
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get Stage 1 data to find the 8-hybrid card
    stage1_data = session.stage1_results.get(image_id, {})
    cards_data = stage1_data.get("cards", [])
    
    # Find 8-hybrid card
    hybrid_card = None
    for card in cards_data:
        if card.get("card_type") == "8_hybrid_card":
            hybrid_card = card
            break
    
    if not hybrid_card:
        raise HTTPException(status_code=400, detail="No 8-hybrid card found for this image")
    
    # Crop card from image
    image_path = image.proxy_3000
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    card_crop = crop_card_from_image(image_path, hybrid_card.get("coordinates"))
    if card_crop is None:
        raise HTTPException(status_code=500, detail="Failed to crop card from image")
    
    # Save to temporary file and serve
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        card_crop.save(tmp_file.name, format='JPEG')
        tmp_path = Path(tmp_file.name)
    
    return FileResponse(
        tmp_path,
        media_type="image/jpeg",
        filename="card_crop.jpg"
    )


@router.get("/{session_id}/masks/{image_id}")
async def serve_mask(
    session_id: str,
    image_id: str,
    path: str = Query(..., description="Relative path to mask file from .ascota folder")
):
    """Serve a mask file from a session.
    
    Args:
        session_id: Session ID
        image_id: Image ID
        path: Relative path to mask (e.g., "masks/1_mask.png")
        
    Returns:
        Mask file response
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Build find path
    context_path = Path(image.context_id)
    find_path = context_path / "finds" / "individual" / image.find_number
    ascota_dir = find_path / ".ascota"
    
    # Resolve mask path
    mask_path = ascota_dir / path
    
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail=f"Mask file does not exist: {mask_path}")
    
    # Ensure path is within .ascota directory (security check)
    try:
        mask_path.resolve().relative_to(ascota_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid mask path")
    
    return FileResponse(
        mask_path,
        media_type="image/png",
        filename=mask_path.name
    )

