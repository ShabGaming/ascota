"""Stage 2: Mask generation routes."""

from fastapi import APIRouter, HTTPException
from typing import Dict
import logging
from pathlib import Path

from app.services.session_store import get_session_store
from app.services.models import Stage2Results, MaskResult, UpdateMaskRequest, WandSelectRequest
from app.services.segmentation import generate_mask, base64_to_mask, mask_to_base64
from app.services.mobilesam import segment_from_point
from app.services.metadata import save_mask, load_preprocess_json, save_preprocess_json, load_stage2_data_for_image, load_mask
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{session_id}/stage2/generate", response_model=Stage2Results)
async def generate_masks(session_id: str):
    """Generate masks for all images using card coordinates from Stage 1."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Ensure Stage 1 results are loaded in session (load from .ascota if needed)
    if not session.stage1_results:
        # Try to load Stage 1 results from .ascota
        from app.services.metadata import load_stage1_data_for_image
        for image_id, image_item in session.images.items():
            context_path = Path(image_item.context_id)
            find_path = context_path / "finds" / "individual" / image_item.find_number
            
            image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
            if image_path:
                image_filename = image_path.stem.replace("-3000", "")
            else:
                image_filename = image_id
            
            existing_data = load_stage1_data_for_image(str(find_path), image_filename)
            if existing_data:
                session.stage1_results[image_id] = existing_data
    
    if not session.stage1_results:
        raise HTTPException(status_code=400, detail="Stage 1 must be completed first")
    
    results: Dict[str, MaskResult] = {}
    
    # Process each image
    for image_id, image_item in session.images.items():
        # Build find path
        context_path = Path(image_item.context_id)
        find_path = context_path / "finds" / "individual" / image_item.find_number
        
        # Get image filename
        image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
        if image_path:
            image_filename = image_path.stem.replace("-3000", "")
        else:
            image_filename = image_id
        
        # Check for existing Stage 2 data
        existing_data = load_stage2_data_for_image(str(find_path), image_filename)
        
        if existing_data and existing_data.get("mask_path"):
            # Check if mask file actually exists
            mask_file_path = find_path / ".ascota" / existing_data.get("mask_path")
            if mask_file_path.exists():
                # Use existing mask
                logger.info(f"Using existing Stage 2 data for image {image_id} in find {image_item.find_number}")
                results[image_id] = MaskResult(
                    image_id=image_id,
                    mask_path=existing_data.get("mask_path"),
                    error=existing_data.get("error")
                )
                continue
        
        # No existing data, generate mask
        # Use -3000 image for mask generation
        image_path_str = image_item.proxy_3000
        if not image_path_str or not Path(image_path_str).exists():
            results[image_id] = MaskResult(
                image_id=image_id,
                mask_path=None,
                error=f"Image file not found: {image_path_str}"
            )
            continue
        
        # Get card coordinates from Stage 1
        stage1_data = session.stage1_results.get(image_id, {})
        cards_data = stage1_data.get("cards", [])
        
        # Convert to format expected by segmentation service
        card_coordinates = None
        if cards_data:
            card_coordinates = [
                {
                    "class": card.get("card_type"),
                    "coordinates": card.get("coordinates"),
                    "confidence": card.get("confidence", 0.0)
                }
                for card in cards_data
            ]
        
        # Generate mask
        mask = generate_mask(image_path_str, card_coordinates, debug=False)
        
        if mask is None:
            results[image_id] = MaskResult(
                image_id=image_id,
                mask_path=None,
                error="Mask generation failed"
            )
        else:
            # Save mask to .ascota folder
            if save_mask(str(find_path), image_filename, mask):
                # Store relative path
                mask_path = f"masks/{image_filename}_mask.png"
                results[image_id] = MaskResult(
                    image_id=image_id,
                    mask_path=mask_path,
                    error=None
                )
            else:
                results[image_id] = MaskResult(
                    image_id=image_id,
                    mask_path=None,
                    error="Failed to save mask"
                )
    
    # Store results in session
    session.stage2_results = {
        img_id: {
            "mask_path": result.mask_path,
            "error": result.error
        }
        for img_id, result in results.items()
    }
    session.updated_at = datetime.now()
    
    return Stage2Results(results=results)


@router.get("/{session_id}/stage2/results", response_model=Stage2Results)
async def get_stage2_results(session_id: str):
    """Get mask generation results, loading from .ascota if available."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Ensure Stage 1 results are loaded in session (needed for mask generation)
    if not session.stage1_results:
        # Load Stage 1 results from .ascota
        from app.services.metadata import load_stage1_data_for_image
        for image_id, image_item in session.images.items():
            context_path = Path(image_item.context_id)
            find_path = context_path / "finds" / "individual" / image_item.find_number
            
            image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
            if image_path:
                image_filename = image_path.stem.replace("-3000", "")
            else:
                image_filename = image_id
            
            existing_data = load_stage1_data_for_image(str(find_path), image_filename)
            if existing_data:
                session.stage1_results[image_id] = existing_data
    
    # Load existing data from .ascota folders and merge with session results
    results: Dict[str, MaskResult] = {}
    
    for image_id, image_item in session.images.items():
        # Build find path
        context_path = Path(image_item.context_id)
        find_path = context_path / "finds" / "individual" / image_item.find_number
        
        # Get image filename
        image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
        if image_path:
            image_filename = image_path.stem.replace("-3000", "")
        else:
            image_filename = image_id
        
        # Check for existing data in .ascota
        existing_data = load_stage2_data_for_image(str(find_path), image_filename)
        
        if existing_data and existing_data.get("mask_path"):
            # Verify mask file exists
            mask_file_path = find_path / ".ascota" / existing_data.get("mask_path")
            if mask_file_path.exists():
                # Use existing data from .ascota
                results[image_id] = MaskResult(
                    image_id=image_id,
                    mask_path=existing_data.get("mask_path"),
                    error=existing_data.get("error")
                )
                continue
        
        # Use session results if no .ascota data
        if image_id in session.stage2_results:
            result_data = session.stage2_results[image_id]
            results[image_id] = MaskResult(
                image_id=image_id,
                mask_path=result_data.get("mask_path"),
                error=result_data.get("error")
            )
    
    return Stage2Results(results=results)


@router.put("/{session_id}/stage2/image/{image_id}/mask")
async def update_mask(
    session_id: str,
    image_id: str,
    request: UpdateMaskRequest
):
    """Update mask from painting edits."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Convert base64 to mask
    mask = base64_to_mask(request.mask_data)
    if mask is None:
        raise HTTPException(status_code=400, detail="Invalid mask data")
    
    # Save mask to .ascota folder
    context_path = Path(image.context_id)
    find_path = context_path / "finds" / "individual" / image.find_number
    
    # Get image filename
    image_path = Path(image.proxy_3000) if image.proxy_3000 else None
    if image_path:
        image_filename = image_path.stem.replace("-3000", "")
    else:
        image_filename = image_id
    
    if save_mask(str(find_path), image_filename, mask):
        mask_path = f"masks/{image_filename}_mask.png"
        
        # Update session
        session.stage2_results[image_id] = {
            "mask_path": mask_path,
            "error": None
        }
        session.updated_at = datetime.now()
        
        return {"message": "Mask updated", "mask_path": mask_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to save mask")


@router.post("/{session_id}/stage2/image/{image_id}/wand-select")
async def wand_select(
    session_id: str,
    image_id: str,
    request: WandSelectRequest
):
    """Use MobileSAM wand tool to select foreground region from a point click."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Use -3000 image for MobileSAM segmentation
    image_path_str = image.proxy_3000
    if not image_path_str or not Path(image_path_str).exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path_str}")
    
    # Validate coordinates are within image bounds
    from PIL import Image as PILImage
    try:
        img = PILImage.open(image_path_str)
        img_width, img_height = img.size
        
        if request.x < 0 or request.x >= img_width or request.y < 0 or request.y >= img_height:
            raise HTTPException(
                status_code=400,
                detail=f"Point coordinates ({request.x}, {request.y}) are out of bounds. Image size: {img_width}x{img_height}"
            )
    except Exception as e:
        logger.error(f"Failed to validate image dimensions: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to validate image: {e}")
    
    # Run MobileSAM segmentation
    try:
        mask = segment_from_point(image_path_str, request.x, request.y, point_label=1)
        
        if mask is None:
            raise HTTPException(status_code=500, detail="MobileSAM segmentation failed")
        
        # Convert mask to base64
        mask_data = mask_to_base64(mask)
        
        return {"mask_data": mask_data}
        
    except Exception as e:
        logger.error(f"Wand select failed for image {image_id} at ({request.x}, {request.y}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Wand selection failed: {str(e)}")


@router.post("/{session_id}/stage2/save")
async def save_stage2(session_id: str):
    """Save Stage 2 results to .ascota/preprocess.json files."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Group images by find path
    finds_data: Dict[str, Dict] = {}
    
    for image_id, image_item in session.images.items():
        # Build find path
        context_path = Path(image_item.context_id)
        find_path = context_path / "finds" / "individual" / image_item.find_number
        
        find_path_str = str(find_path)
        
        if find_path_str not in finds_data:
            finds_data[find_path_str] = {}
        
        # Get result data
        result_data = session.stage2_results.get(image_id, {})
        
        # Get image filename
        image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
        if image_path:
            image_filename = image_path.stem.replace("-3000", "")
        else:
            image_filename = image_id
        
        # Store in finds_data
        finds_data[find_path_str][image_filename] = {
            "mask_path": result_data.get("mask_path"),
            "error": result_data.get("error")
        }
    
    # Save to each find's .ascota folder
    saved_count = 0
    for find_path_str, images_data in finds_data.items():
        # Load existing metadata
        existing_data = load_preprocess_json(find_path_str)
        
        # Update with Stage 2 data
        if "stage2" not in existing_data:
            existing_data["stage2"] = {}
        
        existing_data["stage2"]["masks"] = images_data
        existing_data["stage2"]["timestamp"] = datetime.now().isoformat()
        
        # Save
        if save_preprocess_json(find_path_str, existing_data):
            saved_count += 1
    
    return {
        "message": f"Saved Stage 2 results to {saved_count} find(s)",
        "saved_count": saved_count
    }

