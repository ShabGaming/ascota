"""Stage 3: Scale calculation routes."""

from fastapi import APIRouter, HTTPException
from typing import Dict
import logging
from pathlib import Path
import tempfile

from app.services.session_store import get_session_store
from app.services.models import (
    Stage3Results, ScaleResult, UpdateCentersRequest
)
from app.services.scale_calc import (
    calculate_scale_from_card,
    crop_card_from_image,
    calculate_surface_area
)
from app.services.metadata import load_mask, load_preprocess_json, save_preprocess_json, load_stage3_data_for_image
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{session_id}/stage3/calculate", response_model=Stage3Results)
async def calculate_scale(session_id: str):
    """Calculate scale for all images using card metadata from Stage 1."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.stage1_results:
        raise HTTPException(status_code=400, detail="Stage 1 must be completed first")
    
    results: Dict[str, ScaleResult] = {}
    
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
        
        # Check for existing Stage 3 data
        existing_data = load_stage3_data_for_image(str(find_path), image_filename)
        
        if existing_data and existing_data.get("pixels_per_cm"):
            # Use existing data
            logger.info(f"Using existing Stage 3 data for image {image_id} in find {image_item.find_number}")
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=existing_data.get("pixels_per_cm"),
                surface_area_cm2=existing_data.get("surface_area_cm2"),
                method=existing_data.get("method"),
                card_used=existing_data.get("card_used"),
                error=existing_data.get("error")
            )
            continue
        
        # No existing data, calculate scale
        # Get Stage 1 data
        stage1_data = session.stage1_results.get(image_id, {})
        cards_data = stage1_data.get("cards", [])
        
        if not cards_data:
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=None,
                surface_area_cm2=None,
                method=None,
                card_used=None,
                error="No cards detected in Stage 1"
            )
            continue
        
        # Find a card with scale reference (checker_card or 8_hybrid_card)
        scale_card = None
        for card in cards_data:
            card_type = card.get("card_type")
            if card_type in ["checker_card", "8_hybrid_card"]:
                scale_card = card
                break
        
        if not scale_card:
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=None,
                surface_area_cm2=None,
                method=None,
                card_used=None,
                centers=None,
                error="No scale reference card found (need checker_card or 8_hybrid_card)"
            )
            continue
        
        # Crop card from image
        image_path = image_item.proxy_3000
        if not image_path or not Path(image_path).exists():
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=None,
                surface_area_cm2=None,
                method=None,
                card_used=None,
                centers=None,
                error=f"Image file not found: {image_path}"
            )
            continue
        
        card_crop = crop_card_from_image(image_path, scale_card.get("coordinates"))
        if card_crop is None:
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=None,
                surface_area_cm2=None,
                method=None,
                card_used=None,
                centers=None,
                error="Failed to crop card from image"
            )
            continue
        
        # Save card crop to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            card_crop.save(tmp_file.name, format='JPEG')
            card_crop_path = tmp_file.name
        
        try:
            # Calculate scale
            scale_result = calculate_scale_from_card(
                card_crop_path,
                scale_card.get("card_type"),
                scale_card.get("coordinates"),
                image_path,
                debug=False
            )
            
            if scale_result["error"]:
                results[image_id] = ScaleResult(
                    image_id=image_id,
                    pixels_per_cm=None,
                    surface_area_cm2=None,
                    method=scale_result["method"],
                    card_used=scale_card.get("card_id"),
                    centers=None,
                    error=scale_result["error"]
                )
            else:
                pixels_per_cm = scale_result["pixels_per_cm"]
                
                # Transform centers from card crop coordinates to original image coordinates
                centers_in_image = None
                if scale_result.get("centers") and scale_result["method"] == "8_hybrid_card":
                    import cv2
                    import numpy as np
                    
                    # Centers are in card crop coordinates, need to transform back to image coordinates
                    card_coords = np.array(scale_card.get("coordinates"), dtype=np.float32)
                    centers_crop = np.array(scale_result["centers"], dtype=np.float32)
                    
                    # Calculate perspective transform from card crop to original image
                    # Card crop is rectangular: [0,0], [w,0], [w,h], [0,h]
                    x_coords = card_coords[:, 0]
                    y_coords = card_coords[:, 1]
                    crop_width = float(np.max(x_coords) - np.min(x_coords))
                    crop_height = float(np.max(y_coords) - np.min(y_coords))
                    
                    dst_points = np.array([
                        [0, 0],
                        [crop_width, 0],
                        [crop_width, crop_height],
                        [0, crop_height]
                    ], dtype=np.float32)
                    
                    # Get inverse perspective transform
                    M = cv2.getPerspectiveTransform(dst_points, card_coords)
                    
                    # Transform centers from crop to image coordinates
                    centers_homogeneous = np.hstack([centers_crop, np.ones((centers_crop.shape[0], 1))])
                    centers_transformed = (M @ centers_homogeneous.T).T
                    centers_in_image = (centers_transformed[:, :2] / centers_transformed[:, 2:3]).tolist()
                
                # Calculate surface area if mask exists
                surface_area = None
                if image_id in session.stage2_results:
                    mask_path_data = session.stage2_results[image_id].get("mask_path")
                    if mask_path_data:
                        # Load mask
                        context_path = Path(image_item.context_id)
                        find_path = context_path / "finds" / "individual" / image_item.find_number
                        
                        image_path_obj = Path(image_path)
                        image_filename = image_path_obj.stem.replace("-3000", "")
                        
                        mask = load_mask(str(find_path), image_filename)
                        if mask is not None and pixels_per_cm:
                            try:
                                surface_area = calculate_surface_area(mask, pixels_per_cm, debug=False)
                            except Exception as e:
                                logger.warning(f"Surface area calculation failed: {e}")
                
                results[image_id] = ScaleResult(
                    image_id=image_id,
                    pixels_per_cm=pixels_per_cm,
                    surface_area_cm2=surface_area,
                    method=scale_result["method"],
                    card_used=scale_card.get("card_id"),
                    centers=centers_in_image,
                    error=None
                )
        finally:
            # Clean up temp file
            try:
                Path(card_crop_path).unlink()
            except Exception:
                pass
    
    # Store results in session
    session.stage3_results = {
        img_id: {
            "pixels_per_cm": result.pixels_per_cm,
            "surface_area_cm2": result.surface_area_cm2,
            "method": result.method,
            "card_used": result.card_used,
            "centers": result.centers,
            "error": result.error
        }
        for img_id, result in results.items()
    }
    session.updated_at = datetime.now()
    
    return Stage3Results(results=results)


@router.get("/{session_id}/stage3/results", response_model=Stage3Results)
async def get_stage3_results(session_id: str):
    """Get scale calculation results, loading from .ascota if available."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Ensure Stage 1 results are loaded in session (needed for scale calculation)
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
    results: Dict[str, ScaleResult] = {}
    
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
        existing_data = load_stage3_data_for_image(str(find_path), image_filename)
        
        if existing_data:
            # Use existing data from .ascota
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=existing_data.get("pixels_per_cm"),
                surface_area_cm2=existing_data.get("surface_area_cm2"),
                method=existing_data.get("method"),
                card_used=existing_data.get("card_used"),
                error=existing_data.get("error")
            )
        elif image_id in session.stage3_results:
            # Use session results if no .ascota data
            result_data = session.stage3_results[image_id]
            results[image_id] = ScaleResult(
                image_id=image_id,
                pixels_per_cm=result_data.get("pixels_per_cm"),
                surface_area_cm2=result_data.get("surface_area_cm2"),
                method=result_data.get("method"),
                card_used=result_data.get("card_used"),
                centers=result_data.get("centers"),
                error=result_data.get("error")
            )
    
    return Stage3Results(results=results)


@router.put("/{session_id}/stage3/image/{image_id}/centers")
async def update_centers(
    session_id: str,
    image_id: str,
    request: UpdateCentersRequest
):
    """Update 8-hybrid circle centers and recalculate scale."""
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
    
    # Save card crop to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        card_crop.save(tmp_file.name, format='JPEG')
        card_crop_path = tmp_file.name
    
    try:
        # Convert centers to numpy array for calculation
        import numpy as np
        import sys
        
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        centers_array = np.array(request.centers, dtype=np.float32)
        
        # Calculate scale from updated centers
        from src.ascota_core.scale import calculate_pp_cm_from_centers
        card_array = np.array(card_crop)
        if len(card_array.shape) == 3:
            import cv2
            card_bgr = cv2.cvtColor(card_array, cv2.COLOR_RGB2BGR)
        else:
            card_bgr = card_array
        
        pixels_per_cm, _ = calculate_pp_cm_from_centers(centers_array, card_bgr, debug=False)
        
        # Calculate surface area if mask exists
        surface_area = None
        if image_id in session.stage2_results:
            mask_path_data = session.stage2_results[image_id].get("mask_path")
            if mask_path_data:
                # Load mask
                context_path = Path(image.context_id)
                find_path = context_path / "finds" / "individual" / image.find_number
                
                image_path_obj = Path(image_path)
                image_filename = image_path_obj.stem.replace("-3000", "")
                
                mask = load_mask(str(find_path), image_filename)
                if mask is not None:
                    try:
                        surface_area = calculate_surface_area(mask, float(pixels_per_cm), debug=False)
                    except Exception as e:
                        logger.warning(f"Surface area calculation failed: {e}")
        
        # Update session
        session.stage3_results[image_id] = {
            "pixels_per_cm": float(pixels_per_cm),
            "surface_area_cm2": surface_area,
            "method": "8_hybrid_card",
            "card_used": hybrid_card.get("card_id"),
            "error": None
        }
        session.updated_at = datetime.now()
        
        return {
            "message": "Centers updated and scale recalculated",
            "pixels_per_cm": float(pixels_per_cm),
            "surface_area_cm2": surface_area
        }
    finally:
        # Clean up temp file
        try:
            Path(card_crop_path).unlink()
        except Exception:
            pass


@router.post("/{session_id}/stage3/image/{image_id}/surface_area")
async def calculate_surface_area_for_image(session_id: str, image_id: str):
    """Recalculate surface area for an image using current scale."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get scale from Stage 3
    stage3_data = session.stage3_results.get(image_id, {})
    pixels_per_cm = stage3_data.get("pixels_per_cm")
    
    if not pixels_per_cm:
        raise HTTPException(status_code=400, detail="No scale calculated for this image")
    
    # Load mask
    if image_id not in session.stage2_results:
        raise HTTPException(status_code=400, detail="No mask found for this image")
    
    mask_path_data = session.stage2_results[image_id].get("mask_path")
    if not mask_path_data:
        raise HTTPException(status_code=400, detail="No mask path found")
    
    context_path = Path(image.context_id)
    find_path = context_path / "finds" / "individual" / image.find_number
    
    image_path = Path(image.proxy_3000) if image.proxy_3000 else None
    if image_path:
        image_filename = image_path.stem.replace("-3000", "")
    else:
        image_filename = image_id
    
    mask = load_mask(str(find_path), image_filename)
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask file not found")
    
    # Calculate surface area
    try:
        surface_area = calculate_surface_area(mask, pixels_per_cm, debug=False)
        
        # Update session
        session.stage3_results[image_id]["surface_area_cm2"] = surface_area
        session.updated_at = datetime.now()
        
        return {
            "message": "Surface area calculated",
            "surface_area_cm2": surface_area
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Surface area calculation failed: {str(e)}")


@router.post("/{session_id}/stage3/save")
async def save_stage3(session_id: str):
    """Save Stage 3 results to .ascota/preprocess.json files."""
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
        result_data = session.stage3_results.get(image_id, {})
        
        # Get image filename
        image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
        if image_path:
            image_filename = image_path.stem.replace("-3000", "")
        else:
            image_filename = image_id
        
        # Store in finds_data
        finds_data[find_path_str][image_filename] = {
            "pixels_per_cm": result_data.get("pixels_per_cm"),
            "surface_area_cm2": result_data.get("surface_area_cm2"),
            "method": result_data.get("method"),
            "card_used": result_data.get("card_used"),
            "error": result_data.get("error")
        }
    
    # Save to each find's .ascota folder
    saved_count = 0
    for find_path_str, images_data in finds_data.items():
        # Load existing metadata
        existing_data = load_preprocess_json(find_path_str)
        
        # Update with Stage 3 data
        if "stage3" not in existing_data:
            existing_data["stage3"] = {}
        
        existing_data["stage3"]["images"] = images_data
        existing_data["stage3"]["timestamp"] = datetime.now().isoformat()
        
        # Save
        if save_preprocess_json(find_path_str, existing_data):
            saved_count += 1
    
    # Append preprocess status to context directories
    try:
        from app.services.metadata import append_context_status
        
        # Get unique context paths from images
        context_paths = set()
        for image_id, image_item in session.images.items():
            if image_item.context_id:
                context_paths.add(image_item.context_id)
        
        # Append status to each context directory
        for context_path in context_paths:
            append_context_status(context_path, {
                "preprocess_status": True,
                "export_summary": {
                    "total_images": len(session.images),
                    "saved_finds": saved_count
                }
            })
        
        logger.info(f"Appended preprocess status to {len(context_paths)} context directories")
    except Exception as e:
        logger.error(f"Failed to append context status: {e}", exc_info=True)
        # Don't fail the save if appending context status fails
    
    return {
        "message": f"Saved Stage 3 results to {saved_count} find(s)",
        "saved_count": saved_count
    }

