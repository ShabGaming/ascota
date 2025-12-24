"""Stage 1: Card detection routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import logging
from pathlib import Path

from app.services.session_store import get_session_store
from app.services.models import (
    Stage1Results, ImageCardResult, UpdateCardsRequest, CardDetection
)
from app.services.card_detection import detect_cards_in_image
from app.services.metadata import load_preprocess_json, save_preprocess_json, load_stage1_data_for_image
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{session_id}/stage1/detect", response_model=Stage1Results)
async def detect_cards(session_id: str, background_tasks: BackgroundTasks):
    """Run card detection on all images in the session."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    results: Dict[str, ImageCardResult] = {}
    
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
        
        # Check for existing Stage 1 data
        existing_data = load_stage1_data_for_image(str(find_path), image_filename)
        
        if existing_data and existing_data.get("cards"):
            # Use existing data
            logger.info(f"Using existing Stage 1 data for image {image_id} in find {image_item.find_number}")
            cards = [
                CardDetection(**card_data)
                for card_data in existing_data.get("cards", [])
            ]
            
            results[image_id] = ImageCardResult(
                image_id=image_id,
                image_size=existing_data.get("image_size", []),
                cards=cards,
                error=existing_data.get("error")
            )
            continue
        
        # No existing data, run detection
        # Use -3000 image for detection
        image_path_str = image_item.proxy_3000
        if not image_path_str or not Path(image_path_str).exists():
            results[image_id] = ImageCardResult(
                image_id=image_id,
                image_size=[],
                cards=[],
                error=f"Image file not found: {image_path_str}"
            )
            continue
        
        # Run detection
        detection_result = detect_cards_in_image(image_path_str, debug=False)
        
        if detection_result["error"]:
            results[image_id] = ImageCardResult(
                image_id=image_id,
                image_size=detection_result.get("image_size", []),
                cards=[],
                error=detection_result["error"]
            )
        else:
            # Convert to CardDetection models
            cards = [
                CardDetection(**card_data)
                for card_data in detection_result["cards"]
            ]
            
            results[image_id] = ImageCardResult(
                image_id=image_id,
                image_size=detection_result["image_size"],
                cards=cards,
                error=None
            )
    
    # Store results in session
    session.stage1_results = {
        img_id: {
            "image_size": result.image_size,
            "cards": [card.dict() for card in result.cards],
            "error": result.error
        }
        for img_id, result in results.items()
    }
    session.updated_at = datetime.now()
    
    return Stage1Results(results=results)


@router.get("/{session_id}/stage1/results", response_model=Stage1Results)
async def get_stage1_results(session_id: str):
    """Get card detection results, loading from .ascota if available."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Load existing data from .ascota folders and merge with session results
    results: Dict[str, ImageCardResult] = {}
    
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
        existing_data = load_stage1_data_for_image(str(find_path), image_filename)
        
        if existing_data:
            # Use existing data from .ascota
            cards = [
                CardDetection(**card_data)
                for card_data in existing_data.get("cards", [])
            ]
            results[image_id] = ImageCardResult(
                image_id=image_id,
                image_size=existing_data.get("image_size", []),
                cards=cards,
                error=existing_data.get("error")
            )
        elif image_id in session.stage1_results:
            # Use session results if no .ascota data
            result_data = session.stage1_results[image_id]
            cards = [
                CardDetection(**card_data)
                for card_data in result_data.get("cards", [])
            ]
            results[image_id] = ImageCardResult(
                image_id=image_id,
                image_size=result_data.get("image_size", []),
                cards=cards,
                error=result_data.get("error")
            )
    
    return Stage1Results(results=results)


@router.put("/{session_id}/stage1/image/{image_id}/cards")
async def update_cards(
    session_id: str,
    image_id: str,
    request: UpdateCardsRequest
):
    """Update card coordinates for an image."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Update cards in session
    if image_id not in session.stage1_results:
        # Get image size from image file
        image_path = image.proxy_3000
        if image_path and Path(image_path).exists():
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            image_size = [img.width, img.height]
        else:
            image_size = []
        
        session.stage1_results[image_id] = {
            "image_size": image_size,
            "cards": [],
            "error": None
        }
    
    session.stage1_results[image_id]["cards"] = [
        card.dict() for card in request.cards
    ]
    session.updated_at = datetime.now()
    
    return {"message": "Cards updated", "cards": request.cards}


@router.post("/{session_id}/stage1/image/{image_id}/cards")
async def add_card(
    session_id: str,
    image_id: str,
    card: CardDetection
):
    """Add a new card to an image."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image = session.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Ensure stage1_results exists for this image
    if image_id not in session.stage1_results:
        image_path = image.proxy_3000
        if image_path and Path(image_path).exists():
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            image_size = [img.width, img.height]
        else:
            image_size = []
        
        session.stage1_results[image_id] = {
            "image_size": image_size,
            "cards": [],
            "error": None
        }
    
    # Add card
    session.stage1_results[image_id]["cards"].append(card.dict())
    session.updated_at = datetime.now()
    
    return {"message": "Card added", "card": card}


@router.delete("/{session_id}/stage1/image/{image_id}/cards/{card_id}")
async def delete_card(
    session_id: str,
    image_id: str,
    card_id: str
):
    """Remove a card from an image."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if image_id not in session.stage1_results:
        raise HTTPException(status_code=404, detail="Image results not found")
    
    # Remove card
    cards = session.stage1_results[image_id]["cards"]
    session.stage1_results[image_id]["cards"] = [
        c for c in cards if c.get("card_id") != card_id
    ]
    session.updated_at = datetime.now()
    
    return {"message": "Card deleted"}


@router.post("/{session_id}/stage1/save")
async def save_stage1(session_id: str):
    """Save Stage 1 results to .ascota/preprocess.json files."""
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
        result_data = session.stage1_results.get(image_id, {})
        
        # Get image filename (base name without extension)
        image_path = Path(image_item.proxy_3000) if image_item.proxy_3000 else None
        if image_path:
            image_filename = image_path.stem.replace("-3000", "")
        else:
            image_filename = image_id
        
        # Store in finds_data
        finds_data[find_path_str][image_filename] = {
            "image_size": result_data.get("image_size", []),
            "cards": result_data.get("cards", []),
            "error": result_data.get("error")
        }
    
    # Save to each find's .ascota folder
    saved_count = 0
    for find_path_str, images_data in finds_data.items():
        # Load existing metadata
        existing_data = load_preprocess_json(find_path_str)
        
        # Update with Stage 1 data
        if "stage1" not in existing_data:
            existing_data["stage1"] = {}
        
        existing_data["stage1"]["images"] = images_data
        existing_data["stage1"]["timestamp"] = datetime.now().isoformat()
        
        # Save
        if save_preprocess_json(find_path_str, existing_data):
            saved_count += 1
    
    return {
        "message": f"Saved Stage 1 results to {saved_count} find(s)",
        "saved_count": saved_count
    }

