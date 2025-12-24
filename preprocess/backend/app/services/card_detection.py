"""Service for card detection using imaging module."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import logging

# Add project root to path
# From: preprocess/backend/app/services/card_detection.py
# To: ascota/ (project root)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ascota_core.imaging import detect_color_cards

logger = logging.getLogger(__name__)


def detect_cards_in_image(image_path: str, debug: bool = False) -> Dict[str, Any]:
    """Detect color cards in an image.
    
    Args:
        image_path: Path to image file
        debug: Enable debug output
        
    Returns:
        Dictionary with:
            - cards: List of detection dictionaries
            - image_size: [width, height]
            - error: Error message if detection failed
    """
    try:
        # Load image
        image = Image.open(image_path)
        image_size = [image.width, image.height]
        
        # Run detection
        detections = detect_color_cards(image, debug=debug)
        
        # Convert to our format
        cards = []
        for i, det in enumerate(detections):
            card = {
                "card_id": f"card_{i}",
                "card_type": det['class'],
                "coordinates": det['coordinates'],
                "confidence": det['confidence']
            }
            cards.append(card)
        
        return {
            "cards": cards,
            "image_size": image_size,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Card detection failed for {image_path}: {e}", exc_info=True)
        return {
            "cards": [],
            "image_size": None,
            "error": str(e)
        }

