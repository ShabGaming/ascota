"""Service for background removal and mask generation."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import logging
import base64
from io import BytesIO

# Add project root to path
# From: preprocess/backend/app/services/segmentation.py
# To: ascota/ (project root)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ascota_core.imaging import remove_background_mask

logger = logging.getLogger(__name__)


def generate_mask(
    image_path: str,
    card_coordinates: Optional[List[Dict[str, Any]]] = None,
    debug: bool = False
) -> Optional[np.ndarray]:
    """Generate binary mask for an image.
    
    Args:
        image_path: Path to image file
        card_coordinates: List of card detection dictionaries (from Stage 1)
        debug: Enable debug output
        
    Returns:
        Binary mask as numpy array (0/1), or None if generation failed
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Run background removal
        mask_result = remove_background_mask(
            image,
            card_coordinates=card_coordinates,
            debug=debug
        )
        
        # Handle debug mode return (tuple) vs normal mode (just mask)
        if debug and isinstance(mask_result, tuple):
            mask = mask_result[0]
        else:
            mask = mask_result
        
        # Ensure mask is 0/1 format
        if mask.max() > 1:
            mask = (mask > 128).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        return mask
        
    except Exception as e:
        logger.error(f"Mask generation failed for {image_path}: {e}", exc_info=True)
        return None


def apply_mask_edits(
    base_mask: np.ndarray,
    edits: List[Dict[str, Any]]
) -> np.ndarray:
    """Apply brush edits to a mask.
    
    Args:
        base_mask: Original binary mask (0/1)
        edits: List of edit operations:
            - type: "paint_in" or "paint_out"
            - x, y: Center coordinates
            - radius: Brush radius in pixels
            
    Returns:
        Modified binary mask
    """
    mask = base_mask.copy().astype(np.uint8)
    
    for edit in edits:
        edit_type = edit.get("type")
        x = int(edit.get("x", 0))
        y = int(edit.get("y", 0))
        radius = int(edit.get("radius", 10))
        
        if edit_type == "paint_in":
            # Paint foreground (set to 1)
            _paint_circle(mask, x, y, radius, 1)
        elif edit_type == "paint_out":
            # Paint background (set to 0)
            _paint_circle(mask, x, y, radius, 0)
    
    return mask


def _paint_circle(mask: np.ndarray, cx: int, cy: int, radius: int, value: int):
    """Paint a circle on the mask.
    
    Args:
        mask: Binary mask array
        cx, cy: Circle center
        radius: Circle radius
        value: Value to paint (0 or 1)
    """
    h, w = mask.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Create circular mask
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    circle_mask = dist_sq <= radius ** 2
    
    # Apply paint
    mask[circle_mask] = value


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert mask array to base64 encoded PNG string.
    
    Args:
        mask: Binary mask array (0/1 or 0/255)
        
    Returns:
        Base64 encoded PNG string
    """
    # Ensure mask is in 0/255 format
    if mask.max() <= 1:
        mask_255 = (mask * 255).astype(np.uint8)
    else:
        mask_255 = mask.astype(np.uint8)
    
    # Convert to PIL Image
    mask_image = Image.fromarray(mask_255, mode='L')
    
    # Convert to base64
    buffer = BytesIO()
    mask_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_mask(base64_str: str) -> Optional[np.ndarray]:
    """Convert base64 encoded PNG to mask array.
    
    Args:
        base64_str: Base64 encoded PNG string
        
    Returns:
        Binary mask array (0/1), or None if conversion failed
    """
    try:
        # Decode base64
        image_data = base64.b64decode(base64_str)
        buffer = BytesIO(image_data)
        
        # Load image
        mask_image = Image.open(buffer)
        mask_array = np.array(mask_image.convert('L'), dtype=np.uint8)
        
        # Convert to 0/1 format
        mask_array = (mask_array > 128).astype(np.uint8)
        
        return mask_array
        
    except Exception as e:
        logger.error(f"Failed to convert base64 to mask: {e}", exc_info=True)
        return None

