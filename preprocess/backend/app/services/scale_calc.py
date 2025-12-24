"""Service for scale calculation and surface area measurement."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import numpy as np
import cv2
import logging

# Add project root to path
# From: preprocess/backend/app/services/scale_calc.py
# To: ascota/ (project root)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ascota_core.scale import (
    calculate_pp_cm_checker_card,
    find_circle_centers_8_hybrid_card,
    calculate_pp_cm_from_centers,
    artifact_face_size
)

logger = logging.getLogger(__name__)


def calculate_scale_from_card(
    card_crop_path: str,
    card_type: str,
    card_coordinates: List[List[float]],
    original_image_path: str,
    debug: bool = False
) -> Dict[str, Any]:
    """Calculate pixels per cm from a card crop.
    
    Args:
        card_crop_path: Path to cropped card image
        card_type: Type of card (checker_card or 8_hybrid_card)
        card_coordinates: 4 corner coordinates in original image
        original_image_path: Path to original full image
        debug: Enable debug output
        
    Returns:
        Dictionary with:
            - pixels_per_cm: Calculated scale
            - method: Card type used
            - error: Error message if calculation failed
            - centers: Circle centers (for 8_hybrid_card)
            - debug_image: Debug visualization (if debug=True)
    """
    try:
        # Load card crop
        card_image = Image.open(card_crop_path)
        card_array = np.array(card_image)
        
        # Convert RGB to BGR for OpenCV functions
        if len(card_array.shape) == 3:
            card_bgr = cv2.cvtColor(card_array, cv2.COLOR_RGB2BGR)
        else:
            card_bgr = card_array
        
        if card_type == "checker_card":
            # Use checker card method
            pp_cm, debug_img = calculate_pp_cm_checker_card(card_bgr, debug=debug)
            
            return {
                "pixels_per_cm": float(pp_cm),
                "method": "checker_card",
                "error": None,
                "centers": None,
                "debug_image": debug_img
            }
            
        elif card_type == "8_hybrid_card":
            # Use 8-hybrid card method
            centers, centers_debug_img = find_circle_centers_8_hybrid_card(card_bgr, debug=debug)
            
            if centers is None:
                return {
                    "pixels_per_cm": None,
                    "method": "8_hybrid_card",
                    "error": "Could not detect circle centers",
                    "centers": None,
                    "debug_image": centers_debug_img
                }
            
            # Calculate scale from centers
            pp_cm, calc_debug_img = calculate_pp_cm_from_centers(centers, card_bgr, debug=debug)
            
            # Convert centers to list format
            centers_list = centers.tolist() if isinstance(centers, np.ndarray) else centers
            
            return {
                "pixels_per_cm": float(pp_cm),
                "method": "8_hybrid_card",
                "error": None,
                "centers": centers_list,
                "debug_image": calc_debug_img if calc_debug_img else centers_debug_img
            }
            
        else:
            return {
                "pixels_per_cm": None,
                "method": card_type,
                "error": f"Card type {card_type} has no scale reference",
                "centers": None,
                "debug_image": None
            }
            
    except Exception as e:
        logger.error(f"Scale calculation failed for {card_crop_path}: {e}", exc_info=True)
        return {
            "pixels_per_cm": None,
            "method": card_type,
            "error": str(e),
            "centers": None,
            "debug_image": None
        }


def calculate_surface_area(
    mask: np.ndarray,
    pixels_per_cm: float,
    debug: bool = False
) -> float:
    """Calculate artifact surface area from mask.
    
    Args:
        mask: Binary mask (0/1)
        pixels_per_cm: Scale factor
        debug: Enable debug output
        
    Returns:
        Surface area in cm²
    """
    try:
        area = artifact_face_size(mask, pixels_per_cm, debug=debug)
        return float(area)
    except Exception as e:
        logger.error(f"Surface area calculation failed: {e}", exc_info=True)
        raise


def crop_card_from_image(
    image_path: str,
    card_coordinates: List[List[float]]
) -> Optional[Image.Image]:
    """Crop card region from image using perspective transform.
    
    Args:
        image_path: Path to full image
        card_coordinates: 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        Cropped card image, or None if cropping failed
    """
    try:
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Convert to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Convert coordinates to numpy array
        src_points = np.array(card_coordinates, dtype=np.float32)
        
        # Calculate bounding box dimensions
        x_coords = src_points[:, 0]
        y_coords = src_points[:, 1]
        width = int(np.max(x_coords) - np.min(x_coords))
        height = int(np.max(y_coords) - np.min(y_coords))
        
        # Destination points for perspective transform (rectangular crop)
        dst_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(img_bgr, M, (width, height))
        
        # Convert back to RGB PIL Image
        if len(warped.shape) == 3:
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        else:
            warped_rgb = warped
        
        return Image.fromarray(warped_rgb)
        
    except Exception as e:
        logger.error(f"Card cropping failed for {image_path}: {e}", exc_info=True)
        return None

