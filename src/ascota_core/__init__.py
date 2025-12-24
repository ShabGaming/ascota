"""
Ascota Core - Color card detection and processing pipeline.

This module provides functionality for detecting, extracting, and processing
color cards (ColorChecker 24, ColorChecker 8) from images using computer vision
techniques and machine learning models.
"""

from .imaging import (
    detect_color_cards,
    remove_background_mask,
    generate_swatch
)

from .utils import (
    load_image_any,
    resize_max,
    polygon_to_mask,
    non_max_suppression_polys,
    cv2_to_pil,
    pil_to_cv2,
    create_transparent_image,
    contour_rect_fallback
)

from .scale import (
    calculate_pp_cm_checker_card,
    find_circle_centers_8_hybrid_card,
    calculate_pp_cm_from_centers,
    artifact_face_size
)

__version__ = "0.1.0"
__author__ = "Muhammad Shahab Hasan"

# Main classes and functions for easy access
__all__ = [
    # Main classes
    "CardDetector",
    
    # Core functions
    "process_image_pipeline",
    "detect_card_type",
    "create_card_masks_and_transparent",
    "mask_out_cards",
    
    # RMBG integration
    "setup_rmbg_pipeline",
    "remove_background_and_generate_mask",
    
    # YOLOv8 color card detection
    "detect_color_cards",
    
    # RMBG background removal
    "remove_background_mask",
    
    # Swatch generation
    "generate_swatch",
    
    # Utility functions
    "load_image_any",
    "cv2_to_pil",
    "pil_to_cv2",
    "create_transparent_image",
    
    # Scaling functions
    "calculate_pp_cm_checker_card",
    "calculate_pp_cm_8_hybrid_card",
    "find_circle_centers_8_hybrid_card",
    "calculate_pp_cm_from_centers",
    "artifact_face_size",
]
