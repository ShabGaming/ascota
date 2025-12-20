"""
Image processing functions for color card detection and background removal.

This module provides functions for detecting color reference cards in images
using finetuned YOLOv8 oriented bounding box models, and for generating binary
masks using RMBG-1.4 background removal with optional card area exclusion.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2

# Module-level cached models for performance optimization
_yolo_model: Optional[Any] = None
_rmbg_pipeline: Optional[Any] = None


def _load_yolo_model(debug: bool = False) -> Any:
    """Load and cache YOLOv8 OBB model for color card detection.
    
    Loads the YOLOv8 oriented bounding box model from the models directory
    and caches it at module level to avoid reloading on subsequent calls.
    The model is configured for color card detection with three classes:
    24_color_card, 8_hybrid_card, and checker_card.
    
    Args:
        debug: If True, print debug information during model loading.
            Defaults to False.
    
    Returns:
        Loaded YOLOv8 OBB model instance.
        
    Raises:
        FileNotFoundError: If the model file cannot be found.
        RuntimeError: If model loading fails.
    """
    global _yolo_model
    
    if _yolo_model is not None:
        if debug:
            print("DEBUG: _load_yolo_model - Using cached YOLO model")
        return _yolo_model
    
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics library is required. Install with: pip install ultralytics")
    
    # Get model path relative to this file
    current_dir = Path(__file__).parent
    model_path = current_dir / "models" / "color_card_yolov8m_obb.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLOv8 model file not found at {model_path}. "
            "Please ensure the model file exists in src/ascota_core/models/"
        )
    
    if debug:
        print(f"DEBUG: _load_yolo_model - Loading model from {model_path}")
    
    try:
        _yolo_model = YOLO(str(model_path))
        if debug:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"DEBUG: _load_yolo_model - Model loaded successfully on {device}")
        return _yolo_model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv8 model: {e}") from e


def _load_rmbg_pipeline(debug: bool = False) -> Any:
    """Load and cache RMBG-1.4 pipeline for background removal.
    
    Initializes the RMBG-1.4 transformer pipeline for automated background
    removal and caches it at module level to avoid reloading on subsequent calls.
    The pipeline uses the briaai/RMBG-1.4 model from Hugging Face.
    
    Args:
        debug: If True, print debug information during pipeline initialization.
            Defaults to False.
    
    Returns:
        Configured RMBG pipeline object.
        
    Raises:
        ImportError: If transformers library is not installed.
        RuntimeError: If pipeline initialization fails.
    """
    global _rmbg_pipeline
    
    if _rmbg_pipeline is not None:
        if debug:
            print("DEBUG: _load_rmbg_pipeline - Using cached RMBG pipeline")
        return _rmbg_pipeline
    
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers library is required for RMBG pipeline. "
            "Install with: pip install transformers"
        )
    
    if debug:
        print("DEBUG: _load_rmbg_pipeline - Initializing RMBG-1.4 pipeline")
    
    try:
        _rmbg_pipeline = pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
            use_fast=True
        )
        if debug:
            print("DEBUG: _load_rmbg_pipeline - Pipeline initialized successfully")
        return _rmbg_pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RMBG pipeline: {e}") from e


def detect_color_cards(image: Image.Image, debug: bool = False) -> List[Dict[str, Any]]:
    """Detect color cards in an image using YOLOv8 OBB model.
    
    Processes a PIL image to detect and classify color reference cards
    (24-color, 8-hybrid, or checker cards) using a trained YOLOv8 oriented
    bounding box model. Returns detection results with pixel coordinates
    and classification information for each detected card.
    
    Args:
        image: Input PIL Image to process. Must be in RGB or RGBA format.
        debug: If True, print debug information during processing.
            Defaults to False.
    
    Returns:
        List of dictionaries, one per detected card. Each dictionary contains:
            - class: String class name ('24_color_card', '8_hybrid_card', or 'checker_card')
            - class_id: Integer class ID (0, 1, or 2)
            - confidence: Float confidence score in range [0.0, 1.0]
            - coordinates: List of 4 [x, y] coordinate pairs as pixel coordinates
                representing the four corners of the detected card.
        Returns empty list if no cards are detected.
    
    Raises:
        FileNotFoundError: If the YOLOv8 model file cannot be found.
        RuntimeError: If model loading or inference fails.
        TypeError: If input is not a PIL Image.
    """
    # Validate input
    if not isinstance(image, Image.Image):
        raise TypeError(f"Input must be a PIL Image, got {type(image)}")
    
    # Load model (cached at module level)
    model = _load_yolo_model(debug=debug)
    
    # Convert image to RGB if needed
    if image.mode != 'RGB':
        if debug:
            print(f"DEBUG: detect_color_cards - Converting image from {image.mode} to RGB")
        image = image.convert('RGB')
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if debug:
        print(f"DEBUG: detect_color_cards - Running inference on {device}")
    
    # Run inference
    try:
        results = model.predict(
            source=image,
            imgsz=(960, 640),
            conf=0.25,
            iou=0.45,
            verbose=False,
            device=device
        )
    except Exception as e:
        raise RuntimeError(f"YOLOv8 inference failed: {e}") from e
    
    # Extract detections
    detections = []
    
    for result in results:
        if result.obb is not None:
            for i, (cls, conf) in enumerate(zip(result.obb.cls, result.obb.conf)):
                # Get 4 corner points in pixel coordinates
                points = result.obb.xyxyxyxy[i].cpu().numpy()  # Shape: (4, 2)
                
                # Convert to list of [x, y] pairs
                coordinates = points.tolist()
                
                # Get class name
                class_id = int(cls.item())
                class_name = model.names[class_id]
                
                detection = {
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': float(conf.item()),
                    'coordinates': coordinates
                }
                detections.append(detection)
                
                if debug:
                    print(
                        f"DEBUG: detect_color_cards - Detected {class_name} "
                        f"(ID: {class_id}, confidence: {conf.item():.4f})"
                    )
    
    if debug:
        print(f"DEBUG: detect_color_cards - Found {len(detections)} color card(s)")
    
    return detections


def _calculate_background_color(img_array: np.ndarray,
                                card_coordinates: List[Dict[str, Any]],
                                debug: bool = False) -> Tuple[int, int, int]:
    """Calculate average background color by sampling areas outside detected cards.
    
    Analyzes the image to determine the dominant background color by creating
    masks for detected cards, dilating them to avoid edge effects, and then
    sampling the remaining background pixels. Falls back to corner sampling
    if insufficient background area is available.
    
    Args:
        img_array: Input image as numpy array in RGB format with shape (height, width, 3).
        card_coordinates: List of detection dictionaries from detect_color_cards()
            to exclude from background sampling.
        debug: If True, print debug information about sampling process.
            Defaults to False.
    
    Returns:
        Average background color as RGB tuple (R, G, B) with uint8 values.
    """
    h, w = img_array.shape[:2]
    
    # Create a mask of all detected card areas
    card_mask = np.zeros((h, w), dtype=np.uint8)
    for card in card_coordinates:
        coords = np.array(card['coordinates'], dtype=np.int32)
        cv2.fillPoly(card_mask, [coords], 255)
    
    # Dilate the mask slightly to avoid sampling too close to card edges
    kernel = np.ones((20, 20), np.uint8)
    card_mask_dilated = cv2.dilate(card_mask, kernel, iterations=1)
    
    # Get background pixels (areas not covered by cards)
    background_mask = cv2.bitwise_not(card_mask_dilated)
    
    # Sample background pixels
    background_pixels = img_array[background_mask > 0]
    
    if len(background_pixels) == 0:
        # Fallback: sample from image corners if no background pixels found
        corner_size = min(h, w) // 10  # Sample from corners
        corners = [
            img_array[:corner_size, :corner_size].reshape(-1, 3),  # Top-left
            img_array[:corner_size, -corner_size:].reshape(-1, 3),  # Top-right
            img_array[-corner_size:, :corner_size].reshape(-1, 3),  # Bottom-left
            img_array[-corner_size:, -corner_size:].reshape(-1, 3)  # Bottom-right
        ]
        background_pixels = np.vstack(corners)
        if debug:
            print("DEBUG: _calculate_background_color - Using corner sampling fallback")
    
    # Calculate average color (RGB format)
    avg_color = np.mean(background_pixels, axis=0).astype(np.uint8)
    if debug:
        print(f"DEBUG: _calculate_background_color - Background color (RGB): {avg_color.tolist()}")
    
    return tuple(avg_color.tolist())


def _crop_out_card_regions(img_array: np.ndarray,
                           card_coordinates: List[Dict[str, Any]],
                           background_color: Tuple[int, int, int],
                           debug: bool = False) -> np.ndarray:
    """Crop out (remove) card regions from image by filling them with background color.
    
    Fills the polygon areas of detected color cards with the specified
    background color, effectively removing them from the image while
    maintaining the original image dimensions.
    
    Args:
        img_array: Input image as numpy array in RGB format with shape (height, width, 3).
        card_coordinates: List of detection dictionaries from detect_color_cards().
        background_color: RGB tuple (R, G, B) to fill card regions with.
        debug: If True, print debug information. Defaults to False.
    
    Returns:
        Modified image array with card regions filled with background color.
    """
    img_modified = img_array.copy()
    
    for i, card in enumerate(card_coordinates):
        coords = np.array(card['coordinates'], dtype=np.int32)
        
        # Fill card polygon with background color
        cv2.fillPoly(img_modified, [coords], background_color)
        
        if debug:
            print(f"DEBUG: _crop_out_card_regions - Cropped out card {i+1} ({card['class']})")
    
    return img_modified


def _post_process_mask(binary_mask: np.ndarray,
                       min_component_size_pct: float = 0.000025,
                       max_hole_size_pct: float = 0.000029,
                       edge_tolerance_pct: float = 0.029283,
                       debug: bool = False) -> np.ndarray:
    """Post-process binary mask to clean up artifacts.
    
    Applies several cleaning operations to the binary mask:
    1. Removes edge-touching components (within tolerance)
    2. Removes small isolated components (lone pixels or very small groups)
    3. Fills small holes (small black regions surrounded by white)
    
    All size parameters are specified as percentages of image dimensions to
    work with dynamic resolutions. Values are based on a 2048x1366 reference image:
    - min_component_size: 70 pixels (0.000025 = 0.0025% of total pixels)
    - max_hole_size: 80 pixels (0.000029 = 0.0029% of total pixels)
    - edge_tolerance: 40 pixels (0.029283 = 2.93% of smaller dimension)
    
    Args:
        binary_mask: Binary mask as numpy array with shape (height, width)
            and dtype uint8. Values are 0 (background) or 1 (foreground).
        min_component_size_pct: Minimum component size as decimal fraction of total
            image pixels (e.g., 0.000025 = 0.0025%). Components smaller than this
            will be removed.
        max_hole_size_pct: Maximum hole size as decimal fraction of total image pixels
            (e.g., 0.000029 = 0.0029%). Holes larger than this will remain.
        edge_tolerance_pct: Edge tolerance as decimal fraction of smaller image dimension
            (e.g., 0.029283 = 2.93%). Components within this distance from any edge
            will be removed.
        debug: If True, print debug information about processing.
            Defaults to False.
    
    Returns:
        Post-processed binary mask with same shape and dtype as input.
    """
    h, w = binary_mask.shape
    total_pixels = h * w
    smaller_dimension = min(h, w)
    
    # Calculate pixel values from percentages
    min_component_size = int(total_pixels * min_component_size_pct)
    max_hole_size = int(total_pixels * max_hole_size_pct)
    edge_tolerance = int(smaller_dimension * edge_tolerance_pct)
    
    # Ensure minimum values of 1 pixel
    min_component_size = max(1, min_component_size)
    max_hole_size = max(1, max_hole_size)
    edge_tolerance = max(1, edge_tolerance)
    
    if debug:
        print(
            f"DEBUG: _post_process_mask - Calculated thresholds: "
            f"min_component={min_component_size}px, max_hole={max_hole_size}px, "
            f"edge_tolerance={edge_tolerance}px"
        )
    
    processed_mask = binary_mask.copy()
    
    # Step 1: Remove edge-touching components
    if debug:
        print("DEBUG: _post_process_mask - Removing edge-touching components")
    
    num_labels, labels = cv2.connectedComponents(processed_mask, connectivity=8)
    kept_mask = np.zeros_like(processed_mask, dtype=np.uint8)
    
    for label_id in range(1, num_labels):
        comp = (labels == label_id)
        
        # Check if component touches or is near edge (within tolerance)
        touches_edge = (
            comp[:edge_tolerance, :].any() or  # Top edge
            comp[-edge_tolerance:, :].any() or  # Bottom edge
            comp[:, :edge_tolerance].any() or  # Left edge
            comp[:, -edge_tolerance:].any()  # Right edge
        )
        
        if touches_edge:
            if debug:
                comp_size = np.sum(comp)
                print(f"DEBUG: _post_process_mask - Removing edge-touching component {label_id} (size: {comp_size})")
            continue
        
        # Step 2: Remove small isolated components
        comp_size = np.sum(comp)
        if comp_size < min_component_size:
            if debug:
                print(f"DEBUG: _post_process_mask - Removing small component {label_id} (size: {comp_size})")
            continue
        
        kept_mask[comp] = 1
    
    processed_mask = kept_mask
    
    if debug:
        unique_kept_labels = np.unique(labels[processed_mask > 0])
        # Remove background label 0 if present
        kept_components = len(unique_kept_labels[unique_kept_labels > 0])
        removed_components = (num_labels - 1) - kept_components
        print(f"DEBUG: _post_process_mask - Removed {removed_components} edge-touching/small components, kept {kept_components}")
    
    # Step 3: Fill small holes
    if debug:
        print("DEBUG: _post_process_mask - Filling small holes")
    
    # Invert mask to find holes (black regions in white)
    inverted_mask = 1 - processed_mask
    num_holes, hole_labels = cv2.connectedComponents(inverted_mask, connectivity=8)
    
    # Fill small holes
    for hole_id in range(1, num_holes):
        hole = (hole_labels == hole_id)
        hole_size = np.sum(hole)
        
        # Only fill holes that are small and completely surrounded by foreground
        # Check if hole is not touching edge
        hole_touches_edge = (
            hole[0, :].any() or hole[-1, :].any() or
            hole[:, 0].any() or hole[:, -1].any()
        )
        
        if not hole_touches_edge and hole_size <= max_hole_size:
            if debug:
                print(f"DEBUG: _post_process_mask - Filling hole {hole_id} (size: {hole_size})")
            processed_mask[hole] = 1
    
    if debug:
        final_foreground = np.sum(processed_mask == 1)
        print(f"DEBUG: _post_process_mask - Final foreground pixels: {final_foreground}")
    
    return processed_mask


def remove_background_mask(image: Image.Image,
                          card_coordinates: Optional[List[Dict[str, Any]]] = None,
                          debug: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Image.Image, Image.Image, Image.Image]]:
    """Generate binary mask using RMBG-1.4 background removal with two-pass processing.
    
    Performs background removal on a PIL image using the RMBG-1.4 model in two passes:
    1. First pass: Process the input image (with cards cropped out if provided)
    2. Second pass: Process the first result on a white background
    
    The binary mask is generated from the second pass result, which typically provides
    better quality. The mask is inverted so that foreground objects are marked as 1
    and background as 0.
    
    If card coordinates are provided, the card regions are cropped out (filled
    with background color) from the image before the first pass, and removed from
    the first pass result before the second pass.
    
    Args:
        image: Input PIL Image to process. Must be in RGB or RGBA format.
        card_coordinates: Optional list of detection dictionaries from
            detect_color_cards(). If provided, card polygon areas will be
            cropped out (filled with background color) before the first RMBG pass,
            and removed from the first pass result before the second pass.
        debug: If True, print debug information during processing and return
            additional debug images. Defaults to False.
    
    Returns:
        If debug=False: Binary mask as numpy array with shape (height, width)
            and dtype uint8. Values are 0 (background) or 1 (foreground).
            The mask has the same dimensions as the original input image.
            The mask is generated from the second RMBG pass (white background).
        If debug=True: Tuple of (binary_mask, rmbg_image, rmbg_white_bg_image) where:
            - binary_mask: The inverted mask from the second RMBG pass in original
                image dimensions
            - rmbg_image: The PIL Image result from the first RMBG pass
            - rmbg_white_bg_image: The first RMBG result layered on white background
                (input to the second RMBG pass)
    
    Raises:
        ImportError: If transformers library is not installed.
        RuntimeError: If RMBG pipeline initialization or processing fails.
        TypeError: If input is not a PIL Image.
    """
    # Validate input
    if not isinstance(image, Image.Image):
        raise TypeError(f"Input must be a PIL Image, got {type(image)}")
    
    # Store original dimensions
    original_size = (image.width, image.height)
    
    # Convert image to RGB if needed
    if image.mode != 'RGB':
        if debug:
            print(f"DEBUG: remove_background_mask - Converting image from {image.mode} to RGB")
        image = image.convert('RGB')
    
    # Load RMBG pipeline (cached at module level)
    rmbg_pipe = _load_rmbg_pipeline(debug=debug)
    
    # Prepare image for RMBG processing
    rmbg_input = image
    
    # If card coordinates provided, crop out card regions
    if card_coordinates is not None and len(card_coordinates) > 0:
        if debug:
            print(f"DEBUG: remove_background_mask - Cropping out {len(card_coordinates)} card region(s)")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate background color from areas outside cards
        background_color = _calculate_background_color(
            img_array,
            card_coordinates,
            debug=debug
        )
        
        # Crop out card regions by filling with background color
        img_cropped = _crop_out_card_regions(
            img_array,
            card_coordinates,
            background_color,
            debug=debug
        )
        
        rmbg_input = Image.fromarray(img_cropped)
        
        if debug:
            print("DEBUG: remove_background_mask - Card regions cropped out, image ready for RMBG")
    
    # Run RMBG on the prepared image
    if debug:
        print("DEBUG: remove_background_mask - Running RMBG-1.4 segmentation")
    
    try:
        rmbg_results = rmbg_pipe(rmbg_input)
    except Exception as e:
        raise RuntimeError(f"RMBG processing failed: {e}") from e
    
    # Extract mask and image from RMBG results
    # RMBG can return different formats: list of dicts, single dict, or PIL Image
    mask_image = None
    rmbg_image_result = None
    
    if isinstance(rmbg_results, list) and len(rmbg_results) > 0:
        # Multiple segments: use the first one (RMBG's primary result)
        if debug:
            print(f"DEBUG: remove_background_mask - Found {len(rmbg_results)} RMBG segments, using first")
        first_result = rmbg_results[0]
        mask_image = first_result.get('mask', first_result.get('image', first_result))
        rmbg_image_result = first_result.get('image', first_result)
    elif isinstance(rmbg_results, dict):
        mask_image = rmbg_results.get('mask', rmbg_results.get('image'))
        rmbg_image_result = rmbg_results.get('image', rmbg_results)
    else:
        # Single result or PIL Image
        mask_image = rmbg_results
        rmbg_image_result = rmbg_results
    
    # Convert RMBG image result to PIL Image if needed
    if rmbg_image_result is not None and not isinstance(rmbg_image_result, Image.Image):
        if isinstance(rmbg_image_result, np.ndarray):
            rmbg_image_result = Image.fromarray(rmbg_image_result)
        else:
            rmbg_image_result = None
    
    # Convert mask to numpy array if needed
    if mask_image is None:
        raise RuntimeError("RMBG did not return a valid mask")
    
    if isinstance(mask_image, Image.Image):
        mask_array = np.array(mask_image.convert('L'), dtype=np.uint8)
    elif isinstance(mask_image, np.ndarray):
        mask_array = mask_image.astype(np.uint8)
        if len(mask_array.shape) == 3:
            # Convert to grayscale if needed
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    else:
        raise RuntimeError(f"Unexpected mask format: {type(mask_image)}")
    
    # Ensure mask matches RMBG input image dimensions first
    # PIL size is (width, height), numpy arrays are (height, width)
    rmbg_input_w, rmbg_input_h = rmbg_input.size
    
    if mask_array.shape[:2] != (rmbg_input_h, rmbg_input_w):
        if debug:
            print(
                f"DEBUG: remove_background_mask - "
                f"Resizing mask from {mask_array.shape[:2]} to {(rmbg_input_h, rmbg_input_w)}"
            )
        mask_array = cv2.resize(
            mask_array,
            (rmbg_input_w, rmbg_input_h),  # cv2.resize expects (width, height)
            interpolation=cv2.INTER_NEAREST
        )
    
    # Generate white background version and run RMBG again (always, not just in debug)
    rmbg_white_bg_image = None
    rmbg_white_bg_result = None
    rmbg_white_bg_mask = None
    
    if rmbg_image_result is not None:
        if debug:
            print("DEBUG: remove_background_mask - Creating white background version of RMBG result")
        
        # Remove card regions from first RMBG result if card coordinates were provided
        rmbg_cleaned = rmbg_image_result
        if card_coordinates is not None and len(card_coordinates) > 0:
                if debug:
                    print(f"DEBUG: remove_background_mask - Removing card regions from first RMBG result (2% tolerance)")
                
                # Convert to RGBA if needed
                if rmbg_cleaned.mode != 'RGBA':
                    rmbg_cleaned = rmbg_cleaned.convert('RGBA')
                
                # Get image dimensions
                img_w, img_h = rmbg_cleaned.size
                smaller_dimension = min(img_w, img_h)
                
                # Calculate 2% tolerance in pixels
                tolerance_pct = 0.02
                tolerance_pixels = max(1, int(smaller_dimension * tolerance_pct))
                
                if debug:
                    print(f"DEBUG: remove_background_mask - Card removal tolerance: {tolerance_pixels}px (2% of {smaller_dimension}px)")
                
                # Convert to numpy array for processing
                rmbg_array = np.array(rmbg_cleaned)
                
                # Create mask of card regions with expansion
                card_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for card in card_coordinates:
                    coords = np.array(card['coordinates'], dtype=np.int32)
                    # Card coordinates are in original image dimensions
                    # Need to scale if rmbg_image_result size differs from original
                    if (img_w, img_h) != original_size:
                        # Scale coordinates to match rmbg_image_result dimensions
                        scale_x = img_w / original_size[0]
                        scale_y = img_h / original_size[1]
                        coords_scaled = coords.copy()
                        coords_scaled[:, 0] = (coords_scaled[:, 0] * scale_x).astype(np.int32)
                        coords_scaled[:, 1] = (coords_scaled[:, 1] * scale_y).astype(np.int32)
                        cv2.fillPoly(card_mask, [coords_scaled], 255)
                    else:
                        cv2.fillPoly(card_mask, [coords], 255)
                
                # Expand card mask by tolerance pixels
                if tolerance_pixels > 0:
                    kernel_size = 2 * tolerance_pixels + 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    card_mask = cv2.dilate(card_mask, kernel, iterations=1)
                
                # Remove card regions by setting alpha to 0
                rmbg_array[:, :, 3] = np.where(card_mask > 0, 0, rmbg_array[:, :, 3]).astype(np.uint8)
                
                # Convert back to PIL Image
                rmbg_cleaned = Image.fromarray(rmbg_array, mode='RGBA')
                
                if debug:
                    removed_pixels = np.sum(card_mask > 0)
                    print(f"DEBUG: remove_background_mask - Removed {removed_pixels} pixels from first RMBG result")
        
        # Create white background
        white_bg = Image.new('RGB', rmbg_cleaned.size, (255, 255, 255))
        
        # Layer the cleaned RMBG result on white background
        # If it's RGBA, use alpha channel; if RGB, paste directly
        if rmbg_cleaned.mode == 'RGBA':
            white_bg.paste(rmbg_cleaned, (0, 0), rmbg_cleaned)
        else:
            # Convert to RGB if needed and paste directly
            rmbg_rgb = rmbg_cleaned.convert('RGB')
            white_bg.paste(rmbg_rgb, (0, 0))
        
        rmbg_white_bg_image = white_bg
        
        if debug:
            print("DEBUG: remove_background_mask - Running RMBG again on white background image")
        
        # Run RMBG again on white background image
        try:
            rmbg_white_results = rmbg_pipe(rmbg_white_bg_image)
            
            # Extract result image and mask (similar to first RMBG pass)
            mask_white_image = None
            if isinstance(rmbg_white_results, list) and len(rmbg_white_results) > 0:
                first_result = rmbg_white_results[0]
                mask_white_image = first_result.get('mask', first_result.get('image', first_result))
                rmbg_white_bg_result = first_result.get('image', first_result)
            elif isinstance(rmbg_white_results, dict):
                mask_white_image = rmbg_white_results.get('mask', rmbg_white_results.get('image'))
                rmbg_white_bg_result = rmbg_white_results.get('image', rmbg_white_results)
            else:
                mask_white_image = rmbg_white_results
                rmbg_white_bg_result = rmbg_white_results
            
            # Convert to PIL Image if needed
            if rmbg_white_bg_result is not None and not isinstance(rmbg_white_bg_result, Image.Image):
                if isinstance(rmbg_white_bg_result, np.ndarray):
                    rmbg_white_bg_result = Image.fromarray(rmbg_white_bg_result)
                else:
                    rmbg_white_bg_result = None
            
            # Generate mask from white background RMBG result
            rmbg_white_bg_mask = None
            if rmbg_white_bg_result is not None:
                if debug:
                    print("DEBUG: remove_background_mask - Generating mask from white background RMBG result")
                
                # Extract mask from result image
                # Prefer explicit mask if available, otherwise extract from image
                if mask_white_image is not None and mask_white_image is not rmbg_white_bg_result:
                    # Use explicit mask
                    if isinstance(mask_white_image, Image.Image):
                        mask_white_array = np.array(mask_white_image.convert('L'), dtype=np.uint8)
                    elif isinstance(mask_white_image, np.ndarray):
                        mask_white_array = mask_white_image.astype(np.uint8)
                        if len(mask_white_array.shape) == 3:
                            mask_white_array = cv2.cvtColor(mask_white_array, cv2.COLOR_RGB2GRAY)
                    else:
                        mask_white_array = None
                else:
                    # Extract from result image - prefer alpha channel if available
                    if rmbg_white_bg_result.mode == 'RGBA':
                        # Use alpha channel as mask
                        mask_white_array = np.array(rmbg_white_bg_result.split()[3], dtype=np.uint8)
                    else:
                        # Convert RGB result to grayscale for mask
                        mask_white_array = np.array(rmbg_white_bg_result.convert('L'), dtype=np.uint8)
                
                if mask_white_array is not None:
                    # Get white background image dimensions
                    white_bg_w, white_bg_h = rmbg_white_bg_image.size
                    
                    # Ensure mask matches white background image dimensions
                    if mask_white_array.shape[:2] != (white_bg_h, white_bg_w):
                        if debug:
                            print(
                                f"DEBUG: remove_background_mask - "
                                f"Resizing white BG mask from {mask_white_array.shape[:2]} to {(white_bg_h, white_bg_w)}"
                            )
                        mask_white_array = cv2.resize(
                            mask_white_array,
                            (white_bg_w, white_bg_h),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # Convert to binary mask (0 or 1)
                    # RMBG returns white (255) for foreground, black (0) for background
                    # So >128 = foreground (1), <=128 = background (0)
                    binary_mask_white = (mask_white_array > 128).astype(np.uint8)
                    
                    # Map mask to original image dimensions
                    if (white_bg_w, white_bg_h) != original_size:
                        if debug:
                            print(
                                f"DEBUG: remove_background_mask - "
                                f"Mapping white BG mask from {(white_bg_w, white_bg_h)} to original {original_size}"
                            )
                        binary_mask_white = cv2.resize(
                            binary_mask_white,
                            original_size,
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    rmbg_white_bg_mask = binary_mask_white
            
            if debug:
                print("DEBUG: remove_background_mask - Second RMBG pass completed")
        except Exception as e:
            if debug:
                print(f"DEBUG: remove_background_mask - Second RMBG pass failed: {e}")
            rmbg_white_bg_result = None
            rmbg_white_bg_mask = None
    
    # Use mask from second RMBG pass (white background) as the main mask
    if rmbg_white_bg_mask is not None:
        binary_mask = rmbg_white_bg_mask
        if debug:
            foreground_pixels = np.sum(binary_mask == 1)
            total_pixels = binary_mask.size
            print(
                f"DEBUG: remove_background_mask - "
                f"Using second pass mask - Foreground: {foreground_pixels}/{total_pixels} pixels ({100*foreground_pixels/total_pixels:.1f}%)"
            )
    else:
        # Fallback to first pass mask if second pass failed
        if debug:
            print("DEBUG: remove_background_mask - Second pass mask not available, using first pass mask")
        # Convert first pass mask (already processed above)
        # RMBG returns white (255) for foreground, black (0) for background
        # So >128 = foreground (1), <=128 = background (0)
        binary_mask = (mask_array > 128).astype(np.uint8)
        if (rmbg_input_w, rmbg_input_h) != original_size:
            binary_mask = cv2.resize(
                binary_mask,
                original_size,
                interpolation=cv2.INTER_NEAREST
            )
    
    if debug:
        if rmbg_image_result is not None and rmbg_white_bg_image is not None:
            return binary_mask, rmbg_image_result, rmbg_white_bg_image
    
    return binary_mask


def generate_swatch(pil_image: Image.Image,
                    swatch_size: Tuple[int, int] = (1000, 500),
                    target_dpi: int = 1200,
                    pp_cm_original: Optional[float] = None,
                    pp_cm_target: Optional[float] = None,
                    coarse_angle_step: int = 15,
                    fine_angle_step: int = 1,
                    debug: bool = False) -> Image.Image:
    """Generate a rotated coverage-optimized swatch from a transparent image.

    Given an RGBA image with transparency, this function finds a swatch (default
    1000x500 pixels) centered on the object's (non-transparent area) centroid,
    allowing rotation to maximize the number of foreground pixels captured.

    Strategy:
      1. Ensure RGBA & build binary mask using alpha > 0 (non-transparent pixels).
      2. If object smaller than desired swatch, uniformly upscale image so the
         object's bounding box exceeds the swatch with a modest margin.
      3. Perform a coarse rotation search (0..179 degrees) at
         ``coarse_angle_step`` increments. Track coverage (foreground pixels
         inside the candidate swatch / swatch area).
      4. Refine around best coarse angle using ``fine_angle_step`` steps.
      5. Extract the crop (padding with transparency if crop extends beyond
         image bounds). DPI metadata is set to ``target_dpi``.

    Args:
        pil_image: Input PIL image (preferably RGBA) with transparent background.
        swatch_size: Desired (width, height) of output swatch in pixels.
            Defaults to (1000, 500).
        target_dpi: DPI metadata to tag on the output image. Defaults to 1200.
        pp_cm_original: Original pixels-per-centimeter of the image. If provided
            along with pp_cm_target (>0), the image is uniformly rescaled by
            (pp_cm_target / pp_cm_original) before swatch generation.
        pp_cm_target: Target pixels-per-centimeter to achieve prior to swatch
            rotation/cropping. Ignored unless pp_cm_original also provided.
        coarse_angle_step: Degrees per step for coarse search. Defaults to 15.
        fine_angle_step: Degrees per step for fine search window. Defaults to 1.
        debug: Print diagnostic information if True. Defaults to False.

    Returns:
        PIL Image (RGBA) sized exactly ``swatch_size``.

    Raises:
        ValueError: If no non-transparent pixels are found or swatch dimensions invalid.
    """
    # Validate swatch dimensions
    sw_w, sw_h = swatch_size
    if sw_w <= 0 or sw_h <= 0:
        raise ValueError("generate_swatch: swatch_size must contain positive dimensions")

    # Ensure RGBA
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")
        if debug:
            print("DEBUG: generate_swatch - Converted input to RGBA")

    # Physical scaling to match target pixels-per-centimeter if specified
    if (pp_cm_original is not None and pp_cm_target is not None
            and pp_cm_original > 0 and pp_cm_target > 0):
        scale_pp = pp_cm_target / pp_cm_original
        if abs(scale_pp - 1.0) > 1e-6:
            new_w = max(1, int(round(pil_image.width * scale_pp)))
            new_h = max(1, int(round(pil_image.height * scale_pp)))
            if debug:
                print(
                    f"DEBUG: generate_swatch - Scaling by pp_cm ratio {scale_pp:.4f} "
                    f"to {new_w}x{new_h} (from {pil_image.width}x{pil_image.height})"
                )
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    arr = np.array(pil_image)
    if arr.shape[2] < 4:
        raise ValueError("generate_swatch: input image must have an alpha channel")
    alpha = arr[:, :, 3]
    # Use alpha > 0 instead of alpha_threshold since background removal already handles this
    mask = (alpha > 0).astype(np.uint8)

    if mask.sum() == 0:
        raise ValueError("generate_swatch: no non-transparent pixels detected")

    # Object bounding box
    ys, xs = np.nonzero(mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1

    if debug:
        print(f"DEBUG: generate_swatch - Original bbox (x={x_min}, y={y_min}, w={bbox_w}, h={bbox_h})")

    # Upscale if object smaller than swatch in either dimension
    scale_factor = 1.0
    if bbox_w < sw_w or bbox_h < sw_h:
        scale_factor = max(sw_w / max(bbox_w, 1), sw_h / max(bbox_h, 1)) * 1.05  # add margin
        new_w = max(1, int(round(pil_image.width * scale_factor)))
        new_h = max(1, int(round(pil_image.height * scale_factor)))
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        if debug:
            print(f"DEBUG: generate_swatch - Upscaled by {scale_factor:.2f} to {new_w}x{new_h}")
        arr = np.array(pil_image)
        alpha = arr[:, :, 3]
        mask = (alpha > 0).astype(np.uint8)

        # Recompute bounding box after upscaling
        ys, xs = np.nonzero(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

    # Centroid for horizontal (0°) attempt
    ys, xs = np.nonzero(mask)
    cx = xs.mean()
    cy = ys.mean()
    left0 = int(round(cx - sw_w / 2))
    top0 = int(round(cy - sw_h / 2))
    right0 = left0 + sw_w
    bottom0 = top0 + sw_h

    # Check if the object's bounding box fits entirely inside this 0° swatch
    if (left0 <= x_min and right0 >= x_max + 1 and top0 <= y_min and bottom0 >= y_max + 1):
        # Prepare final swatch canvas (transparent)
        out = Image.new("RGBA", (sw_w, sw_h), (0, 0, 0, 0))
        img_w0, img_h0 = pil_image.size
        src_left = max(left0, 0)
        src_top = max(top0, 0)
        src_right = min(right0, img_w0)
        src_bottom = min(bottom0, img_h0)
        if src_right > src_left and src_bottom > src_top:
            crop0 = pil_image.crop((src_left, src_top, src_right, src_bottom))
            dst_x0 = src_left - left0
            dst_y0 = src_top - top0
            out.paste(crop0, (dst_x0, dst_y0))
        if debug:
            print("DEBUG: generate_swatch - Early exit at 0° (horizontal) orientation; full object covered")
        out.info["dpi"] = (target_dpi, target_dpi)
        return out

    # Helper to evaluate a rotation angle
    def evaluate_angle(angle_deg: float):
        # Rotate image & mask
        r_img = pil_image.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
        r_mask_img = Image.fromarray(mask * 255).rotate(angle_deg, expand=True, resample=Image.NEAREST)
        r_mask = np.array(r_mask_img) > 0
        ry, rx = np.nonzero(r_mask)
        if len(ry) == 0:
            return 0.0, (0, 0, 0, 0), r_img, r_mask
        rcx = rx.mean()
        rcy = ry.mean()
        # Center crop box
        left = int(round(rcx - sw_w / 2))
        top = int(round(rcy - sw_h / 2))
        right = left + sw_w
        bottom = top + sw_h
        # Intersection for coverage
        img_w, img_h = r_img.size
        crop_left = max(left, 0)
        crop_top = max(top, 0)
        crop_right = min(right, img_w)
        crop_bottom = min(bottom, img_h)
        if crop_right <= crop_left or crop_bottom <= crop_top:
            return 0.0, (left, top, right, bottom), r_img, r_mask
        mask_crop = r_mask[crop_top:crop_bottom, crop_left:crop_right]
        coverage = mask_crop.sum() / float(sw_w * sw_h)  # Normalized by full swatch area
        return coverage, (left, top, right, bottom), r_img, r_mask

    # Coarse rotation search
    best_coverage = -1.0
    best_tuple = None  # (coverage, angle, crop_box, rotated_img, rotated_mask)
    for angle in range(0, 180, max(1, coarse_angle_step)):
        coverage, crop_box, r_img, r_mask = evaluate_angle(angle)
        if coverage > best_coverage:
            best_coverage = coverage
            best_tuple = (coverage, angle, crop_box, r_img, r_mask)
        if debug:
            print(f"DEBUG: generate_swatch - Coarse angle {angle:3d}° coverage={coverage:.5f}")
        # Early terminate coarse search if perfect (or near-perfect) coverage achieved
        if best_coverage >= 0.99999:
            if debug:
                print("DEBUG: generate_swatch - Early break from coarse search (coverage >= 0.99999)")
            break

    if best_tuple is None:
        # Fallback: empty transparent swatch
        if debug:
            print("DEBUG: generate_swatch - No coverage found, returning empty swatch")
        empty = Image.new("RGBA", (sw_w, sw_h), (0, 0, 0, 0))
        empty.info["dpi"] = (target_dpi, target_dpi)
        return empty

    _, coarse_angle, _, _, _ = best_tuple

    # Fine search around coarse angle
    fine_start = max(0, int(coarse_angle - coarse_angle_step))
    fine_end = min(179, int(coarse_angle + coarse_angle_step))
    if best_coverage < 0.99999:  # Only refine if not already perfect
        for angle in range(fine_start, fine_end + 1, max(1, fine_angle_step)):
            coverage, crop_box, r_img, r_mask = evaluate_angle(angle)
            if coverage > best_coverage:
                best_coverage = coverage
                best_tuple = (coverage, angle, crop_box, r_img, r_mask)
            if debug and angle % 5 == 0:
                print(f"DEBUG: generate_swatch - Fine angle {angle:3d}° coverage={coverage:.5f}")
            if best_coverage >= 0.99999:
                if debug:
                    print("DEBUG: generate_swatch - Early break from fine search (coverage >= 0.99999)")
                break

    best_coverage, best_angle, crop_box, best_img, _ = best_tuple
    if debug:
        print(f"DEBUG: generate_swatch - Selected angle {best_angle}° with coverage {best_coverage:.5f}")

    # Prepare final swatch canvas (transparent)
    out = Image.new("RGBA", (sw_w, sw_h), (0, 0, 0, 0))
    left, top, right, bottom = crop_box
    img_w, img_h = best_img.size
    # Source coords intersection
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, img_w)
    src_bottom = min(bottom, img_h)
    if src_right > src_left and src_bottom > src_top:
        crop = best_img.crop((src_left, src_top, src_right, src_bottom))
        dst_x = src_left - left  # offset relative to swatch
        dst_y = src_top - top
        out.paste(crop, (dst_x, dst_y))
    else:
        if debug:
            print("DEBUG: generate_swatch - Computed crop outside bounds; returning blank swatch")

    # Attach DPI metadata (persisted on save)
    out.info["dpi"] = (target_dpi, target_dpi)
    return out
