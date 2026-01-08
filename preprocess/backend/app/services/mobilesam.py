"""MobileSAM service for point-based segmentation."""

import sys
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Global model cache
_mobile_sam_model = None
_mobile_sam_predictor = None
_model_loaded = False


def _get_model_path() -> Path:
    """Get path to MobileSAM model weights."""
    # Try weights folder in backend first
    backend_weights = Path(__file__).parent.parent.parent / "weights" / "mobile_sam.pt"
    if backend_weights.exists():
        return backend_weights
    
    # Try project root weights folder
    project_root = Path(__file__).parent.parent.parent.parent.parent
    project_weights = project_root / "weights" / "mobile_sam.pt"
    if project_weights.exists():
        return project_weights
    
    # Default MobileSAM installation location
    try:
        import mobile_sam
        mobile_sam_path = Path(mobile_sam.__file__).parent.parent / "weights" / "mobile_sam.pt"
        if mobile_sam_path.exists():
            return mobile_sam_path
    except ImportError:
        pass
    
    # Fallback: try to download or use default
    return backend_weights


def _load_mobile_sam():
    """Load MobileSAM model with lazy initialization."""
    global _mobile_sam_model, _mobile_sam_predictor, _model_loaded
    
    if _model_loaded:
        return _mobile_sam_model, _mobile_sam_predictor
    
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        
        model_type = "vit_t"
        checkpoint_path = _get_model_path()
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found at {checkpoint_path}. "
                "Please download mobile_sam.pt and place it in the weights folder."
            )
        
        logger.info(f"Loading MobileSAM model from {checkpoint_path}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model
        _mobile_sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        _mobile_sam_model.to(device=device)
        _mobile_sam_model.eval()
        
        # Create predictor
        _mobile_sam_predictor = SamPredictor(_mobile_sam_model)
        
        _model_loaded = True
        logger.info("MobileSAM model loaded successfully")
        
        return _mobile_sam_model, _mobile_sam_predictor
        
    except ImportError as e:
        logger.error(f"Failed to import MobileSAM: {e}")
        raise ImportError(
            "MobileSAM is not installed. Install it with: "
            "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        ) from e
    except Exception as e:
        logger.error(f"Failed to load MobileSAM model: {e}", exc_info=True)
        raise


def segment_from_point(
    image_path: str,
    x: int,
    y: int,
    point_label: int = 1
) -> Optional[np.ndarray]:
    """Generate mask segment from a point prompt using MobileSAM.
    
    Args:
        image_path: Path to image file
        x: X coordinate of point (in image space)
        y: Y coordinate of point (in image space)
        point_label: 1 for foreground (positive point), 0 for background (negative point)
        
    Returns:
        Binary mask as numpy array (0/1), where 1 is foreground, or None if failed
    """
    try:
        # Load model
        model, predictor = _load_mobile_sam()
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Set image in predictor
        predictor.set_image(image_array)
        
        # Prepare point prompt
        # MobileSAM expects points as numpy array of shape (N, 2) where N is number of points
        # and labels as numpy array of shape (N,) where 1 = foreground, 0 = background
        input_point = np.array([[x, y]])
        input_label = np.array([point_label])
        
        # Predict mask
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        # Get the best mask (only one mask when multimask_output=False)
        mask = masks[0]
        
        # Convert to 0/1 format (MobileSAM returns True/False)
        mask_binary = mask.astype(np.uint8)
        
        logger.debug(f"MobileSAM segmentation: point=({x}, {y}), mask_size={mask_binary.shape}, "
                    f"foreground_pixels={np.sum(mask_binary)}")
        
        return mask_binary
        
    except Exception as e:
        logger.error(f"MobileSAM segmentation failed for {image_path} at ({x}, {y}): {e}", exc_info=True)
        return None

