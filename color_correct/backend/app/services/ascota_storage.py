"""Service for managing .ascota folder and color_correct.json storage."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Windows hidden attribute constant
FILE_ATTRIBUTE_HIDDEN = 0x02


def set_hidden_attribute_windows(path: Path):
    """Set hidden attribute on Windows.
    
    Args:
        path: Path to file or directory
    """
    try:
        import ctypes
        import platform
        
        if platform.system() == "Windows":
            # Convert to absolute path and use SetFileAttributesW
            abs_path = str(path.absolute())
            ctypes.windll.kernel32.SetFileAttributesW(abs_path, FILE_ATTRIBUTE_HIDDEN)
            logger.debug(f"Set hidden attribute on {path}")
    except Exception as e:
        logger.warning(f"Failed to set hidden attribute on {path}: {e}")


def ensure_ascota_folder(find_path: str) -> Path:
    """Ensure .ascota folder exists in find directory.
    
    Args:
        find_path: Path to find directory (e.g., context_path/finds/individual/{find_number})
        
    Returns:
        Path to .ascota folder
    """
    find_dir = Path(find_path)
    ascota_dir = find_dir / ".ascota"
    
    # Create directory if it doesn't exist
    ascota_dir.mkdir(parents=True, exist_ok=True)
    
    # Set hidden attribute on Windows
    set_hidden_attribute_windows(ascota_dir)
    
    return ascota_dir


def get_color_correct_json_path(find_path: str) -> Path:
    """Get path to color_correct.json file.
    
    Args:
        find_path: Path to find directory (e.g., context_path/finds/individual/{find_number})
        
    Returns:
        Path to color_correct.json
    """
    ascota_dir = ensure_ascota_folder(find_path)
    return ascota_dir / "color_correct.json"


def load_color_correct_json(find_path: str) -> Dict[str, Any]:
    """Load color_correct.json from find directory.
    
    Args:
        find_path: Path to find directory (e.g., context_path/finds/individual/{find_number})
        
    Returns:
        Dictionary with corrections data, or empty dict if file doesn't exist
    """
    json_path = get_color_correct_json_path(find_path)
    
    if not json_path.exists():
        return {"images": {}}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure images key exists
        if "images" not in data:
            data["images"] = {}
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load color_correct.json from {json_path}: {e}", exc_info=True)
        return {"images": {}}


def save_color_correct_json(
    find_path: str,
    corrections_data: Dict[str, Any]
) -> bool:
    """Save color_correct.json to find directory.
    
    Args:
        find_path: Path to find directory (e.g., context_path/finds/individual/{find_number})
        corrections_data: Dictionary with corrections data (format: {"images": {...}})
        
    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = get_color_correct_json_path(find_path)
        
        # Ensure structure
        if "images" not in corrections_data:
            corrections_data = {"images": corrections_data}
        
        # Atomic write: write to temp file, then rename
        temp_file = json_path.with_suffix(".tmp")
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(corrections_data, f, indent=2, default=str)
        
        # Rename to final file
        temp_file.replace(json_path)
        
        logger.info(f"Saved color_correct.json to {json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save color_correct.json to {find_path}: {e}", exc_info=True)
        return False


def get_raw_filename_no_ext(raw_path: Optional[str]) -> Optional[str]:
    """Extract raw filename without extension.
    
    Args:
        raw_path: Path to RAW file
        
    Returns:
        Filename without extension, or None if path is invalid
    """
    if not raw_path:
        return None
    
    return Path(raw_path).stem


def build_corrections_data(
    images: Dict[str, Any],  # Dict of ImageItem
    clusters: list,  # List of Cluster objects
    session  # Session object with individual corrections
) -> Dict[str, Dict[str, Any]]:
    """Build corrections data structure for saving.
    
    Args:
        images: Dict of image_id -> ImageItem
        clusters: List of Cluster objects
        session: Session object with individual corrections
        
    Returns:
        Dictionary keyed by find_path, containing corrections data
    """
    from datetime import datetime
    
    # Group by find (not context)
    find_data: Dict[str, Dict[str, Any]] = {}
    
    # Build cluster lookup
    clusters_by_image: Dict[str, str] = {}
    for cluster in clusters:
        for image_id in cluster.image_ids:
            clusters_by_image[image_id] = cluster.id
    
    # Process each image
    for image_id, image in images.items():
        # Build photos path: context_path/finds/individual/{find_number}/photos
        context_path = Path(image.context_id)
        photos_path = context_path / "finds" / "individual" / image.find_number / "photos"
        
        if str(photos_path) not in find_data:
            find_data[str(photos_path)] = {"images": {}}
        
        # Get raw filename (without extension) as key
        raw_filename = get_raw_filename_no_ext(image.raw_path)
        if not raw_filename:
            continue
        
        # Get overall corrections from cluster
        cluster_id = clusters_by_image.get(image_id)
        overall_params = None
        if cluster_id:
            cluster = next((c for c in clusters if c.id == cluster_id), None)
            if cluster and cluster.correction_params:
                overall_params = cluster.correction_params.dict()
        
        # Get individual corrections
        individual_params = None
        if hasattr(session, 'get_individual_correction'):
            individual_corr = session.get_individual_correction(image_id)
            if individual_corr:
                individual_params = individual_corr.dict()
        
        # Only save if there are corrections
        if overall_params or individual_params:
            find_data[str(photos_path)]["images"][raw_filename] = {
                "overall": overall_params,
                "individual": individual_params,
                "timestamp": datetime.now().isoformat()
            }
    
    # Add metadata about RAW processing settings used
    for photos_path in find_data:
        find_data[photos_path]["metadata"] = {
            "raw_processing": {
                "use_camera_wb": False,
                "no_auto_bright": False,
                "bright": 1.0,
                "output_bps": 16
            },
            "version": "1.0"
        }
    
    return find_data

