"""Service for managing .ascota folder and preprocess.json storage."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np
from PIL import Image

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
    """Ensure .ascota folder exists in find directory and set hidden attribute on Windows.
    
    Args:
        find_path: Path to find directory (e.g., context_path/finds/individual/{find_number})
        
    Returns:
        Path to .ascota folder
    """
    find_dir = Path(find_path)
    ascota_dir = find_dir / ".ascota"
    
    # Create directory if it doesn't exist
    ascota_dir.mkdir(parents=True, exist_ok=True)
    
    # Always set hidden attribute on Windows (even if folder already existed)
    set_hidden_attribute_windows(ascota_dir)
    
    return ascota_dir


def ensure_context_ascota_folder(context_path: str) -> Path:
    """Ensure .ascota folder exists in main context directory.
    
    Args:
        context_path: Path to context directory (e.g., D:/ararat/data/files/N/38/478020/4419550/1)
        
    Returns:
        Path to .ascota folder in context directory
    """
    context_dir = Path(context_path)
    ascota_dir = context_dir / ".ascota"
    
    # Create directory if it doesn't exist
    ascota_dir.mkdir(parents=True, exist_ok=True)
    
    # Set hidden attribute on Windows
    set_hidden_attribute_windows(ascota_dir)
    
    return ascota_dir


def append_context_status(context_path: str, status_data: Dict[str, Any]) -> bool:
    """Append a status entry to the context metadata JSON file.
    
    Args:
        context_path: Path to context directory
        status_data: Dictionary with status information (e.g., {"preprocess_status": True})
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from datetime import datetime
        
        ascota_dir = ensure_context_ascota_folder(context_path)
        metadata_file = ascota_dir / "context_status.json"
        
        # Load existing data if file exists
        existing_data = {"status_history": []}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # Ensure status_history exists
                    if "status_history" not in existing_data:
                        existing_data["status_history"] = []
            except Exception as e:
                logger.warning(f"Failed to load existing context_status.json: {e}, creating new file")
                existing_data = {"status_history": []}
        
        # Create new status entry with timestamp
        timestamp = datetime.now().isoformat()
        status_entry = {
            "timestamp": timestamp,
            **status_data
        }
        
        # Append to history
        existing_data["status_history"].append(status_entry)
        
        # Atomic write: write to temp file, then rename
        temp_file = metadata_file.with_suffix(".tmp")
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, default=str)
        
        # Rename to final file
        temp_file.replace(metadata_file)
        
        logger.info(f"Appended status to {metadata_file}: {status_data}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to append context status to {context_path}: {e}", exc_info=True)
        return False


def delete_existing_context_data(context_path: str) -> int:
    """Delete masks folder and preprocess.json from all finds in a context.
    
    Args:
        context_path: Path to context directory (e.g., D:/ararat/data/files/N/38/478020/4419550/1)
        
    Returns:
        Number of finds that had data deleted
    """
    deleted_count = 0
    context_dir = Path(context_path)
    
    if not context_dir.exists():
        logger.warning(f"Context path does not exist: {context_path}")
        return deleted_count
    
    # Look for finds/individual/*/ directories
    finds_base = context_dir / "finds" / "individual"
    
    if not finds_base.exists():
        logger.debug(f"No finds directory at: {finds_base}")
        return deleted_count
    
    # Iterate through all find directories
    for find_dir in finds_base.iterdir():
        if not find_dir.is_dir():
            continue
        
        find_path = str(find_dir)
        ascota_dir = find_dir / ".ascota"
        
        if not ascota_dir.exists():
            continue
        
        had_data = False
        
        # Delete masks folder
        masks_dir = ascota_dir / "masks"
        if masks_dir.exists() and masks_dir.is_dir():
            try:
                shutil.rmtree(masks_dir)
                logger.info(f"Deleted masks folder from {find_path}")
                had_data = True
            except Exception as e:
                logger.error(f"Failed to delete masks folder from {find_path}: {e}")
        
        # Delete preprocess.json
        preprocess_json = ascota_dir / "preprocess.json"
        if preprocess_json.exists() and preprocess_json.is_file():
            try:
                preprocess_json.unlink()
                logger.info(f"Deleted preprocess.json from {find_path}")
                had_data = True
            except Exception as e:
                logger.error(f"Failed to delete preprocess.json from {find_path}: {e}")
        
        if had_data:
            deleted_count += 1
    
    logger.info(f"Deleted existing context data from {deleted_count} finds in {context_path}")
    return deleted_count


def get_preprocess_json_path(find_path: str) -> Path:
    """Get path to preprocess.json file.
    
    Args:
        find_path: Path to find directory (e.g., context_path/finds/individual/{find_number})
        
    Returns:
        Path to preprocess.json
    """
    ascota_dir = ensure_ascota_folder(find_path)
    return ascota_dir / "preprocess.json"


def load_preprocess_json(find_path: str) -> Dict[str, Any]:
    """Load preprocess.json from find directory.
    
    Args:
        find_path: Path to find directory
        
    Returns:
        Dictionary with preprocess data, or empty dict if file doesn't exist
    """
    json_path = get_preprocess_json_path(find_path)
    
    if not json_path.exists():
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load preprocess.json from {json_path}: {e}", exc_info=True)
        return {}


def load_stage1_data_for_image(find_path: str, image_filename: str) -> Optional[Dict[str, Any]]:
    """Load Stage 1 data for a specific image from preprocess.json.
    
    Args:
        find_path: Path to find directory
        image_filename: Image filename (without extension, e.g., "1")
        
    Returns:
        Dictionary with stage1 data for the image, or None if not found
    """
    data = load_preprocess_json(find_path)
    stage1_data = data.get("stage1", {})
    images_data = stage1_data.get("images", {})
    return images_data.get(image_filename)


def load_stage2_data_for_image(find_path: str, image_filename: str) -> Optional[Dict[str, Any]]:
    """Load Stage 2 data for a specific image from preprocess.json.
    
    Args:
        find_path: Path to find directory
        image_filename: Image filename (without extension, e.g., "1")
        
    Returns:
        Dictionary with stage2 data for the image, or None if not found
    """
    data = load_preprocess_json(find_path)
    stage2_data = data.get("stage2", {})
    masks_data = stage2_data.get("masks", {})
    return masks_data.get(image_filename)


def load_stage3_data_for_image(find_path: str, image_filename: str) -> Optional[Dict[str, Any]]:
    """Load Stage 3 data for a specific image from preprocess.json.
    
    Args:
        find_path: Path to find directory
        image_filename: Image filename (without extension, e.g., "1")
        
    Returns:
        Dictionary with stage3 data for the image, or None if not found
    """
    data = load_preprocess_json(find_path)
    stage3_data = data.get("stage3", {})
    images_data = stage3_data.get("images", {})
    return images_data.get(image_filename)


def save_preprocess_json(find_path: str, data: Dict[str, Any]) -> bool:
    """Save preprocess.json to find directory.
    
    Args:
        find_path: Path to find directory
        data: Dictionary with preprocess data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = get_preprocess_json_path(find_path)
        
        # Atomic write: write to temp file, then rename
        temp_file = json_path.with_suffix(".tmp")
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Rename to final file
        temp_file.replace(json_path)
        
        logger.info(f"Saved preprocess.json to {json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save preprocess.json to {find_path}: {e}", exc_info=True)
        return False


def get_mask_path(find_path: str, image_id: str) -> Path:
    """Get path to mask file.
    
    Args:
        find_path: Path to find directory
        image_id: Image identifier
        
    Returns:
        Path to mask file
    """
    ascota_dir = ensure_ascota_folder(find_path)
    masks_dir = ascota_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    # Set hidden attribute on masks subdirectory as well (Windows)
    set_hidden_attribute_windows(masks_dir)
    return masks_dir / f"{image_id}_mask.png"


def save_mask(find_path: str, image_id: str, mask_array: np.ndarray) -> bool:
    """Save binary mask as PNG file.
    
    Args:
        find_path: Path to find directory
        image_id: Image identifier
        mask_array: Binary mask as numpy array (0/1 or 0/255)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        mask_path = get_mask_path(find_path, image_id)
        
        # Ensure mask is in 0/255 format
        if mask_array.max() <= 1:
            mask_array = (mask_array * 255).astype(np.uint8)
        else:
            mask_array = mask_array.astype(np.uint8)
        
        # Convert to PIL Image and save
        mask_image = Image.fromarray(mask_array, mode='L')
        mask_image.save(mask_path, format='PNG')
        
        logger.info(f"Saved mask to {mask_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save mask to {find_path}: {e}", exc_info=True)
        return False


def load_mask(find_path: str, image_id: str) -> Optional[np.ndarray]:
    """Load binary mask from file.
    
    Args:
        find_path: Path to find directory
        image_id: Image identifier
        
    Returns:
        Binary mask as numpy array (0/1), or None if not found
    """
    try:
        mask_path = get_mask_path(find_path, image_id)
        
        if not mask_path.exists():
            return None
        
        mask_image = Image.open(mask_path)
        mask_array = np.array(mask_image.convert('L'), dtype=np.uint8)
        
        # Convert to 0/1 format
        mask_array = (mask_array > 128).astype(np.uint8)
        
        return mask_array
        
    except Exception as e:
        logger.error(f"Failed to load mask from {find_path}: {e}", exc_info=True)
        return None

