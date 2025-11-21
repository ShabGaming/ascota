"""Clustering service wrapping ascota_core.color functionality."""

import os
import sys
from typing import List, Dict, Optional
import logging
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path to import ascota_core
src_path = Path(__file__).parent.parent.parent.parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from ascota_core.color import group_similar_images_by_lighting
except ImportError as e:
    logging.error(f"Failed to import ascota_core.color: {e}")
    group_similar_images_by_lighting = None

from app.services.models import ImageItem, Cluster
import uuid

logger = logging.getLogger(__name__)


def is_raw_file(path: str) -> bool:
    """Check if a file is a RAW file.
    
    Args:
        path: File path
        
    Returns:
        True if file is RAW format
    """
    raw_extensions = ['.CR3', '.CR2', '.cr3', '.cr2', '.NEF', '.nef', '.ARW', '.arw']
    return Path(path).suffix.upper() in [ext.upper() for ext in raw_extensions]


def convert_raw_to_jpg(raw_path: str, output_path: str, max_width: int = 1500) -> bool:
    """Convert a RAW file to JPG for clustering.
    
    Args:
        raw_path: Path to RAW file
        output_path: Path to output JPG file
        max_width: Maximum width for converted image (default 1500px)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import rawpy
        
        logger.debug(f"Converting RAW to JPG: {raw_path} -> {output_path}")
        
        with rawpy.imread(raw_path) as raw:
            # Process RAW to RGB with better brightness settings
            # Use auto white balance and allow auto brightness for better matching
            rgb = raw.postprocess(
                use_camera_wb=False,  # Use auto WB (when False, auto WB is used)
                output_bps=16,
                no_auto_bright=False,  # Allow auto brightness adjustment
                bright=1.0             # Brightness multiplier (1.0 = normal)
            )
        
        # Convert to float
        img_float = rgb.astype(np.float32) / 65535.0
        
        # Resize if needed
        h, w = img_float.shape[:2]
        if w > max_width:
            aspect_ratio = h / w
            target_width = max_width
            target_height = int(target_width * aspect_ratio)
            img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='RGB')
            pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='RGB')
        
        # Save as JPG
        pil_img.save(output_path, 'JPEG', quality=95)
        logger.debug(f"Successfully converted RAW to JPG: {output_path}")
        return True
        
    except ImportError:
        logger.error("rawpy not available - cannot convert RAW files")
        return False
    except Exception as e:
        logger.error(f"Failed to convert RAW file {raw_path}: {e}", exc_info=True)
        return False


def cluster_images(
    images: Dict[str, ImageItem],
    k: Optional[int] = None,
    sensitivity: float = 1.0,
    preview_resolution: int = 1500
) -> List[Cluster]:
    """Cluster images by lighting conditions.
    
    Args:
        images: Dict of image_id to ImageItem
        k: Number of clusters (None for auto-detection)
        sensitivity: Clustering sensitivity
        preview_resolution: Resolution to use for RAW conversion (default 1500px)
        
    Returns:
        List of Cluster objects
    """
    if not group_similar_images_by_lighting:
        raise RuntimeError("Clustering function not available - check ascota_core import")
    
    if not images:
        logger.warning("No images to cluster")
        return []
    
    # Create a temporary directory with symlinks/copies for clustering
    # The clustering function expects images in a single directory
    temp_dir = None
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="color_cluster_")
        logger.info(f"Created temp directory for clustering: {temp_dir}")
        
        # Map temp file names to original image IDs
        temp_to_id: Dict[str, str] = {}
        
        # Copy/link primary paths to temp directory
        # Convert RAW files to JPG/PNG for clustering
        for image_id, image_item in images.items():
            if not image_item.primary_path or not os.path.exists(image_item.primary_path):
                logger.warning(f"Primary path not found for image {image_id}")
                continue
            
            src_path = Path(image_item.primary_path)
            
            # Check if it's a RAW file - if so, convert to JPG first
            if is_raw_file(str(src_path)):
                # Convert RAW to JPG for clustering using user's preview resolution
                temp_name = f"{image_id}.jpg"
                temp_path = Path(temp_dir) / temp_name
                
                if convert_raw_to_jpg(str(src_path), str(temp_path), max_width=preview_resolution):
                    temp_to_id[temp_name] = image_id
                    logger.debug(f"Converted RAW {src_path} -> {temp_path} at {preview_resolution}px")
                else:
                    logger.error(f"Failed to convert RAW file {src_path}")
                    continue
            else:
                # For non-RAW files, just copy them
                temp_name = f"{image_id}{src_path.suffix}"
                temp_path = Path(temp_dir) / temp_name
                
                try:
                    shutil.copy2(src_path, temp_path)
                    temp_to_id[temp_name] = image_id
                    logger.debug(f"Copied {src_path} -> {temp_path}")
                except Exception as e:
                    logger.error(f"Failed to copy {src_path}: {e}")
        
        if not temp_to_id:
            logger.error("No valid images for clustering")
            return []
        
        # Determine extensions from temp files
        # Since we convert RAW to JPG, we should only look for JPG/PNG
        extensions = ['.jpg', '.jpeg', '.png']
        
        logger.info(f"Clustering {len(temp_to_id)} images with k={k}, sensitivity={sensitivity}")
        
        # Run clustering
        clustered_paths = group_similar_images_by_lighting(
            directory=temp_dir,
            k=k,
            extensions=extensions,
            sensitivity=sensitivity,
            debug=True
        )
        
        logger.info(f"Clustering complete: {len(clustered_paths)} clusters")
        
        # Convert paths back to image IDs and create Cluster objects
        clusters: List[Cluster] = []
        
        for i, path_list in enumerate(clustered_paths):
            cluster_id = str(uuid.uuid4())
            image_ids: List[str] = []
            
            for path in path_list:
                temp_name = os.path.basename(path)
                if temp_name in temp_to_id:
                    image_ids.append(temp_to_id[temp_name])
                else:
                    logger.warning(f"Temp file {temp_name} not in mapping")
            
            cluster = Cluster(
                id=cluster_id,
                image_ids=image_ids,
                correction_params=None
            )
            clusters.append(cluster)
            logger.info(f"Cluster {i+1}: {len(image_ids)} images")
        
        return clusters
        
    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        raise
    
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

