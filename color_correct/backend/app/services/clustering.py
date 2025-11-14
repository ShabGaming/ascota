"""Clustering service wrapping ascota_core.color functionality."""

import os
import sys
from typing import List, Dict, Optional
import logging
import tempfile
import shutil
from pathlib import Path

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


def cluster_images(
    images: Dict[str, ImageItem],
    k: Optional[int] = None,
    sensitivity: float = 1.0
) -> List[Cluster]:
    """Cluster images by lighting conditions.
    
    Args:
        images: Dict of image_id to ImageItem
        k: Number of clusters (None for auto-detection)
        sensitivity: Clustering sensitivity
        
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
        for image_id, image_item in images.items():
            if not image_item.primary_path or not os.path.exists(image_item.primary_path):
                logger.warning(f"Primary path not found for image {image_id}")
                continue
            
            src_path = Path(image_item.primary_path)
            # Create unique temp filename preserving extension
            temp_name = f"{image_id}{src_path.suffix}"
            temp_path = Path(temp_dir) / temp_name
            
            # Copy file to temp directory
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
        extensions = list(set(Path(name).suffix for name in temp_to_id.keys()))
        
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

