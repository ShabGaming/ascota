"""Scanner service for discovering -3000 images in contexts."""

import hashlib
from typing import List, Dict
from pathlib import Path
import logging

from app.services.models import ImageItem

logger = logging.getLogger(__name__)


def _get_file_hash(path: str) -> str:
    """Generate a unique hash for a file path."""
    return hashlib.md5(path.encode()).hexdigest()[:16]


def scan_context_directory(context_path: str) -> List[ImageItem]:
    """Scan a context directory and discover all images with -3000 variants.
    
    Args:
        context_path: Path to context directory (e.g., D:/ararat/data/files/N/38/478020/4419550/1)
        
    Returns:
        List of ImageItem objects (only images with -3000 proxy)
    """
    images: List[ImageItem] = []
    context_path = Path(context_path)
    
    if not context_path.exists():
        logger.warning(f"Context path does not exist: {context_path}")
        return images
    
    # Look for finds/individual/*/photos/ directories
    finds_base = context_path / "finds" / "individual"
    
    if not finds_base.exists():
        logger.warning(f"No finds directory at: {finds_base}")
        return images
    
    # Search for all find directories
    find_dirs = [d for d in finds_base.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(find_dirs)} find directories in {context_path}")
    
    for find_dir in find_dirs:
        find_number = find_dir.name
        photos_dir = find_dir / "photos"
        
        if not photos_dir.exists():
            continue
        
        # Collect unique base names from -3000 files
        base_names = set()
        
        # Scan for -3000 images (required for preprocess)
        for img_ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            for img_file in photos_dir.glob(f'*-3000{img_ext}'):
                # Extract base name (remove -3000 suffix)
                base_name = img_file.stem[:-5]  # Remove '-3000'
                base_names.add(base_name)
        
        # For each base name, find all variants
        for base_name in base_names:
            variants = {
                'raw_path': None,
                'proxy_3000': None,
                'proxy_1500': None,
                'proxy_450': None
            }
            
            # Look for RAW files
            for raw_ext in ['.CR3', '.CR2', '.cr3', '.cr2']:
                raw_path = photos_dir / f"{base_name}{raw_ext}"
                if raw_path.exists():
                    variants['raw_path'] = str(raw_path)
                    break
            
            # Look for -3000 variant (required)
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                p3000 = photos_dir / f"{base_name}-3000{ext}"
                if p3000.exists():
                    variants['proxy_3000'] = str(p3000)
                    break
            
            # Look for -1500 variant (optional)
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                p1500 = photos_dir / f"{base_name}-1500{ext}"
                if p1500.exists():
                    variants['proxy_1500'] = str(p1500)
                    break
            
            # Look for base (450) variant (optional)
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                pbase = photos_dir / f"{base_name}{ext}"
                if pbase.exists():
                    variants['proxy_450'] = str(pbase)
                    break
            
            # Require -3000 proxy for preprocess
            if not variants['proxy_3000']:
                logger.debug(f"Skipping image {base_name} in find {find_number} - no -3000 proxy found")
                continue
            
            # Use -3000 as primary path for preprocess
            primary_path = variants['proxy_3000']
            
            # Create ImageItem
            image_id = _get_file_hash(primary_path)
            
            image_item = ImageItem(
                id=image_id,
                context_id=str(context_path),
                find_number=find_number,
                raw_path=variants['raw_path'],
                proxy_3000=variants['proxy_3000'],
                proxy_1500=variants['proxy_1500'],
                proxy_450=variants['proxy_450'],
                primary_path=primary_path
            )
            
            images.append(image_item)
            logger.debug(f"Found image {base_name} in find {find_number}")
    
    logger.info(f"Scanned {context_path}: found {len(images)} images with -3000 proxies")
    return images


def scan_all_contexts(context_paths: List[str]) -> Dict[str, ImageItem]:
    """Scan all context directories and return a unified image dictionary.
    
    Args:
        context_paths: List of context directory paths
        
    Returns:
        Dict mapping image_id to ImageItem (only images with -3000 proxies)
    """
    all_images: Dict[str, ImageItem] = {}
    
    for context_path in context_paths:
        images = scan_context_directory(context_path)
        for img in images:
            all_images[img.id] = img
    
    logger.info(f"Scanned {len(context_paths)} contexts: total {len(all_images)} images")
    return all_images

