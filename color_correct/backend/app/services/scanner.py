"""Scanner service for discovering and indexing images in contexts."""

import os
import glob
import hashlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

from app.services.models import ImageItem

logger = logging.getLogger(__name__)


def _get_file_hash(path: str) -> str:
    """Generate a unique hash for a file path."""
    return hashlib.md5(path.encode()).hexdigest()[:16]


def _find_image_variants(photo_dir: Path, base_name: str, extensions: List[str]) -> Dict[str, Optional[str]]:
    """Find all variants of an image (base, -1500, -3000, RAW).
    
    Args:
        photo_dir: Directory containing the photos
        base_name: Base name of the image (without extension)
        extensions: List of extensions to search
        
    Returns:
        Dict with keys: raw_path, proxy_3000, proxy_1500, proxy_450
    """
    variants = {
        'raw_path': None,
        'proxy_3000': None,
        'proxy_1500': None,
        'proxy_450': None
    }
    
    # Look for RAW files
    for raw_ext in ['.CR3', '.CR2', '.cr3', '.cr2']:
        raw_path = photo_dir / f"{base_name}{raw_ext}"
        if raw_path.exists():
            variants['raw_path'] = str(raw_path)
            break
    
    # Look for rendered variants
    for ext in extensions:
        # Check for -3000 variant
        p3000 = photo_dir / f"{base_name}-3000{ext}"
        if p3000.exists():
            variants['proxy_3000'] = str(p3000)
        
        # Check for -1500 variant
        p1500 = photo_dir / f"{base_name}-1500{ext}"
        if p1500.exists():
            variants['proxy_1500'] = str(p1500)
        
        # Check for base (450) variant
        pbase = photo_dir / f"{base_name}{ext}"
        if pbase.exists():
            variants['proxy_450'] = str(pbase)
    
    return variants


def scan_context_directory(context_path: str, image_source: str = "3000px") -> List[ImageItem]:
    """Scan a context directory and discover all images.
    
    Args:
        context_path: Path to context directory (e.g., D:/ararat/data/files/N/38/478020/4419550/1)
        image_source: Image source mode: "450px", "1500px", "3000px", or "raw_mode"
        
    Returns:
        List of ImageItem objects
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
        
        # Collect unique base names from all files
        base_names = set()
        
        # Scan for RAW files
        for raw_pattern in ['*.CR3', '*.CR2', '*.cr3', '*.cr2']:
            for raw_file in photos_dir.glob(raw_pattern):
                base_names.add(raw_file.stem)
        
        # Scan for rendered images
        for img_ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            for img_file in photos_dir.glob(f'*{img_ext}'):
                # Extract base name (remove -3000, -1500 suffixes)
                name = img_file.stem
                if name.endswith('-3000'):
                    base_names.add(name[:-5])
                elif name.endswith('-1500'):
                    base_names.add(name[:-5])
                else:
                    base_names.add(name)
        
        # For each base name, find all variants
        for base_name in base_names:
            extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
            variants = _find_image_variants(photos_dir, base_name, extensions)
            
            # Determine primary path for clustering based on image_source with fallback
            if image_source == "raw_mode":
                # Raw mode: RAW → 3000px → 1500px → 450px
                primary_path = (
                    variants['raw_path'] or
                    variants['proxy_3000'] or
                    variants['proxy_1500'] or
                    variants['proxy_450']
                )
            elif image_source == "3000px":
                # 3000px: 3000px → 1500px → 450px
                primary_path = (
                    variants['proxy_3000'] or
                    variants['proxy_1500'] or
                    variants['proxy_450']
                )
            elif image_source == "1500px":
                # 1500px: 1500px → 450px
                primary_path = (
                    variants['proxy_1500'] or
                    variants['proxy_450']
                )
            elif image_source == "450px":
                # 450px: 450px only
                primary_path = variants['proxy_450']
            else:
                # Default to 3000px fallback chain
                primary_path = (
                    variants['proxy_3000'] or
                    variants['proxy_1500'] or
                    variants['proxy_450']
                )
            
            if not primary_path:
                continue
            
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
    
    logger.info(f"Scanned {context_path}: found {len(images)} images")
    return images


def scan_all_contexts(context_paths: List[str], image_source: str = "3000px") -> Dict[str, ImageItem]:
    """Scan all context directories and return a unified image dictionary.
    
    Args:
        context_paths: List of context directory paths
        image_source: Image source mode: "450px", "1500px", "3000px", or "raw_mode"
        
    Returns:
        Dict mapping image_id to ImageItem
    """
    all_images: Dict[str, ImageItem] = {}
    
    for context_path in context_paths:
        images = scan_context_directory(context_path, image_source)
        for img in images:
            all_images[img.id] = img
    
    logger.info(f"Scanned {len(context_paths)} contexts: total {len(all_images)} images")
    return all_images

