"""Preview cache service for storing converted RAW images."""

import os
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Cache directory in temp folder
CACHE_DIR = Path(tempfile.gettempdir()) / "color_correct_preview_cache"


def ensure_cache_dir():
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(raw_path: str, preview_resolution: int) -> str:
    """Generate a cache key for a RAW file and resolution.
    
    Args:
        raw_path: Path to RAW file
        preview_resolution: Preview resolution width
        
    Returns:
        Cache key (hash-based filename)
    """
    # Create hash from raw path and resolution
    key_string = f"{raw_path}:{preview_resolution}"
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    return f"{key_hash}.jpg"


def get_cached_path(raw_path: str, preview_resolution: int) -> Optional[Path]:
    """Get path to cached JPG if it exists and is valid.
    
    Args:
        raw_path: Path to RAW file
        preview_resolution: Preview resolution width
        
    Returns:
        Path to cached file if valid, None otherwise
    """
    ensure_cache_dir()
    
    cache_key = get_cache_key(raw_path, preview_resolution)
    cached_path = CACHE_DIR / cache_key
    
    # Check if cache exists and is newer than RAW file
    if cached_path.exists():
        try:
            raw_mtime = os.path.getmtime(raw_path)
            cache_mtime = os.path.getmtime(cached_path)
            
            # If cache is newer or same age as RAW, use it
            if cache_mtime >= raw_mtime:
                return cached_path
            else:
                # Cache is stale, remove it
                logger.debug(f"Removing stale cache: {cached_path}")
                cached_path.unlink()
        except OSError:
            # Can't check mtime, assume cache is invalid
            return None
    
    return None


def save_to_cache(raw_path: str, preview_resolution: int, jpg_data: bytes) -> Optional[Path]:
    """Save converted JPG to cache.
    
    Args:
        raw_path: Path to RAW file
        preview_resolution: Preview resolution width
        jpg_data: JPG image data as bytes
        
    Returns:
        Path to cached file if successful, None otherwise
    """
    try:
        ensure_cache_dir()
        
        cache_key = get_cache_key(raw_path, preview_resolution)
        cached_path = CACHE_DIR / cache_key
        
        # Write to cache
        with open(cached_path, 'wb') as f:
            f.write(jpg_data)
        
        logger.debug(f"Cached preview: {raw_path} -> {cached_path}")
        return cached_path
        
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}", exc_info=True)
        return None


def get_preview_source(image, preview_resolution: int) -> Tuple[Optional[str], bool]:
    """Get the best source for preview (cached JPG, existing proxy, or RAW).
    
    Args:
        image: ImageItem object
        preview_resolution: Preview resolution width
        
    Returns:
        Tuple of (source_path, is_cached_or_proxy)
        - source_path: Path to use for preview (None if must use RAW)
        - is_cached_or_proxy: True if using cached/proxy (fast), False if must process RAW
    """
    # Priority 1: Check for cached JPG
    if image.raw_path and os.path.exists(image.raw_path):
        cached_path = get_cached_path(image.raw_path, preview_resolution)
        if cached_path:
            return str(cached_path), True
    
    # Priority 2: Check for existing proxy at matching resolution
    if preview_resolution == 3000 and image.proxy_3000 and os.path.exists(image.proxy_3000):
        return image.proxy_3000, True
    elif preview_resolution == 1500 and image.proxy_1500 and os.path.exists(image.proxy_1500):
        return image.proxy_1500, True
    elif preview_resolution == 450 and image.proxy_450 and os.path.exists(image.proxy_450):
        return image.proxy_450, True
    
    # Priority 3: Use closest available proxy
    if image.proxy_3000 and os.path.exists(image.proxy_3000):
        return image.proxy_3000, True
    elif image.proxy_1500 and os.path.exists(image.proxy_1500):
        return image.proxy_1500, True
    elif image.proxy_450 and os.path.exists(image.proxy_450):
        return image.proxy_450, True
    
    # Priority 4: Must use RAW (will need to convert and cache)
    if image.raw_path and os.path.exists(image.raw_path):
        return image.raw_path, False
    
    return None, False

