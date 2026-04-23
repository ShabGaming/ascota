"""Build RGBA images from find image + mask for classification pipeline."""

from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _get_foreground_bbox(
    mask_array: np.ndarray,
    padding: int = 0,
) -> tuple[int, int, int, int]:
    """Return (cmin, rmin, cmax, rmax) for the foreground (mask > 128)."""
    alpha_threshold = 128
    rows = np.any(mask_array > alpha_threshold, axis=1)
    cols = np.any(mask_array > alpha_threshold, axis=0)
    if not np.any(rows) or not np.any(cols):
        h, w = mask_array.shape
        return 0, 0, w, h
    rmin = int(np.argmax(rows))
    rmax = int(len(rows) - np.argmax(rows[::-1]))
    cmin = int(np.argmax(cols))
    cmax = int(len(cols) - np.argmax(cols[::-1]))
    h, w = mask_array.shape
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding)
    return cmin, rmin, cmax, rmax


def crop_image_to_mask_bounds(
    image_3000_path: str,
    find_path: str,
    mask_path_rel: str,
    padding: int = 8,
) -> Image.Image:
    """
    Load the find image and mask; crop the image (RGB) to the foreground bounding box.
    Returns RGB PIL Image (no transparency) zoomed to the pottery region.
    """
    img_path = Path(image_3000_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_3000_path}")
    find_dir = Path(find_path)
    mask_path = find_dir / ".ascota" / mask_path_rel
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    image = Image.open(img_path).convert("RGB")
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    mask_array = (mask_array > 128).astype(np.uint8) * 255
    if mask_array.shape[:2] != (image.height, image.width):
        mask_pil = Image.fromarray(mask_array, mode="L")
        mask_pil = mask_pil.resize((image.width, image.height), Image.Resampling.NEAREST)
        mask_array = np.array(mask_pil, dtype=np.uint8)
    cmin, rmin, cmax, rmax = _get_foreground_bbox(mask_array, padding=padding)
    return image.crop((cmin, rmin, cmax, rmax))


def crop_to_foreground_bounds(
    rgba: Image.Image,
    alpha_threshold: int = 128,
    padding: int = 0,
) -> Image.Image:
    """
    Crop RGBA image to the bounding box of non-transparent pixels (foreground).
    Returns a new image so the foreground fills the frame (zoomed to object).
    """
    a = np.array(rgba.split()[-1], dtype=np.uint8)
    cmin, rmin, cmax, rmax = _get_foreground_bbox(a, padding=padding)
    return rgba.crop((cmin, rmin, cmax, rmax))


def build_rgba_from_find_and_mask(
    find_path: str,
    image_filename: str,
    image_3000_path: str,
    mask_path_rel: str,
    target_size: Optional[int] = None,
) -> Image.Image:
    """
    Load find image at 3000px, apply mask as alpha, optionally resize to target_size.
    Returns PIL Image in RGBA mode (transparent where mask is 0).
    """
    img_path = Path(image_3000_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_3000_path}")

    find_dir = Path(find_path)
    mask_path = find_dir / ".ascota" / mask_path_rel
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image = Image.open(img_path).convert("RGB")

    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    # Binary: >128 -> 255, else 0
    mask_array = (mask_array > 128).astype(np.uint8) * 255

    # Masks are at 3000px; image may be 3000px - ensure same size
    if mask_array.shape[:2] != (image.height, image.width):
        mask_pil = Image.fromarray(mask_array, mode="L")
        mask_pil = mask_pil.resize((image.width, image.height), Image.Resampling.NEAREST)
        mask_array = np.array(mask_pil, dtype=np.uint8)

    # Compose RGBA: RGB from image, alpha from mask
    r, g, b = image.split()
    alpha = Image.fromarray(mask_array, mode="L")
    rgba = Image.merge("RGBA", (r, g, b, alpha))

    if target_size is not None and target_size != 3000:
        # Resize so longest side is target_size (keep aspect)
        w, h = rgba.size
        if max(w, h) != target_size:
            if w >= h:
                new_w = target_size
                new_h = int(round(h * target_size / w))
            else:
                new_h = target_size
                new_w = int(round(w * target_size / h))
            rgba = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return rgba
