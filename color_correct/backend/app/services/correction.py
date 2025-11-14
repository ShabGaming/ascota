"""Color correction service for applying parametric adjustments."""

import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
import logging
from pathlib import Path

from app.services.models import CorrectionParams

logger = logging.getLogger(__name__)


def apply_temperature_tint(img_rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Apply temperature and tint adjustments.
    
    Args:
        img_rgb: RGB image as float array [0, 1]
        temperature: -100 to 100 (negative = cooler/blue, positive = warmer/yellow)
        tint: -100 to 100 (negative = green, positive = magenta)
        
    Returns:
        Adjusted RGB image
    """
    img = img_rgb.copy()
    
    # Temperature adjustment (modify blue and red channels)
    if temperature != 0:
        temp_factor = temperature / 100.0
        # Warm = increase red, decrease blue
        img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + temp_factor * 0.3), 0, 1)
        img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - temp_factor * 0.3), 0, 1)
    
    # Tint adjustment (modify green and magenta)
    if tint != 0:
        tint_factor = tint / 100.0
        # Magenta = increase red+blue, decrease green
        if tint_factor > 0:
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - tint_factor * 0.2), 0, 1)
        else:
            # Green = increase green
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + abs(tint_factor) * 0.2), 0, 1)
    
    return img


def apply_exposure(img: np.ndarray, exposure: float) -> np.ndarray:
    """Apply exposure adjustment in stops.
    
    Args:
        img: RGB image as float array [0, 1]
        exposure: Exposure adjustment in EV (-2 to 2)
        
    Returns:
        Adjusted image
    """
    if exposure == 0:
        return img
    
    # Each stop = multiply by 2
    multiplier = 2 ** exposure
    return np.clip(img * multiplier, 0, 1)


def apply_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    """Apply contrast adjustment.
    
    Args:
        img: RGB image as float array [0, 1]
        contrast: Contrast multiplier (0.5 to 2.0, 1.0 = no change)
        
    Returns:
        Adjusted image
    """
    if contrast == 1.0:
        return img
    
    # Contrast around midpoint
    midpoint = 0.5
    return np.clip((img - midpoint) * contrast + midpoint, 0, 1)


def apply_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """Apply saturation adjustment.
    
    Args:
        img: RGB image as float array [0, 1]
        saturation: Saturation multiplier (0 to 2.0, 1.0 = no change)
        
    Returns:
        Adjusted image
    """
    if saturation == 1.0:
        return img
    
    # Convert to HSV
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Adjust saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    
    # Convert back
    hsv_uint8 = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
    
    return rgb.astype(np.float32) / 255.0


def apply_rgb_gains(img: np.ndarray, red_gain: float, green_gain: float, blue_gain: float) -> np.ndarray:
    """Apply per-channel RGB gains.
    
    Args:
        img: RGB image as float array [0, 1]
        red_gain, green_gain, blue_gain: Gain multipliers (0.5 to 2.0)
        
    Returns:
        Adjusted image
    """
    result = img.copy()
    result[:, :, 0] = np.clip(result[:, :, 0] * red_gain, 0, 1)
    result[:, :, 1] = np.clip(result[:, :, 1] * green_gain, 0, 1)
    result[:, :, 2] = np.clip(result[:, :, 2] * blue_gain, 0, 1)
    return result


def apply_correction_params(img: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Apply all correction parameters to an image.
    
    Args:
        img: RGB image as float array [0, 1]
        params: Correction parameters
        
    Returns:
        Corrected image
    """
    result = img.copy()
    
    # Apply in order: RGB gains -> exposure -> temp/tint -> contrast -> saturation
    result = apply_rgb_gains(result, params.red_gain, params.green_gain, params.blue_gain)
    result = apply_exposure(result, params.exposure)
    result = apply_temperature_tint(result, params.temperature, params.tint)
    result = apply_contrast(result, params.contrast)
    result = apply_saturation(result, params.saturation)
    
    return result


def load_image_as_float(image_path: str) -> Optional[np.ndarray]:
    """Load an image as a float RGB array.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB array [0, 1] or None if failed
    """
    try:
        img = Image.open(image_path)
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb).astype(np.float32) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def save_image_from_float(img: np.ndarray, output_path: str, quality: int = 95):
    """Save a float RGB array as an image.
    
    Args:
        img: RGB array [0, 1]
        output_path: Output file path
        quality: JPEG quality (if applicable)
    """
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    
    # Determine format from extension
    ext = Path(output_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        pil_img.save(output_path, 'JPEG', quality=quality)
    else:
        pil_img.save(output_path)


def estimate_auto_correction(image_path: str) -> CorrectionParams:
    """Estimate auto-correction parameters from an image.
    
    Uses gray-world assumption on bright areas to estimate white balance.
    
    Args:
        image_path: Path to image
        
    Returns:
        Estimated correction parameters
    """
    img = load_image_as_float(image_path)
    
    if img is None:
        return CorrectionParams()
    
    h, w = img.shape[:2]
    
    # Sample corners (avoid color cards by taking brightest regions)
    corner_size = min(h, w) // 8
    corners = [
        img[:corner_size, :corner_size],  # top-left
        img[:corner_size, -corner_size:],  # top-right
        img[-corner_size:, :corner_size],  # bottom-left
        img[-corner_size:, -corner_size:]  # bottom-right
    ]
    
    # Find brightest corners (likely background)
    corner_brightness = [np.mean(corner) for corner in corners]
    sorted_indices = np.argsort(corner_brightness)[::-1]
    
    # Use top 3 brightest corners
    bright_corners = [corners[i] for i in sorted_indices[:3]]
    combined = np.concatenate([c.reshape(-1, 3) for c in bright_corners], axis=0)
    
    # Calculate mean RGB
    mean_rgb = np.mean(combined, axis=0)
    
    # Gray-world: ideal would be equal RGB
    gray = np.mean(mean_rgb)
    
    if gray < 0.01:  # Too dark
        return CorrectionParams()
    
    # Calculate RGB gains to normalize to gray
    red_gain = gray / (mean_rgb[0] + 1e-6)
    green_gain = gray / (mean_rgb[1] + 1e-6)
    blue_gain = gray / (mean_rgb[2] + 1e-6)
    
    # Clamp gains
    red_gain = np.clip(red_gain, 0.5, 2.0)
    green_gain = np.clip(green_gain, 0.5, 2.0)
    blue_gain = np.clip(blue_gain, 0.5, 2.0)
    
    # Normalize so green is 1.0 (standard reference)
    norm_factor = green_gain
    red_gain /= norm_factor
    blue_gain /= norm_factor
    green_gain = 1.0
    
    logger.info(f"Auto-correction for {image_path}: R={red_gain:.3f}, G={green_gain:.3f}, B={blue_gain:.3f}")
    
    return CorrectionParams(
        red_gain=float(red_gain),
        green_gain=float(green_gain),
        blue_gain=float(blue_gain)
    )

