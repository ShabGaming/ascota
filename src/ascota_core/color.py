"""
Color correction and color grading algorithms. Also includes color based clustering algorithms.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2
import os
import glob
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Helpers
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert sRGB color values to linear RGB.
    
    Applies the inverse sRGB gamma correction curve to convert from sRGB color space
    to linear RGB color space for proper color processing calculations.
    
    Args:
        x: Input array of sRGB values in range [0.0, 1.0].
            
    Returns:
        Array of linear RGB values in range [0.0, 1.0] with same shape as input.
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    low = x <= 0.04045
    high = ~low
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[high] = ((x[high] + a) / (1 + a)) ** 2.4
    return out


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear RGB color values to sRGB.
    
    Applies the sRGB gamma correction curve to convert from linear RGB color space
    to sRGB color space for display or output purposes.
    
    Args:
        x: Input array of linear RGB values in range [0.0, 1.0].
            
    Returns:
        Array of sRGB values in range [0.0, 1.0] with same shape as input.
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    low = x <= 0.0031308
    high = ~low
    out = np.empty_like(x, dtype=np.float32)
    out[low] = 12.92 * x[low]
    out[high] = (1 + a) * (x[high] ** (1 / 2.4)) - a
    return np.clip(out, 0.0, 1.0)


def bgr_linear_to_lab_u8(bgr_lin: np.ndarray) -> np.ndarray:
    """Convert linear BGR image to 8-bit LAB color space.
    
    Converts a linear BGR image to LAB color space for perceptually uniform
    color operations. The output is in 8-bit format suitable for OpenCV operations.
    
    Args:
        bgr_lin: Linear BGR image array with values in range [0.0, 1.0].
            
    Returns:
        8-bit LAB image array with shape matching input.
    """
    bgr_srgb = linear_to_srgb(np.clip(bgr_lin, 0.0, 1.0))
    bgr8 = (bgr_srgb * 255.0 + 0.5).astype(np.uint8)
    lab8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    return lab8


def lab_u8_to_bgr_linear(lab8: np.ndarray) -> np.ndarray:
    """Convert 8-bit LAB image to linear BGR color space.
    
    Converts a LAB color space image back to linear BGR format for further
    processing or color correction operations.
    
    Args:
        lab8: 8-bit LAB image array.
            
    Returns:
        Linear BGR image array with values in range [0.0, 1.0].
    """
    bgr8 = cv2.cvtColor(lab8.astype(np.uint8), cv2.COLOR_LAB2BGR)
    bgr_srgb = np.clip(bgr8.astype(np.float32) / 255.0, 0.0, 1.0)
    return srgb_to_linear(bgr_srgb)


def to_uint8_preview(bgr_lin: np.ndarray) -> Image.Image:
    """Convert linear BGR image to 8-bit RGB PIL Image for preview.
    
    Converts a linear BGR image to sRGB color space and creates a PIL Image
    suitable for display or preview purposes.
    
    Args:
        bgr_lin: Linear BGR image array with values in range [0.0, 1.0].
            
    Returns:
        PIL Image in RGB format with 8-bit precision.
    """
    bgr_srgb = linear_to_srgb(np.clip(bgr_lin, 0.0, 1.0))
    rgb8 = (cv2.cvtColor((bgr_srgb * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB))
    return Image.fromarray(rgb8)


def to_png16_bytes(bgr_lin: np.ndarray) -> bytes:
    """Convert linear BGR image to 16-bit PNG bytes.
    
    Converts a linear BGR image to sRGB color space and encodes it as a 16-bit
    PNG in bytes format for high-quality output or storage.
    
    Args:
        bgr_lin: Linear BGR image array with values in range [0.0, 1.0].
            
    Returns:
        PNG-encoded bytes of the 16-bit image.
        
    Raises:
        RuntimeError: If PNG encoding fails.
    """
    bgr_srgb = linear_to_srgb(np.clip(bgr_lin, 0.0, 1.0))
    bgr16 = (bgr_srgb * 65535.0 + 0.5).astype(np.uint16)
    ok, buf = cv2.imencode('.png', bgr16)
    if not ok:
        raise RuntimeError("Failed to encode PNG16 with OpenCV")
    return buf.tobytes()


def _build_monotone_quantile_mapping(src_vals: np.ndarray, tgt_vals: np.ndarray, n_knots: int = 257) -> Tuple[np.ndarray, np.ndarray]:
    """Build monotonic quantile mapping between source and target values.
    
    Creates a monotonic mapping from source values to target values using
    quantiles. This is used for histogram matching and color transfer operations.
    
    Args:
        src_vals: Source values to map from.
        tgt_vals: Target values to map to.
        n_knots: Number of quantile knots to use for the mapping. Defaults to 257.
        
    Returns:
        Tuple of (source_quantiles, target_quantiles) as float32 arrays for
        use with np.interp().
    """
    q = np.linspace(0.0, 1.0, n_knots, dtype=np.float32)
    s_q = np.quantile(src_vals.astype(np.float32), q)
    t_q = np.quantile(tgt_vals.astype(np.float32), q)
    s_q = np.maximum.accumulate(s_q)
    eps = 1e-4
    for i in range(1, len(s_q)):
        if s_q[i] <= s_q[i-1]:
            s_q[i] = s_q[i-1] + eps
    return s_q.astype(np.float32), t_q.astype(np.float32)


def apply_manual_adjustments_linear(bgr_lin: np.ndarray, brightness: int = 0, contrast: float = 1.0, gamma: float = 1.0,
                                    r_gain: float = 1.0, g_gain: float = 1.0, b_gain: float = 1.0) -> np.ndarray:
    """Apply manual color adjustments to a linear BGR image.
    
    Applies brightness, contrast, gamma, and per-channel gain adjustments to
    a linear BGR image. All adjustments are performed in sRGB space before
    converting back to linear.
    
    Args:
        bgr_lin: Linear BGR image array with values in range [0.0, 1.0].
        brightness: Brightness adjustment in range [-255, 255]. Defaults to 0.
        contrast: Contrast multiplier, values > 1.0 increase contrast. Defaults to 1.0.
        gamma: Gamma correction value, values < 1.0 brighten midtones. Defaults to 1.0.
        r_gain: Red channel gain multiplier. Defaults to 1.0.
        g_gain: Green channel gain multiplier. Defaults to 1.0.
        b_gain: Blue channel gain multiplier. Defaults to 1.0.
        
    Returns:
        Adjusted linear BGR image array with same shape as input.
    """
    bgr_srgb = linear_to_srgb(bgr_lin)
    gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32).reshape(1, 1, 3)
    x = np.clip(bgr_srgb * contrast + (brightness / 255.0), 0.0, 1.0)
    x = np.clip(x * gains, 0.0, 1.0)
    x = np.power(np.clip(x, 1e-6, 1.0), 1.0 / max(gamma, 1e-6))
    return srgb_to_linear(np.clip(x, 0.0, 1.0))


def lab_mean_std_transfer_linear(src_bgr_lin: np.ndarray, tgt_bgr_lin: np.ndarray,
                                 src_mask: np.ndarray, tgt_mask: np.ndarray) -> np.ndarray:
    """Transfer color statistics using LAB mean and standard deviation.
    
    Performs color transfer by matching the mean and standard deviation
    of each LAB channel within the masked regions. This method preserves
    luminance relationships while adjusting color characteristics.
    
    Args:
        src_bgr_lin: Source linear BGR image array.
        tgt_bgr_lin: Target linear BGR image array.
        src_mask: Boolean mask for source image region of interest.
        tgt_mask: Boolean mask for target image region of interest.
        
    Returns:
        Color-corrected linear BGR image array with same shape as source.
    """
    src_lab = bgr_linear_to_lab_u8(src_bgr_lin)
    tgt_lab = bgr_linear_to_lab_u8(tgt_bgr_lin)
    out = src_lab.astype(np.float32)
    for c in range(3):
        s_vals = src_lab[..., c][src_mask].astype(np.float32)
        t_vals = tgt_lab[..., c][tgt_mask].astype(np.float32)
        if len(s_vals) == 0 or len(t_vals) == 0:
            continue
        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals) + 1e-6)
        t_mean, t_std = float(np.mean(t_vals)), float(np.std(t_vals))
        out[..., c] = (out[..., c] - s_mean) * (t_std / s_std) + t_mean
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return lab_u8_to_bgr_linear(out_u8)


def monotone_lut_ab_only(src_bgr_lin: np.ndarray, tgt_bgr_lin: np.ndarray,
                         src_mask: np.ndarray, tgt_mask: np.ndarray,
                         n_knots: int = 257) -> np.ndarray:
    """Apply monotonic LUT mapping to A and B channels only in LAB space.
    
    Performs color transfer by applying quantile-based monotonic lookup tables
    to the A and B channels in LAB color space, while preserving the L channel.
    This method focuses on chromaticity transfer without affecting luminance.
    
    Args:
        src_bgr_lin: Source linear BGR image array.
        tgt_bgr_lin: Target linear BGR image array.
        src_mask: Boolean mask for source image region of interest.
        tgt_mask: Boolean mask for target image region of interest.
        n_knots: Number of quantile knots for the LUT mapping. Defaults to 257.
        
    Returns:
        Color-corrected linear BGR image array with same shape as source.
    """
    src_lab = bgr_linear_to_lab_u8(src_bgr_lin)
    tgt_lab = bgr_linear_to_lab_u8(tgt_bgr_lin)
    out = src_lab.astype(np.float32)
    for c in [1, 2]:
        s_vals = src_lab[..., c][src_mask].astype(np.float32)
        t_vals = tgt_lab[..., c][tgt_mask].astype(np.float32)
        if len(s_vals) == 0 or len(t_vals) == 0:
            continue
        xk, yk = _build_monotone_quantile_mapping(s_vals, t_vals, n_knots=n_knots)
        ch = out[..., c]
        out[..., c] = np.interp(ch, xk, yk).astype(np.float32)
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return lab_u8_to_bgr_linear(out_u8)


def masked_histogram_match_rgb_linear(src_bgr_lin: np.ndarray, tgt_bgr_lin: np.ndarray,
                                      src_mask: np.ndarray, tgt_mask: np.ndarray,
                                      n_knots: int = 1025) -> np.ndarray:
    """Perform histogram matching in RGB space using masked regions.
    
    Applies quantile-based histogram matching to each RGB channel independently
    using only the pixels within the specified mask regions. This method provides
    comprehensive color transfer across all channels.
    
    Args:
        src_bgr_lin: Source linear BGR image array.
        tgt_bgr_lin: Target linear BGR image array.
        src_mask: Boolean mask for source image region of interest.
        tgt_mask: Boolean mask for target image region of interest.
        n_knots: Number of quantile knots for histogram matching. Defaults to 1025.
        
    Returns:
        Color-corrected linear BGR image array with same shape as source.
    """
    src_srgb = linear_to_srgb(src_bgr_lin)
    tgt_srgb = linear_to_srgb(tgt_bgr_lin)
    src_rgb = cv2.cvtColor(src_srgb, cv2.COLOR_BGR2RGB)
    tgt_rgb = cv2.cvtColor(tgt_srgb, cv2.COLOR_BGR2RGB)
    out_rgb = src_rgb.copy()
    for c in range(3):
        s_vals = src_rgb[..., c][src_mask].astype(np.float32)
        t_vals = tgt_rgb[..., c][tgt_mask].astype(np.float32)
        if len(s_vals) == 0 or len(t_vals) == 0:
            continue
        s_vals255 = s_vals * 255.0
        t_vals255 = t_vals * 255.0
        xk, yk = _build_monotone_quantile_mapping(s_vals255, t_vals255, n_knots=n_knots)
        ch = (out_rgb[..., c] * 255.0).astype(np.float32)
        mapped = np.interp(ch, xk, yk) / 255.0
        out_rgb[..., c] = np.clip(mapped, 0.0, 1.0)
    out_bgr_srgb = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return srgb_to_linear(out_bgr_srgb)


def _pil_to_bgr_linear(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL Image to linear BGR array.
    
    Converts a PIL Image to linear BGR format suitable for color processing
    operations. The conversion goes through RGB -> sRGB -> linear RGB -> BGR.
    
    Args:
        img_pil: Input PIL Image in any supported format.
        
    Returns:
        Linear BGR image array with values in range [0.0, 1.0].
    """
    rgb8 = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    rgb_srgb = rgb8.astype(np.float32) / 255.0
    rgb_lin = srgb_to_linear(rgb_srgb)
    bgr_lin = cv2.cvtColor(rgb_lin, cv2.COLOR_RGB2BGR)
    return bgr_lin


def _prepare_mask_bool(mask_pil: Image.Image, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Convert PIL mask to boolean array aligned to expected dimensions.
    
    Converts a PIL mask image to a boolean array where True indicates the
    region of interest (card area). Resizes the mask if needed to match
    the expected dimensions.
    
    Args:
        mask_pil: PIL mask image, will be converted to grayscale.
        expected_shape: Target dimensions as (width, height) tuple.
        
    Returns:
        Boolean mask array with shape (height, width) where True indicates
        the card region (pixels with value >= 200).
    """
    mask = mask_pil.convert("L")
    arr = np.array(mask, dtype=np.uint8)
    # expected_shape is (width, height) to match how read_mask used in streamlit; convert
    expected_w, expected_h = expected_shape
    if (arr.shape[1], arr.shape[0]) != (expected_w, expected_h):
        arr = cv2.resize(arr, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
    return arr >= 200


def match_color_cards_from_pils(
    original_image: Image.Image,
    original_masks: List[Image.Image],
    target_image: Image.Image, 
    target_masks: List[Image.Image],
    original_card_types: List[str],
    target_card_types: List[str],
    method: str = "lab_mean_std_transfer",
    n_knots: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Match color appearance between two images using color card references.
    
    Performs color correction by matching the appearance of color cards between
    a source image and target image. The function automatically finds matching
    color card types and applies the specified color transfer method.
    
    Args:
        original_image: Source PIL Image to correct.
        original_masks: List of PIL mask images for detected cards in source image.
        target_image: Target PIL Image providing the desired color appearance.
        target_masks: List of PIL mask images for detected cards in target image.
        original_card_types: List of card type strings corresponding to original_masks.
        target_card_types: List of card type strings corresponding to target_masks.
        method: Color transfer method to use. One of:
            - "lab_mean_std_transfer" (default): Mean/std matching in LAB space
            - "monotone_lut": Quantile mapping on LAB A/B channels only
            - "histogram_matching": Full RGB histogram matching
        n_knots: Number of quantile knots for LUT-based methods. If None, uses
            method-specific defaults (513 for monotone_lut, 1025 for histogram_matching).
        debug: Enable debug output. Defaults to False.
        
    Returns:
        Dictionary containing:
            - matched_preview: 8-bit PIL Image preview of corrected result
            - matched_png16: 16-bit PNG encoded bytes of corrected result  
            - used_card_type: Name of color card type used ('colorchecker24' or 'colorchecker8')
            - method: The color transfer method that was applied
            
    Raises:
        ValueError: If no matching color card types are found in both images,
            or if an unknown method is specified.
    """

    # select preferred card types
    preferred = ['colorchecker24', 'colorchecker8']

    # Helper to find indices by type list
    def _index_of_type(types_list: List[str], masks_list: List[Image.Image], desired: str) -> Optional[int]:
        for i, t in enumerate(types_list):
            if t == desired and i < len(masks_list):
                return i
        return None

    # Find a matching preferred type present in both sets
    chosen_type = None
    orig_idx = tgt_idx = None
    for t in preferred:
        o_i = _index_of_type(original_card_types, original_masks, t)
        ti = _index_of_type(target_card_types, target_masks, t)
        if o_i is not None and ti is not None:
            chosen_type = t
            orig_idx = o_i
            tgt_idx = ti
            break

    if chosen_type is None:
        raise ValueError("Could not find matching colorchecker24/colorchecker8 masks in both inputs")

    # Convert PIL images to linear BGR arrays
    src_lin = _pil_to_bgr_linear(original_image)
    tgt_lin = _pil_to_bgr_linear(target_image)

    # Prepare boolean masks aligned to images
    src_mask_bool = _prepare_mask_bool(original_masks[orig_idx], (src_lin.shape[1], src_lin.shape[0]))
    tgt_mask_bool = _prepare_mask_bool(target_masks[tgt_idx], (tgt_lin.shape[1], tgt_lin.shape[0]))

    # Choose algorithm
    algo_map = {
        "lab_mean_std_transfer": lambda s, t, sm, tm: lab_mean_std_transfer_linear(s, t, sm, tm),
        "monotone_lut": lambda s, t, sm, tm: monotone_lut_ab_only(s, t, sm, tm, n_knots=(n_knots or 513)),
        "histogram_matching": lambda s, t, sm, tm: masked_histogram_match_rgb_linear(s, t, sm, tm, n_knots=(n_knots or 1025)),
    }

    if method not in algo_map:
        raise ValueError(f"Unknown method '{method}'. Valid: {list(algo_map.keys())}")

    matched_lin = algo_map[method](src_lin, tgt_lin, src_mask_bool, tgt_mask_bool)

    # Prepare outputs
    matched_preview = to_uint8_preview(matched_lin)
    matched_png16 = to_png16_bytes(matched_lin)

    return {
        'matched_preview': matched_preview,
        'matched_png16': matched_png16,
        'used_card_type': chosen_type,
        'method': method
    }


def match_color_cards_from_pipeline_outputs(
    original_pipeline_output: dict,
    target_pipeline_output: dict,
    method: str = "lab_mean_std_transfer",
    n_knots: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Match color cards using outputs from the image processing pipeline.
    
    Convenience wrapper that accepts dictionaries returned by process_image_pipeline()
    and performs color matching between source and target images using their
    detected color cards.
    
    Args:
        original_pipeline_output: Dictionary from process_image_pipeline() for source image.
            Must contain keys: 'original_image', 'masks', 'card_types'.
        target_pipeline_output: Dictionary from process_image_pipeline() for target image.
            Must contain keys: 'original_image', 'masks', 'card_types'.
        method: Color transfer method to use. See match_color_cards_from_pils() for options.
            Defaults to "lab_mean_std_transfer".
        n_knots: Number of quantile knots for LUT-based methods. If None, uses
            method-specific defaults.
        debug: Enable debug output. Defaults to False.
        
    Returns:
        Dictionary with same structure as match_color_cards_from_pils().
        
    Raises:
        TypeError: If pipeline outputs are not dictionaries.
        ValueError: If required keys are missing from pipeline outputs.
    """
    # Basic validation / helpful error messages
    for name, out in (("original_pipeline_output", original_pipeline_output), ("target_pipeline_output", target_pipeline_output)):
        if not isinstance(out, dict):
            raise TypeError(f"{name} must be a dict as returned by process_image_pipeline()")

    try:
        orig_img = original_pipeline_output['original_image']
        orig_masks = original_pipeline_output['masks']
    except KeyError as e:
        raise ValueError("original_pipeline_output missing required key 'original_image' or 'masks'") from e

    try:
        tgt_img = target_pipeline_output['original_image']
        tgt_masks = target_pipeline_output['masks']
    except KeyError as e:
        raise ValueError("target_pipeline_output missing required key 'original_image' or 'masks'") from e

    orig_types = original_pipeline_output.get('card_types')
    tgt_types = target_pipeline_output.get('card_types')

    return match_color_cards_from_pils(
        original_image=orig_img,
        original_masks=orig_masks,
        target_image=tgt_img,
        target_masks=tgt_masks,
        original_card_types=orig_types,
        target_card_types=tgt_types,
        method=method,
        n_knots=n_knots,
        debug=debug,
    )


def _extract_corner_regions(bgr_lin: np.ndarray, corner_size: float = 0.1) -> List[np.ndarray]:
    """Extract four corner regions from an image.
    
    Extracts square regions from each corner of the image. The size of each
    corner region is determined by the corner_size parameter as a fraction
    of the smaller image dimension.
    
    Args:
        bgr_lin: Linear BGR image array with values in range [0.0, 1.0].
        corner_size: Size of corner regions as fraction of smaller dimension.
            Defaults to 0.1 (10%).
            
    Returns:
        List of 4 corner region arrays: [top_left, top_right, bottom_left, bottom_right].
    """
    h, w = bgr_lin.shape[:2]
    corner_pixels = int(min(h, w) * corner_size)
    
    corners = [
        bgr_lin[:corner_pixels, :corner_pixels],           # top_left
        bgr_lin[:corner_pixels, -corner_pixels:],          # top_right  
        bgr_lin[-corner_pixels:, :corner_pixels],          # bottom_left
        bgr_lin[-corner_pixels:, -corner_pixels:]          # bottom_right
    ]
    
    return corners


def _select_whitest_corners(corners: List[np.ndarray], n_select: int = 3) -> List[np.ndarray]:
    """Select the corners closest to white (high L*, low chroma).
    
    Analyzes each corner region and selects the n_select corners that are
    closest to white. This helps avoid selecting corners that contain color
    cards or other colored objects.
    
    Args:
        corners: List of corner region arrays from _extract_corner_regions().
        n_select: Number of corners to select. Defaults to 3.
        
    Returns:
        List of selected corner arrays, ordered by whiteness score.
    """
    corner_scores = []
    
    for i, corner in enumerate(corners):
        # Convert to LAB for perceptual analysis
        lab = bgr_linear_to_lab_u8(corner)
        
        # Calculate mean L*, a*, b* values
        mean_L = np.mean(lab[..., 0])  # Lightness
        mean_a = np.mean(lab[..., 1])  # Green-Red
        mean_b = np.mean(lab[..., 2])  # Blue-Yellow
        
        # Calculate chroma (color saturation)
        chroma = np.sqrt((mean_a - 128) ** 2 + (mean_b - 128) ** 2)
        
        # Whiteness score: high L* (brightness) and low chroma (saturation)
        # Scale L* from 0-255 to 0-1 for scoring
        whiteness_score = (mean_L / 255.0) - (chroma / 128.0)
        
        corner_scores.append((whiteness_score, i, corner))
    
    # Sort by whiteness score (descending) and select top n_select
    corner_scores.sort(key=lambda x: x[0], reverse=True)
    selected_corners = [corner for _, _, corner in corner_scores[:n_select]]
    
    return selected_corners


def _extract_lighting_features(corners: List[np.ndarray]) -> np.ndarray:
    """Extract lighting feature vector from selected corner regions.
    
    Builds a feature vector from the selected corners by computing mean
    L*a*b* values for each corner. This creates a compact representation
    of the lighting conditions in the image.
    
    Args:
        corners: List of selected corner region arrays.
        
    Returns:
        Feature vector as 1D numpy array with shape (n_corners * 3,).
        Each corner contributes 3 values: mean L*, mean a*, mean b*.
    """
    features = []
    
    for corner in corners:
        # Convert to LAB color space
        lab = bgr_linear_to_lab_u8(corner)
        
        # Extract mean values for each channel
        mean_L = np.mean(lab[..., 0])
        mean_a = np.mean(lab[..., 1]) 
        mean_b = np.mean(lab[..., 2])
        
        features.extend([mean_L, mean_a, mean_b])
    
    return np.array(features, dtype=np.float32)


def _find_optimal_k_bic(features: np.ndarray, max_k: int = 10) -> int:
    """Find optimal number of clusters using Bayesian Information Criterion.
    
    Tests different values of k and selects the one that minimizes BIC,
    balancing model fit with complexity. Uses Gaussian Mixture Models
    for BIC calculation.
    
    Args:
        features: Feature matrix with shape (n_samples, n_features).
        max_k: Maximum number of clusters to test. Defaults to 10.
        
    Returns:
        Optimal number of clusters.
    """
    n_samples = features.shape[0]
    if n_samples <= 1:
        return 1
    
    max_k = min(max_k, n_samples)
    bic_scores = []
    
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(features)
            bic = gmm.bic(features)
            bic_scores.append(bic)
        except Exception:
            # If GMM fails for this k, assign a high BIC score
            bic_scores.append(np.inf)
    
    # Return k with minimum BIC
    optimal_k = np.argmin(bic_scores) + 1
    return optimal_k


def _cluster_images(features: np.ndarray, k: Optional[int] = None, max_k: Optional[int] = None) -> np.ndarray:
    """Cluster images based on their lighting features.
    
    Performs k-means clustering on the feature vectors. If k is not provided,
    automatically determines the optimal k using BIC.
    
    Args:
        features: Feature matrix with shape (n_samples, n_features).
        k: Number of clusters. If None, determined automatically via BIC.
        max_k: Maximum k to try when auto-selecting via BIC.
        
    Returns:
        Array of cluster labels with length n_samples.
    """
    n_samples = features.shape[0]
    if n_samples <= 1:
        return np.array([0] * n_samples)
    
    if k is None:
        k = _find_optimal_k_bic(features, max_k=(max_k or 10))
    
    k = min(k, n_samples)  # Can't have more clusters than samples
    
    if k == 1:
        return np.array([0] * n_samples)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels


def group_similar_images_by_lighting(directory: str, k: Optional[int] = None, 
                                   extensions: List[str] = None, sensitivity: float = 1.0,
                                   debug: bool = False) -> List[List[str]]:
    """Group similar images from a directory based on lighting conditions.
    
    Analyzes images in a directory and groups them based on lighting similarity.
    The function examines the four corners of each image, selects the three
    corners closest to white (to avoid color cards), extracts lighting features,
    and clusters the images accordingly.
    
    Args:
        directory: Path to directory containing images to analyze.
        k: Number of clusters. If None, automatically determined via BIC.
        extensions: List of image file extensions to process. If None, uses
            common formats: ['.jpg', '.jpeg', '.png'].
        sensitivity: Float >0 controlling clustering sensitivity. Values >1 increase
            sensitivity (more clusters, amplifies feature differences). Values <1
            decrease sensitivity (fewer clusters, mutes differences). Default 1.0.
        debug: Enable debug output. Defaults to False.
            
    Returns:
        List of lists, where each inner list contains file paths of images
        with similar lighting conditions.
        
    Raises:
        ValueError: If directory doesn't exist or contains no valid images.
        Exception: If image processing fails for critical errors.
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    if debug:
        print(f"DEBUG: group_similar_images_by_lighting - directory={directory}, k={k}, sensitivity={sensitivity}")
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    # Find all image files
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        image_paths.extend(glob.glob(pattern))
        pattern = os.path.join(directory, f"*{ext.upper()}")
        image_paths.extend(glob.glob(pattern))
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {directory}")
    
    # Extract features from each image
    features_list = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            # Load image
            img_pil = Image.open(img_path)
            bgr_lin = _pil_to_bgr_linear(img_pil)
            
            # Extract corner regions
            corners = _extract_corner_regions(bgr_lin)
            
            # Select the 3 whitest corners
            selected_corners = _select_whitest_corners(corners, n_select=3)
            
            # Extract lighting features
            features = _extract_lighting_features(selected_corners)
            
            features_list.append(features)
            valid_paths.append(img_path)
            
        except Exception as e:
            # Only print debug-style message if debug True, otherwise keep warning
            if debug:
                print(f"DEBUG: group_similar_images_by_lighting - failed to process {img_path}: {e}")
            else:
                print(f"Warning: Failed to process {img_path}: {e}")
            continue
    
    if not features_list:
        raise ValueError("No images could be processed successfully")
    
    # Stack features into matrix
    features_matrix = np.vstack(features_list)
    
    # Apply sensitivity: scale features to amplify/mute differences
    try:
        sens = float(sensitivity)
    except Exception:
        sens = 1.0
    sens = max(0.1, sens)  # prevent zero/negative
    features_scaled = features_matrix * sens
    if debug:
        print(f"DEBUG: group_similar_images_by_lighting - features_matrix.shape={features_matrix.shape}, sens={sens}")
    
    n_samples = features_matrix.shape[0]
    
    # 'k' is dependent on the number of samples, with a minimum of 10 or n_samples/6
    number_of_images_factor = max(10, n_samples // 6)
    if debug:
        print(f"DEBUG: group_similar_images_by_lighting - n_samples={n_samples}, number_of_images_factor={number_of_images_factor}")

    # Cluster images (pass scaled features and max_k for BIC)
    if sensitivity == 1.0:
        proposed_max_k = number_of_images_factor
    else:
        # Compute a sensible max_k for BIC based on sensitivity (higher sens will result in more clusters)
        proposed_max_k = max(2, int(number_of_images_factor * sens))
        proposed_max_k = min(proposed_max_k, n_samples)
    if debug:
        print(f"DEBUG: group_similar_images_by_lighting - n_samples={n_samples}, number_of_images_factor={number_of_images_factor}, proposed_max_k={proposed_max_k}")
    labels = _cluster_images(features_scaled, k=k, max_k=proposed_max_k)
    
    # Group images by cluster
    n_clusters = len(np.unique(labels))
    if debug:
        print(f"DEBUG: group_similar_images_by_lighting - n_clusters={n_clusters}, labels_unique={np.unique(labels)}")
    grouped_images = [[] for _ in range(n_clusters)]
    
    for img_path, label in zip(valid_paths, labels):
        grouped_images[label].append(img_path)
    
    # Remove empty groups and sort by group size (largest first)
    grouped_images = [group for group in grouped_images if group]
    grouped_images.sort(key=len, reverse=True)
    
    return grouped_images
