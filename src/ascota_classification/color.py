"""
Color classification module for pottery images with transparent backgrounds.

This module provides three methods for classifying pottery color:
1. lab_threshold: Uses CIELAB color space thresholds
2. kmeans_lab: K-means clustering in CIELAB space
3. clip_vit: CLIP-based image classification
"""

import numpy as np
from PIL import Image
from typing import Dict, Literal, Optional, Tuple
from sklearn.cluster import KMeans
from transformers import pipeline


def _postprocess(output: list) -> Dict[str, float]:
    """
    Postprocess CLIP model output to convert to dictionary format.
    
    Args:
        output: Raw output from CLIP pipeline
        
    Returns:
        Dictionary mapping labels to confidence scores
    """
    return {out["label"]: float(out["score"]) for out in output}


def _get_lab_values(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PIL image to CIELAB color space and extract valid pottery pixels.
    
    Args:
        image: PIL Image with transparency (RGBA)
        
    Returns:
        Tuple of (lab_pixels, valid_mask) where lab_pixels is Nx3 array of L*a*b* values
        and valid_mask is boolean array indicating valid pottery pixels
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if image has alpha channel
    if img_array.shape[2] != 4:
        raise ValueError("Image must have an alpha channel (RGBA format)")
    
    # Extract RGB and alpha
    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    
    # Convert RGB to LAB (using PIL's built-in conversion)
    # First convert to RGB mode (without alpha)
    rgb_image = Image.fromarray(rgb, mode='RGB')
    lab_image = rgb_image.convert('LAB')
    lab_array = np.array(lab_image)
    
    # Normalize LAB values to standard ranges
    # PIL's LAB: L* [0, 255] → [0, 100], a* [0, 255] → [-128, 127], b* [0, 255] → [-128, 127]
    L = lab_array[:, :, 0] * 100.0 / 255.0
    a = lab_array[:, :, 1] - 128.0
    b = lab_array[:, :, 2] - 128.0
    
    # Stack LAB channels
    lab = np.stack([L, a, b], axis=-1)
    
    # Create valid mask: has alpha and not extreme highlights/shadows
    valid_mask = (alpha > 0) & (L > 5) & (L < 95)
    
    return lab, valid_mask


def _classify_by_lab_threshold(image: Image.Image, debug: bool = False) -> str:
    """
    Classify pottery color using CIELAB thresholds.
    
    Args:
        image: PIL Image with transparent background
        debug: If True, print debug information
        
    Returns:
        Classification label: "Red Pottery", "Black Pottery", or "Uncertain"
    """
    lab, valid_mask = _get_lab_values(image)
    
    # Extract LAB channels
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # Calculate chroma
    C = np.sqrt(a**2 + b**2)
    
    # Also get RGB values for additional heuristics
    img_array = np.array(image)
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Apply thresholds
    # Black: low lightness (regardless of hue for very dark pottery)
    is_black = valid_mask & (L < 50)
    
    # Red pottery detection with multiple strategies:
    # Strategy 1: Standard CIELAB (positive a*, positive b*)
    is_red_standard = valid_mask & (a > 15) & (b > 5) & (L > 30) & (L < 70)
    
    # Strategy 2: RGB-based (R channel dominant, not too dark, not too bright)
    is_red_rgb = valid_mask & (R > G + 20) & (R > B + 20) & (L > 35) & (L < 80)
    
    # Strategy 3: High lightness with high chroma (bright colored pottery)
    # This catches pottery that appears reddish/orange despite unusual LAB values
    is_red_bright = valid_mask & (L > 50) & (C > 100) & (L < 80)
    
    # Combine red strategies
    is_red = is_red_standard | is_red_rgb | is_red_bright
    
    # Count pixels
    num_valid = np.sum(valid_mask)
    num_black = np.sum(is_black)
    num_red = np.sum(is_red)
    
    if num_valid == 0:
        return "Uncertain"
    
    ratio_black = num_black / num_valid
    ratio_red = num_red / num_valid
    
    if debug:
        print(f"Valid pixels: {num_valid}")
        print(f"Black pixels: {num_black} ({ratio_black:.2%})")
        print(f"Red pixels: {num_red} ({ratio_red:.2%})")
        print(f"  - Standard LAB: {np.sum(is_red_standard)} ({np.sum(is_red_standard)/num_valid:.2%})")
        print(f"  - RGB-based: {np.sum(is_red_rgb)} ({np.sum(is_red_rgb)/num_valid:.2%})")
        print(f"  - Bright/Chroma: {np.sum(is_red_bright)} ({np.sum(is_red_bright)/num_valid:.2%})")
        print(f"Mean L*: {L[valid_mask].mean():.2f}")
        print(f"Mean a*: {a[valid_mask].mean():.2f}")
        print(f"Mean b*: {b[valid_mask].mean():.2f}")
        print(f"Mean C*: {C[valid_mask].mean():.2f}")
        print(f"Mean RGB: R={R[valid_mask].mean():.1f}, G={G[valid_mask].mean():.1f}, B={B[valid_mask].mean():.1f}")
    
    # Decision with margin (require at least 55% confidence)
    if ratio_red > 0.55 and ratio_red > ratio_black:
        return "Red Pottery"
    elif ratio_black > 0.55 and ratio_black > ratio_red:
        return "Black Pottery"
    elif ratio_red > ratio_black:
        return "Red Pottery"
    elif ratio_black > ratio_red:
        return "Black Pottery"
    else:
        return "Uncertain"


def _classify_by_kmeans_lab(image: Image.Image, n_clusters: int = 2, debug: bool = False) -> str:
    """
    Classify pottery color using K-means clustering in CIELAB space.
    
    Args:
        image: PIL Image with transparent background
        n_clusters: Number of clusters for K-means (default: 2)
        debug: If True, print debug information
        
    Returns:
        Classification label: "Red Pottery", "Black Pottery", "Mixed", or "Uncertain"
    """
    lab, valid_mask = _get_lab_values(image)
    
    # Get valid pottery pixels
    valid_pixels = lab[valid_mask]
    
    if len(valid_pixels) == 0:
        return "Uncertain"
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(valid_pixels)
    centroids = kmeans.cluster_centers_
    
    # Count pixels in each cluster
    cluster_sizes = np.bincount(labels)
    
    if debug:
        print(f"Total valid pixels: {len(valid_pixels)}")
    
    # Get RGB values for additional heuristics
    img_array = np.array(image)
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    valid_rgb = np.stack([R[valid_mask], G[valid_mask], B[valid_mask]], axis=-1)
    
    # Classify each centroid
    cluster_labels = []
    for i, centroid in enumerate(centroids):
        L_c, a_c, b_c = centroid
        C_c = np.sqrt(a_c**2 + b_c**2)
        
        # Get RGB centroid for this cluster
        cluster_rgb = valid_rgb[labels == i].mean(axis=0)
        R_c, G_c, B_c = cluster_rgb
        
        if debug:
            print(f"\nCluster {i} (size: {cluster_sizes[i]}, {cluster_sizes[i]/len(valid_pixels):.2%}):")
            print(f"  L*: {L_c:.2f}, a*: {a_c:.2f}, b*: {b_c:.2f}, C*: {C_c:.2f}")
            print(f"  RGB: R={R_c:.1f}, G={G_c:.1f}, B={B_c:.1f}")
        
        # Apply same thresholds as lab_threshold method
        # Black: low lightness (dark pottery regardless of hue)
        if L_c < 50:
            label = "Black Pottery"
        # Red pottery detection with multiple strategies:
        # Strategy 1: Standard CIELAB
        elif a_c > 15 and b_c > 5 and L_c >= 30 and L_c < 70:
            label = "Red Pottery"
        elif a_c > 10 and b_c > 0 and L_c >= 25 and L_c < 70:
            label = "Red Pottery"
        # Strategy 2: RGB-based (R channel dominant)
        elif R_c > G_c + 20 and R_c > B_c + 20 and L_c > 35 and L_c < 80:
            label = "Red Pottery"
        # Strategy 3: High lightness with high chroma (bright colored pottery)
        elif L_c > 50 and C_c > 100 and L_c < 80:
            label = "Red Pottery"
        else:
            label = "Uncertain"
        
        cluster_labels.append(label)
        if debug:
            print(f"  Label: {label}")
    
    # Weight by cluster sizes to pick overall label
    weighted_votes = {"Red Pottery": 0, "Black Pottery": 0, "Uncertain": 0}
    for label, size in zip(cluster_labels, cluster_sizes):
        weighted_votes[label] += size
    
    # Determine final label
    total_pixels = sum(cluster_sizes)
    red_ratio = weighted_votes["Red Pottery"] / total_pixels
    black_ratio = weighted_votes["Black Pottery"] / total_pixels
    
    if debug:
        print(f"\nWeighted votes:")
        print(f"  Red: {red_ratio:.2%}")
        print(f"  Black: {black_ratio:.2%}")
    
    # Check if it's mixed (both colors present significantly)
    if red_ratio > 0.3 and black_ratio > 0.3:
        return "Mixed"
    elif red_ratio > black_ratio:
        return "Red Pottery"
    elif black_ratio > red_ratio:
        return "Black Pottery"
    else:
        return "Uncertain"


def _classify_by_clip_vit(
    image: Image.Image,
    candidate_labels: str = "Red Pottery, Black Pottery",
    debug: bool = False
) -> Dict[str, float]:
    """
    Classify pottery color using CLIP vit image classification.
    
    Args:
        image: PIL Image with transparent background
        candidate_labels: Comma-separated string of candidate labels
        debug: If True, print debug information
        
    Returns:
        Dictionary mapping labels to confidence scores
    """
    # Initialize CLIP model
    clip_checkpoint = "openai/clip-vit-base-patch16"
    clip_detector = pipeline(model=clip_checkpoint, task="zero-shot-image-classification")
    
    # Convert candidate_labels string to list
    labels_list = [label.strip() for label in candidate_labels.split(",")]
    
    # Run inference
    clip_out = clip_detector(image, candidate_labels=labels_list)
    
    # Postprocess output
    result = _postprocess(clip_out)
    
    if debug:
        print("CLIP classification results:")
        for label, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {score:.4f}")
    
    return result


def classify_pottery_color(
    image: Image.Image,
    method: Literal["lab_threshold", "kmeans_lab", "clip_vit"] = "lab_threshold",
    candidate_labels: str = "Red Pottery, Black Pottery",
    n_clusters: int = 2,
    debug: bool = False
) -> Dict[str, any]:
    """
    Classify pottery color from an image with transparent background.
    
    Args:
        image: PIL Image with transparent background (RGBA format)
        method: Classification method to use:
            - "lab_threshold": CIELAB color space threshold-based classification
            - "kmeans_lab": K-means clustering in CIELAB space
            - "clip_vit": CLIP-based classification
        candidate_labels: Comma-separated candidate labels for CLIP method
        n_clusters: Number of clusters for kmeans_lab method (default: 2)
        debug: If True, print debug information
        
    Returns:
        Dictionary containing:
            - "label": Classification result
            - "method": Method used
            - "scores": (Optional) Confidence scores for CLIP method
            
    Raises:
        ValueError: If image doesn't have alpha channel or method is invalid
        
    Examples:
        >>> from PIL import Image
        >>> img = Image.open("pottery.png")
        >>> result = classify_pottery_color(img, method="lab_threshold", debug=True)
        >>> print(result["label"])
        'Red Pottery'

        >>> result = classify_pottery_color(img, method="clip_vit", 
        ...                                  candidate_labels="Red, Black, White")
        >>> print(result["scores"])
        {'Red': 0.85, 'Black': 0.10, 'White': 0.05}
    """
    # Validate image format
    if image.mode != 'RGBA':
        raise ValueError(f"Image must be in RGBA format (has transparency). Got: {image.mode}")
    
    # Execute appropriate method
    if method == "lab_threshold":
        label = _classify_by_lab_threshold(image, debug=debug)
        return {
            "label": label,
            "method": method
        }
    
    elif method == "kmeans_lab":
        label = _classify_by_kmeans_lab(image, n_clusters=n_clusters, debug=debug)
        return {
            "label": label,
            "method": method,
            "n_clusters": n_clusters
        }
    
    elif method == "clip_vit":
        scores = _classify_by_clip_vit(image, candidate_labels=candidate_labels, debug=debug)
        # Get the top label
        top_label = max(scores, key=scores.get)
        return {
            "label": top_label,
            "method": method,
            "scores": scores
        }
    
    else:
        raise ValueError(f"Invalid method: {method}. Must be one of: "
                        "'lab_threshold', 'kmeans_lab', 'clip_vit'")
