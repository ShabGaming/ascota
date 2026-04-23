"""
Color-based clustering module for images with transparent backgrounds.

This module clusters similar images using a fast pipeline:
1. Convert each image to CIE Lab and compute per-image summary statistics
   (mean and std of L, a, b over opaque pixels only).
2. Stack summaries and reduce dimension with PCA.
3. Cluster the PCA-transformed features with HDBSCAN.

Designed for near-realtime use: Lab summary and PCA are cheap; HDBSCAN
is run on reduced-dimensional data.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
from sklearn.decomposition import PCA
import hdbscan

# Defaults for feature extraction
DEFAULT_ALPHA_THRESHOLD = 128
DEFAULT_RESIZE_MAX = 512
DEFAULT_PCA_COMPONENTS = 0.95  # fraction of variance to retain

# Defaults for HDBSCAN (plan parameters)
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = None  # None => use min_cluster_size
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0
DEFAULT_CLUSTER_SELECTION_METHOD = "eom"


def _image_to_lab_summary(
    image: Image.Image,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    resize_max: Optional[int] = DEFAULT_RESIZE_MAX,
) -> np.ndarray:
    """
    Compute a fixed-length Lab color summary for one image (opaque pixels only).

    Args:
        image: PIL Image (RGBA or RGB). If RGBA, only pixels with alpha >= alpha_threshold are used.
        alpha_threshold: Pixels with alpha >= this value are included in the summary (0-255).
        resize_max: If set, resize image so the longer side is at most this many pixels (for speed).

    Returns:
        Length-6 vector: (mean_L, mean_a, mean_b, std_L, std_a, std_b). If no opaque pixels,
        returns zeros.
    """
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        rgb = np.array(background)
        alpha = np.array(image.split()[-1])
    elif image.mode == "RGB":
        rgb = np.array(image)
        alpha = np.full((image.height, image.width), 255, dtype=np.uint8)
    else:
        image = image.convert("RGB")
        rgb = np.array(image)
        alpha = np.full((image.height, image.width), 255, dtype=np.uint8)

    if resize_max is not None:
        h, w = rgb.shape[:2]
        if max(h, w) > resize_max:
            scale = resize_max / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    mask = alpha >= alpha_threshold
    if not np.any(mask):
        return np.zeros(6, dtype=np.float64)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float64)[mask]
    a = lab[:, :, 1].astype(np.float64)[mask]
    b = lab[:, :, 2].astype(np.float64)[mask]

    mean_L, mean_a, mean_b = np.mean(L), np.mean(a), np.mean(b)
    std_L = np.std(L)
    std_a = np.std(a)
    std_b = np.std(b)
    # Avoid zeros for stability (e.g. flat color)
    std_L = max(std_L, 1e-6)
    std_a = max(std_a, 1e-6)
    std_b = max(std_b, 1e-6)

    return np.array([mean_L, mean_a, mean_b, std_L, std_a, std_b], dtype=np.float64)


def extract_lab_pca_features(
    images: List[Image.Image],
    pca_components: Union[int, float] = DEFAULT_PCA_COMPONENTS,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    resize_max: Optional[int] = DEFAULT_RESIZE_MAX,
) -> Tuple[np.ndarray, PCA]:
    """
    Extract Lab-based summary features and reduce dimension with PCA.

    Args:
        images: List of PIL Images (RGBA or RGB; transparent backgrounds supported).
        pca_components: Number of components (int) or fraction of variance to retain (float in (0, 1]).
        alpha_threshold: Alpha threshold for opaque pixels (0-255).
        resize_max: Max size of the longer side before computing Lab summary; None to disable.

    Returns:
        Tuple of (feature_matrix, pca):
        - feature_matrix: shape (n_images, n_components), float64.
        - pca: Fitted sklearn PCA object (for transforming new images if needed).

    Raises:
        ValueError: If images list is empty.
    """
    if not images:
        raise ValueError("images list must not be empty")

    summaries = []
    for im in images:
        s = _image_to_lab_summary(im, alpha_threshold=alpha_threshold, resize_max=resize_max)
        summaries.append(s)
    X = np.stack(summaries, axis=0)

    n_components = pca_components
    n_features = X.shape[1]
    if isinstance(n_components, float):
        n_comp_int = min(n_features, max(1, int(n_features * n_components)))
        pca = PCA(n_components=n_comp_int)
    else:
        n_comp_int = min(n_components, n_features)
        pca = PCA(n_components=n_comp_int)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def cluster_images_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: Optional[int] = DEFAULT_MIN_SAMPLES,
    cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
    cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
    **kwargs: Any,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Cluster samples by features using HDBSCAN.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        min_cluster_size: Minimum size of clusters; smaller groups are treated as noise.
        min_samples: Core point neighborhood size; if None, defaults to min_cluster_size.
        cluster_selection_epsilon: Distance threshold for merging clusters (0 = no merge by epsilon).
        cluster_selection_method: "eom" (excess of mass) or "leaf".
        **kwargs: Additional arguments passed to hdbscan.HDBSCAN.

    Returns:
        Tuple of (labels, clusters):
        - labels: Array of length n_samples; -1 indicates noise.
        - clusters: List of lists of indices; each inner list is one cluster (excluding noise).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        **kwargs,
    )
    labels = clusterer.fit_predict(features)

    clusters: List[List[int]] = []
    unique = np.unique(labels)
    for c in unique:
        if c == -1:
            continue
        indices = np.where(labels == c)[0].tolist()
        clusters.append(indices)
    # Sort by cluster label so order is stable
    clusters.sort(key=lambda idx_list: labels[idx_list[0]] if idx_list else -1)
    return labels, clusters


def cluster_similar_images(
    images: List[Image.Image],
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: Optional[int] = DEFAULT_MIN_SAMPLES,
    cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
    cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
    pca_components: Union[int, float] = DEFAULT_PCA_COMPONENTS,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    resize_max: Optional[int] = DEFAULT_RESIZE_MAX,
    return_pca_and_labels: bool = False,
    **hdbscan_kwargs: Any,
) -> Union[Tuple[List[List[int]], List[int]], Tuple[List[List[int]], List[int], PCA, np.ndarray]]:
    """
    High-level pipeline: extract Lab+PCA features and cluster with HDBSCAN.

    Args:
        images: List of PIL Images (transparent backgrounds supported).
        min_cluster_size: HDBSCAN min_cluster_size.
        min_samples: HDBSCAN min_samples (None => use min_cluster_size).
        cluster_selection_epsilon: HDBSCAN cluster_selection_epsilon.
        cluster_selection_method: HDBSCAN cluster_selection_method ("eom" or "leaf").
        pca_components: PCA components (int or variance fraction float).
        alpha_threshold: Alpha threshold for opaque pixels.
        resize_max: Max longer side for Lab summary; None to disable.
        return_pca_and_labels: If True, also return fitted PCA and label array.
        **hdbscan_kwargs: Passed to HDBSCAN.

    Returns:
        If return_pca_and_labels is False: (clusters, noise_indices).
        If return_pca_and_labels is True: (clusters, noise_indices, pca, labels).

        - clusters: List of lists of image indices (one list per cluster).
        - noise_indices: List of image indices labeled as noise (label -1).
    """
    if not images:
        return ([], []) if not return_pca_and_labels else ([], [], None, np.array([]))

    features, pca = extract_lab_pca_features(
        images,
        pca_components=pca_components,
        alpha_threshold=alpha_threshold,
        resize_max=resize_max,
    )
    labels, clusters = cluster_images_hdbscan(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        **hdbscan_kwargs,
    )
    noise_indices = np.where(labels == -1)[0].tolist()

    if return_pca_and_labels:
        return clusters, noise_indices, pca, labels
    return clusters, noise_indices
