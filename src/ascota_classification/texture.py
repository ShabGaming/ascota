"""
Texture-based clustering for images with transparent backgrounds.

Clusters similar pottery by surface texture only (grayscale / luminance):
1. Luminance and alpha mask; crop to the opaque bounding box (no white compositing).
2. Local Binary Pattern (LBP) histograms at two radii + Gray-Level Co-occurrence Matrix
   (GLCM) properties (mean over distances and angles).
3. Standardize features, reduce dimension with PCA.
4. Cluster with HDBSCAN (reuse ``cluster_images_hdbscan`` from ``color``).

``eom`` tends to merge clusters and often leaves fewer noise points; ``leaf`` extracts
finer structure but can label many points as noise unless ``min_samples`` is low (e.g. 1).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Any, List, Optional, Tuple, Union

import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .color import cluster_images_hdbscan

# Feature extraction
DEFAULT_ALPHA_THRESHOLD = 128
DEFAULT_RESIZE_MAX = 512
DEFAULT_PCA_COMPONENTS = 0.95

# LBP: uniform patterns, P=8, two radii
LBP_POINTS = 8
LBP_RADII = (1, 2)
LBP_HIST_BINS = LBP_POINTS + 2  # uniform LBP

# GLCM
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2)
GLCM_ANGLES_RAD = (
    0.0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")

# HDBSCAN defaults: min_samples=1 keeps density reachable so fewer points become noise
# (min_samples=2 is often too strict for modest-sized uploads). Pair with leaf or eom in UI.
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_MIN_SAMPLES = 1
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0
DEFAULT_CLUSTER_SELECTION_METHOD = "leaf"


def _luminance_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma from uint8 RGB, shape (H, W)."""
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def _prepare_rgb_and_mask(
    image: Image.Image,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    resize_max: Optional[int] = DEFAULT_RESIZE_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RGB array and boolean mask of opaque pixels. No white compositing (texture-only).
    """
    if image.mode == "RGBA":
        rgb = np.array(image.convert("RGB"))
        alpha = np.array(image.split()[-1])
    elif image.mode == "RGB":
        rgb = np.array(image)
        alpha = np.full((image.height, image.width), 255, dtype=np.uint8)
    else:
        rgb = np.array(image.convert("RGB"))
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
    return rgb, mask


def _image_to_texture_summary(
    image: Image.Image,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    resize_max: Optional[int] = DEFAULT_RESIZE_MAX,
) -> np.ndarray:
    """
    Fixed-length texture vector: LBP histograms (two scales) + pooled GLCM properties.

    Uses luminance from RGB only; crops to opaque bounding box.
    """
    rgb, mask = _prepare_rgb_and_mask(
        image, alpha_threshold=alpha_threshold, resize_max=resize_max
    )
    if not np.any(mask):
        return np.zeros(_texture_feature_dim(), dtype=np.float64)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    rgb_c = rgb[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]

    gray = _luminance_from_rgb(rgb_c)
    fg_mean = float(np.mean(gray[mask_c]))
    gray_filled = gray.copy()
    gray_filled[~mask_c] = np.uint8(np.clip(round(fg_mean), 0, 255))

    if gray_filled.size < 4 or mask_c.sum() < 16:
        return np.zeros(_texture_feature_dim(), dtype=np.float64)

    parts: List[np.ndarray] = []

    for radius in LBP_RADII:
        lbp = local_binary_pattern(
            gray_filled, LBP_POINTS, radius, method="uniform"
        )
        lbp_vals = lbp[mask_c].ravel()
        hist, _ = np.histogram(
            lbp_vals,
            bins=LBP_HIST_BINS,
            range=(0, LBP_HIST_BINS),
        )
        hist = hist.astype(np.float64)
        s = hist.sum()
        if s > 0:
            hist /= s
        parts.append(hist)

    gq = (gray_filled.astype(np.float64) / 255.0 * (GLCM_LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(
        gq,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES_RAD),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    glcm_feats: List[float] = []
    for prop in GLCM_PROPS:
        p = graycoprops(glcm, prop)
        val = float(np.nanmean(p))
        if not np.isfinite(val):
            val = 0.0
        glcm_feats.append(val)
    parts.append(np.array(glcm_feats, dtype=np.float64))

    return np.concatenate(parts, axis=0).astype(np.float64)


def _texture_feature_dim() -> int:
    return len(LBP_RADII) * LBP_HIST_BINS + len(GLCM_PROPS)


def extract_texture_pca_features(
    images: List[Image.Image],
    pca_components: Union[int, float] = DEFAULT_PCA_COMPONENTS,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    resize_max: Optional[int] = DEFAULT_RESIZE_MAX,
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """
    Extract texture features (LBP + GLCM), standardize, and reduce with PCA.

    Returns:
        Tuple of (feature_matrix, pca, scaler):
        - feature_matrix: shape (n_images, n_components), float64.
        - pca: Fitted PCA on standardized features.
        - scaler: Fitted StandardScaler on raw texture vectors.

    Raises:
        ValueError: If images list is empty.
    """
    if not images:
        raise ValueError("images list must not be empty")

    summaries = []
    for im in images:
        s = _image_to_texture_summary(
            im, alpha_threshold=alpha_threshold, resize_max=resize_max
        )
        summaries.append(s)
    X = np.stack(summaries, axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = pca_components
    n_features = X_scaled.shape[1]
    n_samples = X_scaled.shape[0]
    if isinstance(n_components, float):
        n_comp_int = min(n_features, max(1, int(n_features * n_components)))
    else:
        n_comp_int = min(n_components, n_features)
    n_comp_int = min(n_comp_int, max(1, n_samples))
    pca = PCA(n_components=n_comp_int)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca, scaler


def cluster_similar_images_by_texture(
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
) -> Union[
    Tuple[List[List[int]], List[int]],
    Tuple[List[List[int]], List[int], PCA, np.ndarray, StandardScaler],
]:
    """
    High-level pipeline: texture features + StandardScaler + PCA + HDBSCAN.

    Returns:
        If return_pca_and_labels is False: (clusters, noise_indices).
        If True: (clusters, noise_indices, pca, labels, scaler).
    """
    if not images:
        if not return_pca_and_labels:
            return [], []
        return [], [], None, np.array([]), None  # type: ignore[return-value]

    features, pca, scaler = extract_texture_pca_features(
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
        return clusters, noise_indices, pca, labels, scaler
    return clusters, noise_indices
