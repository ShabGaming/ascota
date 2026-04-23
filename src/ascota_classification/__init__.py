"""ascota_classification package

Public API for classification and clustering routines used by the ascota project.

Color clustering (Lab + PCA + HDBSCAN) is implemented in `ascota_classification.color`:
- cluster_similar_images: high-level pipeline for clustering similar images
- extract_lab_pca_features: Lab summary + PCA feature extraction
- cluster_images_hdbscan: HDBSCAN clustering on a feature matrix

Texture clustering (LBP + GLCM + StandardScaler + PCA + HDBSCAN) is in `ascota_classification.texture`:
- cluster_similar_images_by_texture: high-level texture-only pipeline
- extract_texture_pca_features: texture vectors + scaling + PCA
"""

from .color import (
    cluster_similar_images,
    cluster_images_hdbscan,
    extract_lab_pca_features,
)
from .texture import (
    cluster_similar_images_by_texture,
    extract_texture_pca_features,
)

__all__ = [
    "cluster_similar_images",
    "cluster_images_hdbscan",
    "extract_lab_pca_features",
    "cluster_similar_images_by_texture",
    "extract_texture_pca_features",
]

__version__ = "0.1.0"
__author__ = "Muhammad Shahab Hasan"
