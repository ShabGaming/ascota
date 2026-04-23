# ascota_classification

`ascota_classification` is the **second stage of the ASCOTA pipeline**.
It provides model-based classification and feature clustering utilities for
preprocessed pottery imagery.

## Purpose

This package focuses on:

- **Type classification**
  Multi-stage pottery type classification pipelines, including optional
  appendage subtype refinement.
- **Decoration classification**
  DINOv2 feature extraction with a trained classifier for decoration labels.
- **Color clustering**
  Lab-based feature extraction and HDBSCAN clustering for visually similar images.
- **Texture clustering**
  Texture feature extraction (LBP + GLCM), PCA projection, and HDBSCAN grouping.

## Submodules

- [color](color.md): Lab feature extraction and color similarity clustering.
- [texture](texture.md): texture feature extraction and clustering.
- [decoration](decoration.md): pottery decoration classification.
- [type](type.md): pottery type and pottery/non-pottery classification.