# ascota_core

`ascota_core` is the **first stage of the ASCOTA pipeline**.  
It provides the fundamental computer vision and analysis utilities that later
modules build upon.

## Purpose

This package contains tools for:

- **Segmentation & detection**  
  Identifying and isolating key elements in input images (e.g., sherds, measurement cards).  
  Uses models like **SAM (Segment Anything Model)** together with **OpenCV**-based
  preprocessing.

- **Color card classification & correction**  
  Detecting reference color cards, classifying patches, and applying
  color correction to normalize input images across lighting conditions.

- **Geometric scaling & surface estimation**  
  Leveraging detected measurement cards to compute **pixels-per-centimeter**
  ratios and estimate the **surface area of sherds/pottery fragments**.

## Role in the pipeline

`ascota_core` is the core foundational package that **prepares and standardizes
input images** for downstream classification and analysis. It ensures that:
- Sherds are properly segmented from the background.
- Color profiles are normalized to a standard reference.
- Geometric measurements are calibrated to real-world units (via scale cards).

Together, these ensure later classification is performed on calibrated, comparable data.

## Submodules

- [color](color.md): color correction to standard reference and color based clustering.  
- [imaging](imaging.md): segmentation utilities and color card detection.  
- [scale](scale.md): scale estimation and pixel-to-centimeter conversion.  
- [utils](utils.md): shared helpers and lower-level routines.
