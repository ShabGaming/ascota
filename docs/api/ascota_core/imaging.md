# ascota_core.imaging

The `imaging` module implements **segmentation utilities** and **color card
detection**. It leverages different models and traditional computer vision
techniques (OpenCV) to isolate pottery sherds, measurement cards, and other
regions of interest. We also use template matching to enhance detection accuracy.
This module is crucial for preparing images for subsequent color correction 
and scale estimation. Also includes utilities for generating image swatches of
segmented sherds.

---

::: ascota_core.imaging
    options:
      members_order: source
      filters: ["!^__.*__$"]
