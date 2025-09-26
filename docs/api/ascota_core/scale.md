# ascota_core.scale

The `scale` module handles **geometric scaling** tasks. It detects measurement
cards in an image, computes a **pixels-per-centimeter ratio**, and uses this
calibration to estimate the **surface area of pottery sherds**. These
functions provide the quantitative backbone for later size normalization and
comparisons.

---

::: ascota_core.scale
    options:
      members_order: source
      filters: ["!^__.*__$"]
