# ascota_core.mobilesam_wand

The `mobilesam_wand` module provides an interactive MobileSAM session API for
point-prompt segmentation.

It supports:

- lazy model loading with checkpoint path resolution,
- per-image predictor setup (`set_image`), and
- iterative multi-point mask prediction (`predict`) for foreground/background edits.
---

::: ascota_core.mobilesam_wand
    options:
      members_order: source
      filters: ["!^__.*__$"]
