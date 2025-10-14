
"""ascota_classification package

Public API for color classification routines used by the ascota project.

This package currently exposes the `classify_pottery_color` convenience
function and lower-level classification methods implemented in
`ascota_classification.color`.
"""

from .color import (
	classify_pottery_color,
	_classify_by_lab_threshold,
	_classify_by_kmeans_lab,
	_classify_by_clip_vit,
)

__all__ = [
	"classify_pottery_color",
	# exported for testing/advanced usage
	"_classify_by_lab_threshold",
	"_classify_by_kmeans_lab",
	"_classify_by_clip_vit",
]

__version__ = "0.1.0"
__author__ = "Muhammad Shahab Hasan"
