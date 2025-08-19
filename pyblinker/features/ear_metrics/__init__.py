"""EAR baseline and extrema aggregation."""

from .aggregate import aggregate_ear_features
from .features import ear_before_blink_avg_epoch, ear_extrema_epoch

__all__ = [
    "aggregate_ear_features",
    "ear_before_blink_avg_epoch",
    "ear_extrema_epoch",
]
