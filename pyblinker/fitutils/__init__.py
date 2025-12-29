"""MATLAB-style helper functions ported to Python."""

from .forking import (
    corr,
    get_intersection,
    polyfit,
    polyval,
    weighted_corr,
    mad,
)
from .line_intersection import lines_intersection
from .ear_crossing import (
    CrossingPoint,
    ThresholdCrossingError,
    ThresholdCrossingResult,
    compute_threshold_slopes,
    find_threshold_crossing_triplet,
    linear_interpolated_crossing,
)

__all__ = [
    "corr",
    "get_intersection",
    "polyfit",
    "polyval",
    "weighted_corr",
    "lines_intersection",
    "mad",
    "CrossingPoint",
    "ThresholdCrossingError",
    "ThresholdCrossingResult",
    "compute_threshold_slopes",
    "find_threshold_crossing_triplet",
    "linear_interpolated_crossing",
]
