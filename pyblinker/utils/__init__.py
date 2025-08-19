"""Utility functions for pyblinker."""
from .segments import slice_raw_to_segments
from .epochs import (
    slice_raw_into_epochs,
    save_epoch_raws,
    generate_epoch_report,
    slice_into_mini_raws,
)
from .refinement import (
    refine_ear_extrema_and_threshold_stub,
    refine_local_maximum_stub,
    refine_blinks_from_epochs,
    plot_refined_blinks,
)
from .raw_preprocessing import prepare_refined_segments
from .misc import create_annotation

__all__ = [
    "slice_raw_to_segments",
    "slice_raw_into_epochs",
    "save_epoch_raws",
    "generate_epoch_report",
    "slice_into_mini_raws",
    "refine_ear_extrema_and_threshold_stub",
    "refine_local_maximum_stub",
    "refine_blinks_from_epochs",
    "plot_refined_blinks",
    "prepare_refined_segments",
    "create_annotation",
]
