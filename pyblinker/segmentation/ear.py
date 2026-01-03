"""EAR-specific refinement helpers used by segmentation pipelines."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pyblinker.blink_features.blink_events.blink_dataframe import (
    compute_outer_bounds,
)
from pyblinker.blink_features.ear_metrics.refinement import (
    EARRefinementConfig,
    EARThresholdBlinkRefiner,
    _progressive_search,
)
from pyblinker.logging import get_logger

from ..utils.dict_utils import append_to_slot

logger = get_logger(__name__)


def _select_seg_type(seg_type_value: Any) -> Optional[str]:
    """Normalize segmentation type configuration."""

    if isinstance(seg_type_value, str):
        return seg_type_value
    if isinstance(seg_type_value, Sequence):
        seg_types = [str(item) for item in seg_type_value]
        if not seg_types:
            return None
        if "threshold_interpolation" in seg_types:
            return "threshold_interpolation"
        return seg_types[0]
    return None


def _locate_trough(signal: np.ndarray, start: int, end: int) -> Optional[int]:
    """Return the index of the minimum value between ``start`` and ``end``."""

    if signal.size == 0:
        return None
    n = len(signal)
    start_idx = int(np.clip(start, 0, n - 1))
    end_idx = int(np.clip(end, 0, n - 1))
    if end_idx < start_idx:
        start_idx = end_idx
    window = signal[start_idx : end_idx + 1]
    if window.size == 0:
        return start_idx
    finite_mask = np.isfinite(window)
    if not finite_mask.any():
        return start_idx
    local_min = int(np.nanargmin(window))
    return start_idx + local_min


def _empty_interpolated_thresholds() -> Dict[str, float | int | bool]:
    """Return default interpolated threshold metadata with NaN/False values."""

    return {
        "left_interpolated_threshold": float("nan"),
        "right_interpolated_threshold": float("nan"),
        "left_interpolated_threshold_sample": float("nan"),
        "right_interpolated_threshold_sample": float("nan"),
        "left_interpolated_threshold_found": False,
        "right_interpolated_threshold_found": False,
        "interpolated_thresholds_found": False,
    }


def _append_outer_bounds_from_peaks(
    md: Dict[str, List[Any]],
    epoch_index: int,
    peaks: Sequence[int],
    key_prefix: str,
    n_samp_epoch: int,
) -> None:
    if not peaks:
        return
    bounds = compute_outer_bounds(peaks, n_samp_epoch)
    for outer_start, outer_end in bounds:
        md[f"blink_outer_start_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_outer_start_{key_prefix}"][epoch_index], outer_start
        )
        md[f"blink_outer_end_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_outer_end_{key_prefix}"][epoch_index], outer_end
        )


def _refine_ear_blinks_for_epoch(
    segment: np.ndarray,
    blink_starts: Sequence[int],
    blink_ends: Sequence[int],
    sfreq: float,
    segmentation_config: Optional[dict],
) -> List[Dict[str, Any]]:
    """Refine EAR blinks for a single epoch based on segmentation settings."""

    if segment.size == 0 or not blink_starts:
        return []

    ear_config = (segmentation_config or {}).get("ear", {}) if segmentation_config is not None else {}
    seg_type = _select_seg_type(ear_config.get("seg_type"))
    use_threshold_interpolation = seg_type

    def _fallback_refinement(coarse_start: int, coarse_end: int) -> Dict[str, Any]:
        n = len(segment)
        if n == 0:
            start = end = 0
        else:
            start = int(np.clip(coarse_start, 0, n - 1))
            end = int(np.clip(coarse_end, 0, n - 1))
            if end < start:
                end = start
        return {
            "refined_start_sample": int(start),
            "refined_end_sample": int(end),
            "refined_left_threshold": int(start),
            "refined_right_threshold": int(end),
            "search_window_start_sample": int(start),
            "search_window_end_sample": int(end),
            "search_window_start_time": float(start / sfreq),
            "search_window_end_time": float(end / sfreq),
            "refinement_succeeded": False,
            "search_exhausted": True,
            "extension_seconds_used": 0.0,
            "extension_attempts": 0,
        }

    threshold = ear_config.get("threshold")
    if use_threshold_interpolation and threshold is None:
        logger.warning("EAR threshold missing for threshold_interpolation; falling back to coarse bounds.")
        use_threshold_interpolation = False

    config: EARRefinementConfig | None = None
    refiner: EARThresholdBlinkRefiner | None = None
    if use_threshold_interpolation:
        config = EARRefinementConfig(threshold=threshold)
        override_fields = {
            k: v
            for k, v in ear_config.items()
            if k
            in {
                "annotation_time_unit",
                "max_extension",
                "extension_step",
                "padding",
                "extend_before",
                "extend_after",
            }
        }
        if override_fields:
            config = replace(config, **override_fields)
        refiner = EARThresholdBlinkRefiner(segment, sfreq, config)

    refinements: List[Dict[str, Any]] = []
    for coarse_start, coarse_end in zip(blink_starts, blink_ends):
        if use_threshold_interpolation and config is not None:
            refinement = _progressive_search(
                segment,
                threshold,
                coarse_start,
                coarse_end,
                sfreq,
                config,
            )
        else:
            refinement = _fallback_refinement(coarse_start, coarse_end)

        trough_sample = _locate_trough(
            segment,
            refinement["refined_start_sample"],
            refinement["refined_end_sample"],
        )
        refinement["refined_trough_sample"] = trough_sample
        refinement["refined_lowest_point_sample"] = trough_sample
        if use_threshold_interpolation and refiner is not None:
            interp = refiner._compute_interpolated_threshold_crossings(  # noqa: SLF001
                refined_start_sample=refinement["refined_start_sample"],
                refined_end_sample=refinement["refined_end_sample"],
                lowest_point_sample=trough_sample if trough_sample is not None else float("nan"),
            )
        else:
            interp = _empty_interpolated_thresholds()
        refinement.update(interp)
        refinements.append(refinement)

    return refinements


def _append_ear_refinements(
    md: Dict[str, List[Any]],
    epoch_index: int,
    refinements: Sequence[Dict[str, Any]],
    sfreq: float,
    n_samp_epoch: int,
) -> None:
    if not refinements:
        return

    peaks: List[int] = []
    for refinement in refinements:
        start = refinement["refined_start_sample"]
        end = refinement["refined_end_sample"]
        trough = refinement.get("refined_trough_sample")
        lowest = refinement.get("refined_lowest_point_sample", trough)
        left_time_raw = refinement.get("left_interpolated_threshold")
        right_time_raw = refinement.get("right_interpolated_threshold")
        left_time = float(left_time_raw) if left_time_raw is not None else float("nan")
        right_time = float(right_time_raw) if right_time_raw is not None else float("nan")
        left_sample = refinement.get("left_interpolated_threshold_sample")
        right_sample = refinement.get("right_interpolated_threshold_sample")

        has_interp = np.isfinite(left_time) and np.isfinite(right_time)
        onset_time = float(left_time) if np.isfinite(left_time) else start / sfreq
        duration_time = (
            max(0.0, float(right_time) - float(left_time))
            if has_interp
            else max(0.0, (end - start) / sfreq)
        )

        peaks.append(int(trough) if trough is not None else int(start))
        md["blink_onset_ear"][epoch_index] = append_to_slot(
            md["blink_onset_ear"][epoch_index], onset_time
        )
        md["blink_duration_ear"][epoch_index] = append_to_slot(
            md["blink_duration_ear"][epoch_index], duration_time
        )
        md["blink_onset_extremum_ear"][epoch_index] = append_to_slot(
            md["blink_onset_extremum_ear"][epoch_index],
            (trough if trough is not None else start) / sfreq,
        )
        md["refined_start_sample"][epoch_index] = append_to_slot(
            md["refined_start_sample"][epoch_index], int(start)
        )
        md["refined_end_sample"][epoch_index] = append_to_slot(
            md["refined_end_sample"][epoch_index], int(end)
        )
        md["refined_lowest_point_sample"][epoch_index] = append_to_slot(
            md["refined_lowest_point_sample"][epoch_index],
            int(lowest) if lowest is not None and np.isfinite(lowest) else np.nan,
        )
        md["refined_left_threshold"][epoch_index] = append_to_slot(
            md["refined_left_threshold"][epoch_index], int(refinement["refined_left_threshold"])
        )
        md["refined_right_threshold"][epoch_index] = append_to_slot(
            md["refined_right_threshold"][epoch_index], int(refinement["refined_right_threshold"])
        )
        md["left_interpolated_threshold"][epoch_index] = append_to_slot(
            md["left_interpolated_threshold"][epoch_index], float(left_time)
        )
        md["right_interpolated_threshold"][epoch_index] = append_to_slot(
            md["right_interpolated_threshold"][epoch_index], float(right_time)
        )
        md["left_interpolated_threshold_sample"][epoch_index] = append_to_slot(
            md["left_interpolated_threshold_sample"][epoch_index],
            float(left_sample) if left_sample is not None else float("nan"),
        )
        md["right_interpolated_threshold_sample"][epoch_index] = append_to_slot(
            md["right_interpolated_threshold_sample"][epoch_index],
            float(right_sample) if right_sample is not None else float("nan"),
        )
        md["left_interpolated_threshold_found"][epoch_index] = append_to_slot(
            md["left_interpolated_threshold_found"][epoch_index],
            bool(refinement.get("left_interpolated_threshold_found", False)),
        )
        md["right_interpolated_threshold_found"][epoch_index] = append_to_slot(
            md["right_interpolated_threshold_found"][epoch_index],
            bool(refinement.get("right_interpolated_threshold_found", False)),
        )
        md["interpolated_thresholds_found"][epoch_index] = append_to_slot(
            md["interpolated_thresholds_found"][epoch_index],
            bool(refinement.get("interpolated_thresholds_found", False)),
        )
        md["search_window_start_sample"][epoch_index] = append_to_slot(
            md["search_window_start_sample"][epoch_index],
            int(refinement["search_window_start_sample"]),
        )
        md["search_window_end_sample"][epoch_index] = append_to_slot(
            md["search_window_end_sample"][epoch_index],
            int(refinement["search_window_end_sample"]),
        )
        md["search_window_start_time"][epoch_index] = append_to_slot(
            md["search_window_start_time"][epoch_index],
            float(refinement["search_window_start_time"]),
        )
        md["search_window_end_time"][epoch_index] = append_to_slot(
            md["search_window_end_time"][epoch_index],
            float(refinement["search_window_end_time"]),
        )
        md["refinement_succeeded"][epoch_index] = append_to_slot(
            md["refinement_succeeded"][epoch_index],
            bool(refinement["refinement_succeeded"]),
        )
        md["search_exhausted"][epoch_index] = append_to_slot(
            md["search_exhausted"][epoch_index],
            bool(refinement["search_exhausted"]),
        )
        md["extension_seconds_used"][epoch_index] = append_to_slot(
            md["extension_seconds_used"][epoch_index],
            float(refinement["extension_seconds_used"]),
        )
        md["extension_attempts"][epoch_index] = append_to_slot(
            md["extension_attempts"][epoch_index],
            int(refinement["extension_attempts"]),
        )

    _append_outer_bounds_from_peaks(md, epoch_index, peaks, "ear", n_samp_epoch)


__all__ = [
    "_append_ear_refinements",
    "_append_outer_bounds_from_peaks",
    "_empty_interpolated_thresholds",
    "_locate_trough",
    "_refine_ear_blinks_for_epoch",
    "_select_seg_type",
]
