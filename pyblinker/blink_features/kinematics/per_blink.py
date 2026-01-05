"""Per-blink kinematic metrics delegated to the shared blink core."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from .._core_blink import METHODS_BY_MODALITY, compute_blink_core


def _normalize_methods(modality: str, methods: Iterable[str] | None) -> tuple[str, ...]:
    modality_key = modality.lower()
    allowed = METHODS_BY_MODALITY.get(modality_key, ("base",))
    if methods is None:
        return allowed # If method is none, return all allowed methods
    ordered = []
    for method in methods:
        if method not in ordered:
            ordered.append(method)
    return tuple(ordered) if ordered else (allowed[0],)


def compute_segment_kinematics(
    segment: np.ndarray | Mapping[str, np.ndarray],
    sfreq: float,
    *,
    methods: Iterable[str] | None = None,
    modality: str = "eeg",
    include_second_derivative: bool = True,
    use_abs_for_thresholds_and_areas: bool = True,
) -> Dict[str, float]:
    """Compute blink kinematic metrics for one or more segmentation methods.

    Parameters
    ----------
    segment
        Either a one-dimensional array representing the blink waveform or a
        mapping of ``{method: segment}``. When a single array is provided the
        function defaults to the first supported method for the modality (base
        for EEG/EOG, base for EAR). Passing a mapping allows callers to provide
        distinct start/end windows for multiple methods in a single call.
    sfreq
        Sampling frequency of the provided segments.
    methods
        Optional iterable of method names to evaluate when ``segment`` is an
        array. Ignored when ``segment`` is a mapping.
    modality
        Recording modality. ``"eeg"`` (default) enables zero-based metrics
        whereas ``"ear"`` (Eye Aspect Ratio) suppresses them.
    include_second_derivative
        Forwarded to :func:`pyblinker.blink_features._core_blink.compute_blink_core`.
    use_abs_for_thresholds_and_areas
        Forwarded to the shared core. Ignored for EAR data where dip magnitude
        is computed relative to the local baseline.

    Returns
    -------
    dict
        Mapping of metric names with method suffixes to floating point values.
    """

    if isinstance(segment, Mapping):
        segments_by_method = {
            method: np.asarray(data, dtype=float).reshape(-1)
            for method, data in segment.items()
        }
        method_order = tuple(segments_by_method.keys())
    else:
        seg_array = np.asarray(segment, dtype=float).reshape(-1)
        method_order = _normalize_methods(modality, methods)
        segments_by_method = {method: seg_array for method in method_order}

    if not segments_by_method:
        method_order = _normalize_methods(modality, None)
        if isinstance(segment, Mapping):
            seg_array = np.asarray([], dtype=float)
        else:
            seg_array = np.asarray(segment, dtype=float).reshape(-1)
        segments_by_method = {method_order[0]: seg_array}

    metrics: Dict[str, float] = {}
    modality_key = modality.lower()
    for method in method_order:
        metrics.update(
            compute_blink_core(
                segments_by_method[method],
                sfreq,
                start_end_method=method,
                modality=modality_key,
                include_second_derivative=include_second_derivative,
                use_abs_for_thresholds_and_areas=use_abs_for_thresholds_and_areas,
            )
        )
    return metrics
