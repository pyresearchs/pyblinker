"""Per-blink morphology metrics delegated to the shared blink core.
All feature calculations rely only on blink onset and blink duration stored in the metadata.
This design intentionally decouples feature extraction from how blink boundaries are defined.

As a result, users should have full flexibility to define blink onset and duration according to their needs.
See pyblinker/utils/refinement_utils.py

"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from .._core_blink import METHODS_BY_MODALITY, compute_blink_core


def _normalize_methods(modality: str, methods: Iterable[str] | None) -> tuple[str, ...]:
    modality_key = modality.lower()
    allowed = METHODS_BY_MODALITY.get(modality_key, ("base",))
    if methods is None:
        return (allowed[0],)
    ordered: list[str] = []
    for method in methods:
        if method not in ordered:
            ordered.append(method)
    return tuple(ordered) if ordered else (allowed[0],)


def compute_blink_waveform_metrics(
    segment: np.ndarray | Mapping[str, np.ndarray],
    sfreq: float,
    *,
    methods: Iterable[str] | None = None,
    modality: str = "eeg",
    include_second_derivative: bool = True,
    use_abs_for_thresholds_and_areas: bool = True,
) -> Dict[str, float]:
    """Compute morphology-oriented blink metrics for selected methods.

    The signature mirrors :func:`pyblinker.blink_features.kinematics.per_blink.
    compute_segment_kinematics` so callers can interchange the two depending on
    their feature subset needs. The returned key space is identical to the
    kinematic helper and all signal analytics are delegated to
    :func:`pyblinker.blink_features._core_blink.compute_blink_core`.
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
