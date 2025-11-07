"""Formatting, reporting, and visualization utilities for blink event comparisons.

Internally, comparisons rely on sample index units (1-based). Reporting helpers may
convert sample indices to time (seconds) using a caller-provided sampling rate for
readability.
"""

from __future__ import annotations

from typing import Iterable, Optional

import logging

import mne
import numpy as np
import pandas as pd

from . import similarity
from .similarity import Alignment


logger = logging.getLogger(__name__)


def _to_seconds(sample_index: Optional[float], sampling_rate_hz: float) -> Optional[float]:
    if sample_index is None:
        return None
    return (float(sample_index) - 1.0) / sampling_rate_hz


def _duration_seconds(start_sample: int, end_sample: int, sampling_rate_hz: float) -> float:
    return (float(end_sample) - float(start_sample) + 1.0) / sampling_rate_hz


def preview_side_by_side(
    detected_df: pd.DataFrame, ground_truth_df: pd.DataFrame, n: int
) -> pd.DataFrame:
    """Return the first ``n`` blink intervals from each table for quick inspection.

    All values are shown in sample index units (1-based).
    """

    similarity.validate_event_table(detected_df)
    similarity.validate_event_table(ground_truth_df)

    n = max(0, int(n))
    return pd.DataFrame(
        {
            "ground_truth_start": ground_truth_df["start_blink"].astype(int).to_numpy()[:n],
            "detected_start": detected_df["start_blink"].astype(int).to_numpy()[:n],
            "ground_truth_end": ground_truth_df["end_blink"].astype(int).to_numpy()[:n],
            "detected_end": detected_df["end_blink"].astype(int).to_numpy()[:n],
        }
    )


def print_comparison_summary(metrics_dict: dict[str, float], tolerance_samples: int) -> None:
    """Print a human-readable summary of alignment metrics.

    Summaries are expressed in sample index units. Percentages are derived from counts
    in ``metrics_dict``.
    """

    logger.info("[comparison] Blink event alignment summary:")
    logger.info("  • Tolerance: ±%d samples", tolerance_samples)
    logger.info(
        "  • Total ground truth events: %d",
        int(metrics_dict.get("total_ground_truth", 0)),
    )
    logger.info(
        "  • Total detected events: %d",
        int(metrics_dict.get("total_detected", 0)),
    )

    paired = int(metrics_dict.get("paired_events", 0))
    matches = int(metrics_dict.get("matches_within_tolerance", 0))
    outside = int(metrics_dict.get("pairs_outside_tolerance", 0))
    gt_only = int(metrics_dict.get("ground_truth_only", 0))
    det_only = int(metrics_dict.get("detected_only", 0))
    share = metrics_dict.get("share_within_tolerance", float("nan"))

    if paired:
        pct_pairs = (matches / paired) * 100.0
        logger.info(
            "  • Paired events within tolerance: %d/%d (%.2f%% of pairs)",
            matches,
            paired,
            pct_pairs,
        )
    else:
        logger.info("  • Paired events within tolerance: 0")

    if outside:
        logger.info("  • Paired events outside tolerance: %d", outside)

    logger.info("  • Ground truth-only events: %d", gt_only)
    logger.info("  • Detected-only events: %d", det_only)
    if np.isfinite(share):
        logger.info(
            "  • Share of total unique events within tolerance: %d/%d (%.2f%%)",
            matches,
            matches + outside + gt_only + det_only,
            share,
        )
    else:
        logger.info("  • Share of total unique events within tolerance: n/a")


def make_diff_table(
    detected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    alignments: Iterable[Alignment],
    tolerance_samples: int,
    max_rows: int,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Build a diff table summarizing mismatches outside the tolerance window.

    All comparisons are performed in sample index units (1-based). Time values in the
    resulting table are derived from the provided ``sampling_rate_hz`` purely for
    display.
    """

    similarity.validate_event_table(detected_df)
    similarity.validate_event_table(ground_truth_df)

    detected_start = detected_df["start_blink"].astype(int).to_numpy()
    detected_end = detected_df["end_blink"].astype(int).to_numpy()
    gt_start = ground_truth_df["start_blink"].astype(int).to_numpy()
    gt_end = ground_truth_df["end_blink"].astype(int).to_numpy()

    rows: list[dict[str, Optional[float]]] = []

    for alignment in alignments:
        if alignment.ground_truth_idx is None and alignment.detected_idx is None:
            continue

        if alignment.ground_truth_idx is None:
            idx = alignment.detected_idx
            assert idx is not None
            rows.append(
                {
                    "ground_truth_idx": np.nan,
                    "detected_idx": float(idx),
                    "ground_truth_start": np.nan,
                    "ground_truth_end": np.nan,
                    "detected_start": float(detected_start[idx]),
                    "detected_end": float(detected_end[idx]),
                    "start_diff": np.nan,
                    "end_diff": np.nan,
                    "time_sec": _to_seconds(detected_start[idx], sampling_rate_hz),
                }
            )
            continue

        if alignment.detected_idx is None:
            idx = alignment.ground_truth_idx
            rows.append(
                {
                    "ground_truth_idx": float(idx),
                    "detected_idx": np.nan,
                    "ground_truth_start": float(gt_start[idx]),
                    "ground_truth_end": float(gt_end[idx]),
                    "detected_start": np.nan,
                    "detected_end": np.nan,
                    "start_diff": np.nan,
                    "end_diff": np.nan,
                    "time_sec": _to_seconds(gt_start[idx], sampling_rate_hz),
                }
            )
            continue

        if alignment.start_diff is None or alignment.end_diff is None:
            continue

        if (
            abs(alignment.start_diff) <= tolerance_samples
            and abs(alignment.end_diff) <= tolerance_samples
        ):
            continue

        idx_gt = alignment.ground_truth_idx
        idx_det = alignment.detected_idx
        assert idx_gt is not None and idx_det is not None
        midpoint_sample = (gt_start[idx_gt] + detected_start[idx_det]) / 2.0
        rows.append(
            {
                "ground_truth_idx": float(idx_gt),
                "detected_idx": float(idx_det),
                "ground_truth_start": float(gt_start[idx_gt]),
                "ground_truth_end": float(gt_end[idx_gt]),
                "detected_start": float(detected_start[idx_det]),
                "detected_end": float(detected_end[idx_det]),
                "start_diff": float(alignment.start_diff),
                "end_diff": float(alignment.end_diff),
                "time_sec": _to_seconds(midpoint_sample, sampling_rate_hz),
            }
        )

    diff_df = pd.DataFrame(rows)
    if not diff_df.empty:
        diff_df = diff_df.head(max_rows).copy()
        diff_df["time_sec"] = diff_df["time_sec"].round(6)
    return diff_df


def _build_alignment_annotation_payload(
    ground_truth_starts: np.ndarray,
    ground_truth_ends: np.ndarray,
    detected_starts: np.ndarray,
    detected_ends: np.ndarray,
    sampling_rate_hz: float,
    tolerance_samples: int,
    alignments: Optional[Iterable[Alignment]] = None,
) -> tuple[list[Alignment], list[float], list[float], list[str]]:
    """Return alignments alongside onset/duration/description lists for annotations."""

    if alignments is None:
        gt_df = pd.DataFrame({"start_blink": ground_truth_starts, "end_blink": ground_truth_ends})
        det_df = pd.DataFrame({"start_blink": detected_starts, "end_blink": detected_ends})
        alignments_list = similarity.align_events(det_df, gt_df, tolerance_samples)
    else:
        alignments_list = list(alignments)

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    for alignment in alignments_list:
        if alignment.ground_truth_idx is not None and alignment.detected_idx is not None:
            gt_idx = alignment.ground_truth_idx
            det_idx = alignment.detected_idx
            gt_start = int(ground_truth_starts[gt_idx])
            gt_end = int(ground_truth_ends[gt_idx])
            det_start = int(detected_starts[det_idx])
            det_end = int(detected_ends[det_idx])

            gt_onset = _to_seconds(gt_start, sampling_rate_hz)
            det_onset = _to_seconds(det_start, sampling_rate_hz)
            gt_duration = _duration_seconds(gt_start, gt_end, sampling_rate_hz)
            det_duration = _duration_seconds(det_start, det_end, sampling_rate_hz)

            if alignment.is_match(tolerance_samples):
                onsets.append(float(gt_onset + det_onset) / 2.0)
                durations.append(float(gt_duration + det_duration) / 2.0)
                descriptions.append("blink")
            else:
                onsets.extend([float(gt_onset), float(det_onset)])
                durations.extend([float(gt_duration), float(det_duration)])
                descriptions.extend(["blink_ground_truth", "blink_detected"])
        elif alignment.ground_truth_idx is not None:
            gt_idx = alignment.ground_truth_idx
            gt_start = int(ground_truth_starts[gt_idx])
            gt_end = int(ground_truth_ends[gt_idx])
            onsets.append(float(_to_seconds(gt_start, sampling_rate_hz)))
            durations.append(float(_duration_seconds(gt_start, gt_end, sampling_rate_hz)))
            descriptions.append("blink_ground_truth")
        elif alignment.detected_idx is not None:
            det_idx = alignment.detected_idx
            det_start = int(detected_starts[det_idx])
            det_end = int(detected_ends[det_idx])
            onsets.append(float(_to_seconds(det_start, sampling_rate_hz)))
            durations.append(float(_duration_seconds(det_start, det_end, sampling_rate_hz)))
            descriptions.append("blink_detected")

    return alignments_list, onsets, durations, descriptions


def build_diagnostic_raw(
    ground_truth_signal: np.ndarray,
    detected_signal: np.ndarray,
    ground_truth_starts: np.ndarray,
    ground_truth_ends: np.ndarray,
    detected_starts: np.ndarray,
    detected_ends: np.ndarray,
    sampling_rate_hz: float,
    tolerance_samples: int,
    alignments: Optional[Iterable[Alignment]] = None,
) -> mne.io.RawArray:
    """Create an annotated two-channel :class:`mne.io.RawArray` for inspection.

    Annotations are labeled as:
    ``"blink"``
        Detected/ground-truth pairs whose start and end differences are within tolerance.
    ``"blink_ground_truth"``
        Events present only in the ground truth table.
    ``"blink_detected"``
        Events present only in the detected table or outside tolerance.

    All comparisons and tolerance checks are performed in sample index units (1-based).
    """

    ground_truth_signal = np.array(ground_truth_signal, dtype=float, copy=True)
    detected_signal = np.array(detected_signal, dtype=float, copy=True)

    for signal in (ground_truth_signal, detected_signal):
        max_val = np.max(np.abs(signal)) if signal.size else 0.0
        if max_val > 0:
            signal /= max_val

    n_samples = min(len(ground_truth_signal), len(detected_signal))
    if n_samples == 0:
        raise ValueError("Signals must contain at least one sample to create RawArray.")

    logger.warning(
        "Assuming ground truth and detected blink signals are sampled at the same "
        "rate of %.3f Hz for diagnostic visualization.",
        float(sampling_rate_hz),
    )

    if len(ground_truth_signal) != len(detected_signal):
        logger.warning(
            "Signal lengths differ (%d vs %d samples); truncating to the overlapping "
            "window under the shared sampling-rate assumption.",
            len(ground_truth_signal),
            len(detected_signal),
        )

    data = np.vstack([ground_truth_signal[:n_samples], detected_signal[:n_samples]])
    info = mne.create_info(
        ch_names=["ground_truth_blink_signal", "detected_blink_signal"],
        sfreq=float(sampling_rate_hz),
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    (
        alignments_list,
        onsets,
        durations,
        descriptions,
    ) = _build_alignment_annotation_payload(
        ground_truth_starts,
        ground_truth_ends,
        detected_starts,
        detected_ends,
        sampling_rate_hz,
        tolerance_samples,
        alignments,
    )

    if onsets:
        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(annotations)
    else:
        raw.set_annotations(None)

    logger.info(
        "[mne] Created synthetic Raw with %d blink annotations (matched: %d, ground truth-only: %d, detected-only: %d)",
        len(onsets),
        sum(
            1
            for a in alignments_list
            if a.ground_truth_idx is not None and a.detected_idx is not None
        ),
        sum(
            1
            for a in alignments_list
            if a.ground_truth_idx is not None and a.detected_idx is None
        ),
        sum(
            1
            for a in alignments_list
            if a.detected_idx is not None and a.ground_truth_idx is None
        ),
    )

    return raw
