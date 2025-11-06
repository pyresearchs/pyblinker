"""Common routines for comparing blink detections in tutorials."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from .blink_detection import DetectionResult


def build_indicator_signal(n_samples: int, events: pd.DataFrame) -> np.ndarray:
    """Create a binary signal marking blink intervals."""

    signal = np.zeros(int(n_samples), dtype=float)
    starts = events["start_blink"].to_numpy(dtype=int)
    ends = events["end_blink"].to_numpy(dtype=int)

    for start, end in zip(starts, ends, strict=False):
        start_idx = max(start - 1, 0)
        end_idx = min(end - 1, signal.size - 1)
        if end_idx < start_idx:
            continue
        signal[start_idx : end_idx + 1] = 1.0
    return signal


def _print_indicator_diagnostics(
    ground_truth_signal: np.ndarray,
    detected_signal: np.ndarray,
) -> None:
    """Print quick diagnostics comparing binary indicator signals."""

    if ground_truth_signal.size == 0:
        print("\n[metrics] Indicator signal comparison skipped (no samples).")
        return

    diff_signal = ground_truth_signal - detected_signal
    mean_abs_diff = float(np.mean(np.abs(diff_signal)))
    max_abs_diff = float(np.max(np.abs(diff_signal)))
    rms_diff = float(np.sqrt(np.mean(diff_signal**2)))
    if np.std(ground_truth_signal) > 0 and np.std(detected_signal) > 0:
        corr = float(np.corrcoef(ground_truth_signal, detected_signal)[0, 1])
    else:
        corr = float("nan")
    print("\n[metrics] Indicator signal comparison:")
    print(f"  • Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  • RMS difference: {rms_diff:.6f}")
    print(f"  • Max absolute difference: {max_abs_diff:.6f}")
    print(f"  • Pearson correlation: {corr:.6f}")


def compare_detected_vs_ground_truth(
    detected: DetectionResult,
    ground_truth_events: pd.DataFrame,
    *,
    tolerance_samples: int,
    n_preview_rows: int,
    n_diff_rows: int,
    ground_truth_signal: np.ndarray | None = None,
    detected_signal: np.ndarray | None = None,
) -> mne.io.RawArray:
    """Compare detected events with ground truth and build diagnostic visuals."""

    from pyblinker.utils.evaluation import reporting, similarity

    detected_df = detected.events
    ground_truth_df = ground_truth_events.copy()

    similarity.validate_event_table(detected_df)
    similarity.validate_event_table(ground_truth_df)

    print("\n================ COMPARISON: Detected vs Ground Truth ================")
    print(f"Tolerance allowed: ±{tolerance_samples} samples")
    print(f"Detected rows    : {len(detected_df)}")
    print(f"Ground truth rows: {len(ground_truth_df)}")
    print("Row count matches? ->", len(detected_df) == len(ground_truth_df))

    nprev = min(n_preview_rows, len(detected_df), len(ground_truth_df))
    preview = reporting.preview_side_by_side(detected_df, ground_truth_df, nprev)
    print(f"\nFirst {nprev} rows (ground truth vs detected):")
    print(preview)

    alignments = similarity.align_events(
        detected_df=detected_df,
        ground_truth_df=ground_truth_df,
        tolerance_samples=tolerance_samples,
    )
    metrics = similarity.compute_alignment_metrics(alignments, tolerance_samples)

    start_diff, end_diff = similarity.compute_pairwise_differences(detected_df, ground_truth_df)
    ok_start = start_diff <= tolerance_samples if start_diff.size else np.array([], dtype=bool)
    ok_end = end_diff <= tolerance_samples if end_diff.size else np.array([], dtype=bool)
    all_ok = (
        (ok_start.size == 0 or np.all(ok_start))
        and (ok_end.size == 0 or np.all(ok_end))
        and (len(detected_df) == len(ground_truth_df))
    )

    diff_table = reporting.make_diff_table(
        detected_df,
        ground_truth_df,
        alignments,
        tolerance_samples,
        n_diff_rows,
        detected.sampling_rate_hz,
    )

    if all_ok and diff_table.empty:
        print(f"\n✅ PASSED: all blink intervals match within ±{tolerance_samples} samples.")
    else:
        print(f"\n[diff] mismatches beyond ±{tolerance_samples} samples (showing {n_diff_rows}):")
        if diff_table.empty:
            print("No mismatches found despite metric discrepancies.")
        else:
            print(diff_table)

    reporting.print_comparison_summary(metrics, tolerance_samples)

    n_samples = int(len(detected_signal) if detected_signal is not None else len(detected.signal))
    gt_signal = (
        ground_truth_signal
        if ground_truth_signal is not None
        else build_indicator_signal(n_samples, ground_truth_df)
    )
    det_signal = (
        detected_signal
        if detected_signal is not None
        else build_indicator_signal(n_samples, detected_df)
    )

    _print_indicator_diagnostics(gt_signal, det_signal)

    diagnostic_raw = reporting.build_diagnostic_raw(
        ground_truth_signal=gt_signal,
        detected_signal=det_signal,
        ground_truth_starts=ground_truth_df["start_blink"].to_numpy(),
        ground_truth_ends=ground_truth_df["end_blink"].to_numpy(),
        detected_starts=detected_df["start_blink"].to_numpy(),
        detected_ends=detected_df["end_blink"].to_numpy(),
        sampling_rate_hz=detected.sampling_rate_hz,
        tolerance_samples=tolerance_samples,
        alignments=alignments,
    )

    return diagnostic_raw
