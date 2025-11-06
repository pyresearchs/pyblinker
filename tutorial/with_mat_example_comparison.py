#!/usr/bin/env python
"""Compare PyBlinker detections against manual annotations from a MAT dataset.

This tutorial mirrors the workflow used in
``tutorial/blinker/1_blink_position/understand_diff_in_blink_position.py`` but
operates on the CLA subject MAT recording that ships with this repository.

The heavy lifting now lives in :mod:`tutorial.utils`, leaving this file as a
beginner-friendly walkthrough that wires together the individual steps:

1. Download the MAT EEG recording (if necessary) and load it with MNE.
2. Run :class:`pyblinker.blinker.pyblinker.BlinkDetector` on the channels of
   interest.
3. Load the manual annotations stored next to the MAT file.
4. Compare the detected vs. ground-truth blink intervals and visualise the
   differences.

Adjust :data:`TOLERANCE_SAMPLES` to change how strict the comparison is.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mne


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_URL = "https://figshare.com/ndownloader/files/12400409"
MAT_FILENAME = "CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
CSV_FILENAME = "CLA-SubjectJ-170510-3St-LRHand-Inter_annotations.csv"
MAT_PATH = SCRIPT_DIR / MAT_FILENAME
CSV_PATH = SCRIPT_DIR / CSV_FILENAME

SAMPLING_RATE_HZ = 200.0
CHANNELS_TO_KEEP = ("CH1", "CH2", "CH3")
TOLERANCE_SAMPLES = 20
N_PREVIEW_ROWS = 10
N_DIFF_ROWS = 30
RAW_PLOT_SCALINGS = {"eeg": 0.5}


def main() -> mne.io.Raw:
    """Run the full MAT vs PyBlinker comparison workflow."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tutorial.utils.blink_comparison import (
        build_comparison_annotations,
        compare_detected_vs_ground_truth,
        compute_alignments_and_metrics,
    )
    from tutorial.utils.blink_detection import run_pyblinker_detection
    from tutorial.utils.mat_data import (
        annotations_to_event_table,
        ensure_mat_file,
        load_manual_annotations_csv,
        load_raw_from_mat,
        pick_channels,
    )
    from tutorial.utils.pathing import ensure_repo_on_path

    ensure_repo_on_path()

    mat_path = ensure_mat_file(MAT_PATH, DATA_URL)
    raw_full = load_raw_from_mat(mat_path, SAMPLING_RATE_HZ)
    raw = pick_channels(raw_full, CHANNELS_TO_KEEP)
    print(f"[mne] Loaded MAT file with channels: {raw.ch_names}")

    detection = run_pyblinker_detection(raw, sampling_rate_hz=SAMPLING_RATE_HZ)
    print(f"[detector] Event table rows: {len(detection.events)}")

    annotations_df = load_manual_annotations_csv(CSV_PATH)
    ground_truth_events = annotations_to_event_table(annotations_df, detection.sampling_rate_hz)
    print(f"[ground-truth] Loaded {len(ground_truth_events)} manual annotations")

    _diagnostic_raw = compare_detected_vs_ground_truth(
        detection,
        ground_truth_events,
        tolerance_samples=TOLERANCE_SAMPLES,
        n_preview_rows=N_PREVIEW_ROWS,
        n_diff_rows=N_DIFF_ROWS,
    )

    alignments, metrics = compute_alignments_and_metrics(
        detected_df=detection.events,
        ground_truth_df=ground_truth_events,
        tolerance_samples=TOLERANCE_SAMPLES,
    )

    annotations = build_comparison_annotations(
        ground_truth_starts=ground_truth_events["start_blink"].to_numpy(),
        ground_truth_ends=ground_truth_events["end_blink"].to_numpy(),
        detected_starts=detection.events["start_blink"].to_numpy(),
        detected_ends=detection.events["end_blink"].to_numpy(),
        sampling_rate_hz=detection.sampling_rate_hz,
        tolerance_samples=TOLERANCE_SAMPLES,
        alignments=alignments,
    )

    if annotations is not None:
        print(f"[mne] Applying {len(annotations)} comparison annotations to the EEG raw")
        raw.set_annotations(annotations)
    else:
        print("[mne] No blink annotations generated; clearing annotations on the EEG raw")
        raw.set_annotations(None)

    if os.environ.get("PYBLINKER_SKIP_PLOT") == "1":
        print("[info] Skipping raw.plot() because PYBLINKER_SKIP_PLOT=1")
    else:
        matches = int(metrics["matches_within_tolerance"])
        ground_truth_only = int(metrics["ground_truth_only"])
        detected_only = int(metrics["detected_only"])
        pairs_outside = int(metrics["pairs_outside_tolerance"])
        total_differences = ground_truth_only + detected_only + pairs_outside
        plot_title = (
            "Manual vs PyBlinker Blink Comparison — "
            f"Matches: {matches}, Ground Truth Only: {ground_truth_only}, "
            f"PyBlinker Only: {detected_only}, Differences: {total_differences}"
        )
        try:
            raw.plot(
                block=True,
                title=plot_title,
                scalings=RAW_PLOT_SCALINGS,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[warn] Unable to open interactive Raw browser: {exc}")

    return raw


if __name__ == "__main__":
    main()
