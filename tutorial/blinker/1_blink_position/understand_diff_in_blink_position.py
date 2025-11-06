#!/usr/bin/env python
"""Compare pyblinker EDF results against MATLAB ground truth within tolerance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mne


BASE_DIR = Path(__file__).resolve().parents[2]
EDF_PATH = BASE_DIR / "test" / "test_files" / "mne_sample_audvis_raw.edf"
MAT_INPUT_PATH = BASE_DIR / "test" / "migration_files" / "step1bi_data_input_getBlinkPositions.mat"
MAT_OUTPUT_PATH = BASE_DIR / "test" / "migration_files" / "step1bi_data_output_getBlinkPositions.mat"
N_PREVIEW_ROWS = 10
N_DIFF_ROWS = 30
SAMPLING_RATE_HZ = 100.0
RAW_PLOT_SCALINGS = {"eeg": 0.5}
TOLERANCE_SAMPLES = 20  # Allowable sample difference between MATLAB vs Python
PREFERRED_CHANNELS = ("EEG 003", "EEG003", "chan003")


def main() -> mne.io.Raw:
    """Run the EDF vs. MATLAB blink comparison tutorial."""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tutorial.utils.blink_comparison import compare_detected_vs_ground_truth
    from tutorial.utils.blink_position import (
        detect_blinks_from_edf,
        load_ground_truth_from_matlab,
    )
    from tutorial.utils.pathing import ensure_repo_on_path

    ensure_repo_on_path()

    detection = detect_blinks_from_edf(
        EDF_PATH,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        preferred_channel_names=PREFERRED_CHANNELS,
    )
    ground_truth_df, matlab_blink_signal = load_ground_truth_from_matlab(
        MAT_INPUT_PATH, MAT_OUTPUT_PATH
    )

    aligned_samples = min(len(matlab_blink_signal), len(detection.signal))
    ground_truth_signal = matlab_blink_signal[:aligned_samples]
    detected_signal = detection.signal[:aligned_samples]

    diagnostic_raw = compare_detected_vs_ground_truth(
        detection,
        ground_truth_df,
        tolerance_samples=TOLERANCE_SAMPLES,
        n_preview_rows=N_PREVIEW_ROWS,
        n_diff_rows=N_DIFF_ROWS,
        ground_truth_signal=ground_truth_signal,
        detected_signal=detected_signal,
    )

    if os.environ.get("PYBLINKER_SKIP_PLOT") == "1":
        print("[info] Skipping raw.plot() because PYBLINKER_SKIP_PLOT=1")
    else:
        try:
            diagnostic_raw.plot(
                block=True,
                title="MATLAB vs Python blink comparison",
                scalings=RAW_PLOT_SCALINGS,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[warn] Unable to open interactive Raw browser: {exc}")

    return diagnostic_raw


if __name__ == "__main__":
    main()
