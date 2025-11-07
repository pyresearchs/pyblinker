#!/usr/bin/env python
"""Compare pyblinker EDF results against MATLAB ground truth within tolerance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mne
from mne.export import export_raw


BASE_DIR = Path(__file__).resolve().parents[1]
EDF_PATH = BASE_DIR / "test" / "test_files" / "mne_sample_audvis_raw.edf"
MAT_INPUT_PATH = BASE_DIR / "test" / "migration_files" / "step1bi_data_input_getBlinkPositions.mat"
MAT_OUTPUT_PATH = BASE_DIR / "test" / "migration_files" / "step1bi_data_output_getBlinkPositions.mat"
N_PREVIEW_ROWS = 10
N_DIFF_ROWS = 30
SAMPLING_RATE_HZ = 100.0
RAW_PLOT_SCALINGS = {"eeg": 0.5}
TOLERANCE_SAMPLES = 20  # Allowable sample difference between MATLAB vs Python
PREFERRED_CHANNELS = ("EEG 003", "EEG003", "chan003")



def ensure_edf_file(edf_path: Path) -> Path:
    """Ensure the tutorial EDF file exists, converting from MNE sample data if needed."""

    if edf_path.exists():
        return edf_path

    print("[setup] EDF file missing; converting from MNE sample dataset")
    sample_data_folder = Path(mne.datasets.sample.data_path())
    raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    if not raw_file.exists():
        raise FileNotFoundError(f"Sample FIF file not found: {raw_file}")

    edf_path.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_fif(raw_file.as_posix(), preload=True, verbose="ERROR")
    export_raw(edf_path.as_posix(), raw, fmt="edf", physical_range="auto")
    print(f"[setup] Exported EDF to {edf_path}")
    return edf_path
def main() -> mne.io.Raw:
    """Run the EDF vs. MATLAB blink comparison tutorial."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pyblinker.utils.evaluation import blink_comparison, blink_position

    edf_path = ensure_edf_file(EDF_PATH)

    detection = blink_position.detect_blinks_from_edf(
        edf_path,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        preferred_channel_names=PREFERRED_CHANNELS,
    )
    ground_truth_df, matlab_blink_signal = blink_position.load_ground_truth_from_matlab(
        MAT_INPUT_PATH, MAT_OUTPUT_PATH
    )

    aligned_samples = min(len(matlab_blink_signal), len(detection.signal))
    ground_truth_signal = matlab_blink_signal[:aligned_samples]
    detected_signal = detection.signal[:aligned_samples]

    diagnostic_raw = blink_comparison.compare_detected_vs_ground_truth(
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
