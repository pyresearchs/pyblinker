#!/usr/bin/env python
"""

This script show how the preprocessing such as filtering and resampling can affect the results of blink detection when migrating from MATLAB to Python (MNE + pyblinker).
Which can be due to small differences in implementation of filters, resampling, and indexing conventions.

As we see in the tutorial/blinker/step1bi_tutorial_validate_get_blink_positions.py where we are using the input data from MATLAB directly (i.e., test/migration_files/step1bi_data_input_getBlinkPositions.mat)
, the results match exactly,compared to the MATLAB gold standard (i.e.,test/migration_files/step1bi_data_output_getBlinkPositions.mat)
However, when we introduce the preprocessing steps using MNE on the EDF file, we observe small discrepancies in the detected blink positions compared to the MATLAB gold standard (i.e., test/migration_files/step1bi_data_output_getBlinkPositions.mat_.
blink_detection_mne_to_pyblinker_demo.py

Purpose
-------
1. Load EDF with MNE → preprocess → run pyblinker.
2. Load MATLAB gold-standard blink positions.
3. Compare Python vs MATLAB results (1-based indexing).
4. Allow a small integer tolerance (±TOLERANCE_SAMPLES) in start/end indices.
5. Print DIFF REPORT if differences exceed tolerance.

Why tolerance?
--------------
MATLAB and Python can differ by a few samples because of:
- resampling rounding,
- FIR filter group delay,
- 0-based vs 1-based rounding.
Differences ≤ TOLERANCE_SAMPLES are considered acceptable.

"""

import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Return the repository root (directory containing pyproject.toml)."""

    current = Path(__file__).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root relative to this file")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.utils.evaluation import reporting, similarity
from test.blinker_migration.debugging_tools import load_matlab_data


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR = _find_repo_root()
EDF_PATH = BASE_DIR / "test" / "test_files" / "mne_sample_audvis_raw.edf"
MAT_INPUT_PATH = BASE_DIR / "test" / "migration_files" / "step1bi_data_input_getBlinkPositions.mat"
MAT_OUTPUT_PATH = BASE_DIR / "test" / "migration_files" / "step1bi_data_output_getBlinkPositions.mat"
assert EDF_PATH.exists(), f"EDF file not found: {EDF_PATH}"
assert MAT_INPUT_PATH.exists(), f"MAT input file not found: {MAT_INPUT_PATH}"
assert MAT_OUTPUT_PATH.exists(), f"MAT output file not found: {MAT_OUTPUT_PATH}"
N_PREVIEW_ROWS = 10
N_DIFF_ROWS = 30
SAMPLING_RATE_HZ = 100.0
RAW_PLOT_SCALINGS = {"eeg": 0.5}

# ➤ Allowable integer difference between MATLAB vs Python indices
TOLERANCE_SAMPLES = 20  # e.g. 374 vs 376 is considered “same”


# =====================================================================
# PART 1 — run pyblinker on EDF (MNE)
# =====================================================================
def run_detection_from_edf() -> tuple[pd.DataFrame, np.ndarray]:
    """Load EDF, preprocess, pick channel, run pyblinker, return 1-based DataFrame."""
    assert EDF_PATH.exists(), f"EDF file not found: {EDF_PATH}"

    print(f"[info] using mne version: {mne.__version__}")
    print(f"[info] reading EDF: {EDF_PATH}")

    raw = read_raw_edf(EDF_PATH.as_posix(), preload=True, verbose="ERROR")

    raw.filter(1.0, 30.0, fir_design="firwin", n_jobs=1, verbose="ERROR")
    raw.resample(SAMPLING_RATE_HZ, n_jobs=1, verbose="ERROR")

    srate = float(raw.info["sfreq"])
    assert abs(srate - SAMPLING_RATE_HZ) < 1e-6, f"Expected {SAMPLING_RATE_HZ} Hz after resample, got {srate}"

    preferred_channel_names = ["EEG 003", "EEG003", "chan003"]
    picks = next(( [n] for n in preferred_channel_names if n in raw.ch_names ), None)
    if picks is None:
        assert len(raw.ch_names) >= 3, "Need ≥3 channels to pick channel 003."
        picks = [raw.ch_names[2]]

    data, _ = raw.get_data(picks=picks, return_times=True)
    python_blink_signal = np.squeeze(data)
    assert python_blink_signal.ndim == 1

    print(f"[info] python_blink_signal samples: {python_blink_signal.shape[0]}")

    params = {"sfreq": srate, "std_threshold": 1.5, "min_event_len": 0.05}

    assert np.isfinite(python_blink_signal).all()
    assert python_blink_signal.dtype.kind in "fiu"

    result = get_blink_position(
        params=params, blink_component=python_blink_signal, ch="No_channel", progress_bar=False
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["start_blink", "end_blink"]

    py_df_1based = result.copy()
    py_df_1based[["start_blink", "end_blink"]] += 1

    py_df_1based = py_df_1based.sort_values("start_blink", kind="mergesort", ignore_index=True)

    print("\n[detected] first 5 rows:")
    print(py_df_1based.head())
    print(f"[detected] total detected blinks: {len(py_df_1based)}")

    return py_df_1based, python_blink_signal


# =====================================================================
# PART 2 — load MATLAB gold
# =====================================================================
def load_ground_truth_from_matlab() -> tuple[pd.DataFrame, np.ndarray]:
    """Load MATLAB output blink positions (1-based)."""

    assert MAT_OUTPUT_PATH.exists()

    input, output = load_matlab_data(str(MAT_INPUT_PATH), str(MAT_OUTPUT_PATH))
    blink_positions_mat = output["blinkPositions"]
    assert blink_positions_mat.shape[0] == 2
    matlab_blink_signal = input["blinkComp"]
    ground_truth_df = pd.DataFrame(
        {"start_blink": blink_positions_mat[0, :], "end_blink": blink_positions_mat[1, :]}
    )
    ground_truth_df = ground_truth_df.sort_values("start_blink", kind="mergesort", ignore_index=True)
    return ground_truth_df, matlab_blink_signal


# =====================================================================
# PART 3 — comparison helpers
# =====================================================================
def compare_detected_vs_ground_truth(
    detected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    ground_truth_signal: np.ndarray,
    detected_signal: np.ndarray,
) -> mne.io.RawArray:
    """Compare detected vs ground truth blink events within a tolerance."""

    similarity.validate_event_table(detected_df)
    similarity.validate_event_table(ground_truth_df)

    print("\n================ COMPARISON: Detected vs Ground Truth ================")
    print(f"Tolerance allowed: ±{TOLERANCE_SAMPLES} samples")
    print(f"Detected rows    : {len(detected_df)}")
    print(f"Ground truth rows: {len(ground_truth_df)}")
    print("Row count matches? ->", len(detected_df) == len(ground_truth_df))

    nprev = min(N_PREVIEW_ROWS, len(detected_df), len(ground_truth_df))
    preview = reporting.preview_side_by_side(detected_df, ground_truth_df, nprev)
    print(f"\nFirst {nprev} rows (ground truth vs detected):")
    print(preview)

    alignments = similarity.align_events(
        detected_df=detected_df,
        ground_truth_df=ground_truth_df,
        tolerance_samples=TOLERANCE_SAMPLES,
    )
    metrics = similarity.compute_alignment_metrics(alignments, TOLERANCE_SAMPLES)

    start_diff, end_diff = similarity.compute_pairwise_differences(detected_df, ground_truth_df)
    ok_start = start_diff <= TOLERANCE_SAMPLES if start_diff.size else np.array([], dtype=bool)
    ok_end = end_diff <= TOLERANCE_SAMPLES if end_diff.size else np.array([], dtype=bool)
    all_ok = (
        (ok_start.size == 0 or np.all(ok_start))
        and (ok_end.size == 0 or np.all(ok_end))
        and (len(detected_df) == len(ground_truth_df))
    )

    diff_table = reporting.make_diff_table(
        detected_df,
        ground_truth_df,
        alignments,
        TOLERANCE_SAMPLES,
        N_DIFF_ROWS,
        SAMPLING_RATE_HZ,
    )

    if all_ok and diff_table.empty:
        print(f"\n✅ PASSED: all blink intervals match within ±{TOLERANCE_SAMPLES} samples.")
    else:
        print(f"\n[diff] mismatches beyond ±{TOLERANCE_SAMPLES} samples (showing {N_DIFF_ROWS}):")
        if diff_table.empty:
            print("No mismatches found despite metric discrepancies.")
        else:
            print(diff_table)

    reporting.print_comparison_summary(metrics, TOLERANCE_SAMPLES)

    aligned_samples = min(len(ground_truth_signal), len(detected_signal))
    if aligned_samples:
        ground_truth_segment = np.asarray(ground_truth_signal[:aligned_samples], dtype=float)
        detected_segment = np.asarray(detected_signal[:aligned_samples], dtype=float)
        diff_signal = ground_truth_segment - detected_segment
        mean_abs_diff = float(np.mean(np.abs(diff_signal)))
        max_abs_diff = float(np.max(np.abs(diff_signal)))
        rms_diff = float(np.sqrt(np.mean(diff_signal**2)))
        if np.std(ground_truth_segment) > 0 and np.std(detected_segment) > 0:
            corr = float(np.corrcoef(ground_truth_segment, detected_segment)[0, 1])
        else:
            corr = float("nan")
        print("\n[metrics] Signal amplitude comparison (first aligned segment):")
        print(f"  • Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  • RMS difference: {rms_diff:.6f}")
        print(f"  • Max absolute difference: {max_abs_diff:.6f}")
        print(f"  • Pearson correlation: {corr:.6f}")
    else:
        print("\n[metrics] Signal amplitude comparison skipped (no overlapping samples).")

    diagnostic_raw = reporting.build_diagnostic_raw(
        ground_truth_signal=ground_truth_signal,
        detected_signal=detected_signal,
        ground_truth_starts=ground_truth_df["start_blink"].to_numpy(),
        ground_truth_ends=ground_truth_df["end_blink"].to_numpy(),
        detected_starts=detected_df["start_blink"].to_numpy(),
        detected_ends=detected_df["end_blink"].to_numpy(),
        sampling_rate_hz=SAMPLING_RATE_HZ,
        tolerance_samples=TOLERANCE_SAMPLES,
        alignments=alignments,
    )

    return diagnostic_raw


# =====================================================================
# MAIN
# =====================================================================
def main():
    detected_df, detected_signal = run_detection_from_edf()
    ground_truth_df, ground_truth_signal = load_ground_truth_from_matlab()
    raw = compare_detected_vs_ground_truth(
        detected_df, ground_truth_df, ground_truth_signal, detected_signal
    )
    if os.environ.get("PYBLINKER_SKIP_PLOT") == "1":
        print("[info] Skipping raw.plot() because PYBLINKER_SKIP_PLOT=1")
    else:
        try:
            raw.plot(
                block=True,
                title="MATLAB vs Python Blink Signal Comparison",
                scalings=RAW_PLOT_SCALINGS,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[warn] Unable to open interactive Raw browser: {exc}")
    return raw


if __name__ == "__main__":
    RAW_DIAGNOSTIC = main()
