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

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from pyblinker.blinker.get_blink_positions import get_blink_position
from test.blinker_migration.debugging_tools import load_matlab_data


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
EDF_PATH = Path(
    r"C:\Users\balan\IdeaProjects\pyblinker\test\test_files\mne_sample_audvis_raw.edf"
)

MAT_INPUT_PATH = Path(
    r"C:\Users\balan\IdeaProjects\pyblinker\test\migration_files\step1bi_data_input_getBlinkPositions.mat"
)

MAT_OUTPUT_PATH = Path(
    r"C:\Users\balan\IdeaProjects\pyblinker\test\migration_files\step1bi_data_output_getBlinkPositions.mat"
)
N_PREVIEW_ROWS = 10
N_DIFF_ROWS = 30

# ➤ Allowable integer difference between MATLAB vs Python indices
TOLERANCE_SAMPLES = 20  # e.g. 374 vs 376 is considered “same”


# =====================================================================
# PART 1 — run pyblinker on EDF (MNE)
# =====================================================================
def run_python_from_edf() -> pd.DataFrame:
    """Load EDF, preprocess, pick channel, run pyblinker, return 1-based DataFrame."""
    assert EDF_PATH.exists(), f"EDF file not found: {EDF_PATH}"

    print(f"[info] using mne version: {mne.__version__}")
    print(f"[info] reading EDF: {EDF_PATH}")

    raw = read_raw_edf(EDF_PATH.as_posix(), preload=True, verbose="ERROR")

    raw.filter(1.0, 30.0, fir_design="firwin", n_jobs=1, verbose="ERROR")
    raw.resample(100, n_jobs=1, verbose="ERROR")

    srate = float(raw.info["sfreq"])
    assert abs(srate - 100.0) < 1e-6, f"Expected 100 Hz after resample, got {srate}"

    preferred_channel_names = ["EEG 003", "EEG003", "chan003"]
    picks = next(( [n] for n in preferred_channel_names if n in raw.ch_names ), None)
    if picks is None:
        assert len(raw.ch_names) >= 3, "Need ≥3 channels to pick channel 003."
        picks = [raw.ch_names[2]]

    data, _ = raw.get_data(picks=picks, return_times=True)
    blink_component = np.squeeze(data)
    assert blink_component.ndim == 1

    print(f"[info] blink_component samples: {blink_component.shape[0]}")

    params = {"sfreq": srate, "std_threshold": 1.5, "min_event_len": 0.05}

    assert np.isfinite(blink_component).all()
    assert blink_component.dtype.kind in "fiu"

    result = get_blink_position(
        params=params, blink_component=blink_component, ch="No_channel", progress_bar=False
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["start_blink", "end_blink"]

    py_df_1based = result.copy()
    py_df_1based[["start_blink", "end_blink"]] += 1

    print("\n[python/MNE] first 5 rows:")
    print(py_df_1based.head())
    print(f"[python/MNE] total detected blinks: {len(py_df_1based)}")

    return py_df_1based,blink_component


# =====================================================================
# PART 2 — load MATLAB gold
# =====================================================================
def load_matlab_gold() -> pd.DataFrame:
    """Load MATLAB output blink positions (1-based)."""

    assert MAT_OUTPUT_PATH.exists()

    input, output = load_matlab_data(str(MAT_INPUT_PATH), str(MAT_OUTPUT_PATH))
    blink_positions_mat = output["blinkPositions"]
    assert blink_positions_mat.shape[0] == 2
    signal_matlab=input["blinkComp"]
    return pd.DataFrame(
        {"start_blink": blink_positions_mat[0, :], "end_blink": blink_positions_mat[1, :]}
    ),signal_matlab


# =====================================================================
# PART 3 — diff helper
# =====================================================================
def _diff_report(py_df: pd.DataFrame, mat_df: pd.DataFrame, tolerance: int, max_rows: int):
    """Print detailed diff report beyond tolerance."""
    print("\n============== DIFF REPORT (with tolerance) ==============")

    len_py, len_mat = len(py_df), len(mat_df)
    print(f"Python rows : {len_py}")
    print(f"MATLAB rows : {len_mat}")
    min_len = min(len_py, len_mat)

    py_start = py_df["start_blink"].astype(int).to_numpy()
    py_end = py_df["end_blink"].astype(int).to_numpy()
    mat_start = mat_df["start_blink"].astype(int).to_numpy()
    mat_end = mat_df["end_blink"].astype(int).to_numpy()

    mismatches = []
    for i in range(min_len):
        if (abs(py_start[i] - mat_start[i]) > tolerance) or (abs(py_end[i] - mat_end[i]) > tolerance):
            mismatches.append(
                {
                    "idx": i,
                    "mat_start": mat_start[i],
                    "py_start": py_start[i],
                    "Δstart": py_start[i] - mat_start[i],
                    "mat_end": mat_end[i],
                    "py_end": py_end[i],
                    "Δend": py_end[i] - mat_end[i],
                }
            )

    if mismatches:
        print(f"\n[diff] mismatches beyond ±{tolerance} samples (showing {max_rows}):")
        print(pd.DataFrame(mismatches).head(max_rows))
    else:
        print(f"\n[diff] All differences ≤ ±{tolerance} samples (OK).")

    if len_py != len_mat:
        print(f"\n[diff] Row count mismatch: Python={len_py}, MATLAB={len_mat}")

    print("\n============== END DIFF REPORT ==============\n")


# =====================================================================
# PART 4 — compare + assert
# =====================================================================
def compare_python_vs_matlab(py_df: pd.DataFrame, mat_df: pd.DataFrame,mat_signal,blink_component) -> None:
    """Compare with tolerance, print detailed diff report if needed."""
    print("\n================ COMPARISON: Python vs MATLAB ================")
    print(f"Tolerance allowed: ±{TOLERANCE_SAMPLES} samples")
    print(f"Python rows      : {len(py_df)}")
    print(f"MATLAB rows      : {len(mat_df)}")
    print("Row count matches? ->", len(py_df) == len(mat_df))

    # quick preview
    nprev = min(N_PREVIEW_ROWS, len(py_df), len(mat_df))
    preview = pd.DataFrame(
        {
            "mat_start": mat_df["start_blink"].astype(int).to_numpy()[:nprev],
            "py_start": py_df["start_blink"].astype(int).to_numpy()[:nprev],
            "mat_end": mat_df["end_blink"].astype(int).to_numpy()[:nprev],
            "py_end": py_df["end_blink"].astype(int).to_numpy()[:nprev],
        }
    )
    print(f"\nFirst {nprev} rows (MATLAB vs Python):")
    print(preview)

    # convert to arrays
    py_start = py_df["start_blink"].astype(int).to_numpy()
    py_end = py_df["end_blink"].astype(int).to_numpy()
    mat_start = mat_df["start_blink"].astype(int).to_numpy()
    mat_end = mat_df["end_blink"].astype(int).to_numpy()

    # compute absolute diffs (only for overlapping indices)
    min_len = min(len(py_start), len(mat_start))
    diff_start = np.abs(py_start[:min_len] - mat_start[:min_len])
    diff_end = np.abs(py_end[:min_len] - mat_end[:min_len])

    # within tolerance?
    ok_start = diff_start <= TOLERANCE_SAMPLES
    ok_end = diff_end <= TOLERANCE_SAMPLES
    all_ok = np.all(ok_start & ok_end) and (len(py_df) == len(mat_df))

    if all_ok:
        print(f"\n✅ PASSED: all blink intervals match within ±{TOLERANCE_SAMPLES} samples.")
    else:
        _diff_report(py_df, mat_df, tolerance=TOLERANCE_SAMPLES, max_rows=N_DIFF_ROWS)
        # raise AssertionError(
        #     f"Blink positions differ by more than ±{TOLERANCE_SAMPLES} samples."
        # )


# =====================================================================
# MAIN
# =====================================================================
def main():
    py_df,blink_component = run_python_from_edf()
    mat_df,mat_signal = load_matlab_gold()
    compare_python_vs_matlab(py_df, mat_df,mat_signal,blink_component)


if __name__ == "__main__":
    main()
