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
BASE_DIR = Path(__file__).resolve().parents[2]
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
def run_python_from_edf() -> tuple[pd.DataFrame, np.ndarray]:
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

    print("\n[python/MNE] first 5 rows:")
    print(py_df_1based.head())
    print(f"[python/MNE] total detected blinks: {len(py_df_1based)}")

    return py_df_1based, python_blink_signal


# =====================================================================
# PART 2 — load MATLAB gold
# =====================================================================
def load_matlab_gold() -> tuple[pd.DataFrame, np.ndarray]:
    """Load MATLAB output blink positions (1-based)."""

    assert MAT_OUTPUT_PATH.exists()

    input, output = load_matlab_data(str(MAT_INPUT_PATH), str(MAT_OUTPUT_PATH))
    blink_positions_mat = output["blinkPositions"]
    assert blink_positions_mat.shape[0] == 2
    matlab_blink_signal = input["blinkComp"]
    return pd.DataFrame(
        {"start_blink": blink_positions_mat[0, :], "end_blink": blink_positions_mat[1, :]}
    ), matlab_blink_signal


def build_diagnostic_raw(
    matlab_blink_signal: np.ndarray,
    python_blink_signal: np.ndarray,
    mat_start_samples: np.ndarray,
    mat_end_samples: np.ndarray,
    python_start_samples: np.ndarray,
    python_end_samples: np.ndarray,
    tolerance_samples: int,
) -> mne.io.RawArray:
    """Create an annotated two-channel RawArray for visual inspection."""

    matlab_blink_signal = np.array(matlab_blink_signal, dtype=float, copy=True)
    python_blink_signal = np.array(python_blink_signal, dtype=float, copy=True)

    for signal in (matlab_blink_signal, python_blink_signal):
        max_val = np.max(np.abs(signal)) if signal.size else 0.0
        if max_val > 0:
            signal /= max_val

    n_samples = min(len(matlab_blink_signal), len(python_blink_signal))
    if n_samples == 0:
        raise ValueError("Signals must contain at least one sample to create RawArray.")

    data = np.vstack([matlab_blink_signal[:n_samples], python_blink_signal[:n_samples]])
    info = mne.create_info(
        ch_names=["matlab_blink_signal", "python_blink_signal"],
        sfreq=SAMPLING_RATE_HZ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    def _sample_to_seconds(start_sample: int, end_sample: int) -> tuple[float, float]:
        onset_sec = (start_sample - 1) / SAMPLING_RATE_HZ
        duration_sec = (end_sample - start_sample + 1) / SAMPLING_RATE_HZ
        return onset_sec, duration_sec

    min_blinks = min(len(python_start_samples), len(mat_start_samples))
    for i in range(min_blinks):
        start_diff = abs(python_start_samples[i] - mat_start_samples[i])
        end_diff = abs(python_end_samples[i] - mat_end_samples[i])

        if start_diff <= tolerance_samples and end_diff <= tolerance_samples:
            mat_onset, mat_duration = _sample_to_seconds(
                mat_start_samples[i], mat_end_samples[i]
            )
            py_onset, py_duration = _sample_to_seconds(
                python_start_samples[i], python_end_samples[i]
            )
            onsets.append((mat_onset + py_onset) / 2)
            durations.append((mat_duration + py_duration) / 2)
            descriptions.append("blink")
        else:
            mat_onset, mat_duration = _sample_to_seconds(
                mat_start_samples[i], mat_end_samples[i]
            )
            py_onset, py_duration = _sample_to_seconds(
                python_start_samples[i], python_end_samples[i]
            )

            onsets.extend([mat_onset, py_onset])
            durations.extend([mat_duration, py_duration])
            descriptions.extend(["blink_matlab", "blink_python"])

    if len(mat_start_samples) > min_blinks:
        for j in range(min_blinks, len(mat_start_samples)):
            mat_onset, mat_duration = _sample_to_seconds(
                mat_start_samples[j], mat_end_samples[j]
            )
            onsets.append(mat_onset)
            durations.append(mat_duration)
            descriptions.append("blink_matlab")

    if len(python_start_samples) > min_blinks:
        for j in range(min_blinks, len(python_start_samples)):
            py_onset, py_duration = _sample_to_seconds(
                python_start_samples[j], python_end_samples[j]
            )
            onsets.append(py_onset)
            durations.append(py_duration)
            descriptions.append("blink_python")

    if onsets:
        ann = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(ann)
    else:
        raw.set_annotations(None)

    print(f"[mne] Created synthetic Raw with {len(onsets)} blink annotations")

    return raw


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
            approx_start_sample = (py_start[i] + mat_start[i]) / 2
            mismatches.append(
                {
                    "idx": i,
                    "mat_start": mat_start[i],
                    "py_start": py_start[i],
                    "Δstart": py_start[i] - mat_start[i],
                    "mat_end": mat_end[i],
                    "py_end": py_end[i],
                    "Δend": py_end[i] - mat_end[i],
                    "time_sec": (approx_start_sample - 1) / SAMPLING_RATE_HZ,
                }
            )

    if mismatches:
        print(f"\n[diff] mismatches beyond ±{tolerance} samples (showing {max_rows}):")
        diff_df = pd.DataFrame(mismatches)
        diff_df["time_sec"] = diff_df["time_sec"].round(3)
        print(diff_df.head(max_rows))
    else:
        print(f"\n[diff] All differences ≤ ±{tolerance} samples (OK).")

    if len_py != len_mat:
        print(f"\n[diff] Row count mismatch: Python={len_py}, MATLAB={len_mat}")

    print("\n============== END DIFF REPORT ==============\n")


# =====================================================================
# PART 4 — compare + assert
# =====================================================================
def compare_python_vs_matlab(
    py_df: pd.DataFrame,
    mat_df: pd.DataFrame,
    matlab_blink_signal,
    python_blink_signal,
) -> mne.io.RawArray:
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
    python_start_samples = py_df["start_blink"].astype(int).to_numpy()
    python_end_samples = py_df["end_blink"].astype(int).to_numpy()
    mat_start_samples = mat_df["start_blink"].astype(int).to_numpy()
    mat_end_samples = mat_df["end_blink"].astype(int).to_numpy()

    # compute absolute diffs (only for overlapping indices)
    min_len = min(len(python_start_samples), len(mat_start_samples))
    diff_start = np.abs(python_start_samples[:min_len] - mat_start_samples[:min_len])
    diff_end = np.abs(python_end_samples[:min_len] - mat_end_samples[:min_len])

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

    return build_diagnostic_raw(
        matlab_blink_signal=matlab_blink_signal,
        python_blink_signal=python_blink_signal,
        mat_start_samples=mat_start_samples,
        mat_end_samples=mat_end_samples,
        python_start_samples=python_start_samples,
        python_end_samples=python_end_samples,
        tolerance_samples=TOLERANCE_SAMPLES,
    )


# =====================================================================
# MAIN
# =====================================================================
def main():
    py_df, python_blink_signal = run_python_from_edf()
    mat_df, matlab_blink_signal = load_matlab_gold()
    raw = compare_python_vs_matlab(
        py_df, mat_df, matlab_blink_signal, python_blink_signal
    )
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
