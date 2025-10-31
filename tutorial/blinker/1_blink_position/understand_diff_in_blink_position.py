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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class BlinkAlignment:
    """Represents one aligned MATLAB/Python blink interval."""

    mat_idx: Optional[int]
    py_idx: Optional[int]
    start_diff: Optional[float]
    end_diff: Optional[float]


def _align_blink_events(
    mat_start: np.ndarray,
    mat_end: np.ndarray,
    py_start: np.ndarray,
    py_end: np.ndarray,
    tolerance: int,
) -> list[BlinkAlignment]:
    """Greedily align MATLAB and Python blink intervals."""

    alignments: list[BlinkAlignment] = []
    i = j = 0
    while i < len(mat_start) and j < len(py_start):
        start_delta = float(py_start[j] - mat_start[i])
        end_delta = float(py_end[j] - mat_end[i])
        abs_start = abs(start_delta)
        abs_end = abs(end_delta)

        if abs_start <= tolerance and abs_end <= tolerance:
            alignments.append(BlinkAlignment(i, j, start_delta, end_delta))
            i += 1
            j += 1
            continue

        if mat_start[i] < py_start[j]:
            alignments.append(BlinkAlignment(i, None, None, None))
            i += 1
        elif py_start[j] < mat_start[i]:
            alignments.append(BlinkAlignment(None, j, None, None))
            j += 1
        else:
            alignments.append(BlinkAlignment(i, j, start_delta, end_delta))
            i += 1
            j += 1

    while i < len(mat_start):
        alignments.append(BlinkAlignment(i, None, None, None))
        i += 1

    while j < len(py_start):
        alignments.append(BlinkAlignment(None, j, None, None))
        j += 1

    return alignments


def build_diagnostic_raw(
    matlab_blink_signal: np.ndarray,
    python_blink_signal: np.ndarray,
    mat_start_samples: np.ndarray,
    mat_end_samples: np.ndarray,
    python_start_samples: np.ndarray,
    python_end_samples: np.ndarray,
    tolerance_samples: int,
    *,
    alignments: Optional[list[BlinkAlignment]] = None,
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

    alignments = (
        alignments
        if alignments is not None
        else _align_blink_events(
            mat_start=mat_start_samples,
            mat_end=mat_end_samples,
            py_start=python_start_samples,
            py_end=python_end_samples,
            tolerance=tolerance_samples,
        )
    )

    for alignment in alignments:
        if alignment.mat_idx is not None and alignment.py_idx is not None:
            mat_onset, mat_duration = _sample_to_seconds(
                mat_start_samples[alignment.mat_idx],
                mat_end_samples[alignment.mat_idx],
            )
            py_onset, py_duration = _sample_to_seconds(
                python_start_samples[alignment.py_idx],
                python_end_samples[alignment.py_idx],
            )

            if (
                alignment.start_diff is not None
                and alignment.end_diff is not None
                and abs(alignment.start_diff) <= tolerance_samples
                and abs(alignment.end_diff) <= tolerance_samples
            ):
                onsets.append((mat_onset + py_onset) / 2)
                durations.append((mat_duration + py_duration) / 2)
                descriptions.append("blink")
            else:
                onsets.extend([mat_onset, py_onset])
                durations.extend([mat_duration, py_duration])
                descriptions.extend(["blink_matlab", "blink_python"])
        elif alignment.mat_idx is not None:
            mat_onset, mat_duration = _sample_to_seconds(
                mat_start_samples[alignment.mat_idx],
                mat_end_samples[alignment.mat_idx],
            )
            onsets.append(mat_onset)
            durations.append(mat_duration)
            descriptions.append("blink_matlab")
        elif alignment.py_idx is not None:
            py_onset, py_duration = _sample_to_seconds(
                python_start_samples[alignment.py_idx],
                python_end_samples[alignment.py_idx],
            )
            onsets.append(py_onset)
            durations.append(py_duration)
            descriptions.append("blink_python")

    if onsets:
        ann = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(ann)
    else:
        raw.set_annotations(None)

    print(
        "[mne] Created synthetic Raw with "
        f"{len(onsets)} blink annotations (matched: "
        f"{sum(1 for a in alignments if a.mat_idx is not None and a.py_idx is not None)}, "
        f"MATLAB-only: {sum(1 for a in alignments if a.mat_idx is not None and a.py_idx is None)}, "
        f"Python-only: {sum(1 for a in alignments if a.py_idx is not None and a.mat_idx is None)})"
    )

    return raw


# =====================================================================
# PART 3 — diff helper
# =====================================================================
def _diff_report(
    py_df: pd.DataFrame,
    mat_df: pd.DataFrame,
    tolerance: int,
    max_rows: int,
    *,
    precomputed_alignments: Optional[list[BlinkAlignment]] = None,
):
    """Print detailed diff report beyond tolerance, aligning rows greedily."""

    def _time_from_sample(sample: float | None) -> float | None:
        if sample is None:
            return None
        return (sample - 1) / SAMPLING_RATE_HZ

    print("\n============== DIFF REPORT (with tolerance) ==============")

    len_py, len_mat = len(py_df), len(mat_df)
    print(f"Python rows : {len_py}")
    print(f"MATLAB rows : {len_mat}")

    py_start = py_df["start_blink"].astype(int).to_numpy()
    py_end = py_df["end_blink"].astype(int).to_numpy()
    mat_start = mat_df["start_blink"].astype(int).to_numpy()
    mat_end = mat_df["end_blink"].astype(int).to_numpy()
    alignments = (
        precomputed_alignments
        if precomputed_alignments is not None
        else _align_blink_events(
            mat_start=mat_start,
            mat_end=mat_end,
            py_start=py_start,
            py_end=py_end,
            tolerance=tolerance,
        )
    )

    mismatches: list[dict[str, float | int | None]] = []
    for alignment in alignments:
        if alignment.mat_idx is None and alignment.py_idx is None:
            continue

        if alignment.mat_idx is None:
            idx = alignment.py_idx
            assert idx is not None
            mismatches.append(
                {
                    "mat_idx": np.nan,
                    "mat_start": np.nan,
                    "mat_end": np.nan,
                    "py_idx": float(idx),
                    "py_start": py_start[idx],
                    "py_end": py_end[idx],
                    "Δstart": np.nan,
                    "Δend": np.nan,
                    "time_sec": _time_from_sample(py_start[idx]),
                }
            )
            continue

        if alignment.py_idx is None:
            idx = alignment.mat_idx
            mismatches.append(
                {
                    "mat_idx": float(idx),
                    "mat_start": mat_start[idx],
                    "mat_end": mat_end[idx],
                    "py_idx": np.nan,
                    "py_start": np.nan,
                    "py_end": np.nan,
                    "Δstart": np.nan,
                    "Δend": np.nan,
                    "time_sec": _time_from_sample(mat_start[idx]),
                }
            )
            continue

        delta_start = alignment.start_diff if alignment.start_diff is not None else np.nan
        delta_end = alignment.end_diff if alignment.end_diff is not None else np.nan

        if (
            np.isnan(delta_start)
            or np.isnan(delta_end)
            or abs(delta_start) > tolerance
            or abs(delta_end) > tolerance
        ):
            mat_idx = alignment.mat_idx
            py_idx = alignment.py_idx
            mismatches.append(
                {
                    "mat_idx": float(mat_idx),
                    "mat_start": mat_start[mat_idx],
                    "mat_end": mat_end[mat_idx],
                    "py_idx": float(py_idx),
                    "py_start": py_start[py_idx],
                    "py_end": py_end[py_idx],
                    "Δstart": delta_start,
                    "Δend": delta_end,
                    "time_sec": _time_from_sample((py_start[py_idx] + mat_start[mat_idx]) / 2),
                }
            )

    if mismatches:
        print(f"\n[diff] mismatches beyond ±{tolerance} samples (showing {max_rows}):")
        diff_df = pd.DataFrame(mismatches)
        if "time_sec" in diff_df.columns:
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

    alignments = _align_blink_events(
        mat_start=mat_start_samples,
        mat_end=mat_end_samples,
        py_start=python_start_samples,
        py_end=python_end_samples,
        tolerance=TOLERANCE_SAMPLES,
    )

    # compute absolute diffs (only for overlapping indices)
    min_len = min(len(python_start_samples), len(mat_start_samples))
    diff_start = (
        np.abs(python_start_samples[:min_len] - mat_start_samples[:min_len])
        if min_len
        else np.array([], dtype=int)
    )
    diff_end = (
        np.abs(python_end_samples[:min_len] - mat_end_samples[:min_len])
        if min_len
        else np.array([], dtype=int)
    )

    # within tolerance?
    ok_start = diff_start <= TOLERANCE_SAMPLES if diff_start.size else np.array([], dtype=bool)
    ok_end = diff_end <= TOLERANCE_SAMPLES if diff_end.size else np.array([], dtype=bool)
    all_ok = (
        (ok_start.size == 0 or np.all(ok_start))
        and (ok_end.size == 0 or np.all(ok_end))
        and (len(py_df) == len(mat_df))
    )

    if all_ok:
        print(f"\n✅ PASSED: all blink intervals match within ±{TOLERANCE_SAMPLES} samples.")
    else:
        _diff_report(
            py_df,
            mat_df,
            tolerance=TOLERANCE_SAMPLES,
            max_rows=N_DIFF_ROWS,
            precomputed_alignments=alignments,
        )
        # raise AssertionError(
        #     f"Blink positions differ by more than ±{TOLERANCE_SAMPLES} samples."
        # )

    print("\n[metrics] Blink alignment summary:")
    total_mat_events = len(mat_df)
    total_py_events = len(py_df)
    total_pairs = sum(1 for a in alignments if a.mat_idx is not None and a.py_idx is not None)
    matched_within_tol = sum(
        1
        for a in alignments
        if (
            a.mat_idx is not None
            and a.py_idx is not None
            and a.start_diff is not None
            and a.end_diff is not None
            and abs(a.start_diff) <= TOLERANCE_SAMPLES
            and abs(a.end_diff) <= TOLERANCE_SAMPLES
        )
    )
    unmatched_mat = sum(1 for a in alignments if a.mat_idx is not None and a.py_idx is None)
    unmatched_py = sum(1 for a in alignments if a.py_idx is not None and a.mat_idx is None)
    pairs_outside_tol = total_pairs - matched_within_tol
    total_union = total_pairs + unmatched_mat + unmatched_py

    def _pct(n: int, d: int) -> float:
        return (n / d) * 100.0 if d else float("nan")

    print(f"  • Total MATLAB events: {total_mat_events}")
    print(f"  • Total Python events: {total_py_events}")
    print(
        f"  • Paired events within tolerance: {matched_within_tol}/{total_pairs} "
        f"({ _pct(matched_within_tol, total_pairs):.2f}% of pairs)"
        if total_pairs
        else "  • Paired events within tolerance: 0"
    )
    if pairs_outside_tol:
        print(f"  • Paired events outside tolerance: {pairs_outside_tol}")
    print(f"  • MATLAB-only events: {unmatched_mat}")
    print(f"  • Python-only events: {unmatched_py}")
    print(
        f"  • Share of total unique events within tolerance: {matched_within_tol}/{total_union} "
        f"({_pct(matched_within_tol, total_union):.2f}%)"
        if total_union
        else "  • Share of total unique events within tolerance: n/a"
    )

    aligned_samples = min(len(matlab_blink_signal), len(python_blink_signal))
    if aligned_samples:
        matlab_segment = np.asarray(matlab_blink_signal[:aligned_samples], dtype=float)
        python_segment = np.asarray(python_blink_signal[:aligned_samples], dtype=float)
        diff_signal = matlab_segment - python_segment
        mean_abs_diff = float(np.mean(np.abs(diff_signal)))
        max_abs_diff = float(np.max(np.abs(diff_signal)))
        rms_diff = float(np.sqrt(np.mean(diff_signal**2)))
        if np.std(matlab_segment) > 0 and np.std(python_segment) > 0:
            corr = float(np.corrcoef(matlab_segment, python_segment)[0, 1])
        else:
            corr = float("nan")
        print("\n[metrics] Signal amplitude comparison (first aligned segment):")
        print(f"  • Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  • RMS difference: {rms_diff:.6f}")
        print(f"  • Max absolute difference: {max_abs_diff:.6f}")
        print(f"  • Pearson correlation: {corr:.6f}")
    else:
        print("\n[metrics] Signal amplitude comparison skipped (no overlapping samples).")

    return build_diagnostic_raw(
        matlab_blink_signal=matlab_blink_signal,
        python_blink_signal=python_blink_signal,
        mat_start_samples=mat_start_samples,
        mat_end_samples=mat_end_samples,
        python_start_samples=python_start_samples,
        python_end_samples=python_end_samples,
        tolerance_samples=TOLERANCE_SAMPLES,
        alignments=alignments,
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
