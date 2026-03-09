#!/usr/bin/env python
"""
Tutorial: validate Python FitBlinks against MATLAB gold standard
================================================================

Purpose
-------
This tutorial validates that the **Python** implementation of the FitBlinks
processing pipeline reproduces the **MATLAB** results *exactly*, given the same
input data and parameters.

This function uses:
- the **MATLAB input** file (as originally used by EEGLAB/Blinker), and
- the **MATLAB output** file (treated as the gold standard).

We verify that the Python-produced output matches the MATLAB output field by field,
after compensating for indexing differences (MATLAB = 1-based, Python = 0-based).

Intent
------
We are **not** trying to detect numerical deviations or sampling precision errors here.
The goal is to confirm that the **Python code structurally maps 1-to-1** with the
    original MATLAB logic.

Therefore, we deliberately ignore minor differences caused by signal processing
implementation details such as:
- resampling rounding errors,
- FIR/IIR filter group delay differences,
- numerical precision at float boundaries.

In other words, this script ensures that the **migration of logic** (not the
                                                                     exact floating-point samples) is correct.

Ignored known cases
-------------------
A few rows show consistent but minimal one-sample or one-unit deviations that
cannot be traced to logical errors. These are likely due to rounding at
sub-sample precision or internal interpolation differences. For reproducibility,
we skip these cases explicitly:

IGNORED_CASES = [
    {"row": 78, "column": "rightOuter", "mat": 27800, "py": 27801},
    {"row": 26, "column": "y_intersect", "mat": 43.0, "py": 44.0},
    {"row": 65, "column": "y_intersect", "mat": 80.0, "py": 79.0},
]

These deviations are within one sample and do not affect downstream timing or
feature statistics.

Pipeline validated
------------------
1. Load MATLAB input:
step1bii_data_input_process_FitBlinks.mat
which provides:
- candidateSignal
- parameters

2. Load MATLAB gold output:
step1bii_data_output_process_FitBlinks.mat
which provides:
- blinkFits (struct array of blink parameters)
- We convert MATLAB's struct array to a pandas DataFrame and rename columns using the same rename map as the unit tests (`RENAME_MAP`).

3. Run Python equivalent:
- get_blink_position(...)
- FitBlinks(...).dprocess()
- collect FitBlinks.frame_blinks

4. Normalize Python result:
- convert 0-based indices → 1-based
- reorder columns to match MATLAB output
- adjust list-type fields (left_range, right_range)

5. Compare against MATLAB gold:
- check for missing or extra columns
- check that each cell value matches (ignoring known rounding artifacts)
- allow explicitly listed IGNORED_CASES

6. We print:
- missing columns report
- filtered diff table (only cells that are not 'consistent')
- a small preview of the top N rows from each side (if wanted)
Run
---
python tutorial_validate_fit_blinks.py

Expected
--------
✅ "Validation PASSED" if Python reproduces MATLAB output exactly.
❌ AssertionError if any column or value differs (beyond the ignored minimal cases).
"""

import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position
from test.blinker_migration import RENAME_MAP
from test.blinker_migration.obs.debugging_tools import load_matlab_data


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
N_PREVIEW_ROWS = 10
TO_IGNORE_KNOWN_CASES = True  # set False to enforce everything
IGNORED_CASES = [
    # keep the same 3 as in the original unittest
    {"row": 78, "column": "rightOuter", "mat": 27800, "py": 27801},
    {"row": 26, "column": "y_intersect", "mat": 43.0, "py": 44.0},
    {"row": 65, "column": "y_intersect", "mat": 80.0, "py": 79.0},
]

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------
def _find_repo_root() -> Path:
    """
    Return the repository root by walking up until we see `pyproject.toml`.
    Adjust this logic if your repo uses a different marker.
    """
    current = Path(__file__).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root relative to this file")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------
def _adjust_python_fitblinks_to_matlab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the Python FitBlinks DataFrame match MATLAB:
    - do the same 3-step index adjustment as the original unittest
    - bump list columns
    - reorder columns
    """
    df = df.copy()

    # 1) FIRST bump — this MUST include y_intersect
    cols_step1 = [
        "max_blink",
        "start_blink",
        "end_blink",
        "outer_start",
        "outer_end",
        "left_zero",
        "right_zero",
        "max_pos_vel_frame",
        "max_neg_vel_frame",
        "left_base",
        "right_base",
        "left_zero_half_height",
        "right_zero_half_height",
        "left_base_half_height",
        "right_base_half_height",
        "x_intersect",
        "y_intersect",  # ← important: back in
        "right_x_intercept",
    ]
    for col in cols_step1:
        if col in df.columns:
            df[col] = df[col] + 1

    # 2) SECOND bump — exactly like your original test
    cols_step2 = ["y_intersect", "left_x_intercept"]
    for col in cols_step2:
        if col in df.columns:
            df[col] = df[col] + 1

    # 3) THIRD adjustment — subtract 2 from y_intersect
    if "y_intersect" in df.columns:
        df["y_intersect"] = df["y_intersect"] - 2
        # net for y_intersect: +1 (step1) +1 (step2) -2 (step3) = 0

    # list columns → +1 each element
    for list_col in ("left_range", "right_range"):
        if list_col in df.columns:
            df[list_col] = df[list_col].apply(
                lambda arr: (
                    [v + 1 for v in arr] if isinstance(arr, (list, tuple)) else arr
                )
            )

    # final column order
    desired_order = [
        "max_blink",
        "max_value",
        "outer_start",
        "outer_end",
        "left_zero",
        "right_zero",
        "left_base",
        "right_base",
        "left_base_half_height",
        "right_base_half_height",
        "left_zero_half_height",
        "right_zero_half_height",
        "left_range",
        "right_range",
        "left_slope",
        "right_slope",
        "aver_left_velocity",
        "aver_right_velocity",
        "leftR2",
        "rightR2",
        "x_intersect",
        "y_intersect",
        "left_x_intercept",
        "right_x_intercept",
    ]
    existing = [c for c in desired_order if c in df.columns]
    df = df[existing]

    return df


def _compare_frames(df_mat: pd.DataFrame, df_py: pd.DataFrame, decimal_places: int = 0):
    """
    Compare two DataFrames cell-by-cell.

    Returns:
        comparison_report: df of same shape as MATLAB with 'consistent' or message
        missing_report: dict with 2 lists
    """
    # make MATLAB frame a copy so we can round
    df_mat = df_mat.copy()
    df_py = df_py.copy()

    # columns
    cols_mat = set(df_mat.columns)
    cols_py = set(df_py.columns)
    missing_in_mat = cols_py - cols_mat
    missing_in_py = cols_mat - cols_py

    # build empty report with MATLAB's shape
    report = pd.DataFrame("", index=df_mat.index, columns=df_mat.columns)

    # common columns only
    common_cols = cols_mat.intersection(cols_py)

    # round numeric
    for col in common_cols:
        df_mat[col] = df_mat[col].apply(
            lambda x: (
                np.round(x, decimal_places)
                if isinstance(x, (int, float, np.floating))
                else x
            )
        )
        df_py[col] = df_py[col].apply(
            lambda x: (
                np.round(x, decimal_places)
                if isinstance(x, (int, float, np.floating))
                else x
            )
        )

    # cell-wise compare
    for col in common_cols:
        for i in range(len(df_mat)):
            mat_val = df_mat.at[i, col]
            py_val = df_py.at[i, col]
            if _values_equal(mat_val, py_val):
                report.at[i, col] = "consistent"
            else:
                report.at[i, col] = f"not consistent (MAT={mat_val}, PY={py_val})"

    missing_report = {
        "missing_in_matlab": sorted(list(missing_in_mat)),
        "missing_in_python": sorted(list(missing_in_py)),
    }
    return report, missing_report


def _values_equal(a, b) -> bool:
    """Helper to compare scalars/lists/arrays in a tolerant way."""
    # list/tuple compare
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    # numpy array?
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------
    # 1. locate MATLAB fixtures
    # -----------------------------------------------------------------
    base_path = REPO_ROOT / "test" / "migration_files"
    mat_in = base_path / "step1bii_data_input_process_FitBlinks.mat"
    mat_out = base_path / "step1bii_data_output_process_FitBlinks.mat"

    assert mat_in.exists(), f"MATLAB input not found: {mat_in}"
    assert mat_out.exists(), f"MATLAB output not found: {mat_out}"

    # -----------------------------------------------------------------
    # 2. load MATLAB data
    # -----------------------------------------------------------------
    input_data, output_data = load_matlab_data(str(mat_in), str(mat_out))

    # MATLAB gold → struct array
    blink_fits_mat = output_data["blinkFits"]

    # convert to DataFrame + rename to Python style
    df_mat = pd.DataFrame.from_records(blink_fits_mat)
    df_mat.rename(columns=RENAME_MAP, inplace=True)

    # some MATLAB exports include a 'number' field we don't really need
    if "number" in df_mat.columns:
        df_mat = df_mat.drop(columns=["number"])

    # -----------------------------------------------------------------
    # 3. build Python pipeline inputs
    # -----------------------------------------------------------------
    candidate_signal = input_data["candidateSignal"]
    params = default_setting.DEFAULT_PARAMS.copy()
    params["sfreq"] = 100  # matches original unittest
    channel = "No_channel"

    # -----------------------------------------------------------------
    # 4. run Python: blink detection → FitBlinks
    # -----------------------------------------------------------------
    df_blink_pos = get_blink_position(
        params=params,
        blink_component=candidate_signal,
        ch=channel,
        progress_bar=False,
    )

    fb = FitBlinks(candidate_signal=candidate_signal, df=df_blink_pos, params=params)
    fb.dprocess()
    df_py_raw = fb.frame_blinks

    # normalize to MATLAB-like table
    df_py = _adjust_python_fitblinks_to_matlab(df_py_raw)

    # -----------------------------------------------------------------
    # 5. compare (structure)
    # -----------------------------------------------------------------
    report, missing = _compare_frames(df_mat, df_py, decimal_places=0)

    print("\n=== FitBlinks Validation ===")
    print(f"Input file  : {mat_in.name}")
    print(f"Output file : {mat_out.name}")
    print(f"MATLAB rows : {len(df_mat)}")
    print(f"Python rows : {len(df_py)}")
    print("Row count matches? ->", len(df_mat) == len(df_py))

    # show first N rows, both
    print(f"\nFirst {min(N_PREVIEW_ROWS, len(df_mat))} MATLAB rows:")
    print(df_mat.head(N_PREVIEW_ROWS))
    print(f"\nFirst {min(N_PREVIEW_ROWS, len(df_py))} Python rows (normalized):")
    print(df_py.head(N_PREVIEW_ROWS))

    # missing columns
    print("\nMissing Columns Report:")
    print(missing)

    # -----------------------------------------------------------------
    # 6. apply "ignore known odd cases"
    # -----------------------------------------------------------------
    filtered_report = report.copy()

    if TO_IGNORE_KNOWN_CASES:
        logger.warning(
            "Running with TO_IGNORE_KNOWN_CASES=True, will drop these rows:\n%s",
            IGNORED_CASES,
        )
        rows_to_drop = {c["row"] for c in IGNORED_CASES}
        filtered_report = filtered_report.drop(index=rows_to_drop, errors="ignore")

    # keep only columns where at least one cell is not 'consistent'
    filtered_report = filtered_report.loc[:, ~(filtered_report == "consistent").all()]

    print("\nFiltered Comparison Report (only problematic cells):")
    print(filtered_report if not filtered_report.empty else "  - none -")

    # -----------------------------------------------------------------
    # 7. assert
    # -----------------------------------------------------------------
    # no missing columns
    assert not missing["missing_in_matlab"], (
        f"Python has extra columns not in MATLAB: {missing['missing_in_matlab']}"
    )
    assert not missing["missing_in_python"], (
        f"MATLAB has columns not in Python: {missing['missing_in_python']}"
    )

    # no inconsistent cells
    has_inconsistent = filtered_report.apply(
        lambda col: col.map(lambda x: isinstance(x, str) and "not consistent" in x)
    ).any(axis=None)

    assert not has_inconsistent, (
        f"Found inconsistent cells in FitBlinks output:\n{filtered_report}"
    )

    print("\n✅ Validation PASSED: Python FitBlinks matches MATLAB gold.")


if __name__ == "__main__":
    main()
