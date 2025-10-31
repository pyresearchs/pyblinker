#!/usr/bin/env python
"""
Tutorial: validate Python blink-position detection against MATLAB gold standard

Purpose
-------
This tutorial script shows how to:

1. Load **MATLAB input data** that was originally used by EEGLAB/Blinker.
2. Run the Python implementation `get_blink_position(...)` on that exact input.
3. Load the **MATLAB ground-truth output**.
4. Align indexing differences (MATLAB = 1-based, Python = 0-based).
5. Assert that Python and MATLAB produce the **same blink intervals**.
6. Print simple statistics (row counts, mismatches, duplicates).

Why this matters
----------------
When you migrate functionality from MATLAB (e.g. Blinker) to Python
(e.g. `pyblinker`), you need to prove that the Python version reproduces
the MATLAB output. This script treats the MATLAB output as the
**gold standard** and compares the Python output against it.

Files expected
--------------
- Input  (from MATLAB):  `test/migration_files/step1bi_data_input_getBlinkPositions.mat`
  This contains, at minimum:
    - `blinkComp`     : 1D array, the blink-related component/signal
    - `srate`         : sampling rate (Hz)
    - `stdThreshold`  : threshold used in MATLAB
- Output (MATLAB GT):   `test/migration_files/step1bi_data_output_getBlinkPositions.mat`
  This contains:
    - `blinkPositions`: a 2×N matrix (row 0 = start, row 1 = end), **1-based**

Assumptions
-----------
- Python function to test: `pyblinker.blinker.get_blink_positions.get_blink_position`
- Helper loader:          `test.blinker_migration.debugging_tools.load_matlab_data`
- Minimal event length is not stored in the .mat file, so we assume 0.05 s
  (this matches the test logic in the original unit test).

What this script does
---------------------
- Loads .mat → builds params → runs Python detector
- Converts Python output to 1-based so it’s comparable to MATLAB
- Asserts equality (start and end)
- Prints:
    * total rows (MATLAB vs Python)
    * whether shapes match
    * first N rows side by side
    * rows that are duplicated / repeated

Run
---
Just run:

    python tutorial_get_blink_positions.py

If everything matches, you’ll see “✅ Validation PASSED”.
If not, the script will raise an AssertionError and print diffs.

"""

from pathlib import Path
import numpy as np
import pandas as pd

from pyblinker.blinker.get_blink_positions import get_blink_position
from test.blinker_migration.debugging_tools import load_matlab_data


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
N_PREVIEW_ROWS = 10  # how many rows to show for quick visual inspection


def main():
    # -----------------------------------------------------------------
    # 1. Locate test data
    # -----------------------------------------------------------------
    base_path = Path(__file__).resolve().parents[0] / "test" / "migration_files"
    # mat_input_path = base_path / "step1bi_data_input_getBlinkPositions.mat"
    # mat_output_path = base_path / "step1bi_data_output_getBlinkPositions.mat"
    mat_input_path = Path(r"/test/migration_files/step1bi_data_input_getBlinkPositions.mat")
    mat_output_path = Path(r"/test/migration_files/step1bi_data_output_getBlinkPositions.mat")
    assert mat_input_path.exists(), f"Input .mat not found: {mat_input_path}"
    assert mat_output_path.exists(), f"Output .mat not found: {mat_output_path}"

    # -----------------------------------------------------------------
    # 2. Load MATLAB input + MATLAB expected output
    #    load_matlab_data should return two dict-like objects
    # -----------------------------------------------------------------
    input_data, output_data = load_matlab_data(str(mat_input_path), str(mat_output_path))

    # required keys from MATLAB input
    for key in ("blinkComp", "srate", "stdThreshold"):
        assert key in input_data, f"Expected key '{key}' in MATLAB input, got: {list(input_data.keys())}"

    blink_component = input_data["blinkComp"]
    srate = float(input_data["srate"])
    std_threshold = float(input_data["stdThreshold"])

    # MATLAB ground truth blink positions (1-based)
    # expected shape: 2 x N  (row 0 = start, row 1 = end)
    blink_positions_mat = output_data["blinkPositions"]
    assert blink_positions_mat.shape[0] == 2, (
        f"Expected MATLAB blinkPositions to be 2×N, got {blink_positions_mat.shape}"
    )

    # -----------------------------------------------------------------
    # 3. Build params for Python function
    # -----------------------------------------------------------------
    params = {
        "sfreq": srate,
        "std_threshold": std_threshold,
        "min_event_len": 0.05,  # assumed, same as in unit test
    }

    # -----------------------------------------------------------------
    # 4. Run Python function (note: Python uses 0-based indexing)
    # -----------------------------------------------------------------
    result_df = get_blink_position(
        params=params,
        blink_component=blink_component,
        ch="No_channel",
        progress_bar=False,
    )

    # basic shape / type checks
    assert isinstance(result_df, pd.DataFrame), "Python result must be a pandas DataFrame"
    assert list(result_df.columns) == ["start_blink", "end_blink"], (
        f"Unexpected columns: {list(result_df.columns)}"
    )

    # -----------------------------------------------------------------
    # 5. Convert Python → 1-based so we can compare to MATLAB
    # -----------------------------------------------------------------
    result_df_1based = result_df.copy()
    result_df_1based[["start_blink", "end_blink"]] = (
            result_df_1based[["start_blink", "end_blink"]] + 1
    )

    # -----------------------------------------------------------------
    # 6. Build MATLAB dataframe to compare
    # -----------------------------------------------------------------
    expected_df = pd.DataFrame(
        {
            "start_blink": blink_positions_mat[0, :],
            "end_blink": blink_positions_mat[1, :],
        }
    )

    # Ensure integer dtype for comparison
    result_start = result_df_1based["start_blink"].astype(int).to_numpy()
    result_end = result_df_1based["end_blink"].astype(int).to_numpy()
    expected_start = expected_df["start_blink"].astype(int).to_numpy()
    expected_end = expected_df["end_blink"].astype(int).to_numpy()

    # -----------------------------------------------------------------
    # 7. Print statistics BEFORE asserting
    # -----------------------------------------------------------------
    print("\n=== Blink Detection Validation ===")
    print(f"Input file  : {mat_input_path.name}")
    print(f"Output file : {mat_output_path.name}")
    print(f"Python rows : {len(result_df_1based)}")
    print(f"MATLAB rows : {len(expected_df)}")
    print("Row count matches? ->", len(result_df_1based) == len(expected_df))

    # Show first N rows side by side
    print(f"\nFirst {min(N_PREVIEW_ROWS, len(expected_df))} rows (MATLAB vs Python):")
    preview = pd.DataFrame(
        {
            "mat_start": expected_start[:N_PREVIEW_ROWS],
            "mat_end": expected_end[:N_PREVIEW_ROWS],
            "py_start": result_start[:N_PREVIEW_ROWS],
            "py_end": result_end[:N_PREVIEW_ROWS],
        }
    )
    print(preview)

    # Show duplicate (repeated) intervals, if any
    print("\nDuplicate intervals in Python result (if any):")
    dup_py = (
        result_df_1based.value_counts(subset=["start_blink", "end_blink"])
        .reset_index(name="count")
    )
    dup_py = dup_py[dup_py["count"] > 1]
    if dup_py.empty:
        print("  - none -")
    else:
        print(dup_py)

    print("\nDuplicate intervals in MATLAB GT (if any):")
    dup_mat = (
        expected_df.value_counts(subset=["start_blink", "end_blink"])
        .reset_index(name="count")
    )
    dup_mat = dup_mat[dup_mat["count"] > 1]
    if dup_mat.empty:
        print("  - none -")
    else:
        print(dup_mat)

    # -----------------------------------------------------------------
    # 8. ASSERT equality (this will raise if migration is not exact)
    # -----------------------------------------------------------------
    # compare starts
    np.testing.assert_array_equal(
        result_start, expected_start, err_msg="Start indices differ between Python and MATLAB"
    )
    # compare ends
    np.testing.assert_array_equal(
        result_end, expected_end, err_msg="End indices differ between Python and MATLAB"
    )

    print("\n✅ Validation PASSED: Python blink positions match MATLAB ground truth.")


if __name__ == "__main__":
    main()
