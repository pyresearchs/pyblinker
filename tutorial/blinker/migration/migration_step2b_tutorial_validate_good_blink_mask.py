#!/usr/bin/env python
"""
What this tutorial does
-----------------------
This tutorial shows how to reproduce the MATLAB "good blink" selection step in
Python. We rebuild the blink-fit table, run :func:`pyblinker.utils.statistics_utils.get_good_blink_mask`,
and compare the resulting boolean mask to the MATLAB reference exported for the
migration tests.

Based on
--------
test/blinker_migration/test_step2b_getGoodBlinkMask.py

Inputs
------
- ``test/migration_files/step2b_data_input_getGoodBlinkMask.mat``:
  MATLAB structures containing ``blinkFits`` plus supporting statistics
  (``specifiedMedian`` and ``specifiedStd``).
- ``test/migration_files/step2b_data_output_getGoodBlinkMask.mat``:
  MATLAB reference data with the expected ``goodBlinkMask`` column.
- Column rename map from
  ``test/blinker_migration/pyblinker/utils/update_pkl_variables.py`` to align
  MATLAB column names with the Python data frame schema.

Outputs / Validation
--------------------
- Python computes ``good_blink_mask`` using the same z-thresholds as the
  migration test (``[[0.9, 0.98], [2.0, 5.0]]``).
- MATLAB provides the expected ``goodBlinkMask`` array.
- The tutorial prints a short comparison report showing the total number of
  blinks, how many entries match, and any indices where Python and MATLAB
  disagree.
- When the difference report is empty, the migration perfectly reproduces the
  MATLAB selection.

How to run
----------
Execute this file directly from the repository root:

    python tutorial/blinker/migration/migration_step2b_tutorial_validate_good_blink_mask.py

The script prints the comparison summary and exits with code 0 when the masks
match. If discrepancies remain, it prints them and exits with code 1 so that the
failure is visible in automated runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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


from pyblinker.utils.statistics_utils import get_good_blink_mask  # noqa: E402
from test.blinker_migration.obs.debugging_tools import load_matlab_data  # noqa: E402
from test.blinker_migration import (  # noqa: E402
    RENAME_MAP,
)


Z_THRESHOLDS = np.array([[0.9, 0.98], [2.0, 5.0]])


def _format_indices(indices: Iterable[int]) -> str:
    """Return a compact string representation of integer indices."""

    indices = list(indices)
    if not indices:
        return "(none)"
    preview = ", ".join(str(idx) for idx in indices[:10])
    if len(indices) > 10:
        preview += f", … (+{len(indices) - 10} more)"
    return preview


def main() -> int:
    base_path = REPO_ROOT / "test" / "migration_files"
    input_path = base_path / "step2b_data_input_getGoodBlinkMask.mat"
    output_path = base_path / "step2b_data_output_getGoodBlinkMask.mat"

    print("Loading MATLAB input and reference data…")
    input_data, output_data = load_matlab_data(str(input_path), str(output_path))

    print("Preparing blink fit table with Python column names…")
    blink_fits = pd.DataFrame.from_records(input_data["blinkFits"]).rename(
        columns=RENAME_MAP
    )

    specified_median = np.asarray(input_data["specifiedMedian"], dtype=float)
    specified_std = np.asarray(input_data["specifiedStd"], dtype=float)

    print("Running get_good_blink_mask with migration thresholds…")
    python_mask, selected_df = get_good_blink_mask(
        blink_fits=blink_fits,
        specified_median=specified_median,
        specified_std=specified_std,
        z_thresholds=Z_THRESHOLDS,
    )
    python_mask = np.asarray(python_mask, dtype=bool)

    matlab_mask = np.asarray(output_data["goodBlinkMask"], dtype=bool)

    if python_mask.shape != matlab_mask.shape:
        print("⚠️ Shape mismatch between Python and MATLAB outputs")
        print(f"Python mask shape : {python_mask.shape}")
        print(f"MATLAB mask shape : {matlab_mask.shape}")
        return 1

    mismatches = np.where(python_mask != matlab_mask)[0]

    total = python_mask.size
    num_matches = total - mismatches.size

    print("\nComparison summary")
    print("-------------------")
    print(f"Total blink candidates : {total}")
    print(f"Matching entries        : {num_matches}")
    print(f"Mismatching entries     : {mismatches.size}")
    print(f"Mismatch indices        : {_format_indices(mismatches)}")

    if mismatches.size:
        print("\nDetailed mismatches (Python vs MATLAB):")
        comparison = pd.DataFrame(
            {
                "index": mismatches,
                "python": python_mask[mismatches],
                "matlab": matlab_mask[mismatches],
            }
        )
        print(comparison.to_string(index=False))
        print(
            "\n❌ Differences detected. Review the mismatches above to understand why the"
            " selection diverges."
        )
        return 1

    print(
        "\n✅ Validation passed. Python's good blink mask matches the MATLAB reference"
        " for all candidates."
    )

    # selected_df is returned for convenience; display the first few rows so that
    # users can relate the mask to the blink feature table.
    preview_rows = min(5, len(selected_df))
    if preview_rows:
        print("\nPreview of blink fits associated with the mask:")
        print(selected_df.head(preview_rows).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
