"""This tutorial validates the first migration step that locates blink events.

What this tutorial does
-----------------------
Walk through the Step 1a migration workflow by recomputing blink start/end
indices with :func:`pyblinker.blinker.get_blink_position.get_blink_position`
and comparing them to the MATLAB reference output bundled with the test
suite.

Based on
--------
test/blinker_migration/test_step1a.py

Inputs
------
- ``test/migration_files/step1bi_data_input_getBlinkPositions.mat``: legacy
  MATLAB structure containing the blink component signal and detection
  parameters.
- ``test/migration_files/step1bi_data_output_getBlinkPositions.mat``: MATLAB
  reference blink positions exported from the original pipeline.

Outputs / Validation
--------------------
- Python recomputes blink positions and converts them to MATLAB's 1-based
  indexing for a direct comparison.
- A pandas comparison report highlights any column mismatches or row-level
  differences. Values are cast to integers to mirror the unit-test checks.
- If no differences are reported, the migration result matches the MATLAB
  reference.

How to run
----------
Run this file directly (``python migration_step1a_tutorial_validate_candidate_positions.py``)
from the repository root to print a summary of the comparison report.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _find_repo_root() -> Path:
    """Return the repository root by searching for ``pyproject.toml`` upwards."""

    current = Path(__file__).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root relative to this file")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pyblinker.blinker.get_blink_positions import get_blink_position  # noqa: E402
from test.blinker_migration.obs.debugging_tools import load_matlab_data  # noqa: E402


@dataclass
class DataFrameComparison:
    """Simple structure summarising column or row level differences."""

    missing_columns: dict[str, list[str]]
    differing_rows: pd.DataFrame

    def is_match(self) -> bool:
        return (
            not self.missing_columns["missing_in_python"]
            and not self.missing_columns["missing_in_matlab"]
            and self.differing_rows.empty
        )


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MATLAB fixtures and return Python & MATLAB blink tables with aligned schemas."""

    mat_dir = REPO_ROOT / "test" / "migration_files"
    input_mat = mat_dir / "step1bi_data_input_getBlinkPositions.mat"
    output_mat = mat_dir / "step1bi_data_output_getBlinkPositions.mat"

    matlab_input, matlab_output = load_matlab_data(str(input_mat), str(output_mat))

    params = {
        "sfreq": float(matlab_input["srate"]),
        "std_threshold": float(matlab_input["stdThreshold"]),
        "min_event_len": 0.05,
    }

    blink_component = matlab_input["blinkComp"]
    python_positions = get_blink_position(
        params=params,
        blink_component=blink_component,
        ch="No_channel",
        progress_bar=False,
    )

    python_positions = python_positions.copy()
    python_positions[["start_blink", "end_blink"]] += (
        1  # convert to MATLAB's 1-based indexing
    )

    matlab_positions = pd.DataFrame(
        {
            "start_blink": matlab_output["blinkPositions"][0],
            "end_blink": matlab_output["blinkPositions"][1],
        }
    )

    return python_positions, matlab_positions


def _integerise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure comparisons are performed on integer-valued copies."""

    coerced = df.copy()
    for column in coerced.columns:
        coerced[column] = coerced[column].astype(int)
    return coerced


def _compare(python_df: pd.DataFrame, matlab_df: pd.DataFrame) -> DataFrameComparison:
    """Compare DataFrames column-by-column and return mismatches."""

    python_df = _integerise(python_df)
    matlab_df = _integerise(matlab_df)

    missing_in_python = [
        col for col in matlab_df.columns if col not in python_df.columns
    ]
    missing_in_matlab = [
        col for col in python_df.columns if col not in matlab_df.columns
    ]

    aligned_python = python_df[[c for c in matlab_df.columns if c in python_df.columns]]
    aligned_matlab = matlab_df[[c for c in aligned_python.columns]]

    difference_mask = (aligned_python != aligned_matlab).any(axis=1)
    differing_rows = pd.concat(
        [aligned_matlab[difference_mask], aligned_python[difference_mask]],
        axis=1,
        keys=["matlab", "python"],
    )

    return DataFrameComparison(
        missing_columns={
            "missing_in_python": missing_in_python,
            "missing_in_matlab": missing_in_matlab,
        },
        differing_rows=differing_rows,
    )


def _print_report(result: DataFrameComparison) -> None:
    """Print a human-readable report of the comparison outcome."""

    print("\nBlink position comparison (MATLAB vs Python)\n" + "-" * 46)
    if (
        result.missing_columns["missing_in_python"]
        or result.missing_columns["missing_in_matlab"]
    ):
        print("Missing columns detected:")
        for key, columns in result.missing_columns.items():
            print(f"  {key}: {columns if columns else 'None'}")
    else:
        print("No missing columns.")

    if result.differing_rows.empty:
        print("All rows match after converting to MATLAB's 1-based indexing.")
    else:
        print("Differences detected (showing MATLAB vs Python values):")
        print(result.differing_rows)


def main() -> None:
    python_df, matlab_df = _load_inputs()
    comparison = _compare(python_df, matlab_df)
    _print_report(comparison)

    if comparison.is_match():
        print("\nResult: ✅ The migrated blink positions match the MATLAB reference.")
    else:
        print("\nResult: ⚠️ Differences found. Inspect the tables above for details.")


if __name__ == "__main__":
    main()
