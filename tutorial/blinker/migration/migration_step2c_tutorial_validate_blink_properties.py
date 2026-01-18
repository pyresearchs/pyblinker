#!/usr/bin/env python
"""
What this tutorial does
-----------------------
This tutorial walks through migrating MATLAB-computed blink properties to Python.
We recompute those properties with ``pyblinker`` and validate them against the
MATLAB reference export that ships with the test suite.

Based on
--------
test/blinker_migration/compare_BlinkProperties.py

Inputs
------
- ``test/migration_files/step2c_data_input_computeBlinkProperties.mat``:
  legacy MATLAB structures containing ``signalData`` and ``blinkFits``.
- ``test/migration_files/step2c_data_output_computeBlinkProperties.mat``:
  MATLAB reference output with ``blinkFits`` and ``blinkProps`` tables.
- Column rename rules from ``test/blinker_migration/pyblinker/utils/update_pkl_variables.py``
  (applied to mirror the Python field names).

Outputs / Validation
--------------------
- Python rebuilds blink properties into a :class:`pandas.DataFrame` using the
  same parameters as the migration unit test.
- MATLAB reference tables are loaded and aligned to the Python schema with
  ``load_matlab_data``.
- A column-by-column report highlights any differences after rounding to
  1 decimal place (matching the unit test tolerance). Known, tiny timing
  differences from the test are called out and ignored.
- If the printed report shows no remaining differences, the migration matches
  the MATLAB reference.

How to run
----------
Run this file directly (``python migration_step2c_tutorial_validate_blink_properties.py``)
from the repository root. The script prints a summary of the comparison and
any remaining discrepancies.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class Difference:
    """Container describing a mismatch between MATLAB and Python outputs."""

    row: int
    column: str
    matlab_value: object
    python_value: object
    difference: object | None

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "column": self.column,
            "matlab_value": self.matlab_value,
            "python_value": self.python_value,
            "difference": self.difference,
        }


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


from pyblinker.blink_features.waveform_features.extract_blink_properties import (  # noqa: E402
    BlinkProperties,
)
from pyblinker.blinker import default_setting  # noqa: E402
from test.blinker_migration.debugging_tools import load_matlab_data  # noqa: E402
from test.blinker_migration.pyblinker.utils.update_pkl_variables import (  # noqa: E402
    RENAME_MAP,
)


IGNORED_CASES = [
    {"row": 10, "column": "peak_time_blink", "matlab": 59.2, "python": 59.1},
    {"row": 33, "column": "peak_time_blink", "matlab": 147.4, "python": 147.3},
    {"row": 34, "column": "peak_time_blink", "matlab": 154.5, "python": 154.4},
]
COLUMNS_TO_DECREASE = [
    "max_blink",
    "outer_start",
    "outer_end",
    "left_zero",
    "right_zero",
    "left_base",
    "right_base",
    "left_zero_half_height",
    "right_zero_half_height",
    "left_base_half_height",
    "right_base_half_height",
]
COLUMNS_TO_DROP = ["peaks_pos_vel_base", "peaks_pos_vel_zero"]
LAST_ROW_NAN_COLUMNS = [
    "inter_blink_max_amp",
    "inter_blink_max_vel_base",
    "inter_blink_max_vel_zero",
]
ROUND_DECIMALS = 1


def _round_value(value: object, decimals: int) -> object:
    """Round scalars and numpy-like containers to the requested decimals."""

    if value is None:
        return None
    if isinstance(value, (float, int, np.number)):
        return float(np.round(value, decimals))
    if isinstance(value, (pd.Series, list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype == object:
            return np.array([_round_value(v, decimals) for v in array], dtype=object)
        return np.round(array.astype(float), decimals)
    return value


def _is_nan_like(value: object) -> bool:
    """Return True if the value should be treated as NaN (scalar or iterable)."""

    if isinstance(value, (pd.Series, list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=object)
        return all(pd.isna(item) for item in array.reshape(-1))
    return bool(pd.isna(value))


def _values_match(matlab_value: object, python_value: object) -> bool:
    """Return True when two rounded values match (considering NaNs)."""

    if _is_nan_like(matlab_value) and _is_nan_like(python_value):
        return True

    if isinstance(matlab_value, (pd.Series, list, tuple, np.ndarray)) or isinstance(
        python_value, (pd.Series, list, tuple, np.ndarray)
    ):
        matlab_array = np.asarray(matlab_value)
        python_array = np.asarray(python_value)
        if matlab_array.shape != python_array.shape:
            return False
        if np.issubdtype(matlab_array.dtype, np.number) and np.issubdtype(
            python_array.dtype, np.number
        ):
            return np.allclose(matlab_array, python_array, atol=1e-8, rtol=0.0, equal_nan=True)
        return np.array_equal(matlab_array, python_array)

    if isinstance(matlab_value, (float, int, np.number)) and isinstance(
        python_value, (float, int, np.number)
    ):
        return bool(np.isclose(matlab_value, python_value, atol=1e-8, rtol=0.0, equal_nan=True))

    return matlab_value == python_value


def _numeric_difference(matlab_value: object, python_value: object) -> object | None:
    """Return python - matlab when both are numeric with matching shapes."""

    if isinstance(matlab_value, (pd.Series, list, tuple, np.ndarray)) or isinstance(
        python_value, (pd.Series, list, tuple, np.ndarray)
    ):
        matlab_array = np.asarray(matlab_value)
        python_array = np.asarray(python_value)
        if matlab_array.shape == python_array.shape and np.issubdtype(
            matlab_array.dtype, np.number
        ) and np.issubdtype(python_array.dtype, np.number):
            return python_array - matlab_array
        return None

    if isinstance(matlab_value, (float, int, np.number)) and isinstance(
        python_value, (float, int, np.number)
    ):
        return float(python_value) - float(matlab_value)

    return None


def _round_dataframe(df: pd.DataFrame, columns: Iterable[str], decimals: int) -> pd.DataFrame:
    """Return a copy with selected columns rounded element-wise."""

    rounded = df.copy()
    for column in columns:
        rounded[column] = rounded[column].apply(lambda value: _round_value(value, decimals))
    return rounded


def _load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MATLAB fixtures and compute the Python blink properties."""

    base_path = REPO_ROOT / "test" / "migration_files"
    mat_input_path = base_path / "step2c_data_input_computeBlinkProperties.mat"
    mat_output_path = base_path / "step2c_data_output_computeBlinkProperties.mat"

    if not mat_input_path.exists() or not mat_output_path.exists():
        raise FileNotFoundError(
            "Expected MATLAB fixtures under test/migration_files. "
            f"Missing: {mat_input_path if not mat_input_path.exists() else mat_output_path}"
        )

    print("Loading MATLAB fixtures:")
    print(f"  Input : {mat_input_path}")
    print(f"  Output: {mat_output_path}")

    input_data, output_data = load_matlab_data(mat_input_path, mat_output_path)

    signal = input_data["signalData"]["signal"]
    blink_fits_input = pd.DataFrame.from_records(input_data["blinkFits"])
    blink_fits_input.rename(columns=RENAME_MAP, inplace=True)

    blink_fits_matlab = pd.DataFrame.from_records(output_data["blinkFits"])
    blink_props_matlab = pd.DataFrame.from_records(output_data["blinkProps"])
    blink_fits_matlab.rename(columns=RENAME_MAP, inplace=True)
    blink_props_matlab.rename(columns=RENAME_MAP, inplace=True)
    matlab_reference = pd.concat([blink_fits_matlab, blink_props_matlab], axis=1)

    blink_fits_input[COLUMNS_TO_DECREASE] = blink_fits_input[COLUMNS_TO_DECREASE] - 1

    python_df = BlinkProperties(
        signal,
        blink_fits_input,
        input_data["srate"],
        default_setting.DEFAULT_PARAMS,
    ).df

    python_df = python_df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    python_df[COLUMNS_TO_DECREASE] = python_df[COLUMNS_TO_DECREASE] + 1
    python_df.loc[python_df.index[-1], LAST_ROW_NAN_COLUMNS] = np.nan

    return matlab_reference, python_df


def _compare_dataframes(matlab_df: pd.DataFrame, python_df: pd.DataFrame) -> List[Difference]:
    """Return a list of differences after rounding and ignoring known cases."""

    common_columns = sorted(set(matlab_df.columns).intersection(python_df.columns) - {"max_blink"})

    rounded_matlab = _round_dataframe(matlab_df, common_columns, ROUND_DECIMALS)
    rounded_python = _round_dataframe(python_df, common_columns, ROUND_DECIMALS)

    differences: List[Difference] = []
    for column in common_columns:
        for row in range(len(rounded_matlab)):
            matlab_value = rounded_matlab.at[row, column]
            python_value = rounded_python.at[row, column]

            if _values_match(matlab_value, python_value):
                continue

            difference = Difference(
                row=row,
                column=column,
                matlab_value=matlab_value,
                python_value=python_value,
                difference=_numeric_difference(matlab_value, python_value),
            )
            differences.append(difference)

    filtered: List[Difference] = []
    for diff in differences:
        should_ignore = any(
            diff.row == case["row"]
            and diff.column == case["column"]
            and _values_match(diff.matlab_value, case["matlab"])
            and _values_match(diff.python_value, case["python"])
            for case in IGNORED_CASES
        )
        if not should_ignore:
            filtered.append(diff)

    return filtered


def main() -> None:
    print("\n=== Blink Properties Migration Tutorial ===\n")

    matlab_df, python_df = _load_dataframes()

    print("Data overview:")
    print(f"  MATLAB rows : {len(matlab_df)}")
    print(f"  Python rows : {len(python_df)}")
    print(f"  Columns compared : {len(set(matlab_df.columns).intersection(python_df.columns)) - 1} (excluding 'max_blink')")

    differences = _compare_dataframes(matlab_df, python_df)

    if not differences:
        print("\n✅ Validation PASSED: Python blink properties match the MATLAB reference after rounding.")
    else:
        print("\n⚠️ Validation produced differences after rounding. Inspect the details below.")
        max_preview = 10
        for diff in differences[:max_preview]:
            diff_dict = diff.to_dict()
            print(
                f"  Row {diff_dict['row']:<3} | Column: {diff_dict['column']:<25} | "
                f"MATLAB: {diff_dict['matlab_value']} | Python: {diff_dict['python_value']} | "
                f"Δ (py-mat): {diff_dict['difference']}"
            )
        if len(differences) > max_preview:
            print(f"  ... and {len(differences) - max_preview} more differences.")

    print("\nNote: The known peak timing offsets from the unit test are ignored in this report.")
    print("If the difference list above is empty, the migration faithfully reproduces MATLAB.")


if __name__ == "__main__":
    main()
