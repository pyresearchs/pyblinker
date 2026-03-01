"""This tutorial reproduces the end-to-end MATLAB migration check.

What this tutorial does
-----------------------
Rebuild the complete blink-processing pipeline (Steps 1–3) in Python using
the same parameters as the legacy MATLAB workflow and compare the resulting
``combinedStruct`` table to the MATLAB reference export.

Based on
--------
test/blinker_migration/xtest_immitate_full_step.py

Inputs
------
- ``test/migration_files/step1bi_data_input_getBlinkPositions.mat``: legacy
  MATLAB structure containing the blink component signal and default
  detection parameters.
- ``test/migration_files/immitate_full_step.mat``: MATLAB reference output
  after the full pipeline has run, exposing the ``combinedStruct`` table.
- Column rename rules from
  ``test/blinker_migration/pyblinker/utils/update_pkl_variables.py`` to align
  MATLAB field names with the Python implementation.

Outputs / Validation
--------------------
- Python reconstructs blink positions, fits each blink, computes blink
  statistics, filters good blinks, and extracts waveform properties using the
  same settings as the unit test.
- Both MATLAB and Python tables are reduced to the shared set of migration
  columns. Numeric fields are rounded to match the zero-decimal tolerance in
  the test.
- A difference report lists any cells that disagree, excluding the single
  known ``y_intersect`` offset documented in the original test. Missing
  columns are also reported if present.
- When the report is empty, the Python pipeline faithfully reproduces the
  MATLAB ``combinedStruct``.

How to run
----------
Execute this file directly (``python migration_full_pipeline_tutorial_validate_combined_struct.py``)
from the repository root to print the validation summary.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _find_repo_root() -> Path:
    """Return the repository root by searching upwards for ``pyproject.toml``."""

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
from pyblinker.blinker.fit_blink import FitBlinks  # noqa: E402
from pyblinker.blinker.get_blink_positions import get_blink_position  # noqa: E402
from pyblinker.utils.statistics_utils import (  # noqa: E402
    get_blink_statistic,
    get_good_blink_mask,
)
from test.blinker_migration.obs.debugging_tools import load_matlab_data  # noqa: E402
from test.blinker_migration import (  # noqa: E402
    RENAME_MAP,
)


@dataclass
class ComparisonReport:
    missing_in_python: list[str]
    missing_in_matlab: list[str]
    differing_cells: pd.DataFrame

    def is_match(self) -> bool:
        return (
            not self.missing_in_python
            and not self.missing_in_matlab
            and self.differing_cells.empty
        )


IGNORED_CASES = [
    {"row": 41, "column": "y_intersect", "matlab": 43.0, "python": 44.0},
]

ROUND_DECIMALS = 0
COLUMN_ORDER = [
    "max_blink",
    "max_value",
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


def _load_pipeline_inputs():
    """Load MATLAB fixtures and initialise default parameters."""

    mat_dir = REPO_ROOT / "test" / "migration_files"
    input_mat = mat_dir / "step1bi_data_input_getBlinkPositions.mat"
    output_mat = mat_dir / "immitate_full_step.mat"
    matlab_input, matlab_output = load_matlab_data(str(input_mat), str(output_mat))

    params = default_setting.DEFAULT_PARAMS.copy()
    params["sfreq"] = 100

    blink_component = matlab_input["blinkComp"]
    matlab_table = pd.DataFrame.from_records(matlab_output["combinedStruct"])
    matlab_table.rename(columns=RENAME_MAP, inplace=True)

    return params, blink_component, matlab_table


def _adjust_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the MATLAB index adjustments from the unit test."""

    df = df.copy()
    columns_to_increment = [
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
        "y_intersect",
        "right_x_intercept",
    ]
    for column in columns_to_increment:
        if column in df:
            df[column] += 1

    for column in ["y_intersect", "left_x_intercept"]:
        if column in df:
            df[column] += 1

    if "y_intersect" in df:
        df["y_intersect"] -= 2

    for column in ["left_range", "right_range"]:
        if column in df:
            df[column] = df[column].apply(lambda values: [val + 1 for val in values])

    return df


def _round_numeric(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Round numeric columns while preserving nested lists."""

    rounded = df.copy()
    for column in rounded.columns:
        if np.issubdtype(rounded[column].dtype, np.number):
            rounded[column] = rounded[column].round(decimals)
        else:
            rounded[column] = rounded[column].apply(
                lambda value: (
                    [np.round(v, decimals) for v in value]
                    if isinstance(value, (list, tuple, np.ndarray))
                    else value
                )
            )
    return rounded


def _filter_known_offsets(differences: pd.DataFrame) -> pd.DataFrame:
    """Remove the single documented offset from the report."""

    if differences.empty:
        return differences

    mask = pd.Series([True] * len(differences))
    for case in IGNORED_CASES:
        mask &= ~(
            (differences["row"] == case["row"])
            & (differences["column"] == case["column"])
            & (differences["matlab"] == case["matlab"])
            & (differences["python"] == case["python"])
        )
    return differences[mask]


def _build_pipeline_output(params, blink_component) -> pd.DataFrame:
    """Run the Python migration pipeline and return the candidate table."""

    blink_positions = get_blink_position(
        params,
        blink_component=blink_component,
        ch="No_channel",
        progress_bar=False,
    )

    fit_blinks = FitBlinks(
        candidate_signal=blink_component, df=blink_positions, params=params
    )
    fit_blinks.dprocess()
    blink_frame = fit_blinks.frame_blinks

    blink_stats = get_blink_statistic(
        blink_frame, params["z_thresholds"], signal=blink_component
    )

    _, filtered_frame = get_good_blink_mask(
        blink_frame,
        blink_stats["best_median"],
        blink_stats["best_robust_std"],
        params["z_thresholds"],
    )

    properties_df = BlinkProperties(
        blink_component, filtered_frame, params["sfreq"], params
    ).df

    condition_1 = properties_df["pos_amp_vel_ratio_zero"] < params["p_avr_threshold"]
    condition_2 = properties_df["max_value"] < (
        blink_stats["best_median"] - blink_stats["best_robust_std"]
    )
    return properties_df[~(condition_1 & condition_2)]


def _compare_tables(
    matlab_df: pd.DataFrame, python_df: pd.DataFrame
) -> ComparisonReport:
    """Align tables, round values, and capture mismatches."""

    matlab_df = matlab_df.copy()
    python_df = _adjust_indices(python_df)

    missing_in_python = [col for col in COLUMN_ORDER if col not in python_df.columns]
    missing_in_matlab = [col for col in COLUMN_ORDER if col not in matlab_df.columns]

    common_columns = [
        col
        for col in COLUMN_ORDER
        if col in python_df.columns and col in matlab_df.columns
    ]

    if "max_blink" not in common_columns:
        raise ValueError(
            "The comparison requires 'max_blink' to be present in both tables."
        )

    python_indexed = python_df[common_columns].set_index("max_blink").sort_index()
    matlab_indexed = matlab_df[common_columns].set_index("max_blink").sort_index()

    shared_keys = python_indexed.index.intersection(matlab_indexed.index)

    python_aligned = _round_numeric(
        python_indexed.loc[shared_keys].reset_index(), ROUND_DECIMALS
    )
    matlab_aligned = _round_numeric(
        matlab_indexed.loc[shared_keys].reset_index(), ROUND_DECIMALS
    )

    differing_rows = []
    for idx in range(len(matlab_aligned)):
        for column in common_columns:
            matlab_value = matlab_aligned.at[idx, column]
            python_value = python_aligned.at[idx, column]
            if isinstance(matlab_value, (list, tuple, np.ndarray)) or isinstance(
                python_value, (list, tuple, np.ndarray)
            ):
                matlab_arr = np.asarray(matlab_value)
                python_arr = np.asarray(python_value)
                if matlab_arr.shape != python_arr.shape or not np.allclose(
                    matlab_arr, python_arr, atol=1e-6
                ):
                    differing_rows.append(
                        {
                            "row": idx,
                            "column": column,
                            "matlab": matlab_value,
                            "python": python_value,
                        }
                    )
            else:
                if pd.isna(matlab_value) and pd.isna(python_value):
                    continue
                if matlab_value != python_value:
                    differing_rows.append(
                        {
                            "row": idx,
                            "column": column,
                            "matlab": matlab_value,
                            "python": python_value,
                        }
                    )

    differences = pd.DataFrame(
        differing_rows, columns=["row", "column", "matlab", "python"]
    )
    differences = _filter_known_offsets(differences)

    return ComparisonReport(
        missing_in_python=missing_in_python,
        missing_in_matlab=missing_in_matlab,
        differing_cells=differences,
    )


def _print_report(report: ComparisonReport) -> None:
    """Emit a readable summary of the comparison."""

    print("\nFull pipeline comparison (MATLAB combinedStruct vs Python)\n" + "-" * 68)
    print(f"Missing columns in Python table: {report.missing_in_python or 'None'}")
    print(f"Missing columns in MATLAB table: {report.missing_in_matlab or 'None'}")

    if report.differing_cells.empty:
        print("No differing cells after rounding and ignoring documented offsets.")
    else:
        print(
            f"Differences detected (MATLAB vs Python) — {len(report.differing_cells)} mismatching cells:"
        )
        print(report.differing_cells)


def main() -> None:
    params, blink_component, matlab_table = _load_pipeline_inputs()
    python_table = _build_pipeline_output(params, blink_component)
    python_table = python_table[COLUMN_ORDER].reset_index(drop=True)
    matlab_table = matlab_table[COLUMN_ORDER].reset_index(drop=True)

    report = _compare_tables(matlab_table, python_table)
    _print_report(report)

    if report.is_match():
        print(
            "\nResult: ✅ The Python pipeline matches the MATLAB combinedStruct export."
        )
    else:
        print("\nResult: ⚠️ Differences remain. Inspect the table above for details.")


if __name__ == "__main__":
    main()
