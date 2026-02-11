#!/usr/bin/env python
"""
What this tutorial does
-----------------------
This tutorial demonstrates how the channel-selection stage of the blinker
migration chooses a representative signal. We reproduce the filtering logic
implemented in ``pyblinker`` and compare the result with the MATLAB export used
by the migration unit tests.

Based on
--------
test/blinker_migration/test_step3_select_channels.py

Inputs
------
- ``test/migration_files/step3a_input_selectChannel_compact.mat``:
  MATLAB-exported structures containing the blink statistics for every channel.
- ``test/migration_files/step3a_output_selectChannel_compact.mat``:
  MATLAB reference output with the expected per-channel statistics after the
  selection filters are applied.
- Column rename rules from
  ``test/blinker_migration/pyblinker/utils/update_pkl_variables.py`` to align
  MATLAB field names with the Python schema.

Outputs / Validation
--------------------
- Python rebuilds the channel statistics table using
  :func:`pyblinker.blinker.get_representative_channel.filter_*` helpers and the
  default migration parameters.
- The MATLAB reference table is normalised to the same schema (columns renamed,
  non-comparable bookkeeping columns dropped, sorted by channel label).
- A comparison report lists any column-level differences between the Python and
  MATLAB tables. If the report is empty, the migration behaviour matches the
  MATLAB implementation.

How to run
----------
Run this file directly (``python migration_step3_tutorial_validate_channel_selection.py``)
from the repository root. The script prints a summary of the comparison and any
remaining discrepancies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

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

from pyblinker.blinker import default_setting  # noqa: E402
from pyblinker.blinker.get_representative_channel import (  # noqa: E402
    filter_blink_amplitude_ratios,
    filter_good_blinks,
    filter_good_ratio,
    select_max_good_blinks,
)
from test.blinker_migration.obs.debugging_tools import load_matlab_data  # noqa: E402
from test.blinker_migration import (  # noqa: E402
    RENAME_MAP,
)

COLUMNS_TO_IGNORE = ("status", "select")
DROP_COLUMNS = ("signal", "blinkPositions", "signalType", "signalNumber")


def _prepare_signal_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    """Return a normalised :class:`pandas.DataFrame` for channel statistics."""

    df = pd.DataFrame.from_records(records)
    df = df.drop(columns=list(DROP_COLUMNS), errors="ignore")
    df = df.rename(columns={"signalLabel": "ch"})
    df.rename(columns=RENAME_MAP, inplace=True)
    return df


def _load_matlab_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load input and reference tables from the MATLAB migration fixtures."""

    fixtures_dir = REPO_ROOT / "test" / "migration_files"
    input_path = fixtures_dir / "step3a_input_selectChannel_compact.mat"
    output_path = fixtures_dir / "step3a_output_selectChannel_compact.mat"

    matlab_input, matlab_output = load_matlab_data(str(input_path), str(output_path))
    signal_data = _prepare_signal_dataframe(matlab_input["signalData"])

    matlab_reference = pd.DataFrame.from_records(matlab_output["blinks"]["signalData"])
    matlab_reference = matlab_reference.drop(columns=list(DROP_COLUMNS), errors="ignore")
    matlab_reference = matlab_reference.rename(columns={"signalLabel": "ch"})
    matlab_reference.rename(columns=RENAME_MAP, inplace=True)

    return signal_data, matlab_reference


def _apply_channel_selection(signal_data: pd.DataFrame) -> pd.DataFrame:
    """Run the representative-channel selection pipeline."""

    params = default_setting.DEFAULT_PARAMS.copy()
    stats = signal_data.copy()
    stats = filter_blink_amplitude_ratios(stats, params)
    stats = filter_good_blinks(stats, params)
    stats = filter_good_ratio(stats, params)
    stats = select_max_good_blinks(stats)
    return stats


def _drop_non_comparable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without bookkeeping columns that MATLAB does not expose."""

    return df.drop(columns=list(COLUMNS_TO_IGNORE), errors="ignore")


def _compare_tables(python_df: pd.DataFrame, matlab_df: pd.DataFrame) -> pd.DataFrame:
    """Return a :class:`pandas.DataFrame` describing per-cell differences."""

    python_sorted = python_df.sort_values(by="ch").reset_index(drop=True)
    matlab_sorted = matlab_df.sort_values(by="ch").reset_index(drop=True)
    return matlab_sorted.compare(python_sorted, align_axis=1)


def main() -> None:
    """Execute the channel-selection tutorial."""

    signal_data, matlab_reference = _load_matlab_tables()

    print("Loaded MATLAB fixtures with", len(signal_data), "channels to evaluate.")

    python_selection = _apply_channel_selection(signal_data)
    python_selection = _drop_non_comparable_columns(python_selection)
    matlab_selection = _drop_non_comparable_columns(matlab_reference)

    differences = _compare_tables(python_selection, matlab_selection)

    if differences.empty:
        print("\nAll channel statistics match the MATLAB reference after sorting by 'ch'.")
    else:
        print("\nDifferences detected between Python and MATLAB channel statistics:")
        print(differences)
        differing_columns = sorted({col for col, _ in differences.columns})
        print("\nColumns with mismatches:", ", ".join(differing_columns))
        print(
            "Review the rows above to understand how the Python migration differs",
            "from the MATLAB output.",
        )


if __name__ == "__main__":
    main()
