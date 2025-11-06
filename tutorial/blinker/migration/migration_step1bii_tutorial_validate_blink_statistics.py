#!/usr/bin/env python
"""
What this tutorial does
-----------------------
This tutorial reproduces the blink-statistics migration that verifies the Python
implementation of :func:`pyblinker.utils.statistics_utils.get_blink_statistic`
against the MATLAB export. We compute the statistics for the legacy blink fits
and compare every field with the MATLAB reference tables.

Based on
--------
test/blinker_migration/test_step1bii_v_blinkStatProperties.py

Inputs
------
- ``test/migration_files/step1bii_v_input_blinkStatProperties.mat``:
  MATLAB-exported candidate signals and blink fits in the legacy format.
- ``test/migration_files/step1bii_v_output_blinkStatProperties.mat``:
  MATLAB reference output containing the migrated per-channel statistics.
- Column rename and normalisation helpers from
  ``test/blinker_migration/pyblinker/utils/update_pkl_variables.py``.

Outputs / Validation
--------------------
- Python converts the MATLAB blink fits into a :class:`pandas.DataFrame`, then
  calls :func:`get_blink_statistic` with the same ``z_thresholds`` constants that
  the unit test uses.
- MATLAB reference statistics are normalised to Python field names and stripped
  of metadata fields so that only comparable arrays remain.
- A key-by-key report lists any fields whose values differ by more than
  ``atol=1e-6`` (with NaNs treated as equal). If the report is empty, the Python
  migration reproduces the MATLAB behaviour.

How to run
----------
Run this file directly (``python migration_step1bii_tutorial_validate_blink_statistics.py``)
from the repository root. The script prints the comparison summary and any
remaining discrepancies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

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

from pyblinker.utils.statistics_utils import get_blink_statistic  # noqa: E402
from test.blinker_migration.debugging_tools import load_matlab_data  # noqa: E402
from test.blinker_migration.pyblinker.utils.update_pkl_variables import (  # noqa: E402
    RENAME_MAP,
    rename_keys,
)

COMPARISON_ATOL = 1e-6
FIELDS_TO_REMOVE = (
    "signal",
    "blinkPositions",
    "signalType",
    "signalNumber",
    "signalLabel",
)


def _load_inputs() -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Load MATLAB fixtures and convert them to Python-friendly structures."""

    fixtures_dir = REPO_ROOT / "test" / "migration_files"
    input_path = fixtures_dir / "step1bii_v_input_blinkStatProperties.mat"
    output_path = fixtures_dir / "step1bii_v_output_blinkStatProperties.mat"

    matlab_input, matlab_output = load_matlab_data(str(input_path), str(output_path))

    blink_fits = pd.DataFrame.from_records(matlab_input["blinkFits"])
    blink_fits.rename(columns=RENAME_MAP, inplace=True)

    candidate_signals = matlab_input["candidateSignals"]

    matlab_signal_data = rename_keys(matlab_output["blinks"]["signalData"], RENAME_MAP)
    for field in FIELDS_TO_REMOVE:
        matlab_signal_data.pop(field, None)

    return blink_fits, candidate_signals, matlab_signal_data


def _compute_python_statistics(
    blink_fits: pd.DataFrame, candidate_signals: Dict[str, Any]
) -> Dict[str, Any]:
    """Run ``get_blink_statistic`` using the constants from the unit test."""

    z_thresholds = np.array([[0.9, 0.98], [2.0, 5.0]])
    return get_blink_statistic(blink_fits, z_thresholds, signal=candidate_signals)


def _values_match(matlab_value: Any, python_value: Any) -> bool:
    """Return ``True`` when both values agree within ``COMPARISON_ATOL``."""

    if matlab_value is None and python_value is None:
        return True

    matlab_array = np.asarray(matlab_value)
    python_array = np.asarray(python_value)

    if matlab_array.shape != python_array.shape:
        return False

    if np.issubdtype(matlab_array.dtype, np.number) and np.issubdtype(
        python_array.dtype, np.number
    ):
        return bool(
            np.allclose(matlab_array, python_array, atol=COMPARISON_ATOL, equal_nan=True)
        )

    return np.array_equal(matlab_array, python_array)


def _compare_results(
    matlab_signal_data: Dict[str, Any], python_signal_data: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of keys to their mismatched values."""

    differences: Dict[str, Dict[str, Any]] = {}

    matlab_keys = set(matlab_signal_data)
    python_keys = set(python_signal_data)

    for missing in sorted(matlab_keys - python_keys):
        differences[missing] = {"issue": "Missing from Python result"}

    for missing in sorted(python_keys - matlab_keys):
        differences[missing] = {"issue": "Missing from MATLAB reference"}

    for key in sorted(matlab_keys & python_keys):
        matlab_value = matlab_signal_data[key]
        python_value = python_signal_data[key]
        if not _values_match(matlab_value, python_value):
            differences[key] = {
                "matlab": matlab_value,
                "python": python_value,
            }

    return differences


def _print_report(differences: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print the comparison result for tutorial readers."""

    if not differences:
        print(
            "All blink statistics match the MATLAB reference within",
            f"atol={COMPARISON_ATOL} (NaNs treated as equal).",
        )
        return

    print("Differences detected between Python and MATLAB blink statistics:\n")
    for key, payload in differences.items():
        issue = payload.get("issue")
        if issue:
            print(f"- {key}: {issue}")
            continue
        matlab_value = np.asarray(payload["matlab"])
        python_value = np.asarray(payload["python"])
        print(f"- {key}:")
        print("    MATLAB:", matlab_value)
        print("    Python:", python_value)
        if matlab_value.shape == python_value.shape and np.issubdtype(
            matlab_value.dtype, np.number
        ) and np.issubdtype(python_value.dtype, np.number):
            delta = python_value - matlab_value
            print("    Python - MATLAB:", delta)
    print(
        "\nReview the fields above to understand how the migration result differs",
        "from the MATLAB export.",
    )


def main() -> None:
    """Execute the blink-statistics migration comparison."""

    blink_fits, candidate_signals, matlab_signal_data = _load_inputs()
    print("Loaded", len(blink_fits), "blink fits from the MATLAB fixtures.")

    python_signal_data = _compute_python_statistics(blink_fits, candidate_signals)

    differences = _compare_results(matlab_signal_data, python_signal_data)
    _print_report(differences)


if __name__ == "__main__":
    main()
