"""Test utilities for blink feature modules."""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def assert_df_has_columns(testcase, df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Assert that a DataFrame contains the specified columns."""
    missing = [c for c in columns if c not in df.columns]
    testcase.assertFalse(missing, msg=f"Missing columns: {missing}")
