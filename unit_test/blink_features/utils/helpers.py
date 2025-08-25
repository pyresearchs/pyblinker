"""Test utilities for blink feature modules."""
from __future__ import annotations

from typing import Sequence, Iterable

import numpy as np
import pandas as pd
from contextlib import contextmanager


def assert_numeric_or_nan(testcase, values: Iterable[float]) -> None:
    """Assert all values are finite numbers or NaN."""
    arr = np.asarray(list(values), dtype=float)
    condition = np.isfinite(arr) | np.isnan(arr)
    testcase.assertTrue(condition.all(), msg="Values contain non-numeric entries")


@contextmanager
def with_userwarning(testcase):
    """Context manager asserting a :class:`UserWarning` is raised."""
    with testcase.assertWarns(UserWarning):
        yield


def assert_df_has_columns(testcase, df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Assert that a DataFrame contains the specified columns."""
    missing = [c for c in columns if c not in df.columns]
    testcase.assertFalse(missing, msg=f"Missing columns: {missing}")
