"""Test utilities for blink feature modules."""
from __future__ import annotations

from typing import Sequence, Iterable, List

from contextlib import contextmanager
import numpy as np
import pandas as pd

from pyblinker.blink_features.energy.helpers import _safe_stats
from pyblinker.blink_features.morphology.per_blink import WAVEFORM_METRICS


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


def morphology_column_names(channels: Sequence[str]) -> List[str]:
    """Return expected morphology feature columns for given channels."""
    metrics = WAVEFORM_METRICS + ("duration",)
    stats = tuple(_safe_stats([]).keys())
    return [f"{m}_{s}_{ch}" for ch in channels for m in metrics for s in stats]
