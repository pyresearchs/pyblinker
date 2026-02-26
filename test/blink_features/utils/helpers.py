"""Test utilities for blink feature modules."""
from __future__ import annotations

from typing import Sequence, Iterable, List

import numpy as np
import pandas as pd

from pyblinker.blink_features.morphology.core_metrics import MORPHOLOGY_METRIC_STEMS
from pyblinker.blink_features.energy.helpers import compute_basic_statistics
from pyblinker.utils.modality import infer_modality


def assert_numeric_or_nan(testcase, values: Iterable[float]) -> None:
    """Assert all values are finite numbers or NaN."""
    arr = np.asarray(list(values), dtype=float)
    condition = np.isfinite(arr) | np.isnan(arr)
    testcase.assertTrue(condition.all(), msg="Values contain non-numeric entries")


def assert_df_has_columns(testcase, df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Assert that a DataFrame contains the specified columns."""
    missing = [c for c in columns if c not in df.columns]
    testcase.assertFalse(missing, msg=f"Missing columns: {missing}")


def morphology_column_names(channels: Sequence[str]) -> List[str]:
    """Return expected morphology feature columns for given channels."""
    metrics = tuple(f"{stem}_base" for stem in MORPHOLOGY_METRIC_STEMS) + ("duration",)
    stats = tuple(compute_basic_statistics([]).keys())
    columns = []
    for ch in channels:
        modality = infer_modality(ch)
        style = "base"
        for metric in metrics:
            for stat in stats:
                columns.append(f"{modality}__{style}__morphology__{metric}_{stat}__{ch}")
    return columns
