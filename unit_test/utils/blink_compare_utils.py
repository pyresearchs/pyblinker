"""Helpers for comparing blink property tables in tests.

These utilities assist unit tests that verify blink property extraction.
They provide conversions between epoch-level metadata and long-format
per-blink tables as well as DataFrame comparison helpers.
"""

from __future__ import annotations

import ast
import logging
from typing import Sequence

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def metadata_to_long(epochs: mne.Epochs) -> pd.DataFrame:
    """Convert list-based blink metadata to a long-format table.

    Parameters
    ----------
    epochs
        Epochs instance containing list-valued blink metadata.

    Returns
    -------
    pandas.DataFrame
        One row per blink with ``seg_id`` and ``blink_id`` identifiers.
    """
    rows: list[dict[str, float]] = []
    md = epochs.metadata
    for idx, row in md.iterrows():
        n = int(row.get("n_blinks", 0))
        if n <= 0:
            continue
        seg_id = int(epochs.selection[idx])
        for i in range(n):
            rec: dict[str, float] = {"seg_id": seg_id, "blink_id": i}
            for col, val in row.items():
                if col == "n_blinks":
                    continue
                if isinstance(val, list):
                    rec[col] = val[i] if i < len(val) else float("nan")
                else:
                    rec[col] = val
            rows.append(rec)
    return pd.DataFrame(rows)


def report_mismatches(
    result: pd.DataFrame,
    reference: pd.DataFrame,
    key_cols: Sequence[str],
    compare_cols: Sequence[str],
) -> None:
    """Log detailed mismatches between result and reference frames.

    Parameters
    ----------
    result, reference
        DataFrames sorted by ``key_cols``.
    key_cols
        Columns identifying each blink uniquely.
    compare_cols
        All columns to be compared.
    """
    merged = pd.merge(
        reference,
        result,
        on=list(key_cols),
        how="outer",
        suffixes=("_ref", "_res"),
        indicator=True,
    )

    missing = merged[merged["_merge"] == "left_only"][list(key_cols)]
    extra = merged[merged["_merge"] == "right_only"][list(key_cols)]
    if not missing.empty:
        logger.error("Missing rows in result:\n%s", missing)
    if not extra.empty:
        logger.error("Unexpected rows in result:\n%s", extra)

    both = merged[merged["_merge"] == "both"]
    ref_vals = both[[f"{c}_ref" for c in compare_cols]].rename(
        columns=lambda c: c[:-4]
    )
    res_vals = both[[f"{c}_res" for c in compare_cols]].rename(
        columns=lambda c: c[:-4]
    )
    diff = res_vals.compare(ref_vals, keep_equal=False)
    if not diff.empty:
        logger.error("Value mismatches:\n%s", diff)


def scalarize(val: object) -> float:
    """Return the first numeric value from scalars, lists or strings."""
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (SyntaxError, ValueError):
            return float(val)
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return float(val[0]) if len(val) else float(np.nan)
    return float(val)

