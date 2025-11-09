"""Utilities for aligning and comparing blink events between detected and ground truth sets.

All comparisons and tolerance checks in this module are performed in sample index
units (1-based). Callers may convert results to time units (seconds) using the
sampling rate if desired. Reporting helpers in ``pyblinker.utils.evaluation.reporting``
provide such conversions automatically for readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("start_blink", "end_blink")


@dataclass(slots=True)
class Alignment:
    """Represents one greedy alignment between detected and ground truth blink events."""

    ground_truth_idx: Optional[int]
    detected_idx: Optional[int]
    start_diff: Optional[float]
    end_diff: Optional[float]

    def is_match(self, tolerance_samples: int) -> bool:
        """Return ``True`` when both start and end diffs are within tolerance."""

        if self.start_diff is None or self.end_diff is None:
            return False
        return abs(self.start_diff) <= tolerance_samples and abs(self.end_diff) <= tolerance_samples


def coerce_to_one_based(df_or_array_like: Iterable) -> pd.DataFrame:
    """Return a defensive copy of blink events with 1-based integer sample indices.

    Parameters
    ----------
    df_or_array_like
        Either a :class:`pandas.DataFrame` with ``start_blink`` and ``end_blink`` columns
        or any array-like object with shape ``(n_events, 2)`` representing start and end
        samples. Values are coerced to integers. If any sample index is 0-based (i.e.,
        contains a 0), the entire table is shifted by +1.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with ``start_blink`` and ``end_blink`` columns using 1-based
        integer indices.

    Raises
    ------
    ValueError
        If values are missing, non-finite, or negative after coercion.
    """

    if isinstance(df_or_array_like, pd.DataFrame):
        if not set(REQUIRED_COLUMNS).issubset(df_or_array_like.columns):
            raise ValueError(
                "DataFrame must contain 'start_blink' and 'end_blink' columns to coerce."
            )
        df = df_or_array_like.loc[:, REQUIRED_COLUMNS].copy()
    else:
        array = np.asarray(df_or_array_like)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Array-like input must have shape (n_events, 2).")
        df = pd.DataFrame(array, columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Column '{col}' contains non-finite values.")
        df[col] = df[col].astype(int)

    if (df[REQUIRED_COLUMNS] < 0).any().any():
        raise ValueError("Blink sample indices must be non-negative before coercion.")

    if (df[REQUIRED_COLUMNS] == 0).any().any():
        df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS] + 1

    if (df["start_blink"] > df["end_blink"]).any():
        raise ValueError("Each blink event must satisfy start_blink <= end_blink.")

    return df


def validate_event_table(df: pd.DataFrame) -> None:
    """Validate the structure and ordering of a blink event table.

    This function ensures that required columns are present, values are 1-based
    integers, and each event satisfies ``start_blink <= end_blink``. Comparisons are
    always performed in sample index units (1-based).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Blink events must be provided as a pandas DataFrame.")

    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise ValueError(
            "Blink event table must contain 'start_blink' and 'end_blink' columns."
        )

    for col in REQUIRED_COLUMNS:
        if not np.issubdtype(df[col].dtype, np.integer):
            if not np.isfinite(df[col]).all():
                raise ValueError(f"Column '{col}' contains non-finite values.")
            df[col] = df[col].astype(int)
        if (df[col] < 1).any():
            raise ValueError(f"Column '{col}' must use 1-based positive indices.")

    if (df["start_blink"] > df["end_blink"]).any():
        raise ValueError("Each blink event must satisfy start_blink <= end_blink.")

    if not df["start_blink"].is_monotonic_increasing:
        raise ValueError("Blink events must be sorted by start_blink in ascending order.")


def align_events(
    detected_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    tolerance_samples: int,
) -> list[Alignment]:
    """
    Greedily align detected and ground truth blink events by sample index.

    The routine walks both event lists once (O(n)) and tries to pair each
    ground-truth (GT) blink with the earliest *eligible* detected blink.
    A pair is considered a **match** when **both** the start and end deltas
    (detected − ground truth, in samples) are within `tolerance_samples`.
    All indices are **1-based** and inputs must be sorted by ``start_blink``.

    The algorithm is intentionally simple and local (no backtracking):
    it compares the "current" GT blink and "current" detected blink and then:
      1) records a **match** if both deltas are within tolerance, or
      2) records an **unpaired GT** if the GT starts earlier, or
      3) records an **unpaired detection** if the detection starts earlier, or
      4) if starts are equal but outside tolerance, records a **paired-but-out-of-tolerance**
         entry (useful for error analysis) and advances both.

    Parameters
    ----------
    detected_df : pd.DataFrame
        Detected blink events with columns:
          - ``start_blink`` (int): 1-based start index
          - ``end_blink``   (int): 1-based end index
        Must be sorted by ``start_blink``.
    ground_truth_df : pd.DataFrame
        Ground-truth blink events with the same schema and sort order.
    tolerance_samples : int
        Maximum allowed absolute difference (in samples) for both start and end
        to count as a match. Must be non-negative.

    Returns
    -------
    list[Alignment]
        A list of Alignment objects with fields:
          - ``ground_truth_idx`` : int | None
          - ``detected_idx``     : int | None
          - ``start_diff``       : float | None   (detected − GT; None if unpaired)
          - ``end_diff``         : float | None   (detected − GT; None if unpaired)

        Semantics:
          - **Matched pair** → both indices are ints; diffs are floats within tolerance.
          - **Unpaired GT** → ``detected_idx`` is None; diffs are None.
          - **Unpaired detection** → ``ground_truth_idx`` is None; diffs are None.
          - **Paired-but-out-of-tolerance** → both indices are ints; diffs present,
            at least one exceeds tolerance. This captures "near misses" when starts
            are equal but timing is off.

    Notes
    -----
    • Greedy, single-pass alignment favors *earliest plausible* pairings.
      If two detected events are both close to a GT event, only the first can match.
    • Use the presence/absence of diffs to separate *matching quality* (tolerance)
      from *coverage* (paired vs unpaired).

    Flowchart (decision per pair)
    -----------------------------
        +-----------------------------+
        | Compute start_delta,        |
        |         end_delta           |
        +--------------+--------------+
                       |
                       v
        +-----------------------------+
        | |start_delta| <= tol AND    |
        | |end_delta|   <= tol ?      |
        +--------------+--------------+
                       | Yes
                       v
             [Record MATCH; advance both]
                       |
                     (done)
                       |
                       No
                       v
        +-----------------------------+
        | gt_start < det_start ?      |
        +--------------+--------------+
           Yes                        No
           v                          v
    [Record UNPAIRED GT;        +------------------------+
     advance GT only]           | det_start < gt_start ? |
                                +-----------+------------+
                                            | Yes
                                            v
                                   [Record UNPAIRED DET;
                                    advance Det only]
                                            |
                                           No  (starts equal)
                                            v
                       [Record PAIRED-BUT-OUT-OF-TOLERANCE;
                                    advance both]

    Example A — Standard matching, misses, and false positives
    ----------------------------------------------------------
    >>> import pandas as pd
    >>> detected = pd.DataFrame({
    ...     "start_blink": [100, 310, 600],
    ...     "end_blink":   [150, 360, 650],
    ... })
    >>> ground_truth = pd.DataFrame({
    ...     "start_blink": [ 95, 300, 700],
    ...     "end_blink":   [145, 350, 750],
    ... })
    >>> alignments = align_events(detected, ground_truth, tolerance_samples=20)
    >>> for a in alignments:
    ...     print(a)
    Alignment(ground_truth_idx=0, detected_idx=0, start_diff=5.0, end_diff=5.0)
    Alignment(ground_truth_idx=1, detected_idx=1, start_diff=10.0, end_diff=10.0)
    Alignment(ground_truth_idx=2, detected_idx=None, start_diff=None, end_diff=None)
    Alignment(ground_truth_idx=None, detected_idx=2, start_diff=None, end_diff=None)

    Walk-through:
      • GT[0] (95–145) vs Det[0] (100–150): deltas = (+5, +5) → within tol → MATCH.
      • GT[1] (300–350) vs Det[1] (310–360): deltas = (+10, +10) → within tol → MATCH.
      • GT[2] (700–750) vs Det[2] (600–650):
            starts: 700 > 600 → Det starts earlier → UNPAIRED Det[2], advance Det.
            (No detections left) → remaining GT[2] becomes UNPAIRED GT.

    Timeline (1-based samples; ≈ = within tol)
      GT:      [95---------145]   [300--------350]                 [700--------750]
      Det:         [100---------150]   [310--------360]   [600----650]
                 ≈ match #0            ≈ match #1             false positive
                                                               (then GT miss)

    Example B — Starts equal but end outside tolerance → paired-but-out-of-tolerance
    --------------------------------------------------------------------------------
    >>> detected = pd.DataFrame({
    ...     "start_blink": [200],
    ...     "end_blink":   [260],      # ends 30 samples later
    ... })
    >>> ground_truth = pd.DataFrame({
    ...     "start_blink": [200],
    ...     "end_blink":   [230],
    ... })
    >>> align_events(detected, ground_truth, tolerance_samples=20)[0]
    Alignment(ground_truth_idx=0, detected_idx=0, start_diff=0.0, end_diff=30.0)

    Because starts are equal but |end_diff| = 30 > tol, the algorithm records a
    **paired-but-out-of-tolerance** entry and advances both. This preserves the
    near-miss information (useful for bias/latency analysis).

    Example C — Tolerance = 0 (exact matching only)
    -----------------------------------------------
    >>> detected = pd.DataFrame({"start_blink":[100], "end_blink":[150]})
    >>> ground_truth = pd.DataFrame({"start_blink":[101], "end_blink":[151]})
    >>> align_events(detected, ground_truth, tolerance_samples=0)
    [
      Alignment(ground_truth_idx=None, detected_idx=0, start_diff=None, end_diff=None),
      Alignment(ground_truth_idx=0, detected_idx=None, start_diff=None, end_diff=None),
    ]

    With exact matching, any offset produces two unpaired entries (a false positive
    and a miss) unless the indices are identical.
    """

    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative.")

    validate_event_table(detected_df)
    validate_event_table(ground_truth_df)

    det_start = detected_df["start_blink"].to_numpy(dtype=int)
    det_end = detected_df["end_blink"].to_numpy(dtype=int)
    gt_start = ground_truth_df["start_blink"].to_numpy(dtype=int)
    gt_end = ground_truth_df["end_blink"].to_numpy(dtype=int)

    alignments: list[Alignment] = []
    det_idx = gt_idx = 0

    while gt_idx < len(gt_start) and det_idx < len(det_start):
        start_delta = int(det_start[det_idx]) - int(gt_start[gt_idx])
        end_delta = int(det_end[det_idx]) - int(gt_end[gt_idx])

        if abs(start_delta) <= tolerance_samples and abs(end_delta) <= tolerance_samples:
            alignments.append(
                Alignment(
                    ground_truth_idx=gt_idx,
                    detected_idx=det_idx,
                    start_diff=float(start_delta),
                    end_diff=float(end_delta),
                )
            )
            gt_idx += 1
            det_idx += 1
            continue

        if gt_start[gt_idx] < det_start[det_idx]:
            alignments.append(
                Alignment(
                    ground_truth_idx=gt_idx,
                    detected_idx=None,
                    start_diff=None,
                    end_diff=None,
                )
            )
            gt_idx += 1
        elif det_start[det_idx] < gt_start[gt_idx]:
            alignments.append(
                Alignment(
                    ground_truth_idx=None,
                    detected_idx=det_idx,
                    start_diff=None,
                    end_diff=None,
                )
            )
            det_idx += 1
        else:
            alignments.append(
                Alignment(
                    ground_truth_idx=gt_idx,
                    detected_idx=det_idx,
                    start_diff=float(start_delta),
                    end_diff=float(end_delta),
                )
            )
            gt_idx += 1
            det_idx += 1

    while gt_idx < len(gt_start):
        alignments.append(
            Alignment(
                ground_truth_idx=gt_idx,
                detected_idx=None,
                start_diff=None,
                end_diff=None,
            )
        )
        gt_idx += 1

    while det_idx < len(det_start):
        alignments.append(
            Alignment(
                ground_truth_idx=None,
                detected_idx=det_idx,
                start_diff=None,
                end_diff=None,
            )
        )
        det_idx += 1

    return alignments


def compute_alignment_metrics(
    alignments: list[Alignment], tolerance_samples: int
) -> dict[str, float]:
    """Compute summary metrics describing alignment quality.

    Metrics are based on sample index comparisons (1-based) and include:

    ``total_ground_truth``
        Number of ground truth events.
    ``total_detected``
        Number of detected events.
    ``paired_events``
        Count of alignments containing both detected and ground truth events.
    ``matches_within_tolerance``
        Number of paired events whose start and end differences are within tolerance.
    ``pairs_outside_tolerance``
        Number of paired events outside tolerance.
    ``ground_truth_only``
        Ground truth events without a detected counterpart.
    ``detected_only``
        Detected events without a ground truth counterpart.
    ``share_within_tolerance``
        Percentage of all unique events that fall within tolerance.
    """

    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative.")

    total_ground_truth = sum(a.ground_truth_idx is not None for a in alignments)
    total_detected = sum(a.detected_idx is not None for a in alignments)
    paired_events = [a for a in alignments if a.ground_truth_idx is not None and a.detected_idx is not None]
    matches_within_tolerance = sum(a.is_match(tolerance_samples) for a in paired_events)
    pairs_outside_tolerance = len(paired_events) - matches_within_tolerance
    ground_truth_only = sum(a.ground_truth_idx is not None and a.detected_idx is None for a in alignments)
    detected_only = sum(a.detected_idx is not None and a.ground_truth_idx is None for a in alignments)
    unique_total = matches_within_tolerance + pairs_outside_tolerance + ground_truth_only + detected_only

    def _pct(n: int, d: int) -> float:
        return (n / d) * 100.0 if d else float("nan")

    return {
        "total_ground_truth": float(total_ground_truth),
        "total_detected": float(total_detected),
        "paired_events": float(len(paired_events)),
        "matches_within_tolerance": float(matches_within_tolerance),
        "pairs_outside_tolerance": float(pairs_outside_tolerance),
        "ground_truth_only": float(ground_truth_only),
        "detected_only": float(detected_only),
        "share_within_tolerance": _pct(matches_within_tolerance, unique_total),
        "unique_total": float(unique_total),
    }


def compute_pairwise_differences(
    detected_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute start/end sample differences for overlapping events.

    Parameters are validated using :func:`validate_event_table`. Differences are
    computed only for overlapping index positions (``min(len(detected), len(ground_truth))``),
    always in sample index units (1-based).
    """

    validate_event_table(detected_df)
    validate_event_table(ground_truth_df)

    n = min(len(detected_df), len(ground_truth_df))
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    det_start = detected_df["start_blink"].to_numpy(dtype=int)
    det_end = detected_df["end_blink"].to_numpy(dtype=int)
    gt_start = ground_truth_df["start_blink"].to_numpy(dtype=int)
    gt_end = ground_truth_df["end_blink"].to_numpy(dtype=int)

    return np.abs(det_start[:n] - gt_start[:n]), np.abs(det_end[:n] - gt_end[:n])
