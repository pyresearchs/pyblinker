"""Simple blink count summarization used for test comparison."""
from typing import Tuple, Optional

import mne
import pandas as pd

from pyblinker.utils.epochs import slice_raw_into_epochs


def summarize_blink_counts(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, list[mne.io.BaseRaw]]:
    """Return blink counts per segment using pyear utilities."""
    segments, df, _, _ = slice_raw_into_epochs(
        raw, epoch_len=epoch_len, blink_label=blink_label
    )
    return df, segments
