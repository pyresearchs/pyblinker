"""Median absolute deviation of open-eye baseline."""
from typing import List, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)


def baseline_mad_epoch(epoch_signal: np.ndarray, blinks: List[Dict[str, int]]) -> float:
    """Compute baseline median absolute deviation for an epoch."""
    mask = np.ones(len(epoch_signal), dtype=bool)
    for blink in blinks:
        mask[int(blink["refined_start_frame"]): int(blink["refined_end_frame"])+1] = False
    open_signal = epoch_signal[mask]
    if open_signal.size == 0:
        return float("nan")
    median = np.median(open_signal)
    mad = float(np.median(np.abs(open_signal - median)))
    logger.debug("Baseline MAD: %s", mad)
    return mad
