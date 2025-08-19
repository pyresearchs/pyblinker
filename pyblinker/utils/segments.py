"""Raw segmentation helpers."""
from typing import List
import logging

import mne
from tqdm import tqdm

logger = logging.getLogger(__name__)


def slice_raw_to_segments(
    raw: mne.io.BaseRaw, epoch_len: float = 30.0, *, progress_bar: bool = True
) -> List[mne.io.BaseRaw]:
    """Slice a continuous :class:`mne.io.BaseRaw` into fixed-length segments.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous raw recording with blink annotations.
    epoch_len : float, optional
        Length of each segment in seconds, by default ``30.0``.

    Returns
    -------
    list of mne.io.BaseRaw
        List of cropped raw segments containing annotations.
    """
    n_segments = int(raw.times[-1] // epoch_len)
    segments: List[mne.io.BaseRaw] = []
    for i in tqdm(
        range(n_segments), desc="Segmenting", unit="segment", disable=not progress_bar
    ):
        start = i * epoch_len
        stop = start + epoch_len
        seg = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
        segments.append(seg)
    logger.info("Created %d segments", n_segments)
    return segments
