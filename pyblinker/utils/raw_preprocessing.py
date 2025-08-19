"""Utilities for preparing MNE Raw data with refined blink annotations."""
import logging
from pathlib import Path
from typing import Sequence, Union, List, Dict, Any

import mne
from mne.io import BaseRaw
from tqdm import tqdm

from .epochs import slice_raw_into_epochs, EPOCH_LEN
from .refinement import refine_blinks_from_epochs

logger = logging.getLogger(__name__)



def _update_segment_annotations(
    segments: Sequence[mne.io.BaseRaw],
    refined: Sequence[Dict[str, int]],
    *,
    progress_bar: bool = True,
) -> None:
    """Update annotations on each segment with refined blink timings."""
    logger.info("Updating annotations for %d segments", len(segments))
    idx = 0
    for seg_idx, seg in enumerate(
        tqdm(segments, desc="Segments", disable=not progress_bar)
    ):
        sfreq = seg.info["sfreq"]
        orig_anns = seg.annotations
        n_anns = len(orig_anns)
        new_onsets: List[float] = []
        new_durations: List[float] = []
        new_descriptions: List[str] = []
        for ann_i in tqdm(
            range(n_anns), desc=f"Seg {seg_idx} annotations", leave=False, disable=not progress_bar
        ):
            blink_info = refined[idx]
            start_frame = blink_info["refined_start_frame"]
            end_frame = blink_info["refined_end_frame"]
            onset = start_frame / sfreq
            duration = (end_frame - start_frame) / sfreq
            desc = orig_anns.description[ann_i]
            new_onsets.append(onset)
            new_durations.append(duration)
            new_descriptions.append(desc)
            idx += 1
        seg.set_annotations(
            mne.Annotations(
                onset=new_onsets,
                duration=new_durations,
                description=new_descriptions,
            )
        )


def prepare_refined_segments(
    raw: Union[str, Path, mne.io.BaseRaw],
    channel: str,
    *,
    epoch_len: float = EPOCH_LEN,
    keep_epoch_signal: bool = False,
    progress_bar: bool = True,
) -> tuple[list[BaseRaw], list[dict[str, Any]]]:
    """Load and prepare raw segments with refined blink annotations.

    This routine is intended for ``mne.io.Raw`` recordings that already contain
    blink annotations. It performs the standard preprocessing steps required
    prior to feature extraction:

    1. Load the raw file if a path is provided.
    2. Slice the continuous recording into 30-second segments.
    3. Refine each blink's start, peak and end frames using
       :func:`refine_blinks_from_epochs`.
    4. Replace the annotations of each segment with the refined timings.

    Parameters
    ----------
    raw : str | pathlib.Path | mne.io.BaseRaw
        File path or Raw object with blink annotations.
    channel : str
        Channel name used for refinement.
    epoch_len : float, optional
        Length of each segment in seconds. Defaults to ``30``.
    keep_epoch_signal : bool, optional
        If ``True``, keep the ``epoch_signal`` field in the returned refined
        blink dictionaries. This can be useful for manual inspection.

    Returns
    -------
    list of mne.io.BaseRaw
        Segmented raws with updated annotations.
    list of dict
        Refined blink information per annotation.

    Raises
    ------
    ValueError
        If the input Raw contains no annotations.
    """
    logger.info("Preparing raw segments for blink features")
    if isinstance(raw, (str, Path)):
        raw = mne.io.read_raw_fif(raw, preload=False, verbose=False)
    if len(raw.annotations) == 0:
        raise ValueError("Raw recording has no annotations to refine")

    segments, _, _, _ = slice_raw_into_epochs(
        raw, epoch_len=epoch_len, progress_bar=progress_bar
    )
    refined = refine_blinks_from_epochs(segments, channel)

    # segments[1].plot(block=True)
    if not keep_epoch_signal:
        for blink in refined:
            blink.pop("epoch_signal", None)

    _update_segment_annotations(segments, refined, progress_bar=progress_bar)
    # refined = group_refined_by_epoch(refined)
    logger.info("Finished preparing %d segments", len(segments))
    return list(segments), refined
