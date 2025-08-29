"""Epoching utilities for time-series data."""

from typing import List, Tuple, Optional, Dict, Any


import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


def refine_ear_extrema_and_threshold_stub(
        signal_segment: np.ndarray,
        start_rel: int,
        end_rel: int,
        peak_rel_cvat: int | None = None,
) -> Tuple[int, int, int]:
    """Return a crude EAR trough refinement.

    Parameters mirror those of the real refinement routine but the
    implementation merely validates that indices are within bounds and
    estimates a trough location if ``peak_rel_cvat`` is not supplied.

    - outer_start (float): The lower bound index of the left-side search region.
    - outer_end (float): The upper bound index of the right-side search region
    - extremum_point: The frame index of the trough (or minimum) to evaluate crossings around.
    Returns
    -------
    tuple
        ``(start_frame, trough_frame, end_frame)`` indices within the segment.
    """

    extremum_point = peak_rel_cvat
    if not (peak_rel_cvat is not None and 0 <= peak_rel_cvat < len(signal_segment)):
        if end_rel >= start_rel and len(signal_segment) > 0:
            extremum_point = (start_rel + end_rel) // 2
        else:
            extremum_point = 0

    outer_start = max(0, min(start_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    outer_end = max(0, min(end_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    if outer_start > outer_end:
        outer_start = outer_end

    return outer_start, extremum_point, outer_end




def refine_local_maximum_stub(
        signal_segment: np.ndarray,
        start_rel: int,
        end_rel: int,
        peak_rel_cvat: int | None = None,
) -> Tuple[int, int, int]:
    """Return a crude refinement for local maxima in a signal segment.
        This is only suitable for EEG and EOG signals where blinks are detected.
        Eye Aspect Ratio (EAR) is not applicable here, unless it is being inverted.
    Finds start, peak, and end frame indices within the signal segment.
    The peak is the index of the maximum value between start_rel and end_rel.

    Parameters:
    - signal_segment: The signal segment array.
    - start_rel: Relative start index.
    - end_rel: Relative end index.
    - peak_rel_cvat: Optional initial guess for the peak index. If not within bounds, it is ignored.
    - outer_start (float): The lower bound index of the left-side search region.
    - outer_end (float): The upper bound index of the right-side search region
    - extremum_point: The frame index of the peak (or maximum) to evaluate crossings around.

    Returns:
    - Tuple of (start_frame, peak_frame, end_frame), all relative to the segment.
    """
    n = len(signal_segment)
    if n == 0:
        return 0, 0, 0

    outer_start = max(0, min(start_rel, n - 1))
    outer_end = max(0, min(end_rel, n - 1))
    if outer_start > outer_end:
        outer_start = outer_end = min(outer_start, outer_end)

    if peak_rel_cvat is not None and outer_start <= peak_rel_cvat <= outer_end:
        extremum_point = peak_rel_cvat
    else:
        segment = signal_segment[outer_start : outer_end + 1]
        max_idx_local = int(np.argmax(segment))
        extremum_point = outer_start + max_idx_local

    return outer_start, extremum_point, outer_end

# --------------------------- helpers ---------------------------

def _pick_ear_channels(raw: mne.io.BaseRaw) -> List[int]:
    """Heuristic pick for EAR channels (name contains 'ear' or 'eye_aspect_ratio')."""
    picks = []
    for idx, name in enumerate(raw.ch_names):
        nlow = name.lower()
        if ("ear" in nlow) or ("eye_aspect_ratio" in nlow):
            picks.append(idx)
    return picks

def _init_metadata(
        n_epochs: int,
        have_eeg: bool,
        have_eog: bool,
        have_ear: bool,
) -> Dict[str, List[Any]]:
    """Create metadata dict with required (manual) and conditional (refined) fields."""
    md: Dict[str, List[Any]] = {
        "blink_onset": [np.nan] * n_epochs,          # manual (seconds; list or float)
        "blink_duration": [np.nan] * n_epochs,       # manual (seconds; list or float)
        "n_blinks": [0] * n_epochs,
    }
    if have_eeg:
        md["blink_onset_eeg"] = [np.nan] * n_epochs
        md["blink_duration_eeg"] = [np.nan] * n_epochs
        md["blink_onset_extremum_eeg"] = [np.nan] * n_epochs
        md["blink_outer_start_eeg"] = [np.nan] * n_epochs
        md["blink_outer_end_eeg"] = [np.nan] * n_epochs
    if have_eog:
        md["blink_onset_eog"] = [np.nan] * n_epochs
        md["blink_duration_eog"] = [np.nan] * n_epochs
        md["blink_onset_extremum_eog"] = [np.nan] * n_epochs
        md["blink_outer_start_eog"] = [np.nan] * n_epochs
        md["blink_outer_end_eog"] = [np.nan] * n_epochs
    if have_ear:
        md["blink_onset_ear"] = [np.nan] * n_epochs
        md["blink_duration_ear"] = [np.nan] * n_epochs
        md["blink_onset_extremum_ear"] = [np.nan] * n_epochs
        md["blink_outer_start_ear"] = [np.nan] * n_epochs
        md["blink_outer_end_ear"] = [np.nan] * n_epochs
    return md


def _ensure_list_append(slot, value):
    """Append value to a metadata cell, converting NaN/float to list as needed."""
    if isinstance(slot, list):
        slot.append(value)
        return slot
    if pd.isna(slot):
        return [value]
    # already a scalar -> convert to list
    return [slot, value]

def slice_raw_into_mne_epochs_refine_annot(
        raw: mne.io.BaseRaw,
        *,
        epoch_len: float = 30.0,
        blink_label: Optional[str] = "blink",
        progress_bar: bool = True,
) -> mne.Epochs:
    """Convert a continuous recording into equally spaced MNE epochs and refine
    per-epoch blink timing in the metadata.

    This utility targets pipelines where multiple ocular blink modalities
    (EEG, EOG, and eye-aspect-ratio/EAR) may already be combined into a single
    :class:`mne.io.Raw` object and blinks have been coarsely marked via manual
    annotations (e.g., from a GUI). Downstream blink features—such as
    morphology, energy, or zero-crossing–based metrics—often require a precise
    start and end for each blink, which can vary by channel and modality. To
    support that, the function:

    1) splits the continuous recording into fixed-length epochs, and
    2) attaches blink metadata to each epoch, refining blink onset/offset per
       channel according to modality-specific rules when possible.

    Epoching is also helpful when portions of the recording are corrupted
    (e.g., movement, saturation): operating on epoch-level metadata allows you
    to filter, reject, or stratify analyses without mutating the underlying
    raw data.

    Metadata fields (always present / modality-specific)
    ---------------------------------------------------
    The returned ``epochs.metadata`` is a ``pandas.DataFrame`` that always
    includes generic manual-annotation fields and — when available — modality-
    specific refined fields.

    Always-present (manual-annotation defaults)
    - ``blink_onset`` — float, seconds from **epoch start** to the manual
      annotation onset. Present for epochs containing a manual blink
      annotation; ``NaN`` otherwise.
    - ``blink_duration`` — float, seconds of the manual annotation duration.
      Present when a manual annotation exists; ``NaN`` otherwise.

    Modality-specific refined fields (added when corresponding channels are present)
    - ``blink_onset_ear`` — refined blink onset (s from **epoch start**) for
      EAR-derived signals (if an EAR refinement is implemented). ``NaN`` when
      absent.
    - ``blink_duration_ear`` — refined blink duration (seconds) for EAR.
    - ``blink_onset_extremum_ear`` — time (s from **epoch start**) of the EAR
      extremum associated with the blink. For EAR, the extremum is defined as
      the trough (minimum) within the blink interval. ``NaN`` when absent.

    - ``blink_onset_eeg`` — refined blink onset (s from **epoch start**)
      computed from EEG channel(s) using the EEG refinement method (e.g.,
      "blinker"-style rules). ``NaN`` when absent.
    - ``blink_duration_eeg`` — refined blink duration (seconds) for EEG.
    - ``blink_onset_extremum_eeg`` — time (s from **epoch start**) of the EEG
      extremum associated with the blink. For EEG, the extremum is defined as
      the peak (maximum) within the blink interval. ``NaN`` when absent.

    - ``blink_onset_eog`` — refined blink onset (s from **epoch start**)
      computed from EOG channel(s) using the EOG refinement method. ``NaN``
      when absent.
    - ``blink_duration_eog`` — refined blink duration (seconds) for EOG.
    - ``blink_onset_extremum_eog`` — time (s from **epoch start**) of the EOG
      extremum associated with the blink. For EOG, the extremum is defined as
      the peak (maximum) within the blink interval. ``NaN`` when absent.

    Behavior & precedence
    ---------------------
    - By default, ``blink_onset`` / ``blink_duration`` preserve the manual
      annotation timings, so downstream code always has a simple, consistent
      pair to use.
    - When modality-specific refinements are computed, those refined values are
      exposed in the corresponding ``*_ear``, ``*_eeg``, ``*_eog`` columns.
      Extremum times for each modality are exposed as
      ``blink_onset_extremum_{modality}`` and denote the trough (EAR) or peak
      (EEG/EOG) within the refined blink interval.
    - If no modality-specific channels are present (or no refinement is
      performed), the generic ``blink_onset`` and ``blink_duration`` remain the
      only blink timing fields (sourced from the manual annotations).
    - Implementations may optionally provide parameters to (a) overwrite the
      generic ``blink_onset``/``blink_duration`` with a chosen refined value,
      or (b) include associated extremum *values* (signal amplitude at the
      extremum). By default this function does not mutate the original raw
      annotations — it surfaces decisions in epoch metadata.


    Notes & limitations
    -------------------
    - Manual annotations define **candidate** blink intervals; EEG/EOG
      refinements adjust boundaries per channel, while EAR boundaries are kept
      as annotated by default unless an EAR-specific refinement is implemented.
    - Refined times and extremum times can legitimately differ across channels
      and modalities within the same epoch (zero-crossing, baseline return,
      and peak/trough definitions vary by signal).
    - Bad segments (e.g., ``BAD_*`` annotations) are not modified by this
      function; use epoch metadata to exclude or weight epochs downstream.
    - The function does not alter the underlying raw data; all decisions are
      surfaced via epoch metadata.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous raw recording that may contain multiple modalities (EEG,
        EOG, EAR) and manual blink annotations.
    epoch_len : float, optional
        Length of each epoch in seconds. Defaults to ``30.0``.
    blink_label : str | None, optional
        Annotation label that denotes blinks. Use ``None`` to treat all
        annotations as candidate blinks. Defaults to ``"blink"``.
    progress_bar : bool, optional
        Whether to display a progress bar while mapping annotations into epoch
        metadata and performing refinements. Defaults to ``True``.

    Returns
    -------
    mne.Epochs
        Equally spaced epochs with a ``pandas.DataFrame`` attached as
        ``epochs.metadata``. Metadata always contains ``blink_onset`` and
        ``blink_duration`` (manual-annotation defaults) and may include the
        modality-specific refined fields and extremum timing fields described
        above.

    """

    from pyblinker.blink_features.blink_events.blink_dataframe import (
        compute_outer_bounds,
    )

    # --- epoching ---
    events = mne.make_fixed_length_events(raw, duration=epoch_len)
    sfreq = float(raw.info["sfreq"])
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=epoch_len - 1.0 / sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # --- picks / modality availability ---
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, misc=False)
    picks_eog = mne.pick_types(raw.info, eeg=False, eog=True, misc=False)
    picks_ear = _pick_ear_channels(raw)

    have_eeg = len(picks_eeg) > 0
    have_eog = len(picks_eog) > 0
    have_ear = len(picks_ear) > 0

    # --- prefetch data for speed (epochs x channels x samples) ---
    data_eeg = epochs.get_data(picks=picks_eeg) if have_eeg else None
    data_eog = epochs.get_data(picks=picks_eog) if have_eog else None
    data_ear = epochs.get_data(picks=picks_ear) if have_ear else None

    n_epochs = len(epochs)
    n_samp_epoch = epochs.get_data(picks=[0]).shape[-1] if epochs.info["nchan"] > 0 else int(round(epoch_len * sfreq))

    # --- annotation selection ---
    ann = raw.annotations
    if blink_label is None:
        sel = np.ones(len(ann), dtype=bool)
    else:
        # case-insensitive exact match to be conservative
        sel = np.array([(str(d).lower() == blink_label.lower()) for d in ann.description], dtype=bool)

    blink_onsets_sec = np.array(ann.onset)[sel]
    blink_durs_sec = np.array(ann.duration)[sel]

    # --- metadata scaffold ---
    md = _init_metadata(n_epochs, have_eeg, have_eog, have_ear)

    # Iterate epochs, map blinks, and refine
    iterator = range(n_epochs)
    if progress_bar:
        iterator = tqdm(iterator, desc="Refining blink metadata", unit="epoch")

    for ei in iterator:
        epoch_start_samp = int(epochs.events[ei, 0])                  # sample index in raw
        epoch_start_sec = epoch_start_samp / sfreq
        epoch_end_sec = epoch_start_sec + epoch_len

        # gather blinks overlapping this epoch
        blink_starts = []
        blink_ends = []
        for onset_sec, dur_sec in zip(blink_onsets_sec, blink_durs_sec):
            ann_start = float(onset_sec)
            ann_end = float(onset_sec + max(dur_sec, 0.0))
            # overlap?
            if max(ann_start, epoch_start_sec) < min(ann_end, epoch_end_sec):
                # clamp to epoch and convert to RELATIVE samples (inclusive end)
                start_rel = int(np.clip(round((ann_start - epoch_start_sec) * sfreq), 0, n_samp_epoch - 1))
                end_rel = int(np.clip(round((ann_end   - epoch_start_sec) * sfreq) - 1, 0, n_samp_epoch - 1))
                if end_rel < start_rel:
                    end_rel = start_rel
                blink_starts.append(start_rel)
                blink_ends.append(end_rel)

        n_blinks = len(blink_starts)
        md["n_blinks"][ei] = n_blinks
        if n_blinks == 0:
            continue  # leave NaNs

        # --- manual (always-present) fields ---
        for sr, er in zip(blink_starts, blink_ends):
            onset_sec_rel = sr / sfreq
            duration_sec_rel = max(0.0, (er - sr) / sfreq)
            md["blink_onset"][ei] = _ensure_list_append(md["blink_onset"][ei], onset_sec_rel)
            md["blink_duration"][ei] = _ensure_list_append(md["blink_duration"][ei], duration_sec_rel)

        # --- EAR refinement (trough) ---
        if have_ear:
            seg = data_ear[ei].mean(axis=0)  # (n_samples,)
            peaks: List[int] = []
            for sr, er in zip(blink_starts, blink_ends):
                rs, trough, re = refine_ear_extrema_and_threshold_stub(seg, sr, er, peak_rel_cvat=None)
                peaks.append(int(trough))
                md["blink_onset_ear"][ei] = _ensure_list_append(md["blink_onset_ear"][ei], rs / sfreq)
                md["blink_duration_ear"][ei] = _ensure_list_append(md["blink_duration_ear"][ei], max(0.0, (re - rs) / sfreq))
                md["blink_onset_extremum_ear"][ei] = _ensure_list_append(md["blink_onset_extremum_ear"][ei], trough / sfreq)
            if peaks:
                bounds = compute_outer_bounds(peaks, n_samp_epoch)
                for outer_start, outer_end in bounds:
                    md["blink_outer_start_ear"][ei] = _ensure_list_append(
                        md["blink_outer_start_ear"][ei], outer_start
                    )
                    md["blink_outer_end_ear"][ei] = _ensure_list_append(
                        md["blink_outer_end_ear"][ei], outer_end
                    )

        # --- EEG refinement (peak) ---
        if have_eeg:
            seg = data_eeg[ei].mean(axis=0)  # simple robust default; swap to frontal pick if desired
            peaks: List[int] = []
            for sr, er in zip(blink_starts, blink_ends):
                rs, peak, re = refine_local_maximum_stub(seg, sr, er, peak_rel_cvat=None)
                peaks.append(int(peak))
                md["blink_onset_eeg"][ei] = _ensure_list_append(md["blink_onset_eeg"][ei], rs / sfreq)
                md["blink_duration_eeg"][ei] = _ensure_list_append(md["blink_duration_eeg"][ei], max(0.0, (re - rs) / sfreq))
                md["blink_onset_extremum_eeg"][ei] = _ensure_list_append(md["blink_onset_extremum_eeg"][ei], peak / sfreq)
            if peaks:
                bounds = compute_outer_bounds(peaks, n_samp_epoch)
                for outer_start, outer_end in bounds:
                    md["blink_outer_start_eeg"][ei] = _ensure_list_append(
                        md["blink_outer_start_eeg"][ei], outer_start
                    )
                    md["blink_outer_end_eeg"][ei] = _ensure_list_append(
                        md["blink_outer_end_eeg"][ei], outer_end
                    )

        # --- EOG refinement (peak) ---
        if have_eog:
            seg = data_eog[ei].mean(axis=0)
            peaks: List[int] = []
            for sr, er in zip(blink_starts, blink_ends):
                rs, peak, re = refine_local_maximum_stub(seg, sr, er, peak_rel_cvat=None)
                peaks.append(int(peak))
                md["blink_onset_eog"][ei] = _ensure_list_append(md["blink_onset_eog"][ei], rs / sfreq)
                md["blink_duration_eog"][ei] = _ensure_list_append(md["blink_duration_eog"][ei], max(0.0, (re - rs) / sfreq))
                md["blink_onset_extremum_eog"][ei] = _ensure_list_append(md["blink_onset_extremum_eog"][ei], peak / sfreq)
            if peaks:
                bounds = compute_outer_bounds(peaks, n_samp_epoch)
                for outer_start, outer_end in bounds:
                    md["blink_outer_start_eog"][ei] = _ensure_list_append(
                        md["blink_outer_start_eog"][ei], outer_start
                    )
                    md["blink_outer_end_eog"][ei] = _ensure_list_append(
                        md["blink_outer_end_eog"][ei], outer_end
                    )

    # Convert dict to DataFrame
    metadata = pd.DataFrame(md)
    epochs.metadata = metadata

    logger.debug("Epoch metadata head: %s", metadata.head())
    logger.info("Exiting slice_raw_into_mne_epochs_refine_annot")
    return epochs
