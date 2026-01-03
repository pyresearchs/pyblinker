"""Blink refinement utilities shared across pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker.logging import get_logger
from pyblinker.utils.dict_utils import append_to_slot

from .ear import (
    _append_ear_refinements,
    _append_outer_bounds_from_peaks,
    _refine_ear_blinks_for_epoch,
)

logger = get_logger(__name__)


def _init_metadata(
    n_epochs: int,
    have_eeg: bool,
    have_eog: bool,
    have_ear: bool,
) -> Dict[str, List[Any]]:
    """Create metadata dict with required (manual) and conditional fields."""

    md: Dict[str, List[Any]] = {
        "blink_onset": [np.nan] * n_epochs,
        "blink_duration": [np.nan] * n_epochs,
        "n_blinks": [0] * n_epochs,
    }
    # Since we may have different ways to define the blink onset per modality, and per crossing like zero crossing, tent crossing, half base crossing.
    # So here, we just create all the possible fields.
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
        md["refined_start_sample"] = [np.nan] * n_epochs
        md["refined_end_sample"] = [np.nan] * n_epochs
        md["refined_lowest_point_sample"] = [np.nan] * n_epochs
        md["refined_left_threshold"] = [np.nan] * n_epochs
        md["refined_right_threshold"] = [np.nan] * n_epochs
        md["search_window_start_sample"] = [np.nan] * n_epochs
        md["search_window_end_sample"] = [np.nan] * n_epochs
        md["search_window_start_time"] = [np.nan] * n_epochs
        md["search_window_end_time"] = [np.nan] * n_epochs
        md["refinement_succeeded"] = [None] * n_epochs
        md["search_exhausted"] = [None] * n_epochs
        md["extension_seconds_used"] = [np.nan] * n_epochs
        md["extension_attempts"] = [np.nan] * n_epochs
        md["left_interpolated_threshold"] = [np.nan] * n_epochs
        md["right_interpolated_threshold"] = [np.nan] * n_epochs
        md["left_interpolated_threshold_sample"] = [np.nan] * n_epochs
        md["right_interpolated_threshold_sample"] = [np.nan] * n_epochs
        md["left_interpolated_threshold_found"] = [None] * n_epochs
        md["right_interpolated_threshold_found"] = [None] * n_epochs
        md["interpolated_thresholds_found"] = [None] * n_epochs
    return md


def _prepare_segmentation_config(
    segmentation_type: Optional[dict],
    ear_threshold: Optional[float],
) -> Dict[str, Any]:
    """Merge segmentation settings with convenience parameters."""

    config: Dict[str, Any] = dict(segmentation_type or {})
    ear_config = dict(config.get("ear", {}))
    if ear_threshold is not None:
        ear_config.setdefault("threshold", ear_threshold)
        ear_config.setdefault("seg_type", "threshold_interpolation")
        config["ear"] = ear_config
    return config


def _compute_epoch_blink_bounds(
    blink_onsets_sec: np.ndarray,
    blink_durs_sec: np.ndarray,
    epoch_start_sec: float,
    epoch_end_sec: float,
    sfreq: float,
    n_samp_epoch: int,
) -> Tuple[List[int], List[int]]:
    """Return coarse blink start and end samples for a given epoch."""

    blink_starts: List[int] = []
    blink_ends: List[int] = []
    for onset_sec, dur_sec in zip(blink_onsets_sec, blink_durs_sec):
        ann_start = float(onset_sec)
        ann_end = float(onset_sec + max(dur_sec, 0.0))
        if max(ann_start, epoch_start_sec) < min(ann_end, epoch_end_sec):
            start_rel = int(
                np.clip(round((ann_start - epoch_start_sec) * sfreq), 0, n_samp_epoch - 1)
            )
            end_rel = int(
                np.clip(
                    round((ann_end - epoch_start_sec) * sfreq) - 1,
                    0,
                    n_samp_epoch - 1,
                )
            )
            if end_rel < start_rel:
                end_rel = start_rel
            blink_starts.append(start_rel)
            blink_ends.append(end_rel)

    return blink_starts, blink_ends


def _append_peak_refinements(
    md: Dict[str, List[Any]],
    epoch_index: int,
    segment: np.ndarray,
    blink_starts: Sequence[int],
    blink_ends: Sequence[int],
    sfreq: float,
    key_prefix: str,
    refine_func: Callable[[np.ndarray, int, int, int | None], Tuple[int, int, int]],
    n_samp_epoch: int,
) -> None:
    if segment.size == 0 or not blink_starts:
        return
    peaks: List[int] = []
    for start, end in zip(blink_starts, blink_ends):
        refined_start, peak, refined_end = refine_func(segment, start, end, peak_rel_cvat=None)
        peaks.append(int(peak))
        md[f"blink_onset_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_onset_{key_prefix}"][epoch_index], refined_start / sfreq
        )
        md[f"blink_duration_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_duration_{key_prefix}"][epoch_index],
            max(0.0, (refined_end - refined_start) / sfreq),
        )
        md[f"blink_onset_extremum_{key_prefix}"][epoch_index] = append_to_slot(
            md[f"blink_onset_extremum_{key_prefix}"][epoch_index], peak / sfreq
        )

    _append_outer_bounds_from_peaks(md, epoch_index, peaks, key_prefix, n_samp_epoch)


@dataclass
class EpochPreparationResult:
    """Container for epoch-level data and modality picks."""

    epochs: mne.Epochs
    sfreq: float
    n_epochs: int
    n_samp_epoch: int
    data_ear: np.ndarray | None
    data_eeg: np.ndarray | None
    data_eog: np.ndarray | None
    have_ear: bool
    have_eeg: bool
    have_eog: bool
    blink_onsets_sec: np.ndarray
    blink_durs_sec: np.ndarray


def _resolve_single_channel_pick(
    raw: mne.io.BaseRaw,
    config: Dict[str, Any],
    modality: str,
    *,
    required: bool,
) -> int | None:
    """Validate and return the index for a single modality channel.

    Args:
        raw: Raw recording containing channel names.
        config: Segmentation configuration for the modality.
        modality: Modality key such as ``"ear"``, ``"eeg"``, or ``"eog"``.
        required: Whether the modality must be present.

    Returns:
        The integer channel index if enabled, otherwise ``None``.

    Raises:
        ValueError: If a required channel is missing, not found in ``raw``,
            or resolves to multiple indices.
    """

    channel_value = (config or {}).get("channel")
    if channel_value is None or (isinstance(channel_value, str) and not channel_value.strip()):
        if required:
            raise ValueError(f"{modality.upper()} refinement requires a single channel set via segmentation config.")
        return None

    if isinstance(channel_value, Sequence) and not isinstance(channel_value, str):
        if len(channel_value) != 1:
            raise ValueError(
                f"{modality.upper()} refinement expects exactly one channel, but got {len(channel_value)} entries."
            )
        channel_name = str(channel_value[0])
    else:
        channel_name = str(channel_value)

    picks = [idx for idx, name in enumerate(raw.ch_names) if name == channel_name]
    if not picks:
        raise ValueError(f"Configured {modality.upper()} channel '{channel_name}' not found in raw data.")
    if len(picks) > 1:
        raise ValueError(
            f"{modality.upper()} refinement expects a single channel, but multiple matches were found for "
            f"'{channel_name}'."
        )
    return picks[0]


def _prepare_epochs_and_modalities(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float,
    blink_label: str | None,
    segment_config: Dict[str, Any],
) -> EpochPreparationResult:
    """Create epochs, resolve modality channels, and filter annotations.

    Workflow
    --------
    1. Fixed-length events are created with :func:`mne.make_fixed_length_events`.
    2. Epochs are instantiated over the full recording with ``tmin=0`` and
       ``tmax=epoch_len - 1/sfreq`` to preserve the previous inclusive/exclusive
       boundaries.
    3. Each modality resolves an explicit, single channel from ``segment_config``:
       - EAR must provide exactly one ``"channel"`` entry. Missing, empty, or
         multi-channel values raise ``ValueError``.
       - EEG and EOG are optional. If ``"channel"`` is missing/None/empty, the
         modality is disabled (``have_eeg/eog=False``) and the returned data is
         ``None``. If provided but not found or not singular, ``ValueError`` is
         raised.
    4. Epoch data are extracted with ``epochs.get_data(picks=[idx])`` so each
       enabled modality yields an array shaped ``(n_epochs, 1, n_times)``. The
       function retains ``None`` for disabled modalities.
    5. Blink annotations are filtered by ``blink_label``. ``None`` keeps all
       annotations; otherwise, descriptions are matched case-insensitively.

    Returns
    -------
    EpochPreparationResult
        Contains epochs, sampling frequency, derived counts, per-modality data,
        modality enable flags, and blink onset/duration arrays (seconds).

    Notes
    -----
    * ``blink_onsets_sec`` and ``blink_durs_sec`` are in **seconds** relative to
      the raw object. ``n_samp_epoch`` refers to samples within each epoch.
    * Errors intentionally fail fast for misconfigured channels to avoid
      silently averaging or inferring modalities.
    """

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

    ear_idx = _resolve_single_channel_pick(raw, segment_config.get("ear", {}), "ear", required=True)
    eeg_idx = _resolve_single_channel_pick(raw, segment_config.get("eeg", {}), "eeg", required=False)
    eog_idx = _resolve_single_channel_pick(raw, segment_config.get("eog", {}), "eog", required=False)

    have_ear = ear_idx is not None
    have_eeg = eeg_idx is not None
    have_eog = eog_idx is not None

    data_ear = epochs.get_data(picks=[ear_idx]) if have_ear else None
    data_eeg = epochs.get_data(picks=[eeg_idx]) if have_eeg else None
    data_eog = epochs.get_data(picks=[eog_idx]) if have_eog else None

    n_epochs = len(epochs)
    n_samp_epoch = (
        epochs.get_data(picks=[0]).shape[-1]
        if epochs.info["nchan"] > 0
        else int(round(epoch_len * sfreq))
    )

    ann = raw.annotations
    if blink_label is None:
        sel = np.ones(len(ann), dtype=bool)
    else:
        sel = np.array(
            [(str(d).lower() == blink_label.lower()) for d in ann.description],
            dtype=bool,
        )

    blink_onsets_sec = np.array(ann.onset)[sel]
    blink_durs_sec = np.array(ann.duration)[sel]

    return EpochPreparationResult(
        epochs=epochs,
        sfreq=sfreq,
        n_epochs=n_epochs,
        n_samp_epoch=n_samp_epoch,
        data_ear=data_ear,
        data_eeg=data_eeg,
        data_eog=data_eog,
        have_ear=have_ear,
        have_eeg=have_eeg,
        have_eog=have_eog,
        blink_onsets_sec=blink_onsets_sec,
        blink_durs_sec=blink_durs_sec,
    )


def _refine_epoch_modalities(
    *,
    epoch_index: int,
    epoch_len: float,
    epochs: mne.Epochs,
    sfreq: float,
    n_samp_epoch: int,
    blink_onsets_sec: np.ndarray,
    blink_durs_sec: np.ndarray,
    data_ear: np.ndarray | None,
    data_eeg: np.ndarray | None,
    data_eog: np.ndarray | None,
    have_ear: bool,
    have_eeg: bool,
    have_eog: bool,
    segment_config: dict,
    md: Dict[str, List[Any]],
) -> None:
    """Refine blink metadata for a single epoch across modalities.

    Args:
        epoch_index: Index of the epoch to refine.
        epoch_len: Epoch duration in seconds.
        epochs: The epoched object containing metadata and events.
        sfreq: Sampling frequency in Hz.
        n_samp_epoch: Number of samples per epoch.
        blink_onsets_sec: Raw-level blink onsets (seconds).
        blink_durs_sec: Raw-level blink durations (seconds).
        data_ear/eeg/eog: Arrays shaped ``(n_epochs, 1, n_times)`` or ``None``
            when the modality is disabled.
        have_ear/eeg/eog: Flags indicating whether each modality is enabled.
        segment_config: Segmentation configuration passed to EAR refinement.
        md: Metadata dictionary to mutate in-place.

    Behavior
    --------
    * Epoch-relative blink bounds are derived with
      :func:`_compute_epoch_blink_bounds`.
    * ``md["n_blinks"]`` is updated regardless of enabled modalities.
    * When no blinks fall inside the epoch, the function returns early without
      populating modality-specific metadata.
    * EAR refinement validates that exactly one channel sample is available,
      flattens to ``(n_times,)``, and appends detailed thresholds/search fields.
    * EEG and EOG refinements operate on single-channel 1D vectors without
      averaging. Disabled modalities are skipped independently.

    Raises:
        ValueError: If any enabled modality provides more than one channel for
            the epoch or unexpected shapes are encountered.
    """

    epoch_start_samp = int(epochs.events[epoch_index, 0])
    epoch_start_sec = epoch_start_samp / sfreq
    epoch_end_sec = epoch_start_sec + epoch_len

    blink_starts, blink_ends = _compute_epoch_blink_bounds(
        blink_onsets_sec,
        blink_durs_sec,
        epoch_start_sec,
        epoch_end_sec,
        sfreq,
        n_samp_epoch,
    )

    n_blinks = len(blink_starts)
    md["n_blinks"][epoch_index] = n_blinks
    if n_blinks == 0:
        return

    for sr, er in zip(blink_starts, blink_ends):
        onset_sec_rel = sr / sfreq
        duration_sec_rel = max(0.0, (er - sr) / sfreq)
        md["blink_onset"][epoch_index] = append_to_slot(
            md["blink_onset"][epoch_index], onset_sec_rel
        )
        md["blink_duration"][epoch_index] = append_to_slot(
            md["blink_duration"][epoch_index], duration_sec_rel
        )

    if have_ear and data_ear is not None:
        seg_raw = data_ear[epoch_index]
        if seg_raw.ndim != 2 or seg_raw.shape[0] != 1:
            raise ValueError(
                f"EAR refinement expects a single channel, but epoch {epoch_index} contains shape {seg_raw.shape}."
            )
        seg = seg_raw.reshape(-1)
        refinements = _refine_ear_blinks_for_epoch(
            seg,
            blink_starts,
            blink_ends,
            sfreq,
            segment_config,
        )
        _append_ear_refinements(md, epoch_index, refinements, sfreq, n_samp_epoch)

    if have_eeg and data_eeg is not None:
        seg_raw = data_eeg[epoch_index]
        if seg_raw.ndim != 2 or seg_raw.shape[0] != 1:
            raise ValueError(
                f"EEG refinement expects a single channel, but epoch {epoch_index} contains shape {seg_raw.shape}."
            )
        seg = seg_raw.reshape(-1)
        _append_peak_refinements(
            md,
            epoch_index,
            seg,
            blink_starts,
            blink_ends,
            sfreq,
            "eeg",
            refine_local_maximum_stub,
            n_samp_epoch,
        )

    if have_eog and data_eog is not None:
        seg_raw = data_eog[epoch_index]
        if seg_raw.ndim != 2 or seg_raw.shape[0] != 1:
            raise ValueError(
                f"EOG refinement expects a single channel, but epoch {epoch_index} contains shape {seg_raw.shape}."
            )
        seg = seg_raw.reshape(-1)
        _append_peak_refinements(
            md,
            epoch_index,
            seg,
            blink_starts,
            blink_ends,
            sfreq,
            "eog",
            refine_local_maximum_stub,
            n_samp_epoch,
        )


def slice_raw_into_mne_epochs_refine_annot(
    # TODO : We need to enrich this function to support different blink onset selection strategy.
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    progress_bar: bool = True,
    segmentation_type: Optional[dict] = None,
    ear_threshold: Optional[float] = None,
) -> mne.Epochs:
    """Convert a continuous recording into equally spaced epochs with refinement."""

    segment_config = _prepare_segmentation_config(segmentation_type, ear_threshold)
    prep = _prepare_epochs_and_modalities(
        raw,
        epoch_len=epoch_len,
        blink_label=blink_label,
        segment_config=segment_config,
    )
    md = _init_metadata(prep.n_epochs, prep.have_eeg, prep.have_eog, prep.have_ear)

    iterator = range(prep.n_epochs)
    if progress_bar:
        iterator = tqdm(iterator, desc="Refining blink metadata", unit="epoch")

    for ei in iterator:
        _refine_epoch_modalities(
            epoch_index=ei,
            epoch_len=epoch_len,
            epochs=prep.epochs,
            sfreq=prep.sfreq,
            n_samp_epoch=prep.n_samp_epoch,
            blink_onsets_sec=prep.blink_onsets_sec,
            blink_durs_sec=prep.blink_durs_sec,
            data_ear=prep.data_ear,
            data_eeg=prep.data_eeg,
            data_eog=prep.data_eog,
            have_ear=prep.have_ear,
            have_eeg=prep.have_eeg,
            have_eog=prep.have_eog,
            segment_config=segment_config,
            md=md,
        )

    metadata = pd.DataFrame(md)
    prep.epochs.metadata = metadata

    logger.debug("Epoch metadata head: %s", metadata.head())
    logger.debug("Exiting slice_raw_into_mne_epochs_refine_annot")
    return prep.epochs


def refine_local_maximum_stub(
    signal_segment: np.ndarray,
    start_rel: int,
    end_rel: int,
    peak_rel_cvat: int | None = None,
) -> Tuple[int, int, int]:
    """Return a crude refinement for local maxima in a signal segment."""

    n = len(signal_segment)
    if n == 0:
        return 0, 0, 0

    rs_stub = max(0, min(start_rel, n - 1))
    re_stub = max(0, min(end_rel, n - 1))
    if rs_stub > re_stub:
        rs_stub = re_stub = min(rs_stub, re_stub)

    if peak_rel_cvat is not None and rs_stub <= peak_rel_cvat <= re_stub:
        valid_peak = peak_rel_cvat
    else:
        segment = signal_segment[rs_stub : re_stub + 1]
        max_idx_local = int(np.argmax(segment))
        valid_peak = rs_stub + max_idx_local

    return rs_stub, valid_peak, re_stub


def refine_blinks_from_epochs(
    segments: Sequence[mne.io.BaseRaw],
    channel: str,
    *,
    refine_func: Callable[[np.ndarray, int, int, int | None], Tuple[int, int, int]] = refine_local_maximum_stub,
    local_max_prominence: float = 0.01,
    search_expansion_frames: int | None = None,
    value_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Refine blink annotations within pre-sliced raw segments."""

    logger.info("Refining blinks across %d segments", len(segments))
    refined: List[Dict[str, Any]] = []
    if not segments:
        return refined

    sfreq = float(segments[0].info["sfreq"])
    if search_expansion_frames is None:
        search_expansion_frames = int(0.1 * sfreq)

    for epoch_index, segment in enumerate(segments):
        data = segment.get_data(picks=[channel])
        if data.size == 0:
            continue
        signal = data[0]
        ann = segment.annotations
        for ann_idx in range(len(ann)):
            onset = ann.onset[ann_idx]
            duration = ann.duration[ann_idx]
            start_rel = int(max(0, round(onset * sfreq) - search_expansion_frames))
            end_rel = int(round((onset + duration) * sfreq) + search_expansion_frames)
            end_rel = min(end_rel, len(signal) - 1)
            if end_rel < start_rel:
                end_rel = start_rel
            rs, peak, re = refine_func(signal, start_rel, end_rel, peak_rel_cvat=None)
            refined.append(
                {
                    "epoch_index": epoch_index,
                    "refined_start_frame": rs,
                    "refined_peak_frame": peak,
                    "refined_end_frame": re,
                    "epoch_signal": signal,
                }
            )

    logger.info("Generated %d refined blink entries", len(refined))
    return refined


__all__ = [
    "slice_raw_into_mne_epochs_refine_annot",
    "refine_local_maximum_stub",
    "refine_blinks_from_epochs",
]
