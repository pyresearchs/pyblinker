"""Blink refinement utilities shared across pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker.logging import get_logger

from .channel_utils import pick_ear_channels_from_raw
from .dict_utils import append_to_slot

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


def slice_raw_into_mne_epochs_refine_annot(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = 30.0,
    blink_label: Optional[str] = "blink",
    progress_bar: bool = True,
) -> mne.Epochs:
    """Convert a continuous recording into equally spaced epochs with refinement."""

    from pyblinker.blink_features.blink_events.blink_dataframe import (
        compute_outer_bounds,
    )

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

    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, misc=False)
    picks_eog = mne.pick_types(raw.info, eeg=False, eog=True, misc=False)
    picks_ear = pick_ear_channels_from_raw(raw)

    have_eeg = len(picks_eeg) > 0
    have_eog = len(picks_eog) > 0
    have_ear = len(picks_ear) > 0

    data_eeg = epochs.get_data(picks=picks_eeg) if have_eeg else None
    data_eog = epochs.get_data(picks=picks_eog) if have_eog else None
    data_ear = epochs.get_data(picks=picks_ear) if have_ear else None

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

    md = _init_metadata(n_epochs, have_eeg, have_eog, have_ear)

    iterator = range(n_epochs)
    if progress_bar:
        iterator = tqdm(iterator, desc="Refining blink metadata", unit="epoch")

    for ei in iterator:
        epoch_start_samp = int(epochs.events[ei, 0])
        epoch_start_sec = epoch_start_samp / sfreq
        epoch_end_sec = epoch_start_sec + epoch_len

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

        n_blinks = len(blink_starts)
        md["n_blinks"][ei] = n_blinks
        if n_blinks == 0:
            continue

        for sr, er in zip(blink_starts, blink_ends):
            onset_sec_rel = sr / sfreq
            duration_sec_rel = max(0.0, (er - sr) / sfreq)
            md["blink_onset"][ei] = append_to_slot(md["blink_onset"][ei], onset_sec_rel)
            md["blink_duration"][ei] = append_to_slot(
                md["blink_duration"][ei], duration_sec_rel
            )

        if have_ear and data_ear is not None:
            seg = data_ear[ei].mean(axis=0)
            peaks: List[int] = []
            for sr, er in zip(blink_starts, blink_ends):
                rs, trough, re = refine_ear_extrema_and_threshold_stub(
                    seg, sr, er, peak_rel_cvat=None
                )
                peaks.append(int(trough))
                md["blink_onset_ear"][ei] = append_to_slot(
                    md["blink_onset_ear"][ei], rs / sfreq
                )
                md["blink_duration_ear"][ei] = append_to_slot(
                    md["blink_duration_ear"][ei], max(0.0, (re - rs) / sfreq)
                )
                md["blink_onset_extremum_ear"][ei] = append_to_slot(
                    md["blink_onset_extremum_ear"][ei], trough / sfreq
                )
            if peaks:
                bounds = compute_outer_bounds(peaks, n_samp_epoch)
                for outer_start, outer_end in bounds:
                    md["blink_outer_start_ear"][ei] = append_to_slot(
                        md["blink_outer_start_ear"][ei], outer_start
                    )
                    md["blink_outer_end_ear"][ei] = append_to_slot(
                        md["blink_outer_end_ear"][ei], outer_end
                    )

        if have_eeg and data_eeg is not None:
            seg = data_eeg[ei].mean(axis=0)
            peaks = []
            for sr, er in zip(blink_starts, blink_ends):
                rs, peak, re = refine_local_maximum_stub(seg, sr, er, peak_rel_cvat=None)
                peaks.append(int(peak))
                md["blink_onset_eeg"][ei] = append_to_slot(
                    md["blink_onset_eeg"][ei], rs / sfreq
                )
                md["blink_duration_eeg"][ei] = append_to_slot(
                    md["blink_duration_eeg"][ei], max(0.0, (re - rs) / sfreq)
                )
                md["blink_onset_extremum_eeg"][ei] = append_to_slot(
                    md["blink_onset_extremum_eeg"][ei], peak / sfreq
                )
            if peaks:
                bounds = compute_outer_bounds(peaks, n_samp_epoch)
                for outer_start, outer_end in bounds:
                    md["blink_outer_start_eeg"][ei] = append_to_slot(
                        md["blink_outer_start_eeg"][ei], outer_start
                    )
                    md["blink_outer_end_eeg"][ei] = append_to_slot(
                        md["blink_outer_end_eeg"][ei], outer_end
                    )

        if have_eog and data_eog is not None:
            seg = data_eog[ei].mean(axis=0)
            peaks = []
            for sr, er in zip(blink_starts, blink_ends):
                rs, peak, re = refine_local_maximum_stub(seg, sr, er, peak_rel_cvat=None)
                peaks.append(int(peak))
                md["blink_onset_eog"][ei] = append_to_slot(
                    md["blink_onset_eog"][ei], rs / sfreq
                )
                md["blink_duration_eog"][ei] = append_to_slot(
                    md["blink_duration_eog"][ei], max(0.0, (re - rs) / sfreq)
                )
                md["blink_onset_extremum_eog"][ei] = append_to_slot(
                    md["blink_onset_extremum_eog"][ei], peak / sfreq
                )
            if peaks:
                bounds = compute_outer_bounds(peaks, n_samp_epoch)
                for outer_start, outer_end in bounds:
                    md["blink_outer_start_eog"][ei] = append_to_slot(
                        md["blink_outer_start_eog"][ei], outer_start
                    )
                    md["blink_outer_end_eog"][ei] = append_to_slot(
                        md["blink_outer_end_eog"][ei], outer_end
                    )

    metadata = pd.DataFrame(md)
    epochs.metadata = metadata

    logger.debug("Epoch metadata head: %s", metadata.head())
    logger.info("Exiting slice_raw_into_mne_epochs_refine_annot")
    return epochs


def refine_ear_extrema_and_threshold_stub(
    signal_segment: np.ndarray,
    start_rel: int,
    end_rel: int,
    peak_rel_cvat: int | None = None,
    *,
    local_max_prominence: float = 0.01,
    search_expansion_frames: int = 5,
    value_threshold: float | None = None,
) -> Tuple[int, int, int]:
    """Return a crude EAR trough refinement."""

    valid_trough = peak_rel_cvat
    if not (peak_rel_cvat is not None and 0 <= peak_rel_cvat < len(signal_segment)):
        if end_rel >= start_rel and len(signal_segment) > 0:
            valid_trough = (start_rel + end_rel) // 2
        else:
            valid_trough = 0

    rs_stub = max(0, min(start_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    re_stub = max(0, min(end_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    if rs_stub > re_stub:
        rs_stub = re_stub

    return rs_stub, valid_trough, re_stub


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


def plot_refined_blinks(
    refined_blinks: Sequence[Dict[str, Any]],
    sfreq: float,
    epoch_len: float,
    *,
    epoch_indices: Optional[Sequence[int]] = None,
    show: bool = False,
) -> List[plt.Figure]:
    """Plot signal segments with refined blink markers."""

    epochs_to_plot: Dict[int, Dict[str, Any]] = {}
    for blink in refined_blinks:
        idx = blink["epoch_index"]
        if epoch_indices is None or idx in epoch_indices:
            if idx not in epochs_to_plot:
                epochs_to_plot[idx] = {"signal": blink["epoch_signal"], "blinks": []}
            epochs_to_plot[idx]["blinks"].append(blink)

    if not epochs_to_plot:
        logger.warning("No epochs selected for plotting")
        return []

    figs: List[plt.Figure] = []
    time_axis = np.arange(0, epoch_len, 1.0 / sfreq)
    for epoch_index, data in epochs_to_plot.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_axis, data["signal"], label="Signal")
        for blink in data["blinks"]:
            start_t = blink["refined_start_frame"] / sfreq
            peak_t = blink["refined_peak_frame"] / sfreq
            end_t = blink["refined_end_frame"] / sfreq
            ax.axvline(start_t, color="g", linestyle="--")
            ax.axvline(peak_t, color="r")
            ax.axvline(end_t, color="b", linestyle="--")
        ax.set_title(f"Epoch {epoch_index}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        figs.append(fig)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs


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
    "refine_ear_extrema_and_threshold_stub",
    "refine_local_maximum_stub",
    "plot_refined_blinks",
    "refine_blinks_from_epochs",
]
