import math
import logging
from typing import Optional, List, Dict, Any, Iterable

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _pick_ear_channels_from_info(info: mne.Info) -> List[int]:
    """Heuristic for EAR channels (name contains 'ear' or 'eye_aspect_ratio')."""
    picks = []
    for i, name in enumerate(info["ch_names"]):
        nlow = name.lower()
        if "eye_aspect_ratio" in nlow or (("ear" in nlow) and "a1" not in nlow and "a2" not in nlow):
            picks.append(i)
    return picks


def _as_list(x: Any) -> List[float]:
    """Normalize a metadata cell (NaN, scalar, list) into a list of floats."""
    if isinstance(x, list):
        return [float(v) for v in x]
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    return [float(x)]


def add_blink_plots_to_report(
        epochs: mne.Epochs,
        *,
        report: Optional[mne.Report] = None,
        section: str = "Blink validation",
        pad_pre: float = 0.5,
        pad_post: float = 0.5,
        limit_per_epoch: Optional[int] = None,
        decim: Optional[int] = None,
        include_modalities: Iterable[str] = ("eeg", "eog", "ear"),
        progress_bar: bool = True,
) -> mne.Report:
    """
    Add per-epoch/per-blink/per-channel plots into an MNE Report for validation.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs with metadata produced by `slice_raw_into_mne_epochs_refine_annot`.
    report : mne.Report | None
        Existing report to append to; a new one is created if None.
    section : str
        Report section name.
    pad_pre, pad_post : float
        Seconds to pad before/after the base blink window when plotting.
    limit_per_epoch : int | None
        If set, limit number of blinks plotted per epoch (useful to keep reports small).
    decim : int | None
        Optional decimation factor applied before plotting (speed/size).
    include_modalities : Iterable[str]
        Any subset of {"eeg","eog","ear"} to include.
    progress_bar : bool
        Show a progress bar over epochs.

    Returns
    -------
    mne.Report
        The report with added figures.
    """
    if report is None:
        report = mne.Report(title="Blink validation")

    md = epochs.metadata
    if md is None or ("blink_onset" not in md.columns and "blink_duration" not in md.columns):
        raise RuntimeError("Epochs.metadata is missing manual blink fields.")

    sfreq = float(epochs.info["sfreq"])
    times = epochs.times  # relative to epoch start
    n_times = times.size
    epoch_len = times[-1] - times[0] + (1.0 / sfreq)

    # channel picks per modality
    picks_eeg = mne.pick_types(epochs.info, eeg=True, eog=False, misc=False) if "eeg" in include_modalities else []
    picks_eog = mne.pick_types(epochs.info, eeg=False, eog=True, misc=False) if "eog" in include_modalities else []
    picks_ear = _pick_ear_channels_from_info(epochs.info) if "ear" in include_modalities else []

    have_eeg = len(picks_eeg) > 0
    have_eog = len(picks_eog) > 0
    have_ear = len(picks_ear) > 0

    # prefetch data
    data_eeg = epochs.get_data(picks=picks_eeg) if have_eeg else None
    data_eog = epochs.get_data(picks=picks_eog) if have_eog else None
    data_ear = epochs.get_data(picks=picks_ear) if have_ear else None

    # convenience accessor for refined metadata lists
    def _refined_lists(ei: int, modality: str) -> Dict[str, List[float]]:
        out = {"onset": [], "duration": [], "extremum": []}
        for key, tgt in [
            (f"blink_onset_{modality}", "onset"),
            (f"blink_duration_{modality}", "duration"),
            (f"blink_onset_extremum_{modality}", "extremum"),
        ]:
            out[tgt] = _as_list(md.iloc[ei].get(key, np.nan))
        return out

    epoch_iter = range(len(epochs))
    if progress_bar:
        epoch_iter = tqdm(epoch_iter, desc="Building blink report", unit="epoch")

    for ei in epoch_iter:
        # manual (always present; may be lists)
        manual_onsets = _as_list(md.iloc[ei]["blink_onset"])
        manual_durs = _as_list(md.iloc[ei]["blink_duration"])
        n_blinks = min(len(manual_onsets), len(manual_durs))

        if n_blinks == 0:
            continue

        if limit_per_epoch is not None:
            n_blinks = min(n_blinks, int(limit_per_epoch))

        # refined by modality (lists aligned with manual order)
        ref_eeg = _refined_lists(ei, "eeg") if have_eeg else None
        ref_eog = _refined_lists(ei, "eog") if have_eog else None
        ref_ear = _refined_lists(ei, "ear") if have_ear else None

        for bi in range(n_blinks):
            # base window from manual
            m_on = manual_onsets[bi]
            m_dur = manual_durs[bi]
            m_off = m_on + max(0.0, m_dur)

            # expand window with refined bounds (union), if available
            win_start = m_on
            win_end = m_off
            for ref in (ref_eeg, ref_eog, ref_ear):
                if ref is None:
                    continue
                rs = ref["onset"]
                rd = ref["duration"]
                if bi < len(rs) and bi < len(rd):
                    r_on = rs[bi]
                    r_off = rs[bi] + max(0.0, rd[bi])
                    if not math.isnan(r_on):
                        win_start = min(win_start, r_on)
                    if not math.isnan(r_off):
                        win_end = max(win_end, r_off)

            # add padding and clip
            win_start = max(0.0, win_start - pad_pre)
            win_end = min(epoch_len, win_end + pad_post)

            s0 = int(np.clip(round(win_start * sfreq), 0, n_times - 1))
            s1 = int(np.clip(round(win_end   * sfreq), 0, n_times - 1))
            if s1 < s0:
                s1 = s0

            t_seg = times[s0:s1 + 1]
            # per-modality plotting (each channel separately)
            def _plot_mod(mod: str, picks: List[int], data_mod: np.ndarray, ref: Optional[Dict[str, List[float]]]):
                if not picks:
                    return
                for c_rel, ch_idx in enumerate(picks):
                    ch_name = epochs.ch_names[ch_idx]
                    y = data_mod[ei, c_rel, s0:s1 + 1]
                    if decim and decim > 1:
                        y = y[::decim]
                        t = t_seg[::decim]
                    else:
                        t = t_seg

                    fig, ax = plt.subplots(figsize=(7.5, 3.0))
                    ax.plot(t, y, lw=1.0)
                    ax.set_title(f"Epoch {ei} • Blink {bi} • {mod.upper()} • {ch_name}")
                    ax.set_xlabel("Time from epoch start (s)")
                    ax.set_ylabel("Amplitude")

                    # manual window
                    ax.axvline(m_on, linestyle="--", alpha=0.9, label="manual onset")
                    ax.axvline(m_off, linestyle="--", alpha=0.9, label="manual offset")

                    # refined window + extremum
                    if ref is not None:
                        rs = ref["onset"]
                        rd = ref["duration"]
                        rx = ref["extremum"]
                        if bi < len(rs) and not math.isnan(rs[bi]):
                            ax.axvline(rs[bi], linestyle="-", alpha=0.9, label=f"{mod} onset (refined)")
                        if bi < len(rd) and not math.isnan(rd[bi]):
                            ax.axvline(rs[bi] + max(0.0, rd[bi]), linestyle="-", alpha=0.9, label=f"{mod} offset (refined)")
                        if bi < len(rx) and not math.isnan(rx[bi]):
                            ax.axvline(rx[bi], linestyle=":", alpha=0.9, label=f"{mod} extremum")

                    ax.legend(loc="upper right", fontsize=8, ncol=3)
                    ax.grid(True, alpha=0.2)

                    caption = (
                        f"Blink window (manual): {m_on:.3f}s–{m_off:.3f}s. "
                        f"Padding: −{pad_pre:.2f}/+{pad_post:.2f}s. "
                        f"Sampling: {sfreq:.2f} Hz{f', decim={decim}' if decim else ''}."
                    )
                    try:
                        report.add_figure(
                            fig=fig,
                            title=f"Epoch {ei} | Blink {bi} | {mod.upper()} | {ch_name}",
                            caption=caption,
                            section=section,
                            tags=("blink", "validation", mod, ch_name),
                        )
                    finally:
                        plt.close(fig)

            if have_eeg:
                _plot_mod("eeg", picks_eeg, data_eeg, ref_eeg)
            if have_eog:
                _plot_mod("eog", picks_eog, data_eog, ref_eog)
            if have_ear:
                _plot_mod("ear", picks_ear, data_ear, ref_ear)

    return report
