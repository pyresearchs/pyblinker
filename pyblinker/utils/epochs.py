"""Epoching utilities for time-series data."""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BLINK_LABEL = "blink"        # Annotation label for blink events
EPOCH_LEN = 30.0              # Epoch duration in seconds

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core utility functions
# -----------------------------------------------------------------------------

def slice_raw_into_epochs(
    raw: mne.io.BaseRaw,
    *,
    epoch_len: float = EPOCH_LEN,
    blink_label: Optional[str] = BLINK_LABEL,
    progress_bar: bool = True,
) -> Tuple[List[mne.io.BaseRaw], pd.DataFrame, List[Tuple[int, int]], List[Tuple[float, float]]]:
    """Slice a raw recording into epochs and count blink annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording with blink annotations.
    epoch_len : float, optional
        Length of each epoch in seconds. Defaults to :data:`EPOCH_LEN`.
    blink_label : str | None, optional
        Annotation label to filter blinks. ``None`` counts all annotations.

    Returns
    -------
    list of mne.io.BaseRaw
        Raw segments cropped from the input recording with annotations shifted
        relative to each segment.
    pandas.DataFrame
        Blink counts per epoch with columns ``epoch_id`` and ``blink_count``.
    list of tuple
        Pairs of epoch indices where a blink spans the boundary between epochs.
    list of tuple
        Start and stop times for each epoch (seconds) relative to the original
        raw.
    """
    logger.info("Slicing raw into epochs (%.1fs)", epoch_len)

    ann = raw.annotations
    mask = np.ones(len(ann), dtype=bool)
    if blink_label is not None:
        mask &= ann.description == blink_label
    onsets = ann.onset[mask]
    durations = ann.duration[mask]

    total_time = raw.times[-1]
    n_epochs = int(np.ceil(total_time / epoch_len))
    counts: List[int] = [0] * n_epochs
    boundary_pairs: List[Tuple[int, int]] = []
    segments: List[mne.io.BaseRaw] = []
    times: List[Tuple[float, float]] = []

    for i in tqdm(
        range(n_epochs), desc="Cropping epochs", unit="epoch", disable=not progress_bar
    ):
        start = i * epoch_len
        stop = min(start + epoch_len, total_time)
        times.append((start, stop))

        in_epoch = (onsets >= start) & (onsets < stop)
        counts[i] = int(np.sum(in_epoch))
        spans = in_epoch & ((onsets + durations) > stop)
        for _ in np.where(spans)[0]:
            if i + 1 < n_epochs:
                boundary_pairs.append((i, i + 1))

        mini = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
        ann_epoch = mini.annotations
        shifted = mne.Annotations(
            onset=ann_epoch.onset - start,
            duration=ann_epoch.duration,
            description=ann_epoch.description,
        )
        mini.set_annotations(shifted)
        segments.append(mini)

    df = pd.DataFrame({"epoch_id": range(n_epochs), "blink_count": counts})
    logger.debug("Blink counts per epoch: %s", counts)
    logger.debug("Cross-boundary pairs: %s", boundary_pairs)
    return segments, df, boundary_pairs, times

def save_epoch_raws(
    segments: Sequence[mne.io.BaseRaw],
    times: Sequence[Tuple[float, float]],
    out_dir: Path,
    *,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Save cropped raw segments to disk.

    Parameters
    ----------
    segments : sequence of mne.io.BaseRaw
        Segments returned by :func:`slice_raw_into_epochs`.
    times : sequence of tuple
        Start and stop time pairs for file naming.
    out_dir : pathlib.Path
        Directory to write the files.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to ``False``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (segment, span) in enumerate(zip(segments, times)):
        start, stop = span
        fname = out_dir / f"epoch_{idx:04d}_{start:07.2f}s-{stop:07.2f}s_raw.fif"
        if fname.exists() and not overwrite:
            logger.debug("Skipping existing %s", fname)
            continue
        segment.save(fname, overwrite=overwrite, verbose=verbose)


def generate_epoch_report(
    segments: Sequence[mne.io.BaseRaw],
    times: Sequence[Tuple[float, float]],
    *,
    verbose: bool = False,
) -> mne.Report:
    """Create a simple report visualizing each segment.

    Parameters
    ----------
    segments : sequence of mne.io.BaseRaw
        Segment data to plot.
    times : sequence of tuple
        Start and stop time pairs for titles.

    Returns
    -------
    mne.Report
        Report containing one figure per segment.
    """
    report = mne.Report(title="Epoch Overview")
    for idx, (segment, span) in enumerate(zip(segments, times)):
        start, stop = span
        fig = segment.plot(
            n_channels=min(10, len(segment.ch_names)),
            scalings="auto",
            title=f"Epoch {idx} ({start:.2f}-{stop:.2f}s)",
            show=False,
            verbose=verbose,
        )
        report.add_figure(fig, title=f"Epoch {idx}", section="epochs")
        plt.close(fig)
    return report

def slice_into_mini_raws(
    raw: mne.io.BaseRaw,
    out_dir: Path,
    *,
    epoch_len: float = EPOCH_LEN,
    blink_label: Optional[str] = BLINK_LABEL,
    save: bool = True,
    overwrite: bool = False,
    report: bool = False,
    progress_bar: bool = True,
) -> Tuple[List[mne.io.BaseRaw], pd.DataFrame, List[Tuple[int, int]], Optional[mne.Report]]:
    """Slice a raw recording into epochs with optional saving and reporting.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording with blink annotations.
    out_dir : Path
        Directory to save epoch files and/or report.
    epoch_len : float, optional
        Length of each epoch in seconds. Defaults to :data:`EPOCH_LEN`.
    blink_label : str | None, optional
        Annotation label used to filter blinks. ``None`` counts all
        annotations.
    save : bool, optional
        Whether to write epoch files to ``out_dir``. Defaults to ``True``.
    overwrite : bool, optional
        Overwrite any existing files when ``save`` is ``True``. Defaults to
        ``False``.
    report : bool, optional
        Generate and optionally save an ``mne.Report`` to ``out_dir``. The
        report is only produced when both ``report`` and ``save`` are ``True``.

    Returns
    -------
    list of mne.io.BaseRaw
        Raw segments retained in memory.
    pandas.DataFrame
        Blink counts per epoch.
    list of tuple
        Boundary pairs spanning adjacent epochs.
    mne.Report | None
        The generated report if requested, otherwise ``None``.
    """
    logger.info("Entering slice_into_mini_raws")
    segments, df, boundary_pairs, times = slice_raw_into_epochs(
        raw, epoch_len=epoch_len, blink_label=blink_label, progress_bar=progress_bar
    )
    rep: Optional[mne.Report] = None
    if save:
        save_epoch_raws(segments, times, out_dir, overwrite=overwrite, verbose=False)
        if report:
            rep = generate_epoch_report(segments, times, verbose=False)
            rep.save(out_dir / "epoch_report.html", overwrite=overwrite, open_browser=False)
    logger.info("Exiting slice_into_mini_raws")
    return segments, df, boundary_pairs, rep
