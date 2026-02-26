"""Blink energy feature calculations."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set

import mne
import pandas as pd

from pyblinker.logging import get_logger

from .._epoch_context import (
    available_styles_by_modality,
    build_epoch_context,
    empty_feature_frame,
    get_metadata_row,
    frame_from_records,
)
from .._style_windows import style_windows_from_metadata
from .column_headers import METRICS, build_output_columns, make_stat_column
from .common import compute_energy_metrics
from .helpers import compute_basic_statistics

logger = get_logger(__name__)

def _normalize_styles_for_modality(styles: Set[str], modality: str) -> Set[str]:
    if modality in {"eeg", "eog"}:
        normalized_styles: Set[str] = set()
        if "zero" in styles:
            normalized_styles.add("zero")
        if "base" in styles:
            normalized_styles.add("base")
        if "tent" in styles:
            normalized_styles.add("tent")
        if "half_base" in styles or "half_zero" in styles:
            normalized_styles.add("half")
        if "tent" in styles or "base" in styles:
            normalized_styles.add("peak")
        return normalized_styles

    if modality == "ear":
        if "th_point" in styles:
            return {"th_point"}
        if "th_interpolation" in styles:
            return {"th_interpolation"}
        return set()

    return styles

def _compute_epoch_channel_energy_stats(
    *,
    style_windows: Dict[str, List[tuple[int, int]]],
    signal_1d,
    sfreq: float,
    n_times: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-metric summary stats for all style windows in one epoch/channel."""

    style_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for style, windows in style_windows.items():
        energies: List[float] = []
        tkeo_vals: List[float] = []
        lengths: List[float] = []
        vel_ints: List[float] = []

        for start_idx, end_idx in windows:
            if start_idx >= n_times:
                continue
            sl = slice(max(0, start_idx), min(end_idx, n_times))
            if sl.stop <= sl.start:
                continue
            segment = signal_1d[sl]
            if getattr(segment, "size", 0) == 0:
                continue
            metrics = compute_energy_metrics(segment, sfreq)
            energies.append(float(metrics["signal_energy"]))
            tkeo_vals.append(float(metrics["teager_kaiser_energy"]))
            lengths.append(float(metrics["line_length"]))
            vel_ints.append(float(metrics["velocity_integral"]))

        style_stats[style] = {
            METRICS[0]: compute_basic_statistics(energies),
            METRICS[1]: compute_basic_statistics(tkeo_vals),
            METRICS[2]: compute_basic_statistics(lengths),
            METRICS[3]: compute_basic_statistics(vel_ints),
        }

    return style_stats


def compute_energy_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute style-aware energy features for each epoch/channel."""

    ctx = build_epoch_context(epochs, picks)
    available = available_styles_by_modality(
        ctx.metadata_cols,
        set(ctx.modality_by_channel.values()),
        include_eeg_for_eog=True,
    )
    styles_by_modality: Dict[str, Set[str]] = {
        modality: _normalize_styles_for_modality(raw_styles, modality)
        for modality, raw_styles in available.items()
    }

    columns = build_output_columns(ctx.modality_by_channel, styles_by_modality)
    if ctx.n_epochs == 0:
        return empty_feature_frame(ctx.index, columns)

    data = epochs.get_data(picks=ctx.ch_names)
    logger.info("Computing energy features for %d epochs", ctx.n_epochs)
    records: List[Dict[str, float]] = []

    for ei in range(ctx.n_epochs):
        metadata_row = (
            get_metadata_row(epochs, ei)
        )
        record: Dict[str, float] = {}
        for ci, ch in enumerate(ctx.ch_names):
            modality = ctx.modality_by_channel[ch]
            stats_by_style = _compute_epoch_channel_energy_stats(
                style_windows=style_windows_from_metadata(
                    metadata_row=metadata_row,
                    modality=modality,
                    available_styles=available.get(modality, set()),
                    n_times=ctx.n_times,
                    include_half=True,
                    include_peak=True,
                    ear_mode="keep",
                ),
                signal_1d=data[ei, ci, :],
                sfreq=ctx.sfreq,
                n_times=ctx.n_times,
            )
            for style, style_metrics in stats_by_style.items():
                for metric, stats in style_metrics.items():
                    for stat_name, value in stats.items():
                        record[
                            make_stat_column(
                                modality=modality,
                                style=style,
                                metric=metric,
                                stat=stat_name,
                                channel=ch if modality == "eog" else ch.upper(),
                            )
                        ] = value
        records.append(record)

    df = frame_from_records(records, index=ctx.index, columns=columns)
    logger.debug("Energy feature DataFrame shape: %s", df.shape)
    return df
