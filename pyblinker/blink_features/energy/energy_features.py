"""Blink energy feature calculations."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set

import mne
import pandas as pd

from pyblinker.logging import get_logger

from ...utils.iter_utils import ensure_list
from .common import compute_energy_metrics
from .helpers import _safe_stats

logger = get_logger(__name__)

_METRICS = (
    "blink_signal_energy",
    "teager_kaiser_energy",
    "blink_line_length",
    "blink_velocity_integral",
)
_STATS = ("mean", "std", "cv")


def _infer_modality(channel_name: str, info: mne.Info) -> str:
    """Infer modality label (ear/eeg/eog) from channel metadata."""

    ch_type = info.get_channel_types(picks=[channel_name])[0]
    ch_lower = channel_name.lower()
    if "ear" in ch_lower:
        return "ear"
    if ch_type == "eog" or "eog" in ch_lower:
        return "eog"
    if ch_type == "eeg" or "eeg" in ch_lower:
        return "eeg"
    return ch_type.lower()


def _available_styles(metadata_columns: Sequence[str] | None, modality: str) -> Set[str]:
    """Return frame-based segmentation styles present in metadata for a modality."""

    if metadata_columns is None:
        return set()

    styles: Set[str] = set()
    landmark_styles = {
        "base": ("start__left_base", "end__right_base"),
        "zero": ("start__left_zero", "end__right_zero"),
        "tent": ("start__left_x_intercept", "end__right_x_intercept"),
        "half_base": ("start__left_base_half_height", "end__right_base_half_height"),
        "half_zero": ("start__left_zero_half_height", "end__right_zero_half_height"),
    }
    for style, (start_key, end_key) in landmark_styles.items():
        start_col = f"{start_key}__{modality}"
        end_col = f"{end_key}__{modality}"
        if start_col in metadata_columns and end_col in metadata_columns:
            styles.add(style)

    start_prefix = "start__"
    modality_suffix = f"__{modality}"
    metadata_set = set(metadata_columns)
    for col in metadata_columns:
        if not col.startswith(start_prefix) or not col.endswith(modality_suffix):
            continue
        style = col[len(start_prefix) : -len(modality_suffix)]
        if not style:
            continue
        end_col = f"end__{style}__{modality}"
        if end_col in metadata_set:
            styles.add(style)

    return styles


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
    n_times: int,
) -> List[tuple[int, int]]:
    """Extract frame-aligned blink windows as ``(start_sample, end_sample)`` tuples."""

    landmark_style_keys = {
        "base": ("start__left_base", "end__right_base"),
        "zero": ("start__left_zero", "end__right_zero"),
        "tent": ("start__left_x_intercept", "end__right_x_intercept"),
        "half_base": ("start__left_base_half_height", "end__right_base_half_height"),
        "half_zero": ("start__left_zero_half_height", "end__right_zero_half_height"),
    }
    if style in landmark_style_keys:
        start_prefix, end_prefix = landmark_style_keys[style]
        start_key = f"{start_prefix}__{modality}"
        end_key = f"{end_prefix}__{modality}"
    else:
        start_key = f"start__{style}__{modality}"
        end_key = f"end__{style}__{modality}"

    starts = ensure_list(metadata_row.get(start_key))
    ends = ensure_list(metadata_row.get(end_key))

    windows: List[tuple[int, int]] = []
    for start_frame, end_frame in zip(starts, ends):
        if start_frame is None or end_frame is None:
            continue
        if pd.isna(start_frame) or pd.isna(end_frame):
            continue
        start_idx = max(0, int(round(float(start_frame))))
        end_idx = min(n_times, int(round(float(end_frame))))
        if end_idx <= start_idx:
            continue
        windows.append((start_idx, end_idx))
    return windows


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


def _channel_style_windows(
    *,
    metadata_row: Mapping[str, object],
    modality: str,
    available_styles: Set[str],
    n_times: int,
) -> Dict[str, List[tuple[int, int]]]:
    """Resolve output energy styles to frame windows by modality."""

    style_windows: Dict[str, List[tuple[int, int]]] = {}
    if modality in {"eeg", "eog"}:
        if "zero" in available_styles:
            style_windows["zero"] = _style_windows(metadata_row, modality, "zero", n_times)
        if "base" in available_styles:
            style_windows["base"] = _style_windows(metadata_row, modality, "base", n_times)
        if "tent" in available_styles:
            style_windows["tent"] = _style_windows(metadata_row, modality, "tent", n_times)

        if "half_base" in available_styles:
            style_windows["half"] = _style_windows(metadata_row, modality, "half_base", n_times)
        elif "half_zero" in available_styles:
            style_windows["half"] = _style_windows(metadata_row, modality, "half_zero", n_times)

        if "tent" in style_windows:
            style_windows["peak"] = style_windows["tent"]
        elif "base" in style_windows:
            style_windows["peak"] = style_windows["base"]

    elif modality == "ear":
        if "th_point" in available_styles:
            style_windows["th_point"] = _style_windows(metadata_row, modality, "th_point", n_times)
        elif "th_interpolation" in available_styles:
            style_windows["th_interpolation"] = _style_windows(metadata_row, modality, "th_interpolation", n_times)

    return style_windows


def _make_columns(modality_by_channel: Dict[str, str], styles_by_modality: Dict[str, Set[str]]) -> List[str]:
    """Generate ordered output columns for modality/style/metric/stat combinations."""

    columns: List[str] = []
    for ch, modality in modality_by_channel.items():
        for style in sorted(styles_by_modality.get(modality, set())):
            for metric in _METRICS:
                for stat in _STATS:
                    columns.append(f"{modality}__{style}__energy__{metric}_{stat}__{ch.upper()}")
    return columns


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
            _METRICS[0]: _safe_stats(energies),
            _METRICS[1]: _safe_stats(tkeo_vals),
            _METRICS[2]: _safe_stats(lengths),
            _METRICS[3]: _safe_stats(vel_ints),
        }

    return style_stats


def compute_energy_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute style-aware energy features for each epoch/channel."""

    if picks is None:
        ch_names = epochs.ch_names
    elif isinstance(picks, str):
        ch_names = [picks]
    else:
        ch_names = list(picks)

    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Channels not found: {missing}")

    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    n_times = epochs.get_data(picks=[ch_names[0]]).shape[-1] if n_epochs else 0
    modality_by_channel = {ch: _infer_modality(ch, epochs.info) for ch in ch_names}

    metadata_cols: Sequence[str] | None = (
        tuple(epochs.metadata.columns) if isinstance(epochs.metadata, pd.DataFrame) else None
    )
    eeg_styles = _available_styles(metadata_cols, "eeg")
    available_styles_by_modality: Dict[str, Set[str]] = {}
    styles_by_modality: Dict[str, Set[str]] = {}
    for modality in set(modality_by_channel.values()):
        raw_styles = _available_styles(metadata_cols, modality)
        if modality == "eog" and eeg_styles:
            raw_styles = raw_styles | eeg_styles
        available_styles_by_modality[modality] = raw_styles
        styles_by_modality[modality] = _normalize_styles_for_modality(raw_styles, modality)

    columns = _make_columns(modality_by_channel, styles_by_modality)
    index = (
        epochs.metadata.index
        if isinstance(epochs.metadata, pd.DataFrame)
        else pd.RangeIndex(n_epochs)
    )
    if n_epochs == 0:
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    data = epochs.get_data(picks=ch_names)
    logger.info("Computing energy features for %d epochs", n_epochs)
    records: List[Dict[str, float]] = []

    for ei in range(n_epochs):
        metadata_row = (
            epochs.metadata.iloc[ei]
            if isinstance(epochs.metadata, pd.DataFrame)
            else pd.Series(dtype=float)
        )
        record: Dict[str, float] = {}
        for ci, ch in enumerate(ch_names):
            modality = modality_by_channel[ch]
            stats_by_style = _compute_epoch_channel_energy_stats(
                style_windows=_channel_style_windows(
                    metadata_row=metadata_row,
                    modality=modality,
                    available_styles=available_styles_by_modality.get(modality, set()),
                    n_times=n_times,
                ),
                signal_1d=data[ei, ci, :],
                sfreq=sfreq,
                n_times=n_times,
            )
            for style, style_metrics in stats_by_style.items():
                for metric, stats in style_metrics.items():
                    for stat_name, value in stats.items():
                        record[
                            f"{modality}__{style}__energy__{metric}_{stat_name}__{ch.upper()}"
                        ] = value
        records.append(record)

    df = pd.DataFrame.from_records(records, index=index, columns=columns)
    logger.debug("Energy feature DataFrame shape: %s", df.shape)
    return df
