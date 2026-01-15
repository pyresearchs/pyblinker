"""Aggregate blink morphology features from :class:`mne.Epochs`."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Set

import mne
import numpy as np
import pandas as pd
from pyblinker.logging import get_logger

from .core_metrics import (
    MORPHOLOGY_METRIC_STEMS,
    compute_blink_durations,
    compute_blink_peak_times,
    compute_time_base_shut,
    compute_time_zero_shut,
)
from .per_blink import compute_blink_waveform_metrics
from ..energy.helpers import _safe_stats, segment_to_samples
from ..utils.aggregation import prepare_epoch_channel_data
from ...utils.epoch_utils import resolve_channels
from ...utils.iter_utils import ensure_list

logger = get_logger(__name__)

_STATS = ("mean", "std", "cv")
_SHUT_AMP_FRACTION = 0.9
_LEGACY_MORPHOLOGY_METRICS = (
    "duration_zero",
    "duration_base",
    "duration_tent",
    "duration_half_base",
    "duration_half_zero",
    "closing_time_zero",
    "reopening_time_zero",
    "time_shut_zero",
    "time_shut_base",
    "closing_time_tent",
    "reopening_time_tent",
    "time_shut_tent",
    "inter_blink_max_amp",
)
_DURATION_STYLE_MAP = {
    "base": "duration_base",
    "zero": "duration_zero",
    "tent": "duration_tent",
    "half_base": "duration_half_base",
    "half_zero": "duration_half_zero",
}


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


def _default_morphology_channels(epochs: mne.Epochs) -> List[str]:
    """Select default EAR/EOG channels when ``picks`` are unspecified."""

    ch_names = [
        ch for ch in epochs.ch_names if "EOG" in ch.upper() or "EAR" in ch.upper()
    ]
    if not ch_names:
        raise ValueError("No default EAR/EOG channels found")
    return ch_names


def _available_styles(metadata_columns: Sequence[str] | None, modality: str) -> Set[str]:
    """Return segmentation styles present in metadata for a modality."""

    if metadata_columns is None:
        return set()

    styles: Set[str] = set()
    suffix = f"__{modality}"
    for col in metadata_columns:
        if not col.startswith("onset__") or not col.endswith(suffix):
            continue
        style = col[len("onset__") : -len(suffix)]
        if "sample" in style.lower():
            continue
        duration_key = f"duration__{style}__{modality}"
        if duration_key in metadata_columns:
            styles.add(style)

    landmark_styles = {
        "base": (
            f"start__left_base__{modality}",
            f"end__right_base__{modality}",
        ),
        "zero": (
            f"start__left_zero__{modality}",
            f"end__right_zero__{modality}",
        ),
        "tent": (
            f"start__left_x_intercept__{modality}",
            f"end__right_x_intercept__{modality}",
        ),
        "half_base": (
            f"start__left_base_half_height__{modality}",
            f"end__right_base_half_height__{modality}",
        ),
        "half_zero": (
            f"start__left_zero_half_height__{modality}",
            f"end__right_zero_half_height__{modality}",
        ),
    }
    for style, (start_key, end_key) in landmark_styles.items():
        if start_key in metadata_columns and end_key in metadata_columns:
            styles.add(style)

    return styles


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
) -> List[tuple[float, float]]:
    """Extract blink windows for a modality/style pair."""

    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    onsets = (
        ensure_list(metadata_row.get(onset_key))
        if metadata_row.get(onset_key) is not None
        else []
    )
    durations = (
        ensure_list(metadata_row.get(duration_key))
        if metadata_row.get(duration_key) is not None
        else []
    )
    windows: List[tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if pd.isna(onset) or pd.isna(duration):
            continue
        windows.append((float(onset), float(duration)))
    return windows


def _coerce_list(value: object) -> List[float]:
    values = ensure_list(value) if value is not None else []
    cleaned: List[float] = []
    for item in values:
        if item is None or pd.isna(item):
            cleaned.append(float("nan"))
        else:
            cleaned.append(float(item))
    return cleaned


def _pad_to_length(values: List[float], length: int) -> List[float]:
    if len(values) >= length:
        return values[:length]
    return values + [float("nan")] * (length - len(values))


def _peak_times_from_windows(
    signal: np.ndarray,
    windows: Sequence[tuple[float, float]],
    sfreq: float,
    n_times: int,
) -> tuple[List[float], List[float]]:
    peak_indices: List[float] = []
    peak_values: List[float] = []
    for onset_s, duration_s in windows:
        sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
        segment = signal[sl]
        if segment.size == 0:
            peak_indices.append(float("nan"))
            peak_values.append(float("nan"))
            continue
        peak_idx = int(np.argmax(segment))
        peak_indices.append(float(sl.start + peak_idx))
        peak_values.append(float(segment[peak_idx]))
    return peak_indices, peak_values


def _build_blink_landmark_frame(
    metadata_row: Mapping[str, object],
    signal: np.ndarray,
    sfreq: float,
    n_times: int,
    *,
    modality: str,
    styles: Sequence[str],
) -> pd.DataFrame:
    landmark_keys = {
        "left_base": f"start__left_base__{modality}",
        "right_base": f"end__right_base__{modality}",
        "left_zero": f"start__left_zero__{modality}",
        "right_zero": f"end__right_zero__{modality}",
        "left_x_intercept": f"start__left_x_intercept__{modality}",
        "right_x_intercept": f"end__right_x_intercept__{modality}",
        "left_base_half_height": f"start__left_base_half_height__{modality}",
        "right_base_half_height": f"end__right_base_half_height__{modality}",
        "left_zero_half_height": f"start__left_zero_half_height__{modality}",
        "right_zero_half_height": f"end__right_zero_half_height__{modality}",
        "x_intersect": f"x_intersect__{modality}",
        "y_intersect": f"y_intersect__{modality}",
    }
    landmark_lists = {key: _coerce_list(metadata_row.get(col)) for key, col in landmark_keys.items()}

    peak_key_candidates = (
        f"onset__refine_extremum__{modality}",
        f"blink_onset_extremum_{modality}",
    )
    peak_times_sec: List[float] = []
    for peak_key in peak_key_candidates:
        if metadata_row.get(peak_key) is not None:
            peak_times_sec = _coerce_list(metadata_row.get(peak_key))
            if peak_times_sec:
                break

    window_style = None
    for candidate in ("refine", "outer"):
        if candidate in styles:
            window_style = candidate
            break
    if window_style is None and styles:
        window_style = styles[0]
    windows = (
        _style_windows(metadata_row, modality, window_style)
        if window_style is not None
        else []
    )

    lengths = [len(values) for values in landmark_lists.values()]
    lengths.append(len(peak_times_sec))
    lengths.append(len(windows))
    n_blinks = max(lengths) if lengths else 0
    if n_blinks == 0:
        return pd.DataFrame()

    data: Dict[str, List[float]] = {
        key: _pad_to_length(values, n_blinks)
        for key, values in landmark_lists.items()
    }

    if peak_times_sec:
        peak_idx = [
            float(round(time_s * sfreq)) if not pd.isna(time_s) else float("nan")
            for time_s in peak_times_sec
        ]
        peak_idx = _pad_to_length(peak_idx, n_blinks)
        peak_values = []
        for idx in peak_idx:
            if pd.isna(idx):
                peak_values.append(float("nan"))
                continue
            int_idx = int(idx)
            if int_idx < 0 or int_idx >= n_times:
                peak_values.append(float("nan"))
                continue
            peak_values.append(float(signal[int_idx]))
        data["max_blink"] = peak_idx
        data["max_value"] = _pad_to_length(peak_values, n_blinks)
    else:
        peak_idx, peak_values = _peak_times_from_windows(signal, windows, sfreq, n_times)
        data["max_blink"] = _pad_to_length(peak_idx, n_blinks)
        data["max_value"] = _pad_to_length(peak_values, n_blinks)

    return pd.DataFrame(data)


def _apply_morphology_properties(
    blink_df: pd.DataFrame,
    signal: np.ndarray,
    sfreq: float,
    *,
    modality: str,
) -> pd.DataFrame:
    if blink_df.empty:
        return blink_df

    compute_blink_durations(blink_df, sfreq, modality=modality, fitted=True)

    for col in (
        "closing_time_zero",
        "reopening_time_zero",
        "time_shut_zero",
        "time_shut_base",
        "closing_time_tent",
        "reopening_time_tent",
        "time_shut_tent",
        "inter_blink_max_amp",
    ):
        if col not in blink_df.columns:
            blink_df[col] = np.nan

    zero_valid = blink_df[
        ["left_zero", "right_zero", "max_blink", "max_value"]
    ].notna().all(axis=1)
    if zero_valid.any():
        zero_df = blink_df.loc[zero_valid, :].copy()
        compute_time_zero_shut(
            zero_df,
            signal,
            sfreq,
            modality=modality,
            shut_amp_fraction=_SHUT_AMP_FRACTION,
        )
        blink_df.loc[zero_valid, ["closing_time_zero", "reopening_time_zero", "time_shut_zero"]] = (
            zero_df[["closing_time_zero", "reopening_time_zero", "time_shut_zero"]]
        )

    base_valid = blink_df[["left_base", "right_base", "max_value"]].notna().all(axis=1)
    fitted_valid = base_valid & blink_df[
        ["left_x_intercept", "right_x_intercept", "x_intersect"]
    ].notna().all(axis=1)
    if fitted_valid.any():
        fitted_df = blink_df.loc[fitted_valid, :].copy()
        compute_time_base_shut(
            fitted_df,
            signal,
            sfreq,
            shut_amp_fraction=_SHUT_AMP_FRACTION,
            fitted=True,
        )
        blink_df.loc[
            fitted_valid,
            ["time_shut_base", "closing_time_tent", "reopening_time_tent", "time_shut_tent"],
        ] = fitted_df[
            ["time_shut_base", "closing_time_tent", "reopening_time_tent", "time_shut_tent"]
        ]

    base_only = base_valid & ~fitted_valid
    if base_only.any():
        base_df = blink_df.loc[base_only, :].copy()
        compute_time_base_shut(
            base_df,
            signal,
            sfreq,
            shut_amp_fraction=_SHUT_AMP_FRACTION,
            fitted=False,
        )
        blink_df.loc[base_only, ["time_shut_base"]] = base_df[["time_shut_base"]]

    peak_valid = blink_df[["max_blink"]].notna().all(axis=1)
    if peak_valid.any():
        peak_df = blink_df.loc[peak_valid, :].copy()
        compute_blink_peak_times(peak_df, signal, sfreq, fitted=True)
        blink_df.loc[peak_valid, ["inter_blink_max_amp"]] = peak_df[["inter_blink_max_amp"]]

    return blink_df


class MorphologyBlinkFeatureExtractor:
    """Compute blink morphology features from MNE objects."""

    def __init__(self, epochs: mne.Epochs | None = None, raw: mne.io.BaseRaw | None = None):
        self.epochs = epochs
        self.raw = raw

    def _sampling_frequency(self) -> float:
        """Return sampling frequency from available MNE object."""
        if hasattr(self, "epochs") and self.epochs is not None:
            return float(self.epochs.info["sfreq"])
        if hasattr(self, "raw") and self.raw is not None:
            return float(self.raw.info["sfreq"])
        raise ValueError("Neither self.epochs nor self.raw defined (need MNE object).")

    def compute(self, picks: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Compute blink morphology statistics for each epoch.

        Parameters
        ----------
        picks : str | list of str | None, optional
            Channel name(s) to include. ``None`` selects channels containing
            ``"EOG"`` or ``"EAR"``. If any requested channel is missing a
            :class:`ValueError` is raised.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` containing ``mean``, ``std``, and
            ``cv`` aggregates for each morphology metric per channel.

        Raises
        ------
        ValueError
            If required metadata columns are absent or ``picks`` contain unknown
            channels.

        Notes
        -----
        If an epoch contains no blinks, all morphology statistics for that epoch
        are ``NaN``.
        """
        logger.info("Computing morphology features for epochs")

        if self.epochs is None:
            raise ValueError("self.epochs is required for feature computation")
        if self.epochs.metadata is None:
            raise ValueError("epochs.metadata must be provided")

        sfreq = self._sampling_frequency()
        ch_names = resolve_channels(self.epochs, picks, default=_default_morphology_channels)
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=ch_names,
            sfreq=sfreq,
        )

        modality_map: Dict[str, str] = {ch: _infer_modality(ch, self.epochs.info) for ch in ch_names}
        modality_channels: Dict[str, List[str]] = {}
        for ch, mod in modality_map.items():
            modality_channels.setdefault(mod, []).append(ch)
        styles_by_modality: Dict[str, Set[str]] = {
            modality: {"base"} for modality in modality_channels
        }

        metadata_cols: Sequence[str] | None = (
            tuple(self.epochs.metadata.columns)
            if isinstance(self.epochs.metadata, pd.DataFrame)
            else None
        )

        for mod in set(modality_map.values()):
            styles = _available_styles(metadata_cols, mod)
            # fallback_styles[mod] = not styles
            styles_by_modality[mod] = styles

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod, {"base"})):
                metrics_for_style = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]
                metrics_for_style.append("duration")
                for metric in metrics_for_style:
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(f"{mod}__{style}__morphology__{metric}_{stat}__{ch}")
            if mod == "eeg" and channels:
                column_set.update(_LEGACY_MORPHOLOGY_METRICS)

        columns = sorted(column_set)
        if n_epochs == 0:
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        records: List[Dict[str, float]] = []
        logger.info("Computing morphology features for %d epochs", n_epochs)

        for ei in range(n_epochs):
            metadata_row = (
                self.epochs.metadata.iloc[ei]
                if isinstance(self.epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            record: Dict[str, float] = {}
            for modality, channels in modality_channels.items():
                styles = sorted(styles_by_modality.get(modality, {"base"}))
                for ch_index, ch in enumerate(channels):
                    signal = channel_data[ch]["raw"][ei]
                    blink_df = _build_blink_landmark_frame(
                        metadata_row,
                        signal,
                        sfreq,
                        n_times,
                        modality=modality,
                        styles=styles,
                    )
                    blink_df = _apply_morphology_properties(
                        blink_df,
                        signal,
                        sfreq,
                        modality=modality,
                    )
                    for style in styles:
                        metrics_for_style = [
                            f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS
                        ]
                        windows = _style_windows(metadata_row, modality, style)
                        per_metric: Dict[str, List[float]] = {
                            m: [] for m in metrics_for_style
                        }
                        for onset_s, duration_s in windows:
                            sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                            segment = signal[sl]
                            metrics = compute_blink_waveform_metrics(
                                segment,
                                sfreq,
                                method=style,
                                modality=modality,
                            )
                            for metric_name in metrics_for_style:
                                per_metric[metric_name].append(
                                    metrics.get(metric_name, float("nan"))
                                )

                        duration_key = _DURATION_STYLE_MAP.get(style)
                        if duration_key and duration_key in blink_df.columns:
                            per_metric["duration"] = blink_df[duration_key].tolist()
                        else:
                            per_metric["duration"] = []

                        for metric, values in per_metric.items():
                            stats = _safe_stats(values)
                            for stat_name, value in stats.items():
                                column = (
                                    f"{modality}__{style}__morphology__{metric}_{stat_name}__{ch}"
                                )
                                record[column] = value

                    if modality == "eeg" and ch_index == 0 and not blink_df.empty:
                        for legacy_metric in _LEGACY_MORPHOLOGY_METRICS:
                            if legacy_metric in blink_df.columns:
                                record[legacy_metric] = _safe_stats(
                                    blink_df[legacy_metric].tolist()
                                )["mean"]
            records.append(record)

        df = pd.DataFrame.from_records(records, index=index, columns=columns)
        logger.debug("Morphology feature DataFrame shape: %s", df.shape)
        return df


def compute_morphology_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute blink morphology features for each epoch and channel."""

    extractor = MorphologyBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)


def compute_epoch_morphology_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute blink morphology statistics for each epoch."""

    return compute_morphology_features(epochs, picks=picks)
