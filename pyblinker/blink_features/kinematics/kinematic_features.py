"""Blink kinematic feature calculations based on epoch metadata.
All feature calculations rely only on blink onset and blink duration stored in the metadata.
This design intentionally decouples feature extraction from how blink boundaries are defined.

As a result, users should have full flexibility to define blink onset and duration according to their needs.
See pyblinker/segmentation/refinement.py
"""

from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Mapping, Sequence, Set

import mne
import pandas as pd

from .._core_blink import CANONICAL_METRIC_STEMS, METHODS_BY_MODALITY
from .per_blink import compute_segment_kinematics
from ..energy.helpers import segment_to_samples, _safe_stats
from ...utils.iter_utils import ensure_list
from ..utils.aggregation import prepare_epoch_channel_data

logger = get_logger(__name__)

# Base statistic names (kinematics defaults to base per modality)
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
    """Return segmentation styles present in metadata for a modality."""

    if metadata_columns is None:
        return set()

    styles: Set[str] = set()
    suffix = f"__{modality}"
    for col in metadata_columns:
        if col.startswith("onset__") and col.endswith(suffix):
            style = col[len("onset__") : -len(suffix)]
            if f"duration__{style}__{modality}" in metadata_columns:
                styles.add(style)

    mod_onset = f"blink_onset_{modality}"
    mod_duration = f"blink_duration_{modality}"
    if mod_onset in metadata_columns and mod_duration in metadata_columns:
        styles.add("blink")

    if "blink_onset" in metadata_columns and "blink_duration" in metadata_columns:
        styles.add("generic")

    return styles


def _style_windows(
    metadata_row: Mapping[str, object],
    modality: str,
    style: str,
) -> List[tuple[float, float]]:
    """Extract blink windows for a modality/style pair."""

    def _to_windows(onset_val: object, duration_val: object) -> List[tuple[float, float]]:
        onsets = ensure_list(onset_val) if onset_val is not None else []
        durations = ensure_list(duration_val) if duration_val is not None else []
        windows: List[tuple[float, float]] = []
        for onset, duration in zip(onsets, durations):
            if onset is None or duration is None:
                continue
            if pd.isna(onset) or pd.isna(duration):
                continue
            windows.append((float(onset), float(duration)))
        return windows

    if style == "generic":
        return _to_windows(metadata_row.get("blink_onset"), metadata_row.get("blink_duration"))

    mod_onset = f"blink_onset_{modality}"
    mod_duration = f"blink_duration_{modality}"
    if style == "blink":
        onset_val = metadata_row.get(mod_onset, metadata_row.get("blink_onset"))
        duration_val = metadata_row.get(mod_duration, metadata_row.get("blink_duration"))
        return _to_windows(onset_val, duration_val)

    onset_key = f"onset__{style}__{modality}"
    duration_key = f"duration__{style}__{modality}"
    return _to_windows(metadata_row.get(onset_key), metadata_row.get(duration_key))


class KinematicBlinkFeatureExtractor:
    """Compute blink kinematic features from MNE objects."""

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
        """Compute kinematic blink features for each epoch and channel.

        Parameters
        ----------
        picks : str | sequence of str | None, optional
            Channel name or list of channel names to process. ``None`` uses all
            available channels.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` containing aggregated statistics of
            kinematic metrics for each channel.

        Notes
        -----
        If an epoch contains no blinks, all kinematic statistics for that epoch
        are ``NaN``.
        """

        sfreq = self._sampling_frequency()
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=picks,
            sfreq=sfreq,
        )

        modality_map: Dict[str, str] = {ch: _infer_modality(ch, self.epochs.info) for ch in ch_names}
        modality_channels: Dict[str, List[str]] = {}
        for ch, mod in modality_map.items():
            modality_channels.setdefault(mod, []).append(ch)
        methods_by_modality: Dict[str, Sequence[str]] = {
            mod: METHODS_BY_MODALITY.get(mod, ("base",)) for mod in modality_map.values()
        }
        metrics_by_modality: Dict[str, List[str]] = {
            mod: [f"{stem}_{method}" for stem in CANONICAL_METRIC_STEMS for method in methods]
            for mod, methods in methods_by_modality.items()
        }

        metadata_cols: Sequence[str] | None = (
            tuple(self.epochs.metadata.columns) if isinstance(self.epochs.metadata, pd.DataFrame) else None
        )
        styles_by_modality: Dict[str, Set[str]] = {
            mod: _available_styles(metadata_cols, mod) for mod in set(modality_map.values())
        }
        for mod, styles in styles_by_modality.items():
            if not styles:
                styles_by_modality[mod] = {"blink"}

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod, {"blink"})):
                for metric in metrics_by_modality[mod]:
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(f"{mod}__{style}__kinematic__{metric}_{stat}__{ch}")
        columns = sorted(column_set)
        if n_epochs == 0:
            return pd.DataFrame(index=index, columns=columns, dtype=float)

        records: List[Dict[str, float]] = []
        logger.info("Computing kinematic features for %d epochs", n_epochs)

        for ei in range(n_epochs):
            metadata_row = (
                self.epochs.metadata.iloc[ei]
                if isinstance(self.epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            record: Dict[str, float] = {}
            for modality, channels in modality_channels.items():
                styles = styles_by_modality.get(modality, {"blink"})
                metrics_for_modality = metrics_by_modality[modality]
                methods = methods_by_modality[modality]
                for style in sorted(styles):
                    windows = _style_windows(metadata_row, modality, style)
                    for ch in channels:
                        per_metric: Dict[str, List[float]] = {m: [] for m in metrics_for_modality}
                        for onset_s, duration_s in windows:
                            sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                            segment = channel_data[ch]["raw"][ei, sl]
                            if segment.size == 0:
                                continue
                            metrics = compute_segment_kinematics(
                                segment,
                                sfreq,
                                methods=methods,
                                modality=modality,
                            )
                            for m in metrics_for_modality:
                                per_metric[m].append(metrics[m])
                        for metric, values in per_metric.items():
                            stats = _safe_stats(values)
                            for stat_name, value in stats.items():
                                column = f"{modality}__{style}__kinematic__{metric}_{stat_name}__{ch}"
                                record[column] = value
            records.append(record)

        df = pd.DataFrame.from_records(records, index=index, columns=columns)
        logger.debug("Kinematic feature DataFrame shape: %s", df.shape)
        return df


def compute_kinematic_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute kinematic blink features for each epoch and channel."""

    extractor = KinematicBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)
