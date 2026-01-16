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

from .core_metrics import KINEMATIC_METRIC_STEMS, KINEMATIC_METRICS_NO_STYLE
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
        if not col.startswith("onset__") or not col.endswith(suffix):
            continue
        style = col[len("onset__") : -len(suffix)]
        if "sample" in style.lower():
            continue
        duration_key = f"duration__{style}__{modality}"
        if duration_key in metadata_columns:
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
    onsets = ensure_list(metadata_row.get(onset_key)) if metadata_row.get(onset_key) is not None else []
    durations = (
        ensure_list(metadata_row.get(duration_key)) if metadata_row.get(duration_key) is not None else []
    )
    windows: List[tuple[float, float]] = []
    for onset, duration in zip(onsets, durations):
        if onset is None or duration is None:
            continue
        if pd.isna(onset) or pd.isna(duration):
            continue
        windows.append((float(onset), float(duration)))
    return windows


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
        metadata_cols: Sequence[str] | None = (
            tuple(self.epochs.metadata.columns) if isinstance(self.epochs.metadata, pd.DataFrame) else None
        )
        styles_by_modality: Dict[str, Set[str]] = {}
        # fallback_styles: Dict[str, bool] = {}
        for mod in set(modality_map.values()):
            styles = _available_styles(metadata_cols, mod)
            # fallback_styles[mod] = not styles
            styles_by_modality[mod] = styles

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod, {"base"})):
                metrics_for_style = [
                    stem if stem in KINEMATIC_METRICS_NO_STYLE else f"{stem}_{style}"
                    for stem in KINEMATIC_METRIC_STEMS
                ]
                for metric in metrics_for_style:
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
                styles = styles_by_modality.get(modality, {"base"})
                # use_fallback = fallback_styles.get(modality, False)
                for style in sorted(styles):
                    metrics_for_style = [
                        stem if stem in KINEMATIC_METRICS_NO_STYLE else f"{stem}_{style}"
                        for stem in KINEMATIC_METRIC_STEMS
                    ]
                    windows = _style_windows(metadata_row, modality, style)
                    # if use_fallback and not windows:
                    #     onset_key = f"blink_onset_{modality}"
                    #     duration_key = f"blink_duration_{modality}"
                    #     onsets = ensure_list(metadata_row.get(onset_key)) if metadata_row.get(onset_key) is not None else []
                    #     durations = (
                    #         ensure_list(metadata_row.get(duration_key))
                    #         if metadata_row.get(duration_key) is not None
                    #         else []
                    #     )
                    #     windows = [
                    #         (float(o), float(d))
                    #         for o, d in zip(onsets, durations)
                    #         if o is not None and d is not None and not (pd.isna(o) or pd.isna(d))
                    #     ]
                    for ch in channels:
                        per_metric: Dict[str, List[float]] = {m: [] for m in metrics_for_style}
                        for onset_s, duration_s in windows:
                            sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                            segment = {
                                "raw": channel_data[ch]["raw"][ei, sl],
                                "dx1": channel_data[ch]["dx1"][ei, sl],
                                "dx2": channel_data[ch]["dx2"][ei, sl],
                            }
                            # if segment["raw"].size == 0:
                            #     continue
                            metrics = compute_segment_kinematics(
                                segment,
                                sfreq,
                                method=style,
                                modality=modality,
                            )
                            for m in metrics_for_style:
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
