"""Blink kinematic feature calculations based on epoch metadata.
All feature calculations rely only on blink onset and blink duration stored in the metadata.
This design intentionally decouples feature extraction from how blink boundaries are defined.

As a result, users should have full flexibility to define blink onset and duration according to their needs.
See pyblinker/segmentation/refinement.py
"""

from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Sequence

import mne
import pandas as pd

from .._core_blink import CANONICAL_METRIC_STEMS
from .per_blink import compute_segment_kinematics
from ..energy.helpers import extract_blink_windows, segment_to_samples, _safe_stats
from ...utils.epoch_utils import build_metric_stat_columns, resolve_channels
from ...utils.channel_utils import normalize_picks
from ..utils.aggregation import prepare_epoch_channel_data

logger = get_logger(__name__)

# Base-method metric and statistic names (kinematics defaults to base per modality)
_METRICS = tuple(f"{stem}_base" for stem in CANONICAL_METRIC_STEMS)
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

        # Resolve channel names and infer modality per channel to avoid defaulting to EEG.
        resolved_picks = resolve_channels(
            self.epochs, picks, default=lambda ep: normalize_picks(ep.ch_names)
        )
        sfreq = self._sampling_frequency()
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=resolved_picks,
            sfreq=sfreq,
        )

        modalities: List[str] = [_infer_modality(ch, self.epochs.info) for ch in ch_names]
        columns = build_metric_stat_columns(ch_names, _METRICS, _STATS)
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
            for ch, modality in zip(ch_names, modalities):
                windows = extract_blink_windows(metadata_row, ch, ei)
                per_metric: Dict[str, List[float]] = {m: [] for m in _METRICS}
                for onset_s, duration_s in windows:
                    sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                    segment = channel_data[ch][ei, sl]
                    if segment.size == 0:
                        continue
                    metrics = compute_segment_kinematics(segment, sfreq, modality=modality)
                    for m in _METRICS:
                        per_metric[m].append(metrics[m])
                for metric, values in per_metric.items():
                    stats = _safe_stats(values)
                    for stat_name, value in stats.items():
                        record[f"{metric}_{stat_name}_{ch}"] = value
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
