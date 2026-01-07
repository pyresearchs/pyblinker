"""Aggregate blink morphology features from :class:`mne.Epochs`."""
from __future__ import annotations
from pyblinker.logging import get_logger

from typing import Dict, List, Sequence, Set

import mne
import pandas as pd

from .core_metrics import MORPHOLOGY_METRIC_STEMS
from .per_blink import compute_blink_waveform_metrics
from ..energy.helpers import extract_blink_windows, segment_to_samples, _safe_stats
from ..utils.aggregation import prepare_epoch_channel_data
from ...utils.epoch_utils import resolve_channels

logger = get_logger(__name__)

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


def _default_morphology_channels(epochs: mne.Epochs) -> List[str]:
    """Select default EAR/EOG channels when ``picks`` are unspecified."""

    ch_names = [
        ch for ch in epochs.ch_names if "EOG" in ch.upper() or "EAR" in ch.upper()
    ]
    if not ch_names:
        raise ValueError("No default EAR/EOG channels found")
    return ch_names


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

        column_set: Set[str] = set()
        for mod, channels in modality_channels.items():
            for style in sorted(styles_by_modality.get(mod, {"base"})):
                metrics_for_style = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]
                metrics_for_style.append("duration")
                for metric in metrics_for_style:
                    for stat in _STATS:
                        for ch in channels:
                            column_set.add(
                                f"{mod}__{style}__morphology__{metric}_{stat}__{ch}"
                            )
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
                styles = styles_by_modality.get(modality, {"base"})
                for style in sorted(styles):
                    metrics_for_style = [f"{stem}_{style}" for stem in MORPHOLOGY_METRIC_STEMS]
                    metrics_for_style_with_duration = metrics_for_style + ["duration"]
                    for ch in channels:
                        per_metric: Dict[str, List[float]] = {
                            metric: [] for metric in metrics_for_style_with_duration
                        }
                        windows = extract_blink_windows(metadata_row, ch, ei)
                        for onset_s, duration_s in windows:
                            sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                            segment = channel_data[ch]["raw"][ei, sl]
                            if segment.size == 0:
                                continue
                            metrics = compute_blink_waveform_metrics(
                                segment,
                                sfreq,
                                methods=("base",),
                                modality=modality,
                            )
                            for metric_name in metrics_for_style:
                                per_metric[metric_name].append(
                                    metrics.get(metric_name, float("nan"))
                                )
                            per_metric["duration"].append(duration_s)
                        for metric, values in per_metric.items():
                            stats = _safe_stats(values)
                            for stat_name, value in stats.items():
                                column = (
                                    f"{modality}__{style}__morphology__{metric}_{stat_name}__{ch}"
                                )
                                record[column] = value
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
