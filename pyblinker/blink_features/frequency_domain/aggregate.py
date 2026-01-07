"""Aggregate wavelet blink features across epochs."""

from __future__ import annotations
from typing import Dict, List, Mapping, Sequence

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyblinker.logging import get_logger

from .features import _compute_wavelet_energies
from ..energy.helpers import extract_blink_windows, segment_to_samples
from ..utils.aggregation import prepare_epoch_channel_data

logger = get_logger(__name__)


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


def _compute_epoch_wavelet_record(
    *,
    epoch_index: int,
    metadata_row: pd.Series | Mapping[str, object],
    sfreq: float,
    n_times: int,
    channel_data: Dict[str, np.ndarray],
    modality_channels: Dict[str, List[str]],
    modality_order: List[str],
) -> Dict[str, float]:
    """Compute modality-aggregated wavelet energies for a single epoch."""

    record: Dict[str, float] = {}
    for modality in modality_order:
        modality_levels: Dict[int, List[float]] = {i: [] for i in range(1, 5)}
        for ch in modality_channels[modality]:
            windows = extract_blink_windows(metadata_row, ch, epoch_index)
            level_vals: Dict[int, List[float]] = {i: [] for i in range(1, 5)}
            for onset_s, duration_s in windows:
                sl = segment_to_samples(onset_s, duration_s, sfreq, n_times)
                segment = channel_data[ch]["raw"][epoch_index, sl]
                energies = _compute_wavelet_energies(segment, sfreq)
                for lvl, val in enumerate(energies, start=1):
                    level_vals[lvl].append(val)
            for lvl in range(1, 5):
                vals = level_vals[lvl]
                if not vals or np.all(np.isnan(vals)):
                    modality_levels[lvl].append(float("nan"))
                else:
                    modality_levels[lvl].append(float(np.nanmean(vals)))
        for lvl in range(1, 5):
            vals = modality_levels[lvl]
            key = f"wavelet_energy_d{lvl}_{modality}"
            if not vals or np.all(np.isnan(vals)):
                record[key] = float("nan")
            else:
                record[key] = float(np.nanmean(vals))
    return record


class FrequencyDomainBlinkFeatureExtractor:
    """Compute wavelet-energy blink features from MNE objects."""

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

    def compute(
        self,
        picks: str | Sequence[str] | None = None,
        *,
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Compute DWT energies for each epoch.

        Parameters
        ----------
        picks : str | list of str | None, optional
            Channel name(s) to include. When multiple channels are supplied
            they are processed per modality/channel. ``None`` uses all channels.
        progress_bar : bool, optional
            Display a progress bar during epoch processing. Defaults to
            ``True``.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` with columns ``ep`` and
            ``wavelet_energy_d1_{modality}`` .. ``wavelet_energy_d4_{modality}``
            for each detected modality.

        Notes
        -----
        A warning is logged when the sampling frequency is below 30 Hz
        because wavelet features become unreliable at low rates.
        """

        sfreq = self._sampling_frequency()
        ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
            epochs=self.epochs,
            picks=picks,
            sfreq=sfreq,
        )

        modality_channels: Dict[str, List[str]] = {}
        modality_order: List[str] = []
        for ch in ch_names:
            modality = _infer_modality(ch, self.epochs.info)
            if modality not in modality_channels:
                modality_channels[modality] = []
                modality_order.append(modality)
            modality_channels[modality].append(ch)

        records: List[Dict[str, float]] = []
        for ei in tqdm(
            range(n_epochs),
            desc="Wavelet energies",
            unit="epoch",
            disable=not progress_bar,
        ):
            metadata_row = (
                self.epochs.metadata.iloc[ei]
                if isinstance(self.epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            record = _compute_epoch_wavelet_record(
                epoch_index=ei,
                metadata_row=metadata_row,
                sfreq=sfreq,
                n_times=n_times,
                channel_data=channel_data,
                modality_channels=modality_channels,
                modality_order=modality_order,
            )
            record["ep"] = index[ei]
            records.append(record)

        df = pd.DataFrame.from_records(
            records,
            index=index,
        )
        logger.debug("Frequency-domain feature DataFrame shape: %s", df.shape)
        return df


def aggregate_frequency_domain_features(
    epochs: mne.Epochs,
    picks: str | Sequence[str] | None = None,
    *,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Convenience function to compute frequency-domain blink features.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs instance containing the blink data.
    picks : str | list of str | None, optional
        Channel name(s) to include. When multiple channels are provided they
        are aggregated per modality after computing channel-level energies.
        ``None`` uses all channels.
    progress_bar : bool, optional
        Display a progress bar during epoch processing. Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an ``ep`` column denoting the epoch index and
        wavelet-energy features.
    """

    extractor = FrequencyDomainBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks, progress_bar=progress_bar)
