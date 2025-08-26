"""Aggregate wavelet blink features across epochs."""

from __future__ import annotations

from typing import Sequence, Dict, List
import logging
import warnings

import mne
import numpy as np
import pandas as pd

from .features import _compute_wavelet_energies
from ..energy.helpers import _extract_blink_windows, _segment_to_samples

logger = logging.getLogger(__name__)


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

    def compute(self, picks: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Compute DWT energies for each epoch.

        Parameters
        ----------
        picks : str | list of str | None, optional
            Channel name(s) to include. When multiple channels are supplied
            they are averaged before feature extraction. ``None`` uses all
            channels.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed like ``epochs`` with columns
            ``wavelet_energy_d1`` .. ``wavelet_energy_d4``.

        Warns
        -----
        UserWarning
            If the sampling frequency is below 30 Hz, features may be
            unreliable.
        """

        if self.epochs is None:
            raise ValueError("self.epochs is required for feature computation")

        sfreq = self._sampling_frequency()
        if sfreq < 30:
            warnings.warn(
                "Frequency-domain features may be unreliable below 30 Hz", UserWarning
            )

        if picks is None:
            ch_names = self.epochs.ch_names
        elif isinstance(picks, str):
            ch_names = [picks]
        else:
            ch_names = list(picks)

        missing = [ch for ch in ch_names if ch not in self.epochs.ch_names]
        if missing:
            raise ValueError(f"Channels not found: {missing}")

        data = self.epochs.get_data(picks=ch_names).mean(axis=1)
        n_epochs, n_times = data.shape
        index = (
            self.epochs.metadata.index
            if isinstance(self.epochs.metadata, pd.DataFrame)
            else pd.RangeIndex(n_epochs)
        )

        records: List[Dict[str, float]] = []
        for ei in range(n_epochs):
            metadata_row = (
                self.epochs.metadata.iloc[ei]
                if isinstance(self.epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            windows = _extract_blink_windows(metadata_row)
            level_vals: Dict[int, List[float]] = {i: [] for i in range(1, 5)}
            for onset_s, duration_s in windows:
                sl = _segment_to_samples(onset_s, duration_s, sfreq, n_times)
                segment = data[ei, sl]
                energies = _compute_wavelet_energies(segment, sfreq)
                for lvl, val in enumerate(energies, start=1):
                    level_vals[lvl].append(val)
            record: Dict[str, float] = {}
            for lvl in range(1, 5):
                vals = level_vals[lvl]
                if not vals or np.all(np.isnan(vals)):
                    record[f"wavelet_energy_d{lvl}"] = float("nan")
                else:
                    record[f"wavelet_energy_d{lvl}"] = float(np.nanmean(vals))
            records.append(record)

        df = pd.DataFrame.from_records(
            records,
            index=index,
            columns=[f"wavelet_energy_d{i}" for i in range(1, 5)],
        )
        logger.debug("Frequency-domain feature DataFrame shape: %s", df.shape)
        return df


def aggregate_frequency_domain_features(
    epochs: mne.Epochs, picks: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Convenience function to compute frequency-domain blink features."""

    extractor = FrequencyDomainBlinkFeatureExtractor(epochs=epochs)
    return extractor.compute(picks=picks)

