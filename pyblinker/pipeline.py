"""Main pipeline entry for feature extraction."""

import logging
from typing import Iterable, Dict, Sequence, Optional

import pandas as pd
import mne

from .blink_features.blink_events.event_features import (
    aggregate_blink_event_features
)
from .blink_features.morphology import aggregate_morphology_features
from .blink_features.kinematics import aggregate_kinematic_features
from .blink_features.energy import aggregate_energy_features
from .blink_features.open_eye import aggregate_open_eye_features
from .blink_features.ear_metrics import aggregate_ear_features
from .blink_features.waveform_features import aggregate_waveform_features
from .blink_features.frequency_domain import aggregate_frequency_domain_features
from .blink_features.blink_events.classification import aggregate_classification_features

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(
    blinks: Iterable[Dict[str, int]],
    sfreq: float,
    epoch_len: float,
    n_epochs: int,
    features: Sequence[str] | None = None,
    raw_segments: Optional[Sequence[mne.io.BaseRaw]] = None,
) -> pd.DataFrame:
    """Extract blink features using provided blink annotations.

    Parameters
    ----------
    blinks : Iterable[Dict[str, int]]
        Blink annotations for each detected blink. Each record must contain
        ``epoch_index`` (``int``), ``epoch_signal`` (1D array),
        ``refined_start_frame`` (``int``), ``refined_peak_frame`` (``int``),
        and ``refined_end_frame`` (``int``). These fields match the format
        used throughout the feature modules and specify the blink location
        relative to its epoch.
    sfreq : float
        Sampling frequency of the recording.
    epoch_len : float
        Length of each epoch in seconds.
    n_epochs : int
        Total number of epochs.
    features : Sequence[str] | None, optional
        Feature groups to compute. Values from
        :func:`aggregate_blink_event_features` (``"blink_count"``, ``"blink_rate"``,
        ``"ibi"``), ``"morphology``, ``"kinematics``, ``"energy``, ``"open_eye``,
        ``"ear"``, ``"waveform`` and ``"classification"`` are recognized. ``None`` computes all
        available features.
    raw_segments : Sequence[mne.io.BaseRaw] | None, optional
        Collection of 30-second raw segments with annotations. Required when
        ``"blink_interval_dist"`` is among ``features``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated features per epoch.
    """
    logger.info("Starting feature extraction")

    df_events = aggregate_blink_event_features(
        blinks, sfreq, epoch_len, n_epochs, features
    )

    if features is None or "blink_interval_dist" in features:
        if raw_segments is None:
            raise ValueError(
                "raw_segments must be provided when blink_interval_dist is requested"
            )
        df_interval = aggregate_blink_interval_distribution(raw_segments, blink_label=None)
        df_events = pd.concat([df_events, df_interval], axis=1)

    if features is None or "ear" in features:
        df_ear = aggregate_ear_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_ear], axis=1)

    if features is None or "classification" in features:
        df_cls = aggregate_classification_features(
            blinks, sfreq, epoch_len, n_epochs
        )
        df_events = pd.concat([df_events, df_cls], axis=1)

    if features is None or "kinematics" in features:
        df_kin = aggregate_kinematic_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_kin], axis=1)

    if features is None or "energy" in features:
        df_energy = aggregate_energy_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_energy], axis=1)

    if features is None or "open_eye" in features:
        df_open = aggregate_open_eye_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_open], axis=1)

    if features is None or "frequency" in features:
        df_freq = aggregate_frequency_domain_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_freq], axis=1)

    if features is None or "waveform" in features:
        df_wave = aggregate_waveform_features(blinks, sfreq, n_epochs)
        df_events = pd.concat([df_events, df_wave], axis=1)

    if features is None or "morphology" in features:
        df_morph = aggregate_morphology_features(blinks, sfreq, n_epochs)
        df = pd.concat([df_events, df_morph], axis=1)
    else:
        df = df_events

    logger.info("Finished feature extraction")
    return df
