"""End-to-end tutorial: aggregate blink features for EAR, EEG, and EOG channels.

This tutorial demonstrates how to compute and combine epoch-level blink features
across all supported modalities in a single workflow. It mirrors the feature
calculation patterns validated in the unit tests and produces one consolidated
``pandas.DataFrame`` per epoch.

Features calculated
-------------------
The script aggregates the following feature families:

1. Blink count / blink event features
   * ``blink_total_<modality>`` and ``blink_rate_<modality>``
   * ``ibi_<channel>`` (inter-blink interval statistics)
2. Energy features
   * ``blink_signal_energy``
   * ``teager_kaiser_energy``
   * ``blink_line_length``
   * ``blink_velocity_integral``
3. Frequency-domain features
   * ``wavelet_energy_d1`` / ``wavelet_energy_d2`` / ``wavelet_energy_d3`` /
     ``wavelet_energy_d4``
4. Morphology features
   * Blink duration and timing metrics (style-specific variants)
   * Peak timing / amplitude metrics
   * Shape metrics (e.g., half-width and rise/fall timing where applicable)
5. Kinematic features
   * Amplitude-velocity ratios
   * Blink velocity summaries
   * Inter-blink max-velocity metrics
   * Style-window velocity/slope/acceleration summaries

Channel selection in this example
---------------------------------
* EAR: ``EAR-avg_ear``
* EEG: ``EEG-E8``
* EOG: ``EOG-EEG-eog_vert_left``

References
----------
The computation flow in this tutorial aligns with tests that assert the
expected feature schemas:

* ``test/blink_features/test_blink_features_ear_eeg_eog.py``
* ``test/blink_features/blink_events/test_aggregate_event_features.py``
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from pyblinker.blink_features.blink_events.event_features import (
    aggregate_blink_event_features,
)
from pyblinker.blink_features.energy import compute_energy_features
from pyblinker.blink_features.frequency_domain import (
    aggregate_frequency_domain_features,
)
from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
)
from pyblinker.blink_features.morphology import compute_epoch_morphology_features
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

EAR_CHANNEL = "EAR-avg_ear"
EEG_CHANNEL = "EEG-E8"
EOG_CHANNEL = "EOG-EEG-eog_vert_left"
PICKS = [EAR_CHANNEL, EEG_CHANNEL, EOG_CHANNEL]

SEGMENTATION_CONFIG = {
    "ear": {
        "channel": EAR_CHANNEL,
        "seg_type": "threshold_interpolation",
        "threshold": 0.260,
        "annotation_time_unit": "seconds",
        "max_extension": 0.35,
        "extension_step": 0.05,
        "padding": 0.05,
        "extend_before": True,
        "extend_after": True,
    },
    "eeg": {"channel": EEG_CHANNEL, "seg_type": "base"},
    "eog": {"channel": EOG_CHANNEL, "seg_type": "base"},
}


def _load_sample_raw() -> mne.io.BaseRaw:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "test" / "test_files" / "ear_eog_raw.fif"
    return mne.io.read_raw_fif(raw_path, preload=True, verbose=False)


def run_tutorial() -> pd.DataFrame:
    """Compute and merge blink feature families for EAR, EEG, and EOG."""
    raw = _load_sample_raw()
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        progress_bar=False,
        segmentation_type=SEGMENTATION_CONFIG,
    )

    blink_event_df = aggregate_blink_event_features(epochs, picks=PICKS)
    energy_df = compute_energy_features(epochs=epochs, picks=PICKS)
    frequency_df = aggregate_frequency_domain_features(
        epochs=epochs,
        picks=PICKS,
        progress_bar=False,
    )
    kinematic_df = KinematicBlinkFeatureExtractor(epochs=epochs).compute(picks=PICKS)
    morphology_df = compute_epoch_morphology_features(epochs=epochs, picks=PICKS)

    return pd.concat(
        [
            blink_event_df,
            energy_df,
            frequency_df,
            kinematic_df,
            morphology_df,
        ],
        axis=1,
    )


if __name__ == "__main__":
    df_features = run_tutorial()
    output_path = Path(__file__).resolve().parents[1] / "complete_feature.xlsx"
    df_features.to_excel(output_path, index=False)
    print("Combined feature DataFrame shape:", df_features.shape)
    print("Sample columns:")
    print(df_features.columns[:20].tolist())
