"""Combined EAR, EEG, and EOG unit tests for wavelet-based blink frequency features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne

from pyblinker.blink_features.frequency_domain import (
    FrequencyDomainBlinkFeatureExtractor,
    aggregate_frequency_domain_features,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestFrequencyDomainBlinkFeaturesAllModalities(unittest.TestCase):
    """Validate DWT energy features per epoch with all modalities enabled."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        channels = ["EAR-avg_ear", "EEG-E8", "EOG-EEG-eog_vert_left"]
        raw.pick(channels)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )
        self.channels = channels

    def test_schema_and_rows(self) -> None:
        """DataFrame has expected columns and indexing for first epochs."""
        df = aggregate_frequency_domain_features(
            self.epochs, picks=self.channels, progress_bar=False
        )
        assert_df_has_columns(
            self,
            df,
            ["ep"] + [f"wavelet_energy_d{i}" for i in range(1, 5)],
        )
        self.assertEqual(len(df), len(self.epochs))
        for idx in range(4):
            self.assertIn(idx, df.index)
            self.assertEqual(df.iloc[idx]["ep"], idx)
            assert_numeric_or_nan(self, df.iloc[idx].drop(labels="ep"))

    def test_requires_mne_object(self) -> None:
        """Extractor must have epochs or raw defined."""
        extractor = FrequencyDomainBlinkFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.compute()

    def test_low_sampling_frequency_warning(self) -> None:
        """Log a warning and drop Nyquist-touching levels when fs < 30 Hz."""
        epochs = self.epochs.copy().resample(20.0, npad="auto")
        with self.assertLogs("pyblinker", level="WARNING") as cm:
            df = aggregate_frequency_domain_features(
                epochs, picks=self.channels, progress_bar=False
            )
        self.assertTrue(
            any(
                "Frequency-domain features may be unreliable below 30 Hz" in message
                for message in cm.output
            ),
            msg="Expected warning log missing",
        )
        self.assertTrue(df["wavelet_energy_d1"].isna().all())
        assert_df_has_columns(
            self, df, ["ep"] + [f"wavelet_energy_d{i}" for i in range(2, 5)]
        )

    def test_no_blink_epochs(self) -> None:
        """Epochs without blinks yield NaN energies."""
        df = aggregate_frequency_domain_features(
            self.epochs, picks=self.channels, progress_bar=False
        )
        no_blink_idx = self.epochs.metadata.index[
            self.epochs.metadata["blink_onset"].isna()
        ][0]
        self.assertTrue(
            df.loc[no_blink_idx, [f"wavelet_energy_d{i}" for i in range(1, 5)]].isna().all()
        )


if __name__ == "__main__":
    unittest.main()
