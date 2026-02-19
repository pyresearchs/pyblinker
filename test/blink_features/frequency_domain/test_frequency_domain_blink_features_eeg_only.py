# """EEG-only unit tests for wavelet-based blink frequency features."""
#
# from __future__ import annotations
#
# import unittest
# from pathlib import Path
#
# import mne
# import numpy as np
# import pandas as pd
#
# from pyblinker.blink_features.frequency_domain import (
#     FrequencyDomainBlinkFeatureExtractor,
#     aggregate_frequency_domain_features,
# )
# from pyblinker.blink_features.frequency_domain.features import _compute_wavelet_energies
# from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
#
# from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan
#
#
# PROJECT_ROOT = Path(__file__).resolve().parents[3]
#
#
# class TestFrequencyDomainBlinkFeaturesEEGOnly(unittest.TestCase):
#     """Validate DWT energy features per epoch for EEG-only inputs."""
#
#     def setUp(self) -> None:  # noqa: D401
#         raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
#         raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
#         eeg_channel = "EEG-E8"
#         raw.pick([eeg_channel])
#         segmentation_config = {
#             "eeg": {
#                 "channel": eeg_channel,
#             }
#         }
#         self.epochs = slice_raw_into_mne_epochs_refine_annot(
#             raw,
#             epoch_len=30.0,
#             blink_label=None,
#             progress_bar=False,
#             segmentation_type=segmentation_config,
#         )
#         self.eeg_channel = eeg_channel
#
#     def test_schema_and_rows(self) -> None:
#         """DataFrame has expected columns and indexing for first epochs."""
#         df = aggregate_frequency_domain_features(
#             self.epochs, picks=self.eeg_channel, progress_bar=False
#         )
#         assert_df_has_columns(
#             self,
#             df,
#             ["ep"] + [f"wavelet_energy_d{i}_eeg" for i in range(1, 5)],
#         )
#         self.assertEqual(len(df), len(self.epochs))
#         for idx in range(4):
#             self.assertIn(idx, df.index)
#             self.assertEqual(df.iloc[idx]["ep"], idx)
#             assert_numeric_or_nan(self, df.iloc[idx].drop(labels="ep"))
#
#     def test_requires_mne_object(self) -> None:
#         """Extractor must have epochs or raw defined."""
#         extractor = FrequencyDomainBlinkFeatureExtractor()
#         with self.assertRaises(ValueError):
#             extractor.compute()
#
#     def test_low_sampling_frequency_warning(self) -> None:
#         """Log a warning and drop Nyquist-touching levels when fs < 30 Hz."""
#         epochs = self.epochs.copy().resample(20.0, npad="auto")
#         with self.assertLogs("pyblinker", level="WARNING") as cm:
#             df = aggregate_frequency_domain_features(
#                 epochs, picks=self.eeg_channel, progress_bar=False
#             )
#         self.assertTrue(
#             any(
#                 "Frequency-domain features may be unreliable below 30 Hz" in message
#                 for message in cm.output
#             ),
#             msg="Expected warning log missing",
#         )
#         self.assertTrue(df["wavelet_energy_d1_eeg"].isna().all())
#         assert_df_has_columns(
#             self, df, ["ep"] + [f"wavelet_energy_d{i}_eeg" for i in range(2, 5)]
#         )
#
#     def test_no_blink_epochs(self) -> None:
#         """Epochs without blinks yield NaN energies."""
#         df = aggregate_frequency_domain_features(
#             self.epochs, picks=self.eeg_channel, progress_bar=False
#         )
#         no_blink_idx = self.epochs.metadata.index[
#             self.epochs.metadata["blink_onset"].isna()
#         ][0]
#         self.assertTrue(
#             df.loc[no_blink_idx, [f"wavelet_energy_d{i}_eeg" for i in range(1, 5)]].isna().all()
#         )
#
#     def test_multiple_eeg_channels_aggregated_by_modality(self) -> None:
#         """Energies are computed per channel then aggregated by modality."""
#         sfreq = 100.0
#         n_times = 200
#         t = np.arange(n_times) / sfreq
#         sine = np.sin(2 * np.pi * 5 * t)  # non-zero energy channel
#         zeros = np.zeros_like(sine)  # zero-energy channel
#         data = np.stack([sine, zeros])[np.newaxis, ...]
#
#         info = mne.create_info(ch_names=["EEG-1", "EEG-2"], sfreq=sfreq, ch_types="eeg")
#         events = np.array([[0, 0, 1]])
#         metadata = pd.DataFrame({"blink_onset": [0.0], "blink_duration": [2.0]})
#         epochs = mne.EpochsArray(
#             data, info, events=events, event_id={"blink": 1}, metadata=metadata, verbose=False
#         )
#
#         df = aggregate_frequency_domain_features(
#             epochs, picks=["EEG-1", "EEG-2"], progress_bar=False
#         )
#         energies = df.iloc[0][[f"wavelet_energy_d{i}_eeg" for i in range(1, 5)]].to_numpy()
#
#         ch1_energy = _compute_wavelet_energies(sine, sfreq)
#         ch2_energy = _compute_wavelet_energies(zeros, sfreq)
#         expected_modal = np.nanmean(np.vstack([ch1_energy, ch2_energy]), axis=0)
#         np.testing.assert_allclose(energies, expected_modal, rtol=1e-6, atol=1e-6)
#
#         averaged_signal = (sine + zeros) / 2.0
#         averaged_energy = _compute_wavelet_energies(averaged_signal, sfreq)
#         self.assertFalse(np.allclose(energies, averaged_energy, rtol=1e-6, atol=1e-6))
#
#
# if __name__ == "__main__":
#     unittest.main()
