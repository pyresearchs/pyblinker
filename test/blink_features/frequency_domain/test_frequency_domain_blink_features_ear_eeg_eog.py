# """Combined EAR, EEG, and EOG unit tests for wavelet-based blink frequency features."""
#
# from __future__ import annotations
#
# import unittest
# from pathlib import Path
#
# import mne
#
# from pyblinker.blink_features.frequency_domain import aggregate_frequency_domain_features
# from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
#
# from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan
#
#
# PROJECT_ROOT = Path(__file__).resolve().parents[3]
#
#
# class TestFrequencyDomainBlinkFeaturesAllModalities(unittest.TestCase):
#     """Validate DWT energy features per epoch with all modalities enabled."""
#
#     def setUp(self) -> None:  # noqa: D401
#         raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
#         raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
#         channels = ["EAR-avg_ear", "EEG-E8", "EOG-EEG-eog_vert_left"]
#         raw.pick(channels)
#         segmentation_config = {
#             "ear": {
#                 "channel": "EAR-avg_ear",
#                 "seg_type": "threshold_interpolation",
#                 "threshold": 0.260,
#                 "annotation_time_unit": "seconds",
#                 "max_extension": 0.35,
#                 "extension_step": 0.05,
#                 "padding": 0.05,
#                 "extend_before": True,
#                 "extend_after": True,
#             },
#             "eeg": {"channel": "EEG-E8"},
#             "eog": {"channel": "EOG-EEG-eog_vert_left"},
#         }
#         self.epochs = slice_raw_into_mne_epochs_refine_annot(
#             raw,
#             epoch_len=30.0,
#             blink_label=None,
#             progress_bar=False,
#             segmentation_type=segmentation_config,
#         )
#         self.channels = channels
#
#     def test_schema_and_rows(self) -> None:
#         """DataFrame has expected columns and indexing for first epochs."""
#         df = aggregate_frequency_domain_features(
#             self.epochs, picks=self.channels, progress_bar=False
#         )
#         expected_cols = [
#             "ep",
#             *[
#                 f"wavelet_energy_d{i}_{modality}"
#                 for modality in ("ear", "eeg", "eog")
#                 for i in range(1, 5)
#             ],
#         ]
#         assert_df_has_columns(
#             self,
#             df,
#             expected_cols,
#         )
#         self.assertEqual(len(df), len(self.epochs))
#         for idx in range(4):
#             self.assertIn(idx, df.index)
#             self.assertEqual(df.iloc[idx]["ep"], idx)
#             assert_numeric_or_nan(self, df.iloc[idx].drop(labels="ep"))
#         # Ensure modality-specific energies differ (no channel averaging)
#         self.assertTrue(
#             (df["wavelet_energy_d2_ear"] != df["wavelet_energy_d2_eeg"]).any()
#             or (df["wavelet_energy_d2_ear"].isna() & df["wavelet_energy_d2_eeg"].isna()).all(),
#         )
#
#
# if __name__ == "__main__":
#     unittest.main()
