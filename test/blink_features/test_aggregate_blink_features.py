# """End-to-end validation of blink feature aggregation across modalities.
#
# This module exercises :func:`aggregate_blink_features` to ensure it stitches
# EEG, EOG, and EAR blink metrics into a single table. The derived features span
# blink events, energy, frequency-domain, kinematic, morphology, and waveform
# families and rely on the same helpers used in production code, namely
#
# * :func:`aggregate_blink_event_features`
# * :func:`compute_energy_features`
# * :func:`aggregate_frequency_domain_features`
# * :func:`compute_kinematic_features`
# * :func:`compute_epoch_morphology_features`
# * :func:`_compute_waveform_epoch_features`
#
# The regression tests assert schema expectations, data typing, deterministic
# column ordering, CSV merging, and robustness when modalities are missing.
# """
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
# from pyblinker.blink_features.aggregate import aggregate_blink_features
# from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
# from test.segment_config import build_segment_config
#
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
#
#
# class TestAggregateBlinkFeatures(unittest.TestCase):
#     """Validate feature aggregation across modalities and feature families."""
#
#     def setUp(self) -> None:  # noqa: D401 - inherited docstring is sufficient
#         raw_path = (
#             PROJECT_ROOT
#             / "test"
#             / "test_files"
#             / "ear_eog_raw.fif"
#         )
#         raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
#         segmentation_config = build_segment_config(raw)
#         self.epochs = slice_raw_into_mne_epochs_refine_annot(
#             raw,
#             epoch_len=30.0,
#             blink_label=None,
#             progress_bar=False,
#             segmentation_type=segmentation_config,
#         )
#         self.csv_path = (
#             PROJECT_ROOT
#             / "test"
#             / "test_files"
#             / "ear_eog_blink_count_epoch.csv"
#         )
#
#     def _run_aggregate(self, epochs) -> pd.DataFrame:
#         return aggregate_blink_features(
#             epochs,
#             epoch_len=30.0,
#             blink_label=None,
#             progress_bar=False,
#             include_modalities=("EEG", "EOG", "EAR"),
#             feature_families=("events", "energy", "freq", "kin", "morph", "wave"),
#             metadata_csv_path=self.csv_path,
#         )
#
#     def test_aggregate_returns_dataframe(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         self.assertIsInstance(df, pd.DataFrame)
#         self.assertEqual("epoch", df.index.name)
#         self.assertEqual(len(df), len(self.epochs))
#
#     def test_epoch_column_present_and_integer(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         self.assertIn("epoch", df.columns)
#         self.assertTrue(np.issubdtype(df["epoch"].dtype, np.integer))
#         self.assertTrue(df["epoch"].is_monotonic_increasing)
#
#     def test_expected_columns_present(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         try:
#             df.to_excel("output.xlsx", index=False)
#         except ModuleNotFoundError:  # pragma: no cover - environment-specific
#             df.to_csv("output.xlsx.csv", index=False)
#         expected_some = [
#             "EOG__events__blink_total",
#             "EAR__energy__blink_signal_energy_mean_EAR-avg_ear",
#             "EEG__freq__wavelet_energy_d1_eeg",
#             "EEG__wave__peak_time_tent_EEG-E8",
#             "EEG__wave__pos_amp_vel_ratio_tent_EEG-E8",
#         ]
#         for col in expected_some:
#             self.assertIn(col, df.columns)
#
#     def test_csv_data_merged(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         self.assertTrue(
#             any(col.startswith("META__events__blink_count") for col in df.columns)
#         )
#
#     def test_csv_totals_match_output(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         meta_col = "META__events__blink_count"
#         self.assertIn(meta_col, df.columns)
#
#         csv_counts = pd.read_csv(self.csv_path).set_index("epoch_id")
#         csv_counts = csv_counts.reindex(df.index)["blink_count"].fillna(0.0)
#         meta_series = df[meta_col].fillna(0.0)
#
#         pd.testing.assert_series_equal(
#             meta_series.astype(float),
#             csv_counts.astype(float),
#             check_names=False,
#         )
#         self.assertAlmostEqual(float(meta_series.sum()), float(csv_counts.sum()))
#
#     def test_numeric_columns_and_uniqueness(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         self.assertTrue(df.columns.is_unique)
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         self.assertGreaterEqual(len(numeric_cols), 5)
#
#     def test_missing_modality_is_skipped(self) -> None:
#         epochs = self.epochs.copy().pick(
#             ["EOG-EEG-eog_vert_left", "EAR-avg_ear"], verbose=False
#         )
#         df = self._run_aggregate(epochs)
#         self.assertNotIn("EEG__events__blink_total", df.columns)
#         self.assertIn("EOG__events__blink_total", df.columns)
#         self.assertIn("EAR__events__blink_total", df.columns)
#
#     def test_columns_sorted(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         self.assertEqual(df.columns[0], "epoch")
#         self.assertListEqual(list(df.columns[1:]), sorted(df.columns[1:]))
#
#     def test_no_all_nan_columns(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         self.assertTrue(all(~df[col].isna().all() for col in df.columns))
#
#     def test_all_channels_and_feature_families_present(self) -> None:
#         df = self._run_aggregate(self.epochs)
#         for ch_name in self.epochs.info["ch_names"]:
#             self.assertTrue(
#                 any(ch_name in col for col in df.columns),
#                 msg=f"Expected at least one feature column for channel {ch_name}",
#             )
#
#         families = ("events", "energy", "freq", "kin", "morph", "wave")
#         for family in families:
#             self.assertTrue(
#                 any(
#                     col.split("__")[1] == family
#                     for col in df.columns
#                     if "__" in col
#                 ),
#                 msg=f"Expected feature family '{family}' in aggregated output",
#             )
#
#
# if __name__ == "__main__":  # pragma: no cover
#     unittest.main()
