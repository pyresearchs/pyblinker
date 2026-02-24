"""EEG-only unit tests for wavelet-based blink frequency features."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne


from pyblinker.blink_features.frequency_domain import (aggregate_frequency_domain_features)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot



PROJECT_ROOT = Path(__file__).resolve().parents[3]


REQUIRED_ENERGY_METRICS= {
		"zero": {
				"wavelet_energy_d1": [
						"eog__zero__energy__wavelet_energy_d1_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d1_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d1_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d2": [
						"eog__zero__energy__wavelet_energy_d2_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d2_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d2_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d3": [
						"eog__zero__energy__wavelet_energy_d3_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d3_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d3_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d4": [
						"eog__zero__energy__wavelet_energy_d4_mean__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d4_std__EOG-EEG-eog_vert_left",
						"eog__zero__energy__wavelet_energy_d4_cv__EOG-EEG-eog_vert_left",
						],
				},
		"base": {
				"wavelet_energy_d1": [
						"eog__base__energy__wavelet_energy_d1_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d1_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d1_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d2": [
						"eog__base__energy__wavelet_energy_d2_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d2_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d2_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d3": [
						"eog__base__energy__wavelet_energy_d3_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d3_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d3_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d4": [
						"eog__base__energy__wavelet_energy_d4_mean__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d4_std__EOG-EEG-eog_vert_left",
						"eog__base__energy__wavelet_energy_d4_cv__EOG-EEG-eog_vert_left",
						],
				},
		"tent": {
				"wavelet_energy_d1": [
						"eog__tent__energy__wavelet_energy_d1_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d1_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d1_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d2": [
						"eog__tent__energy__wavelet_energy_d2_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d2_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d2_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d3": [
						"eog__tent__energy__wavelet_energy_d3_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d3_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d3_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d4": [
						"eog__tent__energy__wavelet_energy_d4_mean__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d4_std__EOG-EEG-eog_vert_left",
						"eog__tent__energy__wavelet_energy_d4_cv__EOG-EEG-eog_vert_left",
						],
				},
		"half": {
				"wavelet_energy_d1": [
						"eog__half__energy__wavelet_energy_d1_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d1_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d1_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d2": [
						"eog__half__energy__wavelet_energy_d2_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d2_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d2_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d3": [
						"eog__half__energy__wavelet_energy_d3_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d3_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d3_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d4": [
						"eog__half__energy__wavelet_energy_d4_mean__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d4_std__EOG-EEG-eog_vert_left",
						"eog__half__energy__wavelet_energy_d4_cv__EOG-EEG-eog_vert_left",
						],
				},
		"peak": {
				"wavelet_energy_d1": [
						"eog__peak__energy__wavelet_energy_d1_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d1_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d1_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d2": [
						"eog__peak__energy__wavelet_energy_d2_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d2_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d2_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d3": [
						"eog__peak__energy__wavelet_energy_d3_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d3_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d3_cv__EOG-EEG-eog_vert_left",
						],
				"wavelet_energy_d4": [
						"eog__peak__energy__wavelet_energy_d4_mean__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d4_std__EOG-EEG-eog_vert_left",
						"eog__peak__energy__wavelet_energy_d4_cv__EOG-EEG-eog_vert_left",
						],
				},
		}


class TestFrequencyDomainBlinkFeaturesEEGOnly(unittest.TestCase):
	"""Validate DWT energy features per epoch for EEG-only inputs."""

	def setUp(self) -> None:  # noqa: D401
		raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
		raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
		eeg_channel = "EOG-EEG-eog_vert_left"
		raw.pick([eeg_channel])
		segmentation_config = {
				"eog": {
						"channel": eeg_channel,
						}
				}
		self.epochs = slice_raw_into_mne_epochs_refine_annot(
			raw,
			epoch_len=30.0,
			blink_label=None,
			progress_bar=False,
			segmentation_type=segmentation_config,
			)
		self.eeg_channel = eeg_channel

	def test_schema_and_rows(self) -> None:
		"""DataFrame has expected columns and indexing for first epochs."""
		df = aggregate_frequency_domain_features(
			self.epochs, picks=self.eeg_channel, progress_bar=False)

		for style in REQUIRED_ENERGY_METRICS.values():
			for metric in style.values():
				for stat_name in metric:
					self.assertIn(stat_name, df.columns)

if __name__ == "__main__":
	unittest.main()
