"""Integration test for full segment-level feature pipeline.

This test combines frequency-domain metrics, time-domain energy and complexity
features, and averaged blink properties into a single DataFrame for each raw
segment.  It exercises the pipeline with ``run_fit`` disabled and enabled to
ensure both paths complete successfully.
"""
import logging
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.features.blink_events import generate_blink_dataframe
from pyblinker.features.frequency_domain.segment_features import compute_frequency_domain_features
from pyblinker.features.energy_complexity.segment_features import compute_time_domain_features
from pyblinker.segment_blink_properties import compute_segment_blink_properties

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSegmentRawFeaturePipeline(unittest.TestCase):
    """Validate feature aggregation across processing stages."""

    def setUp(self) -> None:
        """Load raw data and prepare blink annotations.

        Parameters
        ----------
        None
        """
        raw_path = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        self.segments, _, _, _ = slice_raw_into_epochs(
            raw, epoch_len=30.0, blink_label=None
        , progress_bar=False)
        self.sfreq = raw.info["sfreq"]
        self.blink_df = generate_blink_dataframe(
            self.segments, channel="EEG-E8", blink_label=None, progress_bar=False
        )
        self.params = {
            "base_fraction": 0.5,
            "shut_amp_fraction": 0.9,
            "p_avr_threshold": 3,
            "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
        }
        csv_path = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_blink_count_epoch.csv"
        self.expected_counts = pd.read_csv(csv_path)["blink_count"].tolist()

    def _build_dataframe(self, *, run_fit: bool) -> pd.DataFrame:
        """Construct a combined feature table.

        Parameters
        ----------
        run_fit : bool
            Whether to run the blink fitting stage.

        Returns
        -------
        pandas.DataFrame
            Table indexed by ``seg_id`` containing spectral metrics, time-domain
            metrics, blink counts and averaged blink properties.
        """
        freq_rows = []
        energy_rows = []
        for seg_id, segment in enumerate(self.segments):
            signal = segment.get_data(picks="EEG-E8")[0]
            fd_feats = compute_frequency_domain_features([], signal, self.sfreq)
            td_feats = compute_time_domain_features(signal, self.sfreq)
            freq_rows.append({"seg_id": seg_id, **fd_feats})
            energy_rows.append({"seg_id": seg_id, **td_feats})

        df_freq = pd.DataFrame(freq_rows)
        df_energy = pd.DataFrame(energy_rows)

        if run_fit:
            with self.assertWarns(RuntimeWarning):
                blink_props = compute_segment_blink_properties(
                    self.segments,
                    self.blink_df,
                    self.params,
                    channel="EEG-E8",
                    run_fit=run_fit,
                    progress_bar=False,
                )
        else:
            blink_props = compute_segment_blink_properties(
                self.segments,
                self.blink_df,
                self.params,
                channel="EEG-E8",
                run_fit=run_fit,
                progress_bar=False,
            )

        blink_averages = (
            blink_props.groupby("seg_id").mean(numeric_only=True).reset_index()
        )
        blink_counts = (
            self.blink_df.groupby("seg_id").size().rename("blink_count").reset_index()
        )

        df = df_freq.merge(df_energy, on="seg_id")
        df = df.merge(blink_counts, on="seg_id", how="left")
        df = df.merge(blink_averages, on="seg_id", how="left")
        df["blink_count"] = df["blink_count"].fillna(0).astype(int)
        return df

    def test_pipeline_run_fit_false(self) -> None:
        """End-to-end feature extraction without blink fitting."""
        df = self._build_dataframe(run_fit=False)
        logger.debug("Combined feature DataFrame (run_fit=False):\n%s", df.head())
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.segments))
        self.assertIn("blink_count", df.columns)
        expected_fd = {
            "blink_rate_peak_freq",
            "blink_rate_peak_power",
            "broadband_power_0_5_2",
            "broadband_com_0_5_2",
            "high_freq_entropy_2_13",
        }
        expected_td = {"energy", "teager", "line_length", "velocity_integral"}
        self.assertTrue(expected_fd.issubset(df.columns))
        self.assertTrue(expected_td.issubset(df.columns))
        self.assertFalse(df[list(expected_fd | expected_td)].isna().any().any())
        counts = df.sort_values("seg_id")["blink_count"].tolist()
        self.assertListEqual(counts, self.expected_counts)

    def test_pipeline_run_fit_true(self) -> None:
        """End-to-end feature extraction with blink fitting enabled."""
        df = self._build_dataframe(run_fit=True)
        logger.debug("Combined feature DataFrame (run_fit=True):\n%s", df.head())
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.segments))
        self.assertIn("blink_count", df.columns)
        expected_fd = {
            "blink_rate_peak_freq",
            "blink_rate_peak_power",
            "broadband_power_0_5_2",
            "broadband_com_0_5_2",
            "high_freq_entropy_2_13",
        }
        expected_td = {"energy", "teager", "line_length", "velocity_integral"}
        self.assertTrue(expected_fd.issubset(df.columns))
        self.assertTrue(expected_td.issubset(df.columns))
        self.assertFalse(df[list(expected_fd | expected_td)].isna().any().any())
        counts = df.sort_values("seg_id")["blink_count"].tolist()
        self.assertListEqual(counts, self.expected_counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
