"""Tests for kinematic blink feature aggregation using epoch metadata."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blink_features.kinematics import compute_kinematic_features
from pyblinker.blink_features.kinematics.per_blink import compute_segment_kinematics
from pyblinker.blink_features.energy.helpers import (
    extract_blink_windows,
    segment_to_samples,
    _safe_stats,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from test.segment_config import build_segment_config

from ..utils.helpers import assert_df_has_columns, assert_numeric_or_nan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestKinematicFeatures(unittest.TestCase):
    """Validate kinematic metrics computed from epoch metadata."""

    def setUp(self) -> None:  # noqa: D401
        raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        segmentation_config = build_segment_config(raw)
        self.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )

    def test_dataframe_and_nan_epochs(self) -> None:
        """DataFrame has expected columns and NaNs for zero-blink epochs."""
        ch = "EEG-E8"
        df = compute_kinematic_features(self.epochs, picks=ch)

        metric_keys = tuple(
            compute_segment_kinematics(np.zeros(3), 1.0, methods=("base",)).keys()
        )
        expected_cols = [
            f"{m}_{s}_{ch}" for m in metric_keys for s in ("mean", "std", "cv")
        ]
        assert_df_has_columns(self, df, expected_cols)
        assert_numeric_or_nan(self, df.iloc[0])

        zero_idx = self.epochs.metadata.index[
            self.epochs.metadata["n_blinks"] == 0
        ][0]
        self.assertTrue(df.loc[zero_idx].isna().all())

    def test_manual_first_epoch(self) -> None:
        """Manual computation for the first epoch matches library output."""
        ch = "EEG-E8"
        df = compute_kinematic_features(self.epochs, picks=ch)
        sfreq = float(self.epochs.info["sfreq"])
        data = self.epochs.get_data(picks=[ch])
        meta = self.epochs.metadata.iloc[0]
        windows = extract_blink_windows(meta, ch, 0)
        metric_keys = tuple(
            compute_segment_kinematics(np.zeros(3), 1.0, methods=("base",)).keys()
        )
        per_metric = {m: [] for m in metric_keys}
        n_times = data.shape[-1]
        for onset, dur in windows:
            sl = segment_to_samples(onset, dur, sfreq, n_times)
            seg = data[0, 0, sl]
            metrics = compute_segment_kinematics(seg, sfreq, methods=("base",))
            for m in metric_keys:
                per_metric[m].append(metrics[m])

        manual = {}
        for metric, values in per_metric.items():
            stats = _safe_stats(values)
            for stat_name, value in stats.items():
                manual[f"{metric}_{stat_name}_{ch}"] = value

        for key, val in manual.items():
            self.assertAlmostEqual(df.iloc[0][key], val, places=7)

    def test_method_suffix_and_modality_guard(self) -> None:
        """Per-blink metrics include method suffixes and respect modality rules."""
        segment = np.array([0.0, 1.0, 0.2, 0.0])
        metrics = compute_segment_kinematics(
            {"base": segment, "half_base": segment},
            100.0,
            modality="eeg",
        )
        self.assertIn("area_abs_total_trapz_base", metrics)
        self.assertIn("area_abs_total_trapz_half_base", metrics)

        ear_metrics = compute_segment_kinematics(
            segment,
            100.0,
            methods=("zero",),
            modality="ear",
        )
        self.assertTrue(
            np.isnan(ear_metrics["area_abs_total_trapz_zero"]),
            msg="EAR zero-crossing metrics should be NaN",
        )


if __name__ == "__main__":
    unittest.main()
