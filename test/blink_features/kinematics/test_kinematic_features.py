"""Tests for kinematic blink feature aggregation using epoch metadata."""

from __future__ import annotations

import unittest
from pathlib import Path

import mne
import numpy as np

from pyblinker.blink_features.kinematics.core_metrics import (
    KINEMATIC_METRIC_STEMS,
    KINEMATIC_METRICS_NO_STYLE,
)
from pyblinker.blink_features.kinematics.per_blink import compute_segment_kinematics
from pyblinker.blink_features.energy.helpers import _safe_stats
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.blink_features.kinematics.kinematic_features import (
    KinematicBlinkFeatureExtractor,
    _available_styles,
    _style_windows,
)
from pyblinker.blink_features.utils.aggregation import prepare_epoch_channel_data
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
        extractor = KinematicBlinkFeatureExtractor(epochs=self.epochs)
        df = extractor.compute(picks=ch)

        styles = _available_styles(tuple(self.epochs.metadata.columns), "eeg")
        metric_keys = [
            stem if stem in KINEMATIC_METRICS_NO_STYLE else f"{stem}_{style}"
            for stem in KINEMATIC_METRIC_STEMS
            for style in styles
        ]
        expected_cols = [
            f"eeg__{style}__kinematic__{m}_{s}__{ch}"
            for m in metric_keys
            for style in styles
            for s in ("mean", "std", "cv")
            if m.endswith(style) or m in KINEMATIC_METRICS_NO_STYLE
        ]
        assert_df_has_columns(self, df, expected_cols)
        assert_numeric_or_nan(self, df.iloc[0])

        zero_idx = self.epochs.metadata.index[
            self.epochs.metadata["n_blinks"] == 0
        ][0]
        self.assertTrue(df.loc[zero_idx, expected_cols].isna().all())

    def test_manual_first_epoch(self) -> None:
        """Manual computation for the first epoch matches library output."""
        ch = "EEG-E8"
        extractor = KinematicBlinkFeatureExtractor(epochs=self.epochs)
        df = extractor.compute(picks=ch)
        sfreq = float(self.epochs.info["sfreq"])
        meta = self.epochs.metadata.iloc[0]
        styles = sorted(_available_styles(tuple(self.epochs.metadata.columns), "eeg"))
        ch_names, channel_data, _, _, n_times = prepare_epoch_channel_data(
            epochs=self.epochs, picks=[ch], sfreq=sfreq
        )
        metrics_by_style = {
            style: [
                stem if stem in KINEMATIC_METRICS_NO_STYLE else f"{stem}_{style}"
                for stem in KINEMATIC_METRIC_STEMS
            ]
            for style in styles
        }
        manual = {}
        for style in styles:
            per_metric = {m: [] for m in metrics_by_style[style]}
            windows = _style_windows(meta, "eeg", style)
            for start_idx, end_idx in windows:
                if start_idx >= n_times:
                    continue
                sl = slice(max(0, start_idx), min(end_idx, n_times))
                seg = {
                    "raw": channel_data[ch_names[0]]["raw"][0, sl],
                    "dx1": channel_data[ch_names[0]]["dx1"][0, sl],
                    "dx2": channel_data[ch_names[0]]["dx2"][0, sl],
                }
                metrics = compute_segment_kinematics(seg, sfreq, method=style, modality="eeg")
                for m in metrics_by_style[style]:
                    per_metric[m].append(metrics[m])

            for metric, values in per_metric.items():
                stats = _safe_stats(values)
                for stat_name, value in stats.items():
                    manual[f"eeg__{style}__kinematic__{metric}_{stat_name}__{ch}"] = value

        for key, val in manual.items():
            self.assertAlmostEqual(df.iloc[0][key], val, places=7)

    def test_style_windows_use_frame_bounds(self) -> None:
        """Style windows should be built from start/end frame metadata."""
        metadata_row = {
            "start__left_base__eeg": [10, 30],
            "end__right_base__eeg": [20, 45],
        }
        self.assertEqual(_style_windows(metadata_row, "eeg", "base"), [(10, 20), (30, 45)])

    def test_method_suffix_and_modality_guard(self) -> None:
        """Per-blink metrics include method suffixes and respect modality rules."""
        segment = np.array([0.0, 1.0, 0.2, 0.0])
        metrics = compute_segment_kinematics(
            {"raw": segment},
            100.0,
            method="base",
            modality="eeg",
        )
        self.assertIn("vel_peak_abs_base", metrics)

        ear_metrics = compute_segment_kinematics(
            segment,
            100.0,
            method="zero",
            modality="ear",
        )
        self.assertTrue(
            np.isnan(ear_metrics["vel_peak_abs_zero"]),
            msg="EAR zero-crossing metrics should be NaN",
        )


if __name__ == "__main__":
    unittest.main()
