"""Tutorial: Evaluate multiple EAR thresholds by refining + extracting per candidate.

This example mirrors ``tutorial/3_ear_threshold_blink_refinement.py`` but runs the full
refinement + feature extraction pipeline independently for each EAR threshold. The
resulting table contains one row per threshold + coarse annotation pair instead of
flattened per-threshold columns.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import mne
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyblinker.blink_features.ear_metrics import load_coarse_blinks  # noqa: E402
from pyblinker.outside_annotation import build_refined_blink_report  # noqa: E402
from pyblinker.segmentation.refinement import (  # noqa: E402
    slice_raw_into_mne_epochs_refine_annot,
)
from test.segment_config import build_segment_config  # noqa: E402


def main() -> None:
    save_reports = os.environ.get("PYBLINKER_SAVE_REPORTS", "1") != "0"
    data_dir = PROJECT_ROOT / "test" / "test_files"
    output_dir = PROJECT_ROOT / "tutorial_outputs"
    if save_reports:
        output_dir.mkdir(parents=True, exist_ok=True)

    annotation_csv = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog_raw.fif"

    # Evaluate several EAR thresholds per blink.
    candidate_thresholds = [
        # 0.18, 0.2, 0.22, 0.24,
        0.26,
    ]

    print("Loading coarse blink annotations from:", annotation_csv)
    annotations = load_coarse_blinks(annotation_csv)
    print(f"{len(annotations)} coarse blinks loaded.")

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    ear_signal = raw.get_data(picks="EAR-avg_ear")[0]
    eeg_overlay = raw.get_data(picks="EEG-E8")[0]
    overlay_sfreq = float(raw.info["sfreq"])

    def _listify(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [] if isinstance(value, float) and value != value else [value]

    def _explode_epoch_metadata(metadata):
        rows = []
        for row in metadata.itertuples(index=False):
            n_blinks = int(getattr(row, "n_blinks", 0))
            if n_blinks == 0:
                continue
            fields = {
                "onset__refine__ear": _listify(
                    getattr(row, "onset__refine__ear", None)
                ),
                "duration__refine__ear": _listify(
                    getattr(row, "duration__refine__ear", None)
                ),
                "onset__th_interpolation__ear": _listify(
                    getattr(row, "onset__th_interpolation__ear", None)
                ),
                "duration__th_interpolation__ear": _listify(
                    getattr(row, "duration__th_interpolation__ear", None)
                ),
                "trough__th_point__ear": _listify(
                    getattr(row, "trough__th_point__ear", None)
                ),
                "refined_left_threshold": _listify(
                    getattr(row, "refined_left_threshold", None)
                ),
                "refined_right_threshold": _listify(
                    getattr(row, "refined_right_threshold", None)
                ),
                "refined_lowest_point_sample": _listify(
                    getattr(row, "refined_lowest_point_sample", None)
                ),
            }
            for idx in range(n_blinks):
                rows.append({key: values[idx] for key, values in fields.items()})
        return rows

    report_threshold = candidate_thresholds[0]
    report_rows = []
    for theta in candidate_thresholds:
        base_config = {
            "ear": {
                "seg_type": "threshold_interpolation",
                "threshold": theta,
                "annotation_time_unit": "seconds",
                "max_extension": 0.5,
                "extension_step": 0.05,
                "padding": 0.05,
                "extend_before": True,
                "extend_after": True,
            },
        }
        segmentation_config = build_segment_config(raw, base_config=base_config)
        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            progress_bar=False,
            segmentation_type=segmentation_config,
        )
        report_rows.extend(_explode_epoch_metadata(epochs.metadata))

    if save_reports:
        user_report_path = (
            output_dir / "ear_multi_threshold_refined_blink_report_user.html"
        )
        report_df = pd.DataFrame.from_records(report_rows)
        build_refined_blink_report(
            results=report_df,
            signal=ear_signal,
            sfreq=sfreq,
            channel_name="EAR-avg_ear",
            plot_overlay=True,
            plot_signal_as_scatter=True,
            mark_threshold_crossings=True,
            threshold_value=report_threshold,
            overlay_signal=eeg_overlay,
            overlay_sfreq=overlay_sfreq,
            overlay_label="EEG-E8",
            output_path=user_report_path,
            epoch_duration=30.0,
        )


if __name__ == "__main__":
    main()
