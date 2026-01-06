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

from pyblinker.blink_features.ear_metrics import (  # noqa: E402
    EARBlinkFeatureExtractor,
    EARFeatureConfig,
    load_coarse_blinks,
    load_ear_channel,
)
from pyblinker.segmentation.refinement.ear import (  # noqa: E402
    EARRefinementConfig,
    refine_annotations_for_threshold,
)
from pyblinker.utils import select_auto_threshold  # noqa: E402
from pyblinker.outside_annotation import build_refined_blink_report  # noqa: E402
from pyblinker.viz import prepare_threshold_report_dataframe  # noqa: E402


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

    feature_config = EARFeatureConfig(
        baseline_window=0.25,
        classification_threshold=None,  # classification uses each threshold value
        context_window=0.1,
        )

    print("Loading coarse blink annotations from:", annotation_csv)
    annotations = load_coarse_blinks(annotation_csv)
    print(f"{len(annotations)} coarse blinks loaded.")

    print("Loading EAR channel from FIF:", fif_path)
    ear_signal, sfreq = load_ear_channel(fif_path, channel="EAR-avg_ear")
    print(f"Sampling rate: {sfreq} Hz; signal length: {len(ear_signal)} samples")

    # Refine and extract independently for each threshold.
    extractor = EARBlinkFeatureExtractor(ear_signal, sfreq, feature_config=feature_config)
    feature_tables = []
    for idx, theta in enumerate(candidate_thresholds):
        refinement_config = EARRefinementConfig(
            threshold=theta,
            annotation_time_unit="seconds",
            max_extension=0.5,
            extension_step=0.05,
            padding=0.05,
            extend_before=True,
            extend_after=True,
            )
        refined = refine_annotations_for_threshold(
            signal=ear_signal,
            sfreq=sfreq,
            annotations=annotations,
            base_config=refinement_config,
            candidate_threshold=theta,
            threshold_index=idx,
            )
        feature_tables.append(extractor.build_feature_table(refined))
    features = pd.concat(feature_tables, ignore_index=True)

    if save_reports:
        output_path = output_dir / "ear_multi_threshold_refined_blinks.csv"
        features.to_csv(output_path, index=False)

    print("Example refined rows (first five across thresholds):")
    print(features.loc[:4, ["candidate_id", "threshold_value", "refined_onset_time", "refined_offset_time"]])

    print("Threshold-dependent metrics for the first blink/threshold row:")
    threshold_cols = [
            "threshold_value",
            "closed_duration_seconds",
            "auc_below_threshold",
            "blink_classification",
            ]
    print(features.loc[0, threshold_cols].to_frame().T)

    # Build visual reports:
    # 1) User-specified plot threshold.
    # 2) Auto-selected threshold with rationale surfaced in the HTML summary.
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    eeg_overlay = raw.get_data(picks="EEG-E8")[0]
    overlay_sfreq = float(raw.info["sfreq"])

    report_threshold = candidate_thresholds[0]
    report_df = prepare_threshold_report_dataframe(features, sfreq, report_threshold)
    user_plot_threshold = report_threshold

    auto_threshold = select_auto_threshold(features)
    auto_report_df = prepare_threshold_report_dataframe(features, sfreq, auto_threshold)
    if save_reports:
        user_report_path = output_dir / "ear_multi_threshold_refined_blink_report_user.html"
        build_refined_blink_report(
            results=report_df,
            signal=ear_signal,
            sfreq=sfreq,
            channel_name="EAR-avg_ear",
            plot_overlay=True,
            plot_signal_as_scatter=True,
            mark_threshold_crossings=True,
            threshold_value=user_plot_threshold,
            overlay_signal=eeg_overlay,
            overlay_sfreq=overlay_sfreq,
            overlay_label="EEG-E8",
            output_path=user_report_path,
            epoch_duration=30.0,
            )

        auto_report_path = output_dir / "ear_multi_threshold_refined_blink_report_auto.html"
        build_refined_blink_report(
            results=auto_report_df,
            signal=ear_signal,
            sfreq=sfreq,
            channel_name="EAR-avg_ear",
            plot_overlay=True,
            plot_signal_as_scatter=True,
            mark_threshold_crossings=True,
            threshold_value=auto_threshold,
            overlay_signal=eeg_overlay,
            overlay_sfreq=overlay_sfreq,
            overlay_label="EEG-E8",
            output_path=auto_report_path,
            epoch_duration=30.0,
            )

    n_success = int(features["refinement_succeeded"].sum())
    print(f"Refined {len(features)} blink-threshold pairs; {n_success} used threshold crossings.")
    print(f"Report threshold used for visualization: {report_threshold}")
    print(f"Automatically selected threshold for report: {auto_threshold}")
    print("Average onset shift (s):", features["onset_offset_seconds"].mean())
    print("Average offset shift (s):", features["offset_offset_seconds"].mean())

    preview_cols = [
            "candidate_id",
            "threshold_value",
            "threshold_index",
            "blink_type",
            "refined_onset_time",
            "refined_offset_time",
            "refined_duration",
            "refinement_succeeded",
            "ear_min",
            "ear_blink_depth",
            "closed_duration_seconds",
            "auc_below_threshold",
            "blink_classification",
            ]
    print("\nExample rows with multi-threshold EAR features:")
    print(features.loc[:, preview_cols].head())

    if save_reports:
        print("\nSaved refined blink table to:", output_path)
        print("Blink validation report with user threshold saved to:", user_report_path)
    else:
        print("\nSaving is disabled (set PYBLINKER_SAVE_REPORTS=1 to enable).")
    print(
        "You can adjust the candidate thresholds to compare how crossings and derived\n"
        "metrics change without rerunning annotation refinement."
        )


if __name__ == "__main__":
    main()
