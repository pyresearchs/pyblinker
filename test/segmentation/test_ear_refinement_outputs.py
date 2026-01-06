from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "test" / "test_files"

import mne
import numpy as np
import pandas as pd
import pandas.testing as pdt

from pyblinker.blink_features.ear_metrics import (
    EARBlinkFeatureExtractor,
    EARFeatureConfig,
    load_coarse_blinks,
    load_ear_channel,
)
from pyblinker.segmentation.refinement.ear import (
    EARRefinementConfig,
    refine_annotations_for_threshold,
)
from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data


def _drop_modality_columns(metadata: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Remove modality-specific columns (e.g., *_eeg) from a metadata frame."""

    suffix = f"_{modality}"
    columns_to_drop = [col for col in metadata.columns if col.endswith(suffix)]
    return metadata.drop(columns=columns_to_drop, errors="ignore")


def _assert_metadata_equivalent(actual: pd.DataFrame, expected: pd.DataFrame, *, atol: float = 1e-8) -> None:
    pdt.assert_index_equal(actual.index, expected.index)
    pdt.assert_index_equal(actual.columns, expected.columns)

    def _sequences_close(left_val: object, right_val: object) -> bool:
        if isinstance(left_val, (list, tuple, np.ndarray)) and isinstance(right_val, (list, tuple, np.ndarray)):
            left_arr = np.asarray(left_val, dtype=float)
            right_arr = np.asarray(right_val, dtype=float)
            if left_arr.shape == right_arr.shape and left_arr.size > 0:
                return np.allclose(left_arr, right_arr, atol=atol, rtol=0, equal_nan=True)
        return False

    for column in actual.columns:
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            pdt.assert_series_equal(
                left,
                right,
                check_names=False,
                check_dtype=False,
                atol=atol,
                rtol=0,
            )
        else:
            for idx, (left_val, right_val) in enumerate(zip(left, right)):
                if _sequences_close(left_val, right_val):
                    continue
                assert left_val == right_val or (
                    pd.isna(left_val) and pd.isna(right_val)
                ), f"Metadata column '{column}' differs at row {idx}"


def test_slice_raw_into_mne_epochs_matches_reference() -> None:
    raw_path = DATA_DIR / "ear_eog_raw.fif"
    csv_path = DATA_DIR / "ear_eog.csv"
    reference_path = DATA_DIR / "ear_metadata_threshold_interpolation.fif"

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
    raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

    ear_channel = "EAR-avg_ear"
    eeg_channel = "EEG-E8"
    raw.pick([ear_channel, eeg_channel])

    segmentation_config = {
        "ear": {
            "channel": ear_channel,
            "seg_type": "threshold_interpolation",
            "threshold": 0.260,
            "annotation_time_unit": "seconds",
            "max_extension": 0.35,
            "extension_step": 0.05,
            "padding": 0.05,
            "extend_before": True,
            "extend_after": True,
        },
        "eeg": {"channel": eeg_channel, "seg_type": [], "threshold": None},
        "eog": {"seg_type": [], "threshold": None},
    }
    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=30.0,
        blink_label=None,
        segmentation_type=segmentation_config,
        progress_bar=False,
    )

    reference_epochs = mne.read_epochs(reference_path, preload=True, verbose="ERROR")

    assert len(epochs) == len(reference_epochs)
    np.testing.assert_allclose(epochs.events, reference_epochs.events)
    np.testing.assert_allclose(
        epochs.get_data(),
        reference_epochs.get_data(),
        rtol=0,
        atol=1e-8,
    )
    # EEG refinement is explicitly disabled (``seg_type=[]``), so EEG metadata
    # columns are omitted even though the channel is present.
    expected_metadata = _drop_modality_columns(reference_epochs.metadata, "eeg")
    _assert_metadata_equivalent(epochs.metadata, expected_metadata)


def test_multi_threshold_refinement_matches_reference_csv() -> None:
    annotation_csv = DATA_DIR / "ear_eog.csv"
    fif_path = DATA_DIR / "ear_eog_raw.fif"
    reference_csv = DATA_DIR / "ear_multi_threshold_refined_blinks.csv"

    annotations = load_coarse_blinks(annotation_csv)
    ear_signal, sfreq = load_ear_channel(fif_path, channel="EAR-avg_ear")

    candidate_thresholds = [0.18, 0.2, 0.22, 0.24, 0.26]
    feature_config = EARFeatureConfig(
        baseline_window=0.25,
        classification_threshold=None,
        context_window=0.1,
    )
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

    reference = pd.read_csv(reference_csv)

    sort_cols = ["candidate_id", "threshold_value", "threshold_index"]
    features_sorted = features.sort_values(sort_cols).reset_index(drop=True)
    reference_sorted = reference.sort_values(sort_cols).reset_index(drop=True)

    features_aligned = features_sorted[reference_sorted.columns]
    reference_aligned = reference_sorted[reference_sorted.columns]

    numeric_cols = reference_aligned.select_dtypes(include=[np.number]).columns
    pdt.assert_frame_equal(
        features_aligned.drop(columns=numeric_cols),
        reference_aligned.drop(columns=numeric_cols),
        check_dtype=False,
    )
    np.testing.assert_allclose(
        features_aligned[numeric_cols].to_numpy(),
        reference_aligned[numeric_cols].to_numpy(),
        rtol=0,
        atol=1e-6,
    )
