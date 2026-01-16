from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "test" / "test_files"

import mne
import pandas as pd

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.evaluation import mat_data



EXPECTED_COLUMNS = [
    "blink_onset",
    "blink_duration",
    "n_blinks",
    "blink_onset_ear",
    "blink_duration_ear",
    "blink_onset_extremum_ear",
    "onset__refine__ear",
    "duration__refine__ear",
    "refined_start_sample",
    "refined_end_sample",
    "refined_lowest_point_sample",
    "refined_left_threshold",
    "refined_right_threshold",
    "search_window_start_sample",
    "search_window_end_sample",
    "search_window_start_time",
    "search_window_end_time",
    "refinement_succeeded",
    "search_exhausted",
    "extension_seconds_used",
    "extension_attempts",
    "onset__th_interpolation__ear",
    "duration__th_interpolation__ear",
    "left_interpolated_threshold",
    "right_interpolated_threshold",
    "left_interpolated_threshold_sample",
    "right_interpolated_threshold_sample",
    "left_interpolated_threshold_found",
    "right_interpolated_threshold_found",
    "interpolated_thresholds_found",
    "onset__th_sample__ear",
    "duration__th_sample__ear",
    "start__th_point__ear",
    "end__th_point__ear",
    "trough__th_point__ear",
    "onset__th__ear",
    "duration__th__ear",
]


def _normalize_for_compare(df: pd.DataFrame, *, expected_columns: list[str]) -> pd.DataFrame:
    """Normalize metadata DataFrame so comparisons are deterministic."""
    # Select columns (and fail loudly if missing)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing expected metadata columns: {missing}")

    out = df.loc[:, expected_columns].copy()

    # Sorting + index reset makes this robust to index label differences.
    # (If row identity needs to be stronger later, we can align by a key.)
    out = out.sort_index().reset_index(drop=True)

    def _coerce_boollike_to_boolean(s: pd.Series) -> pd.Series:
        """Best-effort conversion to pandas nullable boolean.

        Handles True/False, 0/1, and common string forms. If values aren't
        bool-like (e.g., lists/objects), returns the original series.
        """
        # Fast-path for real boolean dtypes
        if pd.api.types.is_bool_dtype(s.dtype) or str(s.dtype) == "boolean":
            arr = pd.array(s, dtype="boolean")
            return pd.Series(arr, index=s.index, name=s.name)

        # Don't try to coerce complex/object columns that include lists/dicts.
        # (Those are not intended to be boolean flags.)
        if s.dtype == object and s.map(lambda x: isinstance(x, (list, dict))).any():
            return s

        def _to_bool_or_na(x):
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return pd.NA
            if isinstance(x, bool):
                return x
            if isinstance(x, (int,)):
                if x in (0, 1):
                    return bool(x)
                return pd.NA
            if isinstance(x, float):
                if x in (0.0, 1.0):
                    return bool(int(x))
                return pd.NA
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("true", "t", "yes", "y", "1"):
                    return True
                if v in ("false", "f", "no", "n", "0"):
                    return False
                return pd.NA
            return pd.NA

        coerced = s.map(_to_bool_or_na)
        # Only adopt the coercion if it actually produced some boolean-ish values
        # (otherwise we'd turn a non-boolean column into all-NA).
        if coerced.notna().any():
            arr = pd.array(coerced, dtype="boolean")
            return pd.Series(arr, index=s.index, name=s.name)
        return s

    # Ensure consistent bool dtype where possible
    for col in (
        "refinement_succeeded",
        "search_exhausted",
        "left_interpolated_threshold_found",
        "right_interpolated_threshold_found",
        "interpolated_thresholds_found",
    ):
        if col in out.columns:
            out[col] = _coerce_boollike_to_boolean(out[col])

    return out


def _assert_metadata_equal(got: pd.DataFrame, expected: pd.DataFrame, *, columns: list[str]) -> None:
    got_n = _normalize_for_compare(got, expected_columns=columns)
    exp_n = _normalize_for_compare(expected, expected_columns=columns)

    try:
        pd.testing.assert_frame_equal(
            got_n,
            exp_n,
            check_dtype=False,
            check_like=False,
            rtol=1e-7,
            atol=1e-9,
        )
    except AssertionError as e:
        # Add a compact diff to help debugging in CI.
        diff = got_n.compare(exp_n, keep_shape=False, keep_equal=False)
        msg = f"Metadata mismatch. Diff (showing changed cells only):\n{diff.head(200)}"
        raise AssertionError(msg) from e


class TestEarRefinementMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw_path = DATA_DIR / "ear_eog_raw.fif"
        csv_path = DATA_DIR / "ear_eog.csv"
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
        raw.set_annotations(mat_data.read_annotations_as_mne(csv_path))

        base_config = {
            "ear": {
                # "channel": ear_channel,
                "seg_type": "threshold_interpolation",
                "threshold": 0.260,
                "annotation_time_unit": "seconds",
                "max_extension": 0.5,
                "extension_step": 0.05,
                "padding": 0.05,
                "extend_before": True,
                "extend_after": True,
            },
        }
        from test.segment_config import build_segment_config  # noqa: E402
        segmentation_config = build_segment_config(raw, base_config=base_config)
        cls.epochs = slice_raw_into_mne_epochs_refine_annot(
            raw,
            epoch_len=30.0,
            blink_label=None,
            segmentation_type=segmentation_config,
            progress_bar=False,
        )

        golden_path = PROJECT_ROOT / "test" / "segmentation" / "ear__metadata.pkl"
        cls.expected_metadata = pd.read_pickle(golden_path)

    def test_metadata_matches_reference(self) -> None:
        got_metadata = self.epochs.metadata
        _assert_metadata_equal(got_metadata, self.expected_metadata, columns=EXPECTED_COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
