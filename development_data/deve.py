"""Development helper script for comparing PyBlinker detections to ground truth."""

from __future__ import annotations

import os
from pathlib import Path

import mne
import pandas as pd

from pyblinker.utils.evaluation import blink_comparison


SAMPLING_RATE_HZ = 200.0
CHANNELS_TO_KEEP = ("CH1",)
TOLERANCE_SAMPLES = 1
N_PREVIEW_ROWS = 10
N_DIFF_ROWS = 20

DATA_DIR = Path(__file__).resolve().parent


def _load_ground_truth() -> pd.DataFrame:
    ground_truth_events = pd.read_pickle(DATA_DIR / "blinker_results.pkl")[
        "frames"
    ]["blinkFits"]
    ground_truth_events = ground_truth_events[["leftZero", "rightZero", "maxValue"]].rename(
        columns={"leftZero": "start_blink", "rightZero": "end_blink"}
    )
    ground_truth_events.loc[len(ground_truth_events)] = {
        "start_blink": 1940,
        "end_blink": 2005,
        "maxValue": None,
    }
    return (
        ground_truth_events.sort_values(by="start_blink").reset_index(drop=True).head(10)
    )


def _load_detections() -> pd.DataFrame:
    detection = pd.read_pickle(DATA_DIR / "pyblinker_results.pkl")["events"]
    detection = detection[["left_zero", "right_zero", "max_value"]].rename(
        columns={
            "left_zero": "start_blink",
            "right_zero": "end_blink",
            "max_value": "maxValue",
        }
    )
    detection.loc[len(detection)] = {
        "start_blink": 1918,
        "end_blink": 2042,
        "maxValue": None,
    }
    return detection.sort_values(by="start_blink").reset_index(drop=True).head(10)


def main() -> None:
    ground_truth_events = _load_ground_truth()
    detection = _load_detections()

    raw = mne.io.read_raw_fif(DATA_DIR / "9636511.fif", preload=True)
    raw.crop(0, 12.0)

    signal = raw.get_data(picks=CHANNELS_TO_KEEP[0])[0]

    comparison = blink_comparison.compare_detected_vs_ground_truth(
        detection,
        ground_truth_events,
        SAMPLING_RATE_HZ,
        tolerance_samples=TOLERANCE_SAMPLES,
        n_preview_rows=N_PREVIEW_ROWS,
        n_diff_rows=N_DIFF_ROWS,
        detected_signal=signal,
    )

    diagnostic_raw = comparison.diagnostic_raw
    metrics = comparison.metrics
    diff_table = comparison.diff_table

    if diff_table.empty:
        print("\n[diff] diff_table is empty")
    else:
        print("\n[diff] Full diff_table:")
        print(diff_table)

    assert int(metrics["detected_only"]) == 4, metrics
    assert int(metrics["share_within_tolerance"]) == 2, metrics

    annotations = diagnostic_raw.annotations if diagnostic_raw is not None else None
    if annotations is not None:
        assert len(annotations) == 6, f"Expected 6 annotations, found {len(annotations)}"
        print(f"[mne] Applying {len(annotations)} comparison annotations to the EEG raw")
        raw.set_annotations(annotations)
    else:
        print("[mne] No blink annotations generated; clearing annotations on the EEG raw")
        raw.set_annotations(None)

    if os.environ.get("PYBLINKER_SKIP_PLOT") == "1":
        print("[info] Skipping raw.plot() because PYBLINKER_SKIP_PLOT=1")
    else:
        raw.plot(block=True)


if __name__ == "__main__":
    main()
