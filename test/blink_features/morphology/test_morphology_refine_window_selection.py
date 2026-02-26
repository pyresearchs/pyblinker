import numpy as np

from pyblinker.blink_features.morphology.epoch_features import (
    _build_blink_landmark_frame,
    _style_windows,
)


def test_style_windows_ignores_onset_duration_fallback() -> None:
    metadata_row = {
        "onset__outer__eeg": [0.01],
        "duration__outer__eeg": [0.03],
    }

    windows = _style_windows(
        metadata_row=metadata_row,
        modality="eeg",
        style="outer",
        n_times=100,
    )

    assert windows == []


def test_build_blink_landmark_frame_does_not_use_style_windows() -> None:
    metadata_row = {
        "start__outer__eeg": [10],
        "end__outer__eeg": [20],
    }
    signal = np.zeros(50)
    signal[15] = 5.0

    blink_df = _build_blink_landmark_frame(
        metadata_row=metadata_row,
        signal=signal,
        sfreq=100.0,
        n_times=50,
        modality="eeg",
    )

    assert blink_df.empty
