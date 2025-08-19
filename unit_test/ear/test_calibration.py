import numpy as np
import mne

from pyblinker.ear.calibration import compute_ear_calibration


def _make_raw_with_annotations() -> mne.io.BaseRaw:
    sfreq = 100.0
    info = mne.create_info(["EAR"], sfreq, ["misc"])
    open_seg = np.full(int(sfreq * 2), 0.31)
    closed_seg = np.full(int(sfreq * 2), 0.11)
    data = np.concatenate([open_seg, closed_seg])[np.newaxis, :]
    raw = mne.io.RawArray(data, info, verbose=False)
    ann = mne.Annotations(
        onset=[0.0, 2.0],
        duration=[2.0, 2.0],
        description=["eyes_open_calib", "eyes_closed_calib"],
    )
    raw.set_annotations(ann)
    return raw


def test_calibration_from_annotations():
    raw = _make_raw_with_annotations()
    cal = compute_ear_calibration(raw, ch_name="EAR")
    assert abs(cal["open"] - 0.31) < 0.02
    assert abs(cal["closed"] - 0.11) < 0.02


def test_calibration_from_separate_arrays():
    rng = np.random.default_rng(0)
    open_arr = 0.32 + 0.005 * rng.normal(size=500)
    closed_arr = 0.09 + 0.005 * rng.normal(size=500)
    cal = compute_ear_calibration(open_arr, closed_arr, ch_name="EAR")
    assert abs(cal["open"] - 0.32) < 0.02
    assert abs(cal["closed"] - 0.09) < 0.02
