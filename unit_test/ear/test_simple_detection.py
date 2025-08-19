"""Basic sanity tests for EAR adaptive MAD detection using mock signals."""
import numpy as np

from pyblinker.ear.blink_epoch_mapper import _get_blink_position_epoching_ear
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear


def test_adaptive_mad_detects_mock_blinks() -> None:
    """Ensure adaptive MAD detects all blinks in the first mock epoch."""
    refined, sfreq, _epoch_len, _n_epochs = _generate_refined_ear()
    epoch_signal = next(r["epoch_signal"] for r in refined if r["epoch_index"] == 0)
    truth_peaks = [r["refined_peak_frame"] for r in refined if r["epoch_index"] == 0]
    params = {
        "sfreq": sfreq,
        "ear_algo": "adaptive_mad",
        "min_width_s": 0.01,
        "baseline_win_s": 0.5,
        "smooth_win_s": 0.02,
        "thresh_k_mad": 1.5,
        "min_drop_abs": 0.02,
        "min_drop_rel": 0.1,
    }
    df = _get_blink_position_epoching_ear(epoch_signal, params, ch="EAR")
    detected = df["pos_blink"].to_numpy()
    assert len(detected) == len(truth_peaks)
    tol = int(0.05 * sfreq)
    for peak in truth_peaks:
        assert np.min(np.abs(detected - peak)) <= tol
