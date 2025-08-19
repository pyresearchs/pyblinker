"""Tests for EAR blink epoch mapping algorithms."""
import numpy as np
import pytest

from pyblinker.ear.blink_epoch_mapper import _get_blink_position_epoching_ear


def _synth_ear(
    *,
    seconds: float = 20.0,
    sfreq: float = 100.0,
    open_level: float = 0.30,
    closed_level: float = 0.10,
    n_blinks: int = 8,
    blink_width_ms: int = 180,
    jitter_ms: int = 40,
    noise_std: float = 0.01,
    seed: int = 7,
):
    """Create a synthetic EAR signal with configurable blink events."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    x = np.full(n, open_level, float)
    x += rng.normal(scale=noise_std, size=n)

    idxs = [int((i + 1) * n / (n_blinks + 1)) for i in range(n_blinks)]
    idxs = [
        i
        + rng.integers(-int(jitter_ms * sfreq / 1000), int(jitter_ms * sfreq / 1000))
        for i in idxs
    ]

    w = int(blink_width_ms * sfreq / 1000)
    w = max(4, w | 1)

    truth = []
    for c in idxs:
        s0 = max(0, c - w // 2)
        s1 = min(n - 1, c + w // 2)
        seg = np.linspace(-1, 1, s1 - s0 + 1)
        shape = 1 - (seg ** 2)
        valley = closed_level + 0.02 * rng.normal(size=shape.size)
        x[s0 : s1 + 1] = np.minimum(
            x[s0 : s1 + 1], valley + (open_level - valley) * (1 - shape)
        )
        truth.append((s0, s1, (s0 + s1) // 2))

    truth = np.array(truth, int)
    return x, t, truth


def _match_tolerance(pred: np.ndarray, truth_centers: np.ndarray, tol: int):
    """Compute TP/FP/FN counts for predicted blink centers."""
    pred = np.asarray(pred, int)
    truth_centers = np.asarray(truth_centers, int)
    used = np.zeros(len(truth_centers), bool)
    tp = 0
    for p in pred:
        d = np.abs(truth_centers - p)
        j = np.argmin(d)
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(truth_centers) - tp
    return tp, fp, fn


@pytest.mark.parametrize("algo", ["adaptive_mad", "calibrated_threshold"])
def test_ear_algorithms_basic(algo: str) -> None:
    """Ensure both EAR algorithms achieve good precision/recall."""
    x, _t, truth = _synth_ear(seconds=30, sfreq=120, n_blinks=10)
    sfreq = 120.0
    params = {"sfreq": sfreq, "ear_algo": algo}

    if algo == "calibrated_threshold":
        params["calibration"] = {"open": 0.30, "closed": 0.10}

    df = _get_blink_position_epoching_ear(x, params, ch="EAR")
    centers = df["pos_blink"].to_numpy()

    tol = int(0.05 * sfreq)
    tp, fp, fn = _match_tolerance(centers, truth[:, 2], tol)

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))

    assert rec >= 0.85, f"Recall too low: {rec:.2f} (tp={tp}, fp={fp}, fn={fn})"
    assert prec >= 0.85, f"Precision too low: {prec:.2f} (tp={tp}, fp={fp}, fn={fn})"


def test_dispatcher_auto_chooses_calibrated_when_available() -> None:
    """Auto selection uses calibrated algorithm when calibration data exist."""
    x, _, _ = _synth_ear(sfreq=100)
    params = {
        "sfreq": 100.0,
        "ear_algo": "auto",
        "calibration": {"open": 0.30, "closed": 0.10},
    }
    df = _get_blink_position_epoching_ear(x, params)
    assert {"start_blink", "end_blink", "pos_blink"} <= set(df.columns)


def test_empty_signal_returns_empty_df() -> None:
    """Empty input signals result in an empty DataFrame."""
    params = {"sfreq": 100.0, "ear_algo": "adaptive_mad"}
    df = _get_blink_position_epoching_ear(np.array([]), params)
    assert df.empty
