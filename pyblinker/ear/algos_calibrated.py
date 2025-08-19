import logging

import numpy as np
import pandas as pd

from .registry import register_ear_algo

logger = logging.getLogger(__name__)


@register_ear_algo("calibrated_threshold")
def detect_calibrated_threshold(x: np.ndarray, sfreq: float, params: dict) -> pd.DataFrame:
    """Calibrated hysteresis thresholding using open/closed EAR levels.

    ### 📈 Pipeline Flowchart

    ```plaintext
    +-----------+     +---------+     +-----------------------+     +------------------+     +------------------+
    | EAR input | --> | Smooth  | --> | Hysteresis thresholds | --> | Merge & filter   | --> | Blink DataFrame  |
    +-----------+     +---------+     +-----------------------+     +------------------+     +------------------+
    ```

    Parameters
    ----------
    x : numpy.ndarray
        Input EAR signal.
    sfreq : float
        Sampling frequency in Hertz.
    params : dict
        Parameters including ``calibration`` with ``open`` and ``closed`` values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``start_blink``, ``end_blink`` and ``pos_blink``
        (in samples).
    """
    p = params or {}
    cal = p.get("calibration", {})
    if not ({"open", "closed"} <= set(cal)):
        raise ValueError(
            "calibrated_threshold requires params['calibration']={'open','closed'}"
        )

    o = float(cal["open"])
    c = float(cal["closed"])

    smooth_win_s = p.get("smooth_win_s", 0.12)
    min_width_s = p.get("min_width_s", 0.06)
    max_width_s = p.get("max_width_s", 0.45)
    merge_gap_s = p.get("merge_gap_s", 0.15)
    refractory_s = p.get("refractory_s", 0.20)
    beta_enter = p.get("beta_enter", 0.50)
    beta_exit = p.get("beta_exit", 0.65)
    min_drop_abs = p.get("min_drop_abs", 0.03)

    if (o - c) < 1e-6:
        return pd.DataFrame({"start_blink": [], "end_blink": [], "pos_blink": []})

    x = np.asarray(x, float).ravel()
    x = np.clip(np.where(np.isnan(x), np.nanmedian(x), x), -0.1, 1.0)
    win = max(3, int(round(smooth_win_s * sfreq)))
    if win % 2 == 0:
        win += 1
    if win > 3:
        ker = np.ones(win) / win
        xs = np.convolve(x, ker, "same")
    else:
        xs = x

    enter_thr = c + beta_enter * (o - c)
    exit_thr = c + beta_exit * (o - c)

    segs = []
    in_evt = False
    s0 = 0
    for i, val in enumerate(xs):
        if not in_evt and val < enter_thr and (o - val) >= min_drop_abs:
            in_evt = True
            s0 = i
        elif in_evt and val > exit_thr:
            in_evt = False
            segs.append((s0, i))
    if in_evt:
        segs.append((s0, len(xs) - 1))

    max_gap = int(round(merge_gap_s * sfreq))
    min_w = int(round(min_width_s * sfreq))
    max_w = int(round(max_width_s * sfreq))
    refr = int(round(refractory_s * sfreq))

    merged = []
    for s, e in segs:
        if not merged or s - merged[-1][1] - 1 > max_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = e

    picks = []
    last_end = -10**9
    for s0, s1 in merged:
        w = s1 - s0 + 1
        if w < min_w or w > max_w:
            continue
        if s0 - last_end < refr:
            continue
        k = s0 + int(np.argmin(xs[s0:s1 + 1]))
        picks.append((s0, s1, k))
        last_end = s1

    return pd.DataFrame(picks, columns=["start_blink", "end_blink", "pos_blink"])
