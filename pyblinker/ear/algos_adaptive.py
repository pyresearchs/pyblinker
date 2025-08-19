import logging
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter  # optional
    from scipy.ndimage import median_filter  # optional
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from .registry import register_ear_algo

logger = logging.getLogger(__name__)


@register_ear_algo("adaptive_mad")
def detect_adaptive_mad(x: np.ndarray, sfreq: float, params: dict) -> pd.DataFrame:
    """Adaptive-baseline + MAD thresholding to detect EAR valleys.

    ### 📈 Pipeline Flowchart

    ```plaintext
    +-----------+     +---------+     +-------------------------+     +--------------------+     +------------------+
    | EAR input | --> | Smooth  | --> | Adaptive baseline & MAD | --> | Segment & score    | --> | Blink DataFrame  |
    +-----------+     +---------+     +-------------------------+     +--------------------+     +------------------+
    ```

    Parameters
    ----------
    x : numpy.ndarray
        Input EAR signal.
    sfreq : float
        Sampling frequency in Hertz.
    params : dict
        Algorithm-specific parameters.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``start_blink``, ``end_blink`` and ``pos_blink``
        (in samples).
    """
    p = params or {}

    # ---- defaults
    smooth_win_s = p.get("smooth_win_s", 0.12)
    smooth_poly = p.get("smooth_poly", 2)
    baseline_win_s = p.get("baseline_win_s", 3.0)
    thresh_k_mad = p.get("thresh_k_mad", 2.5)
    min_drop_abs = p.get("min_drop_abs", 0.05)
    min_drop_rel = p.get("min_drop_rel", 0.35)
    min_width_s = p.get("min_width_s", 0.06)
    max_width_s = p.get("max_width_s", 0.45)
    merge_gap_s = p.get("merge_gap_s", 0.15)
    refractory_s = p.get("refractory_s", 0.20)
    min_score = p.get("min_score", 0.30)
    plateau_policy = p.get("plateau_policy", "midpoint")

    # ---- sanitize
    x = np.asarray(x, float).ravel()
    n = x.size
    if n == 0:
        return pd.DataFrame({"start_blink": [], "end_blink": [], "pos_blink": []})
    if np.isnan(x).any():
        idx = np.where(~np.isnan(x))[0]
        if idx.size == 0:
            return pd.DataFrame({"start_blink": [], "end_blink": [], "pos_blink": []})
        nan_idx = np.where(np.isnan(x))[0]
        x[nan_idx] = np.interp(nan_idx, idx, x[idx])
    x = np.clip(x, -0.1, 1.0)

    # ---- smooth
    win = max(3, int(round(smooth_win_s * sfreq)))
    if win % 2 == 0:
        win += 1
    if _HAVE_SCIPY and win > smooth_poly + 2:
        xs = savgol_filter(x, window_length=win, polyorder=smooth_poly, mode="interp")
    else:
        k = max(3, win)
        ker = np.ones(k, float) / k
        xs = np.convolve(x, ker, mode="same")

    # ---- adaptive baseline & MAD
    b_win = max(3, int(round(baseline_win_s * sfreq)))
    if b_win % 2 == 0:
        b_win += 1

    if _HAVE_SCIPY:
        b = median_filter(xs, size=b_win, mode="nearest")
        mad = median_filter(np.abs(xs - b), size=b_win, mode="nearest") + 1e-9
    else:
        def roll_med(arr, k):
            half = k // 2
            out = np.empty_like(arr)
            for i in range(arr.size):
                lo = max(0, i - half)
                hi = min(arr.size, i + half + 1)
                out[i] = np.median(arr[lo:hi])
            return out
        b = roll_med(xs, b_win)
        mad = roll_med(np.abs(xs - b), b_win) + 1e-9

    drop = b - xs
    theta = b - thresh_k_mad * mad
    hard = np.maximum(min_drop_abs, min_drop_rel * np.clip(b, 1e-6, None))
    mask = (xs < theta) & (drop >= hard)

    # ---- segments from mask
    if not mask.any():
        return pd.DataFrame({"start_blink": [], "end_blink": [], "pos_blink": []})

    m_int = mask.astype(int)
    starts = np.where(np.diff(np.r_[0, m_int]) == 1)[0]
    ends = np.where(np.diff(np.r_[m_int, 0]) == -1)[0] - 1
    segs = list(zip(starts, ends))

    # merge close segments
    max_gap = int(round(merge_gap_s * sfreq))
    merged = []
    for s, e in segs:
        if not merged:
            merged.append([s, e])
        else:
            if s - merged[-1][1] - 1 <= max_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])

    min_w = int(round(min_width_s * sfreq))
    max_w = int(round(max_width_s * sfreq))
    refr = int(round(refractory_s * sfreq))

    picks = []
    last_accept = -10**9
    for s0, s1 in merged:
        w = s1 - s0 + 1
        if w < min_w or w > max_w:
            continue
        if s0 - last_accept < refr:
            continue

        seg = xs[s0:s1 + 1]
        k_rel = int(np.argmin(seg))
        k = s0 + k_rel

        # plateau policy
        eps = 1e-12
        left = k
        while left > s0 and abs(xs[left - 1] - xs[k]) <= eps:
            left -= 1
        right = k
        while right < s1 and abs(xs[right + 1] - xs[k]) <= eps:
            right += 1
        if right > left:
            if plateau_policy == "midpoint":
                k = (left + right) // 2
            elif plateau_policy == "first_min":
                k = left
            elif plateau_policy == "last_min":
                k = right

        amp_norm = float(drop[k] / (mad[k] if mad[k] > 0 else 1.0))
        sharp = max(0.0, xs[max(k - 1, s0)] - 2 * xs[k] + xs[min(k + 1, s1)])
        width_pref = 1.0 - abs((w / sfreq) - 0.20) / 0.20
        width_pref = float(np.clip(width_pref, 0.0, 1.0))
        score = 0.5 * np.tanh(amp_norm / 3.0) + 0.3 * np.clip(sharp, 0.0, 1.0) + 0.2 * width_pref

        if score >= min_score:
            picks.append((s0, s1, int(k)))
            last_accept = k

    return pd.DataFrame(picks, columns=["start_blink", "end_blink", "pos_blink"])
