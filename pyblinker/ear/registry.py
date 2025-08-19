from typing import Callable, Dict
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Central registry for EAR blink detection algorithms
_ALGOS: Dict[str, Callable[[np.ndarray, float, dict], pd.DataFrame]] = {}


def register_ear_algo(name: str):
    """Decorator to register an EAR detection algorithm by name.

    Each registered function must have the signature::

        fn(x: np.ndarray, sfreq: float, params: dict) -> pd.DataFrame

    and return a DataFrame with columns: ``start_blink``, ``end_blink``,
    ``pos_blink`` (samples).
    """
    def _wrap(fn):
        _ALGOS[name] = fn
        return fn
    return _wrap


def available_algos() -> Dict[str, Callable]:
    """Return a mapping of available algorithm names to callables."""
    return dict(_ALGOS)


def choose_algo(params: dict) -> str:
    """Choose an algorithm name based on provided parameters."""
    algo = (params or {}).get("ear_algo", "auto")
    if algo != "auto":
        return algo
    calib = (params or {}).get("calibration", {})
    has_cal = isinstance(calib, dict) and {"open", "closed"} <= set(calib)
    return "calibrated_threshold" if has_cal else "adaptive_mad"
