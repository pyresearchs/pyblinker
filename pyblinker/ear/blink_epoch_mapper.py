"""EAR blink position detection (dispatcher) and public entry point.

This module exposes ``_get_blink_position_epoching_ear``, which dispatches to a
selected algorithm (registered in :mod:`pyblinker.ear.registry`).

Algorithms must return a :class:`pandas.DataFrame` with columns (samples)::

    - start_blink
    - end_blink
    - pos_blink

This function will append the time columns (seconds)::

    - start_time, end_time, pos_time

###  Pipeline Flowchart

```plaintext
                   +-------------+
                   | EAR signal  |
                   +-------------+
                          |
                          v
               +-----------------------+
               | Choose algorithm      |
               | (adaptive MAD or      |
               |  calibrated threshold)|
               +-----------------------+
                          |
                          v
                   +----------------+
                   | Blink DataFrame|
                   +----------------+
```
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd

try:  # optional; logger may live elsewhere in your tree
    from ...utils._logging import logger  # type: ignore
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)

from .registry import available_algos, choose_algo


def _ensure_algos_loaded() -> None:
    """Ensure algorithm modules are imported so decorators register them."""
    try:
        from . import algos_adaptive as _a  # noqa: F401
        from . import algos_calibrated as _b  # noqa: F401
    except Exception as exc:  # pragma: no cover
        logger.debug("EAR algos import skipped/failed: %s", exc)


def _get_blink_position_epoching_ear(
    signal: np.ndarray,
    params: dict,
    ch: Optional[str] = None,
    *,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """Detect blink positions from a 1-D EAR signal (downward valleys).

    Parameters
    ----------
    signal : numpy.ndarray
        1-D EAR signal (flattened if needed).
    params : dict
        Must contain ``sfreq`` or be supplied by the caller upstream.
        Optional keys:
        - ``ear_algo``: name of the algorithm or ``"auto"``.
        - ``calibration``: ``{"open": float, "closed": float}``. These
          thresholds may come from :func:`compute_ear_calibration` or be
          provided manually when experimenting with different levels.
        - Any algorithm-specific parameters.
    ch : str | None
        Channel name for logging only.
    progress_bar : bool
        Unused here; kept for API parity with EEG/EOG path.

    Returns
    -------
    pandas.DataFrame
        Columns (samples): ``start_blink``, ``end_blink``, ``pos_blink``
        Columns (seconds): ``start_time``, ``end_time``, ``pos_time``
    """
    logger.info("Starting EAR blink detection on %s", ch)
    _ensure_algos_loaded()

    x = np.asarray(signal, float).ravel()
    sfreq = float(params.get("sfreq", np.nan))
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("params['sfreq'] must be a positive float for EAR detection")

    name = choose_algo(params)
    algos = available_algos()
    if name not in algos:
        raise ValueError(f"Unknown EAR algo '{name}'. Available: {sorted(algos)}")

    df = algos[name](x, sfreq, params)
    if df is None:
        df = pd.DataFrame({"start_blink": [], "end_blink": [], "pos_blink": []})

    if not {"start_blink", "end_blink", "pos_blink"} <= set(df.columns):
        raise ValueError(
            "Algorithm must return columns: start_blink, end_blink, pos_blink"
        )

    df = df.astype({"start_blink": int, "end_blink": int, "pos_blink": int})
    df["start_time"] = df["start_blink"] / sfreq
    df["end_time"] = df["end_blink"] / sfreq
    df["pos_time"] = df["pos_blink"] / sfreq

    logger.info("EAR algo=%s detected %d blinks on %s", name, len(df), ch)
    return df
