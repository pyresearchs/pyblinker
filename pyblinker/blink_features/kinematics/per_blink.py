"""Per-blink kinematic metrics."""
from typing import Any, Dict

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_blink_kinematics(blink: Dict[str, Any], sfreq: float) -> Dict[str, float]:
    """Compute kinematic quantities for a single blink.

    Blink kinematics capture the dynamics of eyelid motion. Peak velocity and
    acceleration decrease and movements become smoother when drivers grow
    fatigued, while the amplitude-velocity ratio (AVR) rises. These changes have
    been linked to increased crash risk in driving studies.

    Parameters
    ----------
    blink : dict
        Blink annotation containing ``refined_start_frame``, ``refined_peak_frame``,
        ``refined_end_frame`` and ``epoch_signal``.
    sfreq : float
        Sampling frequency of the recording in Hertz.

    Returns
    -------
    dict
        Dictionary with peak velocity, acceleration, jerk and AVR for the blink.
    """
    start = int(blink["refined_start_frame"])
    end = int(blink["refined_end_frame"])
    signal = np.asarray(blink["epoch_signal"], dtype=float)

    segment = signal[start : end + 1]
    dt = 1.0 / sfreq
    # Ensure the segment is long enough to compute gradients
    if len(segment) < 2:
        logger.warning("Segment too short to compute kinematics. Returning NaNs.")
        return {
            "v_max": float("nan"),
            "a_max": float("nan"),
            "j_max": float("nan"),
            "avr": float("nan"),
        }

    velocity = np.gradient(segment, dt)
    acceleration = np.gradient(velocity, dt)
    jerk = np.gradient(acceleration, dt)

    abs_velocity = np.abs(velocity)
    abs_acceleration = np.abs(acceleration)
    abs_jerk = np.abs(jerk)

    v_max = float(np.max(abs_velocity)) if abs_velocity.size > 0 else float("nan")
    a_max = float(np.max(abs_acceleration)) if abs_acceleration.size > 0 else float("nan")
    j_max = float(np.max(abs_jerk)) if abs_jerk.size > 0 else float("nan")

    amplitude = signal[start] - float(np.min(segment))
    avr = amplitude / v_max if v_max != 0 and not np.isnan(v_max) else float("nan")

    logger.debug(
        "Blink kinematics computed: v_max=%s, a_max=%s, j_max=%s, avr=%s",
        v_max,
        a_max,
        j_max,
        avr,
    )

    return {
        "v_max": v_max,
        "a_max": a_max,
        "j_max": j_max,
        "avr": avr,
    }
