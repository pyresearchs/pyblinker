"""EAR blink detection package: pluggable algorithms via registry."""
import logging

from . import algos_adaptive as _algos_adaptive  # noqa: F401
from . import algos_calibrated as _algos_calibrated  # noqa: F401
from .calibration import compute_ear_calibration  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["compute_ear_calibration"]
