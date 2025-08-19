import numpy as np

"""
Default parameters for blink detection and analysis.

This module contains a set of parameters used for configuring the blink detection
algorithm. These parameters are crucial for tuning the performance of the algorithm
based on the characteristics of the input candidate_signal.

Parameters
----------
std_threshold : float
    The standard deviation threshold for identifying blinks. Blinks with a standard
    deviation greater than this value will be considered outliers.

min_event_len : float
    The minimum length of a blink event in seconds. Events shorter than this will
    be discarded.

min_event_sep : float
    The minimum separation time between consecutive blink events in seconds. Events
    that occur closer together than this threshold will be merged.

base_fraction : float
    The fraction of the baseline signal used to compute the baseline for blink events.

correlation_threshold_top : float
    The upper threshold for correlation, above which blink candidates are considered
    highly correlated.

correlation_threshold_bottom : float
    The lower threshold for correlation, below which blink candidates are considered
    poorly correlated.

correlation_threshold_middle : float
    The middle threshold for correlation, used for intermediate correlation assessments.

shut_amp_fraction : float
    The fraction of the amplitude used to determine the shut-off point for blink events.

blink_amp_range_1 : float
    The lower bound of the amplitude range for valid blinks.

blink_amp_range_2 : float
    The upper bound of the amplitude range for valid blinks.

good_ratio_threshold : float
    The threshold for the ratio of good blinks to total blinks. A value below this
    threshold indicates a poor quality of blink detection.

min_good_blinks : int
    The minimum number of good blinks required for a valid analysis.

keep_signals : int
    A flag indicating whether to keep the original signals (1) or discard them (0).

correlation_threshold : float
    The threshold for correlation used in the analysis. Blinks with correlation values
    below this threshold will be considered unreliable.

p_avr_threshold : float
    The threshold for the amplitude-velocity ratio (pAVR). Blink candidates with a pAVR
    value less than or equal to this threshold are likely to be saccades rather than
    normal blinks.

z_thresholds : numpy.ndarray
    A 2D array containing the z-score thresholds for blink detection. The first row
    contains the lower thresholds, and the second row contains the upper thresholds.

sfreq : int
    The sampling frequency of the candidate_signal in Hz. This value is used for time-related
    calculations in the blink detection algorithm.
"""

SCALING_FACTOR = 1.4826  # From original paper: by default, BLINKER eliminates
# “best blinks” more than five robust standard deviations from the median and
# “good” blinks more than two robust standard deviations away from this median.
# Here we define the robust standard deviation as 1.4826 times the median
# absolute deviation from the median.

DEFAULT_PARAMS = {
    'std_threshold': 1.50,
    'min_event_len': 0.05,
    'min_event_sep': 0.05,
    'base_fraction': 0.1,
    'correlation_threshold_top': 0.980,
    'correlation_threshold_bottom': 0.90,
    'correlation_threshold_middle': 0.95,
    'shut_amp_fraction': 0.9,
    'blink_amp_range_1': 3,
    'blink_amp_range_2': 50,
    'good_ratio_threshold': 0.7,
    'min_good_blinks': 10,
    'keep_signals': 0,
    'correlation_threshold': 0.98,
    'p_avr_threshold': 3,  # from original paper: The pAVR criterion captures
    # the difference between the sharp rising edge of saccades and the more
    # curved rise of normal blinks. We have found empirically that blink
    # candidates with pAVR ≤ 3 do not correspond to normal blinks, but rather
    # saccades having short, fast eye movements
    'z_thresholds': np.array([[0.9, 0.98], [2.0, 5.0]]),
    'sfreq': 100,
}
