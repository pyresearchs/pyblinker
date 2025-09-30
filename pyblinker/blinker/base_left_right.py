import numpy as np

from pyblinker.logging import get_logger

from .stroke_utils import max_pos_vel_frame


logger = get_logger(__name__)


def get_left_base(blink_velocity, left_outer, max_pos_vel_frame):
    """Determine the left base frame index.

    The left base is identified by scanning from ``max_pos_vel_frame`` back to
    ``left_outer`` until the blink velocity becomes non-positive. The index of
    the first frame satisfying the condition is returned.
    """

    l_outer = int(left_outer)
    m_pos_vel = int(max_pos_vel_frame)

    left_range = np.arange(l_outer, m_pos_vel + 1)
    reversed_velocity = np.flip(blink_velocity[left_range])

    left_base_index = np.argmax(reversed_velocity <= 0)
    left_base = m_pos_vel - left_base_index - 1
    return left_base


def get_right_base(
    candidate_signal: np.ndarray,
    blink_velocity: np.ndarray,
    right_outer: int,
    max_neg_vel_frame: float | int,
) -> int | float | None:
    """Compute the right base frame index.

    Parameters
    ----------
    candidate_signal : numpy.ndarray
        The filtered blink signal from which baseline metrics are derived.
    blink_velocity : numpy.ndarray
        First derivative of ``candidate_signal`` used to locate zero crossings.
    right_outer : int
        Right boundary frame of the blink segment.
    max_neg_vel_frame : float | int
        Frame index corresponding to the most negative blink velocity.

    Returns
    -------
    int | float | None
        The frame index of the right base. Returns ``numpy.nan`` when
        ``max_neg_vel_frame`` is ``NaN`` and ``None`` when a valid range cannot
        be determined.
    """

    r_outer = int(right_outer)

    # Return ``NaN`` when no negative velocity peak exists.
    if np.isnan(max_neg_vel_frame):
        return np.nan
    m_neg_vel = int(max_neg_vel_frame)

    # Ensure boundaries are valid
    if m_neg_vel > r_outer:
        return None

    max_size = candidate_signal.size
    end_idx = min(r_outer, max_size)
    right_range = np.arange(m_neg_vel, end_idx)

    if right_range.size == 0:
        return None

    # Avoid out-of-bounds indexing for blink_velocity
    if right_range[-1] >= blink_velocity.size:
        right_range = right_range[:-1]
        if right_range.size == 0 or right_range[-1] >= blink_velocity.size:
            logger.warning(
                "Unable to compute right base: right_range %s exceeds blink_velocity length %d",
                right_range,
                blink_velocity.size,
            )
            return None

    right_base_velocity = blink_velocity[right_range]
    right_base_index = np.argmax(right_base_velocity >= 0)
    right_base = m_neg_vel + right_base_index + 1
    return right_base


def create_left_right_base(candidate_signal, df):
    """
    Computes the left and right base values for each row in the DataFrame df,
    using the blink velocity derived from the input signal candidate_signal. The function
    also adds columns for the maximum positive and negative velocity frames to df.

    Parameters
    ----------
    candidate_signal : numpy.ndarray
        A 1D array of signal candidate_signal representing the blink component. The length
        of this array must match the number of rows in the DataFrame df, as it
        is used to compute the blink velocity.

    df : pandas.DataFrame
        A DataFrame containing the following required columns:
        - 'max_blink' (int): The maximum frame index for the blink event.
        - 'left_zero' (int): The index of the left zero crossing.
        - 'right_zero' (int): The index of the right zero crossing.
        - 'outer_start' (int): The starting index of the outer blink event.
        - 'outer_end' (int): The ending index of the outer blink event.
        Additional columns may be present but are not utilized in this function.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the following additional columns:
        - 'max_pos_vel_frame' (int): The frame index of the maximum positive velocity
          calculated from the blink velocity.
        - 'max_neg_vel_frame' (int): The frame index of the maximum negative velocity
          calculated from the blink velocity.
        - 'left_base' (float): The calculated left base value for the blink event,
          derived from the blink velocity and the specified outer start index.
        - 'right_base' (float): The calculated right base value for the blink event,
          derived from the blink velocity and the specified outer end index.
        Rows with NaN values in any of these new columns are dropped from the DataFrame.

    Raises
    ------
    ValueError
        If all rows are removed after NaN filtering, indicating that there are
        no valid blink frames to compute baselines for the current segment.
    """

    # Ensure df is a fresh copy to prevent SettingWithCopyWarning
    df = df.copy()

    # Compute blink velocity by differencing the candidate_signal
    blink_velocity = np.diff(candidate_signal, axis=0)

    # Remove rows with NaNs so we don't pass invalid candidate_signal to our calculations
    df.dropna(inplace=True)

    # Calculate maxPosVelFrame and maxNegVelFrame safely
    df[["max_pos_vel_frame", "max_neg_vel_frame"]] = df.apply(
        lambda row: max_pos_vel_frame(
            blink_velocity=blink_velocity,
            max_blink=row["max_blink"],
            left_zero=row["left_zero"],
            right_zero=row["right_zero"],
        ),
        axis=1,
        result_type="expand",
    )

    # Ensure df is a new variable after filtering
    df = df[df["outer_start"] < df["max_pos_vel_frame"]].copy()

    # Compute leftBase safely using .assign()
    df = df.assign(
        left_base=df.apply(
            lambda row: get_left_base(
                blink_velocity=blink_velocity,
                left_outer=row["outer_start"],
                max_pos_vel_frame=row["max_pos_vel_frame"],
            ),
            axis=1,
        )
    )

    # Drop rows with NaNs again if any were introduced
    df.dropna(inplace=True)

    """If all rows were removed, there are no blink frames left to process."""
    # Downstream calculations require at least one blink frame. Raising an
    # exception allows the calling pipeline to skip this segment gracefully.
    if df.empty:
        raise ValueError("No valid blink frames after baseline computation")

    # Compute rightBase safely using .assign()
    df = df.assign(
        right_base=df.apply(
            lambda row: get_right_base(
                candidate_signal=candidate_signal,
                blink_velocity=blink_velocity,
                right_outer=row["outer_end"],
                max_neg_vel_frame=row["max_neg_vel_frame"],
            ),
            axis=1,
        )
    )

    return df
