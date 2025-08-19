"""Blink refinement utilities for EEG/EOG signals."""
import logging
from typing import Sequence, Dict, Any, Callable, List, Optional
from typing import Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np

logger = logging.getLogger(__name__)


def refine_ear_extrema_and_threshold_stub(
    signal_segment: np.ndarray,
    start_rel: int,
    end_rel: int,
    peak_rel_cvat: int | None = None,
    *,
    local_max_prominence: float = 0.01,
    search_expansion_frames: int = 5,
    value_threshold: float | None = None,
) -> Tuple[int, int, int]:
    """Return a crude EAR trough refinement.

    Parameters mirror those of the real refinement routine but the
    implementation merely validates that indices are within bounds and
    estimates a trough location if ``peak_rel_cvat`` is not supplied.

    Returns
    -------
    tuple
        ``(start_frame, trough_frame, end_frame)`` indices within the segment.
    """

    valid_trough = peak_rel_cvat
    if not (peak_rel_cvat is not None and 0 <= peak_rel_cvat < len(signal_segment)):
        if end_rel >= start_rel and len(signal_segment) > 0:
            valid_trough = (start_rel + end_rel) // 2
        else:
            valid_trough = 0

    rs_stub = max(0, min(start_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    re_stub = max(0, min(end_rel, len(signal_segment) - 1 if len(signal_segment) > 0 else 0))
    if rs_stub > re_stub:
        rs_stub = re_stub

    return rs_stub, valid_trough, re_stub




def refine_local_maximum_stub(
        signal_segment: np.ndarray,
        start_rel: int,
        end_rel: int,
        peak_rel_cvat: int | None = None,
) -> Tuple[int, int, int]:
    """Return a crude refinement for local maxima in a signal segment.
        This is only suitable for EEG and EOG signals where blinks are detected.
        Eye Aspect Ratio (EAR) is not applicable here, unless it is being inverted.
    Finds start, peak, and end frame indices within the signal segment.
    The peak is the index of the maximum value between start_rel and end_rel.

    Parameters:
    - signal_segment: The signal segment array.
    - start_rel: Relative start index.
    - end_rel: Relative end index.
    - peak_rel_cvat: Optional initial guess for the peak index. If not within bounds, it is ignored.

    Returns:
    - Tuple of (start_frame, peak_frame, end_frame), all relative to the segment.
    """
    # n = len(signal_segment)
    n = len(signal_segment)
    if n == 0:
        return 0, 0, 0

    rs_stub = max(0, min(start_rel, n - 1))
    re_stub = max(0, min(end_rel, n - 1))
    if rs_stub > re_stub:
        rs_stub = re_stub = min(rs_stub, re_stub)

    if peak_rel_cvat is not None and rs_stub <= peak_rel_cvat <= re_stub:
        valid_peak = peak_rel_cvat
    else:
        segment = signal_segment[rs_stub : re_stub + 1]
        max_idx_local = int(np.argmax(segment))
        valid_peak = rs_stub + max_idx_local

    return rs_stub, valid_peak, re_stub



def plot_refined_blinks(
    refined_blinks: Sequence[Dict[str, Any]],
    sfreq: float,
    epoch_len: float,
    *,
    epoch_indices: Optional[Sequence[int]] = None,
    show: bool = False,
) -> List[plt.Figure]:
    """Plot signal segments with refined blink markers.

    Parameters
    ----------
    refined_blinks : sequence of dict
        Output from :func:`refine_blinks_from_epochs`.
    sfreq : float
        Sampling frequency of the signals.
    epoch_len : float
        Duration of each epoch in seconds.
    epoch_indices : sequence of int | None, optional
        Specific epochs to plot. If ``None`` all epochs containing blinks are
        shown.
    show : bool, optional
        Whether to display the figures using ``plt.show``. Defaults to ``False``.

    Returns
    -------
    list of matplotlib.figure.Figure
        Figure objects created for each plotted epoch.
    """

    epochs_to_plot: Dict[int, Dict[str, Any]] = {}
    for blink in refined_blinks:
        idx = blink["epoch_index"]
        if epoch_indices is None or idx in epoch_indices:
            if idx not in epochs_to_plot:
                epochs_to_plot[idx] = {"signal": blink["epoch_signal"], "blinks": []}
            epochs_to_plot[idx]["blinks"].append(blink)

    if not epochs_to_plot:
        logger.warning("No epochs selected for plotting")
        return []

    figs: List[plt.Figure] = []
    time_axis = np.arange(0, epoch_len, 1.0 / sfreq)
    for epoch_index, data in epochs_to_plot.items():
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_axis, data["signal"], label="Signal")
        for blink in data["blinks"]:
            start_t = blink["refined_start_frame"] / sfreq
            peak_t = blink["refined_peak_frame"] / sfreq
            end_t = blink["refined_end_frame"] / sfreq
            ax.axvline(start_t, color="g", linestyle="--")
            ax.axvline(peak_t, color="r")
            ax.axvline(end_t, color="b", linestyle="--")
        ax.set_title(f"Epoch {epoch_index}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        figs.append(fig)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs


def refine_blinks_from_epochs(
    segments: Sequence[mne.io.BaseRaw],
    channel: str,
    *,
    refine_func: Callable[[np.ndarray, int, int, int | None], Tuple[int, int, int]] = refine_local_maximum_stub,
    local_max_prominence: float = 0.01,
    search_expansion_frames: int | None = None,
    value_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Refine blink annotations within pre-sliced raw segments.

    Parameters
    ----------
    segments : sequence of mne.io.BaseRaw
        Segments produced by :func:`slice_into_mini_raws` containing annotations.
    channel : str
        Channel name used for refinement.
    refine_func : callable, optional
        Refinement function taking ``(signal_segment, start_rel, end_rel, peak_rel_cvat)``
        and returning ``(start, peak, end)``. Defaults to
        :func:`refine_local_maximum_stub`.
    local_max_prominence : float, optional
        Parameter forwarded to ``refine_func``.
    search_expansion_frames : int | None, optional
        Expansion frames for ``refine_func``. When ``None`` this defaults to
        ``int(0.1 * sfreq)``.
    value_threshold : float | None, optional
        Threshold parameter for ``refine_func``.

    Returns
    -------
    list of dict
        Refined blink annotations with keys ``epoch_index``,
        ``epoch_signal``, ``refined_start_frame``, ``refined_peak_frame`` and
        ``refined_end_frame``.
    """
    logger.info("Refining blinks across %d segments", len(segments))
    refined: List[Dict[str, Any]] = []
    if not segments:
        return refined
    sfreq = segments[0].info["sfreq"]
    if search_expansion_frames is None:
        search_expansion_frames = int(0.1 * sfreq)

    for epoch_index, raw in enumerate(segments):
        signal = raw.get_data(picks=channel)[0]
        for ann in raw.annotations:
            onset = float(ann["onset"]) - raw.first_time
            start_frame = int(round(onset * sfreq))
            end_frame = int(round((onset + float(ann["duration"])) * sfreq))
            r_start, r_peak, r_end = refine_func(
                signal,
                start_frame,
                end_frame,
                None,
            )
            refined.append(
                {
                    "epoch_index": epoch_index,
                    "epoch_signal": signal,
                    "refined_start_frame": r_start,
                    "refined_peak_frame": r_peak,
                    "refined_end_frame": r_end,
                }
            )
    logger.info("Refined %d blink annotations", len(refined))
    return refined
