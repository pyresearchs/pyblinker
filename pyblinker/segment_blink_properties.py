"""Segment-level blink property extraction utilities.

This module exposes :func:`compute_segment_blink_properties`, which can operate
on either legacy ``mne.Raw`` segments with an accompanying blink annotation
table or directly on :class:`mne.Epochs` produced by
``slice_raw_into_mne_epochs_refine_annot``.  In the latter case the epoch
metadata provides refined blink windows that drive the computation.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any, List
import logging
import warnings

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

from .blinker.fit_blink import FitBlinks
from .blinker.extract_blink_properties import BlinkProperties
from .blinker.base_left_right import create_left_right_base
from .blink_features.blink_events.blink_dataframe import left_right_zero_crossing
from .blink_features.waveform_features.aggregate import _sample_windows_from_metadata

logger = logging.getLogger(__name__)


def compute_segment_blink_properties(
    segments: Sequence[mne.io.BaseRaw] | mne.Epochs,
    blink_df: pd.DataFrame | None,
    params: Dict[str, Any],
    *,
    channel: str | Sequence[str] = "EEG-E8",
    run_fit: bool = False,
    progress_bar: bool = True,
) -> mne.Epochs | pd.DataFrame:
    """Calculate blink properties from raw segments or refined epochs.

    Parameters
    ----------
    segments : sequence of mne.io.BaseRaw | mne.Epochs
        Either a sequence of Raw segments with legacy blink annotations or an
        :class:`mne.Epochs` instance returned by
        :func:`slice_raw_into_mne_epochs_refine_annot`.
    blink_df : pandas.DataFrame | None
        Legacy blink annotation table.  Must be ``None`` when ``segments`` is an
        :class:`mne.Epochs` instance as blink metadata are sourced from
        ``epochs.metadata``.
    params : dict
        Parameter dictionary forwarded to :class:`BlinkProperties`.  Required
        keys include ``"shut_amp_fraction"``, ``"p_avr_threshold"`` and
        ``"z_thresholds"``.
    channel : str | sequence of str, optional
        Channel name(s) used for property extraction.  Defaults to ``"EEG-E8"``.
    run_fit : bool, optional
        When using the legacy path this flag controls whether blink fitting is
        executed.  It is ignored for the refined-epoch workflow.
    progress_bar : bool, optional
        Whether to display a progress bar during processing.

    Returns
    -------
    mne.Epochs | pandas.DataFrame
        When ``segments`` is an :class:`mne.Epochs` instance, an
        :class:`~mne.Epochs` object is returned where each epoch represents a
        single blink and ``epochs.metadata`` holds the computed blink
        properties including ``seg_id`` and ``blink_id``. When legacy raw
        segments are provided, a ``pandas.DataFrame`` matching the historic
        behaviour is returned.
    """

    # --- refined epoch workflow -------------------------------------------------
    if isinstance(segments, mne.Epochs) and blink_df is None:
        epochs = segments
        if isinstance(channel, str):
            ch_names = [channel]
        else:
            ch_names = list(channel)

        missing = [ch for ch in ch_names if ch not in epochs.ch_names]
        if missing:
            raise ValueError(f"Channels not found: {missing}")

        sfreq = float(epochs.info["sfreq"])
        n_epochs = len(epochs)
        n_times = (
            epochs.get_data(picks=[ch_names[0]]).shape[-1] if n_epochs else 0
        )

        data = epochs.get_data(picks=ch_names)
        records: List[pd.DataFrame] = []
        logger.info("Computing blink properties for %d epochs", n_epochs)

        for ei in tqdm(range(n_epochs), desc="Epochs", disable=not progress_bar):
            metadata_row = (
                epochs.metadata.iloc[ei]
                if isinstance(epochs.metadata, pd.DataFrame)
                else pd.Series(dtype=float)
            )
            for ci, ch in enumerate(ch_names):
                sample_windows = _sample_windows_from_metadata(
                    metadata_row, ch, sfreq, n_times, ei
                )
                if not sample_windows:
                    continue
                signal = data[ei, ci]
                mod = (
                    "eeg"
                    if "eeg" in ch.lower()
                    else ("eog" if "eog" in ch.lower() else "ear")
                )
                outer_starts = metadata_row.get(f"blink_outer_start_{mod}", [])
                outer_ends = metadata_row.get(f"blink_outer_end_{mod}", [])
                if not isinstance(outer_starts, (list, tuple, np.ndarray, pd.Series)):
                    outer_starts = [outer_starts]
                if not isinstance(outer_ends, (list, tuple, np.ndarray, pd.Series)):
                    outer_ends = [outer_ends]
                blink_rows = []
                for bi, sl in enumerate(sample_windows):
                    start = sl.start
                    end = sl.stop - 1
                    segment = signal[start : end + 1]
                    peak_offset = (
                        int(np.argmin(segment)) if "ear" in ch.lower() else int(np.argmax(segment))
                    )
                    max_blink = start + peak_offset
                    max_value = signal[max_blink]
                    outer_start = (
                        int(outer_starts[bi]) if bi < len(outer_starts) else start
                    )
                    outer_end = (
                        int(outer_ends[bi]) if bi < len(outer_ends) else end
                    )
                    blink_rows.append(
                        {
                            "start_blink": start,
                            "end_blink": end,
                            "outer_start": outer_start,
                            "outer_end": outer_end,
                            "max_blink": max_blink,
                            "max_value": max_value,
                        }
                    )
                rows = pd.DataFrame.from_records(blink_rows)
                if rows.empty:
                    continue
                def _zeros(row: pd.Series) -> tuple[float, float | None]:
                    try:
                        return left_right_zero_crossing(
                            signal,
                            row["max_blink"],
                            row["outer_start"],
                            row["outer_end"],
                        )
                    except Exception:
                        return (np.nan, np.nan)

                zeros = rows.apply(_zeros, axis=1, result_type="expand")
                rows["left_zero"] = zeros[0]
                rows["right_zero"] = zeros[1]
                fitter = FitBlinks(candidate_signal=signal, df=rows.copy(), params=params)
                try:
                    fitter.dprocess_segment_raw(run_fit=run_fit)
                except Exception:  # pragma: no cover - safeguard
                    pass
                try:
                    frame_blinks = create_left_right_base(candidate_signal=signal, df=rows)
                except ValueError:
                    continue
                props = BlinkProperties(
                    signal, frame_blinks, sfreq, params, fitted=run_fit
                ).df
                props["seg_id"] = ei
                props["blink_id"] = range(len(props))
                records.append(props)

        if not records:
            info = mne.create_info(ch_names, sfreq)
            empty_md = pd.DataFrame()
            return mne.EpochsArray(
                np.zeros((0, len(ch_names), 1)), info, metadata=empty_md
            )

        result = pd.concat(records, ignore_index=True)
        info = mne.create_info(ch_names, sfreq)
        dummy = np.zeros((len(result), len(ch_names), 1), dtype=float)
        blink_epochs = mne.EpochsArray(dummy, info, metadata=result)
        logger.info("Computed blink properties for %d blinks", len(result))
        return blink_epochs

    # --- legacy raw-segment workflow --------------------------------------------
    if run_fit:
        warnings.warn(
            "run_fit=True may drop blinks due to NaNs in fit range",
            RuntimeWarning,
        )

    if blink_df is None or blink_df.empty:
        logger.info("Blink DataFrame is empty; nothing to compute")
        return pd.DataFrame()

    if not segments:
        return pd.DataFrame()

    sfreq = segments[0].info["sfreq"]
    all_props = []
    logger.info("Computing blink properties for %d segments", len(segments))

    for seg_id, raw in enumerate(
        tqdm(segments, desc="Segments", disable=not progress_bar)
    ):
        rows = blink_df[blink_df["seg_id"] == seg_id].copy()
        rows["start_blink"] = rows["start_blink"].astype(int)
        rows["end_blink"] = rows["end_blink"].astype(int)
        rows["outer_start"] = rows["outer_start"].astype(int)
        rows["outer_end"] = rows["outer_end"].astype(int)
        rows["left_zero"] = rows["left_zero"].astype(int)
        if "right_zero" in rows.columns:
            rows["right_zero"] = rows["right_zero"].fillna(-1).astype(int)
        if rows.empty:
            continue

        signal = raw.get_data(picks=channel)[0]

        fitter = FitBlinks(candidate_signal=signal, df=rows, params=params)
        try:
            fitter.dprocess_segment_raw(run_fit=run_fit)
        except Exception as exc:  # pragma: no cover - safeguard against bad data
            logger.warning("Skipping segment %d due to fit error: %s", seg_id, exc)
            continue

        if fitter.frame_blinks.empty:
            logger.warning(
                "Skipping segment %d due to no valid blink frames after fitting",
                seg_id,
            )
            continue

        props = BlinkProperties(
            signal,
            fitter.frame_blinks,
            sfreq,
            params,
            fitted=run_fit,
        ).df
        props["seg_id"] = seg_id
        all_props.append(props)

    if not all_props:
        return pd.DataFrame()

    result = pd.concat(all_props, ignore_index=True)
    logger.info("Computed blink properties for %d blinks", len(result))
    return result
