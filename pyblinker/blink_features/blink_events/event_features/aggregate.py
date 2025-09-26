"""Aggregate blink event features using :class:`mne.Epochs` metadata."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from pyblinker.logging import get_logger
from pyblinker.segment_blink_properties import compute_segment_blink_properties
from pyblinker.utils import normalize_picks, require_channels
from pyblinker.utils.refinement_utils import slice_raw_into_mne_epochs_refine_annot

from ...energy.energy_features import compute_energy_features
from ...frequency_domain.aggregate import aggregate_frequency_domain_features
from ...kinematics.kinematic_features import compute_kinematic_features
from ...morphology.epoch_features import compute_epoch_morphology_features
from .blink_count import blink_count
from .inter_blink_interval import inter_blink_interval_epochs

logger = get_logger(__name__)

_DEFAULT_WAVEFORM_PARAMS = {
    "base_fraction": 0.5,
    "shut_amp_fraction": 0.9,
    "p_avr_threshold": 3,
    "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
}


def _get_epoch_index(epochs: mne.Epochs) -> pd.Index:
    """Return the canonical epoch index used for feature alignment."""

    if isinstance(epochs.metadata, pd.DataFrame):
        index = pd.Index(epochs.metadata.index, name="epoch")
    else:
        index = pd.RangeIndex(len(epochs), name="epoch")
    return index


def _get_modality_channels(info: mne.Info, modality: str) -> list[str]:
    """Collect channel names that belong to ``modality``."""

    modality_upper = modality.upper()
    channels: list[str] = []
    for ch_name in info["ch_names"]:
        ch_type = info.get_channel_types(picks=[ch_name])[0]
        ch_lower = ch_name.lower()
        if modality_upper == "EEG" and ch_type == "eeg":
            channels.append(ch_name)
        elif modality_upper == "EOG" and ch_type == "eog":
            channels.append(ch_name)
        elif modality_upper == "EAR":
            if "ear" in ch_lower:
                channels.append(ch_name)
            elif ch_type == "misc" and "ear" in ch_lower:
                channels.append(ch_name)
    return channels


def _align_to_epoch_index(df: pd.DataFrame, epoch_index: pd.Index) -> pd.DataFrame:
    """Align ``df`` to the canonical ``epoch_index``."""

    if df is None or df.empty:
        return pd.DataFrame(index=epoch_index)

    aligned = df.copy()
    if "ep" in aligned.columns:
        aligned = aligned.set_index("ep")

    aligned.index = pd.Index(aligned.index, name="epoch")
    aligned = aligned.reindex(epoch_index)
    return aligned


def _namespace_df(df: pd.DataFrame, modality: str, family: str) -> pd.DataFrame:
    """Namespace columns using ``{modality}__{family}__`` pattern."""

    if df.empty:
        return df

    namespaced = df.copy()
    namespaced.columns = [f"{modality}__{family}__{col}" for col in namespaced.columns]
    return namespaced


def _rename_event_columns(columns: Iterable[str], modality: str) -> dict[str, str]:
    """Remove modality suffixes from blink totals/rates to avoid duplicates."""

    suffix = f"_{modality.lower()}"
    rename: dict[str, str] = {}
    for col in columns:
        if col.startswith("blink_total") and col.endswith(suffix):
            rename[col] = col[: -len(suffix)]
        elif col.startswith("blink_rate") and col.endswith(suffix):
            rename[col] = col[: -len(suffix)]
    return rename


def _compute_waveform_epoch_features(
    epochs: mne.Epochs,
    picks: Sequence[str],
    params: Mapping[str, Any],
    *,
    run_fit: bool,
    progress_bar: bool,
) -> pd.DataFrame:
    """Aggregate waveform features per epoch for the provided channels."""

    epoch_index = _get_epoch_index(epochs)
    if not picks:
        return pd.DataFrame(index=epoch_index)

    params = {**_DEFAULT_WAVEFORM_PARAMS, **dict(params)}
    frames: list[pd.DataFrame] = []
    position_to_epoch = {pos: epoch_index[pos] for pos in range(len(epoch_index))}

    for ch in picks:
        epochs_ch = epochs.copy().pick(ch, verbose=False)
        try:
            blink_df = compute_segment_blink_properties(
                epochs_ch,
                params,
                channel=ch,
                run_fit=run_fit,
                progress_bar=progress_bar,
                long_format=True,
            )
        except Exception as exc:  # pragma: no cover - logged for diagnosis
            logger.warning(
                "Waveform feature computation failed for channel %s: %s",
                ch,
                exc,
            )
            continue

        if not isinstance(blink_df, pd.DataFrame) or blink_df.empty:
            continue
        if "seg_id" not in blink_df.columns:
            continue

        numeric = blink_df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in ("seg_id", "blink_id") if c in blink_df.columns],
            errors="ignore",
        )
        if numeric.empty:
            continue

        grouped = numeric.groupby(blink_df["seg_id"]).mean()
        mapped_index = grouped.index.map(position_to_epoch.get)
        grouped.index = pd.Index(mapped_index, name="epoch")
        grouped = grouped.reindex(epoch_index)
        grouped.columns = [f"{col}_{ch}" for col in grouped.columns]
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(index=epoch_index)

    combined = pd.concat(frames, axis=1)
    combined = combined.reindex(epoch_index)
    return combined


def _compute_family_features(
    epochs: mne.Epochs,
    modality: str,
    family: str,
    picks: Sequence[str],
    epoch_index: pd.Index,
    *,
    progress_bar: bool,
    waveform_params: Mapping[str, Any] | None,
    waveform_run_fit: bool,
) -> pd.DataFrame:
    """Dispatch feature computation for a modality/family pair."""

    family_key = family.lower()
    if family_key == "events":
        df = aggregate_blink_event_features(epochs, picks=picks)
        df = df.rename(columns=_rename_event_columns(df.columns, modality))
    elif family_key == "energy":
        df = compute_energy_features(epochs, picks=picks)
    elif family_key == "freq":
        df = aggregate_frequency_domain_features(
            epochs, picks=picks, progress_bar=progress_bar
        )
    elif family_key == "kin":
        df = compute_kinematic_features(epochs, picks=picks)
    elif family_key == "morph":
        df = compute_epoch_morphology_features(epochs, picks=picks)
    elif family_key == "wave":
        params_dict: dict[str, Any] = dict(_DEFAULT_WAVEFORM_PARAMS)
        if waveform_params is not None:
            params_dict.update(dict(waveform_params))
        df = _compute_waveform_epoch_features(
            epochs,
            picks,
            params_dict,
            run_fit=waveform_run_fit,
            progress_bar=progress_bar,
        )
    else:
        raise ValueError(f"Unknown feature family: {family}")

    aligned = _align_to_epoch_index(df, epoch_index)
    family_label = "freq" if family_key == "freq" else family_key
    return _namespace_df(aligned, modality, family_label)


def _load_extra_features(
    epoch_index: pd.Index, metadata_csv_path: str | Path | None
) -> list[pd.DataFrame]:
    """Load auxiliary inputs such as CSV blink counts."""

    pieces: list[pd.DataFrame] = []
    if not metadata_csv_path:
        return pieces

    try:
        csv_df = pd.read_csv(Path(metadata_csv_path))
    except FileNotFoundError:
        logger.warning("Blink count CSV not found at %s", metadata_csv_path)
    else:
        if "epoch_id" in csv_df.columns:
            csv_df = csv_df.set_index("epoch_id")
        else:
            csv_df = csv_df.set_index(csv_df.columns[0])
        csv_df = csv_df.reindex(epoch_index)
        csv_df.columns = [f"META__events__{col}" for col in csv_df.columns]
        for col in csv_df.columns:
            csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce")
        pieces.append(csv_df)
    return pieces


def aggregate_blink_features(
    raw_or_epochs: mne.io.BaseRaw | mne.Epochs,
    *,
    epoch_len: float = 30.0,
    blink_label: str | None = None,
    progress_bar: bool = False,
    include_modalities: tuple[str, ...] = ("EEG", "EOG", "EAR"),
    feature_families: tuple[str, ...] = (
        "events",
        "energy",
        "freq",
        "kin",
        "morph",
        "wave",
    ),
    waveform_params: Mapping[str, Any] | None = None,
    waveform_run_fit: bool = True,
    metadata_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return consolidated blink features across modalities and families.

    Parameters
    ----------
    raw_or_epochs : mne.io.BaseRaw | mne.Epochs
        Continuous recording or pre-computed epochs used for feature
        extraction.
    epoch_len : float, optional
        Epoch length in seconds when ``raw_or_epochs`` is a Raw instance.
    blink_label : str | None, optional
        Annotation label describing blinks. ``None`` uses all annotations.
    progress_bar : bool, optional
        Whether to display progress bars during computations.
    include_modalities : tuple of str, optional
        Modalities to include (e.g., ``("EEG", "EOG", "EAR")``).
    feature_families : tuple of str, optional
        Feature families to compute (``"events"``, ``"energy"``, ``"freq"``,
        ``"kin"``, ``"morph"``, ``"wave"``).
    waveform_params : Mapping[str, Any] | None, optional
        Optional overrides for waveform feature parameters.
    waveform_run_fit : bool, optional
        Whether to run the blink waveform fitting routine.
    metadata_csv_path : str | Path | None, optional
        Optional CSV file containing per-epoch metadata to merge.

    Returns
    -------
    pandas.DataFrame
        Consolidated blink features indexed by epoch.
    """

    if isinstance(raw_or_epochs, mne.io.BaseRaw):
        epochs = slice_raw_into_mne_epochs_refine_annot(
            raw_or_epochs,
            epoch_len=epoch_len,
            blink_label=blink_label,
            progress_bar=progress_bar,
        )
    elif isinstance(raw_or_epochs, mne.Epochs):
        epochs = raw_or_epochs
    else:  # pragma: no cover - defensive
        raise TypeError("raw_or_epochs must be an mne Raw or Epochs instance")

    epoch_index = _get_epoch_index(epochs)

    modality_channels: dict[str, list[str]] = {}
    for modality in include_modalities:
        picks = _get_modality_channels(epochs.info, modality)
        if picks:
            modality_channels[modality] = picks
        else:
            logger.warning("No channels found for modality %s", modality)

    if not modality_channels:
        raise ValueError("No matching modalities found in the provided data")

    pieces: list[pd.DataFrame] = []
    for modality, picks in modality_channels.items():
        for family in feature_families:
            try:
                features_df = _compute_family_features(
                    epochs,
                    modality,
                    family,
                    picks,
                    epoch_index,
                    progress_bar=progress_bar,
                    waveform_params=waveform_params,
                    waveform_run_fit=waveform_run_fit,
                )
            except Exception as exc:  # pragma: no cover - logged for visibility
                logger.warning(
                    "Failed to compute %s features for %s: %s",
                    family,
                    modality,
                    exc,
                )
                continue
            if features_df.empty:
                continue
            pieces.append(features_df)

    pieces.extend(_load_extra_features(epoch_index, metadata_csv_path))

    if pieces:
        result = pd.concat(pieces, axis=1)
    else:
        result = pd.DataFrame(index=epoch_index)

    if not result.empty:
        all_nan = [col for col in result.columns if result[col].isna().all()]
        if all_nan:
            result = result.drop(columns=all_nan)

    result = result.reindex(epoch_index)
    result.index.name = "epoch"

    if not result.empty:
        result = result.loc[:, sorted(result.columns)]

    return result


def aggregate_blink_event_features(
    epochs: mne.Epochs,
    picks: str | Iterable[str],
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate blink-event metrics for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoch object with metadata containing blink onset and duration
        information. If modality-specific columns such as ``blink_onset_eeg``
        exist they are used; otherwise the generic ``blink_onset`` and
        ``blink_duration`` columns are expected.
    picks : str or iterable of str
        Channel name(s) used when computing inter-blink interval (IBI)
        statistics and to determine which blink metadata columns are
        consulted for ``blink_total`` and ``blink_rate``. When multiple
        modalities are provided, blink counts and rates are reported for each
        modality separately. The blink onset and duration columns are chosen by
        scanning the provided channels for modality-specific metadata, falling
        back to the generic ``blink_onset``/``blink_duration`` pair if none are
        found. The same IBI values are used for all channels because blink
        timing is not channel-specific in the metadata yet.
    features : sequence of str or None, optional
        Subset of feature groups to compute. Valid keys are
        ``"blink_total"``, ``"blink_rate"`` and ``"ibi"``. Passing ``None``
        (default) computes all features.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` containing one row per epoch with the
        requested features. Columns may include ``ep``, ``blink_total`` or
        ``blink_total_<modality>``, ``blink_rate`` or ``blink_rate_<modality>``
        and ``ibi_<channel>`` depending on ``features``.

    Raises
    ------
    ValueError
        If an unknown feature key is requested or if required channels are
        missing from ``epochs`` when ``"ibi"`` is selected.
    """

    logger.info("Aggregating blink features for %d epochs", len(epochs))

    valid = {"blink_total", "blink_rate", "ibi"}
    selected = set(features) if features is not None else valid
    invalid = selected - valid
    if invalid:
        raise ValueError(f"Unknown feature keys: {sorted(invalid)}")

    picks_list = normalize_picks(picks)
    require_channels(epochs, picks_list)

    pieces: list[pd.DataFrame] = []

    if selected & {"blink_total", "blink_rate"}:
        counts_df = blink_count(epochs, picks_list)
        rename_map: dict[str, str] = {}
        for col in counts_df.columns:
            if col == "ep":
                continue
            if col == "blink_count":
                rename_map[col] = "blink_total"
            elif col.startswith("blink_count_"):
                rename_map[col] = col.replace("blink_count_", "blink_total_")
        counts_df = counts_df.rename(columns=rename_map)
        pieces.append(counts_df)

    if "ibi" in selected:
        ibis_df = inter_blink_interval_epochs(epochs, picks_list)
        if pieces:
            ibis_df = ibis_df.drop(columns=["ep"])
        pieces.append(ibis_df)
    elif not selected:
        # If no features selected we still need an empty index-aligned frame
        pieces.append(pd.DataFrame(index=range(len(epochs))))

    df = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=range(len(epochs)))

    if "ep" in df.columns:
        df = df[["ep"] + [c for c in df.columns if c != "ep"]]

    if "blink_rate" in selected:
        epoch_len = epochs.tmax - epochs.tmin + 1.0 / epochs.info["sfreq"]
        for col in df.columns:
            if col.startswith("blink_total"):
                rate_col = col.replace("blink_total", "blink_rate")
                df[rate_col] = df[col] / epoch_len * 60.0

    # Reduce to requested columns if a subset was specified
    if features is not None:
        cols: list[str] = []
        if "blink_total" in selected:
            cols.extend(df.columns[df.columns.str.startswith("blink_total")])
        if "blink_rate" in selected:
            cols.extend(df.columns[df.columns.str.startswith("blink_rate")])
        if "ibi" in selected:
            cols.extend(df.columns[df.columns.str.startswith("ibi_")].tolist())
        df = df[cols]

    logger.debug("Aggregated feature DataFrame shape: %s", df.shape)
    return df


__all__ = ["aggregate_blink_features", "aggregate_blink_event_features"]
