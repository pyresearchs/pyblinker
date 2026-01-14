"""Build an HTML report plotting epoch EEG landmarks with EAR overlay."""

from __future__ import annotations

# ruff: noqa: E402

import logging
from pathlib import Path
from typing import Iterable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot

LOGGER = logging.getLogger(__name__)

LANDMARK_SAMPLE_COLUMNS = (
    "start__left_base__eeg",
    "end__right_base__eeg",
    "start__left_zero__eeg",
    "end__right_zero__eeg",
    "start__left_x_intercept__eeg",
    "end__right_x_intercept__eeg",
    "start__left_base_half_height__eeg",
    "end__right_base_half_height__eeg",
    "start__left_zero_half_height__eeg",
    "end__right_zero_half_height__eeg",
)


def _listify(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [value]


def _explode_epoch_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for epoch_index, row in metadata.iterrows():
        n_blinks = int(row.get("n_blinks", 0))
        if n_blinks <= 0:
            continue
        per_col = {col: _listify(row[col]) for col in metadata.columns if col != "n_blinks"}
        for idx in range(n_blinks):
            entry = {
                col: (values[idx] if idx < len(values) else np.nan)
                for col, values in per_col.items()
            }
            entry["epoch_index"] = epoch_index
            rows.append(entry)
    return pd.DataFrame.from_records(rows)


def _compute_overlay_indices(
    start: int,
    end: int,
    base_sfreq: float,
    overlay_len: int,
    overlay_sfreq: float | None,
) -> tuple[int, int]:
    derived_sfreq = base_sfreq if overlay_sfreq is None else float(overlay_sfreq)
    start_time = start / base_sfreq
    end_time = end / base_sfreq
    overlay_start = int(np.clip(round(start_time * derived_sfreq), 0, overlay_len - 1))
    overlay_end = int(np.clip(round(end_time * derived_sfreq), overlay_start, overlay_len - 1))
    return overlay_start, overlay_end


def _sample_value(
    sample: float | int | None,
    *,
    signal: np.ndarray,
    sfreq: float,
) -> tuple[float, float] | None:
    if sample is None:
        return None
    try:
        idx = int(sample)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= signal.shape[0]:
        return None
    time = idx / sfreq
    value = float(signal[idx])
    if not np.isfinite(value):
        return None
    return time, value


def _pick_boundary(row: pd.Series, columns: Iterable[str]) -> float | None:
    for col in columns:
        value = row.get(col)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def _ensure_1d_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim == 1:
        return signal
    if signal.ndim == 2:
        if signal.shape[0] == 1:
            return signal[0]
        if signal.shape[1] == 1:
            return signal[:, 0]
    return signal.reshape(-1)


def _plot_epoch_blink(
    *,
    row: pd.Series,
    eeg_signal: np.ndarray,
    sfreq: float,
    channel_name: str,
    ear_signal: np.ndarray | None,
    overlay_sfreq: float | None,
    pad_samples: int,
    candidate_id: object,
) -> tuple[plt.Figure, str]:
    eeg_signal = _ensure_1d_signal(eeg_signal)
    if ear_signal is not None:
        ear_signal = _ensure_1d_signal(ear_signal)
    n_samples = eeg_signal.shape[0]
    left = _pick_boundary(
        row,
        (
            "start__left_base__eeg",
            "start__left_zero__eeg",
            "start__left_x_intercept__eeg",
        ),
    )
    right = _pick_boundary(
        row,
        (
            "end__right_base__eeg",
            "end__right_zero__eeg",
            "end__right_x_intercept__eeg",
        ),
    )
    if left is None or right is None:
        left = 0.0
        right = float(n_samples - 1)

    start = max(0, int(min(n_samples - 1, round(left)) - pad_samples))
    end = min(n_samples - 1, int(round(right)) + pad_samples)

    times = np.arange(start, end + 1, dtype=float) / sfreq
    window = eeg_signal[start : end + 1]

    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [5, 1]}
    )
    legend_ax.axis("off")
    ax.plot(times, window, lw=1.0, color="black", label=channel_name)

    overlay_ax = None
    if ear_signal is not None:
        overlay_start, overlay_end = _compute_overlay_indices(
            start=start,
            end=end,
            base_sfreq=sfreq,
            overlay_len=ear_signal.shape[0],
            overlay_sfreq=overlay_sfreq,
        )
        overlay_times = np.arange(overlay_start, overlay_end + 1, dtype=float) / (
            sfreq if overlay_sfreq is None else float(overlay_sfreq)
        )
        overlay_window = ear_signal[overlay_start : overlay_end + 1]
        overlay_ax = ax.twinx()
        overlay_ax.plot(
            overlay_times,
            overlay_window,
            lw=1.0,
            alpha=0.75,
            color="C4",
            label="EAR-avg_ear",
        )
        overlay_ax.set_ylabel("EAR-avg_ear")

    marker_styles = {
        "Left base": ("C0", "o"),
        "Right base": ("C0", "o"),
        "Left zero": ("C1", "^"),
        "Right zero": ("C1", "^"),
        "Left x-intercept": ("C2", "x"),
        "Right x-intercept": ("C2", "x"),
        "Left base half-height": ("C3", "s"),
        "Right base half-height": ("C3", "s"),
        "Left zero half-height": ("C4", "D"),
        "Right zero half-height": ("C4", "D"),
        "Intersection": ("C5", "P"),
    }

    landmark_map = {
        "Left base": row.get("start__left_base__eeg"),
        "Right base": row.get("end__right_base__eeg"),
        "Left zero": row.get("start__left_zero__eeg"),
        "Right zero": row.get("end__right_zero__eeg"),
        "Left x-intercept": row.get("start__left_x_intercept__eeg"),
        "Right x-intercept": row.get("end__right_x_intercept__eeg"),
        "Left base half-height": row.get("start__left_base_half_height__eeg"),
        "Right base half-height": row.get("end__right_base_half_height__eeg"),
        "Left zero half-height": row.get("start__left_zero_half_height__eeg"),
        "Right zero half-height": row.get("end__right_zero_half_height__eeg"),
    }

    for label, sample in landmark_map.items():
        point = _sample_value(sample, signal=eeg_signal, sfreq=sfreq)
        if point is None:
            continue
        time, value = point
        color, marker = marker_styles[label]
        ax.scatter([time], [value], color=color, marker=marker, s=40, label=label)

    x_intersect = row.get("x_intersect__eeg")
    y_intersect = row.get("y_intersect__eeg")
    try:
        x_val = float(x_intersect)
        y_val = float(y_intersect)
    except (TypeError, ValueError):
        x_val = y_val = float("nan")
    if np.isfinite(x_val) and np.isfinite(y_val):
        color, marker = marker_styles["Intersection"]
        ax.scatter([x_val / sfreq], [y_val], color=color, marker=marker, s=50, label="Intersection")

    ax.set_title(f"Candidate {candidate_id} • {channel_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(channel_name)
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if overlay_ax is not None:
        overlay_handles, overlay_labels = overlay_ax.get_legend_handles_labels()
        handles.extend(overlay_handles)
        labels.extend(overlay_labels)
    if handles:
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        legend_ax.legend(unique_handles, unique_labels, loc="upper left", fontsize=8)

    left_time = float(left) / sfreq
    right_time = float(right) / sfreq
    caption_prefix = "Threshold crossings"
    caption_epoch = f"Epoch {row.get('epoch_index', 'unknown')}"
    caption = (
        f"{caption_epoch}. {caption_prefix} at {left_time:.3f}s and {right_time:.3f}s. "
        f"Segment {start}–{end} ({(end - start) / sfreq:.3f}s)."
    )
    caption += (
        " EEG landmarks shown: base, zero-crossings, x-intercepts, half-height points, "
        "and intersection (x_intersect__eeg, y_intersect__eeg)."
    )

    return fig, caption


def build_eeg_landmark_report(
    *,
    raw: mne.io.BaseRaw,
    channel_name: str,
    epoch_len: float,
    blink_label: str | None,
    segmentation_config: dict,
    output_path: Path,
    overlay_ear: bool = True,
) -> mne.Report:
    """Create an HTML report with per-blink EEG landmark plots."""

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=epoch_len,
        blink_label=blink_label,
        segmentation_type=segmentation_config,
        progress_bar=False,
    )

    if epochs.metadata is None or epochs.metadata.empty:
        raise ValueError("No epoch metadata available; cannot build report.")

    results = _explode_epoch_metadata(epochs.metadata)
    if results.empty:
        raise ValueError("No blink metadata found after exploding epochs metadata.")

    eeg_data = epochs.get_data(picks=channel_name)
    if eeg_data.ndim == 3 and eeg_data.shape[1] == 1:
        eeg_data = eeg_data[:, 0, :]
    if eeg_data.ndim != 2:
        raise ValueError("EEG epoch data must be shaped as (n_epochs, n_samples).")

    ear_data = None
    overlay_sfreq = None
    if overlay_ear:
        ear_channel = segmentation_config.get("ear", {}).get("channel")
        if ear_channel:
            ear_data = epochs.get_data(picks=ear_channel)
            if ear_data.ndim == 3 and ear_data.shape[1] == 1:
                ear_data = ear_data[:, 0, :]

    sfreq = float(epochs.info["sfreq"])
    pad_samples = int(round(0.05 * sfreq))

    report = mne.Report(title="EEG Blink Landmarks")
    for idx, row in results.iterrows():
        epoch_index = int(row.get("epoch_index", 0))
        candidate_id = row.get("candidate_id", idx)
        fig, caption = _plot_epoch_blink(
            row=row,
            eeg_signal=eeg_data[epoch_index],
            sfreq=sfreq,
            channel_name=channel_name,
            ear_signal=ear_data[epoch_index] if ear_data is not None else None,
            overlay_sfreq=overlay_sfreq,
            pad_samples=pad_samples,
            candidate_id=candidate_id,
        )
        report.add_figure(
            fig=fig,
            title=f"Blink {idx}",
            caption=caption,
            section="EEG blink landmarks",
            tags=("blink", "refined", channel_name),
        )
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    project_root = PROJECT_ROOT
    raw_path = project_root / "test" / "test_files" / "ear_eog_raw.fif"
    output_path = project_root / "tutorial_outputs" / "eeg_epoch_landmarks.html"

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")

    eeg_channel = "EEG-E8"
    ear_channel = "EAR-avg_ear"
    if eeg_channel not in raw.ch_names:
        raise ValueError(f"Required channel '{eeg_channel}' not found in raw data.")
    if ear_channel not in raw.ch_names:
        raise ValueError(f"Required channel '{ear_channel}' not found in raw data.")

    segmentation_config = {
        "ear": {"channel": ear_channel},
        "eeg": {"channel": eeg_channel},
    }

    build_eeg_landmark_report(
        raw=raw,
        channel_name=eeg_channel,
        epoch_len=30.0,
        blink_label=None,
        segmentation_config=segmentation_config,
        output_path=output_path,
        overlay_ear=True,
    )

    LOGGER.info("Saved EEG landmark report to %s", output_path)


if __name__ == "__main__":
    main()
