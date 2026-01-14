"""Build an HTML report plotting epoch EEG landmarks with optional EAR overlay."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from pyblinker.segmentation.refinement import slice_raw_into_mne_epochs_refine_annot


LOGGER = logging.getLogger(__name__)

LANDMARK_SAMPLE_COLUMNS = (
    "start__left_base__eeg",
    "end__right_base__eeg",
    "start__left_zero__eeg",
    "end__right_zero__eeg",
    "start__left_base_half_height__eeg",
    "end__right_base_half_height__eeg",
    "start__left_zero_half_height__eeg",
    "end__right_zero_half_height__eeg",
)
LANDMARK_XINTERCEPT_COLUMNS = (
    "start__left_x_intercept__eeg",
    "end__right_x_intercept__eeg",
)
LANDMARK_INTERSECT_COLUMNS = ("x_intersect__eeg", "y_intersect__eeg")


def _coerce_values(values: object) -> Sequence[float]:
    if isinstance(values, list):
        return [float(val) for val in values]
    if isinstance(values, np.ndarray):
        return [float(val) for val in values.tolist()]
    if isinstance(values, float) and np.isnan(values):
        return []
    if values is None:
        return []
    return [float(values)]


def _plot_epoch_landmarks(
    *,
    epoch_index: int,
    eeg_signal: np.ndarray,
    sfreq: float,
    metadata_row: dict,
    channel_name: str,
    ear_signal: np.ndarray | None = None,
) -> plt.Figure:
    n_samples = eeg_signal.shape[0]
    times = np.arange(n_samples) / sfreq

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, eeg_signal, color="black", linewidth=1.0, label=channel_name)

    if ear_signal is not None:
        ax2 = ax.twinx()
        ax2.plot(times, ear_signal, color="tab:orange", linewidth=1.0, alpha=0.7, label="EAR-avg_ear")
        ax2.set_ylabel("EAR")

    label_used: set[str] = set()

    for col in LANDMARK_SAMPLE_COLUMNS:
        values = _coerce_values(metadata_row.get(col))
        for value in values:
            if not np.isfinite(value):
                continue
            idx = int(np.clip(round(value), 0, n_samples - 1))
            label = col if col not in label_used else None
            ax.scatter(times[idx], eeg_signal[idx], label=label, s=25)
            if label:
                label_used.add(col)

    for col in LANDMARK_XINTERCEPT_COLUMNS:
        values = _coerce_values(metadata_row.get(col))
        for value in values:
            if not np.isfinite(value):
                continue
            idx = int(np.clip(round(value), 0, n_samples - 1))
            label = col if col not in label_used else None
            ax.scatter(times[idx], 0.0, label=label, marker="x", s=40)
            if label:
                label_used.add(col)

    x_intersects = _coerce_values(metadata_row.get(LANDMARK_INTERSECT_COLUMNS[0]))
    y_intersects = _coerce_values(metadata_row.get(LANDMARK_INTERSECT_COLUMNS[1]))
    for x_val, y_val in zip(x_intersects, y_intersects):
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        idx = int(np.clip(round(x_val), 0, n_samples - 1))
        label = "x_intersect__eeg" if "x_intersect__eeg" not in label_used else None
        ax.scatter(times[idx], y_val, label=label, marker="D", s=35)
        if label:
            label_used.add("x_intersect__eeg")

    ax.set_title(f"Epoch {epoch_index} EEG landmarks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EEG amplitude")
    ax.legend(loc="upper right", fontsize="x-small", ncol=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


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
    """Create an HTML report with per-epoch EEG landmark plots."""

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=epoch_len,
        blink_label=blink_label,
        segmentation_type=segmentation_config,
        progress_bar=False,
    )

    metadata = epochs.metadata
    if metadata is None or metadata.empty:
        raise ValueError("No epoch metadata available; cannot build report.")

    eeg_data = epochs.get_data(picks=channel_name)
    if eeg_data.ndim == 3 and eeg_data.shape[1] == 1:
        eeg_data = eeg_data[:, 0, :]
    if eeg_data.ndim != 2:
        raise ValueError("EEG epoch data must be shaped as (n_epochs, n_samples).")

    ear_data = None
    if overlay_ear:
        ear_channel = segmentation_config.get("ear", {}).get("channel")
        if ear_channel:
            ear_data = epochs.get_data(picks=ear_channel)
            if ear_data.ndim == 3 and ear_data.shape[1] == 1:
                ear_data = ear_data[:, 0, :]

    sfreq = float(epochs.info["sfreq"])
    report = mne.Report(title="EEG Landmark Epoch Report")

    for epoch_index, row in metadata.iterrows():
        fig = _plot_epoch_landmarks(
            epoch_index=epoch_index,
            eeg_signal=eeg_data[epoch_index],
            ear_signal=ear_data[epoch_index] if ear_data is not None else None,
            sfreq=sfreq,
            metadata_row=row.to_dict(),
            channel_name=channel_name,
        )
        report.add_figure(fig, title=f"Epoch {epoch_index} EEG landmarks")
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "test" / "test_files" / "ear_eog_raw.fif"
    output_path = project_root / "artifacts" / "eeg_epoch_landmarks.html"

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
