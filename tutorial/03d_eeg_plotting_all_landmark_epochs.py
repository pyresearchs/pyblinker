"""Plot EEG blinks per epoch with all refined landmarks from metadata."""

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
    "startleftbaseeeg",
    "endrightbaseeeg",
    "startleftzeroeeg",
    "endrightzeroeeg",
    "startleftbasehalfheighteeg",
    "endrightbasehalfheighteeg",
    "startleftzerohalfheighteeg",
    "endrightzerohalfheighteeg",
)
LANDMARK_XINTERCEPT_COLUMNS = (
    "startleftxintercepteeg",
    "endrightxintercepteeg",
)
LANDMARK_INTERSECT_COLUMNS = ("xintersecteeg", "yintersecteeg")


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


def _plot_landmarks_for_epoch(
    *,
    epoch_index: int,
    signal: np.ndarray,
    sfreq: float,
    metadata_row: dict,
    output_dir: Path,
    channel_name: str,
) -> Path:
    n_samples = signal.shape[0]
    times = np.arange(n_samples) / sfreq

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, signal, color="black", linewidth=1.0, label=f"{channel_name}")

    label_used: set[str] = set()

    for col in LANDMARK_SAMPLE_COLUMNS:
        values = _coerce_values(metadata_row.get(col))
        for value in values:
            if not np.isfinite(value):
                continue
            idx = int(np.clip(round(value), 0, n_samples - 1))
            label = col if col not in label_used else None
            ax.scatter(times[idx], signal[idx], label=label, s=25)
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
        label = "xintersecteeg" if "xintersecteeg" not in label_used else None
        ax.scatter(times[idx], y_val, label=label, marker="D", s=35)
        if label:
            label_used.add("xintersecteeg")

    ax.set_title(f"Epoch {epoch_index} EEG landmarks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize="x-small", ncol=2)
    ax.grid(True, alpha=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"epoch_{epoch_index:03d}_eeg_landmarks.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_eeg_landmark_epochs(
    *,
    raw: mne.io.BaseRaw,
    channel_name: str,
    epoch_len: float = 30.0,
    blink_label: str | None = None,
    output_dir: Path,
    segmentation_config: dict,
) -> list[Path]:
    """Create per-epoch landmark plots using epoch metadata."""

    epochs = slice_raw_into_mne_epochs_refine_annot(
        raw,
        epoch_len=epoch_len,
        blink_label=blink_label,
        segmentation_type=segmentation_config,
        progress_bar=False,
    )

    metadata = epochs.metadata
    if metadata is None or metadata.empty:
        LOGGER.warning("No epoch metadata available; skipping plots.")
        return []

    data = epochs.get_data(picks=channel_name)
    if data.ndim == 3 and data.shape[1] == 1:
        data = data[:, 0, :]
    if data.ndim != 2:
        raise ValueError("Epoch data must be shaped as (n_epochs, n_samples).")

    sfreq = float(epochs.info["sfreq"])
    outputs: list[Path] = []
    for epoch_index, row in metadata.iterrows():
        outputs.append(
            _plot_landmarks_for_epoch(
                epoch_index=epoch_index,
                signal=data[epoch_index],
                sfreq=sfreq,
                metadata_row=row.to_dict(),
                output_dir=output_dir,
                channel_name=channel_name,
            )
        )
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "test" / "test_files" / "ear_eog_raw.fif"
    output_dir = project_root / "artifacts" / "eeg_epoch_landmarks"

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

    outputs = plot_eeg_landmark_epochs(
        raw=raw,
        channel_name=eeg_channel,
        epoch_len=30.0,
        blink_label=None,
        output_dir=output_dir,
        segmentation_config=segmentation_config,
    )

    LOGGER.info("Saved %d epoch landmark plots to %s", len(outputs), output_dir)


if __name__ == "__main__":
    main()
